"""Label-free mapping of cell types between FAFB and the MCNS brain, every choice without target
labels.

Per label, with the GNNs saved by transfer.py (model.pt, one per seed):
1. Class probabilities on the source graph and, for every candidate preprocessing (MCNS alignment
   variant x feature normalization, see transfer.py), on the target graph.
2. Prior matching (flybrain.adapt): predicted type sizes on the target are pulled towards the
   type sizes the same GNN predicts on its source (scaled to the target), with strength rho.
3. Selection: the candidate (variant, normalization, rho) whose ensemble labeling (mean of the
   seeds' adjusted probabilities) has the highest type consistency (flybrain.consistency) on the
   scoring graph is chosen.
4. Refinement: NTAC on the chosen target graph, seeded where all seeds agree.
Target labels are only used to report accuracy.

--direction forward (default): GNNs trained on FAFB (--models, one directory); the MCNS target is
  built in each alignment variant; scoring graph: the least preprocessed MCNS graph (`raw`), with
  consistency on the other two graphs reported to show whether the choice depends on it.
--direction reverse: GNNs trained on MCNS, one directory per alignment variant
  (<models>_<variant>, from transfer.py --source mcns); target and scoring graph: FAFB.

Writes results/<out>/<label>/candidates.csv, selected.json, predictions.parquet and summary.md.

Usage: python experiments/b_gnn_celltypes/label_free.py [--models transfer2] [--labels cell_type]
       python experiments/b_gnn_celltypes/label_free.py --direction reverse --models transfer_rev
"""

from __future__ import annotations

import argparse
import itertools
import json

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

from flybrain.adapt import prior_weights
from flybrain.consistency import type_consistency
from flybrain.features import normalize_features
from flybrain.loading import load_graph
from flybrain.mcns import VARIANTS
from flybrain.ntac import ntac_from_predictions
from flybrain.splits import make_split
from flybrain.training import graph_inputs, load_model
from flybrain.utils import REPO_ROOT, write_predictions

LABELS = {"super_class": 1, "cell_type": 20}  # min_class_size, as in the configs
NORMS = ("target", "source")
RHOS = (0.0, 0.25, 0.5, 0.75, 1.0)
SCORING = "raw"  # forward direction: least preprocessed MCNS graph


def model_paths(models: str, label: str) -> list:
    paths = sorted((REPO_ROOT / "results" / models / label / "gnn").glob("s*/model.pt"))
    if not paths:
        raise FileNotFoundError(f"no model.pt under results/{models}/{label}/gnn")
    return paths


def setups(label: str, args):
    """[(variant, model paths, source graph, target graph)], graphs to score on, scoring key."""
    msize = LABELS[label]
    variants = ["synthetic"] if args.synthetic else list(VARIANTS)

    def mcns(v):
        return load_graph(
            label,
            synthetic=args.synthetic,
            min_class_size=msize,
            dataset="mcns",
            mcns_variant="density" if args.synthetic else v,
        )

    fafb = load_graph(label, synthetic=args.synthetic, min_class_size=msize)
    if args.direction == "forward":
        paths = model_paths(args.models, label)
        graphs = {v: mcns(v) for v in variants}
        scoring = variants[0] if args.synthetic else SCORING
        return [(v, paths, fafb, graphs[v]) for v in variants], graphs, scoring
    cases = []
    for v in variants:
        paths = model_paths(args.models if args.synthetic else f"{args.models}_{v}", label)
        cases.append((v, paths, mcns(v), fafb))
    return cases, {"fafb": fafb}, "fafb"


@torch.no_grad()
def probabilities(model, cfg, graph, x) -> np.ndarray:
    xx, adj = graph_inputs(graph, x, cfg, "cpu")
    return torch.softmax(model(xx, *adj), dim=1).numpy()


def score(pred, y) -> tuple[float, float]:
    lab = y >= 0
    return float((pred[lab] == y[lab]).mean()), float(f1_score(y[lab], pred[lab], average="macro"))


def run_label(label: str, args, out_dir) -> dict:
    cases, graphs, scoring_key = setups(label, args)
    targets = {variant: tgt for variant, _, _, tgt in cases}
    y = graphs[scoring_key].y.numpy()
    k = len(cases[0][2].label_names)
    used_models = sorted({str(m) for _, paths, _, _ in cases for m in paths})

    rows, preds = [], {}  # preds[(variant, norm, rho)] = [seed predictions]
    for (variant, paths, src, tgt), norm in itertools.product(cases, NORMS):
        seed_probs, seed_weights = [], []  # float32 probabilities and prior weights per seed
        for path in paths:
            model, cfg, names = load_model(path)
            assert names == src.label_names == tgt.label_names
            train = make_split(src, "random", seed=cfg["seed"])[0]
            ref = probabilities(model, cfg, src, normalize_features(src.x, "global", train))
            x = (
                normalize_features(tgt.x, "global")
                if norm == "target"
                else normalize_features(tgt.x, "global", fit_mask=train, reference=src.x)
            )
            p = probabilities(model, cfg, tgt, x)
            seed_weights.append(prior_weights(p, ref.mean(axis=0) * len(p)))
            seed_probs.append(p.astype(np.float32))
        for rho in RHOS:
            total = np.zeros_like(seed_probs[0])
            seed_preds = []
            for p, v in zip(seed_probs, seed_weights, strict=True):
                q = p * (v**rho).astype(np.float32)
                q /= q.sum(axis=1, keepdims=True)
                total += q
                seed_preds.append(q.argmax(axis=1))
            ensemble = total.argmax(axis=1)
            del total
            preds[(variant, norm, rho)] = seed_preds
            row = {"variant": variant, "norm": norm, "rho": rho}
            for name, g in graphs.items():
                row[f"consistency_{name}"] = type_consistency(
                    g.edge_index, g.edge_weight, ensemble, k
                )[0]
            row["acc"], row["macro_f1"] = score(ensemble, y)
            seed_acc = [score(sp, y)[0] for sp in seed_preds]
            row["seed_acc_mean"], row["seed_acc_std"] = (
                float(np.mean(seed_acc)),
                float(np.std(seed_acc)),
            )
            rows.append(row)
            print(f"[label_free] {label} {row}", flush=True)

    cand = pd.DataFrame(rows)
    best = cand.loc[cand[f"consistency_{scoring_key}"].idxmax()]
    chosen = (best.variant, best.norm, float(best.rho))
    other_choices = {
        c: tuple(cand.loc[cand[c].idxmax(), ["variant", "norm", "rho"]])
        for c in cand.columns
        if c.startswith("consistency_")
    }

    # refinement: NTAC on the chosen graph, seeded where all seeds agree
    g = targets[chosen[0]]
    seed_preds = np.stack(preds[chosen])
    refined, seeds = ntac_from_predictions(
        g.edge_index, g.edge_weight, np.full(g.num_nodes, -1), seed_preds, k
    )
    sel = {
        "label": label,
        "direction": args.direction,
        "models": used_models,
        "chosen": {"variant": chosen[0], "norm": chosen[1], "rho": chosen[2]},
        "choice_by_scoring_graph": {c: list(v) for c, v in other_choices.items()},
        "selected_acc": float(best.acc),
        "selected_macro_f1": float(best.macro_f1),
        "best_possible_acc": float(cand.acc.max()),
        "no_adaptation_acc": float(
            cand[(cand.variant == chosen[0]) & (cand.norm == chosen[1]) & (cand.rho == 0)].acc.iloc[
                0
            ]
        ),
        "seed_share": float(seeds.mean()),
    }
    sel["refined_acc"], sel["refined_macro_f1"] = score(refined, y)
    d = out_dir / label
    d.mkdir(parents=True, exist_ok=True)
    cand.to_csv(d / "candidates.csv", index=False)
    (d / "selected.json").write_text(json.dumps(sel, indent=2))
    lab = torch.from_numpy(y >= 0)
    none = torch.zeros_like(lab)
    write_predictions(d / "predictions.parquet", g, refined, (none, none, lab))
    print(f"[label_free] {label} selected: {json.dumps(sel)}", flush=True)
    return sel


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--direction", default="forward", choices=["forward", "reverse"])
    p.add_argument("--models", default=None, help="results subdirectory (prefix for reverse)")
    p.add_argument("--labels", nargs="+", default=list(LABELS), choices=list(LABELS))
    p.add_argument("--synthetic", action="store_true", help="synthetic stand-ins (for tests)")
    p.add_argument("--out", default=None, help="default: label_free[_rev]")
    args = p.parse_args()
    reverse = args.direction == "reverse"
    args.models = args.models or ("transfer_rev" if reverse else "transfer2")
    args.out = args.out or ("label_free_rev" if reverse else "label_free")
    out_dir = REPO_ROOT / "results" / args.out
    sels = [run_label(label, args, out_dir) for label in args.labels]
    pair = "MCNS -> FAFB" if reverse else "FAFB -> MCNS"
    lines = [f"Label-free {pair} mapping (accuracy % / macro-F1 on labeled target neurons).", ""]
    header = ["label", "chosen (variant, norm., rho)", "no adaptation", "prior matched", "+ NTAC"]
    header += ["best candidate (post hoc)"]
    lines += ["| " + " | ".join(header) + " |", "|---" * len(header) + "|"]
    for s in sels:
        c = s["chosen"]
        cells = [s["label"], f"{c['variant']}, {c['norm']}, {c['rho']}"]
        cells += [f"{100 * s['no_adaptation_acc']:.1f}"]
        cells += [f"{100 * s['selected_acc']:.1f} / {s['selected_macro_f1']:.2f}"]
        cells += [f"**{100 * s['refined_acc']:.1f} / {s['refined_macro_f1']:.2f}**"]
        cells += [f"{100 * s['best_possible_acc']:.1f}"]
        lines.append("| " + " | ".join(cells) + " |")
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
