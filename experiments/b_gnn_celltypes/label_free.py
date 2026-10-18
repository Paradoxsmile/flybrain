"""Label-free mapping of cell types from one connectome to another, every choice without target
labels.

Per label, with the GNNs saved by transfer.py (model.pt, one per seed):
1. Class probabilities on the source graph and, for every candidate preprocessing (alignment
   variant of the source and of the target x feature normalization, see transfer.py), on the
   target graph.
2. Prior matching (flybrain.adapt): predicted type sizes on the target are pulled towards the
   type sizes the same GNN predicts on its source (scaled to the target), with strength rho.
3. Selection: the candidate (variant, normalization, rho) whose ensemble labeling (mean of the
   seeds' adjusted probabilities) has the highest type consistency (flybrain.consistency) on the
   scoring graph, the least preprocessed target graph (`raw` for MCNS and BANC), is chosen.
   --select size_agreement instead takes the best mean rank of predicted type sizes closest to
   the reference and agreement between the seeds (flybrain.selection; chosen on the development
   sandbox after consistency failed on BANC).
4. Refinement: NTAC on the chosen target graph, seeded where all seeds agree.
Target labels are only used to report accuracy.

--source fafb: GNNs trained on FAFB (--models, one directory). --source mcns: GNNs trained on
MCNS, one directory per MCNS alignment variant (<models>_<variant>, from transfer.py --source
mcns). --target mcns | banc: built in each of its alignment variants; --target fafb: one graph.
With --score-all, consistency is also reported on the other target graphs, to show whether the
choice depends on the scoring graph. For BANC targets, accuracy is also reported per region and,
for cell types, against the morphology-matched FAFB type in the central brain
(docs/banc_protocol.md).

For development without touching new connectomes, the target can be made less complete
(--target-retention: synapses binomially thinned before the variant's threshold;
--target-exclude: cell types removed), and prior matching can be bounded (--max-weight).
candidates.csv also records label-free properties of each labeling (number of predicted types,
share of the largest one, Jensen-Shannon divergence of the predicted type sizes from the
reference, agreement between the seeds' labelings), to compare selection rules
(selection_rules.py).

Writes results/<out>/<label>/candidates.csv, selected.json, predictions.parquet and summary.md.

Usage: python experiments/b_gnn_celltypes/label_free.py [--models transfer2] [--score-all]
       python experiments/b_gnn_celltypes/label_free.py --source mcns --target fafb
       python experiments/b_gnn_celltypes/label_free.py --target banc
"""

from __future__ import annotations

import argparse
import itertools
import json

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

from flybrain.adapt import em_prior, prior_weights
from flybrain.banc import VARIANTS as BANC_VARIANTS
from flybrain.consistency import type_consistency
from flybrain.coverage import COVERAGES, TARGET_EXCLUSIONS, coverage_weights
from flybrain.features import normalize_features
from flybrain.hemibrain import VARIANTS as HEMIBRAIN_VARIANTS
from flybrain.loading import encode_labels, load_graph
from flybrain.mcns import VARIANTS as MCNS_VARIANTS
from flybrain.ntac import ntac_from_predictions
from flybrain.selection import RULES, choose, choose_all
from flybrain.splits import make_split
from flybrain.thinning import thin_graph
from flybrain.training import graph_inputs, load_model
from flybrain.utils import REPO_ROOT, write_predictions

LABELS = {"super_class": 1, "cell_type": 20}  # min_class_size, as in the configs
NORMS = ("target", "source")
RHOS = (0.0, 0.25, 0.5, 0.75, 1.0)
DATASET_VARIANTS = {
    "fafb": [None],
    "mcns": list(MCNS_VARIANTS),
    "banc": list(BANC_VARIANTS),
    "hemibrain": list(HEMIBRAIN_VARIANTS),
}
SCORING = {"fafb": "fafb", "mcns": "raw", "banc": "raw", "hemibrain": "raw"}  # least
# preprocessed target graph
DEFAULT_MODELS = {"fafb": "transfer2", "mcns": "transfer_rev"}
THINNING_SOURCE_SEED = 1  # thinning.py's source realization
DEFAULT_OUT = {("fafb", "mcns"): "label_free", ("mcns", "fafb"): "label_free_rev"}


def model_paths(models: str, label: str, retention: float = 1.0) -> list:
    """model.pt files of transfer.py (<models>/<label>/gnn/s*) or, for a thinned source, of
    thinning.py (<models>/<label>/p<retention>/s*)."""
    sub = "gnn" if retention == 1.0 else f"p{retention}"
    if retention == 1.0 and not (REPO_ROOT / "results" / models / label / sub).exists():
        sub = "p1.0"  # full-density models of thinning.py (e.g. with --thin-aug)
    paths = sorted((REPO_ROOT / "results" / models / label / sub).glob("s*/model.pt"))
    if not paths:
        raise FileNotFoundError(f"no model.pt under results/{models}/{label}/{sub}")
    return paths


def variants(dataset: str, synthetic: bool) -> list:
    if dataset == "fafb":
        return [None]
    return ["synthetic"] if synthetic else DATASET_VARIANTS[dataset]


def load(dataset: str, label: str, variant, synthetic: bool, **degrade):
    extra = dict(degrade)
    if dataset != "fafb":  # synthetic graphs ignore the variant
        extra[f"{dataset}_variant"] = DATASET_VARIANTS[dataset][0] if synthetic else variant
    return load_graph(
        label, synthetic=synthetic, min_class_size=LABELS[label], dataset=dataset, **extra
    )


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence (bits) between two distributions."""
    p, q = p / p.sum(), q / q.sum()
    m = (p + q) / 2

    def kl(a):
        nz = a > 0
        return float((a[nz] * np.log2(a[nz] / m[nz])).sum())

    return (kl(p) + kl(q)) / 2


def setups(label: str, args):
    """[(name, model paths, source graph, target graph)], graphs to score on, scoring key.

    name is the target variant for a FAFB source, the source variant for a FAFB target and
    "<source variant>/<target variant>" otherwise."""
    tvs = variants(args.target, args.synthetic)
    exclusions = TARGET_EXCLUSIONS.get(args.target_coverage, {})
    degrade = {
        "retention": args.target_retention,
        "retention_seed": args.retention_seed,
        "exclude_types": tuple(args.target_exclude) + exclusions.get("exclude_types", ()),
        "exclude_super_classes": exclusions.get("exclude_super_classes", ()),
        "crop_fraction": args.target_crop,
    }
    targets = {
        tv or args.target: load(args.target, label, tv, args.synthetic, **degrade) for tv in tvs
    }
    cases = []
    for sv in variants(args.source, args.synthetic):
        suffixed = sv is not None and not args.synthetic
        models = f"{args.models}_{sv}" if suffixed else args.models
        paths = model_paths(models, label, args.source_retention)
        src = load(args.source, label, sv, args.synthetic)
        if args.source_retention < 1:  # the copy the models were trained on (thinning.py)
            src = thin_graph(src, args.source_retention, seed=THINNING_SOURCE_SEED)
        for tkey, tgt in targets.items():
            if sv is None:
                name = tkey
            elif args.target == "fafb":
                name = sv
            else:
                name = f"{sv}/{tkey}"
            cases.append((name, paths, src, tgt))
    scoring = next(iter(targets)) if args.synthetic else SCORING[args.target]
    graphs = targets if args.score_all else {scoring: targets[scoring]}
    return cases, graphs, scoring


@torch.no_grad()
def probabilities(model, cfg, graph, x) -> np.ndarray:
    xx, adj = graph_inputs(graph, x, cfg, "cpu")
    return torch.softmax(model(xx, *adj), dim=1).numpy()


def score(pred, y) -> tuple[float, float]:
    lab = y >= 0
    return float((pred[lab] == y[lab]).mean()), float(f1_score(y[lab], pred[lab], average="macro"))


def breakdown(g, pred, label: str) -> dict:
    """Accuracy per region, for cropped and complete neurons (hemibrain) and, for cell types,
    against the morphology-matched type (BANC)."""
    out = {}
    y = g.y.numpy()
    cropped = getattr(g, "cropped", None)
    if cropped is not None:
        for name, mask in (("cropped", cropped == "cropped"), ("complete", cropped != "cropped")):
            mask = mask & (y >= 0)
            if mask.any():
                out[f"{name}_n"] = int(mask.sum())
                out[f"{name}_acc"], out[f"{name}_macro_f1"] = score(pred[mask], y[mask])
    for region in sorted(set(getattr(g, "region", [])) - {""}):
        mask = (g.region == region) & (y >= 0)
        if not mask.any():
            continue
        out[f"{region}_n"] = int(mask.sum())
        out[f"{region}_acc"], out[f"{region}_macro_f1"] = score(pred[mask], y[mask])
    if label == "cell_type" and getattr(g, "morph_type", None) is not None:
        morph = encode_labels(pd.Series(g.morph_type).replace("", pd.NA), label_names=g.label_names)
        morph = morph[0]
        if not (morph >= 0).any():
            return out
        out["morph_n"] = int((morph >= 0).sum())
        out["morph_acc"], out["morph_macro_f1"] = score(pred, morph)
        both = (morph >= 0) & (y >= 0)
        out["morph_agrees_with_label"] = float((morph[both] == y[both]).mean())
    return out


def target_features(tgt, src, train, norm: str, covered) -> torch.Tensor:
    """Normalized target features. target: the target's own statistics; source: the source's
    training statistics; coverage: standardized like target, then given the mean / std of the
    source neurons inside the covered volume, then normalized like source. Normalizing by the
    target's own statistics assumes the source's composition; for a partial volume the covered
    part of the source is the right reference."""
    if norm == "target":
        return normalize_features(tgt.x, "global")
    x = tgt.x
    if norm == "coverage":
        inside = src.x[torch.from_numpy(covered > 0)]
        z = (x - x.mean(0)) / x.std(0).clamp(min=1e-6)
        x = z * inside.std(0) + inside.mean(0)
    return normalize_features(x, "global", fit_mask=train, reference=src.x)


def type_prior(p, ref, args, covered) -> tuple[np.ndarray, np.ndarray]:
    """(expected type shares on the target, class weights for prior matching).

    source: the shares the GNN predicts on its source; coverage: the same over the source
    neurons inside the volume the target covers (flybrain.coverage); em: estimated from the
    target predictions (flybrain.adapt.em_prior)."""
    shares = ref.mean(axis=0)
    if args.prior == "coverage":
        shares = covered @ ref / covered.sum()
    if args.prior == "em":
        shares, weights = em_prior(p, shares)
        return shares, np.maximum(weights, 1e-12)
    return shares, prior_weights(p, shares * len(p), max_weight=args.max_weight)


def run_label(label: str, args, out_dir) -> dict:
    cases, graphs, scoring_key = setups(label, args)
    targets = {name: tgt for name, _, _, tgt in cases}
    y = graphs[scoring_key].y.numpy()
    k = len(cases[0][2].label_names)
    used_models = sorted({str(m) for _, paths, _, _ in cases for m in paths})

    rows, preds, ensembles = [], {}, {}  # keyed by (name, norm, rho)
    norms = NORMS + (("coverage",) if args.coverage else ())
    covered = {}  # coverage weights per source graph
    for (name, paths, src, tgt), norm in itertools.product(cases, norms):
        if args.coverage and id(src) not in covered:
            covered[id(src)] = coverage_weights(src.node_ids.numpy(), args.coverage)
        assert tgt.num_nodes == len(y)
        seed_probs, seed_weights, refs = [], [], []  # per seed
        for path in paths:
            model, cfg, names = load_model(path)
            assert names == src.label_names == tgt.label_names
            train = make_split(src, "random", seed=cfg["seed"])[0]
            ref = probabilities(model, cfg, src, normalize_features(src.x, "global", train))
            x = target_features(tgt, src, train, norm, covered.get(id(src)))
            p = probabilities(model, cfg, tgt, x)
            shares, weights = type_prior(p, ref, args, covered.get(id(src)))
            seed_weights.append(weights)
            seed_probs.append(p.astype(np.float32))
            refs.append(shares)
        reference = np.mean(refs, axis=0)
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
            preds[(name, norm, rho)] = seed_preds
            ensembles[(name, norm, rho)] = ensemble
            row = {"variant": name, "norm": norm, "rho": rho}
            for gname, g in graphs.items():
                row[f"consistency_{gname}"] = type_consistency(
                    g.edge_index, g.edge_weight, ensemble, k
                )[0]
            sizes = np.bincount(ensemble, minlength=k).astype(float)
            row["n_types"] = int((sizes > 0).sum())
            row["top_share"] = float(sizes.max() / sizes.sum())
            row["ref_top_share"] = float(reference.max() / reference.sum())
            row["size_js"] = js_divergence(sizes, reference)
            pairs = list(itertools.combinations(seed_preds, 2))
            agree = [(a == b).mean() for a, b in pairs]
            row["pairwise_agreement"] = float(np.mean(agree)) if agree else float("nan")
            row["seed_agreement"] = float((np.stack(seed_preds) == seed_preds[0]).all(0).mean())
            row["acc"], row["macro_f1"] = score(ensemble, y)
            seed_acc = [score(sp, y)[0] for sp in seed_preds]
            row["seed_acc_mean"], row["seed_acc_std"] = (
                float(np.mean(seed_acc)),
                float(np.std(seed_acc)),
            )
            rows.append(row)
            print(f"[label_free] {label} {row}", flush=True)

    cand = pd.DataFrame(rows)
    key = f"consistency_{scoring_key}"
    best = cand.loc[choose(cand, args.select, key)]
    chosen = (best.variant, best.norm, float(best.rho))
    post_hoc = cand.loc[cand.acc.idxmax()]
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
    unadapted = (chosen[0], chosen[1], 0.0)
    sel = {
        "label": label,
        "source": args.source,
        "target": args.target,
        "target_retention": args.target_retention,
        "target_exclude": list(args.target_exclude),
        "max_weight": args.max_weight,
        "source_retention": args.source_retention,
        "target_coverage": args.target_coverage,
        "target_crop": args.target_crop,
        "prior": args.prior,
        "coverage": args.coverage,
        "models": used_models,
        "select": args.select,
        "chosen": {"variant": chosen[0], "norm": chosen[1], "rho": chosen[2]},
        "choice_by_rule": {
            rule: [cand.variant[i], cand.norm[i], float(cand.rho[i]), float(cand.acc[i])]
            for rule, i in choose_all(cand, key).items()
        },
        "choice_by_scoring_graph": {c: list(v) for c, v in other_choices.items()},
        "selected_acc": float(best.acc),
        "selected_macro_f1": float(best.macro_f1),
        "best_possible_acc": float(post_hoc.acc),
        "best_possible": [post_hoc.variant, post_hoc.norm, float(post_hoc.rho)],
        "no_adaptation_acc": float(cand.set_index(["variant", "norm", "rho"]).acc[unadapted]),
        "seed_share": float(seeds.mean()),
    }
    sel["refined_acc"], sel["refined_macro_f1"] = score(refined, y)
    labelings = {
        "no_adaptation": ensembles[unadapted],
        "selected": ensembles[chosen],
        "refined": refined,
        "best_possible": ensembles[tuple(sel["best_possible"])],
    }
    parts = {key: breakdown(g, pred, label) for key, pred in labelings.items()}
    if any(parts.values()):
        sel["breakdown"] = parts
    d = out_dir / label
    d.mkdir(parents=True, exist_ok=True)
    cand.to_csv(d / "candidates.csv", index=False)
    (d / "selected.json").write_text(json.dumps(sel, indent=2))
    lab = torch.from_numpy(y >= 0)
    none = torch.zeros_like(lab)
    write_predictions(d / "predictions.parquet", g, refined, (none, none, lab))
    print(f"[label_free] {label} selected: {json.dumps(sel)}", flush=True)
    return sel


def summary_lines(sels: list, source: str, target: str) -> list:
    pair = f"{source.upper()} -> {target.upper()}"
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
    for s in sels:
        if "breakdown" not in s:
            continue
        lines += ["", f"{s['label']}, accuracy % by subset (n labeled neurons):", ""]
        keys = [k[: -len("_acc")] for k in s["breakdown"]["refined"] if k.endswith("_acc")]
        lines += ["| labeling | " + " | ".join(keys) + " |", "|---" * (len(keys) + 1) + "|"]
        for name, part in s["breakdown"].items():
            cells = [f"{100 * part[f'{k}_acc']:.1f} ({part[f'{k}_n']})" for k in keys]
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return lines


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", default="fafb", choices=["fafb", "mcns"])
    p.add_argument("--target", default="mcns", choices=["mcns", "fafb", "banc", "hemibrain"])
    p.add_argument("--models", default=None, help="results subdirectory (prefix for MCNS)")
    p.add_argument("--labels", nargs="+", default=list(LABELS), choices=list(LABELS))
    p.add_argument("--score-all", action="store_true", help="consistency on all target graphs")
    p.add_argument("--select", default="consistency", choices=RULES, help="selection rule")
    p.add_argument("--target-retention", type=float, default=1.0, help="thin the target")
    p.add_argument("--retention-seed", type=int, default=2, help="thinning realization")
    p.add_argument("--target-exclude", nargs="*", default=[], help="cell types to remove")
    p.add_argument("--max-weight", type=float, default=None, help="bound prior matching")
    p.add_argument(
        "--source-retention", type=float, default=1.0, help="GNNs of thinning.py at this density"
    )
    p.add_argument("--target-coverage", default=None, choices=sorted(TARGET_EXCLUSIONS))
    p.add_argument("--target-crop", type=float, default=0.0, help="share of cropped neurons")
    p.add_argument("--prior", default="source", choices=["source", "coverage", "em"])
    p.add_argument(
        "--coverage",
        default=None,
        choices=sorted(COVERAGES),
        help="volume the target covers: adds the coverage normalization, allows --prior coverage",
    )
    p.add_argument("--synthetic", action="store_true", help="synthetic stand-ins (for tests)")
    p.add_argument("--out", default=None, help="default: label_free[_rev | _<source>_<target>]")
    args = p.parse_args()
    degraded = args.target_retention < 1 or args.target_exclude or args.target_crop > 0
    degraded = degraded or args.target_coverage is not None
    if args.coverage and args.source != "fafb":
        p.error("--coverage needs --source fafb (FAFB records the neuropil of each synapse)")
    if args.prior == "coverage" and not args.coverage:
        p.error("--prior coverage needs --coverage")
    if args.source == args.target and not degraded:
        p.error("source and target must differ unless the target is degraded")
    args.models = args.models or DEFAULT_MODELS[args.source]
    pair = (args.source, args.target)
    default = DEFAULT_OUT.get(pair, f"label_free_{args.source}_{args.target}")
    if args.source_retention < 1:
        if args.source != "fafb":
            p.error("--source-retention needs --source fafb (models from thinning.py)")
        args.models = args.models if args.models != DEFAULT_MODELS["fafb"] else "thinning/models"
    if degraded or args.max_weight or args.source_retention < 1:
        default = f"label_free_{args.source}_{args.target}_r{args.target_retention:g}"
        default += "_x" if args.target_exclude else ""
        default += f"_w{args.max_weight:g}" if args.max_weight else ""
        default += f"_s{args.source_retention:g}" if args.source_retention < 1 else ""
        default += f"_{args.target_coverage}" if args.target_coverage else ""
        default += f"_crop{args.target_crop:g}" if args.target_crop else ""
    default += f"_prior-{args.prior}" if args.prior != "source" else ""
    default += f"_{args.select}" if args.select != "consistency" else ""
    args.out = args.out or default
    out_dir = REPO_ROOT / "results" / args.out
    sels = [run_label(label, args, out_dir) for label in args.labels]
    lines = summary_lines(sels, args.source, args.target)
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
