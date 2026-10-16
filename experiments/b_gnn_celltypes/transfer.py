"""Transfer: train on FAFB, predict super classes / cell types on the male CNS (MCNS) brain.

Each method is fit on FAFB (random split; validation picks the training epoch and options, as in
benchmark.py) and applied to the MCNS brain without any MCNS label. Every MCNS graph variant
(flybrain.mcns.VARIANTS) is evaluated with two feature normalizations, from the same fit:
  source  z-score with the FAFB training statistics, as during training
  target  z-score with the MCNS graph's own statistics (uses no labels)
Methods: gnn and gnn_plain (benchmark configs), logreg (own degree features), majority.
Partner-label methods (partner_profile, ntac) need labeled neurons in the target graph, so they
cannot transfer.

For cell types, `set_acc` scores the neurons whose MCNS type maps to several FAFB types: a
prediction counts if it is one of them.

Writes results/<out>/<label>/<method>/s<seed>/metrics.json (FAFB test scores plus one entry per
variant and normalization), predictions for every variant and normalization, the trained GNN
(model.pt: state_dict, config, label names) and runs.csv, summary.csv,
summary.md for all finished runs. Finished runs are skipped.

Usage: python experiments/b_gnn_celltypes/transfer.py [--labels cell_type] [--methods gnn]
       [--seeds 0 1 2] [--variants density raw janelia] [--no-run]
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from flybrain.features import normalize_features
from flybrain.loading import load_graph
from flybrain.mcns import VARIANTS
from flybrain.splits import make_split
from flybrain.training import fit_gnn, graph_inputs, predict
from flybrain.utils import REPO_ROOT, get_device, log_run, set_seed, write_predictions

HERE = Path(__file__).parent
CONFIGS = {
    "super_class": HERE / "configs/default.yaml",
    "cell_type": HERE / "configs/cell_type.yaml",
}
METHODS = ("majority", "logreg", "gnn_plain", "gnn")
PLAIN = {"bidirectional": False, "edge_weight": "none", "class_weight": "none"}
NORMS = ("source", "target")


def scores(pred: np.ndarray, data) -> dict[str, float]:
    """Accuracy and macro-F1 on labeled nodes; set accuracy on nodes with several MCNS types."""
    y = data.y.numpy()
    lab = y >= 0
    out = {
        "acc": float((pred[lab] == y[lab]).mean()),
        "macro_f1": float(f1_score(y[lab], pred[lab], average="macro")),
    }
    types = getattr(data, "flywire_types", None)
    if types is not None:
        multi = np.char.find(types.astype(str), ",") >= 0
        if multi.any():
            names = np.array(data.label_names, dtype=object)
            hits = [
                names[p] in t.split(",") for p, t in zip(pred[multi], types[multi], strict=True)
            ]
            out["set_acc"], out["n_set"] = float(np.mean(hits)), int(multi.sum())
    return out


def target_x(src_raw, train, tgt, norm: str) -> torch.Tensor:
    """Normalized target features: FAFB training statistics ("source") or the target's own."""
    if norm == "source":
        return normalize_features(tgt.x, "global", fit_mask=train, reference=src_raw)
    return normalize_features(tgt.x, "global")


def run_one(label: str, method: str, seed: int, targets: dict, args, run_dir: Path) -> None:
    """Fit one method on FAFB and evaluate it on every target graph and normalization."""
    cfg = yaml.safe_load(CONFIGS[label].read_text())
    cfg.update(seed=seed, **(PLAIN if method == "gnn_plain" else {}))
    if args.epochs:
        cfg["epochs"] = args.epochs
    set_seed(seed)
    device = get_device()
    src = load_graph(
        label, synthetic=args.synthetic, seed=seed, min_class_size=cfg.get("min_class_size", 1)
    )
    train, val, test = make_split(src, "random", seed=seed)
    for tgt in targets.values():
        assert tgt.label_names == src.label_names, "label spaces differ"
    src_raw = src.x
    metrics: dict[str, float | int | str] = {"device": str(device)}

    if method in ("gnn", "gnn_plain"):
        fit = fit_gnn(cfg, src, train, val, device)
        metrics["best_epoch"] = int(max(fit.history, key=lambda h: h["val_macro_f1"])["epoch"])
        src_pred = predict(fit.model, fit.x, fit.adj)
        torch.save(
            {"state_dict": fit.model.state_dict(), "cfg": cfg, "label_names": src.label_names},
            run_dir / "model.pt",
        )

        def apply(tgt, norm):
            return predict(
                fit.model, *graph_inputs(tgt, target_x(src_raw, train, tgt, norm), cfg, device)
            )

    elif method == "logreg":
        x, tr = normalize_features(src_raw, "global", train).numpy(), train.numpy()
        clf = LogisticRegression(max_iter=1000, random_state=seed).fit(x[tr], src.y.numpy()[tr])
        src_pred = clf.predict(x)

        def apply(tgt, norm):
            return clf.predict(target_x(src_raw, train, tgt, norm).numpy())

    else:  # majority
        majority = int(np.bincount(src.y[train].numpy()).argmax())
        src_pred = np.full(src.num_nodes, majority)

        def apply(tgt, norm):
            return np.full(tgt.num_nodes, majority)

    y = src.y.numpy()
    t = test.numpy()
    metrics["fafb_test_acc"] = float((src_pred[t] == y[t]).mean())
    metrics["fafb_test_macro_f1"] = float(f1_score(y[t], src_pred[t], average="macro"))
    for (variant, tgt), norm in itertools.product(targets.items(), NORMS):
        pred = apply(tgt, norm)
        metrics.update({f"{variant}_{norm}_{k}": v for k, v in scores(pred, tgt).items()})
        lab = torch.from_numpy(tgt.y.numpy() >= 0)
        none = torch.zeros_like(lab)
        write_predictions(
            run_dir / f"predictions-{variant}-{norm}.parquet", tgt, pred, (none, none, lab)
        )
    log_run(run_dir, {**cfg, "method": method, "synthetic": args.synthetic}, metrics)
    print(f"[transfer] {label} {method} s{seed}: {metrics}", flush=True)


def summarize(out_dir: Path) -> None:
    """Collect finished runs into runs.csv, summary.csv and summary.md (mean ± std over seeds)."""
    rows = []
    for f in sorted(out_dir.glob("*/*/s*/metrics.json")):
        label, method, seed = f.parts[-4:-1]
        m = json.loads(f.read_text())
        targets = [("fafb", "test")] + [
            (v, n) for v in VARIANTS for n in NORMS if f"{v}_{n}_acc" in m
        ]
        for variant, norm in targets:
            key = "fafb_test" if variant == "fafb" else f"{variant}_{norm}"
            rows.append(
                dict(
                    label=label,
                    method=method,
                    seed=int(seed[1:]),
                    target=f"{variant}/{norm}",
                    acc=m[f"{key}_acc"],
                    macro_f1=m[f"{key}_macro_f1"],
                    set_acc=m.get(f"{key}_set_acc"),
                )
            )
    if not rows:
        return
    runs = pd.DataFrame(rows)
    runs.to_csv(out_dir / "runs.csv", index=False)
    summary = runs.groupby(["label", "method", "target"])[["acc", "macro_f1", "set_acc"]].agg(
        ["mean", "std", "count"]
    )
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    summary.to_csv(out_dir / "summary.csv")
    lines = [
        "Accuracy (%) / macro-F1, mean ± std over seeds. Trained on FAFB; fafb/test is the",
        "",
    ]
    lines[0] += " FAFB test split, all other columns are the MCNS brain (variant/normalization)."
    for label, df in summary.reset_index().groupby("label"):
        df = df.assign(
            cell=[
                f"{100 * a:.1f} ± {100 * (s if s == s else 0):.1f} / {f:.2f}"
                for a, s, f in zip(df.acc_mean, df.acc_std, df.macro_f1_mean, strict=True)
            ]
        )
        table = df.pivot(index="method", columns="target", values="cell")
        table = table.reindex([m for m in METHODS if m in table.index])
        cols = ["fafb/test"] + [c for c in table.columns if c != "fafb/test"]
        table = table[cols]
        lines += [
            f"## {label}",
            "",
            "| method | " + " | ".join(cols) + " |",
            "|---" * (len(cols) + 1) + "|",
        ]
        lines += [f"| {m} | " + " | ".join(table.loc[m].fillna("–")) + " |" for m in table.index]
        lines += [""]
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--labels", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    p.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    p.add_argument("--epochs", type=int, default=None, help="override GNN epochs (for tests)")
    p.add_argument("--synthetic", action="store_true", help="synthetic stand-ins for both graphs")
    p.add_argument("--out", default="transfer", help="results subdirectory")
    p.add_argument("--no-run", action="store_true", help="only summarize finished runs")
    args = p.parse_args()

    out_dir = REPO_ROOT / "results" / args.out
    for label in [] if args.no_run else args.labels:
        min_size = yaml.safe_load(CONFIGS[label].read_text()).get("min_class_size", 1)
        targets = None
        for method, seed in itertools.product(args.methods, args.seeds):
            run_dir = out_dir / label / method / f"s{seed}"
            if (run_dir / "metrics.json").exists():
                continue
            # MCNS graphs depend on neither method nor seed (synthetic stand-ins do on the seed)
            if targets is None or args.synthetic:
                targets = {
                    v: load_graph(
                        label,
                        synthetic=args.synthetic,
                        seed=seed,
                        min_class_size=min_size,
                        dataset="mcns",
                        mcns_variant=v,
                    )
                    for v in args.variants
                }
            run_dir.mkdir(parents=True, exist_ok=True)
            run_one(label, method, seed, targets, args, run_dir)
    summarize(out_dir)
    if (out_dir / "summary.md").exists():
        print((out_dir / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
