"""Final comparison: every method on every label and split, for several seeds.

Runs each (label, split, method, seed) once into results/<out>/<label>/<split>/<method>/s<seed>/
and skips runs that already finished, so the sweep can be divided between processes (--labels,
--splits, --methods) and restarted after an interruption. Then writes results/<out>/runs.csv,
summary.csv (mean and std over seeds) and summary.md.

Methods (in order of the information they use):
  majority         most frequent training class
  logreg           a neuron's own degree features
  label_prop       labels of connected neurons (homophily)
  partner_profile  label shares of synaptic partners
  gnn_plain        GraphSAGE: anonymous structure, inputs only, unweighted
  gnn              DirectedSAGE with the label's config (configs/default.yaml, cell_type.yaml)

Usage: python experiments/b_gnn_celltypes/benchmark.py [--labels cell_type] [--methods gnn]
       [--seeds 0 1 2] [--no-run]
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

from flybrain.splits import SPLITS
from flybrain.utils import REPO_ROOT

HERE = Path(__file__).parent
CONFIGS = {
    "super_class": HERE / "configs/default.yaml",
    "cell_type": HERE / "configs/cell_type.yaml",
}
METHODS = ("majority", "logreg", "label_prop", "partner_profile", "gnn_plain", "gnn")
PLAIN = ["--set", "bidirectional=false", "--set", "edge_weight=none", "--set", "class_weight=none"]
METRICS = ["test_acc", "test_macro_f1", "val_acc", "val_macro_f1"]


def command(label: str, split: str, method: str, seed: int, run: str, args) -> list[str]:
    """The script call for one run; all methods share label, split, seed and output dir."""
    config = CONFIGS[label]
    min_size = yaml.safe_load(config.read_text()).get("min_class_size", 1)
    if method in ("gnn", "gnn_plain"):
        cmd = [str(HERE / "train_gnn.py"), "--config", str(config), "--run-name", run]
        cmd += ["--set", f"split={split}", "--set", f"seed={seed}"]
        cmd += PLAIN if method == "gnn_plain" else []
        cmd += ["--set", f"epochs={args.epochs}"] if args.epochs else []
    else:
        script = "baseline_logreg.py" if method == "logreg" else "baselines.py"
        cmd = [str(HERE / script), "--label-col", label, "--min-class-size", str(min_size)]
        cmd += ["--split", split, "--seed", str(seed), "--run-name", run]
        cmd += [] if method == "logreg" else ["--method", method]
    return [sys.executable, *cmd, *(["--synthetic"] if args.synthetic else [])]


def summarize(out_dir: Path) -> pd.DataFrame:
    """Collect all finished runs and write runs.csv, summary.csv and summary.md."""
    rows = []
    for f in sorted(out_dir.glob("*/*/*/s*/metrics.json")):
        label, split, method, seed = f.parts[-5:-1]
        m = json.loads(f.read_text())
        rows.append(dict(label=label, split=split, method=method, seed=int(seed[1:]),
                         **{k: m[k] for k in METRICS}))  # fmt: skip
    runs = pd.DataFrame(rows)
    runs.to_csv(out_dir / "runs.csv", index=False)
    summary = runs.groupby(["label", "split", "method"])[METRICS].agg(["mean", "std", "count"])
    summary.columns = [f"{m}_{stat}" for m, stat in summary.columns]
    summary.to_csv(out_dir / "summary.csv")

    def cell(r: pd.Series) -> str:
        acc, f1 = 100 * r.test_acc_mean, r.test_macro_f1_mean
        if r.test_acc_count < 2:
            return f"{acc:.1f} / {f1:.2f}"
        return f"{acc:.1f} ± {100 * r.test_acc_std:.1f} / {f1:.2f} ± {r.test_macro_f1_std:.2f}"

    lines = ["Test accuracy (%) / macro-F1, mean ± std over seeds.", ""]
    for label, df in summary.reset_index().groupby("label"):
        table = df.assign(cell=df.apply(cell, axis=1)).pivot(
            index="method", columns="split", values="cell"
        )
        table = table.reindex([m for m in METHODS if m in table.index])
        table = table[[s for s in SPLITS if s in table.columns]]
        lines += [f"## {label}", "", "| method | " + " | ".join(table.columns) + " |"]
        lines += ["|---" * (len(table.columns) + 1) + "|"]
        lines += [f"| {m} | " + " | ".join(table.loc[m].fillna("–")) + " |" for m in table.index]
        lines += [""]
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--labels", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    p.add_argument("--splits", nargs="+", default=list(SPLITS), choices=SPLITS)
    p.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--epochs", type=int, default=None, help="override GNN epochs (for tests)")
    p.add_argument("--synthetic", action="store_true", help="use the synthetic test graph")
    p.add_argument("--out", default="benchmark", help="results subdirectory")
    p.add_argument("--no-run", action="store_true", help="only summarize finished runs")
    args = p.parse_args()

    out_dir = REPO_ROOT / "results" / args.out
    jobs = list(itertools.product(args.labels, args.splits, args.methods, args.seeds))
    for i, (label, split, method, seed) in enumerate(jobs, 1):
        run = f"{args.out}/{label}/{split}/{method}/s{seed}"
        if args.no_run or (REPO_ROOT / "results" / run / "metrics.json").exists():
            continue
        start = time.time()
        cmd = command(label, split, method, seed, run, args)
        subprocess.run(cmd, check=True, cwd=REPO_ROOT, stdout=subprocess.DEVNULL)
        print(
            f"[benchmark] {i}/{len(jobs)} {run}: {(time.time() - start) / 60:.1f} min", flush=True
        )

    summarize(out_dir)
    print((out_dir / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
