"""Ablation over the three DirectedSAGE options: all 2^3 combinations x several seeds.

Runs train_gnn.py once per (bidirectional, edge_weight, class_weight, seed) into
results/<sweep>/<combo>_s<seed>/ and skips runs that already finished, so an interrupted sweep
can simply be restarted. Writes results/<sweep>/runs.csv (one row per run) and summary.csv
(mean and std over seeds). The seed also changes the train/val/test split.

Usage: python experiments/b_gnn_celltypes/ablation.py [--label-col cell_type --min-class-size 20]
       [--seeds 0 1 2] [--split left_right] [--only bidir-true_ew-count_cw-none ...] [--epochs N]
       [--synthetic]
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

from flybrain.splits import SPLITS
from flybrain.utils import REPO_ROOT

TRAIN_SCRIPT = Path(__file__).parent / "train_gnn.py"
OPTIONS = {
    "bidirectional": (False, True),
    "edge_weight": ("none", "count"),
    "class_weight": ("none", "balanced"),
}
SHORT = {"bidirectional": "bidir", "edge_weight": "ew", "class_weight": "cw"}
METRICS = ["test_acc", "test_macro_f1", "val_acc", "val_macro_f1", "best_epoch", "n_test"]


def combo_name(combo: dict) -> str:
    """E.g. bidir-true_ew-count_cw-none."""
    return "_".join(f"{SHORT[k]}-{str(v).lower()}" for k, v in combo.items())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--label-col", default="super_class", choices=["super_class", "cell_type"])
    p.add_argument("--min-class-size", type=int, default=1)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--epochs", type=int, default=None, help="override config epochs")
    p.add_argument("--synthetic", action="store_true", help="use the synthetic test graph")
    p.add_argument("--split", default="random", choices=SPLITS)
    p.add_argument(
        "--only", nargs="+", default=None, metavar="COMBO", help="run only these combos (by name)"
    )
    p.add_argument("--sweep-name", default=None, help="default: ablation_<label-col>[_<split>]")
    args = p.parse_args()

    suffix = "" if args.split == "random" else f"_{args.split}"
    sweep = args.sweep_name or f"ablation_{args.label_col}{suffix}"
    sweep_dir = REPO_ROOT / "results" / sweep
    combos = [
        dict(zip(OPTIONS, values, strict=True)) for values in itertools.product(*OPTIONS.values())
    ]
    if args.only:
        unknown = set(args.only) - {combo_name(c) for c in combos}
        if unknown:
            p.error(f"unknown combos {sorted(unknown)}; names look like {combo_name(combos[0])}")
        combos = [c for c in combos if combo_name(c) in args.only]
    jobs = [(combo, seed) for combo in combos for seed in args.seeds]

    rows = []
    for i, (combo, seed) in enumerate(jobs, 1):
        run = f"{combo_name(combo)}_s{seed}"
        metrics_file = sweep_dir / run / "metrics.json"
        if not metrics_file.exists():
            overrides = {
                **combo,
                "seed": seed,
                "label_col": args.label_col,
                "min_class_size": args.min_class_size,
                "split": args.split,
            }
            if args.epochs is not None:
                overrides["epochs"] = args.epochs
            cmd = [sys.executable, str(TRAIN_SCRIPT), "--run-name", f"{sweep}/{run}"]
            cmd += ["--synthetic"] if args.synthetic else []
            for key, value in overrides.items():
                cmd += ["--set", f"{key}={str(value).lower()}"]
            start = time.time()
            subprocess.run(cmd, check=True, cwd=REPO_ROOT, stdout=subprocess.DEVNULL)
            print(
                f"[ablation] {i}/{len(jobs)} {run}: {(time.time() - start) / 60:.1f} min",
                flush=True,
            )
        metrics = json.loads(metrics_file.read_text())
        rows.append({**combo, "seed": seed, **{m: metrics.get(m) for m in METRICS}})

    runs = pd.DataFrame(rows)
    runs.to_csv(sweep_dir / "runs.csv", index=False)
    summary = runs.groupby(list(OPTIONS))[METRICS].agg(["mean", "std"])
    summary.columns = [f"{m}_{stat}" for m, stat in summary.columns]
    summary.to_csv(sweep_dir / "summary.csv")
    with pd.option_context("display.width", 200, "display.precision", 3):
        print(summary[[c for c in summary.columns if c.startswith("test_")]])
    print(f"results -> {sweep_dir}")


if __name__ == "__main__":
    main()
