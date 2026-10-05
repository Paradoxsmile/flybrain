"""Smoke test: the baseline script runs end-to-end on the synthetic graph."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_baseline_synthetic_runs(tmp_path):
    name = "pytest_logreg"
    cmd = [
        sys.executable,
        str(ROOT / "experiments/b_gnn_celltypes/baseline_logreg.py"),
        "--synthetic",
        "--run-name",
        name,
    ]
    subprocess.run(cmd, check=True, cwd=ROOT)
    run_dir = ROOT / "results" / name
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["test_acc"] > 0.3  # 4 balanced-ish classes: chance is ~0.25
    assert (run_dir / "config.yaml").exists()


def test_gnn_synthetic_runs():
    name = "pytest_gnn"
    cmd = [
        sys.executable,
        str(ROOT / "experiments/b_gnn_celltypes/train_gnn.py"),
        "--synthetic",
        "--epochs",
        "30",
        "--run-name",
        name,
    ]
    subprocess.run(cmd, check=True, cwd=ROOT)
    run_dir = ROOT / "results" / name
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["test_acc"] > 0.3
    assert (run_dir / "curves.png").exists()
