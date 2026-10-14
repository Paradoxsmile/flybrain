"""Smoke test: the baseline script runs end-to-end on the synthetic graph."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

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
    preds = pd.read_parquet(run_dir / "predictions.parquet")
    test = preds[preds.split == "test"]
    assert len(test) == metrics["n_test"]
    assert (test.label == test.pred).mean() == pytest.approx(metrics["test_acc"], abs=1e-6)


def test_gnn_set_overrides_config():
    name = "pytest_gnn_set"
    cmd = [
        sys.executable,
        str(ROOT / "experiments/b_gnn_celltypes/train_gnn.py"),
        "--synthetic",
        "--run-name",
        name,
        "--set",
        "epochs=3",
        "--set",
        "bidirectional=false",
        "--set",
        "edge_weight=none",
    ]
    subprocess.run(cmd, check=True, cwd=ROOT)
    cfg = yaml.safe_load((ROOT / "results" / name / "config.yaml").read_text())
    assert cfg["epochs"] == 3 and cfg["bidirectional"] is False and cfg["edge_weight"] == "none"
    bad = subprocess.run([*cmd[:2], "--set", "no_such_key=1"], cwd=ROOT, capture_output=True)
    assert bad.returncode != 0


def test_ablation_synthetic_runs():
    name = "pytest_ablation"
    cmd = [
        sys.executable,
        str(ROOT / "experiments/b_gnn_celltypes/ablation.py"),
        "--synthetic",
        "--seeds",
        "0",
        "--epochs",
        "2",
        "--sweep-name",
        name,
    ]
    subprocess.run(cmd, check=True, cwd=ROOT)
    summary = pd.read_csv(ROOT / "results" / name / "summary.csv")
    assert len(summary) == 8
    assert "test_macro_f1_mean" in summary.columns


def test_benchmark_synthetic_runs_and_summarizes():
    out = "pytest_benchmark"
    cmd = [
        sys.executable,
        str(ROOT / "experiments/b_gnn_celltypes/benchmark.py"),
        "--synthetic",
        "--seeds",
        "0",
        "--splits",
        "left_right",
        "--methods",
        "majority",
        "gnn",
        "--epochs",
        "3",
        "--out",
        out,
    ]
    subprocess.run(cmd, check=True, cwd=ROOT)
    runs = pd.read_csv(ROOT / "results" / out / "runs.csv")
    assert len(runs) == 4  # 2 labels x 1 split x 2 methods x 1 seed
    assert set(runs.method) == {"majority", "gnn"}
    assert "| gnn |" in (ROOT / "results" / out / "summary.md").read_text()
