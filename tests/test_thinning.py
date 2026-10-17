"""Thinned graph copies and the thinning experiment on synthetic graphs."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from flybrain.features import degree_features
from flybrain.loading import load_graph
from flybrain.thinning import thin_graph

ROOT = Path(__file__).resolve().parents[1]


def test_thin_graph_keeps_nodes_and_recomputes_features():
    g = load_graph("cell_type", synthetic=True)
    t = thin_graph(g, 0.5, min_count=3, seed=0)
    assert t.num_nodes == g.num_nodes and t.label_names == g.label_names
    assert np.array_equal(t.y.numpy(), g.y.numpy())
    w = t.edge_weight.numpy()
    assert (w >= 3).all() and len(w) < g.edge_weight.numel()
    assert w.sum() < 0.5 * g.edge_weight.sum().item()  # thinning plus the threshold
    np.testing.assert_allclose(
        t.x.numpy(), degree_features(t.num_nodes, t.edge_index, t.edge_weight), rtol=1e-6
    )
    # the original graph is untouched; the same seed gives the same copy
    assert g.edge_weight.numel() == load_graph("cell_type", synthetic=True).edge_weight.numel()
    assert np.array_equal(thin_graph(g, 0.5, 3, seed=0).edge_weight.numpy(), w)
    with pytest.raises(ValueError, match="retention"):
        thin_graph(g, 0.0)


def test_thinning_experiment_synthetic_runs():
    out = "pytest_thinning"
    cmd = [sys.executable, str(ROOT / "experiments/b_gnn_celltypes/thinning.py"), "--synthetic"]
    cmd += ["--epochs", "2", "--seeds", "0", "--retentions", "1", "0.5", "--out", out]
    subprocess.run(cmd, check=True, cwd=ROOT)
    grid = pd.read_csv(ROOT / "results" / out / "grid.csv")
    # 2 source x 2 target densities x 2 rho, plus the denser target thinned to the source
    assert len(grid) == 2 * 2 * 2 + 2
    assert set(
        grid[grid.strategy == "thinned_to_source"][["p_source", "p_target"]].itertuples(index=False)
    ) == {(0.5, 1.0)}
    assert (ROOT / "results" / out / "models/cell_type/p0.5/s0/model.pt").exists()
    assert "rho 1" in (ROOT / "results" / out / "summary.md").read_text(encoding="utf-8")
