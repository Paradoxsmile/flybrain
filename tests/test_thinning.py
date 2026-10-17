"""Thinned graph copies and the thinning experiment on synthetic graphs."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from flybrain.features import degree_features
from flybrain.loading import load_graph
from flybrain.splits import make_split
from flybrain.thinning import thin_graph
from flybrain.training import fit_gnn

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


def test_degraded_copies_keep_the_label_space():
    full = load_graph("cell_type", synthetic=True)
    drop = full.label_names[:2]
    for dataset in ("fafb", "mcns", "banc"):
        g = load_graph("cell_type", synthetic=True, dataset=dataset)
        d = load_graph(
            "cell_type", synthetic=True, dataset=dataset, retention=0.5, exclude_types=tuple(drop)
        )
        assert d.label_names == full.label_names
        assert d.num_nodes < g.num_nodes and d.edge_weight.numel() < g.edge_weight.numel()
        assert not np.isin(d.y.numpy(), [0, 1]).any()  # excluded types are gone
        assert (d.edge_weight.numpy() >= 5).all()


def test_mean_builder_with_new_weights_and_density_augmentation():
    from flybrain.models import MeanBuilder

    g = load_graph("cell_type", synthetic=True)
    t = thin_graph(g, 0.5, min_count=1, seed=0)
    full = MeanBuilder(g.edge_index, g.num_nodes, g.edge_weight)
    keep = torch.ones(g.edge_weight.numel(), dtype=torch.bool)
    c = torch.from_numpy(
        np.random.default_rng(0).binomial(g.edge_weight.numpy().astype(int), 0.5)
    ).float()
    keep = c > 0
    mean_in, _ = full(True, keep, c)
    direct, _ = MeanBuilder(g.edge_index[:, keep], g.num_nodes, c[keep])(True)
    h = torch.randn(g.num_nodes, 4)
    torch.testing.assert_close(mean_in(h), direct(h))
    assert t.edge_weight.numel() == keep.sum().item()  # same realization as thin_graph
    cfg = {"seed": 0, "hidden_dim": 8, "num_layers": 2, "dropout": 0.0, "lr": 0.01}
    cfg.update(weight_decay=0.0, epochs=3, thin_aug=0.3, class_weight="none")
    train, val, _ = make_split(g, "random", seed=0)
    fit = fit_gnn(cfg, g, train, val, "cpu")
    assert len(fit.history) == 3


def test_label_free_on_a_degraded_copy_and_selection_rules():
    models = "pytest_transfer_lf"  # trained by test_mcns.py if missing
    if not (ROOT / "results" / models / "super_class/gnn/s0/model.pt").exists():
        train = [sys.executable, str(ROOT / "experiments/b_gnn_celltypes/transfer.py")]
        train += ["--synthetic", "--labels", "super_class", "--methods", "gnn", "--seeds", "0"]
        train += ["--variants", "density", "--epochs", "3", "--out", models]
        subprocess.run(train, check=True, cwd=ROOT)
    out = "pytest_label_free_degraded"
    run = [sys.executable, str(ROOT / "experiments/b_gnn_celltypes/label_free.py"), "--synthetic"]
    run += ["--source", "fafb", "--target", "fafb", "--target-retention", "0.5"]
    run += ["--max-weight", "10", "--models", models, "--labels", "super_class", "--out", out]
    subprocess.run(run, check=True, cwd=ROOT)
    c = pd.read_csv(ROOT / "results" / out / "super_class/candidates.csv")
    assert {"n_types", "top_share", "ref_top_share", "size_js", "seed_agreement"} <= set(c)
    rules = [sys.executable, str(ROOT / "experiments/b_gnn_celltypes/selection_rules.py"), out]
    text = subprocess.run(rules, check=True, cwd=ROOT, capture_output=True, text=True).stdout
    assert "consistency_guarded" in text
