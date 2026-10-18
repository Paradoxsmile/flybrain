"""Partial-volume copies, coverage-weighted type priors and EM prior estimation."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from flybrain.adapt import em_prior
from flybrain.coverage import coverage_weights
from flybrain.loading import degrade_tables, load_graph

ROOT = Path(__file__).resolve().parents[1]


def test_coverage_weights_from_neuropils(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    pd.DataFrame(
        {
            "pre_root_id": [1, 1, 2, 3, 4],
            "post_root_id": [2, 3, 3, 1, 1],
            "neuropil": ["ME_L", "SMP_R", "LO_R", "UNASGD", "MB_CA_L"],
            "syn_count": [90, 10, 50, 7, 5],
            "nt_type": ["ACH"] * 5,
        }
    ).to_csv(raw / "connections.csv", index=False)
    w = coverage_weights(
        np.array([1, 2, 3, 4, 5]), "central_brain", raw_dir=raw, processed_dir=tmp_path
    )
    # assigned synapses in central neuropils: 1: 15 of 105 (14 %); 2: none; 3: 10 of 60; 4: all;
    # 5: no synapses
    assert w.tolist() == [1.0, 0.0, 1.0, 1.0, 0.0]
    assert (tmp_path / "fafb_neuropil_synapses.parquet").exists()


def test_partial_volume_and_cropping():
    neurons = pd.DataFrame(
        {
            "root_id": [1, 2, 3, 4],
            "super_class": ["optic", "central", "central", "sensory"],
            "cell_type": ["T4a", "KCab", "KCg-m", "R1-6"],
        }
    )
    conn = pd.DataFrame(
        {"pre_root_id": [2, 3, 2], "post_root_id": [3, 2, 1], "syn_count": [100, 100, 100]}
    )
    n, c = degrade_tables(neurons, conn, exclude_super_classes=("optic",), exclude_types=("R1-6",))
    assert n.root_id.tolist() == [2, 3] and c.syn_count.tolist() == [100, 100, 100]
    _, c = degrade_tables(
        neurons, conn, crop_fraction=1.0, crop_retention=0.1, seed=0, min_count=None
    )
    assert (c.syn_count < 10).all()  # both ends cropped: about 1 % of the synapses remain
    g = load_graph("cell_type", synthetic=True, dataset="mcns", crop_fraction=0.5)
    full = load_graph("cell_type", synthetic=True, dataset="mcns")
    assert g.num_nodes == full.num_nodes and g.edge_weight.sum() < 0.8 * full.edge_weight.sum()


def test_em_prior_shrinks_a_missing_class():
    rng = np.random.default_rng(0)
    y = rng.choice(3, size=3000, p=[0.6, 0.4, 0.0])
    logits = rng.normal(0, 1, (3000, 3))
    logits[np.arange(3000), y] += 2.0
    p = np.exp(logits)
    p /= p.sum(1, keepdims=True)
    pi, w = em_prior(p, np.ones(3) / 3)
    assert abs(pi[0] - 0.6) < 0.05 and pi[2] < 0.01
    np.testing.assert_allclose(w, pi * 3)


def test_label_free_with_em_prior_on_a_partial_copy():
    models = "pytest_transfer_lf"
    if not (ROOT / "results" / models / "super_class/gnn/s0/model.pt").exists():
        train = [sys.executable, str(ROOT / "experiments/b_gnn_celltypes/transfer.py")]
        train += ["--synthetic", "--labels", "super_class", "--methods", "gnn", "--seeds", "0"]
        train += ["--variants", "density", "--epochs", "3", "--out", models]
        subprocess.run(train, check=True, cwd=ROOT)
    out = "pytest_label_free_em"
    run = [sys.executable, str(ROOT / "experiments/b_gnn_celltypes/label_free.py"), "--synthetic"]
    run += ["--target", "mcns", "--target-crop", "0.2", "--prior", "em", "--models", models]
    run += ["--labels", "super_class", "--select", "size_agreement", "--out", out]
    subprocess.run(run, check=True, cwd=ROOT)
    sel = json.loads((ROOT / "results" / out / "super_class/selected.json").read_text())
    assert sel["prior"] == "em" and sel["target_crop"] == 0.2


def test_coverage_normalization_matches_the_covered_source():
    import importlib.util

    import torch
    from torch_geometric.data import Data

    spec = importlib.util.spec_from_file_location(
        "label_free", ROOT / "experiments/b_gnn_celltypes/label_free.py"
    )
    lf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lf)
    rng = np.random.default_rng(0)
    src = Data(x=torch.from_numpy(rng.normal(0, 1, (200, 3))).float())
    src.x[:100] += 5.0  # the first half lies inside the covered volume
    covered = np.r_[np.ones(100), np.zeros(100)]
    tgt = Data(x=torch.from_numpy(rng.normal(-3, 2, (80, 3))).float())
    train = torch.ones(200, dtype=torch.bool)
    x = lf.target_features(tgt, src, train, "coverage", covered)
    inside = lf.target_features(Data(x=src.x[:100]), src, train, "source", None)
    torch.testing.assert_close(x.mean(0), inside.mean(0), atol=1e-4, rtol=0)
    torch.testing.assert_close(x.std(0), inside.std(0), atol=1e-3, rtol=1e-3)
