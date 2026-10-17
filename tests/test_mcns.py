"""MCNS preprocessing, label alignment with FAFB, and the transfer script on synthetic graphs."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from flybrain.features import normalize_features
from flybrain.loading import encode_labels, load_graph
from flybrain.mcns import brain_neurons, load_mcns_tables

ROOT = Path(__file__).resolve().parents[1]


def _annotations() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "bodyId": [1, 2, 3, 4, 5, 6, 7],
            "status": ["Traced", "Traced", "Orphan", "Traced", "Traced", "Traced", "Traced"],
            "superclass": [
                "ol_intrinsic",
                "cb_sensory",
                "cb_intrinsic",
                "vnc_motor",
                "ENS",
                None,
                "sensory_descending",
            ],
            "flywireType": ["T4c", "VS1,VS2", "x", "MNx", "y", "CB0001", None],
            "somaSide": ["L", None, "R", "R", "L", "M", None],
            "rootSide": [None, "R", None, None, None, None, None],
        }
    )


def test_brain_neurons_selection_and_mapping():
    n = brain_neurons(_annotations())
    # orphan, VNC and enteric neurons are dropped; unmapped superclasses stay as unlabeled nodes
    assert n.root_id.tolist() == [1, 2, 6, 7]
    assert n.super_class.tolist()[:2] == ["optic", "sensory"]
    assert n.super_class.isna().tolist()[2:] == [True, True]
    # a type list is not a single label, but is kept for set-based evaluation
    assert n.cell_type.tolist()[0] == "T4c" and pd.isna(n.cell_type.tolist()[1])
    assert n.flywire_types.tolist()[1] == "VS1,VS2"
    assert n.side.tolist() == ["left", "right", "center", ""]


def test_mcns_variants_threshold_and_scale(tmp_path):
    neurons = brain_neurons(_annotations())
    neurons.to_parquet(tmp_path / "mcns_neurons.parquet", index=False)
    pd.DataFrame(
        {"pre_root_id": [1, 1, 2], "post_root_id": [2, 6, 6], "syn_count": [5, 7, 9]}
    ).to_parquet(tmp_path / "mcns_connections.parquet", index=False)
    expected = {"raw": [5.0, 7.0, 9.0], "density": [7 * 0.581, 9 * 0.581], "janelia": [9 * 0.581]}
    for variant, weights in expected.items():
        _, c = load_mcns_tables(variant, tmp_path / "raw", tmp_path)
        np.testing.assert_allclose(c.syn_count.to_numpy(), weights)
    with pytest.raises(ValueError, match="variant"):
        load_mcns_tables("nope", tmp_path / "raw", tmp_path)


def test_encode_labels_in_a_given_label_space():
    codes, names = encode_labels(pd.Series(["b", "a", "c", None, "a"]), label_names=["a", "b"])
    assert names == ["a", "b"]
    assert codes.tolist() == [1, 0, -1, -1, 0]
    codes, names = encode_labels(pd.Series(["b", "a", "a", None]), min_class_size=2)
    assert names == ["a"] and codes.tolist() == [-1, 0, 0, -1]


def test_synthetic_mcns_shares_the_fafb_label_space():
    fafb = load_graph("super_class", synthetic=True, seed=3)
    mcns = load_graph("super_class", synthetic=True, seed=3, dataset="mcns")
    assert mcns.label_names == fafb.label_names
    assert mcns.num_nodes == fafb.num_nodes and not np.array_equal(mcns.x, fafb.x)


def test_transfer_synthetic_runs():
    out = "pytest_transfer"
    cmd = [
        sys.executable,
        str(ROOT / "experiments/b_gnn_celltypes/transfer.py"),
        "--synthetic",
        "--labels",
        "super_class",
        "--methods",
        "majority",
        "gnn",
        "--seeds",
        "0",
        "--variants",
        "density",
        "--epochs",
        "3",
        "--out",
        out,
    ]
    subprocess.run(cmd, check=True, cwd=ROOT)
    m = json.loads((ROOT / "results" / out / "super_class/gnn/s0/metrics.json").read_text())
    assert {"fafb_test_acc", "density_source_acc", "density_target_macro_f1"} <= set(m)
    assert (
        ROOT / "results" / out / "super_class/gnn/s0/predictions-density-target.parquet"
    ).exists()
    assert "| gnn |" in (ROOT / "results" / out / "summary.md").read_text(encoding="utf-8")

    # the saved model reproduces the saved predictions
    from flybrain.training import graph_inputs, load_model, predict

    model, cfg, names = load_model(ROOT / "results" / out / "super_class/gnn/s0/model.pt")
    tgt = load_graph("super_class", synthetic=True, seed=0, dataset="mcns")
    assert names == tgt.label_names
    x = normalize_features(tgt.x, "global")
    pred = predict(model, *graph_inputs(tgt, x, cfg, "cpu"))
    saved = pd.read_parquet(
        ROOT / "results" / out / "super_class/gnn/s0/predictions-density-target.parquet"
    )
    assert (np.array(names, dtype=object)[pred] == saved.pred.to_numpy()).mean() > 0.99


def test_label_free_synthetic_runs():
    models = "pytest_transfer_lf"
    train = [sys.executable, str(ROOT / "experiments/b_gnn_celltypes/transfer.py"), "--synthetic"]
    # one seed: the label order of synthetic graphs depends on the seed
    train += ["--labels", "super_class", "--methods", "gnn", "--seeds", "0"]
    train += ["--variants", "density", "--epochs", "3", "--out", models]
    subprocess.run(train, check=True, cwd=ROOT)
    run = [sys.executable, str(ROOT / "experiments/b_gnn_celltypes/label_free.py"), "--synthetic"]
    run += ["--models", models, "--labels", "super_class", "--out", "pytest_label_free"]
    subprocess.run(run, check=True, cwd=ROOT)
    out = ROOT / "results" / "pytest_label_free" / "super_class"
    sel = json.loads((out / "selected.json").read_text())
    assert {"chosen", "selected_acc", "refined_acc", "no_adaptation_acc"} <= set(sel)
    assert len(pd.read_csv(out / "candidates.csv")) == 2 * 5  # 2 normalizations x 5 rho


def test_reverse_transfer_and_label_free_synthetic_run():
    models = "pytest_transfer_rev"
    train = [sys.executable, str(ROOT / "experiments/b_gnn_celltypes/transfer.py"), "--synthetic"]
    train += ["--source", "mcns", "--labels", "super_class", "--methods", "gnn", "--seeds", "0"]
    train += ["--epochs", "3", "--out", models]
    subprocess.run(train, check=True, cwd=ROOT)
    m = json.loads((ROOT / "results" / models / "super_class/gnn/s0/metrics.json").read_text())
    assert {"mcns_test_acc", "fafb_source_acc", "fafb_target_acc"} <= set(m)
    run = [sys.executable, str(ROOT / "experiments/b_gnn_celltypes/label_free.py"), "--synthetic"]
    run += ["--source", "mcns", "--target", "fafb", "--models", models, "--labels", "super_class"]
    run += ["--out", "pytest_label_free_rev"]
    subprocess.run(run, check=True, cwd=ROOT)
    sel = json.loads((ROOT / "results/pytest_label_free_rev/super_class/selected.json").read_text())
    assert sel["source"] == "mcns" and sel["target"] == "fafb" and "refined_acc" in sel
