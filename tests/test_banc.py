"""BANC preprocessing, label alignment with FAFB, and label-free transfer to BANC on synthetic
graphs."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as pf
import pytest
import torch
from torch_geometric.data import Data

from flybrain.banc import EDGES_FILE, META_FILE, brain_neurons, build_banc_cache, load_banc_tables
from flybrain.loading import load_graph

ROOT = Path(__file__).resolve().parents[1]


def _meta() -> pd.DataFrame:
    n = 8
    return pd.DataFrame(
        {
            "banc_888_id": [str(720575940000000000 + i) for i in range(1, n + 1)],
            "proofread": ["TRUE", "TRUE", "FALSE", "FALSE", "TRUE", "TRUE", "TRUE", "TRUE"],
            "roughly_proofread": ["FALSE", "FALSE", "TRUE", "FALSE"] + ["FALSE"] * 4,
            "region": [
                "optic_lobe",
                "central_brain",
                "central_brain",
                "central_brain",
                "ventral_nerve_cord",
                "ventral_nerve_cord",
                "central_brain",
                None,
            ],
            "super_class": [
                "optic_lobe_intrinsic",
                "central_brain_intrinsic",
                "visceral_circulatory",
                "central_brain_intrinsic",
                "ventral_nerve_cord_intrinsic",
                "ascending",
                "glia",
                None,
            ],
            "cell_type": ["T4c", "KCab", None, "x", "y", "AN1", None, None],
            "fafb_cell_type": ["auto:L5", "KCg-m", "DSKMP3", "x", "y", "z", None, None],
            "side": ["left", "right", None, "left", "left", "right", "left", None],
        }
    )


def test_brain_neurons_selection_and_mapping():
    n = brain_neurons(_meta())
    ids = n.root_id.to_numpy() - 720575940000000000
    # unproofread, nerve-cord intrinsic, glia and region-less neurons are dropped;
    # ascending neurons are kept without a brain region
    assert ids.tolist() == [1, 2, 3, 6]
    assert n.super_class.tolist() == ["optic", "central", "endocrine", "ascending"]
    assert n.region.tolist() == ["optic_lobe", "central_brain", "central_brain", ""]
    assert n.side.tolist() == ["left", "right", "", "right"]
    # morphology-matched types only in the central brain (optic-lobe entries are placeholders)
    assert pd.isna(n.morph_type[0]) and n.morph_type.tolist()[1:3] == ["KCg-m", "DSKMP3"]


def test_banc_cache_and_variants(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    _meta().to_feather(raw / META_FILE)
    ids = [str(720575940000000000 + i) for i in (1, 2, 3, 6, 5)]
    edges = pa.table(
        {
            "pre": [ids[0], ids[0], ids[1], ids[2], ids[0]],
            "post": [ids[1], ids[2], ids[3], ids[0], ids[4]],  # last: to a nerve-cord neuron
            "count": pa.array([1, 2, 5, 9, 9], pa.int32()),
        }
    )
    pf.write_feather(edges, raw / EDGES_FILE)
    neurons, connections = build_banc_cache(raw, tmp_path)
    assert len(neurons) == 4 and connections.syn_count.tolist() == [2, 5, 9]
    expected = {"raw": [5.0, 9.0], "density": [2.0, 5.0, 9.0], "scaled": [5 * 3.63, 9 * 3.63]}
    for variant, weights in expected.items():
        _, c = load_banc_tables(variant, raw, tmp_path)
        np.testing.assert_allclose(c.syn_count.to_numpy(), weights)
    with pytest.raises(ValueError, match="variant"):
        load_banc_tables("nope", raw, tmp_path)


def test_synthetic_banc_shares_the_fafb_label_space():
    fafb = load_graph("super_class", synthetic=True, seed=3)
    banc = load_graph("super_class", synthetic=True, seed=3, dataset="banc")
    mcns = load_graph("super_class", synthetic=True, seed=3, dataset="mcns")
    assert banc.label_names == fafb.label_names
    assert not np.array_equal(banc.x, fafb.x) and not np.array_equal(banc.x, mcns.x)


def test_breakdown_by_region_and_morphology():
    spec = importlib.util.spec_from_file_location(
        "label_free", ROOT / "experiments/b_gnn_celltypes/label_free.py"
    )
    lf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lf)
    g = Data(y=torch.tensor([0, 1, 1, -1, 0]), num_nodes=5)
    g.label_names = ["A", "B"]
    g.region = np.array(["central_brain", "central_brain", "optic_lobe", "optic_lobe", ""])
    g.morph_type = np.array(["A", "A", "", "B", "B"])
    pred = np.array([0, 1, 0, 1, 1])
    out = lf.breakdown(g, pred, "cell_type")
    assert out["central_brain_n"] == 2 and out["central_brain_acc"] == 1.0
    assert out["optic_lobe_n"] == 1 and out["optic_lobe_acc"] == 0.0
    # morphology reference: A, A, B, B at nodes 0, 1, 3, 4 -> predictions right at 0, 3, 4
    assert out["morph_n"] == 4 and out["morph_acc"] == 0.75
    assert out["morph_agrees_with_label"] == pytest.approx(1 / 3)  # nodes 0, 1, 4 have both
    assert "morph_n" not in lf.breakdown(g, pred, "super_class")


def _train(models: str, source: str) -> None:
    cmd = [sys.executable, str(ROOT / "experiments/b_gnn_celltypes/transfer.py"), "--synthetic"]
    cmd += ["--source", source, "--labels", "super_class", "--methods", "gnn", "--seeds", "0"]
    cmd += ["--epochs", "3", "--out", models]
    if source == "fafb":
        cmd += ["--variants", "density"]
    subprocess.run(cmd, check=True, cwd=ROOT)


@pytest.mark.parametrize("source", ["fafb", "mcns"])
def test_label_free_to_banc_synthetic_runs(source):
    models = f"pytest_transfer_banc_{source}"
    _train(models, source)
    out = f"pytest_label_free_{source}_banc"
    run = [sys.executable, str(ROOT / "experiments/b_gnn_celltypes/label_free.py"), "--synthetic"]
    run += ["--source", source, "--target", "banc", "--models", models]
    run += ["--labels", "super_class", "--out", out]
    subprocess.run(run, check=True, cwd=ROOT)
    sel = json.loads((ROOT / "results" / out / "super_class/selected.json").read_text())
    assert sel["source"] == source and sel["target"] == "banc" and "refined_acc" in sel
    cand = pd.read_csv(ROOT / "results" / out / "super_class/candidates.csv")
    assert len(cand) == 2 * 5  # one synthetic variant each, 2 normalizations x 5 rho
