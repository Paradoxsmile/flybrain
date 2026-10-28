"""The nerve-cord test's data (docs/vnc_rare_protocol.md): MANC as a source, the BANC and MCNS
nerve cords as targets in MANC's label space, the passing-neuron rules, and a synthetic
label-free MANC -> cord run."""

from __future__ import annotations

import importlib
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
import yaml

from flybrain import banc, mcns
from flybrain.loading import build_graph, drop_passing_pairs, drop_self_connections, load_graph
from flybrain.manc import (
    EDGES_FILE,
    META_FILE,
    build_manc_cache,
    cord_neurons,
    load_manc_tables,
    manc_type_names,
)
from flybrain.splits import make_split
from flybrain.training import fit_gnn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/b_gnn_celltypes"))  # the scripts import each other


def _manc_meta() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "manc_121_id": ["10", "11", "12", "13", "14", "15"],
            "region": ["ventral_nerve_cord"] * 3 + ["neck_connective"] * 2 + ["ventral_nerve_cord"],
            "side": ["left", "right", None, "left", "right", "center"],
            "super_class": [
                "ventral_nerve_cord_intrinsic",
                "ventral_nerve_cord_intrinsic",
                "sensory",
                "ascending",
                "descending",
                "motor",
            ],
            "cell_type": ["IN01A001", "IN01A001", "SNxxxx", "AN07B004", "DNg100", "MNa,MNb"],
        }
    )


def test_manc_type_names_and_cord_neurons():
    names = manc_type_names(pd.Series(["IN01A001", "SNxxxx", "SNppxx", "SNxx03", "A,B", None]))
    # placeholders end in "xx"; numbered sensory types such as SNxx03 are real types
    assert names.tolist()[0] == "IN01A001" and names.tolist()[3] == "SNxx03"
    assert names.isna().tolist() == [False, True, True, False, True, True]
    n = cord_neurons(_manc_meta())
    assert n.root_id.tolist() == [10, 11, 12, 13, 14, 15]
    assert n.side.tolist() == ["left", "right", "", "left", "right", "center"]
    assert n.passing.tolist() == [False, False, False, True, True, False]
    assert n.cell_type.isna().tolist() == [False, False, True, False, False, True]


def test_drop_passing_pairs():
    neurons = pd.DataFrame({"root_id": [1, 2, 3], "passing": [True, True, False]})
    c = pd.DataFrame(
        {"pre_root_id": [1, 1, 3, 2], "post_root_id": [2, 3, 1, 1], "syn_count": [5, 6, 7, 8]}
    )
    assert drop_passing_pairs(neurons, c).syn_count.tolist() == [6, 7]  # 1 <-> 2 dropped
    assert drop_passing_pairs(neurons.drop(columns="passing"), c).equals(c)


def test_drop_self_connections():
    c = pd.DataFrame({"pre_root_id": [1, 2, 3], "post_root_id": [1, 3, 3], "syn_count": [5, 6, 7]})
    assert drop_self_connections(c).syn_count.tolist() == [6]


def test_manc_cache_dedupes_thresholds_and_drops_passing_pairs(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    _manc_meta().to_feather(raw / META_FILE)
    edges = pa.table(
        {
            "pre": ["10", "10", "11", "13", "13", "99", "10", "12"],
            "post": ["11", "11", "12", "14", "10", "10", "12", "12"],  # 13 -> 14: both passing
            "count": pa.array([7, 7, 9, 8, 6, 9, 2, 20], pa.int32()),  # 99: not a neuron
        }
    )
    pf.write_feather(edges, raw / EDGES_FILE)
    neurons, connections = build_manc_cache(raw, tmp_path)
    assert len(neurons) == 6
    assert connections.syn_count.tolist() == [7, 9, 8, 6, 2]  # copy, unknown id, 12 -> 12 dropped
    n, c = load_manc_tables("raw", raw, tmp_path)
    assert c.syn_count.tolist() == [7.0, 9.0, 6.0]  # >= 5, without 13 -> 14
    with pytest.raises(ValueError, match="variant"):
        load_manc_tables("density", raw, tmp_path)


def test_banc_cord_neurons_and_references():
    meta = pd.DataFrame(
        {
            "banc_888_id": ["1", "2", "3", "4", "5", "6"],
            "proofread": ["TRUE", "TRUE", "TRUE", "FALSE", "TRUE", "TRUE"],
            "roughly_proofread": ["FALSE"] * 6,
            "region": [
                "ventral_nerve_cord",
                "central_brain",
                "central_brain",
                "ventral_nerve_cord",
                "ventral_nerve_cord",
                "ventral_nerve_cord",
            ],
            "super_class": [
                "ventral_nerve_cord_intrinsic",
                "descending",
                "central_brain_intrinsic",
                "ascending",
                "glia",
                "ascending",
            ],
            "manc_cell_type": ["IN01A001", "DNxl126", None, "AN1", None, "AN07B004"],
            "cell_type": ["IN01A001", "DNge149", "CB1", "AN1", None, "AN07B004"],
            "side": ["left", "right", "left", "left", None, None],
        }
    )
    n = banc.cord_neurons(meta)
    # cord neurons and descending neurons (soma in the brain) kept; brain, glia and
    # unproofread neurons dropped
    assert n.root_id.tolist() == [1, 2, 6]
    assert n.cell_type.tolist() == ["IN01A001", "DNxl126", "AN07B004"]  # reference A
    assert n.alt_type.tolist() == ["IN01A001", "DNge149", "AN07B004"]  # reference B
    assert n.passing.tolist() == [False, True, True]
    assert n.side.tolist() == ["left", "right", ""]


def test_mcns_cord_neurons():
    a = pd.DataFrame(
        {
            "bodyId": [1, 2, 3, 4, 5, 6],
            "status": ["Traced", "Traced", "Traced", "Orphan", "Traced", "Traced"],
            "superclass": [
                "vnc_intrinsic",
                "ascending_neuron",
                "cb_intrinsic",
                "vnc_motor",
                "vnc_sensory",
                "descending_neuron",
            ],
            "mancType": ["IN01A001", "AN07B004", None, "MN1", "SNta02,SNta09", "DNg100"],
            "somaSide": ["L", "R", "L", "R", None, "L"],
            "rootSide": [None, None, None, None, "R", None],
        }
    )
    n = mcns.cord_neurons(a)
    assert n.root_id.tolist() == [1, 2, 5, 6]
    assert n.super_class.tolist() == [
        "ventral_nerve_cord_intrinsic",
        "ascending",
        "sensory",
        "descending",
    ]
    assert pd.isna(n.cell_type.tolist()[2]) and n.manc_types.tolist()[2] == "SNta02,SNta09"
    assert n.side.tolist() == ["left", "right", "right", "left"]
    assert n.passing.tolist() == [False, True, False, True]


def test_cord_variants_threshold(tmp_path):
    meta = pd.DataFrame(
        {"root_id": [1, 2, 3], "cell_type": ["A", "B", "A"], "super_class": ["x"] * 3}
    ).assign(side="left", passing=[True, True, False])
    meta.to_parquet(tmp_path / "mcns_cord_neurons.parquet", index=False)
    pd.DataFrame(
        {"pre_root_id": [1, 1, 3], "post_root_id": [2, 3, 1], "syn_count": [9, 4, 5]}
    ).to_parquet(tmp_path / "mcns_cord_connections.parquet", index=False)
    _, c = mcns.load_mcns_cord_tables("raw", tmp_path, tmp_path)
    assert c.syn_count.tolist() == [5.0]  # 1 -> 2 is a passing pair, 1 -> 3 below 5


def test_synthetic_cords_share_the_manc_label_space():
    manc = load_graph("cell_type", synthetic=True, seed=2, dataset="manc", source="manc")
    for target in ("banc_cord", "mcns_cord"):
        g = load_graph("cell_type", synthetic=True, seed=2, dataset=target, source="manc")
        assert g.label_names == manc.label_names
        src, dst = g.edge_index.numpy()
        assert not (g.passing[src] & g.passing[dst]).any()  # no passing pairs
        assert g.passing.any() and not np.array_equal(g.x, manc.x)
    fafb = load_graph("cell_type", synthetic=True, seed=2)
    assert not fafb.passing.any()  # derived from the (synthetic) super classes
    with pytest.raises(ValueError, match="source"):
        load_graph("cell_type", synthetic=True, dataset="banc_cord", source="mcns")


def test_brain_graphs_get_passing_from_the_super_class():
    neurons = pd.DataFrame(
        {"root_id": [1, 2, 3], "super_class": ["central", "descending", "sensory_ascending"]}
    )
    c = pd.DataFrame({"pre_root_id": [1], "post_root_id": [2], "syn_count": [5]})
    assert build_graph(neurons, c).passing.tolist() == [False, True, True]


def test_source_types_for_manc(tmp_path, monkeypatch):
    stratify = importlib.import_module("stratify")
    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True)
    pd.DataFrame(
        {
            "root_id": [1, 2, 3, 4, 5],
            "cell_type": ["IN1", "IN1", "AN1", "AN1", None],
            "super_class": ["ventral_nerve_cord_intrinsic"] * 2 + ["ascending"] * 2 + ["motor"],
        }
    ).to_parquet(processed / "manc_neurons.parquet", index=False)
    monkeypatch.setattr(stratify, "REPO_ROOT", tmp_path)
    t = stratify.source_types("manc")
    assert t["size"].to_dict() == {"AN1": 2, "IN1": 2}
    assert t.region.to_dict() == {"AN1": "ascending", "IN1": "intrinsic"}


def test_size_threshold_manc_names():
    st = importlib.import_module("size_threshold")
    assert st.label_free_dir("banc_cord", "size_agreement") == (
        "label_free_banc_cord_size_agreement"
    )
    st.set_source("manc")
    try:
        assert st.OUT == ROOT / "results" / "size_threshold_manc"
    finally:
        st.set_source("fafb")
    assert st.OUT == ROOT / "results" / "size_threshold"


def test_label_free_manc_to_cord_synthetic_runs():
    """GNNs trained on the synthetic MANC, mapped label-free to a synthetic cord in MANC's label
    space (label_free.py --source manc)."""
    models = "pytest_manc_models"
    cfg = yaml.safe_load((ROOT / "experiments/b_gnn_celltypes/configs/cell_type.yaml").read_text())
    cfg.update(label_col="super_class", min_class_size=1, epochs=3, seed=0, split="random")
    data = load_graph("super_class", synthetic=True, dataset="manc", source="manc")
    train, val, _ = make_split(data, "random", seed=0)
    fit = fit_gnn(cfg, data, train, val, torch.device("cpu"))
    d = ROOT / "results" / models / "super_class" / "gnn" / "s0"
    d.mkdir(parents=True, exist_ok=True)
    state = {"state_dict": fit.model.state_dict(), "cfg": cfg, "label_names": data.label_names}
    torch.save(state, d / "model.pt")
    out = "pytest_label_free_manc_banc_cord"
    run = [sys.executable, str(ROOT / "experiments/b_gnn_celltypes/label_free.py"), "--synthetic"]
    run += ["--source", "manc", "--target", "banc_cord", "--models", models]
    run += ["--labels", "super_class", "--select", "size_agreement", "--out", out]
    subprocess.run(run, check=True, cwd=ROOT)
    sel = json.loads((ROOT / "results" / out / "super_class/selected.json").read_text())
    assert sel["source"] == "manc" and sel["target"] == "banc_cord" and "refined_acc" in sel
    bad = run[:4] + ["--source", "manc", "--target", "banc", "--models", models]
    assert subprocess.run(bad, cwd=ROOT, capture_output=True).returncode != 0
