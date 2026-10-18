"""Hemibrain preprocessing and label-free transfer to it on synthetic graphs."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as pf

from flybrain.hemibrain import (
    EDGES_FILE,
    META_FILE,
    brain_neurons,
    build_hemibrain_cache,
    load_hemibrain_tables,
)

ROOT = Path(__file__).resolve().parents[1]


def _meta() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "hemibrain_121_id": [11, 12, 13, 14],
            "instance": ["CL141_L", "FS4C(FQ12)_C4_R", "LC4", "x_R"],
            "cell_type": ["CL141", "FS4C", "LC4", None],
            "region": ["central_brain", "central_brain", "optic_lobe", None],
            "super_class": [
                "central_brain_intrinsic",
                "central_brain_intrinsic",
                "visual_projection",
                "glia_like",
            ],
            "status": ["Traced", "Traced", "Traced", "Orphan"],
            "cropped": [False, True, None, False],
        }
    )


def test_brain_neurons_selection_and_mapping():
    n = brain_neurons(_meta())
    assert n.root_id.tolist() == [11, 12, 13]  # orphans dropped
    assert n.super_class.tolist() == ["central", "central", "visual_projection"]
    assert n.side.tolist() == ["left", "right", ""]
    assert n.cropped.tolist() == ["", "cropped", ""]


def test_hemibrain_cache_and_variants(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    _meta().to_feather(raw / META_FILE)
    edges = pa.table(
        {
            "pre": ["11", "12", "12", "11", "13"],  # 12 -> 13 repeated, as in the harmonized file
            "post": ["12", "13", "13", "14", "11"],
            "count": pa.array([4, 5, 5, 9, 20], pa.int32()),
        }
    )
    pf.write_feather(edges, raw / EDGES_FILE)
    _, c = build_hemibrain_cache(raw, tmp_path)
    assert c.syn_count.tolist() == [5, 20]  # copies, below 5 and to an orphan dropped
    for variant, weights in {"raw": [5.0, 20.0], "scaled": [5 * 0.94, 20 * 0.94]}.items():
        np.testing.assert_allclose(
            load_hemibrain_tables(variant, raw, tmp_path)[1].syn_count, weights
        )


def test_label_free_to_hemibrain_synthetic_runs():
    models = "pytest_transfer_lf"
    if not (ROOT / "results" / models / "super_class/gnn/s0/model.pt").exists():
        train = [sys.executable, str(ROOT / "experiments/b_gnn_celltypes/transfer.py")]
        train += ["--synthetic", "--labels", "super_class", "--methods", "gnn", "--seeds", "0"]
        train += ["--variants", "density", "--epochs", "3", "--out", models]
        subprocess.run(train, check=True, cwd=ROOT)
    out = "pytest_label_free_hemibrain"
    run = [sys.executable, str(ROOT / "experiments/b_gnn_celltypes/label_free.py"), "--synthetic"]
    run += ["--target", "hemibrain", "--models", models, "--labels", "super_class"]
    run += ["--select", "size_agreement", "--out", out]
    subprocess.run(run, check=True, cwd=ROOT)
    sel = json.loads((ROOT / "results" / out / "super_class/selected.json").read_text())
    assert sel["target"] == "hemibrain" and sel["select"] == "size_agreement"
