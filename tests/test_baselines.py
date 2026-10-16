"""Baselines on tiny hand-checkable graphs."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from flybrain.baselines import (
    class_mass_normalize,
    edge_homophily,
    label_propagation,
    partner_profiles,
    propagation_matrix,
)

ROOT = Path(__file__).resolve().parents[1]


def test_partner_profiles_shares_over_known_partners():
    # 0 -> 2 (3 syn), 1 -> 2 (1 syn), 3 -> 2 (4 syn, partner 3 not known), 2 -> 0 (2 syn)
    ei = torch.tensor([[0, 1, 3, 2], [2, 2, 2, 0]])
    w = torch.tensor([3.0, 1.0, 4.0, 2.0])
    y = torch.tensor([0, 1, 1, 0])
    known = torch.tensor([True, True, False, False])
    x = partner_profiles(ei, w, y, known, num_classes=2).toarray()
    assert x.shape == (4, 4)
    # neuron 2 inputs: known partners 0 (class 0, 3 syn) and 1 (class 1, 1 syn); 3 is ignored.
    # neuron 2 outputs: only to 0 (class 0)
    np.testing.assert_allclose(x[2], [0.75, 0.25, 1.0, 0.0])
    # neurons 0, 1, 3 only connect to neuron 2, whose class is not known
    np.testing.assert_allclose(x[[0, 1, 3]], np.zeros((3, 4)))


def test_label_propagation_follows_communities():
    # two triangles joined by one weak edge; one known neuron per triangle
    ei = torch.tensor([[0, 1, 2, 3, 4, 5, 2], [1, 2, 0, 4, 5, 3, 3]])
    w = torch.tensor([5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 1.0])
    y = torch.tensor([0, 0, 0, 1, 1, 1])
    known = torch.tensor([True, False, False, False, False, True])
    s = propagation_matrix(ei, w, 6)
    pred = label_propagation(s, y, known, num_classes=2, alpha=0.9).argmax(1)
    assert pred.tolist() == [0, 0, 0, 1, 1, 1]


def test_class_mass_normalize_matches_prior():
    y = torch.tensor([0, 0, 0, 1, -1, -1])
    known = torch.tensor([True, True, True, True, False, False])
    scores = torch.tensor([[1.0, 0.0]] * 4 + [[0.6, 0.4], [0.9, 0.1]])
    out = class_mass_normalize(scores, y, known)
    mass = out[~known].sum(0)
    assert torch.allclose(mass / mass.sum(), torch.tensor([0.75, 0.25]))


def test_edge_homophily():
    ei = torch.tensor([[0, 0, 1], [1, 2, 2]])
    w = torch.tensor([1.0, 3.0, 1.0])
    y = torch.tensor([0, 0, 1])
    h, chance = edge_homophily(ei, w, y)
    assert h == pytest.approx(1 / 5)
    assert 0 <= chance <= 1


@pytest.mark.parametrize("method", ["majority", "label_prop", "partner_profile", "ntac"])
def test_baseline_script_synthetic(method):
    name = f"pytest_baseline_{method}"
    cmd = [
        sys.executable,
        str(ROOT / "experiments/b_gnn_celltypes/baselines.py"),
        "--synthetic",
        "--method",
        method,
        "--run-name",
        name,
    ]
    subprocess.run(cmd, check=True, cwd=ROOT)
    metrics = json.loads((ROOT / "results" / name / "metrics.json").read_text())
    assert 0 <= metrics["test_acc"] <= 1
    assert (ROOT / "results" / name / "predictions.parquet").exists()
