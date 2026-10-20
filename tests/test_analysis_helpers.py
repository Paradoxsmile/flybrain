"""Pure helpers of the analysis scripts (stratify.py, size_threshold.py, label_effort.py,
estimate_accuracy.py) on hand-made inputs."""

from __future__ import annotations

import importlib
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/b_gnn_celltypes"))  # the scripts import each other
stratify = importlib.import_module("stratify")
size_threshold = importlib.import_module("size_threshold")
label_effort = importlib.import_module("label_effort")
estimate_accuracy = importlib.import_module("estimate_accuracy")


def test_size_bins():
    assert [stratify.size_bin(n) for n in (5, 19, 20, 49, 50, 999, 1000, 5000)] == [
        "< 20",
        "< 20",
        "20–49",
        "20–49",
        "50–199",
        "200–999",
        "≥ 1000",
        "≥ 1000",
    ]
    assert [size_threshold.size_bin(n) for n in (1, 2, 4, 5, 9, 10, 19, 20, 1000)] == [
        "1",
        "2–4",
        "2–4",
        "5–9",
        "5–9",
        "10–19",
        "10–19",
        "20–49",
        "≥ 1000",
    ]


def _types() -> pd.DataFrame:
    return pd.DataFrame(
        {"region": ["central brain", "sensory"], "size": [30, 300], "size_bin": ["20–49", "x"]},
        index=["A", "B"],
    )


def test_per_type_and_strata():
    pred = pd.DataFrame(
        {"label": ["A", "A", "B", "B", "B", "B"], "pred_x": ["A", "B", "B", "B", "A", "C"]}
    )
    t = stratify.per_type(pred, "x", _types()).set_index("cell_type")
    assert list(t.index) == ["A", "B"]  # C is predicted but no neuron has it
    assert t.loc["A", ["n", "correct", "predicted"]].tolist() == [2, 1, 2]
    assert t.loc["B", ["n", "correct", "predicted"]].tolist() == [4, 2, 3]
    assert t.f1.tolist() == pytest.approx([0.5, 4 / 7])
    s = stratify.strata(t.reset_index()).set_index("stratum")
    assert s.loc["all", "acc"] == 0.5 and s.loc["all", "macro_f1"] == pytest.approx(
        (0.5 + 4 / 7) / 2
    )
    assert s.loc["central brain", "n_neurons"] == 2 and s.loc["central brain", "acc"] == 0.5
    assert s.loc["sensory", "points"] == pytest.approx(2 / 6)  # correct / all neurons


def test_hand_labels_per_type():
    ref = np.array([0] * 10 + [1] * 3 + [-1] * 5)
    m = label_effort.hand_labels(ref, 50, seed=0)
    assert m[ref == 0].sum() == 5 and m[ref == 1].sum() == 2 and not m[ref < 0].any()
    assert (label_effort.hand_labels(ref, 50, seed=0) == m).all()
    assert not label_effort.hand_labels(ref, 0, seed=0).any()
    assert label_effort.hand_labels(ref, 100, seed=1).sum() == 13


def test_crossing_interpolates_log_linearly():
    ks, acc = [1, 2, 5, 10], [0.5, 0.6, 0.7, 0.8]
    assert label_effort.crossing(ks, acc, 0.65) == pytest.approx(math.sqrt(10))  # half way 2 -> 5
    assert label_effort.crossing(ks, acc, 0.6) == pytest.approx(2.0)
    assert label_effort.crossing(ks, acc, 0.4) == 1
    assert math.isnan(label_effort.crossing(ks, acc, 0.9))


def test_weighted_fit_weights_every_run_equally():
    c = pd.DataFrame(
        {
            "run": ["a", "b", "b", "b"],
            "acc": [0.8, 0.2, 0.3, 0.4],
            "pairwise_agreement": [0.9, 0.5, 0.6, 0.7],
        }
    )
    assert estimate_accuracy.fit(c, "mean")[0] == pytest.approx((0.8 + 0.3) / 2)
    exact = c.assign(acc=0.1 + 0.5 * c.pairwise_agreement)
    np.testing.assert_allclose(estimate_accuracy.fit(exact, "agree"), [0.1, 0.5], atol=1e-12)
    test = pd.DataFrame({"pairwise_agreement": [0.2, 3.0]})
    assert estimate_accuracy.fit_predict(exact, test, "agree").tolist() == pytest.approx([0.2, 1.0])
    assert estimate_accuracy.fit_predict(exact, test, "agree0").tolist() == [0.2, 3.0]


def test_votes_and_error_flagging():
    n = pd.DataFrame(
        {
            "label": ["A"] * 10 + [None],
            "pred_refined": ["A"] * 8 + ["B", "B", "A"],
            "p_refined": [0.9] * 11,
            "seed0": ["A"] * 8 + ["B", "B", "A"],
            "seed1": ["A"] * 8 + ["A", "B", "A"],
            "seed2": ["A"] * 8 + ["A", "A", "B"],
        }
    )
    assert estimate_accuracy.votes(n, "refined").tolist() == pytest.approx(
        [1.0] * 8 + [1 / 3, 2 / 3, 2 / 3]
    )
    rows = pd.DataFrame(estimate_accuracy.flagging(n, "run", "refined"))
    assert (rows.n == 10).all() and (rows.error_rate == 0.2).all()  # unlabeled neuron excluded
    by_votes = rows[rows.score == "votes"].set_index("review")
    assert by_votes.loc[0.1, "caught"] == 0.5 and by_votes.loc[0.2, "caught"] == 1.0
    assert by_votes.loc[0.2, "precision"] == 1.0 and by_votes.loc[0.2, "oracle"] == 1.0
    unanimous = rows[rows.score == "unanimous_share"].iloc[0]
    assert unanimous.caught == 0.8 and unanimous.precision == 1.0
