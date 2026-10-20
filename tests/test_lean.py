"""Memory-lean training and label-free mapping (lean_training.py, lean_label_free.py) give the
same results as flybrain.training.fit_gnn, flybrain.adapt.prior_weights and label_free.py."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from flybrain.adapt import prior_weights
from flybrain.loading import load_graph
from flybrain.models import DirectedSAGE
from flybrain.splits import make_split
from flybrain.training import fit_gnn, graph_inputs, predict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/b_gnn_celltypes"))  # the scripts import each other
label_free = importlib.import_module("label_free")
lean_training = importlib.import_module("lean_training")
lean_label_free = importlib.import_module("lean_label_free")

CFG = {
    "seed": 0,
    "feature_norm": "global",
    "drop_edge": 0.0,
    "bidirectional": True,
    "edge_weight": "count",
    "class_weight": "balanced",
    "hidden_dim": 8,  # <= 8 classes: the lean head needs out_dim >= in_dim
    "num_layers": 3,
    "dropout": 0.3,
    "lr": 0.01,
    "weight_decay": 0.0005,
    "epochs": 20,
}


def test_fit_gnn_lean_matches_fit_gnn():
    data = load_graph("cell_type", synthetic=True)
    train, val, _ = make_split(data, "random", seed=0)
    torch.manual_seed(0)
    ref = fit_gnn(CFG, data, train, val, "cpu")
    torch.manual_seed(0)
    lean = lean_training.fit_gnn_lean(CFG, data, train, val, "cpu", chunk=32)  # several chunks
    assert len(lean.history) == len(ref.history) == CFG["epochs"]
    for a, b in zip(lean.history, ref.history, strict=True):
        assert a["loss"] == pytest.approx(b["loss"], rel=1e-5)
        assert a["val_acc"] == pytest.approx(b["val_acc"], abs=1e-6)  # fit_gnn: float32 mean
        assert a["val_macro_f1"] == pytest.approx(b["val_macro_f1"], abs=1e-12)
    ref_state, lean_state = ref.model.state_dict(), lean.model.state_dict()
    assert ref_state.keys() == lean_state.keys()
    for k, v in ref_state.items():
        torch.testing.assert_close(lean_state[k], v, rtol=1e-5, atol=1e-5)
    pred = predict(ref.model, ref.x, ref.adj)
    assert (lean_training.predict_lean(lean.model, lean.x, lean.adj) == pred).all()
    rows = torch.tensor([5, 0, 17])
    out = lean_training.predict_rows(lean.model, lean.x, lean.adj, rows, chunk=2)
    assert (out.numpy() == pred[rows.numpy()]).all()


def test_fit_gnn_lean_rejects_augmentation():
    data = load_graph("cell_type", synthetic=True)
    train, val, _ = make_split(data, "random", seed=0)
    with pytest.raises(ValueError, match="augmentation"):
        lean_training.fit_gnn_lean({**CFG, "drop_edge": 0.1}, data, train, val, "cpu")


def _probabilities(n: int = 500, k: int = 12, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.dirichlet(np.full(k, 0.3), size=n).astype(np.float32)


@pytest.mark.parametrize("max_weight", [None, 3.0])
def test_prior_weights_lean_matches_prior_weights(max_weight):
    probs = _probabilities()
    target = np.random.default_rng(1).dirichlet(np.ones(probs.shape[1])) * len(probs)
    ref = prior_weights(probs, target, max_weight=max_weight)
    lean = lean_label_free.prior_weights_lean(probs, target, max_weight=max_weight, chunk=64)
    assert not np.allclose(ref, 1.0)  # the weights did move
    np.testing.assert_allclose(lean, ref, rtol=1e-8)
    if max_weight is not None:
        assert ref.max() <= max_weight and ref.min() >= 1 / max_weight


class _Rows:
    """A dense probability matrix behind LazyProbs's interface (rows, chunk, len)."""

    def __init__(self, p: np.ndarray, chunk: int):
        self.p, self.chunk, self.shape = p, chunk, p.shape

    def __len__(self) -> int:
        return len(self.p)

    def rows(self, start: int, stop: int) -> np.ndarray:
        return self.p[start:stop].copy()


def test_adjusted_lean_matches_label_free():
    probs = [_probabilities(seed=s) for s in range(3)]
    weights = [np.random.default_rng(10 + s).lognormal(size=12) for s in range(3)]
    for rho in (0.0, 0.5, 1.0):
        ens, seeds = label_free.adjusted(probs, weights, rho)
        lens, lseeds = lean_label_free.adjusted([_Rows(p, 64) for p in probs], weights, rho)
        assert (lens == ens).all()
        for a, b in zip(lseeds, seeds, strict=True):
            assert (a == b).all()


def test_lazy_probs_match_the_model():
    data = load_graph("cell_type", synthetic=True)
    torch.manual_seed(0)
    model = DirectedSAGE(data.x.size(1), 8, len(data.label_names), 2, 0.3, True)
    x = (data.x - data.x.mean(0)) / data.x.std(0)
    lazy = lean_label_free.LazyProbs(model, CFG, data, x, chunk=64)
    xx, adj = graph_inputs(data, x, CFG, "cpu")
    with torch.no_grad():
        dense = torch.softmax(model(xx, *adj), dim=1).numpy()
    assert lazy.shape == dense.shape
    np.testing.assert_allclose(lazy.dense(), dense, rtol=1e-5, atol=1e-6)
    rows = np.random.default_rng(0).random(len(lazy))
    np.testing.assert_allclose(
        lean_label_free.column_sums(lazy, rows), rows @ dense.astype(np.float64), rtol=1e-5
    )
