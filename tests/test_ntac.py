"""flybrain.ntac must reproduce the official NTAC package step by step."""

from __future__ import annotations

import numba
import numpy as np
import pytest
import scipy.sparse as sp
import torch

from flybrain.ntac import agreement_seeds, ntac_from_predictions, ntac_seeded

ntac = pytest.importorskip("ntac")


def _graph(n=400, k=6, seed=0):
    """Directed weighted block graph; class 0 has no labeled neuron (as in a hemisphere split)."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, k, n)
    p = rng.uniform(0.0, 0.08, (k, k))[y][:, y]
    a = (rng.random((n, n)) < p) * rng.integers(1, 20, (n, n))
    np.fill_diagonal(a, 0)
    src, dst = np.nonzero(a)
    known = (rng.random(n) < 0.3) & (y != 0)
    return sp.csr_matrix(a.astype(float)), src, dst, a[src, dst], y, known


@pytest.mark.parametrize("lr", [1.0, 0.3])
def test_ntac_matches_official_package(lr):
    adj, src, dst, w, y, known = _graph()
    steps = 6
    # the package's similarity kernel writes to shared cells from parallel threads, which makes
    # it nondeterministic; with one thread it is deterministic and is the reference
    threads = numba.get_num_threads()
    numba.set_num_threads(1)
    try:
        official = ntac.Ntac(adj, labels=np.where(known, [f"c{c}" for c in y], "?"), lr=lr)
        expected = []
        for _ in range(steps):
            official.step()
            expected.append(np.array([int(s[1:]) for s in official.get_partition()]))
    finally:
        numba.set_num_threads(threads)

    ours = list(
        ntac_seeded(
            torch.tensor(np.stack([src, dst])),
            torch.tensor(w, dtype=torch.float),
            torch.tensor(y),
            torch.tensor(known),
            num_classes=6,
            lr=lr,
            steps=steps,
        )
    )
    for step, (got, want) in enumerate(zip(ours, expected, strict=True)):
        assert (got == want).mean() == 1.0, f"step {step}: {(got != want).sum()} labels differ"
    assert (ours[-1][known] == y[known]).all()


def test_agreement_seeds():
    preds = np.array([[0, 1, 2, 2], [0, 1, 1, 2], [0, 2, 1, 2]])
    vote, agree = agreement_seeds(preds)
    assert vote.tolist() == [0, 1, 1, 2]
    assert agree.tolist() == [True, False, False, True]


def test_ntac_from_predictions_seeds_known_and_unanimous():
    _, src, dst, w, y, known_mask = _graph()
    known = np.where(known_mask, y, -1)
    rng = np.random.default_rng(1)
    # three "models": correct, except that each one errs on its own random tenth of the nodes
    preds = np.stack([np.where(rng.random(len(y)) < 0.1, (y + 1) % 6, y) for _ in range(3)])
    labels, seeds = ntac_from_predictions(
        torch.tensor(np.stack([src, dst])), torch.tensor(w, dtype=torch.float), known, preds, 6
    )
    unanimous = (preds == preds[0]).all(axis=0)
    assert np.array_equal(seeds, known_mask | unanimous)
    assert (labels[known_mask] == y[known_mask]).all()  # known labels stay fixed
    assert (labels == y).mean() > (preds[0] == y).mean()  # NTAC fixes some model errors
