"""flybrain.ntac must reproduce the official NTAC package step by step."""

from __future__ import annotations

import numba
import numpy as np
import pytest
import scipy.sparse as sp
import torch

from flybrain.ntac import ntac_seeded

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
