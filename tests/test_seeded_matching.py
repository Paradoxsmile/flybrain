"""Seeded graph matching (seeded_matching.py) on small synthetic graphs with a hidden matching."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/b_gnn_celltypes"))
sys.path.insert(0, str(ROOT / "tests"))
gm = importlib.import_module("graph_matching")
gmm = importlib.import_module("graph_matching_mcns")
sm = importlib.import_module("seeded_matching")
tgm = importlib.import_module("test_graph_matching")


def test_nblast_ids_keep_18_digit_ids_exact():
    big = "720575940612821727"  # beyond float64 precision
    meta = pd.DataFrame(
        {"hit": [big, "31170.0", None, "not an id", " 12 "]},
        index=np.array([1, 2, 3, 4, 5], dtype=np.int64),
    )
    out = sm.nblast_ids(meta, "hit", np.array([5, 1, 2, 3, 4, 9], dtype=np.int64))
    assert out.dtype == np.int64
    assert out.tolist() == [12, int(big), 31170, -1, -1, -1]


def test_find_seeds_keeps_unique_hits_in_the_block():
    source_ids = np.array([100, 101, 102, 103, 104], dtype=np.int64)
    f_block = np.array([1, 2, 3])  # source neurons 101-103
    m_block = np.array([0, 1, 2, 3, 4])
    # hits: 102 (unique) / 101 twice (ambiguous) / 104 (other block) / none
    hit = np.array([102, 101, 101, 104, -1], dtype=np.int64)
    rows, cols = sm.find_seeds(f_block, m_block, hit, source_ids)
    assert rows.tolist() == [1] and cols.tolist() == [0]  # 102 is row 1 of the block


def test_seeded_lmo_contains_the_seeds_and_is_optimal_on_the_rest():
    rng = np.random.default_rng(0)
    n = 12
    g = rng.standard_normal((n, n))
    seeds_r, seeds_c = np.array([0, 5]), np.array([7, 2])
    lmo = sm.SeededLMO(seeds_r, seeds_c, n)
    rows, cols = lmo(g)
    assert sorted(rows) == sorted(cols) == list(range(n))
    pairs = dict(zip(rows.tolist(), cols.tolist(), strict=True))
    assert pairs[0] == 7 and pairs[5] == 2
    # brute check against an assignment with the seeds forced by a large bonus
    forced = g.copy()
    forced[seeds_r, seeds_c] += 1e6
    r, c = gmm._lap(forced)
    assert g[rows, cols].sum() == pytest.approx(g[r, c].sum() - 0.0, rel=1e-9)
    p = lmo.start()
    np.testing.assert_allclose(p.sum(axis=0), 1.0, atol=1e-6)
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-6)
    assert (p[seeds_r, seeds_c] == 1).all()


def test_seeded_faq_keeps_the_seeds_and_recovers_the_matching():
    sizes = (70, 60)
    f, m, f_blocks, m_blocks, truth = tgm._two_brains(sizes)
    n = max(sizes)
    F = gmm.as32(gm.subgraph(f, *f_blocks, n))
    M = gmm.as32(gm.subgraph(m, *m_blocks, n))
    lmos = []
    for t, k in zip(truth, sizes, strict=True):
        cols = np.arange(0, k, 5)  # every fifth candidate is a correct seed
        lmos.append(sm.SeededLMO(t[cols], cols, n))
    matches = gmm.faq_blocks(F, M, [lmo.start() for lmo in lmos], iters=30, lmos=lmos)
    for match, t, k, lmo in zip(matches, truth, sizes, lmos, strict=True):
        assert sorted(match) == list(range(n))
        assert (match[lmo.cols] == lmo.rows).all()  # seeds unchanged
        assert (match[:k] == t).all()


@pytest.mark.parametrize(
    ("delta", "lo", "hi", "outcome"),
    [
        (0.05, 0.02, 0.08, "confirmed"),
        (0.02, 0.01, 0.03, "not confirmed: wiring adds less than 3 points"),
        (0.04, -0.01, 0.09, "inconclusive: the interval includes 0"),
        (-0.05, -0.08, -0.02, "failed: NBLAST alone is better"),
    ],
)
def test_decision_rule(delta, lo, hi, outcome):
    assert sm.decide(delta, lo, hi) == outcome
