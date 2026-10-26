"""Graph matching for the rarest cell types (graph_matching.py, graph_matching_mcns.py) on small
synthetic graphs with a hidden matching."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/b_gnn_celltypes"))  # the scripts import each other
gm = importlib.import_module("graph_matching")
gmm = importlib.import_module("graph_matching_mcns")


def _edges(rng, n: int, density: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Random directed edges among n abstract neurons, with synapse counts."""
    a = rng.random((n, n)) < density
    np.fill_diagonal(a, False)
    src, dst = np.nonzero(a)
    return src, dst, rng.integers(1, 20, len(src))


def _graph(src, dst, w, num_nodes: int) -> SimpleNamespace:
    return SimpleNamespace(
        edge_index=torch.tensor(np.stack([src, dst])),
        edge_weight=torch.tensor(w),
        num_nodes=num_nodes,
    )


def _bilateral(n: int, ipsi: bool, contra: bool, seed: int = 0):
    """2n neurons: n abstract neurons, each with a left copy (node a) and a right copy (node
    n + sigma[a]), wired mirror-symmetrically. Returns the graph and the true matching
    (right index r -> left index)."""
    rng = np.random.default_rng(seed)
    sigma = rng.permutation(n)
    left, right = np.arange(n), n + sigma
    src, dst, w = [], [], []
    if ipsi:
        s, d, c = _edges(rng, n, 0.08)
        src += [left[s], right[s]]
        dst += [left[d], right[d]]
        w += [c, c]
    if contra:  # a -> b crossing the midline in both directions
        s, d, c = _edges(rng, n, 0.08)
        src += [left[s], right[s]]
        dst += [right[d], left[d]]
        w += [c, c]
    g = _graph(np.concatenate(src), np.concatenate(dst), np.concatenate(w), 2 * n)
    return g, np.argsort(sigma)


@pytest.mark.parametrize(
    ("ipsi", "contra", "bisected"), [(True, False, False), (False, True, True), (True, True, True)]
)
def test_faq_recovers_a_hidden_matching(ipsi, contra, bisected):
    n = 150
    g, truth = _bilateral(n, ipsi, contra)
    adj = gm.subgraph(g, np.arange(n), np.arange(n, 2 * n), n)
    assert (adj["LL"].nnz > 0) == ipsi and (adj["LR"].nnz > 0) == contra
    perm = gm.faq(adj, np.full((n, n), 1.0 / n), bisected, iters=30)
    assert sorted(perm) == list(range(n))
    assert (perm == truth).mean() == 1.0


def _dense(adj: dict) -> dict:
    return {k: v.toarray() for k, v in adj.items()}


def _f(a: dict, p: np.ndarray, bisected: bool) -> float:
    """f(P) = <LL, P RR P^T> + <LR, P RL P>, dense."""
    val = (a["LL"] * (p @ a["RR"] @ p.T)).sum()
    return float(val + ((a["LR"] * (p @ a["RL"] @ p)).sum() if bisected else 0.0))


@pytest.mark.parametrize("bisected", [False, True])
def test_quad_and_gradient_are_consistent(bisected):
    n, rng = 40, np.random.default_rng(1)
    g, _ = _bilateral(n, ipsi=True, contra=True, seed=1)
    adj = gm.subgraph(g, np.arange(n), np.arange(n, 2 * n), n)
    a = _dense(adj)
    perm = np.eye(n)[rng.permutation(n)]
    for p in (perm, gm.sinkhorn(rng.random((n, n)))):
        f = _f(a, p, bisected)
        assert gm.quad(adj, p, p, bisected) == pytest.approx(f, rel=1e-10)
        grad = gm.gradient(adj, p, bisected)
        assert (grad * p).sum() / 2 == pytest.approx(f, rel=1e-10)  # f is homogeneous of degree 2
        d, e = rng.standard_normal((n, n)), 1e-3
        fd = (_f(a, p + e * d, bisected) - _f(a, p - e * d, bisected)) / (2 * e)
        assert (grad * d).sum() == pytest.approx(fd, rel=1e-6)
    d1, d2 = rng.standard_normal((n, n)), rng.standard_normal((n, n))
    bil = (a["LL"] * (d1 @ a["RR"] @ d2.T)).sum()
    bil += (a["LR"] * (d1 @ a["RL"] @ d2)).sum() if bisected else 0.0
    assert gm.quad(adj, d1, d2, bisected) == pytest.approx(bil, rel=1e-10)


def test_sinkhorn_is_doubly_stochastic():
    m = np.random.default_rng(2).random((30, 30)) + 0.01
    m0 = m.copy()
    s = gm.sinkhorn(m)
    assert (s > 0).all()
    np.testing.assert_allclose(s.sum(axis=0), 1.0, atol=1e-8)
    np.testing.assert_allclose(s.sum(axis=1), 1.0, atol=1e-8)
    assert (m == m0).all()  # input unchanged


# ---------------------------------------------------------------- across connectomes


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_rank1_assignment_is_optimal_for_rank1_matrices(seed):
    rng = np.random.default_rng(seed)
    g = np.outer(rng.standard_normal(25), rng.standard_normal(25))  # mixed signs
    rows, cols = gmm.rank1_assignment(g)
    assert sorted(rows) == sorted(cols) == list(range(25))
    r, c = linear_sum_assignment(g, maximize=True)
    assert g[rows, cols].sum() == pytest.approx(g[r, c].sum(), rel=1e-9)


def test_project_returns_the_best_matching():
    rng = np.random.default_rng(3)
    p = rng.random((20, 20))
    match = gmm.project(p)
    assert sorted(match) == list(range(20))
    r, c = linear_sum_assignment(p, maximize=True)
    assert p[match, np.arange(20)].sum() == pytest.approx(p[r, c].sum())
    hidden = rng.permutation(20)
    assert (gmm.project(np.eye(20)[hidden].T) == hidden).all()  # P[hidden[m], m] = 1


def _two_brains(sizes: tuple[int, int], seed: int = 0):
    """The same two-hemisphere wiring in two brains, the second with its neurons reordered
    within each hemisphere. Returns both graphs, their hemisphere blocks (node indices) and the
    true matching per block (index in brain 2 -> index in brain 1)."""
    rng = np.random.default_rng(seed)
    n = sum(sizes)
    s, d, w = _edges(rng, n, 0.06)  # all four (ipsi- and contralateral) blocks
    f_blocks = [np.arange(sizes[0]), np.arange(sizes[0], n)]
    # brain 2: neuron a of hemisphere h sits at position sigma_h[a] of that hemisphere
    sigmas = [rng.permutation(k) for k in sizes]
    pos = np.concatenate([sigmas[0], sizes[0] + sigmas[1]])
    order = rng.permutation(n)  # node ids in brain 2 do not follow the hemispheres
    m_blocks = [order[: sizes[0]], order[sizes[0] :]]
    node = np.concatenate(m_blocks)[pos]  # brain-2 node of brain-1 neuron a
    f = _graph(s, d, w, n)
    m = _graph(node[s], node[d], w, n)
    return f, m, f_blocks, m_blocks, [np.argsort(sg) for sg in sigmas]


def test_faq_blocks_recovers_a_block_diagonal_matching():
    sizes = (70, 60)  # unequal: the right block is padded with dummies
    f, m, f_blocks, m_blocks, truth = _two_brains(sizes)
    n = max(sizes)
    F = gmm.as32(gm.subgraph(f, *f_blocks, n))
    M = gmm.as32(gm.subgraph(m, *m_blocks, n))
    bary = np.full((n, n), 1.0 / n, dtype=np.float32)
    matches = gmm.faq_blocks(F, M, [bary, bary.copy()], iters=30)
    for match, t, k in zip(matches, truth, sizes, strict=True):
        assert sorted(match) == list(range(n))
        assert (match[:k] == t).all()


def test_block_gradient_and_bilinear_are_consistent():
    f, m, f_blocks, m_blocks, _ = _two_brains((30, 30), seed=4)
    F, M = gm.subgraph(f, *f_blocks, 30), gm.subgraph(m, *m_blocks, 30)
    rng = np.random.default_rng(4)
    P = [gm.sinkhorn(rng.random((30, 30))) for _ in range(2)]
    G = gmm.gradient(F, M, P)
    f2 = 2 * gmm.bilinear(F, M, P, P)
    assert sum((g * p).sum() for g, p in zip(G, P, strict=True)) == pytest.approx(f2, rel=1e-10)


def test_rare_types_and_candidates():
    sizes = pd.Series({"A": 1, "B": 2, "C": 4, "D": 5})
    assert gmm.rare(["A", "B", "C", "D", "unknown"], sizes).tolist() == [0, 1, 1, 0, 0]
    stage1 = pd.DataFrame({"pred": ["B", "D", "C", "A"], "label": ["B", "C", None, "C"]})
    assert gmm.candidates(stage1, sizes, "gnn").tolist() == [0, 2]
    assert gmm.candidates(stage1, sizes, "oracle").tolist() == [0, 1, 3]


def test_file_candidates_two_tier_and_degrade_args(tmp_path):
    sizes = pd.Series({"A": 1, "B": 2, "C": 4, "D": 5, "E": 10, "F": 30})
    stage1 = pd.DataFrame({"root_id": [11, 12, 13, 14], "pred": ["B", "E", "F", "D"]})
    pd.DataFrame({"root_id": [14, 12]}).to_parquet(tmp_path / "candidates_x.parquet")
    assert gmm.candidates(stage1, sizes, "file:x", tmp_path).tolist() == [1, 3]
    assert gmm.cand_tag("file:x") == "x" and gmm.cand_tag("gnn") == "gnn"
    # not assigned and the stage-1 type in the coarse label space (>= 10): coarse type
    pred = np.array(["B", "E", "F", "C"], dtype=object)
    assigned = np.array([False, False, False, True])
    coarse = np.array(["E", "F", "E", "F"], dtype=object)
    out = gmm.two_tier(pred, assigned, stage1.pred.to_numpy(), coarse, 10, sizes)
    assert out.tolist() == ["B", "F", "E", "C"]
    full = gmm.degrade_args({"target_retention": 1.0})
    assert full == {
        "retention": 1.0,
        "retention_seed": 2,
        "exclude_types": (),
        "crop_fraction": 0.0,
    }
    thin = gmm.degrade_args({"target_retention": 0.35, "retention_seed": 3, "target_crop": None})
    assert thin["retention"] == 0.35 and thin["retention_seed"] == 3 and thin["crop_fraction"] == 0


def test_mass_selection_per_side():
    cs = importlib.import_module("candidate_selection")
    side = np.array(["left", "left", "left", "right", "right", ""], dtype=object)
    score = np.array([0.9, 0.1, 0.5, 0.2, 0.8, 1.0])
    none = np.zeros(6, bool)
    k = {"left": 2, "right": 1}
    assert np.nonzero(cs.top_by_side(score, side, k, none))[0].tolist() == [0, 2, 4]
    base = np.array([False, True, False, False, False, True])  # kept; filled up to k per side
    assert np.nonzero(cs.top_by_side(score, side, k, base))[0].tolist() == [0, 1, 4, 5]


def test_size_threshold_run_names():
    st = importlib.import_module("size_threshold")
    assert st.label_free_dir("mcns", "consistency") == "label_free"
    assert st.label_free_dir("banc", "size_agreement") == "label_free_banc_size_agreement"
    assert st.label_free_dir("mcns", "size_agreement") == "label_free_mcns_size_agreement"
    thin = {"retention": 0.35}
    assert (
        st.label_free_dir("mcns", "size_agreement", thin) == "label_free_mcns_r0.35_size_agreement"
    )
    other = {"retention": 0.25, "retention_seed": 3, "exclude": ("JO-A",)}
    assert (
        st.label_free_dir("mcns", "consistency", other)
        == "label_free_mcns_r0.25_seed3_x_consistency"
    )


def test_lap_capacity():
    rng = np.random.default_rng(0)
    g = rng.random((6, 6))
    rows, cols = gmm.lap_capacity(g, 1, 4)
    ref = linear_sum_assignment(g, maximize=True)
    assert g[rows, cols].sum() == pytest.approx(g[ref].sum())
    g = np.zeros((4, 4))
    g[0, :3] = 1.0  # three columns prefer row 0; with capacity 2 it takes two of them
    rows, cols = gmm.lap_capacity(g, 2, 3)
    assert sorted(cols) == [0, 1, 2, 3] and (rows == 0).sum() == 2


def test_centered_gradient_matches_dense_centering():
    f, m, f_blocks, m_blocks, _ = _two_brains((30, 25), seed=5)
    F, M = gm.subgraph(f, *f_blocks, 30), gm.subgraph(m, *m_blocks, 30)
    rng = np.random.default_rng(5)
    P = [gm.sinkhorn(rng.random((30, 30))).astype(np.float32) for _ in range(2)]

    def centered(a):
        a = a.toarray().astype(np.float64)
        return a - np.outer(a.sum(1), a.sum(0)) / a.sum()

    Fc = {k: centered(v) for k, v in F.items()}
    Mc = {k: centered(v) for k, v in M.items()}
    ref = gmm.gradient(Fc, Mc, [p.astype(np.float64) for p in P])
    got = gmm.centered_gradient(gmm.as32(F), gmm.as32(M), P)
    for r, g in zip(ref, got, strict=True):
        assert np.allclose(g, r, atol=1e-3 * np.abs(r).max())


def test_absorb_reject():
    gc = np.array([[2.0, -1.0, 0.5], [0.1, 0.3, -2.0], [0.0, 0.0, 0.0]])  # row 2: dummy
    fm = np.array([0, 1, 2])  # candidate 2 sits on the dummy
    absorbed, rejected, own = gmm.absorb_reject(gc, fm, 2)
    assert absorbed.tolist() == [0, 1, 0]  # absorbed onto FAFB 0 (gain 0.5 > 0)
    assert np.isnan(own[2]) and own[:2] == pytest.approx([2.0, 0.3])
    gc[1, 1] = -0.3
    absorbed, rejected, _ = gmm.absorb_reject(gc, fm, 2)
    assert rejected.tolist() == [0, -1, 0]


@pytest.mark.skipif(
    not (ROOT / "data/processed/neurons.parquet").exists(), reason="needs the FAFB type sizes"
)
def test_match_end_to_end_with_absorb_and_capacity(tmp_path):
    """match() on two synthetic brains labeled with real 2-4-neuron FAFB types: the matching
    recovers the types, and the absorb / reject / capacity variants are reported."""
    stratify = importlib.import_module("stratify")
    types = stratify.fafb_types()
    sizes = (40, 30)
    rng = np.random.default_rng(1)
    n = sum(sizes)
    s_, d_, w = _edges(rng, n, 0.08)
    sigmas = [rng.permutation(k) for k in sizes]
    pos = np.concatenate([sigmas[0], sizes[0] + sigmas[1]])
    order = rng.permutation(n)
    node = np.concatenate([order[: sizes[0]], order[sizes[0] :]])[pos]
    extra = 4  # isolated left candidates beyond FAFB's left neurons: they go to dummies
    names = list(types.index[types["size"].between(2, 4)][:n])
    fside = np.array(["left"] * sizes[0] + ["right"] * sizes[1], dtype=object)
    fafb = _graph(s_, d_, w, n)
    fafb.label_names, fafb.y = names, torch.arange(n)
    fafb.side, fafb.node_ids = fside, torch.arange(1000, 1000 + n)
    tgt = _graph(node[s_], node[d_], w, n + extra)
    tside = np.empty(n + extra, dtype=object)
    tside[node] = fside
    tside[n:] = "left"
    tlabel = np.empty(n + extra, dtype=object)
    tlabel[node] = names
    tgt.side, tgt.node_ids = tside, torch.arange(n + extra)
    stage1 = pd.DataFrame(
        {
            "root_id": np.arange(n + extra),
            "label": tlabel,
            "pred": names[0],
            "pred_selected": names[0],
        }
    )
    s = SimpleNamespace(
        sel={"chosen": {"variant": "raw"}},
        fafb=fafb,
        tgt=tgt,
        stage1=stage1,
        sizes=types["size"],
        cand=np.arange(n + extra),
        out=tmp_path,
    )
    args = SimpleNamespace(
        candidates="gnn",
        target="mcns",
        fallback=[],
        start="bary",
        iters=30,
        lam=0.0,
        variant="raw",
        capacity=2,
        cap_iters=3,
        stage1="synthetic",
    )
    gmm.match(s, args)
    m = pd.read_csv(tmp_path / "metrics.csv")
    acc = m[m.subset == "2-4 (MCNS label)"].set_index("method").acc
    for method in ("gm_bary", "gm_bary_absorb", "gm_bary_absorb_reject", "gm_bary_cap2"):
        assert method in acc.index
    assert acc["gm_bary"] == 1.0 and acc["stage 1 (label-free GNN)"] < 0.1
    out = pd.read_parquet(tmp_path / "matches_gnn_gm_bary.parquet")
    assert (out.fafb_root_id[out.label.notna()] >= 1000).all()
    assert (out.fafb_root_id[out.label.isna()] == -1).all()  # isolated: on dummies
