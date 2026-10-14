"""Tests on a tiny synthetic graph (no real data needed)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from torch_geometric.nn import SAGEConv

from flybrain.features import FEATURE_NAMES, degree_features, normalize_features
from flybrain.loading import build_graph, clean_tables, load_tables
from flybrain.models import DirectedSAGELayer, SparseMean, class_weights, mean_adjacency
from flybrain.splits import hemisphere_split, make_split, stratified_split
from flybrain.synthetic import make_synthetic_tables
from flybrain.utils import set_seed


@pytest.fixture(scope="module")
def tables():
    return make_synthetic_tables(n_nodes=120, n_classes=3, seed=1)


@pytest.fixture(scope="module")
def data(tables):
    return build_graph(*tables)


def test_graph_shapes(data):
    assert data.num_nodes == 120
    assert data.x.shape == (120, len(FEATURE_NAMES))
    assert data.edge_index.shape[0] == 2
    assert data.edge_weight.shape[0] == data.edge_index.shape[1]
    assert len(data.label_names) == 3
    assert int(data.edge_index.max()) < data.num_nodes


def test_min_class_size_unlabels_small_classes(tables):
    neurons, conns = tables
    neurons = neurons.copy()
    neurons.loc[:1, "super_class"] = "rare"  # a class with only two neurons
    data = build_graph(neurons, conns, min_class_size=3)
    assert "rare" not in data.label_names
    assert (data.y[:2] == -1).all()
    assert len(data.label_names) == 3


def test_degree_features_simple():
    ei = torch.tensor([[0, 0, 1], [1, 2, 2]])
    feats = degree_features(3, ei, torch.tensor([2.0, 3.0, 5.0]))
    # node 0: out_degree 2, out_synapses 5; node 2: in_degree 2, in_synapses 8
    assert feats[0, 1] == pytest.approx(torch.log1p(torch.tensor(2.0)).item())
    assert feats[0, 3] == pytest.approx(torch.log1p(torch.tensor(5.0)).item())
    assert feats[2, 2] == pytest.approx(torch.log1p(torch.tensor(8.0)).item())


def test_split_disjoint_and_reproducible(data):
    a = stratified_split(data.y, seed=3)
    b = stratified_split(data.y, seed=3)
    assert all(torch.equal(m1, m2) for m1, m2 in zip(a, b, strict=True))
    tr, va, te = a
    assert not (tr & va).any() and not (tr & te).any() and not (va & te).any()
    assert int((tr | va | te).sum()) == int((data.y >= 0).sum())
    assert not torch.equal(a[2], stratified_split(data.y, seed=4)[2])


def test_clean_tables_sums_and_drops(tables):
    neurons, conns = tables
    extra = pd.DataFrame(
        {
            "pre_root_id": [conns.pre_root_id[0], -1],
            "post_root_id": [conns.post_root_id[0], conns.post_root_id[0]],
            "syn_count": [10, 5],
        }
    )
    _, cleaned = clean_tables(neurons, pd.concat([conns, extra]))
    assert (cleaned.pre_root_id != -1).all()
    row = cleaned[
        (cleaned.pre_root_id == conns.pre_root_id[0])
        & (cleaned.post_root_id == conns.post_root_id[0])
    ]
    assert row.syn_count.iloc[0] == conns.syn_count[0] + 10


def test_parquet_cache_roundtrip(tables, tmp_path):
    raw, proc = tmp_path / "raw", tmp_path / "proc"
    raw.mkdir()
    tables[0].to_csv(raw / "classification.csv", index=False)
    tables[1].to_csv(raw / "connections.csv", index=False)
    n1, c1 = load_tables(raw, proc)
    assert (proc / "neurons.parquet").exists()
    (raw / "classification.csv").unlink()  # second call must come from the cache
    n2, c2 = load_tables(raw, proc)
    pd.testing.assert_frame_equal(n1, n2)
    pd.testing.assert_frame_equal(c1, c2)


def test_missing_raw_files_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="data/README.md"):
        load_tables(tmp_path / "raw", tmp_path / "proc")


def test_sage_forward_and_seeding(data):
    def run() -> torch.Tensor:
        set_seed(0)
        return SAGEConv(data.x.size(1), 4)(data.x, data.edge_index)

    out = run()
    assert out.shape == (data.num_nodes, 4)
    assert torch.allclose(out, run())


def test_codex_filenames_and_cell_type_merge(tables, tmp_path):
    raw, proc = tmp_path / "raw", tmp_path / "proc"
    raw.mkdir()
    neurons = tables[0].drop(columns="cell_type", errors="ignore")
    neurons.to_csv(raw / "classification.csv.gz", index=False)
    tables[1].to_csv(raw / "connections_princeton.csv.gz", index=False)
    ids = neurons.root_id.tolist()
    types = pd.DataFrame({"root_id": ids[:2], "primary_type": ["T4a", "Mi1"]})
    types.to_csv(raw / "consolidated_cell_types.csv.gz", index=False)
    n, c = load_tables(raw, proc)
    assert len(n) == len(neurons) and len(c) > 0
    by_id = n.set_index("root_id").cell_type
    assert by_id[ids[0]] == "T4a" and by_id[ids[1]] == "Mi1"
    assert by_id[ids[2:]].isna().all()


def test_mean_adjacency_directions_and_weights():

    ei = torch.tensor([[0, 1, 2], [2, 2, 0]])  # 0->2, 1->2, 2->0
    w = torch.tensor([1.0, 3.0, 5.0])
    x = torch.tensor([[1.0], [2.0], [4.0]])
    a_in = mean_adjacency(ei, 3, w, "count")
    # node 2 averages its inputs 0 and 1, weighted 1:3; node 1 has no inputs
    assert torch.allclose(torch.sparse.mm(a_in, x).squeeze(), torch.tensor([4.0, 0.0, 1.75]))
    a_out = mean_adjacency(ei, 3, w, "none", reverse=True)
    # node 0 projects to 2, node 1 to 2, node 2 to 0
    assert torch.allclose(torch.sparse.mm(a_out, x).squeeze(), torch.tensor([4.0, 4.0, 1.0]))


def test_directed_sage_matches_sageconv(data):

    set_seed(0)
    conv = SAGEConv(data.x.size(1), 4)
    layer = DirectedSAGELayer(data.x.size(1), 4, bidirectional=False)
    layer.lin_in.load_state_dict(conv.lin_l.state_dict())
    layer.lin_self.load_state_dict(conv.lin_r.state_dict())
    mean = SparseMean(mean_adjacency(data.edge_index, data.num_nodes, weighting="none"))
    expected = conv(data.x, data.edge_index)
    assert torch.allclose(layer(data.x, mean, None), expected, atol=1e-5)
    # projecting before aggregating gives the same result
    layer.project_first = True
    assert torch.allclose(layer(data.x, mean, None), expected, atol=1e-5)


def test_sparse_mean_gradient_matches_dense():
    ei = torch.tensor([[0, 1, 2, 0], [2, 2, 0, 1]])
    adj = mean_adjacency(ei, 3, torch.tensor([1.0, 3.0, 5.0, 2.0]), "count")
    x = torch.randn(3, 4, requires_grad=True)
    SparseMean(adj)(x).pow(2).sum().backward()
    expected = torch.autograd.grad((adj.to_dense() @ x).pow(2).sum(), x)[0]
    assert torch.allclose(x.grad, expected, atol=1e-6)


def test_class_weights_balanced():

    w = class_weights(torch.tensor([0, 0, 0, 1]), 2)
    assert torch.allclose(w, torch.tensor([4 / 6, 2.0]))
    assert class_weights(torch.tensor([0, 1]), 2, "none") is None


def test_hemisphere_split(data):
    side = data.side
    train, val, test = hemisphere_split(data.y, side, "left", seed=0)
    assert (side[train.numpy()] == "left").all() and (side[val.numpy()] == "left").all()
    assert (side[test.numpy()] == "right").all()
    assert not (train & val).any() and val.any() and test.any()
    assert set(data.y[test].tolist()) <= set(data.y[train].tolist())
    flipped = make_split(data, "right_left", seed=0)
    assert (side[flipped[2].numpy()] == "left").all()


def test_hemisphere_split_needs_side(data):
    no_side = data.clone()
    del no_side.side
    with pytest.raises(ValueError, match="side"):
        make_split(no_side, "left_right")
    assert stratified_split(data.y, test_frac=0.0)[2].sum() == 0


def test_normalize_features_modes():
    g = torch.Generator().manual_seed(0)
    x = torch.rand(200, 3, generator=g)
    side = np.array(["left"] * 100 + ["right"] * 100)
    denser = x.clone()
    denser[100:] = denser[100:] * 2 + 1  # right side: same order, different scale

    glob = normalize_features(x, "global")
    assert torch.allclose(glob.mean(0), torch.zeros(3), atol=1e-5)

    per_side = normalize_features(denser, "per_side", groups=side)
    assert torch.allclose(per_side, normalize_features(x, "per_side", groups=side), atol=1e-5)
    assert torch.allclose(per_side[100:].mean(0), torch.zeros(3), atol=1e-5)

    # ranks ignore any monotone change within a side, e.g. log of the denser side
    warped = denser.clone()
    warped[100:] = torch.log(warped[100:])
    ranked = normalize_features(warped, "rank_per_side", groups=side)
    assert torch.allclose(ranked, normalize_features(x, "rank_per_side", groups=side))
    assert torch.isfinite(ranked).all()

    with pytest.raises(ValueError, match="groups"):
        normalize_features(x, "per_side")


def test_mean_builder_matches_mean_adjacency(data):
    from flybrain.models import MeanBuilder, drop_edges

    x = torch.randn(data.num_nodes, 5)
    ei, n, w = data.edge_index, data.num_nodes, data.edge_weight
    builder = MeanBuilder(ei, n, w, "count")
    mean_in, mean_out = builder(bidirectional=True)
    expected_out = SparseMean(mean_adjacency(ei, n, w, "count", reverse=True))
    assert torch.allclose(mean_in(x), SparseMean(mean_adjacency(ei, n, w, "count"))(x), atol=1e-5)
    assert torch.allclose(mean_out(x), expected_out(x), atol=1e-5)

    keep = drop_edges(ei, n, 0.5, torch.Generator().manual_seed(0))
    assert 0.6 < keep.float().mean() < 0.9  # mean drop rate is max_rate / 2
    sub_in, _ = builder(bidirectional=False, keep=keep)
    expected = SparseMean(mean_adjacency(ei[:, keep], n, w[keep], "count"))
    assert torch.allclose(sub_in(x), expected(x), atol=1e-5)
    # gradient through the subgraph mean uses the subgraph transpose
    xg = x.clone().requires_grad_(True)
    sub_in(xg).pow(2).sum().backward()
    dense = mean_adjacency(ei[:, keep], n, w[keep], "count").to_dense()
    xd = x.clone().requires_grad_(True)
    (dense @ xd).pow(2).sum().backward()
    assert torch.allclose(xg.grad, xd.grad, atol=1e-5)


def test_normalize_features_reference():
    x = torch.rand(50, 3)
    ref = x * 2
    out = normalize_features(x, "global", reference=ref)
    assert torch.allclose(out, (x - ref.mean(0)) / ref.std(0))
