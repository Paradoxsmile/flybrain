"""Tests on a tiny synthetic graph (no real data needed)."""

from __future__ import annotations

import pandas as pd
import pytest
import torch
from torch_geometric.nn import SAGEConv

from flybrain.features import FEATURE_NAMES, degree_features
from flybrain.loading import build_graph, clean_tables, load_tables
from flybrain.splits import stratified_split
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
