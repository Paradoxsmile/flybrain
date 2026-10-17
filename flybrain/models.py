"""GNN models for node classification on the connectome."""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn

EDGE_WEIGHTINGS = ("none", "count", "log")


def _edge_values(
    edge_index: torch.Tensor, edge_weight: torch.Tensor | None, weighting: str
) -> torch.Tensor:
    """Per-edge weight before normalization: 1, the synapse count, or log1p of it."""
    if weighting not in EDGE_WEIGHTINGS:
        raise ValueError(f"weighting must be one of {EDGE_WEIGHTINGS}, got {weighting!r}")
    if weighting == "none" or edge_weight is None:
        return torch.ones(edge_index.size(1), device=edge_index.device)
    w = edge_weight.float()
    return torch.log1p(w) if weighting == "log" else w


def mean_adjacency(
    edge_index: torch.Tensor,
    num_nodes: int,
    edge_weight: torch.Tensor | None = None,
    weighting: str = "count",
    reverse: bool = False,
) -> torch.Tensor:
    """Sparse [num_nodes, num_nodes] matrix A with A @ h = (weighted) mean over neighbors.

    Default: row i averages over the presynaptic partners j of i (edges j -> i).
    reverse=True: row i averages over the postsynaptic partners k of i (edges i -> k).
    weighting: "none" (plain mean), "count" (synapse counts), "log" (log1p of synapse counts).
    Rows of nodes without such neighbors are zero.
    """
    src, dst = edge_index
    w = _edge_values(edge_index, edge_weight, weighting)
    rows, cols = (src, dst) if reverse else (dst, src)
    total = torch.zeros(num_nodes, device=w.device).index_add_(0, rows, w)
    w = w / total[rows]
    adj = torch.sparse_coo_tensor(
        torch.stack([rows, cols]), w, (num_nodes, num_nodes), check_invariants=False
    )
    return adj.coalesce()


class _CSRMatMul(torch.autograd.Function):
    """y = A @ x with a constant sparse A; the backward pass uses a precomputed A^T (CSR)."""

    @staticmethod
    def forward(ctx, a: torch.Tensor, a_t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        ctx.a_t = a_t
        return a @ x

    @staticmethod
    def backward(ctx, grad: torch.Tensor) -> tuple[None, None, torch.Tensor]:
        return None, None, ctx.a_t @ grad


class SparseMean:
    """Applies a fixed sparse aggregation matrix (from mean_adjacency) to node features.

    Stores A and A^T in CSR format: ~10x faster than torch.sparse.mm on COO, forward and backward.
    """

    def __init__(self, adj: torch.Tensor | None = None):
        if adj is not None:
            adj = adj.coalesce()
            self.a = adj.to_sparse_csr()
            self.a_t = adj.t().coalesce().to_sparse_csr()

    @classmethod
    def from_csr(cls, a: torch.Tensor, a_t: torch.Tensor) -> SparseMean:
        mean = cls()
        mean.a, mean.a_t = a, a_t
        return mean

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return _CSRMatMul.apply(self.a, self.a_t, x)


class MeanBuilder:
    """Builds the input / output SparseMeans of a graph, optionally on a subset of its edges.

    Same result as SparseMean(mean_adjacency(...)), but the edge orders needed for the CSR
    matrices are sorted once, so a new subgraph (e.g. with dropped edges) costs O(edges).
    """

    def __init__(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        edge_weight: torch.Tensor | None = None,
        weighting: str = "count",
    ):
        self.src, self.dst = edge_index
        self.n = num_nodes
        self.edge_index, self.weighting = edge_index, weighting
        self.w = _edge_values(edge_index, edge_weight, weighting)
        self.by_dst = torch.argsort(self.dst * num_nodes + self.src)  # rows = targets
        self.by_src = torch.argsort(self.src * num_nodes + self.dst)  # rows = sources

    def _csr(self, order, rows, cols, values, keep) -> torch.Tensor:
        if keep is not None:
            order = order[keep[order]]
        crow = torch.zeros(self.n + 1, dtype=torch.long, device=rows.device)
        crow[1:] = torch.bincount(rows[order], minlength=self.n).cumsum(0)
        return torch.sparse_csr_tensor(
            crow, cols[order], values[order], (self.n, self.n), check_invariants=False
        )

    def _mean(self, rows, cols, by_rows, by_cols, keep, w) -> SparseMean:
        w = w if keep is None else w * keep
        total = torch.zeros(self.n, device=w.device).index_add_(0, rows, w)
        values = w / total[rows].clamp(min=1e-12)
        a = self._csr(by_rows, rows, cols, values, keep)
        a_t = self._csr(by_cols, cols, rows, values, keep)
        return SparseMean.from_csr(a, a_t)

    def __call__(
        self,
        bidirectional: bool = True,
        keep: torch.Tensor | None = None,
        edge_weight: torch.Tensor | None = None,
    ) -> tuple[SparseMean, SparseMean | None]:
        """(mean over inputs, mean over outputs or None); keep: bool mask of edges to use;
        edge_weight: new synapse counts for this call (e.g. thinned), weighted as configured."""
        w = self.w
        if edge_weight is not None:
            w = _edge_values(self.edge_index, edge_weight, self.weighting)
        mean_in = self._mean(self.dst, self.src, self.by_dst, self.by_src, keep, w)
        if not bidirectional:
            return mean_in, None
        return mean_in, self._mean(self.src, self.dst, self.by_src, self.by_dst, keep, w)


def drop_edges(
    edge_index: torch.Tensor, num_nodes: int, max_rate: float, generator: torch.Generator | None
) -> torch.Tensor:
    """Bool mask of kept edges. Every neuron draws a drop rate in [0, max_rate]; an edge is
    dropped with the mean rate of its two neurons, so some regions lose more edges than others
    (like parts of a connectome that are reconstructed less completely)."""
    src, dst = edge_index
    rate = torch.rand(num_nodes, generator=generator, device=src.device) * max_rate
    p_drop = (rate[src] + rate[dst]) / 2
    return torch.rand(src.numel(), generator=generator, device=src.device) >= p_drop


class DirectedSAGELayer(nn.Module):
    """h_i' = W_self h_i + W_in mean_in(h) + [W_out mean_out(h)].

    Without the out term and with unweighted means this equals PyG's SAGEConv (mean aggr).
    mean_in / mean_out are callables (SparseMean). Since the mean is linear, the projection is
    applied before aggregating when that is cheaper (out_dim < in_dim); the result is the same.
    """

    def __init__(self, in_dim: int, out_dim: int, bidirectional: bool):
        super().__init__()
        self.lin_self = nn.Linear(in_dim, out_dim, bias=False)
        self.lin_in = nn.Linear(in_dim, out_dim)
        self.lin_out = nn.Linear(in_dim, out_dim, bias=False) if bidirectional else None
        self.project_first = out_dim < in_dim

    def _aggregate(self, lin: nn.Linear, mean: Callable, x: torch.Tensor) -> torch.Tensor:
        if self.project_first:
            h = mean(F.linear(x, lin.weight))
            return h if lin.bias is None else h + lin.bias
        return lin(mean(x))

    def forward(
        self, x: torch.Tensor, mean_in: Callable, mean_out: Callable | None
    ) -> torch.Tensor:
        h = self.lin_self(x) + self._aggregate(self.lin_in, mean_in, x)
        if self.lin_out is not None:
            h = h + self._aggregate(self.lin_out, mean_out, x)
        return h


class DirectedSAGE(nn.Module):
    """Stack of DirectedSAGELayers with ReLU + dropout.

    bidirectional=True aggregates over inputs and outputs separately, so a neuron also sees
    whom it projects to (e.g. sensory neurons, which have almost no inputs).
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool = True,
    ):
        super().__init__()
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        self.layers = nn.ModuleList(
            DirectedSAGELayer(a, b, bidirectional) for a, b in zip(dims[:-1], dims[1:], strict=True)
        )
        self.dropout = dropout
        self.bidirectional = bidirectional

    def forward(
        self, x: torch.Tensor, mean_in: Callable, mean_out: Callable | None = None
    ) -> torch.Tensor:
        if self.bidirectional and mean_out is None:
            raise ValueError("bidirectional model needs mean_out")
        for layer in self.layers[:-1]:
            x = F.dropout(F.relu(layer(x, mean_in, mean_out)), self.dropout, self.training)
        return self.layers[-1](x, mean_in, mean_out)


def class_weights(y: torch.Tensor, num_classes: int, mode: str = "balanced") -> torch.Tensor | None:
    """Loss weights per class from the labels y (no -1). "balanced": n / (k * count_c),
    "sqrt": its square root (milder for long-tailed labels), "none": None."""
    if mode == "none":
        return None
    counts = torch.bincount(y, minlength=num_classes).float().clamp(min=1)
    w = len(y) / (num_classes * counts)
    if mode == "sqrt":
        w = w.sqrt()
    elif mode != "balanced":
        raise ValueError(f"class weight mode must be none, balanced or sqrt, got {mode!r}")
    return w
