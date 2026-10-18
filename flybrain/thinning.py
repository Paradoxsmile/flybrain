"""Thinned copies of a connectome graph, to simulate a less complete reconstruction.

Each synapse is kept with probability `retention` (binomial thinning of the synapse counts) and
connections with fewer than `min_count` remaining synapses are dropped, as Codex drops them. On
FAFB at retention 0.35-0.4 the copy has about BANC's density (8.3 connections and a median of 27
input synapses per neuron at 0.35; BANC: 8.6 and 30; FAFB: 26.8 and 152).
"""

from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import Data

from flybrain.features import degree_features


def thin_counts(counts: np.ndarray, retention: float, seed: int = 0) -> np.ndarray:
    """Binomially thinned integer synapse counts (each synapse kept with prob. retention)."""
    if not 0.0 < retention <= 1.0:
        raise ValueError(f"retention must be in (0, 1], got {retention}")
    counts = np.asarray(counts).round().astype(np.int64)
    if retention == 1.0:
        return counts
    return np.random.default_rng(seed).binomial(counts, retention)


def thin_graph(data: Data, retention: float, min_count: int = 5, seed: int = 0) -> Data:
    """Copy of `data` with binomially thinned synapse counts and recomputed degree features.

    Labels, node ids and other node attributes are kept. retention = 1 returns the graph
    unchanged (connections below min_count are still dropped)."""
    counts = thin_counts(data.edge_weight.numpy(), retention, seed)
    keep = counts >= min_count
    edge_index = data.edge_index[:, torch.from_numpy(keep)]
    edge_weight = torch.from_numpy(counts[keep]).float()
    x = degree_features(data.num_nodes, edge_index, edge_weight)
    out = data.clone()
    out.edge_index, out.edge_weight = edge_index, edge_weight
    out.x = torch.from_numpy(x).float()
    return out
