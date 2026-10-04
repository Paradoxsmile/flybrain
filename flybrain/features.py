"""Degree / synapse-count node features for baselines and GNN inputs."""

from __future__ import annotations

import numpy as np
import torch

FEATURE_NAMES = [
    "in_degree",
    "out_degree",
    "in_synapses",
    "out_synapses",
    "mean_in_syn",
    "mean_out_syn",
]


def degree_features(
    num_nodes: int, edge_index: torch.Tensor, edge_weight: torch.Tensor | None = None
) -> np.ndarray:
    """Return a [num_nodes, 6] log1p-scaled array (see FEATURE_NAMES)."""
    src, dst = edge_index.numpy()
    w = np.ones(src.shape[0]) if edge_weight is None else edge_weight.numpy().astype(float)
    out_deg = np.bincount(src, minlength=num_nodes).astype(float)
    in_deg = np.bincount(dst, minlength=num_nodes).astype(float)
    out_syn = np.bincount(src, weights=w, minlength=num_nodes)
    in_syn = np.bincount(dst, weights=w, minlength=num_nodes)
    mean_out = out_syn / np.maximum(out_deg, 1)
    mean_in = in_syn / np.maximum(in_deg, 1)
    feats = np.stack([in_deg, out_deg, in_syn, out_syn, mean_in, mean_out], axis=1)
    return np.log1p(feats)
