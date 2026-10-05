"""Degree / synapse-count node features for baselines and GNN inputs."""

from __future__ import annotations

import numpy as np
import pandas as pd
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


FEATURE_NORMS = ("global", "per_side", "rank_per_side")


def normalize_features(
    x: torch.Tensor,
    mode: str = "global",
    fit_mask: torch.Tensor | None = None,
    groups: np.ndarray | None = None,
    reference: torch.Tensor | None = None,
) -> torch.Tensor:
    """Scale node features column-wise.

    global: z-score with the mean / std of the fit_mask rows (the training nodes), taken from
        reference instead of x if given (e.g. the full graph when x comes from a subgraph).
    per_side: z-score each group (e.g. hemisphere, from data.side) with its own mean / std.
    rank_per_side: within each group, map each feature to the normal quantile of its rank
        (ties share their average rank), which removes any monotone scale difference.
    The per-group modes use the features of all nodes in a group, no labels.
    """
    if mode not in FEATURE_NORMS:
        raise ValueError(f"feature_norm must be one of {FEATURE_NORMS}, got {mode!r}")
    if mode == "global":
        ref = x if reference is None else reference
        rows = ref if fit_mask is None else ref[fit_mask]
        return (x - rows.mean(0)) / rows.std(0).clamp(min=1e-6)
    if groups is None:
        raise ValueError(f"feature_norm={mode!r} needs groups (a side column)")
    out = torch.empty_like(x)
    for g in np.unique(groups):
        idx = torch.from_numpy(np.nonzero(groups == g)[0])
        xg = x[idx]
        if mode == "per_side":
            out[idx] = (xg - xg.mean(0)) / xg.std(0, unbiased=False).clamp(min=1e-6)
        else:
            ranks = pd.DataFrame(xg.numpy()).rank(method="average").to_numpy()
            q = torch.from_numpy((ranks - 0.5) / len(idx)).to(x.dtype)
            out[idx] = torch.special.ndtri(q)
    return out
