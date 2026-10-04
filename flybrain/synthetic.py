"""Tiny synthetic connectome (stochastic block model) for tests and --synthetic runs."""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_synthetic_tables(
    n_nodes: int = 300, n_classes: int = 4, seed: int = 0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (neurons, connections) DataFrames in the raw-table schema.

    Classes differ in connection probability, so degree features carry signal.
    """
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, n_classes, n_nodes)
    ids = rng.choice(10**6, n_nodes, replace=False) + 720575940000000000
    # Class k is denser the larger k is; same-class pairs get a small extra boost.
    base = 0.01 + 0.03 * np.arange(n_classes)
    same = labels[:, None] == labels[None, :]
    prob = np.minimum(0.4, base[labels][:, None] + 0.01 * same)
    adj = rng.random((n_nodes, n_nodes)) < prob
    np.fill_diagonal(adj, False)
    pre, post = np.nonzero(adj)
    neurons = pd.DataFrame(
        {
            "root_id": ids,
            "super_class": [f"class_{k}" for k in labels],
            "cell_type": [f"type_{k}_{rng.integers(0, 2)}" for k in labels],
        }
    )
    connections = pd.DataFrame(
        {
            "pre_root_id": ids[pre],
            "post_root_id": ids[post],
            "syn_count": rng.integers(1, 20, len(pre)),
        }
    )
    return neurons, connections
