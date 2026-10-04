"""Reproducible, seeded, stratified train/val/test splits over labeled nodes."""

from __future__ import annotations

import numpy as np
import torch


def stratified_split(
    y: torch.Tensor, seed: int = 0, val_frac: float = 0.15, test_frac: float = 0.15
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return boolean (train, val, test) masks. Nodes with y < 0 are unlabeled and excluded.

    Classes with fewer than 3 labeled members go entirely to train.
    """
    rng = np.random.default_rng(seed)
    labels = y.numpy()
    train = np.zeros(len(labels), dtype=bool)
    val = np.zeros_like(train)
    test = np.zeros_like(train)
    for c in np.unique(labels[labels >= 0]):
        idx = rng.permutation(np.nonzero(labels == c)[0])
        if len(idx) < 3:
            train[idx] = True
            continue
        n_test = max(1, round(len(idx) * test_frac))
        n_val = max(1, round(len(idx) * val_frac))
        test[idx[:n_test]] = True
        val[idx[n_test : n_test + n_val]] = True
        train[idx[n_test + n_val :]] = True
    return torch.from_numpy(train), torch.from_numpy(val), torch.from_numpy(test)
