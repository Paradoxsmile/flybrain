"""The label-free consistency score must prefer the true labeling of a structured graph."""

from __future__ import annotations

import numpy as np
import torch

from flybrain.consistency import type_consistency


def _block_graph(n=600, k=6, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, k, n)
    p = rng.uniform(0.0, 0.06, (k, k))[y][:, y]
    a = (rng.random((n, n)) < p) * rng.integers(1, 20, (n, n))
    np.fill_diagonal(a, 0)
    src, dst = np.nonzero(a)
    return torch.tensor(np.stack([src, dst])), torch.tensor(a[src, dst], dtype=torch.float), y


def test_true_labels_score_higher_than_corrupted_ones():
    ei, w, y = _block_graph()
    rng = np.random.default_rng(1)
    scores = []
    for noise in (0.0, 0.3, 0.6, 1.0):
        labels = np.where(rng.random(len(y)) < noise, rng.integers(0, 6, len(y)), y)
        scores.append(type_consistency(ei, w, labels, 6)[0])
    assert scores == sorted(scores, reverse=True)
    assert scores[0] > 0.8 and abs(scores[-1]) < 0.1


def test_single_label_has_zero_kappa():
    ei, w, y = _block_graph()
    kappa, agree = type_consistency(ei, w, np.zeros_like(y), 6)
    assert agree == 1.0 and kappa == 0.0
