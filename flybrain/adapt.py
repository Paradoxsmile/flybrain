"""Label-free adaptation of a cell-type classifier to a new connectome.

Cell-type sizes are stereotyped across individuals: a type has about the same number of neurons
in every fly brain. A classifier carried over to another connectome, however, tends to pour whole
types into a neighbor type (e.g. one Kenyon cell subtype into another). prior_weights rescales
the class probabilities so that the predicted type sizes match a reference distribution, as in
prior-shift correction or entropic optimal transport with fixed class marginals (Sinkhorn
scaling). The strength rho interpolates between the raw prediction (0) and full matching (1).
"""

from __future__ import annotations

import numpy as np


def prior_weights(
    probs: np.ndarray, target: np.ndarray, iters: int = 200, tol: float = 1e-6
) -> np.ndarray:
    """Class weights v such that the soft class counts of probs * v (rows renormalized) match
    `target` (expected number of neurons per class; sums to the number of rows).

    Sinkhorn iterations on the probability matrix with damped (square-root) updates. Classes
    that no neuron can take (zero probability everywhere) keep weight 1.
    """
    probs = np.asarray(probs, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    v = np.ones(probs.shape[1])
    for _ in range(iters):
        q = probs * v
        q /= q.sum(axis=1, keepdims=True)
        counts = q.sum(axis=0)
        ratio = np.where(counts > 0, target / np.maximum(counts, 1e-300), 1.0)
        v *= np.sqrt(ratio)
        if np.max(np.abs(counts - target)) < tol * len(probs):
            break
    return v


def adapt_predictions(probs: np.ndarray, weights: np.ndarray, rho: float) -> np.ndarray:
    """Class index per row after reweighting with weights ** rho."""
    return (np.asarray(probs) * np.asarray(weights) ** rho).argmax(axis=1)
