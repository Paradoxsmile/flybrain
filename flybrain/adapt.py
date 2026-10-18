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
    probs: np.ndarray,
    target: np.ndarray,
    iters: int = 200,
    tol: float = 1e-6,
    max_weight: float | None = None,
) -> np.ndarray:
    """Class weights v such that the soft class counts of probs * v (rows renormalized) match
    `target` (expected number of neurons per class; sums to the number of rows).

    Sinkhorn iterations on the probability matrix with damped (square-root) updates. Classes
    that no neuron can take (zero probability everywhere) keep weight 1. With max_weight, every
    weight stays within [1 / max_weight, max_weight]: a class that is missing or much smaller
    in the target cannot be filled by forcing neurons into it (bounded, partial matching).
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
        if max_weight is not None:  # weights only matter relative to each other
            v /= np.exp((target * np.log(v)).sum() / target.sum())
            v = np.clip(v, 1.0 / max_weight, max_weight)
        if np.max(np.abs(counts - target)) < tol * len(probs):
            break
    return v


def adapt_predictions(probs: np.ndarray, weights: np.ndarray, rho: float) -> np.ndarray:
    """Class index per row after reweighting with weights ** rho."""
    return (np.asarray(probs) * np.asarray(weights) ** rho).argmax(axis=1)


def em_prior(
    probs: np.ndarray, source_prior: np.ndarray, iters: int = 500, tol: float = 1e-7
) -> tuple[np.ndarray, np.ndarray]:
    """Target class prior estimated by EM (Saerens, Latinne & Decaestecker, 2002) and the
    weights target / source prior.

    Unlike prior_weights, which imposes the reference sizes, EM estimates the sizes from the
    classifier's outputs on the target: classes that the target lacks shrink towards zero
    instead of being filled. probs are the posteriors under the source prior."""
    probs = np.asarray(probs, dtype=np.float64)
    pi_s = np.asarray(source_prior, dtype=np.float64)
    pi_s = pi_s / pi_s.sum()
    pi_t = pi_s.copy()
    for _ in range(iters):
        q = probs * (pi_t / np.maximum(pi_s, 1e-300))
        q /= q.sum(axis=1, keepdims=True)
        new = q.mean(axis=0)
        done = np.max(np.abs(new - pi_t)) < tol
        pi_t = new
        if done:
            break
    return pi_t, pi_t / np.maximum(pi_s, 1e-300)
