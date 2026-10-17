"""Memory-lean reimplementation of seeded NTAC (Schwartzman et al., Nature Communications 2026).

NTAC assigns cell types from connectivity alone. Each neuron is described by the synapses it
receives from and sends to every type (a 2k-dimensional vector, inputs first); every unlabeled
neuron takes the label of the most similar labeled neuron under weighted Jaccard similarity; the
type vectors are then recomputed from the new labels, blended into the previous ones with a
learning rate, and the step repeats.

The official package (`pip install ntac`, class `ntac.Ntac`) materializes two dense
[all neurons x labeled neurons] float32 matrices, about 40 GB each on the full FlyWire brain.
This version computes the same nearest labeled neuron row by row with a buffer of one row, and
reproduces the package's steps exactly (tests/test_ntac.py checks this against the package run
with one thread: the package's parallel similarity kernel accumulates into shared cells from
several threads, so its multi-threaded results vary from run to run; this kernel gives each
thread its own buffer and is deterministic):
- unlabeled neurons start with the smallest label among the labeled neurons, as in the package;
- similarities are computed in float32; ties go to the lowest labeled-neuron index;
- after the labels are reassigned, embedding <- (1 - lr) * embedding + lr * E(labels).
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import scipy.sparse as sp
import torch
from numba import njit, prange


def type_embedding(adj: sp.csr_matrix, labels: np.ndarray, k: int) -> sp.csr_matrix:
    """[n, 2k]: synapses received from (first k) and sent to (last k) neurons of each type."""
    n = adj.shape[0]
    onehot = sp.csr_matrix((np.ones(n), (np.arange(n), labels)), shape=(n, k))
    return sp.hstack([adj.T @ onehot, adj @ onehot]).tocsr()


@njit(parallel=True, cache=True)
def _nearest_labeled(
    q_indptr, q_indices, q_data, q_sum, f_indptr, f_indices, f_data, f_sum, num_labeled
):
    """For each query row, the index of the labeled row with the highest weighted Jaccard
    similarity sum(min) / (sum_q + sum_f - sum(min)), computed through an inverted index over
    the labeled rows (f_* is the CSC form of the labeled embedding)."""
    nq = q_indptr.shape[0] - 1
    best_idx = np.zeros(nq, dtype=np.int64)
    for r in prange(nq):
        inter = np.zeros(num_labeled, dtype=np.float32)
        for p in range(q_indptr[r], q_indptr[r + 1]):
            d = q_indices[p]
            a = q_data[p]
            for t in range(f_indptr[d], f_indptr[d + 1]):
                b = f_data[t]
                inter[f_indices[t]] += a if a < b else b
        best = np.float32(-1.0)
        for j in range(num_labeled):
            denom = q_sum[r] + f_sum[j] - inter[j]
            s = np.float32(0.0) if denom < 1e-9 else inter[j] / denom
            if s > best:
                best = s
                best_idx[r] = j
    return best_idx


def _assign(emb: sp.csr_matrix, labeled: np.ndarray, unlabeled: np.ndarray) -> np.ndarray:
    """Index into `labeled` of the most similar labeled neuron, for every unlabeled neuron."""
    q = emb[unlabeled].astype(np.float32).tocsr()
    f = emb[labeled].astype(np.float32).tocsc()
    q_sum = np.asarray(emb[unlabeled].sum(axis=1)).ravel().astype(np.float32)
    f_sum = np.asarray(emb[labeled].sum(axis=1)).ravel().astype(np.float32)
    return _nearest_labeled(
        q.indptr, q.indices, q.data, q_sum, f.indptr, f.indices, f.data, f_sum, len(labeled)
    )


def ntac_seeded(
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    y: torch.Tensor,
    known: torch.Tensor,
    num_classes: int,
    lr: float = 0.3,
    steps: int = 15,
) -> Iterator[np.ndarray]:
    """Run seeded NTAC; yields the label of every neuron after each step.

    Labels of `known` neurons (y >= 0) are fixed; all other neurons are re-labeled each step.
    Edge weights are synapse counts, pre -> post.
    """
    n = y.numel()
    src, dst = edge_index.cpu().numpy()
    adj = sp.csr_matrix((edge_weight.cpu().numpy().astype(np.float64), (src, dst)), shape=(n, n))
    y_np = y.cpu().numpy()
    labeled = np.sort(np.nonzero(known.cpu().numpy() & (y_np >= 0))[0])
    unlabeled = np.setdiff1d(np.arange(n), labeled)

    # the package starts every unlabeled neuron at its label index 0, i.e. the smallest label
    # among the labeled neurons
    labels = np.full(n, y_np[labeled].min(), dtype=np.int64)
    labels[labeled] = y_np[labeled]
    emb = type_embedding(adj, labels, num_classes)
    for _ in range(steps):
        labels = labels.copy()
        labels[unlabeled] = labels[labeled][_assign(emb, labeled, unlabeled)]
        yield labels
        emb = ((1 - lr) * emb + lr * type_embedding(adj, labels, num_classes)).tocsr()


def agreement_seeds(predictions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(majority vote, unanimous mask) of several models' predicted labels [models, nodes]."""
    predictions = np.asarray(predictions)
    vote = np.array([np.bincount(col).argmax() for col in predictions.T])
    return vote, (predictions == predictions[0]).all(axis=0)


def ntac_from_predictions(
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    known: np.ndarray,
    predictions: np.ndarray,
    num_classes: int,
    lr: float = 0.3,
    steps: int = 15,
) -> tuple[np.ndarray, np.ndarray]:
    """Seeded NTAC whose seeds are the known labels plus the predictions of several models
    where they all agree (for neurons without a known label). `known` holds a label or -1 per
    neuron. Returns (labels after the last step, seed mask)."""
    vote, agree = agreement_seeds(predictions)
    seeds = np.where(known >= 0, known, np.where(agree, vote, -1))
    mask = seeds >= 0
    *_, labels = ntac_seeded(
        edge_index,
        edge_weight,
        torch.from_numpy(seeds),
        torch.from_numpy(mask),
        num_classes,
        lr,
        steps,
    )
    return labels, mask
