"""Label-free quality of a cell-type assignment: does the labeling explain the wiring?

Connectomic cell types are defined by connectivity: neurons of a type have similar numbers of
synapses to and from each type (Matsliah et al., 2024; the basis of NTAC). A correct labeling of
an unlabeled connectome should therefore be self-consistent: describing each neuron by its
synapses per partner type, its most similar neuron (weighted Jaccard, as in NTAC) should carry
the same label.

type_consistency splits the neurons in two random halves, gives each neuron of one half the label
of its most similar neuron in the other half, and reports the chance-corrected agreement with the
labeling (Cohen's kappa; chance = sum of squared label shares). The correction matters: a
labeling with a single type agrees with itself perfectly but has kappa 0. Labelings refined by
NTAC optimize this criterion directly, so scores are only comparable between labelings that were
not produced by it.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch

from flybrain.ntac import _assign, type_embedding


def type_consistency(
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    labels: np.ndarray,
    num_classes: int,
    seed: int = 0,
) -> tuple[float, float]:
    """(kappa, raw agreement) of `labels` (one class index per neuron) with the graph."""
    n = len(labels)
    src, dst = edge_index.cpu().numpy()
    w = edge_weight.cpu().numpy().astype(np.float64)
    adj = sp.csr_matrix((w, (src, dst)), shape=(n, n))
    labels = np.asarray(labels, dtype=np.int64)
    perm = np.random.default_rng(seed).permutation(n)
    a, b = np.sort(perm[: n // 2]), np.sort(perm[n // 2 :])
    emb = type_embedding(adj, labels, num_classes)
    agree = float(
        np.mean([(labels[q] == labels[r][_assign(emb, r, q)]).mean() for r, q in ((a, b), (b, a))])
    )
    share = np.bincount(labels, minlength=num_classes) / n
    chance = float(share @ share)
    kappa = 0.0 if chance >= 1.0 else (agree - chance) / (1.0 - chance)
    return kappa, agree
