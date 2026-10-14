"""Baselines that use the labels of other neurons, unlike the GNN.

The GNN sees only anonymous structure (degree features of a neuron and its neighborhood). These
baselines instead read the labels of training neurons:

- label propagation assumes connected neurons share a label (homophily),
- partner profiles describe a neuron by the classes of its synaptic partners, which is what
  connectivity-based cell typing compares.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch


def synapse_matrix(edge_index: torch.Tensor, edge_weight: torch.Tensor, num_nodes: int):
    """scipy CSR matrix A with A[pre, post] = synapse count."""
    src, dst = edge_index.cpu().numpy()
    w = edge_weight.cpu().numpy().astype(float)
    return sp.csr_matrix((w, (src, dst)), shape=(num_nodes, num_nodes))


def edge_homophily(edge_index: torch.Tensor, edge_weight: torch.Tensor, y: torch.Tensor):
    """(synapse-weighted share of connections between neurons of the same class, chance level).

    Only connections with both neurons labeled count. Chance is the share expected if pre- and
    postsynaptic classes were independent, given how many synapses each class sends / receives.
    """
    src, dst = edge_index.cpu().numpy()
    w = edge_weight.cpu().numpy().astype(float)
    y = y.cpu().numpy()
    both = (y[src] >= 0) & (y[dst] >= 0)
    pre, post, w = y[src][both], y[dst][both], w[both]
    k = int(y.max()) + 1
    p_pre = np.bincount(pre, weights=w, minlength=k) / w.sum()
    p_post = np.bincount(post, weights=w, minlength=k) / w.sum()
    return float(w[pre == post].sum() / w.sum()), float(p_pre @ p_post)


def _row_normalize(m):
    total = np.asarray(m.sum(axis=1)).ravel()
    total[total == 0] = 1.0
    return sp.diags(1.0 / total) @ m


def partner_profiles(
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    y: torch.Tensor,
    known: torch.Tensor,
    num_classes: int,
):
    """Sparse [num_nodes, 2 * num_classes] features: the share of a neuron's input synapses
    (first half) and output synapses (second half) that come from / go to each class.

    Only partners in `known` (the training neurons) count, and shares are taken over those
    partners, so the varying share of unlabeled or held-out partners does not shift the profile.
    Rows of neurons without known partners are zero.
    """
    n = y.numel()
    a = synapse_matrix(edge_index, edge_weight, n)
    idx = np.nonzero(known.cpu().numpy())[0]
    onehot = sp.csr_matrix((np.ones(len(idx)), (idx, y.cpu().numpy()[idx])), shape=(n, num_classes))
    inputs = _row_normalize(a.T @ onehot)  # row i: synapses from known partners j -> i
    outputs = _row_normalize(a @ onehot)  # row i: synapses to known partners i -> k
    return sp.hstack([inputs, outputs]).tocsr()


def propagation_matrix(
    edge_index: torch.Tensor, edge_weight: torch.Tensor, num_nodes: int
) -> torch.Tensor:
    """S = D^-1/2 W D^-1/2 (torch CSR), W = synapse counts with direction ignored."""
    src, dst = edge_index
    w = edge_weight.float()
    idx = torch.cat([torch.stack([src, dst]), torch.stack([dst, src])], dim=1)
    shape = (num_nodes, num_nodes)
    W = torch.sparse_coo_tensor(idx, torch.cat([w, w]), shape, check_invariants=False).coalesce()
    rows, cols = W.indices()
    deg = torch.zeros(num_nodes).index_add_(0, rows, W.values())
    d = deg.clamp(min=1e-12).rsqrt()
    values = W.values() * d[rows] * d[cols]
    s = torch.sparse_coo_tensor(W.indices(), values, shape, check_invariants=False)
    return s.to_sparse_csr()


def label_propagation(
    s: torch.Tensor,
    y: torch.Tensor,
    known: torch.Tensor,
    num_classes: int,
    alpha: float = 0.9,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> torch.Tensor:
    """Label spreading (Zhou et al., 2004): F <- alpha S F + (1 - alpha) Y0 until convergence.

    s: propagation_matrix(...). Y0 is one-hot for the `known` neurons and zero elsewhere.
    Returns [num_nodes, num_classes] scores; alpha sets how far labels spread.
    """
    y0 = torch.zeros(y.numel(), num_classes)
    y0[known, y[known]] = 1.0
    f = y0
    for _ in range(max_iter):
        f_next = alpha * (s @ f) + (1 - alpha) * y0
        converged = (f_next - f).abs().max() < tol
        f = f_next
        if converged:
            break
    return f


def class_mass_normalize(scores: torch.Tensor, y: torch.Tensor, known: torch.Tensor):
    """Rescale each class column so that its total score over the unknown neurons matches the
    class's share among the known ones (Zhu et al., 2003). Without this, large classes absorb
    most of the propagated mass and small classes are rarely predicted."""
    prior = torch.bincount(y[known], minlength=scores.size(1)).float()
    mass = scores[~known].sum(0).clamp(min=1e-12)
    return scores * (prior / prior.sum()) / mass
