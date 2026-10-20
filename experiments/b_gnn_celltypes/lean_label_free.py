"""Memory-lean label-free mapping (label_free.py) for label spaces with thousands of classes
(size_threshold.py at min type size 3 or 2).

label_free.run_label keeps every seed's class probabilities on the target as a dense N x K float32
matrix, and flybrain.adapt.prior_weights converts each to float64 and builds a second N x K copy
in every Sinkhorn iteration: about 7 GB + 9 GB per seed for MCNS at K = 4,175, more than 16 GB of
RAM. run_label uses the probabilities only through seed_outputs and adjusted, so `patched()` swaps
in lean versions of those two (label_free.py itself is unchanged):
- LazyProbs keeps the input of the GNN's last layer (N x 3 hidden, see lean_training) and computes
  probability rows in chunks when needed;
- prior matching runs the same damped Sinkhorn iterations as adapt.prior_weights, in float64 over
  row chunks, with the soft counts written as counts_j = v_j * sum_i p_ij / (p_i . v);
- adjusted computes each chunk's adjusted probabilities with the same float32 operations as
  label_free.adjusted.
Peak memory is one seed's N x K float32 probabilities (during its prior matching) plus chunks.
Results agree with label_free.py up to float rounding (checked at min type size 10).
"""

from __future__ import annotations

from contextlib import contextmanager

import label_free
import numpy as np
import torch
from lean_training import features, head

from flybrain.features import normalize_features
from flybrain.splits import make_split
from flybrain.training import graph_inputs, load_model

CHUNK = 8192


class LazyProbs:
    """Class probabilities (softmax) of one GNN on one graph, computed in row chunks."""

    def __init__(self, model, cfg, graph, x, chunk: int = CHUNK):
        xx, adj = graph_inputs(graph, x, cfg, "cpu")
        with torch.no_grad():
            model.eval()
            self.z = features(model, xx, *adj)
            w, b = head(model)
            self.w, self.b = w.detach(), b.detach()
        self.shape = (self.z.shape[0], self.w.shape[0])
        self.chunk = chunk

    def __len__(self) -> int:
        return self.shape[0]

    @torch.no_grad()
    def rows(self, start: int, stop: int) -> np.ndarray:
        logits = self.z[start:stop] @ self.w.T + self.b
        return torch.softmax(logits, dim=1).numpy()

    def chunks(self):
        for start in range(0, len(self), self.chunk):
            stop = min(start + self.chunk, len(self))
            yield start, stop, self.rows(start, stop)

    def dense(self) -> np.ndarray:
        out = np.empty(self.shape, dtype=np.float32)
        for start, stop, p in self.chunks():
            out[start:stop] = p
        return out


def column_sums(probs: LazyProbs, row_weights: np.ndarray | None = None) -> np.ndarray:
    """sum_i w_i p_i (float64), w = 1 by default."""
    total = np.zeros(probs.shape[1])
    for start, stop, p in probs.chunks():
        p = p.astype(np.float64)
        total += p.sum(axis=0) if row_weights is None else row_weights[start:stop] @ p
    return total


def prior_weights_lean(
    probs: np.ndarray,
    target: np.ndarray,
    iters: int = 200,
    tol: float = 1e-6,
    max_weight: float | None = None,
    chunk: int = CHUNK,
) -> np.ndarray:
    """flybrain.adapt.prior_weights on a float32 matrix, in float64 row chunks."""
    target = np.asarray(target, dtype=np.float64)
    n, k = probs.shape
    v = np.ones(k)
    for _ in range(iters):
        vt = torch.from_numpy(v)
        acc = torch.zeros(k, dtype=torch.float64)
        for start in range(0, n, chunk):
            p = torch.from_numpy(probs[start : start + chunk]).double()
            acc += p.T @ (1.0 / (p @ vt))
        counts = v * acc.numpy()
        ratio = np.where(counts > 0, target / np.maximum(counts, 1e-300), 1.0)
        v = v * np.sqrt(ratio)
        if max_weight is not None:  # weights only matter relative to each other
            v /= np.exp((target * np.log(v)).sum() / target.sum())
            v = np.clip(v, 1.0 / max_weight, max_weight)
        if np.max(np.abs(counts - target)) < tol * n:
            break
    return v


def type_prior(p: LazyProbs, ref: LazyProbs, args, covered) -> tuple[np.ndarray, np.ndarray]:
    """label_free.type_prior for priors "source" and "coverage"."""
    if args.prior == "em":
        raise NotImplementedError("lean label-free mapping supports --prior source / coverage")
    if args.prior == "coverage":
        shares = column_sums(ref, covered) / covered.sum()
    else:
        shares = column_sums(ref) / len(ref)
    dense = p.dense()  # one seed at a time
    weights = prior_weights_lean(dense, shares * len(p), max_weight=args.max_weight)
    return shares, weights


def seed_outputs(paths, src, tgt, norm: str, args, covered: dict) -> tuple[list, list, list]:
    """label_free.seed_outputs with LazyProbs in place of dense probabilities."""
    seed_probs, seed_weights, refs = [], [], []
    for path in paths:
        model, cfg, names = load_model(path)
        assert names == src.label_names == tgt.label_names
        train = make_split(src, "random", seed=cfg["seed"])[0]
        ref = LazyProbs(model, cfg, src, normalize_features(src.x, "global", train))
        x = label_free.target_features(tgt, src, train, norm, covered.get(id(src)))
        p = LazyProbs(model, cfg, tgt, x)
        shares, weights = type_prior(p, ref, args, covered.get(id(src)))
        seed_weights.append(weights)
        seed_probs.append(p)
        refs.append(shares)
    return seed_probs, seed_weights, refs


def adjusted(seed_probs: list, seed_weights: list, rho: float) -> tuple[np.ndarray, list]:
    """label_free.adjusted, chunk by chunk."""
    n = len(seed_probs[0])
    ensemble = np.empty(n, dtype=np.int64)
    seed_preds = [np.empty(n, dtype=np.int64) for _ in seed_probs]
    factors = [(v**rho).astype(np.float32) for v in seed_weights]
    chunk = seed_probs[0].chunk
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        total = None
        for i, (p, f) in enumerate(zip(seed_probs, factors, strict=True)):
            q = p.rows(start, stop) * f
            q /= q.sum(axis=1, keepdims=True)
            total = q if total is None else total + q
            seed_preds[i][start:stop] = q.argmax(axis=1)
        ensemble[start:stop] = total.argmax(axis=1)
    return ensemble, seed_preds


@contextmanager
def patched():
    """Run label_free.run_label with the lean seed_outputs and adjusted."""
    saved = label_free.seed_outputs, label_free.adjusted
    label_free.seed_outputs, label_free.adjusted = seed_outputs, adjusted
    try:
        yield
    finally:
        label_free.seed_outputs, label_free.adjusted = saved
