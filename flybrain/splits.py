"""Reproducible, seeded, stratified train/val/test splits over labeled nodes.

split="random": stratified random split over all labeled nodes.
split="left_right" / "right_left": train and validate on one hemisphere, test on the other.
"""

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
        n_test = max(1, round(len(idx) * test_frac)) if test_frac > 0 else 0
        n_val = max(1, round(len(idx) * val_frac))
        test[idx[:n_test]] = True
        val[idx[n_test : n_test + n_val]] = True
        train[idx[n_test + n_val :]] = True
    return torch.from_numpy(train), torch.from_numpy(val), torch.from_numpy(test)


SPLITS = ("random", "left_right", "right_left")


def hemisphere_split(
    y: torch.Tensor, side: np.ndarray, train_side: str, seed: int = 0, val_frac: float = 0.15
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Train/val masks from one hemisphere ("left" or "right"), test mask from the other.

    Validation is a stratified fraction of the training hemisphere. Neurons on the midline
    (side "center" or missing) are left out. Test neurons whose class has no training neuron are
    left out too, since no model can predict them.
    """
    test_side = {"left": "right", "right": "left"}[train_side]
    labeled = y.numpy() >= 0
    on_train = torch.from_numpy(labeled & (side == train_side))
    y_train_side = torch.where(on_train, y, torch.full_like(y, -1))
    train, val, _ = stratified_split(y_train_side, seed=seed, val_frac=val_frac, test_frac=0.0)
    seen = np.isin(y.numpy(), y.numpy()[train.numpy()])
    test = torch.from_numpy(labeled & (side == test_side) & seen)
    return train, val, test


def make_split(
    data, split: str = "random", seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dispatch to stratified_split or hemisphere_split (needs data.side)."""
    if split == "random":
        return stratified_split(data.y, seed=seed)
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    if getattr(data, "side", None) is None:
        raise ValueError(f"split={split!r} needs a side column in the neurons table")
    return hemisphere_split(data.y, data.side, split.split("_")[0], seed=seed)
