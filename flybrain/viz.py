"""Basic plotting helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _finish(fig: plt.Figure, path: str | Path | None) -> plt.Figure:
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=150)
    return fig


def plot_class_counts(
    y: np.ndarray, names: list[str], path: str | Path | None = None
) -> plt.Figure:
    """Bar plot of labeled-node counts per class (log scale)."""
    counts = np.bincount(y[y >= 0], minlength=len(names))
    order = np.argsort(counts)[::-1]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(np.arange(len(names)), counts[order])
    ax.set_xticks(np.arange(len(names)), [names[i] for i in order], rotation=90)
    ax.set_yscale("log")
    ax.set_ylabel("neurons")
    return _finish(fig, path)


def plot_degree_distribution(
    degrees: np.ndarray, path: str | Path | None = None, label: str = "degree"
) -> plt.Figure:
    """Log-log histogram of node degrees."""
    fig, ax = plt.subplots(figsize=(5, 4))
    bins = np.logspace(0, np.log10(max(degrees.max(), 2)), 40)
    ax.hist(degrees[degrees > 0], bins=bins)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(label)
    ax.set_ylabel("count")
    return _finish(fig, path)


def plot_training_curves(history: pd.DataFrame, path: str | Path | None = None) -> plt.Figure:
    """Plot loss and val accuracy per epoch from a history frame (epoch, loss, val_acc)."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.5))
    a1.plot(history["epoch"], history["loss"])
    a1.set_xlabel("epoch")
    a1.set_ylabel("train loss")
    a2.plot(history["epoch"], history["val_acc"])
    a2.set_xlabel("epoch")
    a2.set_ylabel("val accuracy")
    return _finish(fig, path)
