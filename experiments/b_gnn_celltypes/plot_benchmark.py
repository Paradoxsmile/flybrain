"""README figure: test accuracy of every method on every split, from the benchmark runs.

One dot per split (mean over seeds, whiskers = std); a hairline joins a method's dots, so its
length shows how much the method loses on an unlabeled hemisphere. Writes a light and a dark
version for GitHub's theme switch.

Usage: python experiments/b_gnn_celltypes/plot_benchmark.py [--runs results/benchmark/runs.csv]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from flybrain.utils import REPO_ROOT

# top to bottom: the GNN first, then baselines by decreasing information
METHODS = {
    "gnn": "GNN (inputs + outputs)",
    "gnn_plain": "GraphSAGE (inputs only)",
    "partner_profile": "Partner profile (partner labels)",
    "label_prop": "Label propagation (partner labels)",
    "logreg": "Logistic regression (own degrees)",
    "majority": "Majority class",
}
SPLITS = {"random": "random split", "left_right": "left → right", "right_left": "right → left"}
MARKERS = {"random": "o", "left_right": "s", "right_left": "^"}
DODGE = {"random": -0.22, "left_right": 0.0, "right_left": 0.22}  # keeps equal values visible
LABELS = {"cell_type": "Cell types (434 classes)", "super_class": "Super classes (10 classes)"}
THEMES = {
    "light": dict(
        surface="#fcfcfb",
        ink="#0b0b0b",
        ink2="#52514e",
        muted="#898781",
        grid="#e1e0d9",
        axis="#c3c2b7",
        span="#c3c2b7",
        series={"random": "#2a78d6", "left_right": "#eb6834", "right_left": "#1baf7a"},
    ),
    "dark": dict(
        surface="#1a1a19",
        ink="#f0efec",
        ink2="#c3c2b7",
        muted="#898781",
        grid="#2c2c2a",
        axis="#383835",
        span="#52514e",
        series={"random": "#3987e5", "left_right": "#d95926", "right_left": "#199e70"},
    ),
}


def plot(stats: pd.DataFrame, theme: dict, path: Path) -> None:
    """stats: mean and std of test accuracy (%) indexed by (label, method, split)."""
    plt.rcParams.update({"font.size": 10, "font.family": "sans-serif"})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True, facecolor=theme["surface"])
    rows = list(METHODS)
    for ax, label in zip(axes, LABELS, strict=True):
        ax.set_facecolor(theme["surface"])
        for y, method in enumerate(rows):
            means = stats.loc[(label, method), "mean"]
            ax.plot([means.min(), means.max()], [y, y], color=theme["span"], lw=1.5, zorder=1)
            for split in SPLITS:
                m, s = stats.loc[(label, method, split)]
                color = theme["series"][split]
                yd = y + DODGE[split]
                ax.errorbar(m, yd, xerr=s, fmt="none", ecolor=color, elinewidth=1.2, zorder=2)
                ax.plot(
                    m,
                    yd,
                    MARKERS[split],
                    ms=8,
                    color=color,
                    mec=theme["surface"],
                    mew=1.5,
                    zorder=3,
                )
        ax.set_title(LABELS[label], loc="left", color=theme["ink"], fontsize=11, pad=8)
        ax.set_xlim(-2, 102)
        ax.set_xticks(range(0, 101, 25))
        ax.set_xlabel("test accuracy (%)", color=theme["ink2"])
        ax.grid(axis="x", color=theme["grid"], lw=1)
        ax.set_axisbelow(True)
        ax.tick_params(colors=theme["muted"], length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(theme["axis"])
    axes[0].set_yticks(range(len(rows)), [METHODS[m] for m in rows], color=theme["ink2"])
    axes[0].set_ylim(len(rows) - 0.5, -0.5)
    handles = [
        plt.Line2D(
            [],
            [],
            ls="",
            marker=MARKERS[s],
            ms=8,
            color=theme["series"][s],
            mec=theme["surface"],
            mew=1.5,
            label=SPLITS[s],
        )
        for s in SPLITS
    ]
    legend = fig.legend(
        handles=handles,
        loc="upper right",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.99, 1.0),
        handletextpad=0.3,
        columnspacing=1.2,
    )
    for text in legend.get_texts():
        text.set_color(theme["ink2"])
    fig.suptitle(
        "FlyWire FAFB, mean ± std over 3 seeds",
        x=0.01,
        ha="left",
        y=0.98,
        color=theme["ink"],
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=200, facecolor=theme["surface"])
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=Path, default=REPO_ROOT / "results/benchmark/runs.csv")
    p.add_argument("--out", type=Path, default=REPO_ROOT / "docs")
    args = p.parse_args()

    runs = pd.read_csv(args.runs)
    runs["test_acc"] *= 100
    stats = runs.groupby(["label", "method", "split"]).test_acc.agg(["mean", "std"]).fillna(0)
    args.out.mkdir(parents=True, exist_ok=True)
    for name, theme in THEMES.items():
        plot(stats, theme, args.out / f"benchmark_{name}.png")
        print(f"wrote {args.out / f'benchmark_{name}.png'}")


if __name__ == "__main__":
    main()
