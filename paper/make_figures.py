"""Figures of the paper draft, from the result files (results/ is not versioned; see README).

  rare_types.png / .pdf main figure: the two-stage pipeline and its accuracy on rare types
                        per source -> target pair, with the morphology rows
  transfer.png / .pdf   label-free transfer per scenario: without prior matching, the
                        label-free choice, + NTAC, the size-and-agreement rule, and the
                        accuracy of a GNN trained with labels on the target
  thinning.png / .pdf   transfer between thinned FAFB copies (source x target density), with
                        prior matching, and the density-augmented GNN

Usage: python paper/make_figures.py [--out paper/figures]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from flybrain.selection import choose
from flybrain.utils import REPO_ROOT

RESULTS = REPO_ROOT / "results"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1baf7a"

# (row label, run of the reported pipeline, run with the label-free columns for the
# size-and-agreement rule, accuracy % of a GNN trained with labels on the target (README),
# pre-registered)
SCENARIOS = [
    (
        "FAFB → MCNS",
        "label_free",
        "label_free_fafb_mcns_r1",
        {"cell_type": 96.6, "super_class": 98.4},
        False,
    ),
    (
        "MCNS → FAFB",
        "label_free_rev",
        "label_free_mcns_fafb_r1",
        {"cell_type": 95.2, "super_class": 97.8},
        False,
    ),
    ("FAFB → BANC", "label_free_fafb_banc", "label_free_fafb_banc_cols", {}, True),
    ("MCNS → BANC", "label_free_mcns_banc", "label_free_mcns_banc_cols", {}, True),
    ("FAFB → hemibrain", "label_free_hemibrain", "label_free_hemibrain", {}, True),
]
LABELS = {"cell_type": "Cell types", "super_class": "Super classes"}


def style(ax) -> None:
    ax.grid(axis="x", color=GRID, lw=1)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)


def revised_choice(run: str, label: str) -> float:
    """Accuracy % of the candidate the revised rule (size_agreement) chooses."""
    path = RESULTS / run / label / "candidates.csv"
    if not path.exists():  # MCNS -> BANC super classes were rerun in a separate directory
        path = RESULTS / f"{run}_super" / label / "candidates.csv"
    c = pd.read_csv(path)
    return 100 * c.acc[choose(c, "size_agreement")]


def transfer_figure(out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True, facecolor="white")
    for ax, label in zip(axes, LABELS, strict=True):
        for y, (_, run, cols_run, ceiling, _) in enumerate(SCENARIOS):
            sel = json.loads((RESULTS / run / label / "selected.json").read_text())
            pts = {
                "none": 100 * sel["no_adaptation_acc"],
                "chosen": 100 * sel["selected_acc"],
                "ntac": 100 * sel["refined_acc"],
                "revised": revised_choice(cols_run, label),
            }
            lo, hi = min(pts.values()), max(pts.values())
            ax.plot([lo, hi], [y, y], color=AXIS, lw=1.5, zorder=1)
            if label in ceiling:
                ax.plot(ceiling[label], y, "|", ms=14, mew=2, color=INK2, zorder=2)
            ax.plot(pts["none"], y, "o", ms=8, mfc="white", mec=BLUE, mew=1.8, zorder=3)
            ax.plot(pts["chosen"], y, "o", ms=8, color=BLUE, mec="white", mew=1.2, zorder=4)
            ax.plot(pts["ntac"], y, "D", ms=7, color=ORANGE, mec="white", mew=1.2, zorder=5)
            ax.plot(pts["revised"], y, "^", ms=8, color=GREEN, mec="white", mew=1.2, zorder=6)
        ax.set_title(LABELS[label], loc="left", color=INK, fontsize=11, pad=8)
        ax.set_xlim(-2, 102)
        ax.set_xticks(range(0, 101, 25))
        ax.set_xlabel("accuracy on labeled target neurons (%)", color=INK2)
        style(ax)
    names = [f"{s[0]}{' *' if s[4] else ''}" for s in SCENARIOS]
    axes[0].set_yticks(range(len(SCENARIOS)), names, color=INK2)
    axes[0].set_ylim(len(SCENARIOS) - 0.5, -0.5)
    handles = [
        plt.Line2D(
            [],
            [],
            ls="",
            marker="o",
            ms=8,
            mfc="white",
            mec=BLUE,
            mew=1.8,
            label="without prior matching",
        ),
        plt.Line2D([], [], ls="", marker="o", ms=8, color=BLUE, label="label-free choice"),
        plt.Line2D([], [], ls="", marker="D", ms=7, color=ORANGE, label="+ NTAC"),
        plt.Line2D([], [], ls="", marker="^", ms=8, color=GREEN, label="size-and-agreement rule"),
        plt.Line2D(
            [], [], ls="", marker="|", ms=14, mew=2, color=INK2, label="trained with target labels"
        ),
    ]
    leg = fig.legend(
        handles=handles,
        loc="lower center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
        handletextpad=0.3,
        columnspacing=1.2,
    )
    for t in leg.get_texts():
        t.set_color(INK2)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(out / f"transfer.{ext}", dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def thinned_means(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Mean accuracy % over seeds with prior matching, source x target density."""
    d = df[(df.label == label) & (df.strategy == "as_is") & (df.rho == 1.0)]
    return d.groupby(["p_source", "p_target"]).acc.mean().unstack() * 100


def thinning_figure(out: Path) -> None:
    grid = pd.read_csv(RESULTS / "thinning" / "grid.csv")
    aug = pd.read_csv(RESULTS / "thinning_aug" / "grid.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), facecolor="white")
    for ax, label in zip(axes, LABELS, strict=True):
        m = thinned_means(grid, label).sort_index(ascending=False)
        m = m[sorted(m.columns, reverse=True)]
        a = thinned_means(aug, label)[m.columns]
        a.index = ["augmented"]
        table = pd.concat([m, a])
        values = table.to_numpy()
        ax.imshow(values, cmap="Blues", vmin=20, vmax=100, aspect="auto")
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                v = values[i, j]
                ax.text(
                    j,
                    i,
                    f"{v:.1f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="white" if v > 75 else INK,
                    fontweight="bold" if i == j else None,
                )
        ax.axhline(len(m) - 0.5, color="white", lw=3)
        ax.set_xticks(range(values.shape[1]), [f"{p:g}" for p in table.columns], color=INK2)
        rows = [f"{p:g}" for p in m.index] + ["augmented"]
        ax.set_yticks(range(values.shape[0]), rows, color=INK2)
        ax.set_xlabel("target: share of synapses kept", color=INK2)
        ax.set_ylabel("source GNN trained at", color=INK2)
        ax.set_title(f"{LABELS[label]}, accuracy (%)", loc="left", color=INK, fontsize=11)
        ax.tick_params(length=0)
        for side in ax.spines.values():
            side.set_visible(False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"thinning.{ext}", dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def _acc(path: str, **where) -> float:
    """Accuracy % of the one row of results/<path> matching all column = value pairs."""
    m = pd.read_csv(RESULTS / path)
    for col, val in where.items():
        m = m[m[col] == val]
    assert len(m) == 1, (path, where, len(m))
    return 100 * float(m.acc.iloc[0])


def rare_values() -> list:
    """Per source -> target pair (label, pre-registered, {marker: accuracy %}) on the 2-4 stratum
    (reference A)."""
    ci = "bootstrap_ci/metrics.csv"
    morph = "morphology_baseline/metrics.csv"
    mb = {"target": "banc", "reference": "ref A", "subset": "2-4"}
    mc = {"target": "banc_cord", "reference": "ref A", "subset": "2-4"}
    mcns = {"candidates": "union", "reference": "MCNS label", "subset": "2-4 (MCNS label)"}
    return [
        (
            "FlyWire left → right",
            False,
            {
                "stage 1": _acc(
                    "graph_matching/two_stage.csv", subset="2-4", method="gnn (majority)"
                ),
                "two-stage": _acc(
                    "graph_matching/two_stage.csv", subset="2-4", method="gnn + graph matching"
                ),
                "oracle": _acc("graph_matching/metrics.csv", method="gm_bisected"),
            },
        ),
        (
            "FlyWire → MCNS brain",
            False,
            {
                "stage 1": _acc(
                    "graph_matching_mcns_cand/metrics.csv",
                    method="stage 1 (label-free GNN)",
                    **mcns,
                ),
                "two-stage": _acc("graph_matching_mcns_cand/metrics.csv", method="gm_bary", **mcns),
                "oracle": _acc(
                    "graph_matching_mcns/metrics.csv",
                    candidates="oracle",
                    method="gm_bary",
                    subset="2-4 (MCNS label)",
                ),
            },
        ),
        (
            "FlyWire → BANC brain",
            True,
            {
                "stage 1": _acc(ci, test="BANC brain (Table 6)", labeling="stage 1"),
                "two-stage, pre-registered": _acc(
                    ci, test="BANC brain (Table 6)", labeling="two-stage, gnn"
                ),
                "two-stage": _acc(morph, method="two-stage [union]", **mb),
                "oracle": _acc(ci, test="BANC brain (Table 6)", labeling="oracle candidates"),
                "NBLAST": _acc(morph, method="NBLAST", **mb),
                "matched, else NBLAST": _acc(
                    morph, method="two-stage matched, else NBLAST [union]", **mb
                ),
            },
        ),
        (
            "MANC → BANC cord",
            True,
            {
                "stage 1": _acc(ci, test="BANC cord (Table 8)", labeling="stage 1"),
                "two-stage": _acc(ci, test="BANC cord (Table 8)", labeling="two-stage, union"),
                "oracle": _acc(ci, test="BANC cord (Table 8)", labeling="oracle candidates"),
                "NBLAST": _acc(morph, method="NBLAST", **mc),
                "matched, else NBLAST": _acc(
                    morph, method="two-stage matched, else NBLAST [union]", **mc
                ),
            },
        ),
        (
            "MANC → MCNS cord",
            True,
            {
                "stage 1": _acc(ci, test="MCNS cord (Table 8)", labeling="stage 1"),
                "two-stage": _acc(ci, test="MCNS cord (Table 8)", labeling="two-stage, union"),
                "oracle": _acc(ci, test="MCNS cord (Table 8)", labeling="oracle candidates"),
            },
        ),
    ]


def _box(ax, x, y, text, color, w=0.2, h=0.2) -> None:
    ax.add_patch(
        plt.Rectangle(
            (x - w / 2, y - h / 2), w, h, facecolor="white", edgecolor=color, lw=1.4, zorder=2
        )
    )
    ax.text(x, y, text, ha="center", va="center", fontsize=8.5, color=INK, zorder=3)


def _arrow(ax, x0, y0, x1, y1, color=MUTED) -> None:
    ax.annotate(
        "", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color=color, lw=1.2)
    )


def rare_figure(out: Path) -> None:
    """Main figure: the two-stage pipeline (a) and its accuracy on rare types per pair (b)."""
    fig = plt.figure(figsize=(10, 6.4), facecolor="white")
    a = fig.add_axes([0.02, 0.62, 0.96, 0.34])
    a.set_xlim(0, 1)
    a.set_ylim(0, 1)
    a.axis("off")
    a.set_title("a  Two-stage typing of rare cell types", loc="left", color=INK, fontsize=11)
    _box(a, 0.1, 0.72, "Annotated\nreference\nconnectome", INK2)
    _box(a, 0.1, 0.25, "Unlabeled\ntarget\nconnectome", INK2)
    _box(a, 0.36, 0.25, "Stage 1: GNN on\nanonymous wiring,\nlabel-free transfer", BLUE, w=0.22)
    _box(a, 0.62, 0.25, "Candidates:\nneurons assigned\na 2–4-neuron type", BLUE, w=0.22)
    _box(a, 0.62, 0.72, "Reference neurons\nof 2–4-neuron\ntypes", INK2, w=0.22)
    _box(a, 0.88, 0.48, "Stage 2: graph\nmatching, blocked\nby hemisphere", BLUE, w=0.2, h=0.26)
    _arrow(a, 0.2, 0.72, 0.51, 0.72)
    _arrow(a, 0.1, 0.62, 0.25, 0.32, color=MUTED)
    _arrow(a, 0.2, 0.25, 0.25, 0.25)
    _arrow(a, 0.47, 0.25, 0.51, 0.25)
    _arrow(a, 0.73, 0.25, 0.78, 0.42)
    _arrow(a, 0.73, 0.72, 0.78, 0.55)
    a.text(0.17, 0.48, "trains", fontsize=8, color=MUTED)
    a.text(
        0.88,
        0.12,
        "each matched candidate\ntakes its partner's type",
        ha="center",
        fontsize=8,
        color=MUTED,
    )

    b = fig.add_axes([0.2, 0.08, 0.77, 0.44])
    rows = rare_values()
    markers = {
        "stage 1": dict(marker="o", mfc="white", mec=MUTED, ms=7, label="Stage 1 (GNN alone)"),
        "two-stage, pre-registered": dict(
            marker="o",
            mfc="white",
            mec=BLUE,
            ms=7,
            label="Two-stage, strict candidates (pre-registered)",
        ),
        "two-stage": dict(
            marker="o", mfc=BLUE, mec=BLUE, ms=8, label="Two-stage (broad candidates where run)"
        ),
        "NBLAST": dict(marker="X", mfc=ORANGE, mec=ORANGE, ms=7, label="NBLAST (morphology only)"),
        "matched, else NBLAST": dict(
            marker="^", mfc=GREEN, mec=GREEN, ms=8, label="Matching-first combination"
        ),
        "oracle": dict(
            marker="D", mfc=INK2, mec=INK2, ms=5.5, label="Oracle candidates (uses labels)"
        ),
    }
    seen = set()
    n = len(rows)
    for i, (_, _, vals) in enumerate(rows):
        y = n - 1 - i
        b.plot([vals["stage 1"], vals["two-stage"]], [y, y], color=GRID, lw=3, zorder=1)
        for key, v in vals.items():
            kw = dict(markers[key])
            lab = kw.pop("label")
            lab = None if key in seen else lab
            seen.add(key)
            dy = -0.22 if key in ("NBLAST", "matched, else NBLAST") else 0.0
            b.plot([v], [y + dy], ls="none", zorder=3, label=lab, **kw)
    b.set_yticks(range(n))
    b.set_yticklabels([f"{r[0]}{' *' if r[1] else ''}" for r in rows][::-1], color=INK)
    b.set_xlim(15, 100)
    b.set_ylim(-0.6, n - 0.4)
    style(b)
    b.set_title(
        "b  Neurons of 2–4-neuron types, accuracy (%) without any target label",
        loc="left",
        color=INK,
        fontsize=11,
        x=-0.23,
    )
    b.legend(loc="upper center", bbox_to_anchor=(0.4, -0.1), fontsize=8, frameon=False, ncol=3)
    b.text(
        -0.23,
        -0.3,
        "* pre-registered test (morphology rows exploratory)",
        transform=b.transAxes,
        fontsize=8,
        color=MUTED,
    )
    for ext in ("png", "pdf"):
        fig.savefig(out / f"rare_types.{ext}", dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=REPO_ROOT / "paper" / "figures")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "font.family": "sans-serif"})
    rare_figure(args.out)
    transfer_figure(args.out)
    thinning_figure(args.out)
    print(f"figures written to {args.out}")


if __name__ == "__main__":
    main()
