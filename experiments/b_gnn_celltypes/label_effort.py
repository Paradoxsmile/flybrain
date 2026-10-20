"""Labeling effort: how many hand labels does the label-free GNN save an annotation team?

On a target connectome, k % of the neurons with a reference cell type are labeled by hand
(drawn at random within each type, round(k % x type size) per type; 3 draws), and the rest are
scored. Three labelings:
- NTAC: seeded NTAC (flybrain.ntac, 15 steps, learning rate 0.3) with the k % hand labels only;
- GNN + NTAC: seeded with the hand labels plus the label-free GNN predictions where all three
  GNNs agree (hand labels override), as in hybrid.py; at k = 0 this is the label-free pipeline;
- NTAC + GNN fallback (no extra run): the NTAC labeling, except where the label-free pipeline
  predicts a type that has no hand label (NTAC cannot assign it), which keeps the pipeline's;
- GNN: the label-free pipeline without hand labels (ensemble with prior matching, and + NTAC),
  a flat line.

Targets (the label-free runs of the paper, GNNs trained on FAFB, nothing retrained):
- mcns: FAFB -> MCNS (results/label_free), reference = MCNS cell types;
- banc_a: FAFB -> BANC (results/label_free_fafb_banc), reference A = BANC typing;
- banc_b: the same run, reference B = morphology-matched FAFB type (central brain only); hand
  labels come from reference B (docs/banc_protocol.md).
NTAC runs on the target graph of the chosen candidate (as label_free.py's refinement). The
per-seed GNN labelings of the chosen candidate are recomputed once from the saved models
(stratify.py's arguments) and cached; GNN + NTAC at k = 0 must reproduce label_free.py's
refined labeling exactly.

Scores on the unlabeled neurons with a reference type: accuracy and macro-F1 (over the types
present among them), overall and by region (the FAFB super class of the true type,
stratify.py). Headline: the k at which NTAC alone matches GNN + NTAC at k = 1, 2 and 5 %
(log-linear interpolation of mean accuracy between the measured k).

Writes results/<out>/runs/<target>_<arm>_k<k>_s<seed>.npy (labels, cached), metrics.csv,
strata.csv, tables.md, label_effort.png / .pdf.

Usage: python experiments/b_gnn_celltypes/label_effort.py [--targets mcns banc_a banc_b]
"""

from __future__ import annotations

import argparse
import json
import time

import label_free
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import stratify
import torch

from flybrain.loading import encode_labels
from flybrain.ntac import ntac_from_predictions, ntac_seeded
from flybrain.utils import REPO_ROOT

RESULTS = REPO_ROOT / "results"
LABEL = "cell_type"
KS = (0, 1, 2, 5, 10, 20, 50)
SEEDS = (0, 1, 2)
NTAC_LR, NTAC_STEPS = 0.3, 15
TARGETS = {  # name: (label-free run, source, target, reference, title)
    "mcns": ("label_free", "fafb", "mcns", "y", "MCNS"),
    "banc_a": ("label_free_fafb_banc", "fafb", "banc", "y", "BANC, reference A (typing)"),
    "banc_b": ("label_free_fafb_banc", "fafb", "banc", "morph", "BANC, reference B (morphology)"),
}
ARMS = {"ntac": "NTAC (hand labels)", "gnn_ntac": "GNN + NTAC (hand labels + GNN seeds)"}
CURVES = {**ARMS, "fallback": "NTAC, GNN pipeline for types without hand labels"}
FLAT = {"gnn": "GNN (no hand labels)", "pipeline": "GNN pipeline + NTAC (no hand labels)"}
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
COLORS = {
    "ntac": "#eb6834",
    "gnn_ntac": "#2a78d6",
    "fallback": "#8a5cd1",
    "pipeline": "#1baf7a",
    "gnn": MUTED,
}


def prepare(run: str, source: str, target: str, cache_dir):
    """(chosen target graph, per-seed GNN labelings [seeds, n], ensemble, selected.json)."""
    d = RESULTS / run / LABEL
    sel = json.loads((d / "selected.json").read_text())
    args = stratify.run_args(sel, source, target)
    chosen = sel["chosen"]
    cases, _, _ = label_free.setups(LABEL, args)
    _, paths, src, g = next(c for c in cases if c[0] == chosen["variant"])
    cache = cache_dir / f"{run}_seed_preds.npz"
    if not cache.exists():
        print(f"[label_effort] {run}: applying the saved GNNs to the chosen candidate", flush=True)
        probs, weights, _ = label_free.seed_outputs(paths, src, g, chosen["norm"], args, {})
        ensemble, seed_preds = label_free.adjusted(probs, weights, chosen["rho"])
        np.savez_compressed(cache, seed_preds=np.stack(seed_preds), ensemble=ensemble)
    z = np.load(cache)
    names = np.array(g.label_names, dtype=object)
    saved = pd.read_parquet(d / "predictions.parquet")
    assert (saved.root_id.to_numpy() == g.node_ids.numpy()).all()
    if "pred_selected" not in saved.columns:  # older runs: stratify.py's cache
        stages = pd.read_parquet(d / "stages.parquet")
        saved = saved.merge(stages, on="root_id", how="left", validate="1:1")
    assert (names[z["ensemble"]] == saved.pred_selected.to_numpy()).all(), "ensemble differs"
    return g, z["seed_preds"], z["ensemble"], saved.pred.to_numpy()


def reference(g, kind: str) -> np.ndarray:
    if kind == "y":
        return g.y.numpy()
    morph = pd.Series(g.morph_type).replace("", pd.NA)
    return np.asarray(encode_labels(morph, label_names=g.label_names)[0])


def hand_labels(ref: np.ndarray, k: float, seed: int) -> np.ndarray:
    """Mask of neurons labeled by hand: round(k % of each type), drawn at random."""
    rng = np.random.default_rng(seed)
    mask = np.zeros(len(ref), dtype=bool)
    for t in np.unique(ref[ref >= 0]):
        idx = np.nonzero(ref == t)[0]
        mask[rng.permutation(idx)[: int(round(k / 100 * len(idx)))]] = True
    return mask


def run_ntac(g, known: np.ndarray, seed_preds, arm: str) -> np.ndarray:
    k = len(g.label_names)
    if arm == "gnn_ntac":
        return ntac_from_predictions(
            g.edge_index, g.edge_weight, known, seed_preds, k, NTAC_LR, NTAC_STEPS
        )[0]
    *_, labels = ntac_seeded(
        g.edge_index,
        g.edge_weight,
        torch.from_numpy(known),
        torch.from_numpy(known >= 0),
        k,
        NTAC_LR,
        NTAC_STEPS,
    )
    return labels


def evaluate(name, labels: np.ndarray, ref, ev, names, types) -> list:
    """Overall and per-region rows (stratify.per_type / strata) on the evaluated neurons."""
    pred = pd.DataFrame({"label": names[ref[ev]], "pred_x": names[labels[ev]]})
    s = stratify.strata(stratify.per_type(pred, "x", types))
    s = s[s.kind.isin(["all", "region"])]
    return [dict(r, **name) for r in s.to_dict("records")]


def crossing(ks, acc, level) -> float:
    """Smallest k (log-linear interpolation) at which the curve reaches level; nan if never."""
    pairs = list(zip(ks, acc, strict=True))
    for (k0, a0), (k1, a1) in zip(pairs, pairs[1:], strict=False):
        if a0 >= level:
            return k0
        if a1 >= level:
            f = (level - a0) / (a1 - a0)
            return float(np.exp(np.log(k0) + f * (np.log(k1) - np.log(k0))))
    return float("nan")


def figure(m: pd.DataFrame, targets: list, out) -> None:
    fig, axes = plt.subplots(
        2, len(targets), figsize=(4.0 * len(targets), 6.2), facecolor="white", squeeze=False
    )
    for j, t in enumerate(targets):
        for i, (metric, ylabel) in enumerate((("acc", "accuracy"), ("macro_f1", "macro-F1"))):
            ax = axes[i, j]
            d = m[(m.target == t) & (m.stratum == "all")]
            for arm in ("pipeline", "gnn"):
                v = d[d.arm == arm][metric].mean()
                ax.axhline(v, color=COLORS[arm], lw=1.4, ls=(0, (4, 3)), label=FLAT[arm])
            for arm in CURVES:
                g = d[(d.arm == arm) & (d.k > 0)].groupby("k")[metric]
                mean, lo, hi = g.mean(), g.min(), g.max()
                ax.fill_between(mean.index, lo, hi, color=COLORS[arm], alpha=0.18, lw=0)
                ax.plot(
                    mean.index, mean, color=COLORS[arm], marker="o", ms=4, lw=1.8, label=CURVES[arm]
                )
            ax.set_xscale("log")
            ax.set_xticks([k for k in KS if k > 0], [f"{k}" for k in KS if k > 0])
            ax.minorticks_off()
            ax.set_ylim(0, 1)
            ax.grid(axis="y", color=GRID, lw=1)
            ax.set_axisbelow(True)
            ax.tick_params(colors=MUTED, length=0)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(AXIS)
            if i == 0:
                ax.set_title(TARGETS[t][4], loc="left", color=INK, fontsize=10)
            if i == 1:
                ax.set_xlabel("hand-labeled neurons (% of each type)", color=INK2)
            if j == 0:
                ax.set_ylabel(f"{ylabel} on unlabeled neurons", color=INK2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    leg = fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=8,
        bbox_to_anchor=(0.5, -0.04),
    )
    for text in leg.get_texts():
        text.set_color(INK2)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(out / f"label_effort.{ext}", dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def summary(m: pd.DataFrame, counts: dict, targets: list) -> list:
    regions = [r for r in stratify.REGION_STYLE if r in set(m.stratum)]
    lines = [
        "Labeling effort: accuracy % / macro-F1 on the neurons without a hand label, mean of 3",
        "random draws of the hand labels (min-max of accuracy in brackets for k > 0). k = % of",
        "each type's reference neurons labeled by hand. GNN and pipeline need no hand labels.",
        "",
    ]
    for t in targets:
        d = m[m.target == t]
        lines += [f"### {TARGETS[t][4]} ({counts[t]:,} reference neurons)", ""]
        cols = ["all"] + regions
        lines += ["| method | k | " + " | ".join(cols) + " |", "|---" * (len(cols) + 2) + "|"]
        for arm in ("gnn", "pipeline", "ntac", "fallback", "gnn_ntac"):
            name = {**FLAT, **CURVES}[arm]
            for k in [k for k in KS if k > 0] if arm in CURVES else (0,):
                e = d[(d.arm == arm) & (d.k == k)]
                if e.empty:
                    continue
                cells = []
                for c in cols:
                    s = e[e.stratum == c]
                    if s.empty:
                        cells.append("–")
                        continue
                    cell = f"{100 * s.acc.mean():.1f} / {s.macro_f1.mean():.2f}"
                    if c == "all" and k > 0:
                        cell += f" [{100 * s.acc.min():.1f}–{100 * s.acc.max():.1f}]"
                    cells.append(f"**{cell}**" if c == "all" else cell)
                lines.append(f"| {name} | {k} | " + " | ".join(cells) + " |")
        lines.append("")
    lines += [
        "### Hand labels NTAC alone needs to match GNN + NTAC",
        "",
        "k* = % of each type that NTAC alone needs (log-linear interpolation of the mean over",
        "draws) to reach GNN + NTAC at k, by accuracy and by macro-F1; '> 50' if it never does",
        "in the range, '< 1' if NTAC alone at 1 % is already better. Hand labels in neurons.",
        "",
        "| target | k | GNN + NTAC acc / F1 | k* by accuracy | k* by macro-F1 | hand labels at k "
        "| NTAC alone, by acc / by F1 |",
        "|---|---|---|---|---|---|---|",
    ]
    for t in targets:
        d = m[(m.target == t) & (m.stratum == "all")]
        nt = d[(d.arm == "ntac") & (d.k > 0)].groupby("k")[["acc", "macro_f1"]].mean()
        for k in (1, 2, 5):
            e = d[(d.arm == "gnn_ntac") & (d.k == k)]
            ks, ns = [], []
            for metric in ("acc", "macro_f1"):
                level = e[metric].mean()
                x = crossing(list(nt.index), list(nt[metric]), level)
                if np.isnan(x):
                    ks.append("> 50")
                    ns.append(f"> {0.5 * counts[t]:,.0f}")
                elif x <= nt.index[0] and nt[metric].iloc[0] >= level:
                    ks.append("< 1")
                    ns.append(f"< {0.01 * counts[t]:,.0f}")
                else:
                    ks.append(f"{x:.1f}")
                    ns.append(f"~{x / 100 * counts[t]:,.0f}")
            lines.append(
                f"| {TARGETS[t][4]} | {k} | {100 * e.acc.mean():.1f} / {e.macro_f1.mean():.2f} | "
                f"{ks[0]} | {ks[1]} | {e.n_hand.mean():,.0f} | {ns[0]} / {ns[1]} |"
            )
    return lines


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--targets", nargs="+", default=list(TARGETS), choices=list(TARGETS))
    p.add_argument("--ks", nargs="+", type=float, default=list(KS))
    p.add_argument("--out", default="label_effort")
    args = p.parse_args()
    out = RESULTS / args.out
    (out / "runs").mkdir(parents=True, exist_ok=True)
    types = stratify.fafb_types()
    rows, counts, graphs = [], {}, {}
    for t in args.targets:
        run, source, target, kind, _ = TARGETS[t]
        if run not in graphs:
            graphs[run] = prepare(run, source, target, out)
        g, seed_preds, ensemble, saved_refined = graphs[run]
        names = np.array(g.label_names, dtype=object)
        ref = reference(g, kind)
        counts[t] = int((ref >= 0).sum())
        index = {n: i for i, n in enumerate(names)}
        pipeline = np.array([index[n] for n in saved_refined])
        for k in args.ks:
            k = int(k) if float(k).is_integer() else k
            for seed in SEEDS if k > 0 else (0,):
                hand = hand_labels(ref, k, seed)
                known = np.where(hand, ref, -1)
                ev = (ref >= 0) & ~hand
                tag = {"target": t, "k": k, "seed": seed, "n_hand": int(hand.sum())}
                for arm in ARMS:
                    if arm == "ntac" and k == 0:
                        continue
                    path = out / "runs" / f"{t}_{arm}_k{k}_s{seed}.npy"
                    if arm == "gnn_ntac" and k == 0:  # the same run for every reference
                        path = out / "runs" / f"{run}_gnn_ntac_k0.npy"
                    if not path.exists():
                        t0 = time.time()
                        labels = run_ntac(g, known, seed_preds, arm)
                        np.save(path, labels.astype(np.int32))
                        print(
                            f"[label_effort] {t} {arm} k={k} s={seed}: {time.time() - t0:.0f} s",
                            flush=True,
                        )
                    labels = np.load(path)
                    if arm == "gnn_ntac" and k == 0:
                        assert (names[labels] == saved_refined).all(), "k = 0 is not the pipeline"
                        rows += evaluate({**tag, "arm": "pipeline"}, labels, ref, ev, names, types)
                        rows += evaluate({**tag, "arm": "gnn"}, ensemble, ref, ev, names, types)
                    rows += evaluate({**tag, "arm": arm}, labels, ref, ev, names, types)
                    if arm == "ntac":  # NTAC can only assign hand-labeled types
                        fb = np.where(np.isin(pipeline, np.unique(known[hand])), labels, pipeline)
                        rows += evaluate({**tag, "arm": "fallback"}, fb, ref, ev, names, types)
    m = pd.DataFrame(rows)
    m.to_csv(out / "strata.csv", index=False)
    m[m.stratum == "all"].to_csv(out / "metrics.csv", index=False)
    lines = summary(m, counts, args.targets)
    (out / "tables.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    plt.rcParams.update({"font.size": 9, "font.family": "sans-serif"})
    figure(m, args.targets, out)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
