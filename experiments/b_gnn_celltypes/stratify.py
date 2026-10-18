"""Stratified accuracy of the label-free transfers reported in the paper, by region and by type
size, at the three stages of the pipeline (without prior matching, with prior matching, + NTAC).

Strata are sets of cell types, so every labeled target neuron falls into exactly one:
- region: the FAFB super class of the neuron's (true) type: optic-lobe intrinsic, visual
  projection, central brain, sensory, other (ascending, descending, sensory ascending, visual
  centrifugal, motor, endocrine);
- type size: the number of FAFB neurons of the type (20-49, 50-199, 200-999, >= 1000).
Accuracy is the share of labeled neurons of the stratum that get their type; macro-F1 is the
mean over the stratum's types of each type's F1 on the whole labeled target (a wrong prediction
lowers the precision of the predicted type, whichever stratum it belongs to). Over all strata
this macro-F1 averages over the types present in the target only, so it can be slightly higher
than label_free.py's, which also counts predicted types that are absent from the target.
"points" is a stratum's contribution to the overall accuracy (its correct neurons / all labeled
neurons). Cell types only; for super classes the region strata would be the classes themselves.

No retraining. label_free.py saves the + NTAC labeling per neuron (predictions.parquet, column
pred); runs from before it also saved the other two stages (columns pred_no_adaptation,
pred_selected) are completed by applying the saved GNNs (3 seeds) once more with the chosen
alignment and normalization, at rho 0 and at the chosen rho (cached in stages.parquet), and
checked against the accuracies in selected.json.

Writes results/stratified/{strata,types}.csv and summary.md, and paper/figures/type_f1.png /
.pdf (per-type F1 against FAFB type size, FAFB -> MCNS after NTAC).

Usage: python experiments/b_gnn_celltypes/stratify.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import label_free
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from flybrain.coverage import coverage_weights
from flybrain.utils import REPO_ROOT

RESULTS = REPO_ROOT / "results"
LABEL = "cell_type"
RUNS = {  # the runs of the paper's Sections 4.2, 4.3 and 4.5: (directory, source, target)
    "FAFB → MCNS": ("label_free", "fafb", "mcns"),
    "MCNS → FAFB": ("label_free_rev", "mcns", "fafb"),
    "FAFB → BANC": ("label_free_fafb_banc", "fafb", "banc"),
    "MCNS → BANC": ("label_free_mcns_banc", "mcns", "banc"),
    "FAFB → hemibrain": ("label_free_hemibrain", "fafb", "hemibrain"),
}
STAGES = {
    "no_adaptation": "without prior matching",
    "selected": "with prior matching",
    "refined": "+ NTAC",
}
REGIONS = {
    "optic": "optic-lobe intrinsic",
    "visual_projection": "visual projection",
    "central": "central brain",
    "sensory": "sensory",
}
OTHER = "other"
SIZE_BINS = [(20, 49), (50, 199), (200, 999), (1000, np.inf)]
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
REGION_STYLE = {  # categorical slots 1-5 (validated); markers as a second channel
    "optic-lobe intrinsic": ("#2a78d6", "o"),
    "visual projection": ("#eb6834", "s"),
    "central brain": ("#1baf7a", "^"),
    "sensory": ("#eda100", "D"),
    OTHER: ("#e87ba4", "v"),
}


def size_bin(n: float) -> str:
    for lo, hi in SIZE_BINS:
        if lo <= n <= hi:
            return f"≥ {lo}" if hi == np.inf else f"{lo}–{hi}"
    return "< 20"


def fafb_types() -> pd.DataFrame:
    """Per FAFB cell type: region and size (number of FAFB neurons)."""
    n = pd.read_parquet(REPO_ROOT / "data" / "processed" / "neurons.parquet")
    n = n.dropna(subset=["cell_type"])
    t = n.groupby("cell_type").agg(super_class=("super_class", "first"), size=("root_id", "size"))
    t["region"] = t.super_class.map(REGIONS).fillna(OTHER)
    t["size_bin"] = t["size"].map(size_bin)
    return t


def run_args(sel: dict, source: str, target: str) -> argparse.Namespace:
    """label_free.py's arguments for a run, from its selected.json (older runs lack some)."""
    return argparse.Namespace(
        source=source,
        target=target,
        models=label_free.DEFAULT_MODELS[source],
        synthetic=False,
        score_all=False,
        target_retention=sel.get("target_retention", 1.0),
        retention_seed=2,
        target_exclude=sel.get("target_exclude", []),
        max_weight=sel.get("max_weight"),
        source_retention=sel.get("source_retention", 1.0),
        target_coverage=sel.get("target_coverage"),
        target_crop=sel.get("target_crop", 0.0),
        prior=sel.get("prior", "source"),
        coverage=sel.get("coverage"),
    )


def recompute(sel: dict, source: str, target: str) -> pd.DataFrame:
    """The labelings without and with prior matching of the chosen candidate."""
    args = run_args(sel, source, target)
    chosen = sel["chosen"]
    cases, _, _ = label_free.setups(LABEL, args)
    _, paths, src, tgt = next(c for c in cases if c[0] == chosen["variant"])
    covered = {}
    if args.coverage:
        covered[id(src)] = coverage_weights(src.node_ids.numpy(), args.coverage)
    probs, weights, _ = label_free.seed_outputs(paths, src, tgt, chosen["norm"], args, covered)
    names = np.array(tgt.label_names, dtype=object)
    out = pd.DataFrame({"root_id": tgt.node_ids.numpy()})
    for stage, rho in (("no_adaptation", 0.0), ("selected", chosen["rho"])):
        out[f"pred_{stage}"] = names[label_free.adjusted(probs, weights, rho)[0]]
    return out


def labelings(run: str, source: str, target: str) -> pd.DataFrame:
    """Labeled target neurons with their label and the prediction of every stage."""
    d = RESULTS / run / LABEL
    sel = json.loads((d / "selected.json").read_text())
    pred = pd.read_parquet(d / "predictions.parquet").rename(columns={"pred": "pred_refined"})
    if "pred_selected" not in pred.columns:
        cache = d / "stages.parquet"
        if not cache.exists():
            print(f"[stratify] {run}: applying the saved GNNs to the chosen candidate", flush=True)
            recompute(sel, source, target).to_parquet(cache, index=False)
        pred = pred.merge(pd.read_parquet(cache), on="root_id", how="left", validate="1:1")
    pred = pred[pred.label.notna()]
    reported = {"no_adaptation": "no_adaptation_acc", "selected": "selected_acc"}
    reported["refined"] = "refined_acc"
    for stage, key in reported.items():
        acc = (pred[f"pred_{stage}"] == pred.label).mean()
        if abs(acc - sel[key]) > 1e-6:
            raise RuntimeError(f"{run} {stage}: accuracy {acc:.6f}, selected.json {sel[key]:.6f}")
    return pred


def per_type(pred: pd.DataFrame, stage: str, types: pd.DataFrame) -> pd.DataFrame:
    """Per true type: neurons, correct, predicted (among labeled neurons), F1."""
    p = pred[f"pred_{stage}"]
    hit = p == pred.label
    t = pd.DataFrame(
        {
            "n": pred.label.value_counts(),
            "correct": pred.label[hit].value_counts(),
            "predicted": p.value_counts(),
        }
    )
    t = t[t.n.notna()].fillna(0)
    t["f1"] = 2 * t.correct / (t.n + t.predicted)
    t = t.join(types[["region", "size", "size_bin"]], how="left")
    t["region"] = t.region.fillna(OTHER)
    return t.rename_axis("cell_type").reset_index()


def strata(t: pd.DataFrame) -> pd.DataFrame:
    rows = []
    total = t.n.sum()
    groups = [("all", "all", t)]
    groups += [("region", r, g) for r, g in t.groupby("region")]
    groups += [("type size", b, g) for b, g in t.groupby("size_bin")]
    for kind, name, g in groups:
        rows.append(
            {
                "kind": kind,
                "stratum": name,
                "n_neurons": int(g.n.sum()),
                "n_types": len(g),
                "share": g.n.sum() / total,
                "acc": g.correct.sum() / g.n.sum(),
                "macro_f1": g.f1.mean(),
                "points": g.correct.sum() / total,
            }
        )
    return pd.DataFrame(rows)


def percent(share: float) -> str:
    return f"{100 * share:.1f} %" if share < 0.01 else f"{100 * share:.0f} %"


def markdown(df: pd.DataFrame, kind: str, order: list) -> list:
    """One row per run and stage, one column per stratum (accuracy % / macro-F1), and a row with
    each stratum's share of the labeled neurons."""
    cols = ["all"] + order
    lines = ["| Run | Stage | " + " | ".join(cols) + " |", "|---" * (len(cols) + 2) + "|"]
    for run in RUNS:
        d = df[(df.run == run) & df.kind.isin([kind, "all"])].set_index(["stage", "stratum"])
        share = d.loc["refined"]
        cells = [f"{int(share.loc['all', 'n_neurons']):,} neurons"]
        cells += [percent(share.loc[c, "share"]) if c in share.index else "–" for c in order]
        lines.append(f"| **{run}** | share of labeled neurons | " + " | ".join(cells) + " |")
        for stage, name in STAGES.items():
            s = d.loc[stage]
            cells = [
                f"{100 * s.loc[c, 'acc']:.1f} / {s.loc[c, 'macro_f1']:.2f}" if c in s.index else "–"
                for c in cols
            ]
            lines.append(f"| | {name} | " + " | ".join(cells) + " |")
    return lines


def figure(t: pd.DataFrame, out) -> None:
    """Per-type F1 against FAFB type size, by region; vertical lines: the size bins."""
    fig, ax = plt.subplots(figsize=(8.4, 3.8), facecolor="white")
    for region, (color, marker) in REGION_STYLE.items():
        g = t[t.region == region]
        if g.empty:
            continue
        label = f"{region}\n{len(g)} types, {percent(g.n.sum() / t.n.sum())} of neurons"
        ax.scatter(
            g["size"],
            g.f1,
            s=24,
            c=color,
            marker=marker,
            edgecolors="white",
            linewidths=0.6,
            label=label,
            zorder=3,
        )
    for lo, _ in SIZE_BINS[1:]:
        ax.axvline(lo, color=AXIS, lw=1, ls=(0, (3, 3)), zorder=1)
    ax.set_xscale("log")
    edges = [lo for lo, _ in SIZE_BINS] + [5000]
    ax.set_xticks(edges, [f"{e:,}" for e in edges])
    ax.minorticks_off()
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("neurons of the type in FAFB (log scale)", color=INK2)
    ax.set_ylabel("F1 on MCNS", color=INK2)
    ax.set_title("FAFB → MCNS after NTAC, per cell type", loc="left", color=INK, fontsize=11)
    ax.grid(axis="y", color=GRID, lw=1)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, length=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    leg = ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
        frameon=False,
        fontsize=8,
        handletextpad=0.3,
        labelspacing=0.9,
    )
    for text in leg.get_texts():
        text.set_color(INK2)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"type_f1.{ext}", dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="stratified", help="results subdirectory")
    p.add_argument("--figures", type=Path, default=REPO_ROOT / "paper" / "figures")
    args = p.parse_args()
    types = fafb_types()
    all_strata, all_types = [], []
    for run, (directory, source, target) in RUNS.items():
        pred = labelings(directory, source, target)
        for stage in STAGES:
            t = per_type(pred, stage, types)
            all_types.append(t.assign(run=run, stage=stage))
            all_strata.append(strata(t).assign(run=run, stage=stage))
    s = pd.concat(all_strata, ignore_index=True)
    t = pd.concat(all_types, ignore_index=True)
    out = RESULTS / args.out
    out.mkdir(parents=True, exist_ok=True)
    first = ["run", "stage"]
    s[first + [c for c in s.columns if c not in first]].to_csv(out / "strata.csv", index=False)
    t[first + [c for c in t.columns if c not in first]].to_csv(out / "types.csv", index=False)
    regions = list(REGION_STYLE)
    bins = [size_bin(lo) for lo, _ in SIZE_BINS]
    lines = ["Cell types by region of the type (FAFB super class), accuracy % / macro-F1.", ""]
    lines += markdown(s, "region", regions)
    lines += ["", "Cell types by FAFB type size (neurons), accuracy % / macro-F1.", ""]
    lines += markdown(s, "type size", bins)
    lines += ["", "Points of the overall accuracy contributed by each region (+ NTAC).", ""]
    lines += ["| Run | all | " + " | ".join(regions) + " |", "|---" * (len(regions) + 2) + "|"]
    for run in RUNS:
        d = s[(s.run == run) & (s.stage == "refined")].set_index("stratum").points
        cells = [f"{100 * d.get(c, 0):.1f}" for c in ["all"] + regions]
        lines.append(f"| {run} | " + " | ".join(cells) + " |")
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    plt.rcParams.update({"font.size": 10, "font.family": "sans-serif"})
    args.figures.mkdir(parents=True, exist_ok=True)
    figure(t[(t.run == "FAFB → MCNS") & (t.stage == "refined")], args.figures)
    print(f"[stratify] wrote {out / 'summary.md'} and {args.figures / 'type_f1.png'}")


if __name__ == "__main__":
    main()
