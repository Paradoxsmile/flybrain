"""95 % intervals for the pre-registered rare-type tests (paper Tables 6 and 8).

For each test, the accuracy on the 2-4 stratum (reference A, and B where it exists) of stage 1,
the two-stage labelings and the oracle-candidate diagnostic, and each labeling's gain over stage
1, with a 95 % interval from resampling the true types (2,000 draws; neurons of one type move
together, since they share their errors). Labelings are rebuilt from the saved stage-1
predictions and matchings, so nothing is rerun; the point estimates must equal the reported ones.

Writes results/bootstrap_ci/{metrics.csv, SUMMARY.md}.

Usage: python experiments/b_gnn_celltypes/bootstrap_ci.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import stratify
from graph_matching_mcns import STAGE1_ROOTS, rare

from flybrain.utils import REPO_ROOT

OUT = REPO_ROOT / "results" / "bootstrap_ci"
N_BOOT = 2000

# test: source, stage-1 run, {labeling: saved matching}
TESTS = {
    "BANC brain (Table 6)": (
        "fafb",
        "label_free_banc_size_agreement",
        {
            "two-stage, gnn": "graph_matching_banc/matches_gnn_gm_bary.parquet",
            "oracle candidates": "graph_matching_banc/matches_oracle_gm_bary.parquet",
        },
    ),
    "BANC cord (Table 8)": (
        "manc",
        "label_free_banc_cord_size_agreement",
        {
            "two-stage, gnn": "graph_matching_banc_cord/matches_gnn_gm_bary_pass.parquet",
            "two-stage, union": "graph_matching_banc_cord/matches_union_gm_bary_pass.parquet",
            "oracle candidates": "graph_matching_banc_cord/matches_oracle_gm_bary_pass.parquet",
        },
    ),
    "MCNS cord (Table 8)": (
        "manc",
        "label_free_mcns_cord_size_agreement",
        {
            "two-stage, gnn": "graph_matching_mcns_cord/matches_gnn_gm_bary_pass.parquet",
            "two-stage, union": "graph_matching_mcns_cord/matches_union_gm_bary_pass.parquet",
            "oracle candidates": "graph_matching_mcns_cord/matches_oracle_gm_bary_pass.parquet",
        },
    ),
}


def two_stage(stage1: pd.DataFrame, matches: pd.DataFrame) -> np.ndarray:
    pred = stage1.pred.to_numpy(dtype=object).copy()
    pos = pd.Series(np.arange(len(stage1)), index=stage1.root_id)
    pred[pos.loc[matches.root_id].to_numpy()] = matches.pred.to_numpy(dtype=object)
    return pred


def per_type(truth: np.ndarray, preds: dict, subset: np.ndarray) -> pd.DataFrame:
    """Per true type: neurons and correct neurons of each labeling."""
    df = pd.DataFrame({"t": truth[subset]})
    for name, p in preds.items():
        df[name] = p[subset] == truth[subset]
    return df.groupby("t").agg(n=("t", "size"), **{k: (k, "sum") for k in preds})


def intervals(g: pd.DataFrame, names: list, rng) -> list:
    idx = rng.integers(0, len(g), size=(N_BOOT, len(g)))
    n = g.n.to_numpy(float)
    nb = n[idx].sum(axis=1)
    base = g["stage 1"].to_numpy(float)
    rows = []
    for name in names:
        c = g[name].to_numpy(float)
        acc = c[idx].sum(axis=1) / nb
        gain = (c - base)[idx].sum(axis=1) / nb
        rows.append(
            {
                "labeling": name,
                "acc": c.sum() / n.sum(),
                "acc_lo": np.percentile(acc, 2.5),
                "acc_hi": np.percentile(acc, 97.5),
                "gain": (c - base).sum() / n.sum(),
                "gain_lo": np.percentile(gain, 2.5),
                "gain_hi": np.percentile(gain, 97.5),
            }
        )
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    rows = []
    for test, (source, run, matchings) in TESTS.items():
        d = STAGE1_ROOTS[source] / run / "cell_type"
        stage1 = pd.read_parquet(d / "predictions.parquet")
        sizes = stratify.source_types(source)["size"]
        preds = {"stage 1": stage1.pred.to_numpy(dtype=object)}
        for name, path in matchings.items():
            preds[name] = two_stage(stage1, pd.read_parquet(REPO_ROOT / "results" / path))
        refs = {"ref A": stage1.label.to_numpy(dtype=object)}
        for name, truth in refs.items():
            k = pd.notna(truth) & rare(truth, sizes)
            g = per_type(truth, preds, k)
            for r in intervals(g, list(preds), rng):
                rows.append(
                    {"test": test, "reference": name, "neurons": int(k.sum()), "types": len(g), **r}
                )
    ci = pd.DataFrame(rows)
    ci.to_csv(OUT / "metrics.csv", index=False)

    def pct(x):
        return f"{100 * x:.1f}"

    lines = [
        "# 95 % intervals for the pre-registered rare-type tests",
        "",
        "Accuracy % on the 2-4 stratum (reference A) and gain over stage 1, points; 95 % "
        f"intervals from resampling true types ({N_BOOT} draws, seed 0). "
        "Command: `python experiments/b_gnn_celltypes/bootstrap_ci.py`.",
        "",
        "| Test | Labeling | Accuracy [95 %] | Gain over stage 1 [95 %] |",
        "|---|---|---|---|",
    ]
    for r in ci.itertuples():
        gain = (
            "–"
            if r.labeling == "stage 1"
            else (f"{pct(r.gain)} [{pct(r.gain_lo)}, {pct(r.gain_hi)}]")
        )
        lines.append(
            f"| {r.test} ({r.neurons:,} neurons, {r.types:,} types) | {r.labeling} | "
            f"{pct(r.acc)} [{pct(r.acc_lo)}, {pct(r.acc_hi)}] | {gain} |"
        )
    (OUT / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
