"""Morphology baseline for the rare-type tests (docs/finalize_plan.md, A1-A2; exploratory).

BANC's metadata holds each neuron's NBLAST top match before review: `fafb_nblast_match` (a FAFB
root id) and `manc_nblast_match` (a MANC body id). Typing a BANC neuron by the type of that match
is the morphology baseline; it uses no BANC label and no connectivity.

Targets: the BANC brain (source FAFB) and the BANC nerve cord (source MANC). Labelings, all on the
same neurons and scored like graph_matching_mcns.py (accuracy; macro-F1 over the true types):
- stage 1: the label-free GNN pipeline (predictions.parquet, `pred`);
- NBLAST: the NBLAST match's source type, no prediction where there is no match;
- NBLAST, else stage 1;
- per saved matching [tag] (TARGETS; `gnn` and `union` candidates, and `nblast_union` of A2-ii):
  two-stage (stage 1 with the matched candidates' types), NBLAST, else two-stage (A2-i), and
  two-stage matched, else NBLAST (the matching's type where it assigned a source neuron, NBLAST
  elsewhere, stage 1 where neither).

agreement.csv: on the ref-A 2-4 stratum, accuracy where NBLAST and the first matching agree or
disagree, and each combination's accuracy minus NBLAST alone with a 95 % interval from resampling
true types (the delta of docs/finalize_plan.md, A4).

References: A is the stage-1 `label`. B is fafb_cell_type (brain, central brain only; built from
NBLAST with review, so circular for the NBLAST rows) or BANC's curated type (cord). Both are
restricted to the source label space (types with >= 2 source neurons).

--candidates-out also writes A2-ii's candidate set: the stage-1 `union` candidates plus the
neurons whose NBLAST type is a 2-4-neuron source type, as candidates_nblast_union.parquet in the
stage-1 directory, for graph_matching_mcns.py --candidates file:nblast_union.

Writes results/morphology_baseline/{metrics.csv, agreement.csv}.

Usage: python experiments/b_gnn_celltypes/morphology_baseline.py [--candidates-out]
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import stratify
from graph_matching_mcns import STAGE1_ROOTS, rare, score

from flybrain.utils import REPO_ROOT

RAW = REPO_ROOT / "data" / "raw"
PROCESSED = REPO_ROOT / "data" / "processed"
OUT = REPO_ROOT / "results" / "morphology_baseline"

# target: source, stage-1 run, saved matching (the reported primary two-stage run), NBLAST column,
# reference B column of the BANC metadata
TARGETS = {
    "banc": dict(
        source="fafb",
        stage1="label_free_banc_size_agreement",
        matchings={
            "gnn": "graph_matching_banc/matches_gnn_gm_bary.parquet",
            "union": "morphology_baseline/gm_banc/matches_union_gm_bary.parquet",
            "nblast_union": "morphology_baseline/gm_banc/matches_nblast_union_gm_bary.parquet",
        },
        nblast="fafb_nblast_match",
        ref_b="fafb_cell_type",
        source_table="neurons.parquet",
    ),
    "banc_cord": dict(
        source="manc",
        stage1="label_free_banc_cord_size_agreement",
        matchings={
            "union": "graph_matching_banc_cord/matches_union_gm_bary_pass.parquet",
            "nblast_union": (
                "morphology_baseline/gm_banc_cord/matches_nblast_union_gm_bary_pass.parquet"
            ),
        },
        nblast="manc_nblast_match",
        ref_b="cell_type",
        source_table="manc_neurons.parquet",
    ),
}


def banc_meta() -> pd.DataFrame:
    cols = ["banc_888_id", "region", "cell_type", "fafb_cell_type"]
    cols += ["fafb_nblast_match", "manc_nblast_match"]
    m = pd.read_feather(RAW / "banc" / "banc_888_meta.feather", columns=cols)
    m = m.dropna(subset=["banc_888_id"])
    m["banc_888_id"] = m["banc_888_id"].astype(np.int64)
    return m.drop_duplicates("banc_888_id").set_index("banc_888_id")


def nblast_types(meta: pd.DataFrame, column: str, source_table: str) -> pd.Series:
    """Per BANC neuron: the source type of its NBLAST match (missing: no match or untyped)."""
    src = pd.read_parquet(PROCESSED / source_table, columns=["root_id", "cell_type"])
    by_id = src.dropna(subset=["cell_type"]).set_index("root_id").cell_type
    # parse the ids as integers directly: FlyWire root ids have 18 digits, beyond float64
    raw = meta[column].dropna().astype(str).str.strip()
    raw = raw[raw.str.fullmatch(r"\d+")]
    return raw.astype(np.int64).map(by_id).reindex(meta.index)


def labelings(stage1: pd.DataFrame, matches: pd.DataFrame, nb: np.ndarray, tag: str) -> dict:
    s1 = stage1.pred.to_numpy(dtype=object)
    pos = pd.Series(np.arange(len(stage1)), index=stage1.root_id)
    two = s1.copy()
    k = pos.loc[matches.root_id].to_numpy()
    two[k] = matches.pred.to_numpy(dtype=object)
    assigned = np.zeros(len(s1), bool)
    if "fafb_root_id" in matches:
        assigned[k] = matches.fafb_root_id.to_numpy() >= 0
    else:  # older matchings (BANC brain): assigned approximated as "matching changed the type"
        assigned[k] = (matches.pred != matches.stage1).to_numpy()
    has = pd.notna(nb)
    return {
        f"two-stage [{tag}]": two,
        f"NBLAST, else two-stage [{tag}]": np.where(has, nb, two),
        f"two-stage matched, else NBLAST [{tag}]": np.where(assigned, two, np.where(has, nb, s1)),
    }


def bootstrap_delta(truth, a, b, subset, n_boot: int = 2000, seed: int = 0) -> tuple:
    """Accuracy of a minus b on subset, with a 95 % interval from resampling true types."""
    df = pd.DataFrame(
        {
            "t": truth[subset],
            "a": np.asarray(a, dtype=object)[subset] == truth[subset],
            "b": np.asarray(b, dtype=object)[subset] == truth[subset],
        }
    )
    g = df.groupby("t").agg(a=("a", "sum"), b=("b", "sum"), n=("a", "size"))
    diff, n = (g.a - g.b).to_numpy(float), g.n.to_numpy(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(g), size=(n_boot, len(g)))
    boot = diff[idx].sum(axis=1) / n[idx].sum(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(diff.sum() / n.sum()), float(lo), float(hi)


def run(target: str, cfg: dict, meta: pd.DataFrame, write_candidates: bool):
    d = STAGE1_ROOTS[cfg["source"]] / cfg["stage1"] / "cell_type"
    stage1 = pd.read_parquet(d / "predictions.parquet")
    types = stratify.source_types(cfg["source"])
    sizes = types["size"]
    space = set(types.index)

    m = meta.reindex(stage1.root_id.to_numpy())
    nb = nblast_types(m, cfg["nblast"], cfg["source_table"]).to_numpy(dtype=object)
    ref_a = stage1.label.to_numpy(dtype=object)
    ref_b = m[cfg["ref_b"]].astype(object)
    if target == "banc":
        ref_b = ref_b.where(m.region.to_numpy() == "central_brain")
    ref_b = ref_b.where(ref_b.isin(space)).to_numpy(dtype=object)
    region = m.region.fillna("").to_numpy()

    rows = []
    has = pd.notna(nb)
    s1 = stage1.pred.to_numpy(dtype=object)
    labs = {
        "stage 1 (label-free GNN)": s1,
        "NBLAST": np.where(has, nb, None),
        "NBLAST, else stage 1": np.where(has, nb, s1),
    }
    for tag, path in cfg["matchings"].items():
        f = REPO_ROOT / "results" / path
        if f.exists():
            labs.update(labelings(stage1, pd.read_parquet(f), nb, tag))
    for ref_name, truth in (("ref A", ref_a), ("ref B", ref_b)):
        r = pd.notna(truth) & rare(truth, sizes)
        subsets = {
            "2-4": r,
            "2-4, with NBLAST match": r & has,
            "2-4, without NBLAST match": r & ~has,
            "all labeled": np.ones(len(truth), bool),
        }
        if target == "banc" and ref_name == "ref A":
            subsets["2-4, central brain"] = r & (region == "central_brain")
        for method, pred in labs.items():
            for subset, k in subsets.items():
                rows.append(
                    {
                        "target": target,
                        "reference": ref_name,
                        "method": method,
                        "subset": subset,
                        **score(truth, np.asarray(pred, dtype=object), k, types),
                    }
                )

    # A2: who is right where NBLAST and two-stage agree or disagree (ref A, 2-4 stratum)
    r = pd.notna(ref_a) & rare(ref_a, sizes) & has
    first = next(iter(cfg["matchings"]))
    two, nbl = labs[f"two-stage [{first}]"], labs["NBLAST"]
    agree = []
    for name, k in (("agree", r & (two == nbl)), ("disagree", r & (two != nbl))):
        agree.append(
            {
                "target": target,
                "cell": name,
                "n": int(k.sum()),
                "two-stage correct": float((two[k] == ref_a[k]).mean()),
                "NBLAST correct": float((nbl[k] == ref_a[k]).mean()),
            }
        )
    in_rare = pd.notna(ref_a) & rare(ref_a, sizes)
    # A4's delta: each combination minus NBLAST alone (ref A, 2-4), 95 % CI over types
    for method in labs:
        if "else NBLAST" in method or "NBLAST, else two-stage" in method:
            d, lo, hi = bootstrap_delta(ref_a, labs[method], labs["NBLAST"], in_rare)
            agree.append(
                {
                    "target": target,
                    "cell": f"delta vs NBLAST: {method}",
                    "n": int(in_rare.sum()),
                    "delta": d,
                    "ci_low": lo,
                    "ci_high": hi,
                }
            )
    print(
        f"{target}: NBLAST match for {has.mean():.1%} of neurons, "
        f"{has[in_rare].mean():.1%} of the ref-A 2-4 stratum",
        flush=True,
    )

    if write_candidates:  # A2-ii: union candidates plus NBLAST rare-type neurons
        cand = rare(stage1.pred, sizes) | rare(stage1.pred_selected, sizes) | rare(nb, sizes)
        recall_u = (rare(stage1.pred, sizes) | rare(stage1.pred_selected, sizes))[in_rare].mean()
        print(
            f"{target}: candidates union {recall_u:.1%} recall -> with NBLAST "
            f"{cand[in_rare].mean():.1%}, {cand.sum()} candidates",
            flush=True,
        )
        pd.DataFrame({"root_id": stage1.root_id[cand]}).to_parquet(
            d / "candidates_nblast_union.parquet"
        )
    return rows, agree


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--targets", nargs="+", default=list(TARGETS), choices=list(TARGETS))
    ap.add_argument("--candidates-out", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    meta = banc_meta()
    rows, agree = [], []
    for t in args.targets:
        r, a = run(t, TARGETS[t], meta, args.candidates_out)
        rows += r
        agree += a
    m = pd.DataFrame(rows)
    m.to_csv(OUT / "metrics.csv", index=False)
    pd.DataFrame(agree).to_csv(OUT / "agreement.csv", index=False)
    show = m[m.subset.isin(["2-4", "all labeled"])]
    table = show.pivot_table(
        index=["target", "method"], columns=["reference", "subset"], values="acc"
    )
    print(table.round(3).to_string())
    print(pd.DataFrame(agree).round(3).to_string())


if __name__ == "__main__":
    main()
