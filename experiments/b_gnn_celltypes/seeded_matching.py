"""Seeded graph matching for the rare types (docs/preregistration_confirmatory.md).

The exploratory morphology baseline (morphology_baseline.py) found that on BANC's rare types the
matching's type where it assigned a FAFB neuron, NBLAST's elsewhere, beats NBLAST alone by 6.0
points. This script runs the pre-registered follow-up: the same FAQ matching as
graph_matching_mcns.py (barycenter start, A2-ii candidates `nblast_union`, blocks by hemisphere,
and for the cord also the passing block), with confident NBLAST matches fixed as seeds.

Seeds (label-free, per matching block): a candidate whose NBLAST top hit (BANC metadata,
`fafb_nblast_match` / `manc_nblast_match`) is a source neuron of a 2-4-neuron type in the same
block, and that source neuron is the NBLAST top hit of no other candidate in the block. Seed pairs
stay fixed; Frank-Wolfe runs on the remaining rows and columns. Every linear step, including the
first and the final projection, is an exact assignment on the free part (SeededLMO).

Stages:
- seeds: count the seeds per block (no labels are read). Writes seeds_<target>.json.
- match: the seeded matching. Writes matches_<target>.parquet (candidates: root_id, label,
  stage1, pred, fafb_root_id = assigned source id or -1, seed) and info_<target>.json.
- score: labelings and the pre-registered deltas for every target with a saved matching. Writes
  metrics.csv, delta.csv and decision.json.

Labelings (morphology_baseline.labelings, on the same neurons): NBLAST alone; for the unseeded
A2-ii matching (results/morphology_baseline/gm_banc*) and the seeded one, two-stage and
"two-stage matched, else NBLAST" (the matching's type where it assigned a source neuron, NBLAST
elsewhere, stage 1 where neither).

Outputs in results/seeded_matching/.

Usage: python experiments/b_gnn_celltypes/seeded_matching.py --stage seeds|match|score
           [--targets banc banc_cord]
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import pandas as pd
import stratify
from graph_matching_mcns import (
    STAGE1_MANC,
    STAGE1_ROOTS,
    _lap,
    as32,
    block_adjacency,
    faq_blocks,
    labels,
    rare,
    score,
    setup,
    side_blocks,
)
from morphology_baseline import TARGETS, banc_meta, bootstrap_delta, labelings, nblast_types

from flybrain.utils import REPO_ROOT

RAW = REPO_ROOT / "data" / "raw"
OUT = REPO_ROOT / "results" / "seeded_matching"
CANDIDATES = "file:nblast_union"  # A2-ii: union candidates plus NBLAST rare-type neurons
UNSEEDED = "nblast_union"  # the tag of the unseeded matching on the same candidates
N_BOOT = 10_000
BOOT_SEED = 0
MIN_DELTA = 0.03  # docs/finalize_plan.md A4
PRIMARY_TARGET = "banc"
PRIMARY = "two-stage matched, else NBLAST [seeded]"


def load_meta() -> pd.DataFrame:
    """morphology_baseline.banc_meta plus BANC's reviewed matches (fafb_match, manc_match)."""
    m = banc_meta()
    extra = pd.read_feather(
        RAW / "banc" / "banc_888_meta.feather", columns=["banc_888_id", "fafb_match", "manc_match"]
    )
    extra = extra.dropna(subset=["banc_888_id"])
    extra["banc_888_id"] = extra["banc_888_id"].astype(np.int64)
    return m.join(extra.drop_duplicates("banc_888_id").set_index("banc_888_id"))


def nblast_ids(meta: pd.DataFrame, column: str, root_ids: np.ndarray) -> np.ndarray:
    """Per target neuron: the NBLAST top hit's source id (int64), -1 where there is none. Ids
    are parsed from strings: 18-digit FlyWire ids do not survive a float."""
    raw = meta[column].reindex(root_ids).astype("string").str.strip()
    raw = raw.str.replace(r"\.0$", "", regex=True)
    ok = raw.str.fullmatch(r"\d+").fillna(False).to_numpy(bool)
    out = np.full(len(root_ids), -1, dtype=np.int64)
    out[ok] = raw[ok].astype(str).map(int).to_numpy(dtype=np.int64)
    return out


def find_seeds(f_block: np.ndarray, m_block: np.ndarray, hit: np.ndarray, source_ids) -> tuple:
    """Seed pairs of one block, as (row, col) positions in the block: candidate m_block[col] whose
    NBLAST hit (hit[target index], a source id) is source neuron f_block[row], where no other
    candidate of the block has the same hit."""
    row_of = pd.Series(np.arange(len(f_block)), index=np.asarray(source_ids)[f_block])
    h = pd.Series(hit[m_block])
    in_block = h.isin(row_of.index).to_numpy()
    unique = ~h.duplicated(keep=False).to_numpy()
    cols = np.nonzero(in_block & unique)[0]
    rows = row_of.loc[h.to_numpy()[cols]].to_numpy()
    return rows.astype(np.int64), cols.astype(np.int64)


class SeededLMO:
    """Linear step of Frank-Wolfe over the assignments that contain the seed pairs: an exact
    assignment on the free rows and columns of the gradient, plus the seeds."""

    def __init__(self, rows: np.ndarray, cols: np.ndarray, n: int):
        self.rows, self.cols = np.asarray(rows), np.asarray(cols)
        self.free_r = np.setdiff1d(np.arange(n), self.rows)
        self.free_c = np.setdiff1d(np.arange(n), self.cols)

    def __call__(self, g: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        r, c = _lap(g[np.ix_(self.free_r, self.free_c)])
        return (
            np.concatenate([self.free_r[r], self.rows]),
            np.concatenate([self.free_c[c], self.cols]),
        )

    def start(self) -> np.ndarray:
        """Seeds at 1, the free part at its barycenter (doubly stochastic)."""
        n = len(self.free_r) + len(self.rows)
        p = np.zeros((n, n), dtype=np.float32)
        p[np.ix_(self.free_r, self.free_c)] = 1.0 / max(len(self.free_r), 1)
        p[self.rows, self.cols] = 1.0
        return p


def stage_args(target: str) -> argparse.Namespace:
    source = TARGETS[target]["source"]
    return argparse.Namespace(
        target=target,
        source=source,
        stage1=TARGETS[target]["stage1"],
        candidates=CANDIDATES,
        variant=None,
        out_dir=OUT,
        passing_block=target in STAGE1_MANC,
    )


def blocks_and_seeds(s, target: str, meta: pd.DataFrame) -> tuple:
    """Matching blocks (as graph_matching_mcns.match) and the seeds of each block."""
    args = stage_args(target)
    flabel = labels(s.fafb)
    frare = np.nonzero(rare(flabel, s.sizes))[0]
    f_blocks = side_blocks(frare, s.fafb.side, getattr(s.fafb, "passing", None), args.passing_block)
    m_blocks = side_blocks(s.cand, s.tgt.side, getattr(s.tgt, "passing", None), args.passing_block)
    hit = nblast_ids(meta, TARGETS[target]["nblast"], s.tgt.node_ids.numpy())
    source_ids = s.fafb.node_ids.numpy()
    seeds = [find_seeds(fb, mb, hit, source_ids) for fb, mb in zip(f_blocks, m_blocks, strict=True)]
    return flabel, f_blocks, m_blocks, hit, seeds


def run_seeds(target: str, meta: pd.DataFrame) -> dict:
    s = setup(stage_args(target))
    _, f_blocks, m_blocks, hit, seeds = blocks_and_seeds(s, target, meta)
    info = {
        "target": target,
        "candidates": int(len(s.cand)),
        "candidates_with_nblast_hit": int((hit[s.cand] >= 0).sum()),
        "source_rare_per_block": [len(b) for b in f_blocks],
        "candidates_per_block": [len(b) for b in m_blocks],
        "seeds_per_block": [len(r) for r, _ in seeds],
    }
    info["seeds"] = sum(info["seeds_per_block"])
    print(info, flush=True)
    (OUT / f"seeds_{target}.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


def run_match(target: str, meta: pd.DataFrame, iters: int) -> None:
    s = setup(stage_args(target))
    flabel, f_blocks, m_blocks, _, seeds = blocks_and_seeds(s, target, meta)
    n = max(len(b) for b in f_blocks + m_blocks)
    print(f"blocks {[len(b) for b in f_blocks]} / {[len(b) for b in m_blocks]}, n = {n}")
    print(f"seeds {[len(r) for r, _ in seeds]}", flush=True)
    F = as32(block_adjacency(s.fafb, f_blocks, n))
    M = as32(block_adjacency(s.tgt, m_blocks, n))
    lmos = [SeededLMO(r, c, n) for r, c in seeds]
    t = time.time()
    matches = faq_blocks(F, M, [lmo.start() for lmo in lmos], iters, lmos=lmos)
    seconds = time.time() - t

    pred = s.stage1.pred.to_numpy(dtype=object).copy()
    assigned = np.full(len(pred), -1, dtype=np.int64)
    is_seed = np.zeros(len(pred), dtype=bool)
    source_ids = s.fafb.node_ids.numpy()
    for fb, mb, mt, (_, cols) in zip(f_blocks, m_blocks, matches, seeds, strict=True):
        a = mt[: len(mb)]
        real = a < len(fb)
        pred[mb[real]] = flabel[fb[a[real]]]
        assigned[mb[real]] = source_ids[fb[a[real]]]
        is_seed[mb[cols]] = True
    out = pd.DataFrame(
        {
            "root_id": s.tgt.node_ids.numpy(),
            "label": s.stage1.label.to_numpy(dtype=object),
            "stage1": s.stage1.pred.to_numpy(dtype=object),
            "pred": pred,
            "fafb_root_id": assigned,
            "seed": is_seed,
        }
    ).iloc[s.cand]
    out.to_parquet(OUT / f"matches_{target}.parquet")
    info = {
        "target": target,
        "candidates": UNSEEDED,
        "iters": iters,
        "n": n,
        "seeds_per_block": [len(r) for r, _ in seeds],
        "assigned": int((out.fafb_root_id >= 0).sum()),
        "seconds": round(seconds),
    }
    (OUT / f"info_{target}.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(info, flush=True)


def decide(delta: float, lo: float, hi: float) -> str:
    """The pre-registered rule (docs/preregistration_confirmatory.md)."""
    if delta >= MIN_DELTA and lo > 0:
        return "confirmed"
    if hi < 0:
        return "failed: NBLAST alone is better"
    if lo > 0:
        return "not confirmed: wiring adds less than 3 points"
    return "inconclusive: the interval includes 0"


def run_score(target: str, meta: pd.DataFrame) -> tuple[list, list]:
    cfg = TARGETS[target]
    d = STAGE1_ROOTS[cfg["source"]] / cfg["stage1"] / "cell_type"
    stage1 = pd.read_parquet(d / "predictions.parquet")
    types = stratify.source_types(cfg["source"])
    sizes = types["size"]
    m = meta.reindex(stage1.root_id.to_numpy())
    nb = nblast_types(m, cfg["nblast"], cfg["source_table"]).to_numpy(dtype=object)
    has = pd.notna(nb)
    seeded = pd.read_parquet(OUT / f"matches_{target}.parquet")
    unseeded = pd.read_parquet(REPO_ROOT / "results" / cfg["matchings"][UNSEEDED])
    assert (seeded.root_id.to_numpy() == unseeded.root_id.to_numpy()).all(), "other candidates"
    labs = {
        "stage 1 (label-free GNN)": stage1.pred.to_numpy(dtype=object),
        "NBLAST": np.where(has, nb, None),
    }
    labs.update(labelings(stage1, unseeded, nb, "unseeded"))
    labs.update(labelings(stage1, seeded, nb, "seeded"))
    labs = {k: v for k, v in labs.items() if not k.startswith("NBLAST, else")}

    ref_a = stage1.label.to_numpy(dtype=object)
    ref_b = m[cfg["ref_b"]].astype(object)
    if target == "banc":
        ref_b = ref_b.where(m.region.to_numpy() == "central_brain")
    ref_b = ref_b.where(ref_b.isin(set(types.index))).to_numpy(dtype=object)
    pos = pd.Series(np.arange(len(stage1)), index=stage1.root_id)
    seed = np.zeros(len(stage1), bool)
    seed[pos.loc[seeded.root_id[seeded.seed]].to_numpy()] = True
    # BANC's reviewed match (fafb_match / manc_match) against the NBLAST top hit
    reviewed = nblast_ids(meta, cfg["nblast"].replace("_nblast", ""), stage1.root_id.to_numpy())
    top = nblast_ids(meta, cfg["nblast"], stage1.root_id.to_numpy())
    kept = (top >= 0) & (reviewed == top)
    overridden = (top >= 0) & (reviewed >= 0) & (reviewed != top)

    rows, deltas = [], []
    for ref_name, truth in (("ref A", ref_a), ("ref B", ref_b)):
        r = pd.notna(truth) & rare(truth, sizes)
        subsets = {"2-4": r}
        if ref_name == "ref A":
            subsets.update(
                {
                    "2-4, seeds": r & seed,
                    "2-4, not seeds": r & ~seed,
                    "2-4, NBLAST hit kept by review": r & kept,
                    "2-4, NBLAST hit overridden by review": r & overridden,
                    "all labeled": pd.notna(truth),
                }
            )
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
        contrasts = [
            (
                "primary" if target == PRIMARY_TARGET and ref_name == "ref A" else "secondary",
                PRIMARY,
                "NBLAST",
                "2-4",
            ),
            ("secondary", PRIMARY, "two-stage matched, else NBLAST [unseeded]", "2-4"),
            ("secondary", "two-stage [seeded]", "two-stage [unseeded]", "2-4"),
        ]
        if ref_name == "ref A":
            contrasts += [
                ("descriptive", PRIMARY, "NBLAST", "2-4, not seeds"),
                ("descriptive", PRIMARY, "NBLAST", "2-4, NBLAST hit kept by review"),
                ("descriptive", PRIMARY, "NBLAST", "2-4, NBLAST hit overridden by review"),
            ]
        for role, a, b, subset in contrasts:
            k = subsets[subset]
            delta, lo, hi = bootstrap_delta(truth, labs[a], labs[b], k, N_BOOT, BOOT_SEED)
            deltas.append(
                {
                    "target": target,
                    "reference": ref_name,
                    "role": role,
                    "comparison": f"{a} - {b}",
                    "subset": subset,
                    "n": int(k.sum()),
                    "types": int(pd.Series(truth[k]).nunique()),
                    "delta": delta,
                    "ci_low": lo,
                    "ci_high": hi,
                    # the rule is defined for contrasts against NBLAST alone
                    "outcome": decide(delta, lo, hi)
                    if role != "descriptive" and b == "NBLAST"
                    else "",
                }
            )
    return rows, deltas


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--stage", choices=["seeds", "match", "score"], required=True)
    ap.add_argument("--targets", nargs="+", default=list(TARGETS), choices=list(TARGETS))
    ap.add_argument("--iters", type=int, default=30)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    meta = load_meta()
    if args.stage == "seeds":
        for t in args.targets:
            run_seeds(t, meta)
        return
    if args.stage == "match":
        for t in args.targets:
            run_match(t, meta, args.iters)
        return
    rows, deltas = [], []
    for t in args.targets:
        if not (OUT / f"matches_{t}.parquet").exists():
            print(f"{t}: no seeded matching, skipped", flush=True)
            continue
        r, d = run_score(t, meta)
        rows += r
        deltas += d
    pd.DataFrame(rows).to_csv(OUT / "metrics.csv", index=False, lineterminator="\n")
    dl = pd.DataFrame(deltas)
    dl.to_csv(OUT / "delta.csv", index=False, lineterminator="\n")
    prim = dl[dl.role == "primary"]
    if len(prim):
        p = prim.iloc[0]
        decision = {k: p[k] for k in ("comparison", "subset", "n", "types")}
        decision.update(
            delta=float(p.delta),
            ci_low=float(p.ci_low),
            ci_high=float(p.ci_high),
            outcome=p.outcome,
            n_boot=N_BOOT,
            boot_seed=BOOT_SEED,
        )
        decision = {k: (int(v) if isinstance(v, np.integer) else v) for k, v in decision.items()}
        (OUT / "decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
        print(json.dumps(decision, indent=2))
    with pd.option_context("display.width", 200, "display.max_colwidth", 70):
        print(dl.round(4).to_string())


if __name__ == "__main__":
    main()
