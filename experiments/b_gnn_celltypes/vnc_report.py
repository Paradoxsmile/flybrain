"""Step R of the nerve-cord test (docs/vnc_rare_protocol.md): collect the saved outputs of the
runs into the tables the protocol reports, and check the four expectations.

Reads, per target (banc_cord, mcns_cord):
- results/size_threshold_manc/m{2,10}/label_free_<target>_size_agreement/cell_type/selected.json;
- results/graph_matching_<target>/metrics.csv, info_*.json and matches_*.parquet (runs 3, 4, 5, 7)
  and results/graph_matching_<target>_<variant>/ (run 6);
- results/vnc_rare/logs/*.log (runtime and peak memory, from the peakmem lines).

Writes results/vnc_rare/report.json and prints markdown tables. Labels are used only for scoring.

Usage: python experiments/b_gnn_celltypes/vnc_report.py
"""

from __future__ import annotations

import json
import re
import sys

import pandas as pd

from flybrain.utils import REPO_ROOT

RESULTS = REPO_ROOT / "results"
OUT = RESULTS / "vnc_rare"
TARGETS = {"banc_cord": "b", "mcns_cord": "m"}
STAGE1 = "stage 1 (label-free GNN)"
TT = " + two-tier m10"
RUNS = {  # protocol run -> (directory suffix, candidates, method)
    "3 union (primary)": ("", "union", "gm_bary_pass"),
    "4 gnn (old)": ("", "gnn", "gm_bary_pass"),
    "5 oracle": ("", "oracle", "gm_bary_pass"),
    "7 union, no passing block": ("", "union", "gm_bary"),
}


def subset_names(ref: str) -> dict[str, str]:
    s = "" if ref == "ref A" else " (ref B)"
    return {
        "2-4": f"2-4 ({ref})",
        "2-4 matched": f"2-4, matched candidates{s}",
        "all": f"all labeled{s}",
    }


def row(m: pd.DataFrame, cand: str, method: str, ref: str, subset: str) -> dict:
    r = m[
        (m.candidates == cand) & (m.method == method) & (m.reference == ref) & (m.subset == subset)
    ]
    if len(r) != 1:
        return {}
    r = r.iloc[0]
    return {"n": int(r.n), "acc": float(r.acc), "macro_f1": float(r.macro_f1)}


def logs(target: str) -> dict:
    out = {}
    for f in sorted((OUT / "logs").glob("*.log")):
        if not re.fullmatch(rf"[SM]\d{TARGETS[target]}(_\w+)?", f.stem):
            continue
        hit = re.findall(r"\[peakmem\] exit (\d+), (\d+) s, peak ([\d.]+) GB", f.read_text("utf-8"))
        if hit:
            code, sec, gb = hit[-1]
            out[f.stem] = {"exit": int(code), "seconds": int(sec), "peak_gb": float(gb)}
    return out


def stage1(target: str) -> dict:
    out = {}
    for m in (2, 10):
        d = RESULTS / "size_threshold_manc" / f"m{m}" / f"label_free_{target}_size_agreement"
        s = json.loads((d / "cell_type" / "selected.json").read_text())
        out[f"m{m}"] = {
            k: s.get(k)
            for k in (
                "chosen",
                "choice_by_rule",
                "no_adaptation_acc",
                "selected_acc",
                "selected_macro_f1",
                "best_possible",
                "best_possible_acc",
                "refined_acc",
                "refined_macro_f1",
                "breakdown",
            )
        }
    return out


def candidate_counts(target: str, rare_types: set) -> dict:
    d = RESULTS / f"graph_matching_{target}"
    out = {}
    for cand in ("union", "gnn", "oracle"):
        info = json.loads((d / f"info_{cand}_gm_bary_pass.json").read_text())
        mt = pd.read_parquet(d / f"matches_{cand}_gm_bary_pass.parquet")
        lab = mt["label"]
        is_rare = lab.isin(rare_types)
        out[cand] = {
            "candidates": int(len(mt)),
            "per_block": info["target_candidates"],
            "source_rare_per_block": info["source_rare"],
            "without_side": info.get("candidates_without_side"),
            "rare_type_share": float(is_rare.mean()),
            "other_type_share": float((lab.notna() & ~is_rare).mean()),
            "unlabeled_share": float(lab.isna().mean()),
            "rare_selected": int(is_rare.sum()),
        }
    return out


def report(target: str) -> dict:
    d = RESULTS / f"graph_matching_{target}"
    m = pd.read_csv(d / "metrics.csv")
    refs = ["ref A", "ref B"] if target == "banc_cord" else ["ref A"]
    rows = {}
    for ref in refs:
        names = subset_names(ref)
        rows[ref] = {}
        rows[ref]["stage 1"] = {k: row(m, "union", STAGE1, ref, v) for k, v in names.items()}
        rows[ref]["stage 1" + TT] = {
            k: row(m, "union", STAGE1 + TT, ref, v) for k, v in names.items()
        }
        for name, (_, cand, method) in RUNS.items():
            for suffix in ("", TT):
                rows[ref][name + suffix] = {
                    k: row(m, cand, method + suffix, ref, v) for k, v in names.items()
                }
        for v in ("raw", "density", "scaled"):
            dv = RESULTS / f"graph_matching_{target}_{v}"
            if dv.exists():
                mv = pd.read_csv(dv / "metrics.csv")
                for suffix in ("", TT):
                    rows[ref][f"6 union, variant {v}{suffix}"] = {
                        k: row(mv, "union", f"gm_bary_{v}_pass" + suffix, ref, s)
                        for k, s in names.items()
                    }
    groups = m[
        (m.candidates == "union")
        & m.method.isin([STAGE1, "gm_bary_pass"])
        & m.subset.str.startswith("2-4 (ref A), ")
    ][["method", "subset", "n", "acc", "macro_f1"]]
    bins = m[
        (m.candidates == "union")
        & m.method.isin([STAGE1, STAGE1 + TT, "gm_bary_pass", "gm_bary_pass" + TT])
        & m.subset.isin(["5-9", "10-19", ">= 20"])
    ][["method", "subset", "n", "acc", "macro_f1"]]
    types = pd.read_parquet(RESULTS.parent / "data" / "processed" / "manc_neurons.parquet")
    size = types["cell_type"].value_counts()
    rare_types = set(size[size.between(2, 4)].index)
    counts = candidate_counts(target, rare_types)
    n_rare = rows["ref A"]["stage 1"]["2-4"]["n"]
    for cand in ("union", "gnn"):
        counts[cand]["recall"] = counts[cand]["rare_selected"] / n_rare
    unmatched = {}
    for cand in ("union", "gnn", "oracle"):  # the run's own counts (block candidates only)
        r = m[(m.candidates == cand) & (m.method == "gm_bary_pass")].iloc[0]
        counts[cand]["on_padding"] = int(r.to_dummy)
        unmatched[cand] = int(r.fafb_unmatched)
    return {
        "stage1": stage1(target),
        "rows": rows,
        "groups_2_4": groups.to_dict("records"),
        "bins": bins.to_dict("records"),
        "counts": counts,
        "source_rows_unmatched": unmatched,
        "runs": logs(target),
    }


def expectations(r: dict, target: str) -> dict:
    a = r["rows"]["ref A"]
    final = a["3 union (primary)" + TT]
    old = a["4 gnn (old)" + TT]
    s1 = a["stage 1"]
    oracle = a["5 oracle"]
    e = {}
    d1 = final["2-4"]["acc"] - old["2-4"]["acc"]
    f1 = final["2-4"]["macro_f1"] - old["2-4"]["macro_f1"]
    e["1"] = {"union_minus_gnn": d1, "macro_f1_diff": f1, "met": d1 >= 0.03 and f1 >= -0.01}
    e["2"] = {}
    for ref in r["rows"]:
        g = (
            r["rows"][ref]["3 union (primary)" + TT]["2-4"]["acc"]
            - r["rows"][ref]["stage 1"]["2-4"]["acc"]
        )
        e["2"][ref] = {"final_minus_stage1": g, "met": g >= 0.15}
    gap = oracle["2-4"]["acc"] - s1["2-4"]["acc"]
    share = (final["2-4"]["acc"] - s1["2-4"]["acc"]) / gap if gap > 0 else None
    e["3"] = {
        "oracle": oracle["2-4"]["acc"],
        "share_closed": share,
        "met": share is not None and share >= 0.5,
    }
    no_tt = a["3 union (primary)"]
    da = final["all"]["acc"] - no_tt["all"]["acc"]
    df = final["all"]["macro_f1"] - no_tt["all"]["macro_f1"]
    same = final["2-4"]["acc"] == no_tt["2-4"]["acc"]
    e["4"] = {
        "all_acc_gain": da,
        "all_macro_f1_change": df,
        "2_4_unchanged": same,
        "met": da >= 0.005 and df >= -0.01 and same,
    }
    return e


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    out = {}
    for target in TARGETS:
        if not (RESULTS / f"graph_matching_{target}" / "metrics.csv").exists():
            print(target, "not run yet")
            continue
        r = report(target)
        r["expectations"] = expectations(r, target)
        out[target] = r
        print(f"## {target}")
        for ref, rr in r["rows"].items():
            print(f"\n{ref}: | run | 2-4 acc | 2-4 F1 | matched acc | all acc | all F1 |")
            for name, v in rr.items():
                if v.get("2-4"):
                    print(
                        f"| {name} | {v['2-4']['acc']:.1%} | {v['2-4']['macro_f1']:.3f} | "
                        f"{v['2-4 matched'].get('acc', float('nan')):.1%} | "
                        f"{v['all']['acc']:.1%} | "
                        f"{v['all']['macro_f1']:.3f} |"
                    )
        print(json.dumps(r["expectations"], indent=1))
        print(json.dumps(r["counts"], indent=1))
        print(r["source_rows_unmatched"], r["runs"])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
