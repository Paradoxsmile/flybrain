"""Compare label-free rules for choosing among the candidate labelings of label_free.py.

Each rule sees only the label-free columns of candidates.csv; accuracy is used to score the
choice. Rules:
  consistency          highest type consistency on the scoring graph (the rule used so far)
  consistency_guarded  the same among candidates whose largest predicted type holds at most
                       twice the share of the largest reference type (excludes collapsed ones)
  agreement            highest mean agreement between the seeds' labelings (disagreement
                       between independently trained models tracks their error under shift)
  agreement_guarded    the same with the collapse guard
  rank_mean            best mean rank of consistency and agreement, with the collapse guard
  size_match           predicted type sizes closest to the reference (smallest Jensen-Shannon
                       divergence): a misaligned model cannot produce the expected sizes, even
                       after prior matching

Prints and writes, per run and label: the accuracy of the best candidate (post hoc) and, per
rule, the accuracy of its choice and the gap to the best.

Usage: python experiments/b_gnn_celltypes/selection_rules.py RUN [RUN ...] [--out file.md]
       (RUN: a results subdirectory written by label_free.py)
"""

from __future__ import annotations

import argparse

import pandas as pd

from flybrain.utils import REPO_ROOT


def guard(c: pd.DataFrame) -> pd.DataFrame:
    ok = c[c.top_share <= 2 * c.ref_top_share]
    return ok if len(ok) else c


def scoring_column(c: pd.DataFrame) -> str:
    cols = [k for k in c.columns if k.startswith("consistency_")]
    return "consistency_raw" if "consistency_raw" in cols else cols[0]


def choose(c: pd.DataFrame) -> dict[str, pd.Series]:
    key = scoring_column(c)
    g = guard(c)
    chosen = {"consistency": c.loc[c[key].idxmax()], "consistency_guarded": g.loc[g[key].idxmax()]}
    chosen["size_match"] = c.loc[c.size_js.idxmin()]
    if c.pairwise_agreement.notna().any():  # needs several seeds
        ranks = g[key].rank(ascending=False) + g.pairwise_agreement.rank(ascending=False)
        chosen["agreement"] = c.loc[c.pairwise_agreement.idxmax()]
        chosen["agreement_guarded"] = g.loc[g.pairwise_agreement.idxmax()]
        chosen["rank_mean"] = g.loc[ranks.idxmin()]
    return chosen


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("runs", nargs="+")
    p.add_argument("--out", default=None, help="markdown file under results/")
    args = p.parse_args()
    rows = []
    for run in args.runs:
        for path in sorted((REPO_ROOT / "results" / run).glob("*/candidates.csv")):
            c = pd.read_csv(path)
            if "pairwise_agreement" not in c.columns:
                print(f"skipping {path}: written before the agreement columns")
                continue
            row = {"run": run, "label": path.parent.name, "best": c.acc.max()}
            for rule, chosen in choose(c).items():
                row[rule] = chosen.acc
            rows.append(row)
    df = pd.DataFrame(rows)
    rules = [k for k in df.columns if k not in ("run", "label", "best")]
    lines = ["Accuracy % of the candidate each rule chooses (best: post hoc).", ""]
    header = ["run", "label", "best"] + rules
    lines += ["| " + " | ".join(header) + " |", "|---" * len(header) + "|"]
    for r in df.itertuples(index=False):
        cells = [r.run, r.label] + [f"{100 * getattr(r, k):.1f}" for k in header[2:]]
        lines.append("| " + " | ".join(cells) + " |")
    gaps = {k: 100 * (df.best - df[k]) for k in rules}
    lines += ["", "Gap to the best candidate, points: mean / worst", ""]
    lines += [f"- {k}: {g.mean():.1f} / {g.max():.1f}" for k, g in gaps.items()]
    text = "\n".join(lines)
    print(text)
    if args.out:
        (REPO_ROOT / "results" / args.out).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
