"""Label-free rules for choosing among candidate labelings of an unlabeled connectome.

Every rule sees only label-free properties of the candidates (the columns label_free.py writes
to candidates.csv): type consistency on the scoring graph, mean agreement between the seeds'
labelings, the Jensen-Shannon divergence of the predicted type sizes from the reference
(size_js), and the share of the largest predicted type. "consistency" is the rule of the
FAFB <-> MCNS and pre-registered BANC runs; "size_agreement" was chosen on the development
sandbox (thinned FAFB / MCNS copies with known labels, selection_rules.py) because its worst
case was smallest.
"""

from __future__ import annotations

import pandas as pd

RULES = (
    "consistency",
    "consistency_guarded",
    "size_match",
    "agreement",
    "agreement_guarded",
    "rank_mean",
    "size_agreement",
    "rank_all",
)


def scoring_column(c: pd.DataFrame) -> str:
    cols = [k for k in c.columns if k.startswith("consistency_")]
    return "consistency_raw" if "consistency_raw" in cols else cols[0]


def guard(c: pd.DataFrame) -> pd.DataFrame:
    """Candidates whose largest predicted type holds at most twice the share of the largest
    reference type (excludes collapsed labelings); all candidates if none is left."""
    ok = c[c.top_share <= 2 * c.ref_top_share]
    return ok if len(ok) else c


def choose_all(c: pd.DataFrame, key: str | None = None) -> dict[str, int]:
    """Index of the candidate each rule chooses (agreement rules need several seeds)."""
    key = key or scoring_column(c)
    g = guard(c)
    chosen = {
        "consistency": c[key].idxmax(),
        "consistency_guarded": g[key].idxmax(),
        "size_match": c.size_js.idxmin(),
    }
    if c.pairwise_agreement.notna().any():
        agree_rank = c.pairwise_agreement.rank(ascending=False)
        g_ranks = g[key].rank(ascending=False) + g.pairwise_agreement.rank(ascending=False)
        two = c.size_js.rank() + agree_rank
        chosen["agreement"] = c.pairwise_agreement.idxmax()
        chosen["agreement_guarded"] = g.pairwise_agreement.idxmax()
        chosen["rank_mean"] = g_ranks.idxmin()
        chosen["size_agreement"] = two.idxmin()
        chosen["rank_all"] = (two + c[key].rank(ascending=False)).idxmin()
    return chosen


def choose(c: pd.DataFrame, rule: str, key: str | None = None) -> int:
    """Index of the candidate `rule` chooses; size_agreement falls back to size_match with a
    single seed."""
    if rule not in RULES:
        raise ValueError(f"rule must be one of {RULES}, got {rule!r}")
    chosen = choose_all(c, key)
    if rule not in chosen:
        rule = "size_match" if rule == "size_agreement" else "consistency"
    return chosen[rule]
