"""Label-free selection rules on hand-made candidate tables."""

from __future__ import annotations

import pandas as pd
import pytest

from flybrain.selection import RULES, choose, choose_all


def _candidates() -> pd.DataFrame:
    # 0: collapsed (one type holds 77 %), self-consistent but wrong; 1: well aligned;
    # 2: misaligned, sizes off
    return pd.DataFrame(
        {
            "consistency_raw": [0.65, 0.61, 0.55],
            "pairwise_agreement": [0.40, 0.65, 0.50],
            "size_js": [0.78, 0.01, 0.30],
            "top_share": [0.77, 0.02, 0.05],
            "ref_top_share": [0.06, 0.06, 0.06],
            "acc": [0.014, 0.58, 0.30],
        }
    )


def test_rules_on_a_collapsed_candidate():
    c = _candidates()
    chosen = choose_all(c)
    assert set(chosen) == set(RULES)
    assert chosen["consistency"] == 0  # the BANC failure
    assert chosen["consistency_guarded"] == 1 and chosen["size_agreement"] == 1
    assert choose(c, "size_agreement") == 1


def test_single_seed_falls_back_and_unknown_rule_fails():
    c = _candidates().assign(pairwise_agreement=float("nan"))
    assert set(choose_all(c)) == {"consistency", "consistency_guarded", "size_match"}
    assert choose(c, "size_agreement") == choose(c, "size_match") == 1
    with pytest.raises(ValueError, match="rule"):
        choose(c, "nope")
