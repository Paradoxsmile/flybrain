"""Coverage of a partial reconstruction, for type-size priors that fit its volume.

Prior matching and the size-based selection rule assume that the target has the source's type
composition. A partial volume breaks this: the hemibrain covers the central brain, but 77 % of
FAFB's labeled neurons belong to optic-lobe types. FAFB records the neuropil of every
connection, so the expected composition of the target is the type distribution of the FAFB
neurons inside its volume: those with at least 10 % of their synapses (inputs and outputs) in
the neuropils it covers. Visual projection neurons thus count, with their optic-lobe arbors cut
off, as in the hemibrain. Only the list of covered neuropils is needed, no target label.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from flybrain.loading import (
    CONNECTIONS_STEM,
    ID_COL,
    POST_COL,
    PRE_COL,
    PROCESSED_DIR,
    RAW_DIR,
    WEIGHT_COL,
    find_raw_file,
    read_table,
)

OPTIC_NEUROPILS = ("ME", "AME", "LO", "LOP", "LA")  # FAFB optic lobe, without the _L / _R suffix
COVERAGES = {"central_brain": OPTIC_NEUROPILS + ("OCG",)}  # coverage -> neuropils not covered

# a partial-volume copy for development (label_free.py --target-coverage): optic-lobe neurons
# and the visual sensory neurons (FAFB class "visual") removed
TARGET_EXCLUSIONS = {
    "central_brain": {
        "exclude_super_classes": ("optic",),
        "exclude_types": ("R1-6", "R7", "R8", "ocellar_retinula_cell"),
    }
}


def neuropil_synapses(raw_dir: Path = RAW_DIR, processed_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    """Synapses (inputs + outputs) of every FAFB neuron per neuropil, cached as
    data/processed/fafb_neuropil_synapses.parquet (columns root_id, neuropil, synapses)."""
    cache = processed_dir / "fafb_neuropil_synapses.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    path = find_raw_file(raw_dir, CONNECTIONS_STEM)
    if path is None:
        raise FileNotFoundError(f"FAFB connections with neuropils not found in {raw_dir}")
    c = read_table(path)[[PRE_COL, POST_COL, "neuropil", WEIGHT_COL]]
    parts = [
        c.groupby([col, "neuropil"])[WEIGHT_COL].sum().rename_axis([ID_COL, "neuropil"])
        for col in (PRE_COL, POST_COL)
    ]
    out = pd.concat(parts).groupby(level=[0, 1]).sum().rename("synapses").reset_index()
    processed_dir.mkdir(parents=True, exist_ok=True)
    out.to_parquet(cache, index=False)
    return out


def coverage_weights(
    node_ids: np.ndarray, coverage: str, min_share: float = 0.1, **kw
) -> np.ndarray:
    """1 for FAFB neurons with at least `min_share` of their synapses inside the neuropils
    `coverage` covers, else 0 (unassigned synapses ignored; neurons without synapses: 0)."""
    if coverage not in COVERAGES:
        raise ValueError(f"coverage must be one of {sorted(COVERAGES)}, got {coverage!r}")
    s = neuropil_synapses(**kw)
    s = s[s.neuropil != "UNASGD"]
    base = s.neuropil.str.replace(r"_[LR]$", "", regex=True)
    inside = s.synapses.where(~base.isin(COVERAGES[coverage]), 0)
    frac = inside.groupby(s[ID_COL]).sum() / s.synapses.groupby(s[ID_COL]).sum()
    share = pd.Series(np.asarray(node_ids)).map(frac).fillna(0.0).to_numpy()
    return (share >= min_share).astype(float)
