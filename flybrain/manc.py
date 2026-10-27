"""MANC v1.2.1 (male adult nerve cord, Janelia FlyEM; Takemura et al., 2024; Marin et al., 2024;
Cheong et al., 2024) as the source of the nerve-cord test (docs/vnc_rare_protocol.md), in the FAFB
table schema.

Raw files (harmonized by the BANC project, CC BY 4.0, from
gs://lee-lab_brain-and-nerve-cord-fly-connectome/compiled_data/manc_121/) go to data/raw/manc/;
see data/README.md. They are reduced once to a cache:
- neurons: all 23,650 of the harmonized table (the cord, and the neck-connective fragments of
  ascending and descending neurons);
- super_class: the harmonized super class, unchanged (the cord targets are mapped to it);
- cell_type: the MANC type; placeholder names ending in "xx" (SNxxxx, SNppxx, ...) and
  comma-joined names are unlabeled;
- side: "left", "right", "center" or "" (for passing neurons the side of the neck connective);
- passing: ascending, descending, sensory ascending / descending and ascending visceral neurons,
  the neurons that pass through the neck (PASSING);
- connections: with >= CACHE_MIN_RAW synapses; exact duplicate (pre, post) rows are dropped.

Like FAFB, the source graph keeps connections with >= 5 synapses (one variant, "raw"). As in the
cord targets, connections between two passing neurons are dropped (loading.drop_passing_pairs):
in a brain-and-cord connectome they can lie in the brain, which MANC lacks.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.feather as pf

from flybrain.loading import (
    ID_COL,
    POST_COL,
    PRE_COL,
    PROCESSED_DIR,
    RAW_DIR,
    WEIGHT_COL,
    degrade_tables,
    drop_passing_pairs,
    drop_self_connections,
)

MANC_RAW_DIR = RAW_DIR / "manc"
META_FILE = "manc_121_meta.feather"
EDGES_FILE = "manc_121_simple_edgelist.feather"
CACHE_MIN_RAW = 1  # D2 measures the target thresholds against MANC's statistics
VARIANTS = {"raw": (5, 1.0)}  # (min raw, scale)
# super classes (harmonized names, also BANC's) of neurons that pass through the neck
PASSING = (
    "ascending",
    "descending",
    "sensory_ascending",
    "sensory_descending",
    "ascending_visceral_circulatory",
)


def manc_type_names(types: pd.Series) -> pd.Series:
    """MANC type names with placeholders ("...xx") and comma-joined names unlabeled."""
    t = types.astype("string")
    bad = t.str.endswith("xx", na=False) | t.str.contains(",", na=False)
    return t.where(~bad)


def cord_neurons(meta: pd.DataFrame) -> pd.DataFrame:
    """All MANC neurons in the FAFB schema: root_id, super_class, cell_type, side, passing."""
    side = meta["side"].astype("string").where(meta["side"].isin(["left", "right", "center"]))
    return pd.DataFrame(
        {
            ID_COL: meta["manc_121_id"].astype(np.int64).to_numpy(),
            "super_class": meta["super_class"].astype("string").to_numpy(),
            "cell_type": manc_type_names(meta["cell_type"]).to_numpy(),
            "side": side.fillna("").to_numpy(dtype=object),
            "passing": meta["super_class"].isin(PASSING).to_numpy(),
        }
    )


def build_manc_cache(
    raw_dir: Path = MANC_RAW_DIR, processed_dir: Path = PROCESSED_DIR
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reduce the raw MANC files to neurons and their connections (>= CACHE_MIN_RAW)."""
    for name in (META_FILE, EDGES_FILE):
        if not (raw_dir / name).exists():
            raise FileNotFoundError(f"{raw_dir / name} not found; see data/README.md")
    neurons = cord_neurons(pd.read_feather(raw_dir / META_FILE))
    ids = pa.array(neurons[ID_COL].astype(str).to_numpy())
    edges = pf.read_table(raw_dir / EDGES_FILE, columns=["pre", "post", "count"])
    edges = pa.Table.from_pandas(
        edges.to_pandas().drop_duplicates(["pre", "post"]), preserve_index=False
    )
    keep = pc.and_(
        pc.and_(pc.is_in(edges["pre"], value_set=ids), pc.is_in(edges["post"], value_set=ids)),
        pc.greater_equal(edges["count"], CACHE_MIN_RAW),
    )
    edges = edges.filter(keep)
    connections = pd.DataFrame(
        {
            PRE_COL: pc.cast(edges["pre"], pa.int64()).to_numpy(),
            POST_COL: pc.cast(edges["post"], pa.int64()).to_numpy(),
            WEIGHT_COL: edges["count"].to_numpy(),
        }
    )
    connections = drop_self_connections(connections).reset_index(drop=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    neurons.to_parquet(processed_dir / "manc_neurons.parquet", index=False)
    connections.to_parquet(processed_dir / "manc_connections.parquet", index=False)
    return neurons, connections


def manc_cache(
    raw_dir: Path = MANC_RAW_DIR, processed_dir: Path = PROCESSED_DIR
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The cached (neurons, connections), all counts >= CACHE_MIN_RAW, passing pairs kept."""
    n_cache = processed_dir / "manc_neurons.parquet"
    c_cache = processed_dir / "manc_connections.parquet"
    if n_cache.exists() and c_cache.exists():
        return pd.read_parquet(n_cache), pd.read_parquet(c_cache)
    return build_manc_cache(raw_dir, processed_dir)


def load_manc_tables(
    variant: str = "raw",
    raw_dir: Path = MANC_RAW_DIR,
    processed_dir: Path = PROCESSED_DIR,
    degrade: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(neurons, connections) of MANC, preprocessed as in VARIANTS[variant], without connections
    between two passing neurons. degrade: arguments of loading.degrade_tables, applied to the
    raw counts before the threshold."""
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {sorted(VARIANTS)}, got {variant!r}")
    neurons, connections = manc_cache(raw_dir, processed_dir)
    if degrade:
        neurons, connections = degrade_tables(neurons, connections, **degrade, min_count=None)
    min_raw, scale = VARIANTS[variant]
    connections = connections[connections[WEIGHT_COL] >= min_raw].copy()
    connections[WEIGHT_COL] = connections[WEIGHT_COL].to_numpy(dtype=np.float64) * scale
    return neurons, drop_passing_pairs(neurons, connections)
