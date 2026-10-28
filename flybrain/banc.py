"""BANC v888 (brain and nerve cord, female; Bates et al., 2026) as a third dataset in the FAFB
table schema. Used only as an unlabeled transfer target; the evaluation protocol is fixed in
docs/banc_protocol.md.

Raw files (CC-BY 4.0, from gs://lee-lab_brain-and-nerve-cord-fly-connectome/compiled_data/
banc_888/) go to data/raw/banc/; see data/README.md. They are reduced once to a cache of the
brain:
- neurons: proofread or roughly proofread, not glia / trachea / non-neuronal, in the central
  brain or optic lobe, plus ascending and descending neurons (their nerve-cord partners are
  dropped with the nerve cord);
- super_class: BANC super classes mapped to the 10 FAFB super classes (SUPER_CLASS_MAP);
- cell_type: BANC's curated `cell_type` (named after FAFB types in the brain);
- morph_type: `fafb_cell_type`, the type of the FAFB neuron matched by morphology (NBLAST, with
  review), central brain only; in the optic lobe this column holds automatic placeholders;
- region: "central_brain", "optic_lobe" or "" (ascending neurons);
- connections: between kept neurons, with >= CACHE_MIN_RAW synapses (v2 synapse set).

The BANC brain graph is much sparser than FAFB's: at FAFB's threshold of 5 synapses it has 32 %
of FAFB's connections per neuron, with a similar number of synapses per connection. Each
variant aligns one statistic with FAFB (FAFB v783: 26.8 connections and 364 synapses per
neuron; the values below follow from these by the stated rule, without labels):
"raw" applies FAFB's threshold of 5 and nothing else; "density" lowers the threshold to the one
whose number of connections per neuron is closest to FAFB's (>= 2); "scaled" keeps the
threshold of 5 and multiplies counts so that synapses per neuron match FAFB's (x 3.63).
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

BANC_RAW_DIR = RAW_DIR / "banc"
META_FILE = "banc_888_meta.feather"
EDGES_FILE = "banc_888_edgelist_simple_v2.feather"
CACHE_MIN_RAW = 2  # smallest threshold of all variants
VARIANTS = {"raw": (5, 1.0), "density": (2, 1.0), "scaled": (5, 3.63)}  # (min raw, scale)

SUPER_CLASS_MAP = {
    "optic_lobe_intrinsic": "optic",
    "central_brain_intrinsic": "central",
    "sensory": "sensory",
    "visual_projection": "visual_projection",
    "visual_centrifugal": "visual_centrifugal",
    "ascending": "ascending",
    "sensory_ascending": "sensory_ascending",
    "descending": "descending",
    "motor": "motor",
    "visceral_circulatory": "endocrine",
}
NOT_NEURONS = ("glia", "trachea", "not_a_neuron")
PASSING = ("ascending", "sensory_ascending", "descending", "sensory_descending")


def brain_neurons(meta: pd.DataFrame) -> pd.DataFrame:
    """Proofread brain neurons in the FAFB schema: root_id, super_class, cell_type, side, region,
    morph_type."""
    super_class = meta["super_class"].fillna("")
    region = meta["region"].fillna("")
    proofread = (meta["proofread"] == "TRUE") | (meta["roughly_proofread"] == "TRUE")
    in_brain = region.isin(["central_brain", "optic_lobe"]) | super_class.isin(PASSING)
    m = meta[proofread & in_brain & ~super_class.isin(NOT_NEURONS)]
    region = m["region"].where(m["region"].isin(["central_brain", "optic_lobe"]))
    return pd.DataFrame(
        {
            ID_COL: m["banc_888_id"].astype(np.int64).to_numpy(),
            "super_class": m["super_class"].map(SUPER_CLASS_MAP).to_numpy(),
            "cell_type": m["cell_type"].astype("string").to_numpy(),
            "side": m["side"].fillna("").to_numpy(),
            "region": region.fillna("").to_numpy(),
            "morph_type": m["fafb_cell_type"]
            .astype("string")
            .where(region == "central_brain")
            .to_numpy(),
        }
    )


def build_banc_cache(
    raw_dir: Path = BANC_RAW_DIR, processed_dir: Path = PROCESSED_DIR
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reduce the raw BANC files to brain neurons and their connections (>= CACHE_MIN_RAW)."""
    for name in (META_FILE, EDGES_FILE):
        if not (raw_dir / name).exists():
            raise FileNotFoundError(f"{raw_dir / name} not found; see data/README.md")
    neurons = brain_neurons(pd.read_feather(raw_dir / META_FILE))
    ids = pa.array(neurons[ID_COL].astype(str).to_numpy())
    edges = pf.read_table(raw_dir / EDGES_FILE, columns=["pre", "post", "count"])
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
    processed_dir.mkdir(parents=True, exist_ok=True)
    neurons.to_parquet(processed_dir / "banc_neurons.parquet", index=False)
    connections.to_parquet(processed_dir / "banc_connections.parquet", index=False)
    return neurons, connections


def load_banc_tables(
    variant: str = "raw",
    raw_dir: Path = BANC_RAW_DIR,
    processed_dir: Path = PROCESSED_DIR,
    degrade: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(neurons, connections) of the BANC brain, preprocessed as in VARIANTS[variant]; the cache
    is built from the raw files on first use. degrade: arguments of loading.degrade_tables,
    applied to the raw counts before the threshold, for a less complete copy."""
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {sorted(VARIANTS)}, got {variant!r}")
    n_cache = processed_dir / "banc_neurons.parquet"
    c_cache = processed_dir / "banc_connections.parquet"
    if n_cache.exists() and c_cache.exists():
        neurons, connections = pd.read_parquet(n_cache), pd.read_parquet(c_cache)
    else:
        neurons, connections = build_banc_cache(raw_dir, processed_dir)
    if degrade:
        neurons, connections = degrade_tables(neurons, connections, **degrade, min_count=None)
    min_raw, scale = VARIANTS[variant]
    connections = connections[connections[WEIGHT_COL] >= min_raw].copy()
    connections[WEIGHT_COL] = connections[WEIGHT_COL].to_numpy(dtype=np.float64) * scale
    return neurons, connections


# --- nerve cord (docs/vnc_rare_protocol.md): a target of the MANC source -----------------------

CORD_CACHE_MIN_RAW = 1  # smallest threshold D2 considered
# Variants by the BANC rules, relative to MANC at threshold 5 instead of FAFB: "raw" threshold
# 5; "density" the threshold whose connections per neuron are closest to MANC's; "scaled"
# threshold 5, counts x the factor that matches MANC's synapses per neuron. From D2
# (results/vnc_rare/d2_graph_stats.json; no labels): MANC at >= 5 has 55.1 connections and 969
# synapses per neuron, the BANC cord 16.9 and 235 without self-connections (31 % of MANC's
# connections, about what the brain had of FAFB's); >= 2 gives 47.8 connections per neuron.
CORD_VARIANTS = {"raw": (5, 1.0), "density": (2, 1.0), "scaled": (5, 4.13)}  # (min raw, scale)
CORD_PASSING = (
    "ascending",
    "descending",
    "sensory_ascending",
    "sensory_descending",
    "ascending_visceral_circulatory",
)


def cord_neurons(meta: pd.DataFrame) -> pd.DataFrame:
    """Proofread nerve-cord neurons in the MANC schema: root_id, super_class, cell_type (reference
    A: manc_cell_type, the type of the matched MANC body), alt_type (reference B: BANC's curated
    cell_type), side, passing. Descending neurons, whose somata lie in the brain, are included."""
    super_class = meta["super_class"].fillna("")
    proofread = (meta["proofread"] == "TRUE") | (meta["roughly_proofread"] == "TRUE")
    in_cord = (meta["region"] == "ventral_nerve_cord") | super_class.isin(
        ["descending", "sensory_descending"]
    )
    m = meta[proofread & in_cord & ~super_class.isin(NOT_NEURONS)]
    return pd.DataFrame(
        {
            ID_COL: m["banc_888_id"].astype(np.int64).to_numpy(),
            "super_class": m["super_class"].astype("string").to_numpy(),
            "cell_type": m["manc_cell_type"].astype("string").to_numpy(),
            "alt_type": m["cell_type"].astype("string").to_numpy(),
            "side": m["side"].fillna("").to_numpy(dtype=object),
            "passing": m["super_class"].isin(CORD_PASSING).to_numpy(),
        }
    )


def build_banc_cord_cache(
    raw_dir: Path = BANC_RAW_DIR, processed_dir: Path = PROCESSED_DIR
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reduce the raw BANC files to cord neurons and their connections (>= CORD_CACHE_MIN_RAW)."""
    for name in (META_FILE, EDGES_FILE):
        if not (raw_dir / name).exists():
            raise FileNotFoundError(f"{raw_dir / name} not found; see data/README.md")
    neurons = cord_neurons(pd.read_feather(raw_dir / META_FILE))
    ids = pa.array(neurons[ID_COL].astype(str).to_numpy())
    edges = pf.read_table(raw_dir / EDGES_FILE, columns=["pre", "post", "count"])
    keep = pc.and_(
        pc.and_(pc.is_in(edges["pre"], value_set=ids), pc.is_in(edges["post"], value_set=ids)),
        pc.greater_equal(edges["count"], CORD_CACHE_MIN_RAW),
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
    neurons.to_parquet(processed_dir / "banc_cord_neurons.parquet", index=False)
    connections.to_parquet(processed_dir / "banc_cord_connections.parquet", index=False)
    return neurons, connections


def banc_cord_cache(
    raw_dir: Path = BANC_RAW_DIR, processed_dir: Path = PROCESSED_DIR
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The cached cord (neurons, connections), all counts >= CORD_CACHE_MIN_RAW, passing pairs
    kept."""
    n_cache = processed_dir / "banc_cord_neurons.parquet"
    c_cache = processed_dir / "banc_cord_connections.parquet"
    if n_cache.exists() and c_cache.exists():
        return pd.read_parquet(n_cache), pd.read_parquet(c_cache)
    return build_banc_cord_cache(raw_dir, processed_dir)


def load_banc_cord_tables(
    variant: str = "raw",
    raw_dir: Path = BANC_RAW_DIR,
    processed_dir: Path = PROCESSED_DIR,
    degrade: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(neurons, connections) of the BANC nerve cord, preprocessed as in CORD_VARIANTS[variant],
    without connections between two passing neurons (loading.drop_passing_pairs)."""
    if variant not in CORD_VARIANTS:
        raise ValueError(f"variant must be one of {sorted(CORD_VARIANTS)}, got {variant!r}")
    neurons, connections = banc_cord_cache(raw_dir, processed_dir)
    if degrade:
        neurons, connections = degrade_tables(neurons, connections, **degrade, min_count=None)
    min_raw, scale = CORD_VARIANTS[variant]
    connections = connections[connections[WEIGHT_COL] >= min_raw].copy()
    connections[WEIGHT_COL] = connections[WEIGHT_COL].to_numpy(dtype=np.float64) * scale
    return neurons, drop_passing_pairs(neurons, connections)
