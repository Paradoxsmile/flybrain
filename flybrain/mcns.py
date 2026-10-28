"""Male CNS connectome (MCNS v1.0, Janelia FlyEM) as a second dataset in the FAFB table schema.

Raw files (CC-BY, from gs://flyem-male-cns/v1.0/connectome-data/flat-connectome/) go to
data/raw/mcns/; see data/README.md. They are reduced once to a cache of the brain:
- neurons: status "Traced", without ventral nerve cord ("vnc_*") and enteric ("ENS") superclasses;
- super_class: MCNS superclasses mapped to the 10 FAFB super classes (SUPER_CLASS_MAP; others
  are unlabeled);
- cell_type: `flywireType` where it names one FAFB type; entries listing several types are
  unlabeled, but kept in `flywire_types` for set-based evaluation;
- connections: between kept neurons, with >= CACHE_MIN_RAW synapses.

MCNS detects more synapses per connection than FAFB, so the graph depends on how the synapse
counts are aligned (VARIANTS): "density" (default) keeps connections with >= 7 synapses, which
gives about FAFB's number of connections (3.50 M vs. 3.73 M), and scales counts by 0.581, the
factor Janelia uses to align MCNS weights with FlyWire; "raw" applies FAFB's threshold of 5
without scaling; "janelia" scales first and then applies the threshold of 5 (>= 9 raw).
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

MCNS_RAW_DIR = RAW_DIR / "mcns"
ANNOTATIONS_FILE = "body-annotations-male-cns-v1.0-minconf-0.5.feather"
WEIGHTS_FILE = "connectome-weights-male-cns-v1.0-minconf-0.5.feather"
CACHE_MIN_RAW = 5  # smallest threshold of all variants
VARIANTS = {"density": (7, 0.581), "raw": (5, 1.0), "janelia": (9, 0.581)}  # (min raw, scale)

SUPER_CLASS_MAP = {
    "ol_intrinsic": "optic",
    "cb_intrinsic": "central",
    "visual_projection": "visual_projection",
    "visual_projection_tbc": "visual_projection",
    "visual_centrifugal": "visual_centrifugal",
    "ol_sensory": "sensory",
    "cb_sensory": "sensory",
    "cb_sensory_tbc": "sensory",
    "ascending_neuron": "ascending",
    "descending_neuron": "descending",
    "descending_neuron_tbc": "descending",
    "sensory_ascending": "sensory_ascending",
    "sensory_ascending_tbc": "sensory_ascending",
    "cb_motor": "motor",
    "cb_endocrine": "endocrine",
}
SIDE_MAP = {"L": "left", "R": "right", "M": "center"}


def brain_neurons(annotations: pd.DataFrame) -> pd.DataFrame:
    """Traced brain neurons in the FAFB schema: root_id, super_class, cell_type, side,
    flywire_types (the raw `flywireType`, possibly several comma-separated types)."""
    superclass = annotations["superclass"].fillna("")
    keep = (annotations["status"] == "Traced") & ~superclass.str.startswith(("vnc_", "ENS"))
    a = annotations[keep]
    flywire = a["flywireType"].astype("string")
    side = a["somaSide"].map(SIDE_MAP).fillna(a["rootSide"].map(SIDE_MAP))
    return pd.DataFrame(
        {
            ID_COL: a["bodyId"].to_numpy(),
            "super_class": a["superclass"].map(SUPER_CLASS_MAP).to_numpy(),
            "cell_type": flywire.where(~flywire.str.contains(",", na=False)).to_numpy(),
            "side": side.fillna("").to_numpy(),
            "flywire_types": flywire.to_numpy(),
        }
    )


def build_mcns_cache(
    raw_dir: Path = MCNS_RAW_DIR, processed_dir: Path = PROCESSED_DIR
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reduce the raw MCNS files to brain neurons and their connections (>= CACHE_MIN_RAW)."""
    for name in (ANNOTATIONS_FILE, WEIGHTS_FILE):
        if not (raw_dir / name).exists():
            raise FileNotFoundError(f"{raw_dir / name} not found; see data/README.md")
    neurons = brain_neurons(pd.read_feather(raw_dir / ANNOTATIONS_FILE))
    ids = pa.array(neurons[ID_COL].to_numpy())
    weights = pf.read_table(raw_dir / WEIGHTS_FILE)
    keep = pc.and_(
        pc.and_(
            pc.is_in(weights["body_pre"], value_set=ids),
            pc.is_in(weights["body_post"], value_set=ids),
        ),
        pc.greater_equal(weights["weight"], CACHE_MIN_RAW),
    )
    edges = weights.filter(keep).to_pandas()
    connections = pd.DataFrame(
        {
            PRE_COL: edges["body_pre"].to_numpy(),
            POST_COL: edges["body_post"].to_numpy(),
            WEIGHT_COL: edges["weight"].to_numpy(),
        }
    )
    processed_dir.mkdir(parents=True, exist_ok=True)
    neurons.to_parquet(processed_dir / "mcns_neurons.parquet", index=False)
    connections.to_parquet(processed_dir / "mcns_connections.parquet", index=False)
    return neurons, connections


def load_mcns_tables(
    variant: str = "density",
    raw_dir: Path = MCNS_RAW_DIR,
    processed_dir: Path = PROCESSED_DIR,
    degrade: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(neurons, connections) of the MCNS brain, preprocessed as in VARIANTS[variant]; the cache
    is built from the raw files on first use. degrade: arguments of loading.degrade_tables,
    applied to the raw counts before the threshold, for a less complete copy."""
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {sorted(VARIANTS)}, got {variant!r}")
    n_cache = processed_dir / "mcns_neurons.parquet"
    c_cache = processed_dir / "mcns_connections.parquet"
    if n_cache.exists() and c_cache.exists():
        neurons, connections = pd.read_parquet(n_cache), pd.read_parquet(c_cache)
    else:
        neurons, connections = build_mcns_cache(raw_dir, processed_dir)
    if degrade:
        neurons, connections = degrade_tables(neurons, connections, **degrade, min_count=None)
    min_raw, scale = VARIANTS[variant]
    connections = connections[connections[WEIGHT_COL] >= min_raw].copy()
    connections[WEIGHT_COL] = connections[WEIGHT_COL].to_numpy(dtype=np.float64) * scale
    return neurons, connections


# --- nerve cord (docs/vnc_rare_protocol.md): a target of the MANC source -----------------------

CORD_CACHE_MIN_RAW = 1  # smallest threshold D2 considered
# Variants by the BANC rules, relative to MANC at threshold 5 (the FlyWire factor 0.581 of the
# brain variants is specific to FAFB): "raw" threshold 5; "density" the threshold whose
# connections per neuron are closest to MANC's; "scaled" threshold 5, counts x the factor that
# matches MANC's synapses per neuron. From D2 (results/vnc_rare/d2_graph_stats.json; no labels):
# the MCNS cord at >= 5 has 43.4 connections and 759 synapses per neuron (MANC: 55.1 and 969);
# >= 4 gives 52.8 connections per neuron.
CORD_VARIANTS = {"raw": (5, 1.0), "density": (4, 1.0), "scaled": (5, 1.28)}  # (min raw, scale)
CORD_PASSING = (
    "ascending_neuron",
    "descending_neuron",
    "descending_neuron_tbc",
    "sensory_ascending",
    "sensory_ascending_tbc",
    "sensory_descending",
    "efferent_ascending",
    "efferent_descending",
)
CORD_SUPER_CLASS_MAP = {  # to the harmonized MANC super classes
    "vnc_intrinsic": "ventral_nerve_cord_intrinsic",
    "vnc_tbc": "ventral_nerve_cord_intrinsic",
    "vnc_sensory": "sensory",
    "vnc_sensory_tbc": "sensory",
    "vnc_motor": "motor",
    "vnc_efferent": "visceral_circulatory",
    "vnc_endocrine": "visceral_circulatory",
    "ascending_neuron": "ascending",
    "descending_neuron": "descending",
    "descending_neuron_tbc": "descending",
    "sensory_ascending": "sensory_ascending",
    "sensory_ascending_tbc": "sensory_ascending",
    "sensory_descending": "sensory_descending",
    "efferent_ascending": "ascending_visceral_circulatory",
}


def cord_neurons(annotations: pd.DataFrame) -> pd.DataFrame:
    """Traced nerve-cord neurons in the MANC schema: root_id, super_class, cell_type (`mancType`,
    the type of the matched MANC body; entries listing several types are unlabeled), manc_types
    (the raw `mancType`), side, passing."""
    superclass = annotations["superclass"].fillna("")
    in_cord = superclass.str.startswith("vnc_") | superclass.isin(CORD_PASSING)
    a = annotations[(annotations["status"] == "Traced") & in_cord]
    manc = a["mancType"].astype("string")
    side = a["somaSide"].map(SIDE_MAP).fillna(a["rootSide"].map(SIDE_MAP))
    return pd.DataFrame(
        {
            ID_COL: a["bodyId"].to_numpy(),
            "super_class": a["superclass"].map(CORD_SUPER_CLASS_MAP).to_numpy(),
            "cell_type": manc.where(~manc.str.contains(",", na=False)).to_numpy(),
            "manc_types": manc.to_numpy(),
            "side": side.fillna("").to_numpy(dtype=object),
            "passing": a["superclass"].isin(CORD_PASSING).to_numpy(),
        }
    )


def build_mcns_cord_cache(
    raw_dir: Path = MCNS_RAW_DIR, processed_dir: Path = PROCESSED_DIR
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reduce the raw MCNS files to cord neurons and their connections (>= CORD_CACHE_MIN_RAW)."""
    for name in (ANNOTATIONS_FILE, WEIGHTS_FILE):
        if not (raw_dir / name).exists():
            raise FileNotFoundError(f"{raw_dir / name} not found; see data/README.md")
    neurons = cord_neurons(pd.read_feather(raw_dir / ANNOTATIONS_FILE))
    ids = pa.array(neurons[ID_COL].to_numpy())
    weights = pf.read_table(raw_dir / WEIGHTS_FILE)
    keep = pc.and_(
        pc.and_(
            pc.is_in(weights["body_pre"], value_set=ids),
            pc.is_in(weights["body_post"], value_set=ids),
        ),
        pc.greater_equal(weights["weight"], CORD_CACHE_MIN_RAW),
    )
    edges = weights.filter(keep).to_pandas()
    connections = pd.DataFrame(
        {
            PRE_COL: edges["body_pre"].to_numpy(),
            POST_COL: edges["body_post"].to_numpy(),
            WEIGHT_COL: edges["weight"].to_numpy(),
        }
    )
    connections = drop_self_connections(connections).reset_index(drop=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    neurons.to_parquet(processed_dir / "mcns_cord_neurons.parquet", index=False)
    connections.to_parquet(processed_dir / "mcns_cord_connections.parquet", index=False)
    return neurons, connections


def mcns_cord_cache(
    raw_dir: Path = MCNS_RAW_DIR, processed_dir: Path = PROCESSED_DIR
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The cached cord (neurons, connections), all counts >= CORD_CACHE_MIN_RAW, passing pairs
    kept."""
    n_cache = processed_dir / "mcns_cord_neurons.parquet"
    c_cache = processed_dir / "mcns_cord_connections.parquet"
    if n_cache.exists() and c_cache.exists():
        return pd.read_parquet(n_cache), pd.read_parquet(c_cache)
    return build_mcns_cord_cache(raw_dir, processed_dir)


def load_mcns_cord_tables(
    variant: str = "raw",
    raw_dir: Path = MCNS_RAW_DIR,
    processed_dir: Path = PROCESSED_DIR,
    degrade: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(neurons, connections) of the MCNS nerve cord, preprocessed as in CORD_VARIANTS[variant],
    without connections between two passing neurons (loading.drop_passing_pairs)."""
    if variant not in CORD_VARIANTS:
        raise ValueError(f"variant must be one of {sorted(CORD_VARIANTS)}, got {variant!r}")
    neurons, connections = mcns_cord_cache(raw_dir, processed_dir)
    if degrade:
        neurons, connections = degrade_tables(neurons, connections, **degrade, min_count=None)
    min_raw, scale = CORD_VARIANTS[variant]
    connections = connections[connections[WEIGHT_COL] >= min_raw].copy()
    connections[WEIGHT_COL] = connections[WEIGHT_COL].to_numpy(dtype=np.float64) * scale
    return neurons, drop_passing_pairs(neurons, connections)
