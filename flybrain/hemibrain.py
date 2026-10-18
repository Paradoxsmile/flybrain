"""Hemibrain v1.2.1 (Janelia FlyEM, female; Scheffer et al., 2020) as a fourth dataset in the
FAFB table schema. Used only as an unlabeled transfer target; the evaluation protocol is fixed
in docs/hemibrain_protocol.md.

Raw files (harmonized by the BANC project, CC BY 4.0, from
gs://lee-lab_brain-and-nerve-cord-fly-connectome/compiled_data/hemibrain_121/) go to
data/raw/hemibrain/; see data/README.md. They are reduced once to a cache:
- neurons: status "Traced" (all of them; the volume is the central brain, mostly of the right
  hemisphere, without the optic lobes);
- super_class: hemibrain super classes mapped to the 10 FAFB super classes (SUPER_CLASS_MAP);
- cell_type: the hemibrain type (FAFB adopted most central-brain type names from the hemibrain;
  Schlegel et al., 2024);
- side: from the instance suffix (_L / _R); region: the harmonized region of the arbor;
  cropped: "cropped" for neurons cut by the volume boundary (16 %), else "";
- connections: between traced neurons, with >= CACHE_MIN_RAW synapses. The harmonized edgelist
  repeats many (pre, post) rows exactly (931,329 copies, up to 12 per pair, always with the
  same count); copies are dropped before anything else.

Within its volume the hemibrain is about as dense as FAFB: at FAFB's threshold of 5 synapses
it has 26.7 connections and 387 synapses per neuron (FAFB: 26.8 and 364). The variants follow
the rules used for BANC: "raw" applies FAFB's threshold of 5 and nothing else; the threshold
whose connections per neuron are closest to FAFB's is 5 as well, so the "density" variant
equals "raw" and is left out; "scaled" keeps the threshold and multiplies counts so that
synapses per neuron match FAFB's (x 0.94).
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
)

HEMIBRAIN_RAW_DIR = RAW_DIR / "hemibrain"
META_FILE = "hemibrain_121_meta.feather"
EDGES_FILE = "hemibrain_121_simple_edgelist.feather"
CACHE_MIN_RAW = 5  # smallest threshold of all variants
VARIANTS = {"raw": (5, 1.0), "scaled": (5, 0.94)}  # (min raw, scale)

SUPER_CLASS_MAP = {
    "central_brain_intrinsic": "central",
    "optic_lobe_intrinsic": "optic",
    "visual_projection": "visual_projection",
    "visual_centrifugal": "visual_centrifugal",
    "sensory": "sensory",
    "descending": "descending",
    "ascending": "ascending",
    "visceral_circulatory": "endocrine",
    "endocrine": "endocrine",
}
SIDE_MAP = {"L": "left", "R": "right"}


def brain_neurons(meta: pd.DataFrame) -> pd.DataFrame:
    """Traced hemibrain neurons in the FAFB schema: root_id, super_class, cell_type, side,
    region, cropped."""
    m = meta[meta["status"] == "Traced"]
    side = m["instance"].astype("string").str.extract(r"_([LR])$")[0].map(SIDE_MAP)
    cropped = m["cropped"].astype("boolean").fillna(False).to_numpy(dtype=bool)
    return pd.DataFrame(
        {
            ID_COL: m["hemibrain_121_id"].astype(np.int64).to_numpy(),
            "super_class": m["super_class"].map(SUPER_CLASS_MAP).to_numpy(),
            "cell_type": m["cell_type"].astype("string").to_numpy(),
            "side": side.fillna("").to_numpy(),
            "region": m["region"].fillna("").to_numpy(),
            "cropped": np.where(cropped, "cropped", ""),
        }
    )


def build_hemibrain_cache(
    raw_dir: Path = HEMIBRAIN_RAW_DIR, processed_dir: Path = PROCESSED_DIR
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reduce the raw hemibrain files to traced neurons and their connections."""
    for name in (META_FILE, EDGES_FILE):
        if not (raw_dir / name).exists():
            raise FileNotFoundError(f"{raw_dir / name} not found; see data/README.md")
    neurons = brain_neurons(pd.read_feather(raw_dir / META_FILE))
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
    processed_dir.mkdir(parents=True, exist_ok=True)
    neurons.to_parquet(processed_dir / "hemibrain_neurons.parquet", index=False)
    connections.to_parquet(processed_dir / "hemibrain_connections.parquet", index=False)
    return neurons, connections


def load_hemibrain_tables(
    variant: str = "raw",
    raw_dir: Path = HEMIBRAIN_RAW_DIR,
    processed_dir: Path = PROCESSED_DIR,
    degrade: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(neurons, connections) of the hemibrain, preprocessed as in VARIANTS[variant]; the cache
    is built from the raw files on first use. degrade: arguments of loading.degrade_tables."""
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {sorted(VARIANTS)}, got {variant!r}")
    n_cache = processed_dir / "hemibrain_neurons.parquet"
    c_cache = processed_dir / "hemibrain_connections.parquet"
    if n_cache.exists() and c_cache.exists():
        neurons, connections = pd.read_parquet(n_cache), pd.read_parquet(c_cache)
    else:
        neurons, connections = build_hemibrain_cache(raw_dir, processed_dir)
    if degrade:
        neurons, connections = degrade_tables(neurons, connections, **degrade, min_count=None)
    min_raw, scale = VARIANTS[variant]
    connections = connections[connections[WEIGHT_COL] >= min_raw].copy()
    connections[WEIGHT_COL] = connections[WEIGHT_COL].to_numpy(dtype=np.float64) * scale
    return neurons, connections
