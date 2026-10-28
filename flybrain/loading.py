"""Raw tables -> cleaned Parquet cache -> PyG Data object.

Raw files are placed manually in data/raw/ (see data/README.md). Filename stems and column
names below match the FlyWire Codex FAFB v783 exports: adjust them here if yours differ.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from flybrain.features import degree_features
from flybrain.thinning import thin_counts
from flybrain.utils import REPO_ROOT

RAW_DIR = REPO_ROOT / "data" / "raw"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

# Stems are tried in order; the Codex original names work unchanged.
NEURONS_STEM = ("classification",)  # one row per neuron
CONNECTIONS_STEM = ("connections", "connections_princeton")  # one row per (pre, post, neuropil)
CELL_TYPES_STEM = ("consolidated_cell_types",)  # optional, one row per typed neuron
EXTENSIONS = (".parquet", ".feather", ".csv", ".csv.gz")

ID_COL = "root_id"
PRE_COL = "pre_root_id"
POST_COL = "post_root_id"
WEIGHT_COL = "syn_count"
CELL_TYPE_SRC_COL = "primary_type"  # column in the cell-types table, merged in as "cell_type"
LABEL_COLS = ("super_class", "cell_type")
# FAFB super classes of the neurons that pass through the neck (graph attribute `passing`)
BRAIN_PASSING = ("ascending", "descending", "sensory_ascending", "sensory_descending")
MIN_COUNT = 5  # Codex threshold of the FAFB connections


def find_raw_file(raw_dir: Path, stems: str | tuple[str, ...]) -> Path | None:
    """Return the first existing raw file for the given stem(s), trying known extensions."""
    for stem in (stems,) if isinstance(stems, str) else stems:
        for ext in EXTENSIONS:
            path = raw_dir / f"{stem}{ext}"
            if path.exists():
                return path
    return None


def read_table(path: Path) -> pd.DataFrame:
    """Read a csv / csv.gz / parquet / feather table."""
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".feather":
        return pd.read_feather(path)
    return pd.read_csv(path)


def clean_tables(
    neurons: pd.DataFrame, connections: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate columns, dedupe neurons, sum synapses per (pre, post), drop unknown ids."""
    for df, cols, name in (
        (neurons, [ID_COL], "neurons"),
        (connections, [PRE_COL, POST_COL, WEIGHT_COL], "connections"),
    ):
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"{name} table is missing columns: {missing}")
    neurons = neurons.drop_duplicates(ID_COL).reset_index(drop=True)
    known = set(neurons[ID_COL])
    connections = connections[connections[PRE_COL].isin(known) & connections[POST_COL].isin(known)]
    connections = connections.groupby([PRE_COL, POST_COL], as_index=False)[WEIGHT_COL].sum()
    return neurons, connections


def merge_cell_types(neurons: pd.DataFrame, raw_dir: Path) -> pd.DataFrame:
    """Attach `cell_type` from the optional cell-types table if the neurons table lacks it."""
    if "cell_type" in neurons.columns:
        return neurons
    path = find_raw_file(raw_dir, CELL_TYPES_STEM)
    if path is None:
        return neurons
    types = read_table(path)[[ID_COL, CELL_TYPE_SRC_COL]].drop_duplicates(ID_COL)
    types = types.rename(columns={CELL_TYPE_SRC_COL: "cell_type"})
    return neurons.merge(types, on=ID_COL, how="left")


def load_tables(
    raw_dir: Path = RAW_DIR, processed_dir: Path = PROCESSED_DIR
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load cleaned tables from the Parquet cache, building it from raw files if needed."""
    n_cache = processed_dir / "neurons.parquet"
    c_cache = processed_dir / "connections.parquet"
    if n_cache.exists() and c_cache.exists():
        return pd.read_parquet(n_cache), pd.read_parquet(c_cache)
    n_raw = find_raw_file(raw_dir, NEURONS_STEM)
    c_raw = find_raw_file(raw_dir, CONNECTIONS_STEM)
    if n_raw is None or c_raw is None:
        raise FileNotFoundError(
            f"Expected one of {NEURONS_STEM} and one of {CONNECTIONS_STEM} (any of "
            f"{EXTENSIONS}) in {raw_dir}. See data/README.md or run data/download_fafb.py."
        )
    neurons = merge_cell_types(read_table(n_raw), raw_dir)
    neurons, connections = clean_tables(neurons, read_table(c_raw))
    processed_dir.mkdir(parents=True, exist_ok=True)
    neurons.to_parquet(n_cache, index=False)
    connections.to_parquet(c_cache, index=False)
    return neurons, connections


def encode_labels(
    labels: pd.Series, min_class_size: int = 1, label_names: list[str] | None = None
) -> tuple[np.ndarray, list[str]]:
    """Integer codes (-1 = unlabeled) and class names for a label column.

    Without label_names, classes with fewer than min_class_size members are unlabeled and the
    remaining classes are numbered in order of appearance. With label_names (e.g. the classes of
    another dataset), labels are coded in that label space and labels outside it are unlabeled.
    """
    labels = labels.astype("string")
    if label_names is None:
        counts = labels.value_counts()
        codes, names = pd.factorize(labels.where(labels.map(counts) >= min_class_size))
        return codes, [str(n) for n in names]
    lookup = {name: i for i, name in enumerate(label_names)}
    codes = labels.astype(object).map(lookup).fillna(-1).to_numpy(dtype=np.int64, copy=True)
    return codes, list(label_names)


def build_graph(
    neurons: pd.DataFrame,
    connections: pd.DataFrame,
    label_col: str = "super_class",
    min_class_size: int = 1,
    label_names: list[str] | None = None,
) -> Data:
    """Build a PyG Data object. y is the integer label (-1 if unlabeled).

    Classes with fewer than `min_class_size` neurons are treated as unlabeled; with
    `label_names`, labels are coded in that label space instead (see encode_labels).

    Attributes: x (degree features), edge_index, edge_weight, y, node_ids, label_names, and
    side (array of "left" / "right" / "center" / "", if the neurons table has a side column);
    flywire_types (MCNS), region and morph_type (BANC), cropped (hemibrain), alt_type (BANC
    cord) and manc_types (MCNS cord) likewise, as string arrays; passing (the neurons that pass
    through the neck: the table's `passing` column, else super_class in BRAIN_PASSING) as a
    boolean array.
    """
    neurons, connections = clean_tables(neurons, connections)
    index = pd.Series(np.arange(len(neurons)), index=neurons[ID_COL])
    src = index[connections[PRE_COL]].to_numpy()
    dst = index[connections[POST_COL]].to_numpy()
    edge_index = torch.from_numpy(np.stack([src, dst])).long()
    edge_weight = torch.tensor(connections[WEIGHT_COL].to_numpy(), dtype=torch.float)

    codes, names = encode_labels(neurons[label_col], min_class_size, label_names)
    x = torch.from_numpy(degree_features(len(neurons), edge_index, edge_weight)).float()
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_weight=edge_weight,
        y=torch.from_numpy(codes).long(),
        num_nodes=len(neurons),
    )
    data.node_ids = torch.from_numpy(neurons[ID_COL].to_numpy().copy())
    data.label_names = names
    if "passing" in neurons.columns:
        data.passing = neurons["passing"].astype(bool).to_numpy(copy=True)
    elif "super_class" in neurons.columns:  # brain graphs (FAFB super classes)
        data.passing = neurons["super_class"].isin(BRAIN_PASSING).to_numpy()
    for col in (
        "side",
        "flywire_types",
        "region",
        "morph_type",
        "cropped",
        "alt_type",
        "manc_types",
    ):
        if col in neurons.columns:
            setattr(data, col, neurons[col].astype("string").fillna("").to_numpy(dtype=str))
    return data


CROP_RETENTION = 0.1  # cropped hemibrain neurons: median 30 vs. 340 input synapses


def degrade_tables(
    neurons: pd.DataFrame,
    connections: pd.DataFrame,
    retention: float = 1.0,
    seed: int = 0,
    exclude_types: tuple = (),
    min_count: int | None = MIN_COUNT,
    exclude_super_classes: tuple = (),
    crop_fraction: float = 0.0,
    crop_retention: float = CROP_RETENTION,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A less complete copy of a connectome, for development with known labels.

    Neurons of `exclude_types` (cell_type) or `exclude_super_classes` are removed (a partial
    volume); synapse counts are binomially thinned with `retention` (flybrain.thinning); a
    random `crop_fraction` of the remaining neurons is cropped, i.e. their connections keep
    each synapse only with probability crop_retention (per cropped end), as for neurons cut by
    the volume boundary; with min_count, connections with fewer remaining synapses are
    dropped."""
    drop = neurons["cell_type"].isin(list(exclude_types))
    drop |= neurons["super_class"].isin(list(exclude_super_classes))
    neurons = neurons[~drop]
    if retention == 1.0 and crop_fraction == 0.0:
        return neurons, connections
    rng = np.random.default_rng(seed)
    if crop_fraction > 0.0:
        cropped = neurons[ID_COL].to_numpy()[rng.random(len(neurons)) < crop_fraction]
        ends = connections[PRE_COL].isin(cropped).to_numpy().astype(int)
        ends = ends + connections[POST_COL].isin(cropped).to_numpy().astype(int)
        p = retention * crop_retention**ends
        counts = connections[WEIGHT_COL].to_numpy().round().astype(np.int64)
        counts = rng.binomial(counts, p)
    else:
        counts = thin_counts(connections[WEIGHT_COL].to_numpy(), retention, seed)
    connections = connections.assign(**{WEIGHT_COL: counts})
    if min_count is not None:
        connections = connections[connections[WEIGHT_COL] >= min_count]
    return neurons, connections


def drop_passing_pairs(neurons: pd.DataFrame, connections: pd.DataFrame) -> pd.DataFrame:
    """The connections without those between two passing neurons (neurons["passing"]; all of
    them if the table has no such column). In a brain-and-cord connectome a connection between
    two neurons that pass through the neck can lie in the brain, which a cord-only source (MANC)
    lacks, so the nerve-cord graphs drop them all (docs/vnc_rare_protocol.md)."""
    if "passing" not in neurons.columns:
        return connections
    ids = neurons[ID_COL].to_numpy()[neurons["passing"].to_numpy(dtype=bool)]
    both = connections[PRE_COL].isin(ids).to_numpy() & connections[POST_COL].isin(ids).to_numpy()
    return connections[~both]


def drop_self_connections(connections: pd.DataFrame) -> pd.DataFrame:
    """The connections without those from a neuron onto itself. MANC, the nerve-cord source,
    has none, so the nerve-cord graphs drop them all (docs/vnc_rare_protocol.md, decision 5)."""
    return connections[connections[PRE_COL].to_numpy() != connections[POST_COL].to_numpy()]


SOURCES = ("fafb", "manc")  # datasets whose label space other graphs can be coded in
CORD_DATASETS = ("manc", "banc_cord", "mcns_cord")
SYNTHETIC_OFFSETS = {
    "fafb": 0,
    "mcns": 1000,
    "banc": 2000,
    "hemibrain": 3000,
    "manc": 4000,
    "banc_cord": 5000,
    "mcns_cord": 6000,
}


def _synthetic(dataset: str, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Synthetic tables standing in for a dataset; for the nerve-cord datasets the neurons of
    class_0 are passing neurons, and connections between two of them are dropped."""
    from flybrain.synthetic import make_synthetic_tables

    neurons, connections = make_synthetic_tables(seed=seed + SYNTHETIC_OFFSETS[dataset])
    if dataset in CORD_DATASETS:
        neurons = neurons.assign(passing=(neurons["super_class"] == "class_0").to_numpy())
        connections = drop_passing_pairs(neurons, connections)
    return neurons, connections


def load_graph(
    label_col: str = "super_class",
    raw_dir: Path = RAW_DIR,
    processed_dir: Path = PROCESSED_DIR,
    synthetic: bool = False,
    seed: int = 0,
    min_class_size: int = 1,
    dataset: str = "fafb",
    mcns_variant: str = "density",
    banc_variant: str = "raw",
    hemibrain_variant: str = "raw",
    retention: float = 1.0,
    retention_seed: int = 0,
    exclude_types: tuple = (),
    exclude_super_classes: tuple = (),
    crop_fraction: float = 0.0,
    source: str = "fafb",
    manc_variant: str = "raw",
    banc_cord_variant: str = "raw",
    mcns_cord_variant: str = "raw",
) -> Data:
    """Load the real graph (raw -> Parquet cache -> Data), or a synthetic one if requested.

    dataset="mcns" loads the male CNS brain (flybrain.mcns, preprocessing `mcns_variant`) with
    its labels coded in the FAFB label space (FAFB classes with >= min_class_size neurons), so
    class indices mean the same in both graphs; dataset="banc" likewise loads the BANC brain
    (flybrain.banc, preprocessing `banc_variant`) and dataset="hemibrain" the hemibrain
    (flybrain.hemibrain, `hemibrain_variant`). With synthetic=True, other synthetic graphs
    stand in for them.

    source="manc" codes the labels in MANC's label space instead (the nerve-cord test,
    docs/vnc_rare_protocol.md), for dataset="manc" (flybrain.manc), "banc_cord" and
    "mcns_cord" (the nerve cords of BANC and MCNS, flybrain.banc / flybrain.mcns, preprocessing
    `banc_cord_variant` / `mcns_cord_variant`). The nerve-cord graphs have no connections
    between two passing neurons (drop_passing_pairs).

    retention < 1, exclude_types, exclude_super_classes or crop_fraction give a less complete
    copy (degrade_tables; for MCNS and BANC before the variant's threshold), labeled in the
    label space of the full source graph.
    """
    if dataset not in SYNTHETIC_OFFSETS:
        raise ValueError(f"dataset must be one of {sorted(SYNTHETIC_OFFSETS)}, got {dataset!r}")
    if source not in SOURCES:
        raise ValueError(f"source must be one of {SOURCES}, got {source!r}")
    degrade = {
        "retention": retention,
        "seed": retention_seed,
        "exclude_types": exclude_types,
        "exclude_super_classes": exclude_super_classes,
        "crop_fraction": crop_fraction,
    }
    full = retention == 1.0 and not exclude_types and not exclude_super_classes
    full = full and crop_fraction == 0.0
    if synthetic:
        src = _synthetic(source, seed)
        if dataset == source and full:
            return build_graph(*src, label_col, min_class_size)
        names = encode_labels(src[0][label_col], min_class_size)[1]
        other = src if dataset == source else _synthetic(dataset, seed)
        other = degrade_tables(*other, **degrade)
        return build_graph(*other, label_col, 1, names)
    if source == "fafb":
        src_neurons, src_connections = load_tables(raw_dir, processed_dir)
    else:
        from flybrain.manc import load_manc_tables

        src_neurons, src_connections = load_manc_tables(
            manc_variant, raw_dir / "manc", processed_dir
        )
    if dataset == source and full:
        return build_graph(src_neurons, src_connections, label_col, min_class_size)
    names = encode_labels(src_neurons[label_col], min_class_size)[1]
    if dataset == "fafb":
        if source == "fafb":
            neurons, connections = src_neurons, src_connections
        else:
            neurons, connections = load_tables(raw_dir, processed_dir)
        neurons, connections = degrade_tables(neurons, connections, **degrade)
    elif dataset == "manc":
        from flybrain.manc import load_manc_tables

        neurons, connections = load_manc_tables(
            manc_variant, raw_dir / "manc", processed_dir, None if full else degrade
        )
    elif dataset == "banc_cord":
        from flybrain.banc import load_banc_cord_tables

        neurons, connections = load_banc_cord_tables(
            banc_cord_variant, raw_dir / "banc", processed_dir, None if full else degrade
        )
    elif dataset == "mcns_cord":
        from flybrain.mcns import load_mcns_cord_tables

        neurons, connections = load_mcns_cord_tables(
            mcns_cord_variant, raw_dir / "mcns", processed_dir, None if full else degrade
        )
    elif dataset == "mcns":
        from flybrain.mcns import load_mcns_tables

        neurons, connections = load_mcns_tables(
            mcns_variant, raw_dir / "mcns", processed_dir, None if full else degrade
        )
    elif dataset == "banc":
        from flybrain.banc import load_banc_tables

        neurons, connections = load_banc_tables(
            banc_variant, raw_dir / "banc", processed_dir, None if full else degrade
        )
    else:
        from flybrain.hemibrain import load_hemibrain_tables

        neurons, connections = load_hemibrain_tables(
            hemibrain_variant, raw_dir / "hemibrain", processed_dir, None if full else degrade
        )
    return build_graph(neurons, connections, label_col, 1, names)
