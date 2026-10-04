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


def build_graph(
    neurons: pd.DataFrame, connections: pd.DataFrame, label_col: str = "super_class"
) -> Data:
    """Build a PyG Data object. y is the integer label (-1 if unlabeled).

    Attributes: x (degree features), edge_index, edge_weight, y, node_ids, label_names.
    """
    neurons, connections = clean_tables(neurons, connections)
    index = pd.Series(np.arange(len(neurons)), index=neurons[ID_COL])
    src = index[connections[PRE_COL]].to_numpy()
    dst = index[connections[POST_COL]].to_numpy()
    edge_index = torch.from_numpy(np.stack([src, dst])).long()
    edge_weight = torch.tensor(connections[WEIGHT_COL].to_numpy(), dtype=torch.float)

    codes, names = pd.factorize(neurons[label_col].astype("string"))  # NA -> -1
    x = torch.from_numpy(degree_features(len(neurons), edge_index, edge_weight)).float()
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_weight=edge_weight,
        y=torch.from_numpy(codes).long(),
        num_nodes=len(neurons),
    )
    data.node_ids = torch.from_numpy(neurons[ID_COL].to_numpy().copy())
    data.label_names = [str(n) for n in names]
    return data


def load_graph(
    label_col: str = "super_class",
    raw_dir: Path = RAW_DIR,
    processed_dir: Path = PROCESSED_DIR,
    synthetic: bool = False,
    seed: int = 0,
) -> Data:
    """Load the real graph (raw -> Parquet cache -> Data), or a synthetic one if requested."""
    if synthetic:
        from flybrain.synthetic import make_synthetic_tables

        return build_graph(*make_synthetic_tables(seed=seed), label_col)
    neurons, connections = load_tables(raw_dir, processed_dir)
    return build_graph(neurons, connections, label_col)
