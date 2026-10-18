"""Seeding, device selection and run logging."""

from __future__ import annotations

import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """CUDA if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_run_dir(run_name: str | None, prefix: str, results_dir: Path | None = None) -> Path:
    """Create results/<run_name>/ (timestamped default name)."""
    name = run_name or f"{prefix}_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir = (results_dir or REPO_ROOT / "results") / name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def log_run(run_dir: Path, config: dict[str, Any], metrics: dict[str, Any]) -> None:
    """Write config.yaml and metrics.json into the run directory."""
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=True))
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))


def write_predictions(
    path: Path, data, pred: np.ndarray, masks: tuple, extra: dict | None = None
) -> None:
    """Write one row per neuron: root_id, side (if known), true and predicted label, split.

    pred: predicted class index per neuron; masks: boolean (train, val, test) tensors;
    extra: further labelings (column name -> class index per neuron), written as names.
    """
    names = np.array(list(data.label_names) + [None], dtype=object)  # index -1 -> None
    split = np.full(data.num_nodes, "", dtype=object)
    for name, mask in zip(("train", "val", "test"), masks, strict=True):
        split[mask.cpu().numpy()] = name
    df = pd.DataFrame(
        {
            "root_id": data.node_ids.cpu().numpy(),
            "label": names[data.y.cpu().numpy()],
            "pred": names[np.asarray(pred)],
            "split": split,
        }
    )
    for col, labeling in (extra or {}).items():
        df[col] = names[np.asarray(labeling)]
    if getattr(data, "side", None) is not None:
        df.insert(1, "side", data.side)
    df.to_parquet(path, index=False)
