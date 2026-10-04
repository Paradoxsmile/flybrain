"""Seeding, device selection and run logging."""

from __future__ import annotations

import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
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
