"""Step D2 of the nerve-cord test (docs/vnc_test_plan.md): build the cord caches and compute the
label-free graph statistics that fix the target variants (docs/vnc_rare_protocol.md).

Uses no labels. For MANC (the source) and the BANC and MCNS nerve cords:
- local MD5s (base64) of the raw files;
- neurons, passing neurons, and at threshold 5 the connections and synapses per neuron and the
  synapses per connection, after dropping the connections between two passing neurons;
- the share of connections and synapses (threshold 5) that dropping those pairs removes;
- for each target, the variants by the BANC rules, relative to MANC at threshold 5:
  "density" = the threshold (1-20) whose connections per neuron are closest to MANC's (left
  out if it is 5); "scaled" = threshold 5, counts x MANC's synapses per neuron / the target's.

Writes results/vnc_rare/d2_graph_stats.json. The variants are then copied into
flybrain.banc.CORD_VARIANTS and flybrain.mcns.CORD_VARIANTS and the protocol.

Usage: python experiments/b_gnn_celltypes/vnc_data.py
"""

from __future__ import annotations

import base64
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from flybrain import banc, manc, mcns
from flybrain.loading import POST_COL, PRE_COL, RAW_DIR, WEIGHT_COL, drop_passing_pairs
from flybrain.utils import REPO_ROOT

OUT = REPO_ROOT / "results" / "vnc_rare"
THRESHOLD = 5  # the source graph's threshold (FAFB's convention)
THRESHOLDS = range(1, 21)
RAW_FILES = {
    "manc": [RAW_DIR / "manc" / manc.META_FILE, RAW_DIR / "manc" / manc.EDGES_FILE],
    "banc_cord": [RAW_DIR / "banc" / banc.META_FILE, RAW_DIR / "banc" / banc.EDGES_FILE],
    "mcns_cord": [RAW_DIR / "mcns" / mcns.ANNOTATIONS_FILE, RAW_DIR / "mcns" / mcns.WEIGHTS_FILE],
}
CACHES = {
    "manc": manc.manc_cache,
    "banc_cord": banc.banc_cord_cache,
    "mcns_cord": mcns.mcns_cord_cache,
}


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return base64.b64encode(h.digest()).decode()


def passing_pairs(neurons: pd.DataFrame, connections: pd.DataFrame) -> np.ndarray:
    """Boolean mask of the connections between two passing neurons."""
    return ~connections.index.isin(drop_passing_pairs(neurons, connections).index)


def per_neuron(neurons: pd.DataFrame, connections: pd.DataFrame, t: int) -> tuple[float, float]:
    """(connections, synapses) per neuron at threshold t, without passing pairs."""
    c = connections[connections[WEIGHT_COL] >= t]
    c = drop_passing_pairs(neurons, c)
    return len(c) / len(neurons), float(c[WEIGHT_COL].sum()) / len(neurons)


def stats(neurons: pd.DataFrame, connections: pd.DataFrame) -> dict:
    c5 = connections[connections[WEIGHT_COL] >= THRESHOLD].reset_index(drop=True)
    pp = passing_pairs(neurons, c5)
    kept = c5[~pp]
    n = len(neurons)
    return {
        "neurons": n,
        "passing_neurons": int(neurons["passing"].sum()),
        "connections_per_neuron": len(kept) / n,
        "synapses_per_neuron": float(kept[WEIGHT_COL].sum()) / n,
        "synapses_per_connection": float(kept[WEIGHT_COL].mean()),
        "passing_pair_share_connections": float(pp.mean()),
        "passing_pair_share_synapses": float(c5[WEIGHT_COL][pp].sum() / c5[WEIGHT_COL].sum()),
        "cache_connections": len(connections),
        "self_connections": int((connections[PRE_COL] == connections[POST_COL]).sum()),
    }


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    out = {"threshold": THRESHOLD, "md5": {}, "graphs": {}, "variants": {}}
    tables = {}
    for name, files in RAW_FILES.items():
        t = time.time()
        out["md5"][name] = {f.name: md5(f) for f in files}
        tables[name] = CACHES[name]()  # builds the cache on first use
        out["graphs"][name] = stats(*tables[name])
        print(name, f"{time.time() - t:.0f} s", out["md5"][name], out["graphs"][name], flush=True)
    ref = out["graphs"]["manc"]
    for name in ("banc_cord", "mcns_cord"):
        neurons, connections = tables[name]
        curve = {t: per_neuron(neurons, connections, t) for t in THRESHOLDS}
        density = min(THRESHOLDS, key=lambda t: abs(curve[t][0] - ref["connections_per_neuron"]))
        scale = ref["synapses_per_neuron"] / curve[THRESHOLD][1]
        variants = {"raw": [THRESHOLD, 1.0]}
        if density != THRESHOLD:
            variants["density"] = [density, 1.0]
        variants["scaled"] = [THRESHOLD, round(scale, 2)]
        out["variants"][name] = variants
        out["graphs"][name]["connections_per_neuron_by_threshold"] = {
            str(t): curve[t][0] for t in THRESHOLDS
        }
        print(name, "variants", variants, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "d2_graph_stats.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out["variants"], indent=2))


if __name__ == "__main__":
    main()
