# Data

Raw data is never committed (`data/raw/` and `data/processed/` are git-ignored), and this repo does not download anything. Obtain the FlyWire (FAFB) tables yourself, for example from FlyWire Codex, and place them in `data/raw/`.

## Expected files (`data/raw/`)

Extensions may be `.parquet`, `.feather`, `.csv` or `.csv.gz`.

| File stem | One row per | Required columns |
|---|---|---|
| `classification` | neuron | `root_id`; labels `super_class` and/or `cell_type` |
| `connections` | (pre, post[, neuropil]) pair | `pre_root_id`, `post_root_id`, `syn_count` |

These filenames and column names are assumptions about the Codex exports. If yours differ, rename the files or edit the constants at the top of `flybrain/loading.py` (`NEURONS_STEM`, `CONNECTIONS_STEM`, `*_COL`).

Check your files:

```bash
uv run python data/download_fafb.py
```

## Processed cache

On first use, `flybrain.loading` cleans the tables (dedupes neurons, sums synapses per pre/post pair, drops unknown ids) and writes `data/processed/neurons.parquet` and `data/processed/connections.parquet`. Delete that folder to rebuild after changing raw files.

## Licenses

FlyWire and MCNS data are released under CC-BY; cite the original publications and keep the attribution when redistributing derived data. Check the license terms of the exact release you use.
