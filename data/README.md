# Data

Raw data is never committed (the contents of `data/raw/` and `data/processed/` are git-ignored), and this repo does not download anything. Download the FlyWire (FAFB) tables yourself from FlyWire Codex (release **v783**) and place them in `data/raw/` with their original filenames.

## Expected files (`data/raw/`)

| Codex file (v783) | One row per | Used columns | Required |
|---|---|---|---|
| `classification.csv.gz` | neuron | `root_id`, `super_class` (also `flow`, `class`, `sub_class`, `hemilineage`, `side`, `nerve`) | yes |
| `connections_princeton.csv.gz` | (pre, post, neuropil) | `pre_root_id`, `post_root_id`, `syn_count` (also `neuropil`, `nt_type`) | yes |
| `consolidated_cell_types.csv.gz` | typed neuron | `root_id`, `primary_type` | optional |

`classification` has no `cell_type` column; if `consolidated_cell_types` is present, its `primary_type` is left-merged on `root_id` as `cell_type` (untyped neurons get no label). Without it only `super_class` labels are available.

Extensions may also be `.parquet`, `.feather` or `.csv`, and `connections.*` is accepted instead of `connections_princeton.*`. If your exports differ, edit the constants at the top of `flybrain/loading.py` (`NEURONS_STEM`, `CONNECTIONS_STEM`, `CELL_TYPES_STEM`, `*_COL`).

Check your files:

```bash
uv run python data/download_fafb.py
```

## Processed cache

On first use, `flybrain.loading` cleans the tables (dedupes neurons, sums synapses per pre/post pair, drops unknown ids) and writes `data/processed/neurons.parquet` and `data/processed/connections.parquet`. Delete that folder to rebuild after changing raw files.

## Licenses

FlyWire and MCNS data are released under CC-BY; cite the original publications and keep the attribution when redistributing derived data. Check the license terms of the exact release you use.
