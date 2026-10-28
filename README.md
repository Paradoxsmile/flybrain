# flybrain

[![CI](https://github.com/Paradoxsmile/flybrain/actions/workflows/ci.yml/badge.svg)](https://github.com/Paradoxsmile/flybrain/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](pyproject.toml)
[![Data: FlyWire FAFB v783](https://img.shields.io/badge/data-FlyWire%20FAFB%20v783-lightgrey.svg)](https://codex.flywire.ai)

Typing the neurons of an unlabeled *Drosophila* connectome from wiring alone. A graph neural network sees only anonymous connectivity, never a partner's label, and is transferred from an annotated connectome to a new one without any target label. For the rarest cell types (2–4 neurons, about three quarters of FlyWire's types), the transferred GNN selects the candidate neurons and graph matching against the reference types them. The approach was tested on five connectomes, four of them under protocols committed before any run.

Paper draft: [paper/draft.md](paper/draft.md). Every experiment in detail: [docs/results.md](docs/results.md). Command and result file for every number: [docs/reproduce.md](docs/reproduce.md).

## Main results

**Rare types: a transferred GNN selects, graph matching types.** Accuracy % on neurons of 2–4-neuron source types, without any target label (pre-registered tests in bold):

| Source → target | Stage 1: GNN alone | Two-stage | Oracle candidates (uses labels) | Data |
|---|---|---|---|---|
| FlyWire left → right hemisphere | 78.9 | 85.8 | 92.2 | development |
| FlyWire → male CNS (MCNS) brain | 54.2 | 74.5; 81.6 with `union` candidates | 85.3 | development |
| **FlyWire → BANC brain** | **22.2** | **41.3** (morphology reference: 40.6) | 65.9 | [protocol](docs/banc_rare_protocol.md) |
| **MANC → BANC nerve cord** | **39.5** | **61.0** with `union` (curated reference: 61.0) | 75.5 | [protocol](docs/vnc_rare_protocol.md) |
| **MANC → MCNS nerve cord** | **72.0** | **82.2** with `union` | 87.0 | [protocol](docs/vnc_rare_protocol.md) |

The left → right stage 1 is the GNN's majority vote over three seeds. Of the pre-registered expectations, all four held on the BANC brain and two of four on both nerve cords. A gain of +15 points was out of reach on MCNS, whose stage 1 is already at 72 %. A two-tier fallback that had helped on the brain lowered accuracy on both cords and was dropped.

**Common types: wiring alone, and label-free transfer.** Cell types with ≥ 20 FlyWire neurons (434 types), accuracy %:

| Setting | GNN (wiring only) | Comparison |
|---|---|---|
| FlyWire, random split | 95.3 | NTAC with labeled partners: 97.0 |
| FlyWire, unlabeled hemisphere (left → right) | 94.5 | NTAC: 8.4; partner profile: 8.7 |
| FlyWire → MCNS, label-free (394 types) | 92.6 (optic lobe 97.2, all other neurons 74.0) | trained on MCNS labels: 96.6 |
| MCNS → FlyWire, label-free | 83.4 | trained on FlyWire labels: 95.2 |
| **FlyWire → BANC brain**, label-free | 61.7 (central brain, morphology reference: 32.3) | MCNS → BANC: 1.4 (the original rule chose a collapsed labeling) |
| **FlyWire → hemibrain** (partial volume), label-free | 19.6 | best candidate, post hoc: 25.5 |

- Label-free transfer works between complete connectomes and keeps 84.2 % with 7,064 types down to two neurons per type (FlyWire → MCNS). Its accuracy comes mostly from the optic lobe.
- Prior matching of predicted type sizes carried over to every complete or sparse target (+3.6 to +17.5 points).
- The original label-free selection rule failed on the sparse BANC brain. A rule revised on a sandbox of degraded development connectomes held afterwards.
- Reconstruction density explains about half of the loss on BANC. Partial volumes such as the hemibrain cannot be typed this way.
- With hand labels, NTAC overtakes the GNN on common types at about 1.3 % of each type labeled. The model is most useful with no labels, for the rare types a labeling budget reaches last, and for directing review: the 20 % least-agreed neurons hold 77 % of the errors on MCNS.

## Pipeline

1. **GNN** (`flybrain/models.py`, `train_gnn.py`): Dir-SAGE (Rossi et al., 2023) with synapse-weighted means over inputs and outputs, on six degree and synapse-count features per neuron.
2. **Label-free transfer** (`label_free.py`; `lean_label_free.py` for thousands of types):
   - the target is built in a few alignment variants, and features are normalized by source or target statistics;
   - prior matching rescales the predicted type sizes to the reference (`flybrain/adapt.py`);
   - a label-free rule chooses among the candidates: type sizes plus seed agreement, `--select size_agreement` (`flybrain/selection.py`);
   - NTAC refines the result, seeded where three GNNs agree (`flybrain/ntac.py`).
3. **Rare types** (`graph_matching_mcns.py`):
   - candidates are the target neurons whose stage-1 type is a 2–4-neuron type before or after NTAC (`--candidates union`);
   - graph matching against the source's rare-type neurons uses Frank–Wolfe on the relaxed quadratic assignment problem, blocked by hemisphere and with ipsilateral and contralateral edges;
   - in the nerve cord, a third block holds the neurons that pass through the neck (`--passing-block`);
   - matched candidates take the matched neuron's type.

Pre-registration: each test on a new dataset was fixed in a protocol before any run ([BANC](docs/banc_protocol.md), [hemibrain](docs/hemibrain_protocol.md), [BANC rare types](docs/banc_rare_protocol.md), [nerve cords](docs/vnc_rare_protocol.md)). Between tests, methods were changed only on degraded copies of the development connectomes.

## Quickstart

```bash
uv sync
uv run pytest
uv run python experiments/b_gnn_celltypes/train_gnn.py --synthetic
uv run python experiments/b_gnn_celltypes/baselines.py --method ntac --synthetic
```

On the real data, the benchmark takes several hours on a CPU (resumable; split it with `--labels` / `--methods`). CUDA is used automatically when available:

```bash
uv run python experiments/b_gnn_celltypes/benchmark.py
```

The commands for every other result are in [docs/reproduce.md](docs/reproduce.md).

## Data

FlyWire (FAFB v783), MCNS v1.0, BANC v888, the hemibrain v1.2.1 and MANC v1.2.1 are public under CC BY. Raw files are **not** in this repository; place them in `data/raw/` as described in [data/README.md](data/README.md). `python data/download_fafb.py` checks the FlyWire files. The small summaries of the reported runs are committed under `results/`; models and predictions are not.

## Repository

| Path | Contents |
|---|---|
| `flybrain/` | data loading (`loading.py`, `mcns.py`, `banc.py`, `hemibrain.py`, `manc.py`), model and training, prior matching (`adapt.py`), label-free scores (`consistency.py`, `selection.py`, `coverage.py`), NTAC, thinning |
| `experiments/b_gnn_celltypes/` | one script per experiment: benchmark and baselines, transfer, label-free transfer, type-size sweep, graph matching, candidate selection, labeling effort, accuracy estimation, nerve-cord test |
| `docs/` | [results in detail](docs/results.md), [reproduction map](docs/reproduce.md), the four protocols, development plans, [literature](docs/literature.md) |
| `paper/` | draft and figures (`make_figures.py`) |
| `results/` | committed summaries of the reported runs |
| `tests/` | unit and smoke tests (`uv run pytest`) |

## Limitations

- **Circularity.** Cell types were defined partly from connectivity, and most cross-dataset references come from matching that uses connectivity. The GNN never sees partner identities, which weakens but does not remove this concern. The rare-type gain is the same against BANC's morphology-matched and curated references.
- **Scope of the rare-type tests.** The rare-type pipeline was developed on one brain pair and tested on one held-out brain and two nerve cords. Single-neuron types are not predicted. The matching is one-to-one, while type sizes differ between animals.
- **Label-free transfer.** It was developed on FlyWire → MCNS while looking at its errors. It degrades with reconstruction density and fails for partial volumes. Run-level accuracy cannot be estimated without labels.
- **Setting.** It is transductive within a connectome. Most common-type results use types with ≥ 20 FlyWire neurons; we recommend ≥ 10 for new results.

More caveats are listed in [docs/results.md](docs/results.md#limitations).

## Possible follow-up

Reproducing the connectome-constrained visual-system model of Lappalainen et al. (2024, "flyvis") and testing it under shuffled wiring was planned as a part c. It is not part of this project.

## Citation

Citation metadata are in [CITATION.cff](CITATION.cff) (GitHub's "Cite this repository" button), or as BibTeX:

```bibtex
@misc{vogt2026flybrain,
  author = {Vogt, Maximilian},
  title  = {flybrain: label-free cell typing across fly connectomes from wiring alone},
  year   = {2026},
  url    = {https://github.com/paradoxsmile/flybrain}
}
```

Please also cite the data sources: FlyWire (Dorkenwald et al., 2024; Schlegel et al., 2024), MCNS (Berg et al., 2026), BANC (Bates et al., 2026), the hemibrain (Scheffer et al., 2020) and MANC (Takemura et al., 2024; Marin et al., 2024).

## License

Code: MIT (see [LICENSE](LICENSE)). Data: see the licenses of the respective sources (CC BY).
