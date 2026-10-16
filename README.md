# flybrain

[![CI](https://github.com/Paradoxsmile/flybrain/actions/workflows/ci.yml/badge.svg)](https://github.com/Paradoxsmile/flybrain/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](pyproject.toml)
[![Data: FlyWire FAFB v783](https://img.shields.io/badge/data-FlyWire%20FAFB%20v783-lightgrey.svg)](https://codex.flywire.ai)

Machine learning on the *Drosophila* connectome: how much of a neuron's identity is encoded in its wiring alone, and what does connectome-constrained modelling of the fly visual system tell us about the role of wiring?

> Status: part b has results on FlyWire (FAFB v783), including a cross-hemisphere test. The MCNS transfer and part c are open.

## Motivation

Whole-brain connectomes (FlyWire/FAFB, Male CNS) list every neuron and synapse of an adult fly brain. Cell types are defined largely by morphology and connectivity, which raises two questions: can a model recover them from connectivity alone, and does the wiring diagram by itself constrain function?

## Research Questions

- **Part b.** Can a GNN predict super class and cell type of a neuron from pure connectivity on FlyWire (FAFB)? Does a model trained on FAFB transfer to the Male CNS (MCNS) connectome?
- **Part c.** Can the connectome-constrained visual-system network of Lappalainen et al. (2024, "flyvis") be reproduced, and how do its predictions change under ablations: real vs. degree-preserving shuffled vs. random wiring?

## Data

FlyWire (FAFB) and MCNS data are released under CC-BY. Raw files are **not** in this repository and must be placed manually in `data/raw/`; see [data/README.md](data/README.md) for expected files and licenses. Check them with `python data/download_fafb.py`.

## Methods

- **Graph:** neurons are nodes; directed edges pre → post, weighted by synapse count (Codex FAFB v783, which keeps connections with ≥ 5 synapses). Raw tables are cleaned once and cached as Parquet.
- **Labels:** `super_class` (10 classes) and `cell_type`, restricted to types with ≥ 20 neurons (434 types, 78 % of neurons).
- **GNN** (`train_gnn.py`, model in `flybrain/models.py`): input features are six degree / synapse-count statistics per neuron, so the model sees anonymous wiring structure and never the labels of other neurons. Each layer averages separately over presynaptic inputs and postsynaptic outputs, weighted by synapse count; this is Dir-SAGE (Rossi et al., 2023) with synapse-weighted means. Configs: `configs/default.yaml` (super_class: 4 layers, 1000 epochs), `configs/cell_type.yaml` (3 layers, 300 epochs, class-weighted loss); depth and options were chosen on validation.
- **Baselines** (`baselines.py`, `baseline_logreg.py`), ordered by the information they use: majority class; logistic regression on a neuron's own degree features; label propagation (connected neurons share a label); partner profile, i.e. logistic regression on the share of a neuron's synapses per partner class, using the labels of training neurons, which is what connectivity-based cell typing compares; NTAC (Schwartzman et al., 2026), which gives each neuron the label of the labeled neuron with the most similar partner-type synapse counts and then re-labels partners iteratively (memory-lean, deterministic reimplementation in `flybrain/ntac.py`, checked step by step against the `ntac` package); plain GraphSAGE (inputs only, unweighted, same depth as the GNN).
- **Splits:** `random` (stratified 70 / 15 / 15) and `left_right` / `right_left` (train and validate on one hemisphere, test on the other).
- **Model selection:** hyperparameters and the training epoch are chosen by validation macro-F1; the test set is only used for the final numbers.
- **Benchmark:** `benchmark.py` runs every method on every label and split for three seeds and writes `results/benchmark/summary.md`. All runs log config, metrics and per-neuron predictions to `results/<run_name>/`.
- **Part c:** plan only, see [experiments/c_constrained_rnn](experiments/c_constrained_rnn/README.md).

## Quickstart

```bash
uv sync
uv run pytest
uv run python experiments/b_gnn_celltypes/baseline_logreg.py --synthetic
uv run python experiments/b_gnn_celltypes/train_gnn.py --synthetic
uv run python experiments/b_gnn_celltypes/baselines.py --method partner_profile --synthetic
```

Full comparison on the real data (several hours on CPU; resumable, and can be split with `--labels` / `--methods`):

```bash
uv run python experiments/b_gnn_celltypes/benchmark.py
```

CPU works by default; CUDA is used automatically when available.

## Results

FlyWire FAFB v783, test accuracy (%) / macro-F1, mean ± std over 3 seeds, produced by `benchmark.py` (baselines on CPU at commit `2000b7b`, NTAC on CPU at `c54198c`, GNN rows on a Kaggle T4 GPU at `343125f`). Hemisphere columns: trained on the first hemisphere, tested on the second.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/benchmark_dark.png">
  <img alt="Test accuracy per method and split. The GNN stays at 87 to 98 percent on every split. Methods that read partner labels (partner profile, NTAC) reach 91 to 97 percent on the random split but fall to 1 to 9 percent on cell types and 46 to 82 percent on super classes when tested on the unlabeled hemisphere." src="docs/benchmark_light.png">
</picture>

Each dot is one split (mean over seeds, whiskers = std); the grey line spans a method's best and worst split. Figure: `experiments/b_gnn_celltypes/plot_benchmark.py`.

**cell_type** (434 types)

| Method | random | left → right | right → left |
|---|---|---|---|
| Majority class | 7.8 / 0.00 | 7.5 / 0.00 | 8.1 / 0.00 |
| Logistic regression (own degrees) | 40.9 ± 0.2 / 0.12 | 37.3 ± 0.0 / 0.09 | 39.0 ± 0.1 / 0.11 |
| Label propagation | 14.7 ± 0.2 / 0.28 | 10.7 ± 1.0 / 0.19 | 10.3 ± 1.0 / 0.20 |
| Partner profile (partner labels) | 92.2 ± 0.9 / 0.79 | 8.7 ± 0.0 / 0.10 | 1.2 ± 0.0 / 0.10 |
| NTAC (partner labels, iterative) | **97.0 ± 0.2 / 0.93** | 8.4 ± 0.1 / 0.20 | 8.8 ± 0.5 / 0.19 |
| GraphSAGE | 87.1 ± 0.3 / 0.77 | 85.2 ± 0.3 / 0.72 | 79.2 ± 0.2 / 0.69 |
| **GNN (inputs + outputs, weighted)** | 95.3 ± 0.2 / **0.93** | **94.5 ± 0.9 / 0.90** | **86.7 ± 1.0 / 0.87** |

**super_class** (10 classes)

| Method | random | left → right | right → left |
|---|---|---|---|
| Majority class | 55.9 / 0.07 | 55.9 / 0.07 | 56.1 / 0.07 |
| Logistic regression (own degrees) | 67.6 ± 0.4 / 0.31 | 67.2 ± 0.0 / 0.32 | 66.7 ± 0.0 / 0.32 |
| Label propagation | 80.1 ± 0.1 / 0.30 | 73.4 ± 7.8 / 0.22 | 72.6 ± 8.8 / 0.21 |
| Partner profile (partner labels) | 90.9 ± 0.1 / 0.60 | 47.6 ± 3.0 / 0.28 | 45.7 ± 3.0 / 0.23 |
| NTAC (partner labels, iterative) | 94.4 ± 0.3 / 0.78 | 78.4 ± 2.1 / 0.49 | 82.4 ± 3.8 / 0.58 |
| GraphSAGE | 95.3 ± 0.0 / 0.62 | 95.1 ± 0.3 / 0.63 | 94.4 ± 0.2 / 0.63 |
| **GNN (inputs + outputs, weighted)** | **97.9 ± 0.1 / 0.85** | **97.9 ± 0.1 / 0.84** | **96.6 ± 0.0 / 0.76** |

Findings:

- **With labeled partners, partner identity is hard to beat; anonymous wiring comes close.** On the random split NTAC is the best cell-type method (97.0 vs. 95.3 % for the GNN, equal macro-F1 0.93), while the GNN is best on super classes (97.9 vs. 94.4 %). The GNN gets there without seeing any partner's label.
- **Only the anonymous model works on an unlabeled hemisphere.** In the hemisphere splits the median test neuron has no synapses to a labeled partner (optic-lobe circuits stay on their side), so the partner profile and NTAC collapse on cell types (1–9 %; NTAC stays below 9.4 % even at its best of 40 steps), while the GNN keeps 87–95 %. On super classes NTAC keeps 78–82 % with its validation-chosen step count and 91 % at its best step, still below the GNN's 97–98 %. Labels do spread across the midline, but only through the few cross-hemisphere connections.
- **Seeing outputs matters most.** Averaging over postsynaptic partners as well as inputs adds 2–9 points accuracy and 0.13–0.23 macro-F1 over GraphSAGE of the same depth in every column. In the ablation (`ablation.py`), synapse weighting helps only cell types, and class weighting (with bidirectional aggregation) helps cell types but hurts super classes.
- **Depth helps up to a point.** A third layer mainly separates sibling optic-lobe types (T4 / T5 subtypes, which differ in the motion direction they encode); a fourth helps only super classes, mostly their rare classes. Deeper models were not better on validation.
- **Homophily explains little.** 71 % of synapses connect neurons of the same super class (35 % by chance), but only 3 % connect neurons of the same cell type (1.3 % by chance): the brain is wired heterophilously at the type level, so label propagation fails on cell types.
- **Reconstruction differences hurt.** Right → left cell typing is 8 points below left → right, with 2 or 3 layers alike. The hemispheres differ in the number of connections per neuron for several optic-lobe types, and the errors concentrate there (photoreceptors R1-6 and lamina neurons); outgoing photoreceptor synapses are known to be strongly underdetected in FlyWire (Matsliah et al., 2024). Per-hemisphere feature normalization and edge-dropping augmentation (`feature_norm`, `drop_edge`) did not close the gap, and neither did more depth.

Related work and how this project relates to it: [docs/literature.md](docs/literature.md).

## Limitations

- Cell types were defined partly from connectivity, including matching across hemispheres and brains (Schlegel et al., 2024), so recovering them from wiring is partly circular. The GNN never sees partner identities, which weakens this concern but does not remove it. Super classes are mostly defined anatomically.
- Only types with ≥ 20 neurons are predicted (434 of 8,772 types); the many rare types are left out.
- The setting is transductive: the whole graph is visible during training, only the labels of validation and test neurons are hidden.
- One animal. The hemisphere split shows sensitivity to reconstruction completeness, so a transfer to another dataset (MCNS) is expected to be harder.
- Each seed changes both the data split and the initialization; the std covers both.
- The cell-type GNN is trained for 300 epochs and its best epoch is close to the end, so it may be slightly underestimated.
- Codex drops connections with fewer than 5 synapses.
- FAFB and MCNS differ in proofreading state, synapse detection and annotation schemes; transfer results depend on how neurons and classes are aligned.
- The synthetic graph is for testing only and says nothing about biology.

## Citation

Citation metadata are in [CITATION.cff](CITATION.cff) (GitHub's "Cite this repository" button), or as BibTeX:

```bibtex
@misc{vogt2026flybrain,
  author = {Vogt, Maximilian},
  title  = {flybrain: ML on the Drosophila connectome},
  year   = {2026},
  url    = {https://github.com/paradoxsmile/flybrain}
}
```

Please also cite the data sources (FlyWire / FAFB, MCNS) and Lappalainen et al. (2024), *Connectome-constrained networks predict neural activity across the fly visual system*, Nature.

## License

Code: MIT (see [LICENSE](LICENSE)). Data: see the licenses of the respective sources (CC-BY).
