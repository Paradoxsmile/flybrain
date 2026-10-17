# flybrain

[![CI](https://github.com/Paradoxsmile/flybrain/actions/workflows/ci.yml/badge.svg)](https://github.com/Paradoxsmile/flybrain/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](pyproject.toml)
[![Data: FlyWire FAFB v783](https://img.shields.io/badge/data-FlyWire%20FAFB%20v783-lightgrey.svg)](https://codex.flywire.ai)

Machine learning on the *Drosophila* connectome: how much of a neuron's identity is encoded in its wiring alone, and what does connectome-constrained modelling of the fly visual system tell us about the role of wiring?

> Status: part b has results on FlyWire (FAFB v783), including a cross-hemisphere test, a replication within the male CNS (MCNS v1.0) and a fully label-free FAFB → MCNS transfer (92.6 % over 394 cell types; 83.4 % when run unchanged from MCNS to FAFB), and an independent test on a third connectome (BANC) with a protocol fixed in advance, where part of the pipeline did not hold up. Part c is open.

## Motivation

Whole-brain connectomes (FlyWire/FAFB, Male CNS) list every neuron and synapse of an adult fly brain. Cell types are defined largely by morphology and connectivity, which raises two questions: can a model recover them from connectivity alone, and does the wiring diagram by itself constrain function?

## Research Questions

- **Part b.** Can a GNN predict super class and cell type of a neuron from pure connectivity on FlyWire (FAFB)? Does a model trained on FAFB transfer to the Male CNS (MCNS) connectome?
- **Part c.** Can the connectome-constrained visual-system network of Lappalainen et al. (2024, "flyvis") be reproduced, and how do its predictions change under ablations: real vs. degree-preserving shuffled vs. random wiring?

## Data

FlyWire (FAFB) and MCNS data are released under CC-BY. Raw files are **not** in this repository and must be placed manually in `data/raw/` (MCNS in `data/raw/mcns/`); see [data/README.md](data/README.md) for expected files and licenses. Check them with `python data/download_fafb.py`.

## Methods

- **Graph:** neurons are nodes; directed edges pre → post, weighted by synapse count (Codex FAFB v783, which keeps connections with ≥ 5 synapses). Raw tables are cleaned once and cached as Parquet.
- **Labels:** `super_class` (10 classes) and `cell_type`, restricted to types with ≥ 20 neurons (434 types, 78 % of neurons).
- **GNN** (`train_gnn.py`, model in `flybrain/models.py`): input features are six degree / synapse-count statistics per neuron, so the model sees anonymous wiring structure and never the labels of other neurons. Each layer averages separately over presynaptic inputs and postsynaptic outputs, weighted by synapse count; this is Dir-SAGE (Rossi et al., 2023) with synapse-weighted means. Configs: `configs/default.yaml` (super_class: 4 layers, 1000 epochs), `configs/cell_type.yaml` (3 layers, 300 epochs, class-weighted loss); depth and options were chosen on validation.
- **Baselines** (`baselines.py`, `baseline_logreg.py`), ordered by the information they use: majority class; logistic regression on a neuron's own degree features; label propagation (connected neurons share a label); partner profile, i.e. logistic regression on the share of a neuron's synapses per partner class, using the labels of training neurons, which is what connectivity-based cell typing compares; NTAC (Schwartzman et al., 2026), which gives each neuron the label of the labeled neuron with the most similar partner-type synapse counts and then re-labels partners iteratively (memory-lean, deterministic reimplementation in `flybrain/ntac.py`, checked step by step against the `ntac` package); plain GraphSAGE (inputs only, unweighted, same depth as the GNN).
- **Splits:** `random` (stratified 70 / 15 / 15) and `left_right` / `right_left` (train and validate on one hemisphere, test on the other).
- **Model selection:** hyperparameters and the training epoch are chosen by validation macro-F1; the test set is only used for the final numbers.
- **Male CNS** (`flybrain/mcns.py`): the traced brain neurons of MCNS v1.0 (no ventral nerve cord), with MCNS superclasses mapped to the FAFB super classes and `flywireType` to FAFB cell types (394 of the 434 types occur; entries listing several types are left out). MCNS detects more synapses per connection, so its graph is built in three alignments: `density` (>= 7 synapses, about FAFB's number of connections, counts × 0.581 as in Janelia's FlyWire comparison), `raw` (FAFB's threshold of 5, no scaling) and `janelia` (counts × 0.581, then threshold 5).
- **Transfer** (`transfer.py`): methods are trained on FAFB as in the benchmark and applied to MCNS without any MCNS label, with features normalized by FAFB statistics (`source`) or by MCNS's own (`target`). `--source mcns` trains on MCNS (one alignment per run) and applies to FAFB.
- **GNN-seeded NTAC** (`hybrid.py`): neurons on which three GNNs (seeds) agree become NTAC seeds, together with any truly known labels; NTAC labels the rest. Nothing is tuned (unanimous predictions; NTAC defaults: 15 steps, learning rate 0.3).
- **Label-free transfer** (`label_free.py`): every transfer choice is made without labels of the target connectome. Type sizes are matched to the reference (`flybrain/adapt.py`), and the alignment, normalization and matching strength are chosen by a label-free type-consistency score (`flybrain/consistency.py`); see below. `--source mcns --target fafb` runs the same pipeline from MCNS to FAFB, `--target banc` to BANC (protocol fixed in advance: [docs/banc_protocol.md](docs/banc_protocol.md)).
- **Benchmark:** `benchmark.py` runs every method on every label and split for three seeds and writes `results/benchmark/summary.md` (`--dataset mcns`: the same within MCNS). All runs log config, metrics and per-neuron predictions to `results/<run_name>/`.
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
uv run python experiments/b_gnn_celltypes/benchmark.py --dataset mcns
uv run python experiments/b_gnn_celltypes/transfer.py
```

CPU works by default; CUDA is used automatically when available.

## Results

FlyWire FAFB v783, test accuracy (%) / macro-F1, mean ± std over 3 seeds, produced by `benchmark.py` (baselines on CPU at commit `2000b7b`, NTAC on CPU at `bfbd103`, GNN rows on a Kaggle T4 GPU at `343125f`). Hemisphere columns: trained on the first hemisphere, tested on the second.

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

### Replication within the male CNS

The same comparison within MCNS (trained and tested on MCNS; NTAC, logistic regression and majority on CPU, GNN rows on a Kaggle T4 GPU, both at `4ec2a8a`):

| Method | cell types: random | left → right | right → left | super classes: random | left → right | right → left |
|---|---|---|---|---|---|---|
| Logistic regression (own degrees) | 40.5 / 0.16 | 35.1 / 0.14 | 35.8 / 0.13 | 69.9 / 0.29 | 68.9 / 0.27 | 69.4 / 0.28 |
| NTAC (partner labels, iterative) | **97.3 / 0.93** | 7.8 / 0.16 | 9.5 / 0.22 | 95.0 / 0.78 | 80.0 / 0.54 | 82.4 / 0.56 |
| GraphSAGE (inputs only) | 87.5 / 0.78 | 85.1 / 0.74 | 82.4 / 0.73 | 95.5 / 0.74 | 94.6 / 0.74 | 94.7 / 0.74 |
| **GNN (inputs + outputs, weighted)** | 96.6 / **0.95** | **95.4 / 0.93** | **93.7 / 0.91** | **98.4 / 0.88** | **97.6 / 0.86** | **97.9 / 0.86** |

The hemisphere result replicates in a second animal: NTAC collapses on the unlabeled hemisphere, the GNN does not. Unlike FAFB, the two directions differ by less than 2 points, so the 8-point right → left gap in FAFB is specific to that reconstruction.

### Transfer: trained on FAFB, tested on the male CNS

No MCNS label is used. Methods that read partner labels (partner profile, NTAC) cannot be applied, because no MCNS neuron is labeled. Mean ± std over 3 seeds (`transfer.py` at `4ec2a8a`, GNN rows on a Kaggle T4 GPU).

| Method | FAFB test | MCNS, `density`, target norm. | MCNS, `raw`, target norm. | MCNS, `janelia`, target norm. | MCNS, `raw`, source norm. (best, post hoc) |
|---|---|---|---|---|---|
| **cell types** | | | | | |
| Logistic regression (own degrees) | 41.0 ± 0.2 / 0.12 | 19.0 ± 0.1 / 0.04 | 21.2 ± 0.1 / 0.05 | 15.3 ± 0.1 / 0.04 | 26.8 ± 0.2 / 0.08 |
| GraphSAGE (inputs only) | 87.2 ± 0.3 / 0.77 | 58.8 ± 0.5 / 0.36 | 63.7 ± 1.1 / 0.37 | 45.3 ± 0.6 / 0.25 | 75.4 ± 0.2 / 0.49 |
| **GNN (inputs + outputs, weighted)** | 95.2 ± 0.1 / 0.93 | **73.4 ± 0.5 / 0.50** | **74.5 ± 0.6 / 0.49** | **65.3 ± 1.6 / 0.47** | **85.7 ± 1.1 / 0.66** |
| **super classes** | | | | | |
| Majority class | 55.9 / 0.07 | 62.0 / 0.08 | 62.0 / 0.08 | 62.0 / 0.08 | 62.0 / 0.08 |
| Logistic regression (own degrees) | 67.6 ± 0.4 / 0.31 | 58.0 ± 0.2 / 0.19 | 58.8 ± 0.1 / 0.19 | 57.5 ± 0.1 / 0.19 | 68.1 ± 0.0 / 0.29 |
| GraphSAGE (inputs only) | 95.1 ± 0.2 / 0.64 | 82.2 ± 2.5 / 0.43 | 87.5 ± 1.9 / 0.50 | 75.9 ± 3.1 / 0.34 | 93.4 ± 0.2 / 0.51 |
| **GNN (inputs + outputs, weighted)** | 97.8 ± 0.1 / 0.84 | **94.0 ± 0.6 / 0.68** | **95.4 ± 0.3 / 0.70** | **90.5 ± 1.4 / 0.59** | **96.2 ± 0.3 / 0.68** |

- **The raw transfer depends strongly on preprocessing:** 65–86 % on cell types, 90–96 % on super classes, compared with 96.6 % / 98.4 % when trained within MCNS. Matching feature distributions does not tell the alignments apart (mean 1-D Wasserstein distance 0.195–0.203 for all MCNS-normalized variants). The label-free pipeline below resolves this.
- **Aggregating over outputs matters more under transfer:** the GNN is 10–20 points above GraphSAGE on cell types.
- **Where transfer fails:** optic-lobe types transfer well (neuron-weighted F1 0.81 vs. 0.97 within MCNS; `density`, target norm., seed 0), central-brain types poorly (0.30 vs. 0.95). Several lobula columnar types (LC9, LC12, LC21) and the Kenyon cells KCab (predicted as KCg-m) are lost completely; sibling types such as T4/T5 subtypes and Dm3p/q are confused.

### Label-free transfer: type consistency and type-size priors

Two observations make every transfer choice possible without MCNS labels (`label_free.py`, GNNs of 3 seeds saved by `transfer.py`):

1. **A labeling can be scored by how well it explains the wiring.** Cell types are defined by connectivity: neurons of a type have similar numbers of synapses to and from each type (Matsliah et al., 2024). The type-consistency score (`flybrain/consistency.py`) gives each neuron of a random half the label of its most similar neuron in the other half (synapses per partner type, weighted Jaccard as in NTAC) and measures the chance-corrected agreement (Cohen's kappa). It needs no labels, and across the 30 candidate labelings per label below it ranks them like their true accuracy (Spearman 0.95–0.97 for cell types, 0.86–0.95 for super classes, depending on the scoring graph).
2. **Cell-type sizes are stereotyped across flies.** The transferred GNN shifts neurons between neighboring types, so some types are over-filled and others almost empty (T5a: 2,252 predicted vs. 1,664 true neurons; LC21: 13 vs. 153). Rescaling its class probabilities so that the predicted type sizes match the sizes it predicts on FAFB (Sinkhorn scaling, `flybrain/adapt.py`, strength rho) corrects this.

The pipeline scores every combination of alignment (3), normalization (2) and rho (0, 0.25, 0.5, 0.75, 1) by type consistency on the target graph (for MCNS the least preprocessed one, `raw`), keeps the best, and refines it with NTAC seeded where all three GNNs agree. Target labels are used only to report accuracy:

| Direction, label | Chosen (alignment, norm., rho) | Without prior matching | With prior matching | **+ NTAC** | Best of the 30 candidates (post hoc) | Trained within the target (with labels) |
|---|---|---|---|---|---|---|
| FAFB → MCNS, cell types | `raw`, source, 0.5 | 87.6 / 0.68 | 91.2 / 0.75 | **92.6 / 0.76** | 91.4 | 96.6 / 0.95 |
| FAFB → MCNS, super classes | `raw`, source, 0.25 | 96.2 / 0.68 | 96.3 / 0.68 | 96.2 / 0.65 | 96.3 | 98.4 / 0.88 |
| MCNS → FAFB, cell types | `density`, target, 0.75 | 72.0 / 0.57 | 81.3 / 0.68 | **83.4 / 0.69** | 82.9 | 95.2 / 0.93 |
| MCNS → FAFB, super classes | `raw`, source, 0 | 95.2 / 0.66 | 95.2 / 0.66 | **95.5 / 0.67** | 95.2 | 97.8 / 0.84 |

- **Without a single label in the male brain, 92.6 % of its neurons get the right cell type out of 394**, within 4 points of a model trained on male labels. The single GNN with a label-free normalization reached 73–75 %.
- The consistency score picks a candidate within 0.2 points of the best one, and all three scoring graphs pick the same one.
- Prior matching fixes the types the raw transfer over- or under-filled (seed 0, `raw`/source, before → after the pipeline): T5a 2,252 → 1,714 predicted neurons (1,664 true), F1 0.84 → 0.98; LC21 13 → 115 (153 true), F1 0.16 → 0.86. By region, the neuron-weighted F1 rises from 0.40 to 0.61 in the central brain and from 0.83 to 0.96 for visual projection neurons. Kenyon cell subtypes (KCab vs. KCg-m) and some olfactory receptor neuron types stay hard.

**Check in the reverse direction.** The pipeline was developed on FAFB → MCNS. Run unchanged from MCNS to FAFB (GNNs trained on MCNS in each alignment with `transfer.py --source mcns`, then `label_free.py --source mcns --target fafb`):

- **Both label-free components carry over.** Prior matching helps in all six alignment × normalization combinations (best rho 0.5–1 in each; +9.3 points for the chosen one), and the consistency score again ranks the 30 labelings like their accuracy (Spearman 0.87 for cell types, 0.80 for super classes). Its choice is 1.6 points below the best candidate (0.2 in the forward direction) and above every candidate without prior matching (at most 78.5 %). The single GNN with a label-free normalization reaches 60–74 %.
- **The level is lower:** 83.4 % vs. 95.2 % within FAFB, a gap of 12 points (forward: 4). Three error sources are specific to FAFB as the target. 2.2 % of its neurons belong to the 40 types missing from MCNS's mapping (Dm8a/b, ocellar retinula cells, several Johnston's organ types) and cannot be predicted; on the 394 shared types accuracy is 85.3 % (macro-F1 0.78). Photoreceptors (R1-6, R7, R8; 10 % of the neurons) cause 21 % of the errors: 51 % of them are typed correctly in the left FAFB hemisphere and 83 % in the right, the neurons that also make FAFB right → left typing 8 points worse (in the forward direction, 95.6 % of MCNS photoreceptors are correct). Kenyon cell subtypes cause 15 % (1,606 KCg-m predicted as KCab), the confusion that also remains in the forward direction. On shared types without photoreceptors, 87.5 % are correct.
- Super classes: the score picks the best candidate and, as in the forward direction, no prior matching (rho 0).

### Independent test on BANC, with a protocol fixed in advance

BANC v888 (Bates et al., 2026) is a third connectome that played no part in developing the pipeline: an independently reconstructed female brain and nerve cord. Data, labels, runs, reported numbers and three expectations were committed in [docs/banc_protocol.md](docs/banc_protocol.md) (`85516a2`) before any model was run on BANC. The GNNs already trained on FAFB and MCNS were applied unchanged (`label_free.py --target banc`, `--source mcns --target banc`); nothing was trained on BANC. Two caveats come from the data. BANC's cell types were assigned by an alignment to FAFB that uses connectivity (reference A below), so the cleanest test is the central brain against the type of the FAFB neuron matched by morphology (reference B, 11,656 neurons, 159 types). And BANC's brain graph is much sparser: the median neuron has 30 input synapses from other neurons in the graph (FAFB: 152) and 4 input partners (FAFB: 14).

| Run, label | Chosen (alignment, norm., rho) | Without prior matching | With prior matching | **+ NTAC** | Best candidate (post hoc) |
|---|---|---|---|---|---|
| FAFB → BANC, cell types | `density`, target, 1 | 38.2 / 0.21 | 55.7 / 0.37 | **61.7 / 0.40** | 55.7 |
| FAFB → BANC, super classes | `scaled`, source, 1 | 42.2 / 0.11 | 72.3 / 0.24 | **73.3 / 0.22** | 92.6 |
| MCNS → BANC, cell types | `density`/`scaled`, source, 0 | 1.4 / 0.00 | 1.4 / 0.00 | **1.4 / 0.00** | 58.0 |
| MCNS → BANC, super classes | `raw`/`raw`, source, 1 | 69.0 / 0.34 | 78.7 / 0.33 | **79.0 / 0.31** | 94.3 |

FAFB → BANC cell types after NTAC, by subset: central brain 33.7 % (reference A) and 32.3 % (reference B), optic lobe 67.4 %. The expectations stated in advance:

1. **Selected candidate within 2 points of the best: met once in four.** For FAFB → BANC cell types the score picks the best candidate. For MCNS → BANC cell types it picks a collapsed labeling that puts 77 % of the neurons into one type (1.4 % correct): such a labeling is self-consistent on the graph, and the chance correction (Cohen's kappa) does not penalize the collapse enough. The best candidate (58.0 %) ranks third by consistency, after two collapsed ones. For super classes it picks source normalization with full prior matching, 16–20 points below the best candidate (Spearman 0.61 and 0.86 between consistency and accuracy).
2. **The pipeline beats every candidate without prior matching on cell types: met for FAFB → BANC** (61.7 vs. at most 38.2 %), not for MCNS → BANC (1.4 vs. 45.0 %, because of the collapsed choice).
3. **Cell-type accuracy below FAFB → MCNS: met** (61.7 vs. 92.6 %).

What carries over and what does not:

- **Prior matching carries over.** In both runs the best cell-type candidate uses full prior matching (rho 1), 13.0 and 17.5 points above the best candidate without it, although BANC has no R1-6 photoreceptors: prior matching fills R1-6 with 10,800 neurons (FAFB → BANC), 20 % of the remaining errors, as anticipated in the protocol.
- **Label-free selection by type consistency does not carry over reliably.** It worked on the two complete reconstructions, where all candidates were sensible labelings, but it was never tested against collapsed ones.
- **Reconstruction completeness limits transfer.** The best candidates reach 92.6–94.3 % on super classes, but at most 58 % on cell types, and in the central brain only about a third of the neurons get the right type, also against the morphology-based reference.

### Reconstruction density: transfer between thinned copies of FAFB

To separate density from everything else that differs between connectomes, `thinning.py` transfers between copies of FAFB that keep each synapse with probability p and drop connections with fewer than 5 remaining synapses (`flybrain/thinning.py`). At p = 0.4 a copy has about BANC's density (9.8 connections and a median of 34 input synapses per neuron; BANC: 8.6 and 30). GNNs trained on one copy (3 seeds, Kaggle T4 GPU) are tested on an independently thinned copy of every density, on their test split. Cell types with prior matching (rho 1), accuracy %:

| Source \ target density | 1 | 0.6 | 0.4 | 0.25 |
|---|---|---|---|---|
| 1 | **95.1** | 89.4 | 72.9 | 41.7 |
| 0.6 | 93.2 | **91.3** | 82.4 | 54.0 |
| 0.4 | 87.0 | 88.8 | **83.9** | 64.5 |
| 0.25 | 56.3 | 72.6 | 76.5 | **67.7** |

- **Density alone explains about half of the drop on BANC.** The full-density GNN reaches 72.9 % on a copy at BANC's density (62.4 % without prior matching), against 55.7 % (38.2 %) on BANC.
- **A sparser target is best served by a source thinned to its density** (83.9 vs. 72.9 % at p = 0.4). Even then accuracy falls with density (95.1 → 67.7 % on the diagonal): a sparse reconstruction loses information that no alignment recovers.
- **A denser target, as future more complete reconstructions will be, is less of a problem when prior matching is used.** A GNN trained at p = 0.4 reaches 87.0 % on the full-density copy, more than at its own density. Without prior matching it reaches 70.0 %. Thinning the denser target down to the source density helps only without prior matching (70.0 → 82.1 %) or for a 4× gap (p = 0.25 → 1: 56.3 → 67.3 %).
- Super classes stay at 90.8–97.8 % except when the sparsest target (p = 0.25) meets a source at p ≥ 0.6 (86.4 and 74.7 %).
- **On BANC itself, a density-matched source did not help** (exploratory): GNNs trained at p = 0.4 reach at best 51.2 % on cell types, against 55.7 % for the full-density GNNs. On BANC's `raw` graph they gain 12.5 points, as the copies predict, but the full-density GNNs do better on the `density` graph, which keeps BANC's connections with 2 to 4 synapses. In a sparse reconstruction many strong connections survive as weak ones, and lowering the threshold recovers them. FAFB's public connections start at 5 synapses, so the thinned copies cannot show this.

### Development sandbox and a revised selection rule

The failures on BANC show that the label-free choices need a test bed with known labels that differs from the target in the ways new connectomes do. Thinned copies of the development connectomes (`label_free.py --target-retention`, `--target-exclude`) provide one without touching a new dataset: FAFB → MCNS at 100, 50 and 20 % of MCNS's synapses (20 % is about BANC's density relative to FAFB), MCNS → FAFB at 100 and 35 %, and FAFB → MCNS at 50 % without its R1-6 photoreceptors, with and without bounded prior matching. For every candidate, `label_free.py` records label-free properties besides type consistency: the agreement between the three seeds' labelings (disagreement between independently trained models tracks their error under distribution shift; Jiang et al., 2022) and how far the predicted type sizes are from the reference (Jensen-Shannon divergence). `selection_rules.py` compares rules built from them (`flybrain/selection.py`). Gap between the chosen and the best candidate, points, mean / worst over the cases (one per run and label):

| Rule | Sandbox (14 cases) | BANC re-scored (8 cases, exploratory) |
|---|---|---|
| Type consistency (used so far) | 4.1 / 26.7 | 19.9 / 56.6 |
| Type sizes closest to the reference | 4.1 / 26.0 | 0.6 / 1.8 |
| Consistency + seed agreement, collapse guard | 1.1 / 8.9 | 1.5 / 5.8 |
| **Type sizes + seed agreement** (`--select size_agreement`) | **1.3 / 4.0** | **1.1 / 6.3** |

- At low density type consistency ranks the candidates inversely to their accuracy (Spearman −0.68 for cell types at 20 %), the BANC failure reproduced with known labels. On complete graphs it is the best single signal, and seed agreement tracks accuracy closely there (Spearman 0.98 for MCNS → FAFB cell types).
- The rule combining type sizes and seed agreement had the smallest worst case on the sandbox and was chosen there, before it was computed on BANC. On BANC (re-run with the extra columns; the candidates' accuracies were known from the pre-registered runs) it picks a candidate within 0.5 points of the best in all four pre-registered runs, including the collapsed one (58.0 instead of 1.4 % for MCNS → BANC cell types, 92.1 instead of 72.3 % for FAFB → BANC super classes).
- Missing types cost little: without R1-6 the best candidate loses 1.1 points, and prior matching still adds 11 points. Bounding prior matching (`--max-weight 10`) cuts the neurons wrongly assigned to the missing type from 7,833 to 2,627 but costs 0.5 points accuracy and 0.06 macro-F1, so it is not used.
- The sandbox covers only density and missing types, and the rule was chosen from 14 cases; its worst case on BANC (6.3 points) comes from the exploratory run with density-matched GNNs. It needs a new connectome to be tested on.

### GNN-seeded NTAC: combining both

NTAC needs labeled partners, the GNN does not. Seeding NTAC with the predictions on which three GNNs agree (`hybrid.py`) labels the neurons that have no labeled partners better than the GNNs alone. Cell types, accuracy (%) / macro-F1 on the scored neurons (the unlabeled hemisphere, or all labeled MCNS neurons for the transfer):

| Scenario | NTAC alone | GNN (single) | GNN (vote of 3) | Seeds: share, accuracy | **GNN + NTAC** |
|---|---|---|---|---|---|
| FAFB, left → right | 8.4 / 0.20 | 94.5 / 0.90 | 94.8 / 0.91 | 94 %, 98.0 | **96.5 / 0.93** |
| FAFB, right → left | 8.8 / 0.19 | 86.7 / 0.87 | 87.8 / 0.89 | 85 %, 95.7 | **90.8 / 0.92** |
| MCNS, left → right | 7.8 / 0.16 | 95.4 / 0.93 | 96.0 / 0.94 | 95 %, 98.1 | **97.1 / 0.95** |
| MCNS, right → left | 9.5 / 0.22 | 93.7 / 0.91 | 94.6 / 0.92 | 92 %, 97.7 | **96.2 / 0.93** |
| FAFB → MCNS transfer (`density`, target norm.) | not applicable | 73.4 / 0.50 | 75.5 / 0.53 | 71 %, 89.0 | **79.5 / 0.57** |

- On an unlabeled hemisphere the combination reaches 91–97 %, close to what NTAC achieves with labeled partners (97 % on the random split), while NTAC alone stays below 10 %.
- Under transfer it adds 4 points over the GNN vote (79.5 vs. 75.5 %), without any MCNS label. The seeds limit it: 11 % of them are wrong and NTAC keeps seeds fixed.
- For super classes the GNNs are already near the ceiling; the combination changes accuracy by at most 0.4 points.

Related work and how this project relates to it: [docs/literature.md](docs/literature.md).

## Limitations

- Cell types were defined partly from connectivity, including matching across hemispheres and brains (Schlegel et al., 2024), so recovering them from wiring is partly circular. The GNN never sees partner identities, which weakens this concern but does not remove it. Super classes are mostly defined anatomically.
- Only types with ≥ 20 neurons are predicted (434 of 8,772 types); the many rare types are left out.
- The setting is transductive: the whole graph is visible during training, only the labels of validation and test neurons are hidden.
- Two animals of different sex, reconstructed and annotated by different teams. MCNS labels come from Janelia's `flywireType` mapping; 40 of the 434 FAFB types do not occur in it (some are split differently, e.g. Dm8a/b vs. yDm8/pDm8), and 2,751 neurons with several mapped types are not scored (a prediction falls within their type list for only 3–4 % of them).
- The label-free pipeline (type-consistency selection, prior matching) was developed on the FAFB → MCNS pair while looking at its transfer errors. Run unchanged in the reverse direction it still helps (83.4 % on cell types vs. at most 78.5 % without prior matching). In the independent test on BANC, prior matching carried over but the selection by type consistency failed in three of four runs (see above). The revised rule (type sizes + seed agreement) was chosen on a sandbox of the same two development connectomes and still needs a connectome it has not seen.
- Prior matching assumes that type sizes are similar across flies; sex-specific and dimorphic types (about 5 % of the central brain) violate this.
- Each seed changes both the data split and the initialization; the std covers both.
- The cell-type GNN is trained for 300 epochs and its best epoch is close to the end, so it may be slightly underestimated.
- Codex drops connections with fewer than 5 synapses. The thinned FAFB copies therefore cannot contain the weak connections that a sparse reconstruction still has, and they overstate what a density-matched source gains on BANC.
- FAFB, MCNS and BANC differ in proofreading state, synapse detection and annotation schemes; transfer results depend on how neurons and classes are aligned. BANC's cell types come from an alignment to FAFB that uses connectivity.
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
