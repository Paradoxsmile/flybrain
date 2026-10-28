# Results in detail

The full record of part b: every experiment, table and caveat, in the order the work was done. The README gives the overview, [paper/main.pdf](../paper/main.pdf) the argument, and [reproduce.md](reproduce.md) the command and result file for each number.

## Methods

- **Graph:** neurons are nodes; directed edges pre → post, weighted by synapse count (Codex FAFB v783, which keeps connections with ≥ 5 synapses). Raw tables are cleaned once and cached as Parquet.
- **Labels:** `super_class` (10 classes) and `cell_type`, restricted to types with ≥ 20 neurons (434 types, 78 % of neurons).
- **GNN** (`train_gnn.py`, model in `flybrain/models.py`): input features are six degree / synapse-count statistics per neuron, so the model sees anonymous wiring structure and never the labels of other neurons. Each layer averages separately over presynaptic inputs and postsynaptic outputs, weighted by synapse count; this is Dir-SAGE (Rossi et al., 2023) with synapse-weighted means. Configs: `configs/default.yaml` (super_class: 4 layers, 1000 epochs), `configs/cell_type.yaml` (3 layers, 300 epochs, class-weighted loss); depth and options were chosen on validation.
- **Baselines** (`baselines.py`, `baseline_logreg.py`), ordered by the information they use: majority class; logistic regression on a neuron's own degree features; label propagation (connected neurons share a label); partner profile, i.e. logistic regression on the share of a neuron's synapses per partner class, using the labels of training neurons, which is what connectivity-based cell typing compares; NTAC (Schwartzman et al., 2026), which gives each neuron the label of the labeled neuron with the most similar partner-type synapse counts and then re-labels partners iteratively (memory-lean, deterministic reimplementation in `flybrain/ntac.py`, checked step by step against the `ntac` package); plain GraphSAGE (inputs only, unweighted, same depth as the GNN).
- **Splits:** `random` (stratified 70 / 15 / 15) and `left_right` / `right_left` (train and validate on one hemisphere, test on the other).
- **Model selection:** hyperparameters and the training epoch are chosen by validation macro-F1; the test set is only used for the final numbers.
- **Male CNS** (`flybrain/mcns.py`): the traced brain neurons of MCNS v1.0 (no ventral nerve cord), with MCNS superclasses mapped to the FAFB super classes and `flywireType` to FAFB cell types (394 of the 434 types occur; entries listing several types are left out). MCNS detects more synapses per connection, so its graph is built in three alignments: `density` (>= 7 synapses, about FAFB's number of connections, counts × 0.581 as in Janelia's FlyWire comparison), `raw` (FAFB's threshold of 5, no scaling) and `janelia` (counts × 0.581, then threshold 5).
- **Nerve cords** (`flybrain/manc.py`; `cord_neurons` in `flybrain/banc.py` and `flybrain/mcns.py`): MANC v1.2.1 (harmonized by the BANC project) is the source of the nerve-cord test, the nerve cords of BANC and MCNS are its targets. Labels are MANC cell types; self-connections and connections between two ascending or descending neurons are dropped in all three graphs.
- **Transfer** (`transfer.py`): methods are trained on FAFB as in the benchmark and applied to MCNS without any MCNS label, with features normalized by FAFB statistics (`source`) or by MCNS's own (`target`). `--source mcns` trains on MCNS (one alignment per run) and applies to FAFB.
- **GNN-seeded NTAC** (`hybrid.py`): neurons on which three GNNs (seeds) agree become NTAC seeds, together with any truly known labels; NTAC labels the rest. Nothing is tuned (unanimous predictions; NTAC defaults: 15 steps, learning rate 0.3).
- **Label-free transfer** (`label_free.py`): every transfer choice is made without labels of the target connectome. Type sizes are matched to the reference (`flybrain/adapt.py`), and the alignment, normalization and matching strength are chosen by a label-free type-consistency score (`flybrain/consistency.py`); see below. `--source mcns --target fafb` runs the same pipeline from MCNS to FAFB, `--target banc` to BANC (protocol fixed in advance: [docs/banc_protocol.md](banc_protocol.md)).
- **Stratified accuracy** (`stratify.py`): accuracy and macro-F1 of the reported label-free runs by region of the type and by FAFB type size, at each stage of the pipeline, from the saved predictions (no retraining).
- **Benchmark:** `benchmark.py` runs every method on every label and split for three seeds and writes `results/benchmark/summary.md` (`--dataset mcns`: the same within MCNS). All runs log config, metrics and per-neuron predictions to `results/<run_name>/`.

## Results

FlyWire FAFB v783, test accuracy (%) / macro-F1, mean ± std over 3 seeds, produced by `benchmark.py` (baselines on CPU at commit `2000b7b`, NTAC on CPU at `bfbd103`, GNN rows on a Kaggle T4 GPU at `343125f`). Hemisphere columns: trained on the first hemisphere, tested on the second.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="benchmark_dark.png">
  <img alt="Test accuracy per method and split. The GNN stays at 87 to 98 percent on every split. Methods that read partner labels (partner profile, NTAC) reach 91 to 97 percent on the random split but fall to 1 to 9 percent on cell types and 46 to 82 percent on super classes when tested on the unlabeled hemisphere." src="benchmark_light.png">
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

### Where the transfer accuracy comes from: region and type size

The overall accuracy is dominated by the optic lobe: 80 % of MCNS's labeled neurons (71 % of FAFB's and BANC's) belong to optic-lobe intrinsic types, most of them large columnar types. `stratify.py` splits each label-free cell-type run reported here and in the sections below by the region of the neuron's true type (its FAFB super class) and by the type's size in FAFB, at the three stages of the pipeline. It reads the saved predictions and retrains nothing; for runs that saved only the + NTAC labeling, the two earlier stages are recomputed with the saved GNNs for the chosen candidate and checked against `selected.json` (`results/stratified/`). Accuracy % / macro-F1, where macro-F1 is the mean, over the stratum's types present in the target, of each type's F1 on the whole target. The "all" column therefore gives a slightly higher macro-F1 than the tables above (0.81 vs. 0.76 for FAFB → MCNS), which also count predicted types absent from the target.

| Run | Stage | all | optic-lobe intrinsic | visual projection | central brain | sensory | other |
|---|---|---|---|---|---|---|---|
| **FAFB → MCNS** | share of labeled neurons | 109,901 neurons | 80 % | 7 % | 6 % | 7 % | 0.2 % |
| | without prior matching | 87.6 / 0.73 | 92.9 / 0.91 | 86.4 / 0.84 | 52.6 / 0.63 | 55.4 / 0.21 | 88.2 / 0.74 |
| | with prior matching | 91.2 / 0.79 | 95.6 / 0.94 | 94.0 / 0.92 | 63.9 / 0.77 | 59.8 / 0.29 | 90.7 / 0.76 |
| | + NTAC | 92.6 / 0.81 | 97.2 / 0.96 | 95.7 / 0.96 | 65.1 / 0.80 | 59.1 / 0.26 | 95.4 / 0.73 |
| **MCNS → FAFB** | share of labeled neurons | 108,647 neurons | 71 % | 6 % | 7 % | 15 % | 0.4 % |
| | without prior matching | 72.0 / 0.57 | 78.5 / 0.78 | 73.2 / 0.73 | 35.5 / 0.38 | 59.5 / 0.20 | 46.7 / 0.40 |
| | with prior matching | 81.3 / 0.68 | 90.2 / 0.86 | 85.7 / 0.83 | 51.5 / 0.51 | 52.7 / 0.36 | 48.3 / 0.38 |
| | + NTAC | 83.4 / 0.69 | 92.6 / 0.90 | 87.8 / 0.86 | 54.0 / 0.52 | 53.6 / 0.31 | 52.8 / 0.41 |
| **FAFB → BANC** | share of labeled neurons | 63,332 neurons | 71 % | 9 % | 8 % | 12 % | 0.3 % |
| | without prior matching | 38.2 / 0.22 | 44.0 / 0.32 | 31.5 / 0.22 | 10.7 / 0.16 | 26.1 / 0.03 | 57.8 / 0.42 |
| | with prior matching | 55.7 / 0.39 | 62.0 / 0.47 | 56.8 / 0.45 | 36.9 / 0.37 | 28.2 / 0.15 | 70.2 / 0.48 |
| | + NTAC | 61.7 / 0.42 | 68.6 / 0.52 | 63.1 / 0.51 | 44.2 / 0.38 | 30.3 / 0.14 | 77.6 / 0.50 |
| **MCNS → BANC** | share of labeled neurons | 63,332 neurons | 71 % | 9 % | 8 % | 12 % | 0.3 % |
| | without prior matching | 1.4 / 0.00 | 1.8 / 0.00 | 0.0 / 0.00 | 0.3 / 0.00 | 1.1 / 0.00 | 0.0 / 0.00 |
| | with prior matching | 1.4 / 0.00 | 1.8 / 0.00 | 0.0 / 0.00 | 0.3 / 0.00 | 1.1 / 0.00 | 0.0 / 0.00 |
| | + NTAC | 1.4 / 0.00 | 1.8 / 0.00 | 0.0 / 0.00 | 0.5 / 0.00 | 0.7 / 0.00 | 0.0 / 0.00 |
| **FAFB → hemibrain** | share of labeled neurons | 7,639 neurons | 2 % | 35 % | 32 % | 30 % | 0.2 % |
| | without prior matching | 17.3 / 0.15 | 0.0 / 0.00 | 3.1 / 0.01 | 33.6 / 0.32 | 18.5 / 0.15 | 0.0 / 0.00 |
| | with prior matching | 19.1 / 0.13 | 0.0 / 0.00 | 2.6 / 0.02 | 46.0 / 0.25 | 11.9 / 0.15 | 0.0 / 0.00 |
| | + NTAC | 19.6 / 0.12 | 0.0 / 0.00 | 5.0 / 0.03 | 44.5 / 0.24 | 12.3 / 0.13 | 0.0 / 0.00 |

| Run | Stage | all | 20–49 | 50–199 | 200–999 | ≥ 1000 |
|---|---|---|---|---|---|---|
| **FAFB → MCNS** | share of labeled neurons | 109,901 neurons | 5 % | 13 % | 23 % | 58 % |
| | without prior matching | 87.6 / 0.73 | 69.2 / 0.62 | 83.2 / 0.77 | 85.2 / 0.81 | 91.1 / 0.90 |
| | with prior matching | 91.2 / 0.79 | 75.8 / 0.72 | 87.4 / 0.83 | 90.0 / 0.86 | 93.9 / 0.92 |
| | + NTAC | 92.6 / 0.81 | 77.0 / 0.74 | 88.2 / 0.83 | 92.4 / 0.88 | 95.1 / 0.93 |
| **MCNS → FAFB** | share of labeled neurons | 108,647 neurons | 6 % | 12 % | 22 % | 60 % |
| | without prior matching | 72.0 / 0.57 | 51.0 / 0.44 | 74.6 / 0.67 | 66.7 / 0.65 | 75.4 / 0.76 |
| | with prior matching | 81.3 / 0.68 | 64.4 / 0.56 | 81.0 / 0.77 | 78.1 / 0.76 | 84.1 / 0.85 |
| | + NTAC | 83.4 / 0.69 | 64.6 / 0.57 | 82.9 / 0.78 | 80.7 / 0.79 | 86.3 / 0.87 |
| **FAFB → BANC** | share of labeled neurons | 63,332 neurons | 7 % | 15 % | 16 % | 62 % |
| | without prior matching | 38.2 / 0.22 | 18.5 / 0.14 | 22.6 / 0.23 | 45.4 / 0.32 | 42.5 / 0.41 |
| | with prior matching | 55.7 / 0.39 | 33.7 / 0.32 | 42.4 / 0.39 | 60.7 / 0.43 | 60.2 / 0.63 |
| | + NTAC | 61.7 / 0.42 | 35.3 / 0.34 | 46.5 / 0.43 | 66.9 / 0.49 | 67.3 / 0.69 |
| **MCNS → BANC** | share of labeled neurons | 63,332 neurons | 7 % | 15 % | 16 % | 62 % |
| | without prior matching | 1.4 / 0.00 | 0.5 / 0.00 | 0.0 / 0.00 | 0.1 / 0.00 | 2.2 / 0.01 |
| | with prior matching | 1.4 / 0.00 | 0.5 / 0.00 | 0.0 / 0.00 | 0.1 / 0.00 | 2.2 / 0.01 |
| | + NTAC | 1.4 / 0.00 | 0.8 / 0.00 | 0.0 / 0.00 | 0.0 / 0.00 | 2.1 / 0.01 |
| **FAFB → hemibrain** | share of labeled neurons | 7,639 neurons | 34 % | 42 % | 12 % | 11 % |
| | without prior matching | 17.3 / 0.15 | 24.7 / 0.20 | 18.7 / 0.11 | 1.0 / 0.00 | 7.8 / 0.03 |
| | with prior matching | 19.1 / 0.13 | 20.1 / 0.17 | 10.2 / 0.08 | 1.1 / 0.02 | 68.8 / 0.12 |
| | + NTAC | 19.6 / 0.12 | 19.3 / 0.16 | 12.1 / 0.08 | 0.3 / 0.01 | 69.2 / 0.12 |

Points of the overall accuracy (+ NTAC) contributed by each region:

| Run | all | optic-lobe intrinsic | visual projection | central brain | sensory | other |
|---|---|---|---|---|---|---|
| FAFB → MCNS | 92.6 | 78.0 | 6.6 | 3.7 | 4.1 | 0.2 |
| MCNS → FAFB | 83.4 | 65.8 | 5.4 | 4.0 | 8.1 | 0.2 |
| FAFB → BANC | 61.7 | 48.8 | 5.7 | 3.6 | 3.5 | 0.2 |
| MCNS → BANC | 1.4 | 1.3 | 0.0 | 0.0 | 0.1 | 0.0 |
| FAFB → hemibrain | 19.6 | 0.0 | 1.8 | 14.2 | 3.7 | 0.0 |

- **78.0 of the 92.6 points of FAFB → MCNS come from optic-lobe intrinsic neurons**, 97.2 % of which are correct. Outside the optic lobe (20 % of the neurons) 74.0 % are correct (macro-F1 0.68); in MCNS → FAFB 60.9 % and in FAFB → BANC 44.9 %.
- **Central brain, FAFB → MCNS: 65.1 %** (neuron-weighted F1 0.61, macro-F1 0.80). Kenyon cells are 65 % of the central-brain neurons in the label set and 84 % of its errors; the other central-brain types reach 84.5 %. Only 71 central-brain types have at least 20 neurons in FAFB, so the label set excludes most of the central brain to begin with.
- **Sensory:** photoreceptors are typed well (95.6 % in MCNS), olfactory receptor neurons are not (17.7 %); 33 of the 40 sensory types with F1 below 0.1 are ORN types.
- **Type size:** accuracy falls from 95.1 % for types with at least 1,000 FAFB neurons to 77.0 % for types with 20–49 (FAFB → MCNS). Size and region are confounded: 33 of the 39 largest types are optic-lobe intrinsic.
- **Prior matching helps the central brain most** (+11, +16 and +26 points in FAFB → MCNS, MCNS → FAFB and FAFB → BANC), and costs sensory neurons 7 points in MCNS → FAFB.
- The BANC and hemibrain subsets in their sections group neurons by the dataset's own region of the neuron, not by the super class of the type, hence different numbers (central brain: 33.7 % there vs. 44.2 % for central-brain types here on BANC, 28 % vs. 44.5 % on the hemibrain). On the hemibrain the large-type bin is almost only KCg-m, which prior matching fills; MCNS → BANC is the collapsed labeling (1.4 % everywhere).

![Per-type F1, FAFB → MCNS](../paper/figures/type_f1.png)

### Independent test on BANC, with a protocol fixed in advance

BANC v888 (Bates et al., 2026) is a third connectome that played no part in developing the pipeline: an independently reconstructed female brain and nerve cord. Data, labels, runs, reported numbers and three expectations were committed in [docs/banc_protocol.md](banc_protocol.md) (`85516a2`) before any model was run on BANC. The GNNs already trained on FAFB and MCNS were applied unchanged (`label_free.py --target banc`, `--source mcns --target banc`); nothing was trained on BANC. Two caveats come from the data. BANC's cell types were assigned by an alignment to FAFB that uses connectivity (reference A below), so the cleanest test is the central brain against the type of the FAFB neuron matched by morphology (reference B, 11,656 neurons, 159 types). And BANC's brain graph is much sparser: the median neuron has 30 input synapses from other neurons in the graph (FAFB: 152) and 4 input partners (FAFB: 14).

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

| Rule | Sandbox (14 cases) | BANC re-scored (6 cases, exploratory) |
|---|---|---|
| Type consistency (used so far) | 4.1 / 26.7 | 19.9 / 56.6 |
| Type sizes closest to the reference | 4.1 / 26.0 | 0.6 / 1.8 |
| Consistency + seed agreement, collapse guard | 1.1 / 8.9 | 1.5 / 5.8 |
| **Type sizes + seed agreement** (`--select size_agreement`) | **1.3 / 4.0** | **1.1 / 6.3** |

- At low density type consistency ranks the candidates inversely to their accuracy (Spearman −0.68 for cell types at 20 %), the BANC failure reproduced with known labels. On complete graphs it is the best single signal, and seed agreement tracks accuracy closely there (Spearman 0.98 for MCNS → FAFB cell types).
- The rule combining type sizes and seed agreement had the smallest worst case on the sandbox and was chosen there, before it was computed on BANC. On BANC (re-run with the extra columns; the candidates' accuracies were known from the pre-registered runs) it picks a candidate within 0.5 points of the best in all four pre-registered runs, including the collapsed one (58.0 instead of 1.4 % for MCNS → BANC cell types, 92.1 instead of 72.3 % for FAFB → BANC super classes).
- Missing types cost little: without R1-6 the best candidate loses 1.1 points, and prior matching still adds 11 points. Bounding prior matching (`--max-weight 10`) cuts the neurons wrongly assigned to the missing type from 7,833 to 2,627 but costs 0.5 points accuracy and 0.06 macro-F1, so it is not used.
- The sandbox covers only density and missing types, and the rule was chosen from 14 cases; its worst case on BANC (6.3 points) comes from the exploratory run with density-matched GNNs. It needs a new connectome to be tested on.

### Partial volumes, and a second pre-registered test on the hemibrain

The hemibrain (Scheffer et al., 2020) covers the central brain, mostly of the right hemisphere, without the optic lobes; 16 % of its neurons are cut by the volume boundary. 77 % of FAFB's labeled neurons belong to types it does not contain, which breaks the assumption shared by prior matching, the size-based selection rule and normalization by the target's own statistics: that the target has the source's composition.

**Development sandbox.** MCNS restricted to the central brain, with 16 % of its neurons cropped (each synapse of a cropped neuron kept with probability 0.1), labels known. For the same central-brain neurons, the best candidate falls from 69.9 % (full MCNS volume) to 46.9 % (partial volume) and 36.3 % (partial and cropped): partners outside the volume carry about 23 points, cropping about 10. Two fixes were tested. A **coverage prior** (`flybrain/coverage.py`) uses the type distribution of the FAFB neurons inside the covered volume (those with at least 10 % of their synapses outside the optic-lobe neuropils), which needs only the list of covered neuropils; it raised the best candidate from 30.5 to 36.3 %, and the size-based rule then chose it. Estimating the target's type sizes by EM (Saerens et al., 2002) amplified the classifier's errors (23.1 %), and a normalization against the covered part of the source did not beat the source normalization.

**Pre-registered test.** Protocol committed before any model ran on the hemibrain ([docs/hemibrain_protocol.md](hemibrain_protocol.md), `161cf7b`). The harmonized edgelist repeats rows (931,329 exact copies), which are dropped; within its volume the hemibrain then has FAFB's density (26.7 connections and 387 synapses per neuron at 5 synapses; FAFB: 26.8 and 364). Labels: hemibrain types that are among the 434 FAFB types (7,639 neurons, 202 types).

| Label | Chosen (alignment, norm., rho) | Without prior matching | With prior matching | **+ NTAC** | Best candidate (post hoc) | Old consistency rule |
|---|---|---|---|---|---|---|
| cell types | `raw`, source, 1 | 17.3 / 0.09 | 19.1 / 0.09 | **19.6 / 0.09** | 25.5 | 7.8 |
| super classes | `raw`, source, 0.5 | 72.8 / 0.20 | 74.3 / 0.20 | **74.4 / 0.19** | 78.4 | 72.8 |

1. **Selected candidate within 5 points of the best: met for super classes (4.1), not for cell types (6.4).** The old consistency rule would have chosen 7.8 %.
2. **The coverage prior gives a better best candidate than the source prior: met, but by little for cell types** (25.5 vs. 24.9 %). For super classes it matters most after selection: with the source prior the pipeline forces the expected optic-lobe classes onto central-brain neurons and ends at 34.5 % (74.4 % with the coverage prior).
3. **Complete neurons are typed better than cropped ones: met** (22.8 vs. 12.2 % on cell types, 76.1 vs. 65.5 % on super classes, after NTAC).
4. **Cell-type accuracy below 60 %: met** (19.6 %).

- **A partial volume is beyond what the transferred GNN can type.** Super-class accuracy (74.4 %) is barely above always predicting the most frequent class (73.1 %). Visual projection neurons, whose dendrites lie in the missing optic lobe, are typed correctly in 5 % of cases; central-brain neurons in 28 % (38 % for the best candidate).
- The label-free safeguards still matter: without the coverage prior and the new rule, the pipeline would end at 34.5 % (super classes) and 7.8 % (cell types).

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

### Rarer cell types: lowering the minimum type size

The experiments above predict only FAFB types with at least 20 neurons (434 types). `size_threshold.py` repeats the within-FAFB training (3 seeds per split) and the label-free FAFB → MCNS transfer at a minimum of 10, 5, 3 and 2 neurons per type (`load_graph(min_class_size=...)`; nothing else changed). The GNNs were trained on a Kaggle T4, the label-free stage ran on CPU. Below 5 neurons per type (thousands of classes) training uses `lean_training.py` (output layer in row chunks; identical to the standard training at ≥ 20 up to 1e-5) and the label-free stage `lean_label_free.py` (probabilities and prior matching in chunks; reproduces the standard run at ≥ 10, 93.02 vs. 93.04 %). Accuracy % / macro-F1 (macro-F1 here: mean over the true types present, as in `stratify.py`); results in `results/size_threshold/`:

| Min. size | Types (FAFB neurons labeled) | FAFB random split | FAFB left → right | FAFB → MCNS label-free, + NTAC (MCNS types) | of which central-brain types |
|---|---|---|---|---|---|
| 20 | 434 (108,647; 78 %) | 95.1 ± 0.1 / 0.93 | 94.3 ± 0.7 / 0.92 | 92.6 / 0.81 (394) | 65.1 / 0.80 (71 types) |
| **10** | **779 (112,996; 81 %)** | 94.2 ± 0.5 / 0.90 | 93.1 ± 1.1 / 0.87 | **93.0 / 0.80 (649)** | **77.7 / 0.79 (243 types)** |
| 5 | 2,164 (121,799; 87 %) | 90.9 ± 0.8 / 0.82 | 89.1 ± 0.9 / 0.77 | 90.3 / 0.71 (1,694) | 72.3 / 0.70 (1,181 types) |
| 3 | 4,175 (129,247; 93 %) | 87.5 ± 0.4 / 0.72 | 87.2 ± 0.8 / 0.72 | 87.6 / 0.57 (3,206) | 62.5 / 0.54 (2,542 types) |
| 2 | 8,658 (138,213; 99 %) | 82.9 ± 0.9 / 0.68 * | 83.8 ± 1.1 / 0.71 | 84.2 / 0.51 (7,064) | 64.3 / 0.54 (5,371 types) |

\* In the random split, types with 2 neurons go entirely to training, so this row tests the same 4,175 types as ≥ 3. Only the left → right split tests 2-neuron types, typically one neuron per hemisphere: the model has seen a single example. The ≥ 20 left → right row is the sweep's own run (the benchmark's separate run: 94.5 ± 0.9).

By FAFB type size: within FAFB, left → right, and FAFB → MCNS after NTAC, each at the threshold that first includes the bin (accuracy % / macro-F1, number of types):

| | 2–4 (at ≥ 2) | 5–9 (at ≥ 5) | 10–19 (at ≥ 10) | 20–49 (at ≥ 20) | ≥ 1000 (at ≥ 20) |
|---|---|---|---|---|---|
| FAFB left → right | 75.9 / 0.72 (6,438) | 76.4 / 0.74 | 86.6 / 0.84 | 90.7 / 0.89 | 95.4 / 0.95 |
| FAFB → MCNS, label-free | 54.2 / 0.49 (5,370) | 71.6 / 0.68 (1,045) | 75.5 / 0.74 (255) | 77.0 / 0.74 | 95.1 / 0.93 |

The same types at every threshold, FAFB → MCNS after NTAC:

| Min. size | 2–4 | 5–9 | 10–19 | 20–49 | 50–199 | 200–999 | ≥ 1000 |
|---|---|---|---|---|---|---|---|
| 20 | – | – | – | 77.0 / 0.74 | 88.2 / 0.83 | 92.4 / 0.88 | 95.1 / 0.93 |
| 10 | – | – | 75.5 / 0.74 | 80.4 / 0.76 | 90.8 / 0.87 | 93.7 / 0.90 | 95.2 / 0.93 |
| 5 | – | 71.6 / 0.68 | 68.6 / 0.66 | 79.6 / 0.75 | 90.8 / 0.87 | 90.4 / 0.88 | 94.0 / 0.92 |
| 3 | 54.2 / 0.48 | 66.5 / 0.59 | 63.6 / 0.59 | 78.3 / 0.72 | 89.8 / 0.85 | 89.3 / 0.87 | 93.5 / 0.91 |
| 2 | 54.2 / 0.49 | 58.9 / 0.51 | 61.8 / 0.57 | 73.7 / 0.68 | 88.2 / 0.83 | 88.8 / 0.85 | 92.6 / 0.91 |

- **On the rarest types within one connectome, classic graph matching beats the GNN.** Having seen one or two neurons of a type on the left hemisphere, the GNN types its right-hemisphere neurons at 75.9 % (6,438 types of 2–4 neurons, out of 8,582). These types were defined by matching neurons across hemispheres (Schlegel et al., 2024), so the task is close to pairing homologs, which graph matching does without labels (Pedigo et al., 2023). `graph_matching.py` matches the left and right neurons of 2–4-neuron types by Frank–Wolfe on the relaxed problem (FAQ; Vogelstein et al., 2015), with ipsilateral and contralateral edges (bisected matching), and gives each right neuron the type of its matched left neuron. This setup knows which right neurons are rare-type, which the GNN does not. Results in `results/graph_matching/`, on the same 8,142 right neurons:

  | Method | Accuracy % / macro-F1 |
  |---|---|
  | GNN (left → right, mean of 3 seeds) | 75.9 ± 0.7 / 0.74 |
  | GNN votes, restricted to the rare candidates and made one-to-one | 79.8 / 0.81 |
  | Graph matching, ipsilateral edges | 91.2 / 0.92 |
  | **Graph matching, bisected** | **92.2 / 0.93** |
  | **Label-free two-stage**: the GNN picks the right neurons of rare types, graph matching types them | **85.8 / 0.88** (GNN majority alone: 78.9 / 0.78) |

  Graph matching wins in every region (central brain 91.7 vs. 76.8 %, ascending, descending and other types 93.5 vs. 67.2 %), although it sees only the edges among about 8,000 neurons per side; starting it from the GNN's votes changes nothing. In the two-stage version, the GNN's rare-type candidates (9,041 neurons) contain 93 % of the true rare-type neurons, and 84 % of them are rare-type. Over all 68,593 right test neurons, the two stages raise macro-F1 from 0.75 to 0.84 and accuracy from 86.6 to 87.4 %.
- **Rare types transfer label-free, with a gap that widens below 10 neurons.** Without any MCNS label, types with 10–19 FAFB neurons are typed at 75.5 %, 5–9 neurons at 71.6 % and 2–4 neurons at 54.2 % (5,370 types, 13,742 neurons). Against within-FAFB left → right, the gap is about 11, 5 and 22 points.
- **Across connectomes, the two-stage approach lifts the rarest types from 54 to 75 % without labels** (development pair; `graph_matching_mcns.py`, results in `results/graph_matching_mcns/`). Stage 1 is the label-free FAFB → MCNS labeling at ≥ 2; the 18,479 MCNS neurons it assigns to a 2–4-neuron type are the candidates (84 % of the labeled rare-type neurons). Stage 2 matches the 16,365 FAFB neurons of 2–4-neuron types to them by graph matching, one block per hemisphere, with ipsilateral and contralateral edges (about 9,300 × 9,300 per block, 27 min, 4.5 GB). Surplus candidates and all other MCNS neurons keep their stage-1 type. On the 13,742 MCNS neurons of 2–4-neuron FAFB types (5,370 types), accuracy % / macro-F1:

  | Method | 2–4-neuron types | All labeled MCNS neurons |
  |---|---|---|
  | Stage 1: label-free GNN pipeline | 54.2 / 0.49 | 84.2 / 0.51 |
  | Stage-1 probabilities made one-to-one (no wiring) | 59.9 / 0.62 | 84.8 / 0.61 |
  | **Two-stage, graph matching from the barycenter** | **74.5 / 0.78** | **86.3 / 0.73** |
  | Two-stage, graph matching started from the GNN probabilities | 74.6 / 0.78 | 86.3 / 0.74 |
  | Oracle candidates (the labeled rare-type neurons) + graph matching | 85.3 / 0.87 | 87.4 / 0.81 |

  - The wiring does the work: graph matching without any GNN input (barycenter start) is as good as starting from the GNN's probabilities, and both agree on 92.6 % of candidates. As within FAFB, the GNN's role is to select which neurons to match.
  - On the rare-type neurons that stage 1 selects, matching reaches 89.1 % (stage 1: 64.8 %); it fixes 3,288 stage-1 errors and breaks 491 correct types. The largest gains are where the GNN is weakest: ascending, descending and other types 25.2 → 66.1 %, central brain 58.0 → 75.2 %.
  - Between two animals graph matching is nearly as accurate as between two hemispheres (oracle candidates: 85.3 vs. 92.2 %). The remaining gap to the oracle is candidate selection: stage 1 misses 16 % of the rare-type neurons.
  - Caveats: FAFB → MCNS is the development pair (the stage-1 pipeline was developed on it; the three matching variants were all run and reported). MCNS's `flywireType` for these types comes from cross-dataset matching that uses connectivity, so agreement is partly circular. One run per variant; the matching is one-to-one, while type sizes differ between the animals. From the barycenter, the first Frank–Wolfe step pairs the leading singular vectors of the gradient by rank, because the exact assignment solver takes 5–10 min on that near-degenerate matrix; the soft start, which avoids this, converges to nearly the same matching.
- **Pre-registered test on BANC: the two-stage gain holds on a held-out, three times sparser connectome.** Protocol, code and four expectations were committed in [docs/banc_rare_protocol.md](banc_rare_protocol.md) (`0d1ee67`) before either stage was run on BANC at this label set. Stage 1 was selected by `size_agreement` (it chose `density`, target normalization, rho 1, which was also the best of the 30 candidates); stage 2 was unchanged from FAFB → MCNS. Results in `results/graph_matching_banc/`. BANC neurons of 2–4-neuron FAFB types, accuracy % / macro-F1, against BANC's own typing (reference A) and the morphology-matched central-brain types (reference B):

  | Method | 2–4, reference A (14,512 neurons, 5,416 types) | 2–4, reference B (13,848) | All reference-A neurons (88,639) |
  |---|---|---|---|
  | Stage 1: label-free GNN pipeline | 22.2 / 0.19 | 21.3 / 0.18 | 47.9 / 0.19 |
  | **Two-stage (primary run)** | **41.3 / 0.47** | **40.6 / 0.47** | **51.0 / 0.40** |
  | Oracle candidates + graph matching (uses labels) | 65.9 / 0.70 | 65.5 / 0.70 | 55.0 / 0.57 |

  - **All four expectations were met:** (1) two-stage beats stage 1 by at least 10 points on both references (+19.1 and +19.3; MCNS +20.3); (2) over all reference-A neurons macro-F1 rises by at least 0.05 and accuracy does not fall (0.19 → 0.40, 47.9 → 51.0 %); (3) the rare stratum stays below FAFB → MCNS (41.3 vs. 74.5 %); (4) matching withstands the sparsity better than the GNN (with oracle candidates it keeps 0.77 of its MCNS accuracy, stage 1 keeps 0.41).
  - The morphology-matched reference gives the same numbers within 1 point, so the gain is not an artifact of BANC's connectivity-based typing. The graph variant hardly matters (`raw` and `scaled`, with 40 % of the edges, lose 1 point).
  - **Candidate selection is the bottleneck:** stage 1 selects only 54 % of the rare-type neurons (MCNS: 84 %). On those it does select, matching reaches 77.1 % (MCNS: 89.1 %). The largest gains are for descending and other types (24.2 → 61.8 %) and visual projection types (40.6 → 58.7 %); central-brain types rise from 21.1 to 38.7 %.
  - The fine label set costs the common types: stage 1 at ≥ 2 types 47.9 % of all reference-A neurons, against 61.7 % for the ≥ 20 pipeline of the first BANC test (different neuron sets: 88,639 vs. 63,332 labeled neurons, more of them rare). In practice, common types are better typed with a coarser model and the ≥ 2 model used only to select rare-type candidates.
  - Deviation: peak memory was not recorded. One run per setting.
- **Better candidate selection (development only, after the BANC test).** Plan: [docs/candidate_selection_plan.md](candidate_selection_plan.md); script `candidate_selection.py`; results in `results/candidate_selection/`. BANC was not used. As a BANC stand-in, MCNS thinned to 25 % of its synapses reproduces BANC's candidate selection: recall 54.4 % (BANC 54.3 %), 6,110 / 5,986 candidates per side (BANC 5,932 / 6,049), stage 1 on the rare stratum 19.2 % (BANC 22.2 %). At 35 % the copy was too mild (recall 70 %). Accuracy % / macro-F1 on the 2–4 stratum:

  | Candidates | Recall (sandbox) | Sandbox, 25 % | Full MCNS |
  |---|---|---|---|
  | Current: stage-1 type after NTAC is 2–4 | 54.4 % | 47.3 / 0.53 | 74.5 / 0.78 |
  | `mass_fill`: current, topped up by rare-type probability mass to the label-free expected count | 68.4 % | 59.4 / 0.65 | = current (already above the count) |
  | **`union`: stage-1 type before or after NTAC is 2–4** | 70.0 % | **60.1 / 0.65** | **81.6 / 0.83** |
  | Oracle candidates (uses labels) | – | 82.0 / 0.84 | 85.3 / 0.87 |

  - **Selecting more candidates is what helps, not ranking them better.** At equal count, rare-type probability mass beats the argmax by under 1 recall point everywhere. Matching stays accurate on the added rare-type neurons (81 % on the sandbox), and most added false positives end on dummies.
  - **`union` is adopted:** it has no hyperparameter, ties with `mass_fill` on the sandbox, and also helps on full MCNS (+7.1 points, two thirds of the gap to oracle candidates), where `mass_fill` cannot act. Its cost is a larger matching (up to 11.4k per block, 34 min, 6.2 GB) and −2.2 points on 5–19-neuron types.
  - **Two-tier fallback adopted:** neurons not typed by the matching take the ≥ 10 model's type where their ≥ 2 type has at least 10 FAFB neurons. Over all labeled neurons this adds 2.5 points on full MCNS (`union`: 86.9 → 89.4 %) and 3.7 on the sandbox (44.0 → 47.7 %), with the rare stratum unchanged. Central-brain types lose 0.4–1.0 points, and sensory types lose 4 points on the sandbox. In the nerve-cord test below it cost accuracy and was dropped.
  - **Many-to-one matching is rejected:** letting each FAFB neuron take two candidates drops oracle-candidate accuracy on full MCNS from 85.3 to 76.1 %, because candidates pile onto a few well-connected FAFB neurons. Absorbing dummy-matched candidates lowers macro-F1, and rejecting low-gain matches changes nothing.
  - A second thinning realization (seed 3) replicates every finding within 1–2 points.
  - These are development results. The sandbox is harsher than BANC for stage 1 but milder for matching (oracle 82 vs. 66 %), so the expected BANC-level gain is uncertain. The improved pipeline was then tested on the nerve cord (next item).
- **Pre-registered test on two nerve cords: the improved two-stage pipeline holds on a new source and two new targets; two of four expectations were met on both.** Protocol, code and four expectations were committed in [docs/vnc_rare_protocol.md](vnc_rare_protocol.md) (`5119069`) before any model ran on a target cord; the report collector `vnc_report.py` followed in `2519cf8` and only reads saved outputs. Source: MANC v1.2.1 (male nerve cord; GNNs trained on MANC only, at minimum type sizes 2 and 10). Targets: the BANC v888 cord (female, 31 % of MANC's connections per neuron) and the MCNS v1.0 cord (a second male). No nerve cord was used in developing the pipeline. Method as developed: `union` candidates, graph matching from the barycenter, two-tier fallback; one disclosed change, a pooled block for ascending and descending neurons (their `side` follows the neck in MANC and the soma in the targets), decided from a label-based side count before any model output on a cord. Connections between two passing neurons and self-connections are dropped in all three graphs. Results in `results/vnc_rare/SUMMARY.md`. Neurons of 2–4-neuron MANC types, accuracy % / macro-F1, against each target's alignment to MANC (reference A) and, for BANC, its curated typing (reference B):

  | Method | BANC cord, ref A (5,989 neurons) | BANC cord, ref B (5,257) | MCNS cord, ref A (5,945) |
  |---|---|---|---|
  | Stage 1: label-free GNN pipeline | 39.5 / 0.36 | 40.7 / 0.37 | 72.0 / 0.70 |
  | Old two-stage (`gnn` candidates) | 52.1 / 0.55 | 51.7 / – | 77.1 / 0.79 |
  | **Improved two-stage (`union`, primary)** | **61.0 / 0.61** | **61.0 / 0.61** | **82.2 / 0.82** |
  | Oracle candidates + graph matching (uses labels) | 75.5 / 0.73 | 75.5 / – | 87.0 / 0.86 |

  | Expectation (fixed in advance) | BANC cord | MCNS cord |
  |---|---|---|
  | 1. `union` beats `gnn` by ≥ 3 points on the 2–4 stratum, macro-F1 not > 0.01 lower | met (+8.9) | met (+5.1) |
  | 2. Final ≥ stage 1 + 15 points | met (+21.4; ref B +20.3) | **not met** (+10.2) |
  | 3. The pipeline closes ≥ half of the gap to oracle candidates | met (0.60) | met (0.68) |
  | 4. Two-tier fallback: ≥ +0.5 points over all labeled neurons, rare types unchanged | **not met** (−0.4) | **not met** (−1.2) |

  - **The core result replicates:** matching lifts the rarest types by 21 and 10 points, `union` beats the old candidates on both cords, and every group gains on both (BANC: intrinsic 37.3 → 56.7 %, ascending 54.1 → 71.2 %, descending 41.4 → 74.2 %, motor 16.0 → 48.4 %). Reference B, BANC's curated typing, gives the same numbers. Intrinsic neurons, whose references rest partly on connectivity, gain no more than the others.
  - **Expectation 2 missed on MCNS because stage 1 is already strong there** (72.0 %, against 19–54 % in the earlier tests). The oracle ceiling is 87.0 %, so +15 would have meant closing the whole gap; the pipeline closed 0.68 of it. The threshold should have been set relative to that gap. Reported as not met; the ceiling is a post-hoc explanation.
  - **The two-tier fallback does not transfer to the cord.** On both cords the min-size-10 model types the neurons it covers worse than the min-size-2 model (10–19 bin: BANC 55.7 → 52.8 %, MCNS 76.1 → 72.5 %), so it costs 0.4 and 1.2 points over all labeled neurons. Within MANC the min-size-10 model is not much better either (78.1 vs. 70.9 %, FAFB 94.2 vs. 82.9 %); this is a likely reason, not tested. The fallback is therefore no longer part of the recommended pipeline.
  - **The passing block matters:** without it the rare types lose 5.7 (BANC) and 5.6 (MCNS) points. The other graph variants would have cost 4–5 points on BANC and 0.3 on MCNS; stage 1 chose the best one on both.
  - **Costs:** types of 5–19 neurons lose 1.5–2.5 points, because 31 % (BANC) and 17 % (MCNS) of the `union` candidates are of other types and receive a rare type. Over all labeled neurons the pipeline still gains (BANC 44.8 → 49.4 %, MCNS 73.1 → 75.0 % without the fallback). Each target took under 30 min on the CPU (peak 2.1 GB). One run per setting.
- **≥ 10 is better than ≥ 20 everywhere after transfer**: 255 more types, overall 93.0 vs. 92.6 % on 3,100 more neurons, and no size bin worse (20–49: 80.4 vs. 77.0 %). We therefore use ≥ 10 as the default label set for new cell-type results. The experiments above, including the two pre-registered tests, keep ≥ 20, the label set their protocols and saved models fix, and the code defaults (`configs/cell_type.yaml`, `label_free.LABELS`) stay at 20 so that their commands reproduce them.
- **Below 10, each threshold trades common-type accuracy for coverage.** At ≥ 2 the 20–49 bin falls from 77.0 to 73.7 % after transfer and the ≥ 1000 bin from 95.1 to 92.6 %; within FAFB the cost is larger (≥ 1000, left → right: 95.4 to 89.0 %). A two-stage model, one classifier for common types and a separate stage for rare ones, may avoid this.
- **The central brain and "other" carry the rare types and their errors.** Central-brain accuracy after transfer peaks at 77.7 % (243 types at ≥ 10) and settles at 62–64 % with thousands of rare types; at ≥ 20 its 71 types are dominated by Kenyon cell subtypes. Ascending, descending and other types fall to 31.6 % at ≥ 2, sensory types stay near 60 %. Optic-lobe and visual projection types stay at 93–97 % at every threshold.
- **The pipeline steps behave differently with many classes.** Prior matching adds more as the number of types grows (+13.2 points at ≥ 2, +3.6 at ≥ 20). NTAC still raises accuracy by 3–4 points but lowers macro-F1 at ≥ 3 and ≥ 2 (0.62 → 0.57, 0.59 → 0.51): it pulls neurons of rare types into common ones, so for rare types the prior-matched labeling is the better endpoint.

### Labeling effort: how many hand labels the GNN saves

An annotation team typing a new connectome can label some neurons by hand. `label_effort.py` labels k % of each type's reference neurons at random (3 draws) and compares seeded NTAC with the hand labels only, GNN + NTAC (hand labels plus the unanimous label-free GNN predictions as fixed seeds, as in `hybrid.py`), and NTAC with the label-free pipeline as a fallback for types that have no hand label. Scored on the neurons without a hand label; ≥ 20 label set; nothing retrained; results in `results/label_effort/`. Accuracy % / macro-F1:

| Target | Method | k = 0 | 1 % | 2 % | 5 % | 10 % | 50 % |
|---|---|---|---|---|---|---|---|
| MCNS | NTAC, hand labels only | – | 91.7 / 0.51 | 95.5 / 0.74 | 97.2 / 0.87 | 97.7 / 0.90 | 98.5 / 0.95 |
| | GNN + NTAC | 92.6 / 0.81 | 92.7 / 0.81 | 92.9 / 0.82 | 93.2 / 0.83 | 93.6 / 0.84 | 95.1 / 0.89 |
| | NTAC + GNN fallback | – | **94.1 / 0.81** | 95.3 / 0.83 | 96.7 / 0.87 | 97.2 / 0.90 | 97.9 / 0.94 |
| BANC, reference A | NTAC, hand labels only | – | 58.5 / 0.17 | 68.4 / 0.30 | 73.9 / 0.45 | 76.0 / 0.51 | 80.0 / 0.59 |
| | GNN + NTAC | 61.7 / 0.42 | 62.1 / 0.43 | 62.5 / 0.44 | 63.5 / 0.46 | 65.0 / 0.48 | 70.3 / 0.55 |
| | NTAC + GNN fallback | – | 57.7 / 0.37 | 64.3 / 0.38 | 68.7 / 0.42 | 70.6 / 0.47 | 73.9 / 0.53 |
| BANC, reference B (morphology) | NTAC, hand labels only | – | 25.8 / 0.04 | 44.9 / 0.25 | 51.0 / 0.42 | 52.6 / 0.45 | 61.2 / 0.56 |
| | GNN + NTAC | 32.3 / 0.27 | 32.8 / 0.27 | 34.7 / 0.31 | 37.1 / 0.35 | 40.5 / 0.40 | 52.1 / 0.51 |
| | NTAC + GNN fallback | – | 25.9 / 0.25 | 33.6 / 0.28 | 40.2 / 0.34 | 41.5 / 0.36 | 47.5 / 0.44 |

- **By accuracy, the label-free GNN is worth about 1.3 % hand labels.** On all three references, NTAC alone with 1.2–1.5 % of each type labeled matches GNN + NTAC at k = 1–5 % (MCNS: about 1,300–1,500 neurons). From k = 2 % on, plain NTAC is better.
- **The saving is real for small types at very low budgets.** Random labeling at 1 % leaves types with fewer than 50 neurons without a hand label, which NTAC cannot assign. GNN + NTAC at 1 % matches the macro-F1 of NTAC with 3.4 % (MCNS), 4.4 % (BANC A) and 2.3 % (BANC B) of the neurons labeled, 2.3–4.4 times as many hand labels. The advantage is gone by 5–10 %, and a team that labels at least one neuron of every type would shrink it.
- **Fixed GNN seeds are the wrong combination once hand labels exist.** NTAC never revises them, and each hand label corrects only its own neuron: in the MCNS central brain GNN + NTAC goes from 65.8 to 75.0 % between 1 and 50 % hand labels, NTAC alone from 69.9 to 96.6 %. Using the GNN only as a fallback is the best of the three at 1 % on MCNS (central brain 88.1 %) but hurts on BANC, where the transferred labels are less reliable. Revisable GNN seeds (only hand labels fixed) are not tested yet.
- The GNN's niche is therefore typing with no or almost no labels, and covering the types a small hand-labeling budget misses.

### Estimating accuracy and flagging errors without labels

`estimate_accuracy.py` asks whether the label-free columns of `label_free.py` (seed agreement, type-size divergence, type consistency, share of the largest type) predict a labeling's accuracy. Simple linear predictors are fitted on the development targets (FAFB ↔ MCNS and the sandbox; 1,200 candidates over all runs) and tested on BANC and the hemibrain, and in every leave-one-connectome-out rotation. Results in `results/accuracy_estimation/`.

- **Run-level accuracy cannot be estimated reliably.** The pre-specified predictor (seed agreement + type-size divergence) misses the held-out runs by 12.8 points on average (median 8.4; 5 of 12 within 5 points); a five-parameter model has a median miss of 3.6 but the same worst case (49 points). It works on complete targets whose labeling has not collapsed (FAFB → BANC cell types: +0.5 points) and fails in two ways no label-free column reveals: a collapsed labeling on which the seeds agree (MCNS → BANC: predicted 44.5 %, true 1.4 %), and a shared systematic error in a partial volume (hemibrain super classes with the source prior: 0.94 seed agreement, near-perfect size match, 34.7 % correct).
- **Seed agreement ranks types and neurons where the transfer basically works.** Per predicted type, agreement correlates with F1 (Spearman 0.63 FAFB → MCNS, 0.66 MCNS → FAFB, 0.54 FAFB → BANC). Share of all errors caught by reviewing the least-agreed neurons (random review catches 10 / 20 / 30 %):

| Transfer | Errors | 10 % reviewed | 20 % | 30 % | Oracle at 20 % |
|---|---|---|---|---|---|
| FAFB → MCNS | 7.4 % | 52 % | **77 %** | 87 % | 100 % |
| MCNS → FAFB | 16.6 % | 35 % | 55 % | 72 % | 100 % |
| FAFB → BANC | 38.3 % | 20 % | 34 % | 47 % | 52 % |
| FAFB → hemibrain | 80.4 % | 12 % | 23 % | 35 % | 25 % |

- When most labels are wrong (hemibrain, collapsed MCNS → BANC), ranking cannot help: flagging there is no better than random.

## Limitations

- Cell types were defined partly from connectivity, including matching across hemispheres and brains (Schlegel et al., 2024), so recovering them from wiring is partly circular. The GNN never sees partner identities, which weakens this concern but does not remove it. Super classes are mostly defined anatomically.
- Most results predict only types with ≥ 20 neurons (434 of 8,772 types). Lower minimums down to 2 neurons (8,658 types) were tested within FAFB, for the FAFB → MCNS transfer and, with graph matching, in pre-registered tests on the BANC brain and on two nerve cords (from MANC); graph matching was also tested within FAFB and on the FAFB → MCNS development pair. All rare-type references come from alignments that use connectivity in part. Types with a single neuron are left out.
- The setting is transductive: the whole graph is visible during training, only the labels of validation and test neurons are hidden.
- Two animals of different sex, reconstructed and annotated by different teams. MCNS labels come from Janelia's `flywireType` mapping; 40 of the 434 FAFB types do not occur in it (some are split differently, e.g. Dm8a/b vs. yDm8/pDm8), and 2,751 neurons with several mapped types are not scored (a prediction falls within their type list for only 3–4 % of them).
- The label-free pipeline (type-consistency selection, prior matching) was developed on the FAFB → MCNS pair while looking at its transfer errors. Run unchanged in the reverse direction it still helps (83.4 % on cell types vs. at most 78.5 % without prior matching). In the independent test on BANC, prior matching carried over but the selection by type consistency failed in three of four runs (see above). The revised rule (type sizes + seed agreement) was chosen on a sandbox of the same two development connectomes; on the hemibrain it stayed within 4.1 points of the best candidate for super classes but missed it by 6.4 points for cell types.
- Prior matching assumes that type sizes are similar across flies; sex-specific and dimorphic types (about 5 % of the central brain) violate this. For partial volumes the coverage prior needs the list of covered neuropils, and the transfer itself fails (hemibrain).
- Each seed changes both the data split and the initialization; the std covers both.
- The cell-type GNN is trained for 300 epochs and its best epoch is close to the end, so it may be slightly underestimated.
- Codex drops connections with fewer than 5 synapses. The thinned FAFB copies therefore cannot contain the weak connections that a sparse reconstruction still has, and they overstate what a density-matched source gains on BANC.
- FAFB, MCNS and BANC differ in proofreading state, synapse detection and annotation schemes; transfer results depend on how neurons and classes are aligned. BANC's cell types come from an alignment to FAFB that uses connectivity.
- The synthetic graph is for testing only and says nothing about biology.
