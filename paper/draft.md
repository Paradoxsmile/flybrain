# Typing the rare neurons of an unlabeled connectome from wiring alone: a transferred GNN selects the candidates, graph matching types them

*Draft v0.2, October 2026. Maximilian Vogt. Every number comes from a result file in the repository; [docs/reproduce.md](../docs/reproduce.md) lists the command and file for each table. Figures are made by `paper/make_figures.py`, `stratify.py` and `label_effort.py`. Items in [brackets] are open.*

## Abstract

New connectome reconstructions are typed by matching their neurons to an annotated reference, and about three quarters of the 8,772 cell types of the FlyWire brain have only two to four neurons. Connectivity-based typing methods read the labels of a neuron's partners, so they need a partly labeled target. We type rare neurons without any target label by combining two label-free steps. A graph neural network that sees only anonymous wiring, transferred from the reference, selects the neurons likely to belong to rare types. Graph matching against the reference's rare-type neurons then types them. Within FlyWire, graph matching pairs rare-type neurons across hemispheres far better than the GNN (92 vs. 76 %), and the label-free combination reaches 86 %. Across animals it lifts FlyWire's types of 2–4 neurons in the male CNS from 54 to 75 % (82 % with candidate selection improved on development data). Three pre-registered tests on held-out data confirm the gain. On the sparse BANC brain it rises from 22 to 41 %. From the male nerve cord MANC to the nerve cords of BANC and the male CNS, it rises from 40 to 61 % and from 72 to 82 %, closing 60–68 % of the gap to oracle candidates. References built from morphology or from curated typing give the same gains. The GNN underneath is a directed GNN with synapse-weighted aggregation. It types 434 common FlyWire types at 95 % and keeps 87–95 % on a hemisphere without labels, where methods that read partner labels fall below 10 %. With prior matching of type sizes and label-free model selection, it transfers to the male CNS at 92.6 %, mostly in the optic lobe. The pre-registered tests also show the limits of the approach. On the sparse BANC brain the original selection rule failed. A partial volume, the hemibrain, cannot be typed (19.6 %). A fallback tuned on the brain did not carry over to the nerve cord. On complete targets, NTAC with about 1.3 % of each type labeled by hand matches the transferred GNN on common types.

## 1 Introduction

Electron-microscopy connectomes of the adult *Drosophila* central nervous system now exist for several individuals: the female FAFB/FlyWire brain (Dorkenwald et al., 2024; Schlegel et al., 2024), the hemibrain (Scheffer et al., 2020), the male ventral nerve cord MANC (Takemura et al., 2024; Marin et al., 2024), the male CNS (MCNS; Berg et al., 2026) and the female brain and nerve cord BANC (Bates et al., 2026). Their value rests on cell-type annotations. New reconstructions are typed by matching their neurons to an annotated reference, by morphology and by connectivity (Schlegel et al., 2024; Bates et al., 2026). Most of the types are small: of FlyWire's 8,772 cell types, about three quarters have two to four neurons. Connectivity-based typing compares each neuron's synaptic partners by type, as in the definition of optic-lobe types (Matsliah et al., 2024) and in NTAC (Schwartzman et al., 2026), and therefore needs labeled partners in the target.

We study the complementary setting: methods that see only the anonymous wiring diagram, the directed graph of neurons and synapse counts, applied to a target in which no neuron is labeled. For common types, a trained GNN can be transferred between connectomes. For rare types, a trained classifier has seen one or two examples per type. These types were defined by matching neurons across hemispheres and brains, and pairing neurons is what graph matching does without labels (Pedigo et al., 2023). We therefore let the transferred GNN decide which target neurons to match and graph matching decide their types. Our contributions:

1. **Label-free typing of rare types by a GNN and graph matching.** Within FlyWire, graph matching types the right-hemisphere neurons of 2–4-neuron types at 92 %, against 76 % for the GNN; when the GNN selects the candidates, the label-free combination reaches 86 %. Between FlyWire and the male CNS it raises these types from 54 to 75 %, and to 82 % with improved candidates (Sections 3.3, 4.3).
2. **Three pre-registered tests of it on held-out data.** On BANC's brain (22 → 41 %), and from MANC to the nerve cords of BANC (40 → 61 %) and MCNS (72 → 82 %), the gain held, also against references built from morphology or curated typing. Candidate selection is the bottleneck: the pipeline closes 44–68 % of the gap to oracle candidates. A fallback added on development data failed, and one expectation was set too high for a strong baseline (Section 4.4).
3. **A wiring-only GNN and label-free transfer.** A Dir-SAGE GNN (Rossi et al., 2023) with synapse-weighted means reaches 95.3 % on 434 FlyWire cell types, close to NTAC (97.0 %), and keeps 87–95 % on an unlabeled hemisphere, where NTAC falls to 8–9 % (Section 4.1). Prior matching of predicted type sizes and a label-free choice of preprocessing transfer it to the male CNS at 92.6 % (83.4 % in the reverse direction), and down to types of 2 neurons at 84.2 % over 7,064 types (Section 4.2). These are development numbers, carried by the optic lobe.
4. **Where label-free transfer fails.** Two further pre-registered tests and a sandbox of degraded development connectomes show that the original selection rule fails on sparse reconstructions, that density explains about half of the loss on BANC, and that partial volumes are beyond the approach (Section 4.5).
5. **What the model is worth to an annotation team.** For common types, the GNN is worth about 1.3 % hand labels by accuracy, and 2–4 times that for small types at very low budgets. Seed agreement flags most errors where transfer works, but run-level accuracy cannot be estimated without labels (Section 4.6).

We report the failures with the same care as the successes. Each of the four pre-registered tests revealed something the development data had hidden.

## 2 Related work

**Cell typing from connectomes.** Fly connectomes have been annotated with thousands of types from morphology and connectivity (Scheffer et al., 2020; Schlegel et al., 2024; Matsliah et al., 2024; Marin et al., 2024). In the optic lobe, types were defined from each neuron's input and output vectors over types (Matsliah et al., 2024). NTAC (Schwartzman et al., 2026) automates this from labeled seeds and exceeds 90 % on the FlyWire brain. Learned approaches mostly use morphology (Troidl et al., 2024) or combine it with connectivity (NeuNet; Liao et al., 2024), where connectivity alone reached 59 % on hemibrain classes. Across datasets, types are matched by NBLAST, connectivity alignment and combinations of both (Schlegel et al., 2024; Bates et al., 2026; Berg et al., 2026). The typing methods read partner labels or are evaluated within one labeled graph; we know of no evaluation of a trained typing model transferred to an unlabeled connectome.

**Graph matching and label-free structure.** Graph matching finds the correspondence between the nodes of two graphs that best aligns their edges; the relaxed quadratic assignment problem is solved efficiently by Frank–Wolfe iterations (FAQ; Vogelstein et al., 2015). Bisected graph matching, which also aligns the connections between hemispheres, pairs bilaterally homologous neurons of the larval connectome from connectivity alone (Pedigo et al., 2023). Adjacency spectral embedding with Gaussian mixtures recovers neuron groups in the larval mushroom body without labels (Priebe et al., 2017). These methods match or cluster neuron by neuron; we use graph matching as the second stage after a transferred classifier has selected the candidates, across animals and connectomes.

**Directed and heterophilic graph learning.** Neurons mostly connect to neurons of other types, a heterophilic setting in which standard message passing struggles (Zhu et al., 2020). Aggregating separately over in- and out-neighbors (Dir-GNN; Rossi et al., 2023) raises effective homophily; our model is Dir-SAGE (Hamilton et al., 2017; Rossi et al., 2023) with synapse-weighted means.

**Domain adaptation without target labels.** Prior matching is a form of label-shift correction (Saerens et al., 2002) and of optimal transport with fixed class marginals (Sinkhorn scaling), and it mirrors reference mapping in single-cell genomics, where batch effects separate a query from an annotated atlas. Our revised selection rule uses agreement between independently trained models, which tracks their error under distribution shift (Jiang et al., 2022).

## 3 Data and methods

**Connectomes** (Table 1). FlyWire FAFB v783 is the source and development connectome for the brain. MCNS v1.0 (brain only) is the second development connectome; its `flywireType` annotation maps FlyWire types onto its neurons. BANC v888 and the hemibrain v1.2.1 were used only in pre-registered tests. For the nerve-cord test, MANC v1.2.1 (harmonized by the BANC project) is the source and the nerve cords of BANC and MCNS are the targets; no nerve cord was used in development. Neurons are nodes, directed edges are connections weighted by synapse count. FAFB connections have at least 5 synapses (Codex); the other connectomes are aligned to the source by fixed, label-free rules (Section 3.4).

*Table 1. Connectomes. Brain columns: labels are the 434 FlyWire types with at least 20 neurons; nerve-cord columns: MANC types with at least 2 neurons (reference A).*

| | FAFB v783 | MCNS v1.0 brain | BANC v888 brain | Hemibrain v1.2.1 | MANC v1.2.1 | BANC v888 cord | MCNS v1.0 cord |
|---|---|---|---|---|---|---|---|
| Role | source, development | development (both directions) | pre-registered tests | pre-registered test | source (cord test) | pre-registered test | pre-registered test |
| Neurons in the graph | 139,255 | 144,725 | 133,549 | 25,348 | 23,650 | 25,929 | 24,073 |
| Connections / synapses per neuron (≥ 5 synapses) | 26.8 / 364 | 35.9 / 494 | 8.6 / 100 | 26.7 / 387 | 55.1 / 969 | 16.9 / 235 | 43.4 / 759 |
| Labeled neurons (types) | 108,647 (434) | 109,901 (394) | 63,332 (409) | 7,639 (202) | 22,883 (3,901) | 22,451 (3,677) | 21,402 (3,670) |
| Particularities | | male; synapse detection differs | sparse; types assigned by connectivity alignment to FAFB; no R1-6 photoreceptors | central brain only; 16 % of neurons cropped | male | female; sparse | male |

**Labels.** For common types we use super classes (10) and cell types with at least 20 neurons in FAFB (434 types, 78 % of neurons). Section 4.2 lowers the minimum to 10 (779 types, 81 %), 5, 3 and 2 neurons (8,658 types, 99 %); the rare-type pipeline uses the ≥ 2 set. The **rare stratum** of a target is its labeled neurons whose type has 2–4 neurons in the source. In the nerve cord, labels are MANC cell types (3,901 types at ≥ 2, 2,452 of them with 2–4 neurons). In every target the labels are coded in the source's label space and used only for evaluation. For new common-type results we recommend the ≥ 10 set; Sections 4.1, 4.5 and 4.6 use ≥ 20, the set their protocols and saved models fix.

### 3.1 Model

Each neuron has six features: in- and out-degree, input and output synapses, and mean synapses per input and output connection (log-scaled, z-scored). A layer computes h'ᵢ = W_self hᵢ + W_in mean_in(h) + W_out mean_out(h), with means over pre- and postsynaptic partners weighted by synapse count. Cell types use 3 layers (300 epochs, class-weighted loss), super classes 4 layers (1,000 epochs); depth and options were chosen by validation macro-F1. With thousands of types the output layer and prior matching are computed in chunks, which reproduces the standard code where both can run. The model never sees a partner's label, so it applies unchanged to an unlabeled graph.

Baselines, ordered by the information they use: majority class; logistic regression on a neuron's own features; label propagation; a partner profile (logistic regression on the share of synapses per partner type, using training labels); NTAC (Schwartzman et al., 2026; a memory-lean reimplementation checked step by step against the published package); GraphSAGE (inputs only, unweighted). Splits: random (70/15/15, stratified) and hemisphere (train and validate on one hemisphere, test on the other). Three seeds throughout.

### 3.2 Label-free transfer

A model trained on the source is applied to the target. Every choice is made without target labels:

- **Candidates.** Each target is built in a few alignment variants (Section 3.4); target features are normalized either with the source's training statistics or with the target's own; prior matching is applied with strength ρ ∈ {0, 0.25, 0.5, 0.75, 1}.
- **Prior matching.** Cell-type sizes are stereotyped across flies, but a transferred classifier shifts neurons between neighboring types. We rescale its class probabilities with weights v (Sinkhorn iterations) so that the expected number of neurons per type matches the type distribution the same model predicts on its source, scaled to the target; ρ interpolates between the raw prediction and full matching (pₖ · vₖ^ρ, renormalized).
- **Selection.** Originally, the candidate whose labeling (mean over three seeds) is most self-consistent on the target: each neuron of a random half gets the label of its most similar neuron in the other half (synapses per partner type, weighted Jaccard), and agreement is chance-corrected (Cohen's κ). The revised rule (Section 4.5), used in all later tests, takes the best mean rank of two other label-free properties: the Jensen-Shannon divergence of the predicted type sizes from the prior, and the agreement between the three seeds' labelings.
- **Refinement.** NTAC on the target, seeded with the neurons on which all three seeds agree.

### 3.3 Two-stage typing of rare types

- **Stage 1** is the label-free transfer of Section 3.2 at the ≥ 2 label set.
- **Candidates** are the target neurons that stage 1 assigns to a rare type (2–4 source neurons). In the first tests, the type after NTAC decides (`gnn`). After the BANC rare-type test, candidate selection was improved on development data (Section 4.4); the improved rule, `union`, takes a neuron if its type before or after NTAC is rare.
- **Stage 2** matches the source's rare-type neurons one-to-one to the candidates. It maximizes the agreement of the two log1p(synapse) adjacency matrices among the selected neurons by Frank–Wolfe on the relaxed quadratic assignment problem (FAQ; Vogelstein et al., 2015), from the barycenter, for 30 iterations. The assignment is block-diagonal by hemisphere and aligns both ipsilateral and contralateral connections (bisected matching; Pedigo et al., 2023). In the nerve cord a third, pooled block holds the neurons that pass through the neck (ascending and descending), whose side annotation follows the neck in MANC and the soma in the targets. Each block is padded to a square; candidates matched to padding are not assigned.
- **Final labeling.** A matched candidate takes the type of its source neuron; every other neuron keeps its stage-1 type.
- **Diagnostic.** With **oracle candidates** (the labeled rare-type target neurons), the matching shows what perfect candidate selection would give.

At the barycenter the gradient is nearly of rank one and the exact assignment solver takes 5–10 minutes, so the first step pairs the leading singular vectors by rank. A block of about 9,300 neurons per side takes 27 minutes and 4.5 GB on a CPU, the nerve-cord blocks under 3 minutes.

### 3.4 Alignment variants

Connectomes differ in synapse detection and completeness. Each target gets variants that align one statistic with the source by a fixed rule: `raw` (threshold 5, nothing aligned), `density` (the threshold whose connections per neuron are closest to the source's) and `scaled` (threshold 5, counts scaled so that synapses per neuron match the source's). For MCNS against FAFB we use Janelia's published scaling factor (0.581) for the last rule. In the nerve-cord test, self-connections and connections between two passing neurons, which in the targets can lie in the brain, are dropped in all three graphs.

### 3.5 Pre-registration and development sandbox

For each test on a new dataset, data, labels, candidates, selection rule, method, the reported numbers and explicit expectations were committed to the repository before any model was run on it: `docs/banc_protocol.md` (BANC brain, common types), `docs/hemibrain_protocol.md`, `docs/banc_rare_protocol.md` (BANC brain, rare types) and `docs/vnc_rare_protocol.md` (nerve cords). The protocols allow only data-loading and memory fixes afterwards and require every deviation to be reported. Between tests, methods were changed only on a sandbox built from the development connectomes, with known labels: synapse counts binomially thinned before the threshold, cell types removed, the volume restricted to the central brain, or a fraction of neurons cropped (each synapse of a cropped neuron kept with probability 0.1). For partial volumes, a coverage prior uses the type distribution predicted for the FAFB neurons inside the covered volume, those with at least 10 % of their synapses outside the optic-lobe neuropils; it needs only the list of covered neuropils.

## 4 Results

### 4.1 Within a connectome: anonymous wiring suffices, and is the only thing that works on an unlabeled hemisphere

![Benchmark](../docs/benchmark_light.png)

*Figure 1. Test accuracy on FlyWire per method and split (mean over 3 seeds, whiskers: std).*

*Table 2. FlyWire, test accuracy % / macro-F1 (mean ± std, 3 seeds).*

| Method | Cell types, random | left → right | right → left | Super classes, random | left → right | right → left |
|---|---|---|---|---|---|---|
| Majority class | 7.8 / 0.00 | 7.5 / 0.00 | 8.1 / 0.00 | 55.9 / 0.07 | 55.9 / 0.07 | 56.1 / 0.07 |
| Logistic regression (own degrees) | 40.9 / 0.12 | 37.3 / 0.09 | 39.0 / 0.11 | 67.6 / 0.31 | 67.2 / 0.32 | 66.7 / 0.32 |
| Label propagation | 14.7 / 0.28 | 10.7 / 0.19 | 10.3 / 0.20 | 80.1 / 0.30 | 73.4 / 0.22 | 72.6 / 0.21 |
| Partner profile (partner labels) | 92.2 / 0.79 | 8.7 / 0.10 | 1.2 / 0.10 | 90.9 / 0.60 | 47.6 / 0.28 | 45.7 / 0.23 |
| NTAC (partner labels, iterative) | **97.0 / 0.93** | 8.4 / 0.20 | 8.8 / 0.19 | 94.4 / 0.78 | 78.4 / 0.49 | 82.4 / 0.58 |
| GraphSAGE | 87.1 / 0.77 | 85.2 / 0.72 | 79.2 / 0.69 | 95.3 / 0.62 | 95.1 / 0.63 | 94.4 / 0.63 |
| GNN (Dir-SAGE, weighted) | 95.3 / **0.93** | **94.5 / 0.90** | **86.7 / 0.87** | **97.9 / 0.85** | **97.9 / 0.84** | **96.6 / 0.76** |

With labeled partners, NTAC is the best cell-type method; the GNN comes close (95.3 vs. 97.0 %, equal macro-F1) without seeing any partner label and is best on super classes. On an unlabeled hemisphere, the median test neuron has no synapse to a labeled partner (optic-lobe circuits stay on their side), and partner-label methods collapse on cell types (1–9 %; NTAC stays below 9.4 % at any of 40 steps), while the GNN keeps 87–95 %. Aggregating over outputs as well as inputs adds 2–9 points over GraphSAGE of the same depth. Cell types are strongly heterophilic: only 3 % of synapses connect neurons of the same type. The right → left gap of 8 points concentrates on photoreceptors and lamina neurons, whose outgoing synapses are underdetected in FlyWire (Matsliah et al., 2024) and whose connection counts differ between the hemispheres.

The result replicates within MCNS: the GNN reaches 96.6 % (random), 95.4 % and 93.7 % (hemispheres) on cell types, NTAC 7.8–9.5 % on the hemispheres. Seeding NTAC with the predictions on which three GNNs agree yields 96.5 / 90.8 % (FlyWire) and 97.1 / 96.2 % (MCNS) on the unlabeled hemisphere.

### 4.2 Between complete connectomes: label-free transfer, down to types of two neurons

![Transfer](figures/transfer.png)

*Figure 2. Label-free transfer per scenario (accuracy on labeled target neurons). Open circles: chosen preprocessing without prior matching; filled circles: the label-free choice with prior matching; diamonds: after NTAC; triangles: the revised selection rule; bars: a GNN trained with labels on the target. \* pre-registered tests; their filled circles and diamonds use the rule fixed in the protocol (consistency for BANC, the revised rule for the hemibrain).*

A single FlyWire GNN applied to MCNS with a label-free normalization reaches 73–75 % on cell types; the best alignment, found only post hoc, 85.7 %. The label-free pipeline chooses the alignment and normalization by type consistency, which ranks the 30 candidates like their accuracy (Spearman 0.95–0.97), and prior matching corrects the type sizes: 87.6 % without and 91.2 % with prior matching, and **92.6 % after NTAC, over 394 cell types without a single MCNS label**, against 96.6 % for a GNN trained on MCNS. Prior matching repairs the types the transfer over- or under-fills (T5a: 2,252 → 1,714 predicted neurons for 1,664 true ones; LC21: 13 → 115 for 153). Run unchanged in the reverse direction, the pipeline reaches 83.4 % on FlyWire (72.0 % without prior matching; 95.2 % trained within FlyWire). The larger gap is specific to FlyWire as a target: 2.2 % of its neurons belong to types absent from MCNS's mapping, photoreceptors cause 21 % of the errors (51 % correct in the left hemisphere, 83 % in the right), and Kenyon cell subtypes 15 %.

**Where the accuracy comes from.** The headline number is mostly the optic lobe (Figure 3). 80 % of MCNS's labeled neurons belong to optic-lobe intrinsic types; they are typed at 97.2 % after NTAC and contribute 78.0 of the 92.6 points. The other 20 % are typed at 74.0 % (macro-F1 0.68). Visual projection neurons transfer almost as well as the optic lobe (95.7 %), central-brain types much worse (65.1 %; neuron-weighted F1 0.61). The 71 central-brain types with at least 20 FlyWire neurons are a small part of the central brain dominated by Kenyon cells, whose subtypes account for 65 % of these neurons and 84 % of their errors (the other central-brain types: 84.5 %). Among sensory neurons, photoreceptors are typed at 96 %, olfactory receptor neurons at 18 %. Accuracy also falls with type size, from 95.1 % for types with at least 1,000 FlyWire neurons to 77.0 % for types with 20–49, but size and region are confounded: 33 of the 39 largest types are optic-lobe intrinsic. Averaged over the 394 MCNS types, macro-F1 is 0.81 (0.76 when predicted types absent from MCNS are counted). The pattern repeats in the reverse direction (optic lobe 92.6 %, all other neurons 60.9 %) and on BANC (68.6 % and 44.9 %). Prior matching helps central-brain types most (+11, +16 and +26 points in the three runs).

![Per-type F1](figures/type_f1.png)

*Figure 3. FAFB → MCNS after the label-free pipeline (+ NTAC): F1 per cell type on MCNS against the type's number of neurons in FlyWire, by region (FlyWire super class of the type). Dashed lines: the type-size bins of the stratified table.*

**Smaller types.** We retrained the GNN with minimums of 10, 5, 3 and 2 neurons per type, everything else unchanged, and repeated the label-free FAFB → MCNS transfer (Table 3).

*Table 3. Cell types by minimum type size, accuracy % / macro-F1 (macro-F1: mean over the true types present). FlyWire: mean ± std over 3 seeds. Transfer: label-free pipeline + NTAC.*

| Min. size | Types (FlyWire neurons) | FlyWire, left → right | FAFB → MCNS (MCNS types) | MCNS central-brain types | Rarest bin: left → right / transfer |
|---|---|---|---|---|---|
| 20 | 434 (78 %) | 94.3 ± 0.7 / 0.92 | 92.6 / 0.81 (394) | 65.1 / 0.80 (71) | 20–49: 90.7 / 77.0 |
| 10 | 779 (81 %) | 93.1 ± 1.1 / 0.87 | **93.0 / 0.80** (649) | 77.7 / 0.79 (243) | 10–19: 86.6 / 75.5 |
| 5 | 2,164 (87 %) | 89.1 ± 0.9 / 0.77 | 90.3 / 0.71 (1,694) | 72.3 / 0.70 (1,181) | 5–9: 76.4 / 71.6 |
| 3 | 4,175 (93 %) | 87.2 ± 0.8 / 0.72 | 87.6 / 0.57 (3,206) | 62.5 / 0.54 (2,542) | 3–4: see ≥ 2 |
| 2 | 8,658 (99 %) | 83.8 ± 1.1 / 0.71 | 84.2 / 0.51 (7,064) | 64.3 / 0.54 (5,371) | 2–4: 75.9 / 54.2 |

Lowering the minimum from 20 to 10 costs nothing after transfer: no type-size bin is worse (20–49 neurons: 80.4 vs. 77.0 %), and 255 types are added. Below 10, each step trades accuracy on common types for coverage: at ≥ 2 the 20–49 bin falls to 73.7 % and the largest types (≥ 1,000 neurons) from 95.1 to 92.6 %. With 243 central-brain types at ≥ 10, central-brain accuracy rises to 77.7 %, and with thousands of rare ones it settles at 62–64 %. Optic-lobe and visual projection types stay at 93–97 % at every threshold, while small ascending, descending and other types fall to 32 % at ≥ 2. Prior matching matters more with many types (+13.2 points at ≥ 2, +3.6 at ≥ 20). NTAC still adds 3–4 points of accuracy but lowers macro-F1 at ≥ 3 and ≥ 2 (0.62 → 0.57, 0.59 → 0.51), because it pulls neurons of rare types into common ones. Rare types transfer, but with a widening gap: types of 10–19 FlyWire neurons are typed in MCNS at 75.5 %, of 5–9 neurons at 71.6 % (1,045 types) and of 2–4 neurons at 54.2 % (5,370 types, 13,742 neurons), against 86.6, 76.4 and 75.9 % within FlyWire across hemispheres.

### 4.3 Rare types: the GNN selects, graph matching types

Having seen one or two neurons of a type on the left hemisphere, the GNN types its right-hemisphere neurons at 75.9 % (6,438 types of 2–4 neurons, out of 8,582). These types were defined by matching neurons across hemispheres (Schlegel et al., 2024), so the task is close to pairing homologs. We matched the left and right neurons of 2–4-neuron types by graph matching (Section 3.3) and gave each right neuron the type of its matched left neuron (Table 4). Knowing which right neurons are rare-type is an advantage the GNN does not have, so we also restricted the GNN's votes to the same candidates and made them one-to-one.

*Table 4. The 8,142 right-hemisphere FlyWire neurons of 2–4-neuron types, typed from the left hemisphere, accuracy % / macro-F1.*

| Method | Labels used on the right | Accuracy / macro-F1 |
|---|---|---|
| GNN (mean of 3 seeds) | none | 75.9 / 0.74 |
| GNN votes, restricted to rare candidates, one-to-one | which neurons are rare-type | 79.8 / 0.81 |
| Graph matching, ipsilateral | which neurons are rare-type | 91.2 / 0.92 |
| Graph matching, bisected | which neurons are rare-type | **92.2 / 0.93** |
| GNN selects rare-type candidates, graph matching types them | none | **85.8 / 0.88** |

Graph matching beats the equally restricted GNN by 12 points and wins in every region, although it sees only the edges among about 8,000 neurons per side; starting it from the GNN's votes changes nothing. The two methods combine without labels: the GNN's rare-type candidates contain 93 % of the true rare-type neurons (84 % precision), and matching them raises the rare types from 78.9 to 85.8 % (GNN majority vote) and macro-F1 over all right-hemisphere neurons from 0.75 to 0.84.

Across animals the same division of labor works (Table 5). On FAFB → MCNS, stage 1 assigns 18,479 MCNS neurons to a 2–4-neuron type; these candidates contain 84 % of the labeled rare-type neurons. We matched them to the 16,365 FlyWire neurons of such types.

*Table 5. MCNS neurons of 2–4-neuron FlyWire types (13,742 neurons, 5,370 types), without any MCNS label, accuracy % / macro-F1. FAFB → MCNS is the development pair.*

| Method | 2–4-neuron types | All labeled MCNS neurons |
|---|---|---|
| Label-free GNN pipeline (stage 1) | 54.2 / 0.49 | 84.2 / 0.51 |
| Stage-1 probabilities, made one-to-one | 59.9 / 0.62 | 84.8 / 0.61 |
| Stage 1 selects candidates, graph matching types them | **74.5 / 0.78** | **86.3 / 0.73** |
| Same, matching started from the GNN probabilities | 74.6 / 0.78 | 86.3 / 0.74 |
| Labeled rare-type neurons as candidates (oracle) | 85.3 / 0.87 | 87.4 / 0.81 |

The two stages raise the rarest types by 20 points and macro-F1 over all labeled MCNS neurons from 0.51 to 0.73. The wiring does the work: matching without any GNN input is as good as matching started from the GNN's probabilities (they agree on 93 % of candidates), while making the GNN's own probabilities one-to-one gains only 6 points. On the rare-type neurons stage 1 selects, matching reaches 89.1 % (stage 1: 64.8 %), and the largest gains are where the GNN is weakest (ascending, descending and other types: 25 → 66 %; central brain: 58 → 75 %). Graph matching between two animals is nearly as accurate as between two hemispheres (85.3 % with oracle candidates, against 92.2 %), so the remaining loss is mostly candidate selection. Two caveats limit this result: the stage-1 pipeline was developed on this pair, and MCNS's mapping to FlyWire types for these rare types itself comes from cross-dataset matching that uses connectivity.

### 4.4 Pre-registered tests of the rare-type pipeline

**BANC brain.** We tested the two-stage pipeline on BANC under a protocol (`docs/banc_rare_protocol.md`, committed with the code in `0d1ee67` before either stage ran on BANC at this label set). BANC had been used before only for the ≥ 20 pipeline (Section 4.5). The protocol fixed the stage-1 selection rule (the revised rule, which here chose the best of the 30 candidates), the matching (unchanged, `gnn` candidates), two references and four expectations (Table 6). Reference B, the type of the FlyWire neuron matched to each BANC neuron by morphology, is independent of connectivity.

*Table 6. Pre-registered test: BANC neurons of 2–4-neuron FlyWire types, without any BANC label, accuracy % / macro-F1. Reference A: BANC's typing; reference B: the type of the morphology-matched FlyWire neuron (central brain).*

| Method | Reference A (14,512 neurons) | Reference B (13,848) | All reference-A neurons (88,639) |
|---|---|---|---|
| Label-free GNN pipeline (stage 1) | 22.2 / 0.19 | 21.3 / 0.18 | 47.9 / 0.19 |
| Two-stage | **41.3 / 0.47** | **40.6 / 0.47** | **51.0 / 0.40** |
| Labeled rare-type neurons as candidates (oracle) | 65.9 / 0.70 | 65.5 / 0.70 | 55.0 / 0.57 |

All four expectations held: the two-stage pipeline beat stage 1 by at least 10 points on both references (+19.1 and +19.3, against +20.3 on MCNS); over all labeled neurons macro-F1 rose by at least 0.05 (0.19 → 0.40) without a loss in accuracy (+3.1 points); the rare stratum stayed below FAFB → MCNS (41.3 vs. 74.5 %); and matching withstood BANC's sparsity better than the GNN (with oracle candidates it kept 77 % of its MCNS accuracy, stage 1 41 %). Unlike for common types (Section 4.5), the morphology-matched reference gives the same result, so the gain is not an artifact of connectivity-based labels. The absolute level is limited by candidate selection: stage 1 finds only 54 % of the rare-type neurons on BANC (84 % on MCNS), while matching types 77 % of those it finds correctly.

**Improving candidate selection on development data.** Because BANC was then spent for rare types, we improved candidate selection on development data only. MCNS thinned to 25 % of its synapses reproduces BANC's stage 1 (candidate recall 54.4 vs. 54.3 %, rare-stratum accuracy 19.2 vs. 22.2 %) and serves as a stand-in (Table 7).

*Table 7. Candidate selection on development data: accuracy % / macro-F1 on the 2–4 stratum of MCNS, full and thinned to 25 % of its synapses.*

| Candidates | Recall, thinned | Thinned (25 %) | Full MCNS |
|---|---|---|---|
| Stage-1 type after NTAC is 2–4 (`gnn`, as tested on BANC) | 54.4 % | 47.3 / 0.53 | 74.5 / 0.78 |
| Stage-1 type before or after NTAC is 2–4 (`union`) | 70.0 % | 60.1 / 0.65 | 81.6 / 0.83 |
| Labeled rare-type neurons (oracle) | – | 82.0 / 0.84 | 85.3 / 0.87 |

Selecting more candidates helps; ranking them better does not. At equal count, the GNN's total probability on rare types selects no better than its top type (under 1 recall point everywhere), but the union of the rare-type predictions before and after NTAC raises recall from 54 to 70 % on the thinned copy and from 84 to 92 % on full MCNS. Matching stays accurate on the added neurons (81 % on the rare-type neurons added in the thinned copy), and most added false positives end on padding. The rule has no free parameter, and a second thinning realization gives the same result within 1–2 points. Two further changes were tested. A two-tier fallback, which types the neurons that are not matched with the coarser ≥ 10 model, added 2.5–3.7 points over all labeled neurons without changing the rare types; it was adopted and then failed in the nerve cord. Letting a FlyWire neuron match two candidates, to absorb differences in type size, failed already here: with oracle candidates it fell from 85.3 to 76.1 %, because candidates piled onto a few well-connected neurons.

**Nerve cords.** We tested the improved pipeline in the nerve cord under a fourth protocol (`docs/vnc_rare_protocol.md`, committed with the code in `5119069` before any model ran on a target cord). The source is MANC, on which the GNNs were trained unchanged; the targets are the nerve cords of BANC (female, 31 % of MANC's connections per neuron) and MCNS (a second male). The method was the one developed above (`union` candidates, two-tier fallback), with one disclosed change made before any model output on a cord: the pooled block for neurons that pass through the neck (Section 3.3). Reference A is each target's alignment to MANC, which for intrinsic neurons rests partly (BANC) or mainly (MCNS) on connectivity; reference B is BANC's curated typing (Table 8).

*Table 8. Pre-registered nerve-cord test: neurons of 2–4-neuron MANC types, without any target label, accuracy % / macro-F1.*

| Method | BANC cord, A (5,989) | BANC cord, B (5,257) | MCNS cord, A (5,945) |
|---|---|---|---|
| Label-free GNN pipeline (stage 1) | 39.5 / 0.36 | 40.7 / 0.37 | 72.0 / 0.70 |
| Two-stage, `gnn` candidates (as on the BANC brain) | 52.1 / 0.55 | 51.7 / – | 77.1 / 0.79 |
| Two-stage, `union` candidates | **61.0 / 0.61** | **61.0 / 0.61** | **82.2 / 0.82** |
| Labeled rare-type neurons as candidates (oracle) | 75.5 / 0.73 | 75.5 / – | 87.0 / 0.86 |

Two of the four expectations held on both cords, and two did not. The improved candidates beat the old ones by at least 3 points (+8.9 and +5.1), and the pipeline closed at least half of the gap between stage 1 and oracle candidates (0.60 and 0.68, against 0.44 for the old pipeline on the BANC brain). The gain of at least 15 points over stage 1 held on BANC (+21.4; +20.3 against reference B) but not on MCNS (+10.2). There stage 1 was already at 72.0 % and the oracle at 87.0 %, so the threshold could only have been met by closing the whole gap; we had set it from earlier tests, where stage 1 lay at 19–54 %, and should have set it relative to the gap. The two-tier fallback failed on both cords: the coarser model typed the neurons it covers worse than the fine one (types of 10–19 neurons: 55.7 vs. 52.8 % on BANC, 76.1 vs. 72.5 % on MCNS) and cost 0.4 and 1.2 points over all labeled neurons, the opposite of the brain. Within MANC, the coarser model has little advantage either (78.1 vs. 70.9 %; FlyWire 94.2 vs. 82.9 %), a likely but untested reason. We therefore drop the fallback from the recommended pipeline; the rare-type numbers do not depend on it, and without it the pipeline still gains over all labeled neurons (BANC 44.8 → 49.4 %, MCNS 73.1 → 75.0 %). Every neuron group gained on both cords (on BANC, descending neurons 41 → 74 %, motor neurons 16 → 48 %), and intrinsic neurons gained no more than the others. Without the pooled block the rare types lost 5.6–5.7 points. The cost is borne by types of 5–19 neurons (−1.5 to −2.5 points), because a fifth to a third of the candidates belong to other types and receive a rare type.

### 4.5 Where label-free transfer fails: sparse and partial reconstructions

**BANC brain, common types.** The first pre-registered test applied the unchanged ≥ 20 pipeline to BANC (`docs/banc_protocol.md`, `85516a2`), with selection by type consistency (Table 9).

*Table 9. BANC, accuracy % / macro-F1, as pre-registered (selection by type consistency).*

| Run, label | Without prior matching | With prior matching | + NTAC | Best candidate (post hoc) |
|---|---|---|---|---|
| FAFB → BANC, cell types | 38.2 / 0.21 | 55.7 / 0.37 | **61.7 / 0.40** | 55.7 |
| FAFB → BANC, super classes | 42.2 / 0.11 | 72.3 / 0.24 | **73.3 / 0.22** | 92.6 |
| MCNS → BANC, cell types | 1.4 / 0.00 | 1.4 / 0.00 | **1.4 / 0.00** | 58.0 |
| MCNS → BANC, super classes | 69.0 / 0.34 | 78.7 / 0.33 | **79.0 / 0.31** | 94.3 |

Of three pre-registered expectations, the first (selected candidate within 2 points of the best) held in one of four runs; the second (the pipeline beats every candidate without prior matching on cell types) held for FAFB → BANC only; the third (cell-type accuracy below FAFB → MCNS) held. Prior matching carried over: the best cell-type candidates use full prior matching, 13.0 and 17.5 points above the best candidates without it, although BANC lacks the R1-6 photoreceptors that the prior expects (prior matching fills R1-6 with 10,800 neurons, 20 % of the remaining errors). The selection did not: for MCNS → BANC cell types it chose a labeling that puts 77 % of the neurons into one type; such a labeling is self-consistent on the graph, and the chance correction does not penalize the collapse enough. Against the type of the morphologically matched FAFB neuron (central brain only), FAFB → BANC reaches 32.3 % after NTAC; the optic lobe reaches 67.4 % against BANC's own types.

**Density.** BANC's brain graph is sparse: the median neuron has 30 input synapses from other neurons in the graph and 4 input partners (FlyWire: 152 and 14). Thinned copies of FlyWire, which keep each synapse with probability p before the threshold of 5, reproduce BANC's statistics at p ≈ 0.35–0.4. Transfer between independently thinned copies isolates density from all other differences (Figure 4).

![Thinning](figures/thinning.png)

*Figure 4. Transfer between thinned FlyWire copies, with prior matching (test split, mean over 3 seeds). Rows: density of the source copy the GNN was trained on; "augmented": one GNN trained with a random density in [0.25, 1] per epoch.*

- **Density explains about half of the drop on BANC**: the full-density GNN reaches 72.9 % on a copy at BANC's density, against 55.7 % on BANC.
- **A sparser target is best served by a source of its density** (83.9 vs. 72.9 % at p = 0.4), but accuracy falls even then (95.1 → 67.7 % on the diagonal): a sparse reconstruction loses information.
- **A denser target is benign with prior matching**: a GNN trained at p = 0.4 reaches 87.0 % on the full-density copy (70.0 % without prior matching).
- **One GNN trained with random thinning approaches the density-matched GNNs on every target** (94.0, 90.9, 83.7, 63.7 % against 95.1, 91.3, 83.9, 67.7 %), without knowing the target's density.
- On BANC itself, neither density-matched nor augmented GNNs beat the full-density GNN (51.2 and 55.0 vs. 55.7 %, exploratory). There, the `density` variant, which lowers the threshold to 2 synapses, recovers strong connections that survive as weak ones; the thinned copies cannot show this, because FlyWire's public connections start at 5 synapses.

**Selection.** Degraded copies of the development connectomes with known labels (FAFB → MCNS at 100, 50 and 20 % of MCNS's synapses, MCNS → FAFB at 100 and 35 %, FAFB → MCNS without R1-6, with and without bounded prior matching) reproduce the BANC failure: at 20 % density, type consistency ranks the candidates inversely to their accuracy (Spearman −0.68 for cell types). Table 10 compares label-free selection rules.

*Table 10. Gap between the chosen and the best candidate, points, mean / worst.*

| Rule | Sandbox (14 cases) | BANC re-scored (6 cases, exploratory) |
|---|---|---|
| Type consistency (original) | 4.1 / 26.7 | 19.9 / 56.6 |
| Type sizes closest to the prior | 4.1 / 26.0 | 0.6 / 1.8 |
| Consistency + seed agreement, collapse guard | 1.1 / 8.9 | 1.5 / 5.8 |
| **Type sizes + seed agreement (revised)** | **1.3 / 4.0** | **1.1 / 6.3** |

*Cases are runs × labels. BANC: the four pre-registered runs (FAFB → BANC and MCNS → BANC, cell types and super classes) and, exploratory, FAFB → BANC with GNNs trained on FlyWire thinned to p = 0.4 (both labels). The worst case of the revised rule (6.3) is the density-matched super-class run.*

Type consistency is the best single signal on complete graphs and fails on sparse ones; the type-size divergence behaves the other way round; seed agreement tracks accuracy closely on complete graphs (Spearman 0.98 for MCNS → FAFB cell types). The revised rule had the smallest worst case on the sandbox and was chosen there, before it was computed on BANC. Re-scored on BANC (the candidates' accuracies were known from the pre-registered runs), it lands within 0.5 points of the best candidate in all four pre-registered runs, including the collapsed one (58.0 instead of 1.4 %). In the two pre-registered rare-type tests on new targets it chose the best candidate in four of five stage-1 runs and missed by 1.9 points in the fifth. Two candidate fixes did not help: bounding prior matching to avoid filling missing types (fewer wrong R1-6, but lower accuracy), and estimating target type sizes by EM (Saerens et al., 2002), which amplified the classifier's errors.

**Partial volumes.** The hemibrain covers the central brain without the optic lobes; 77 % of FlyWire's labeled neurons belong to types it does not contain. This breaks the shared assumption of prior matching, the size-based selection and normalization by the target's own statistics: that the target has the source's composition. In the sandbox (MCNS restricted to the central brain, 16 % cropped), the best candidate for the same central-brain neurons fell from 69.9 % (full volume) to 46.9 % (partial) and 36.3 % (partial and cropped); the coverage prior raised it from 30.5 to 36.3 % and was adopted. The second pre-registered test (`docs/hemibrain_protocol.md`, `161cf7b`) was also the first test of the revised rule. The harmonized hemibrain edgelist repeats rows (931,329 exact copies), which we drop; within its volume the hemibrain then has FlyWire's density (Table 11).

*Table 11. Hemibrain, accuracy % / macro-F1, as pre-registered (revised rule, coverage prior).*

| Label | Without prior matching | With prior matching | + NTAC | Best candidate (post hoc) | Original consistency rule |
|---|---|---|---|---|---|
| Cell types | 17.3 / 0.09 | 19.1 / 0.09 | **19.6 / 0.09** | 25.5 | 7.8 |
| Super classes | 72.8 / 0.20 | 74.3 / 0.20 | **74.4 / 0.19** | 78.4 | 72.8 |

Of four expectations, the selection was within 5 points of the best candidate for super classes (4.1) but not for cell types (6.4); the coverage prior gave a better best candidate than the FlyWire prior, by little for cell types (25.5 vs. 24.9 %) and decisively after selection for super classes (74.4 vs. 34.5 %); complete neurons were typed better than cropped ones (22.8 vs. 12.2 % on cell types); and cell-type accuracy stayed below 60 %. Super-class accuracy is barely above predicting the most frequent class (73.1 %); visual projection neurons, whose dendrites lie in the missing optic lobe, are typed correctly in 5 % of cases, central-brain neurons in 28 %. Partners outside the volume and cut neurons remove the context the model relies on.

*Table 12. The four pre-registered tests.*

| Test | Protocol (commit) | What was tested | Expectations |
|---|---|---|---|
| BANC brain, common types | `banc_protocol.md` (`85516a2`) | unchanged label-free pipeline, selection by type consistency | 3: one held in every run, two only in some runs |
| Hemibrain | `hemibrain_protocol.md` (`161cf7b`) | revised rule, coverage prior, partial volume | 4: three held, one held for super classes only |
| BANC brain, rare types | `banc_rare_protocol.md` (`0d1ee67`) | two-stage pipeline, `gnn` candidates | 4: all held |
| Nerve cords, rare types (MANC → BANC, MCNS) | `vnc_rare_protocol.md` (`5119069`) | `union` candidates, two-tier fallback, passing block | 4 per cord: two held on both, one on BANC only, the fallback failed on both |

### 4.6 What the model is worth to an annotation team

A team typing a new connectome would not rely on wiring alone; it can label neurons by hand and propagate the labels with NTAC. On the ≥ 20 label set, we labeled k % of each type's neurons at random (three draws) and compared NTAC seeded with the hand labels only, NTAC seeded with the hand labels plus the GNN predictions on which three seeds agree (fixed seeds, as in Section 4.1), and NTAC with the label-free pipeline as a fallback for types that have no hand label (Figure 5, Table 13).

![Labeling effort](figures/label_effort.png)

*Figure 5. Accuracy and macro-F1 on the neurons without a hand label, against the share of each type labeled by hand (mean over three draws, ≥ 20 label set). Dashed: the label-free pipeline without hand labels.*

*Table 13. Hand labels NTAC alone needs (% of each type) to match GNN-seeded NTAC at k = 1 %.*

| Target | GNN + NTAC at 1 % | NTAC alone, by accuracy | NTAC alone, by macro-F1 |
|---|---|---|---|
| MCNS | 92.7 / 0.81 | 1.2 % | 3.4 % |
| BANC, BANC's types | 62.1 / 0.43 | 1.3 % | 4.4 % |
| BANC, morphology-matched types | 32.8 / 0.27 | 1.3 % | 2.3 % |

By accuracy, the GNN is worth about 1.3 % hand labels on all three references; from 2 % on, NTAC alone is better (MCNS at 5 %: 97.2 vs. 93.2 %; BANC: 73.9 vs. 63.5 %). The GNN's contribution is larger for small types: random labeling at 1 % leaves types with fewer than 50 neurons unlabeled, which NTAC cannot assign, and GNN-seeded NTAC matches the macro-F1 that NTAC alone reaches only with 2.3–4.4 times as many hand labels. Fixed GNN seeds are the wrong way to combine both once hand labels exist, because NTAC never revises them: in the MCNS central brain, GNN-seeded NTAC gains 9 points between 1 and 50 % hand labels, NTAC alone 27 points. The fallback variant is best at 1 % on MCNS (94.1 %, macro-F1 0.81) but hurts on BANC, where the transferred labels are less reliable. The rare-type pipeline was not part of this comparison.

**Can one tell without labels how good a labeling is?** We fitted simple linear predictors of accuracy from the label-free properties of all 1,200 candidate labelings of the development runs and the sandbox, and tested them on BANC and the hemibrain. The pre-specified predictor (seed agreement and type-size divergence) missed by 12.8 points on average (median 8.4). It was accurate on complete targets with a non-collapsed labeling (FAFB → BANC cell types: 0.5 points). It failed when all seeds made the same error: on the collapsed MCNS → BANC labeling (predicted 44.5 %, true 1.4 %) and on hemibrain super classes with the FlyWire prior (seed agreement 0.94, 34.7 % correct). Seed agreement does rank neurons and types where the transfer works: reviewing the 20 % least-agreed neurons finds 77 % of the errors on MCNS, 55 % on FlyWire and 34 % on BANC (random review: 20 %), and per type it correlates with F1 (Spearman 0.54–0.66). On the hemibrain, where 80 % of the labels are wrong, it is no better than random.

## 5 Discussion

**Rare types: pairing beats classifying.** For types with two to four neurons, a trained classifier has seen one or two examples, while graph matching uses the wiring among all candidate neurons at once. Within FlyWire, where the types were defined by pairing neurons across hemispheres, matching wins by 12–16 points. Between animals and connectomes it remains the stronger principle, and the transferred GNN's role is to decide which neurons to match. That division of labor held in three pre-registered tests: across sex, at a third of the source's density, and from the brain to the nerve cord with a new source dataset. It held against references built from morphology (BANC brain) or curated by hand (BANC cord). Intrinsic nerve-cord neurons, whose references rest most on connectivity, gained no more than the others. Candidate selection remains the bottleneck: oracle candidates are 5–25 points better than the selected ones in the three tests. Selecting more generously helped (`union`), and its cost falls on the 5–19-neuron types, which receive some wrong rare types. One-to-one matching ignores that type sizes differ between animals; a looser assignment failed on development data, because candidates piled onto a few well-connected neurons.

**Common types.** Anonymous wiring identifies cell types without labeled partners, within a connectome and across complete connectomes. Prior matching is the most robust ingredient of label-free transfer: it helped on every complete or sparse target (+3.6 to +17.5 points), even with types missing from the target. Of the selection criteria, none is reliable alone; combining a size-based and an agreement-based signal chose the best or a near-best candidate in every later test except the partial volume.

**What does not carry over.** Transfer degrades with reconstruction density and fails for partial volumes. Both remove context the model relies on: in a sparse reconstruction, connections; in a partial volume, partners outside it and parts of cut neurons. For typing a partial volume, a model would need to be trained on correspondingly truncated graphs or combined with morphology. Add-ons tuned on development data are fragile: the two-tier fallback helped on the brain and hurt in the nerve cord.

**Where the approach is useful.** On a complete, dense target, a small hand-labeling budget with NTAC beats the transferred GNN on common types, so for these types the model's niche is the start of an annotation: typing with no labels, covering types that a small budget misses, and pointing reviewers to the neurons most likely to be wrong. Rare types are where label-free typing helps most, because a hand-labeling budget reaches them last; there the two-stage pipeline types 41–82 % of the neurons without a label. No label-free signal we tried detects an error that all seeds share, so a transferred labeling still needs a labeled spot check.

**Circularity.** Cell types were defined partly from connectivity, and cross-dataset labels (MCNS's `flywireType` and `mancType`, BANC's types, FlyWire names for hemibrain types) come from matching that uses connectivity. Agreement with them is therefore partly agreement with another connectivity-based procedure. The GNN never sees partner identities, which weakens but does not remove this concern. For common types, BANC's morphology-matched central-brain types are the cleanest reference we found, and there FAFB → BANC reaches only 32 %; for rare types the same reference gives the same gain as the connectivity-based one.

**On pre-registration.** The first two pre-registered tests contradicted part of what development data suggested, and the third confirmed it. The fourth confirmed the core method but rejected an add-on tuned on development data, and it showed that an absolute threshold can be out of reach when the baseline is already strong; thresholds relative to an oracle are the better form. The sandbox let us diagnose and repair failures without spending a new dataset. Degraded copies of annotated connectomes are a cheap test bed for any method meant to type new reconstructions; we recommend them.

**Limitations.** The rare-type pipeline was developed on one brain pair and tested on one held-out brain and two nerve cords; every rare-type reference except BANC's morphology matches rests partly on connectivity. Single-neuron types are not predicted, and serial homologs in the nerve cord are not separated by the matching blocks. The recommended pipeline drops the fallback after the last test, so its all-neuron numbers come from runs reported in that test but not from a fresh one. The labeling-effort comparison draws hand labels at random within types, while a real team would label every type at least once. The setting is transductive within a connectome. The revised selection rule was chosen on 14 sandbox cases. Thinned copies cannot contain sub-threshold connections. Results depend on how neurons and classes are aligned across datasets.

## 6 Conclusion

Rare cell types, most of a fly connectome's types, can be typed without any target label by letting a transferred wiring-only GNN select the candidates and graph matching type them. The approach roughly doubles accuracy on a held-out brain (22 → 41 %) and, with improved candidate selection, carries over to two nerve cords from a new source (40 → 61 % and 72 → 82 %), in pre-registered tests. The GNN alone types common types of an unlabeled hemisphere or another complete connectome at 83–93 %, mainly in the optic lobe. Sparser reconstructions cost accuracy and need a robust selection rule, partial volumes are out of reach for connectivity alone, and on complete targets a few percent of hand labels with NTAC type common types better. The model is best used to start an annotation, to type the rare types a labeling budget reaches last, and to direct the review.

## Reproducibility

Code, configs, protocols and the commands for every number are in the repository [URL]; `docs/reproduce.md` maps each table to its command and result file, and the small result summaries are committed under `results/`. Data are public (FlyWire Codex, Janelia MCNS, BANC, and the hemibrain and MANC via the BANC project's bucket). Protocols: `docs/banc_protocol.md` (commit `85516a2`), `docs/hemibrain_protocol.md` (`161cf7b`), `docs/banc_rare_protocol.md` (`0d1ee67`), `docs/vnc_rare_protocol.md` (`5119069`).

## References

[To be completed and checked; see `docs/literature.md`, section 5, for open checks.]

- Berg et al. (2026). Sexual dimorphism in the complete connectome of the *Drosophila* male central nervous system. *Cell* 189(18), 5504–5526. doi:10.1016/j.cell.2026.08.015 (bioRxiv 10.1101/2025.10.09.680999). [Check volume, pages and date against the journal page.]
- Bates, Phelps, Kim, Yang et al. (2026). Distributed control circuits across a brain-and-cord connectome. *Nature*.
- Dorkenwald et al. (2024). Neuronal wiring diagram of an adult brain. *Nature*.
- Hamilton, Ying & Leskovec (2017). Inductive representation learning on large graphs. NeurIPS.
- Jiang, Nagarajan, Baek & Kolter (2022). Assessing generalization of SGD via disagreement. ICLR.
- Liao, Wan & Du (2024). NeuNet. AAAI.
- Marin et al. (2024). Systematic annotation of a complete adult male *Drosophila* nerve cord connectome reveals principles of functional organisation. *eLife*.
- Matsliah et al. (2024). Neuronal parts list and wiring diagram for a visual system. *Nature* 634.
- Pedigo, Winding, Priebe & Vogelstein (2023). Bisected graph matching improves automated pairing of bilaterally homologous neurons from connectomes. *Network Neuroscience* 7(2), 522–538. doi:10.1162/netn_a_00287
- Priebe, Park, Tang, Athreya, Lyzinski, Vogelstein, Qin, Cocanougher, Eichler, Zlatic & Cardona (2017). Semiparametric spectral modeling of the *Drosophila* connectome. arXiv:1705.03297.
- Rossi, Charpentier, Di Giovanni, Frasca, Günnemann & Bronstein (2023). Edge directionality improves learning on heterophilic graphs. LoG.
- Saerens, Latinne & Decaestecker (2002). Adjusting the outputs of a classifier to new a priori probabilities. *Neural Computation* 14.
- Scheffer et al. (2020). A connectome and analysis of the adult *Drosophila* central brain. *eLife*.
- Schlegel et al. (2024). Whole-brain annotation and multi-connectome cell typing of *Drosophila*. *Nature*.
- Schwartzman, Jourdan, García-Soriano & Matsliah (2026). NTAC: Neuronal type assignment from connectivity. *Nature Communications* 17.
- Takemura et al. (2024). A connectome of the male *Drosophila* ventral nerve cord. *eLife*.
- Troidl et al. (2024). Global neuron shape reasoning with point affinity transformers. bioRxiv.
- Vogelstein, Conroy, Lyzinski, Podrazik, Kratzer, Harley, Fishkind, Vogelstein & Priebe (2015). Fast approximate quadratic programming for graph matching. *PLoS ONE* 10(4), e0121002.
- Zhu, Yan, Zhao, Heimann, Akoglu & Koutra (2020). Beyond homophily in graph neural networks. NeurIPS.
