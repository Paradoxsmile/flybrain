# Supplementary material

*To the paper draft [draft.md](draft.md). Every number comes from a result file in the repository; [docs/reproduce.md](../docs/reproduce.md) lists the command and file for each table and figure.*

## S1 Smaller types (Section 4.2)

**Smaller types.** We retrained the GNN with minimums of 10, 5, 3 and 2 neurons per type, everything else unchanged, and repeated the label-free FAFB → MCNS transfer (Table S1).

*Table S1. Cell types by minimum type size, accuracy % / macro-F1 (macro-F1: mean over the true types present). FlyWire: mean ± std over 3 seeds. Transfer: label-free pipeline + NTAC.*

| Min. size | Types (FlyWire neurons) | FlyWire, left → right | FAFB → MCNS (MCNS types) | MCNS central-brain types | Rarest bin: left → right / transfer |
|---|---|---|---|---|---|
| 20 | 434 (78 %) | 94.3 ± 0.7 / 0.92 | 92.6 / 0.81 (394) | 65.1 / 0.80 (71) | 20–49: 90.7 / 77.0 |
| 10 | 779 (81 %) | 93.1 ± 1.1 / 0.87 | **93.0 / 0.80** (649) | 77.7 / 0.79 (243) | 10–19: 86.6 / 75.5 |
| 5 | 2,164 (87 %) | 89.1 ± 0.9 / 0.77 | 90.3 / 0.71 (1,694) | 72.3 / 0.70 (1,181) | 5–9: 76.4 / 71.6 |
| 3 | 4,175 (93 %) | 87.2 ± 0.8 / 0.72 | 87.6 / 0.57 (3,206) | 62.5 / 0.54 (2,542) | 3–4: see ≥ 2 |
| 2 | 8,658 (99 %) | 83.8 ± 1.1 / 0.71 | 84.2 / 0.51 (7,064) | 64.3 / 0.54 (5,371) | 2–4: 75.9 / 54.2 |

Lowering the minimum from 20 to 10 costs nothing after transfer: no type-size bin is worse (20–49 neurons: 80.4 vs. 77.0 %), and 255 types are added. Below 10, each step trades accuracy on common types for coverage: at ≥ 2 the 20–49 bin falls to 73.7 % and the largest types (≥ 1,000 neurons) from 95.1 to 92.6 %. With 243 central-brain types at ≥ 10, central-brain accuracy rises to 77.7 %, and with thousands of rare ones it settles at 62–64 %. Optic-lobe and visual projection types stay at 93–97 % at every threshold, while small ascending, descending and other types fall to 32 % at ≥ 2. Prior matching matters more with many types (+13.2 points at ≥ 2, +3.6 at ≥ 20). NTAC still adds 3–4 points of accuracy but lowers macro-F1 at ≥ 3 and ≥ 2 (0.62 → 0.57, 0.59 → 0.51), because it pulls neurons of rare types into common ones. Rare types transfer, but with a widening gap: types of 10–19 FlyWire neurons are typed in MCNS at 75.5 %, of 5–9 neurons at 71.6 % (1,045 types) and of 2–4 neurons at 54.2 % (5,370 types, 13,742 neurons), against 86.6, 76.4 and 75.9 % within FlyWire across hemispheres.

## S2 Density and selection (Section 4.5)

**Density.** BANC's brain graph is sparse: the median neuron has 30 input synapses from other neurons in the graph and 4 input partners (FlyWire: 152 and 14). Thinned copies of FlyWire, which keep each synapse with probability p before the threshold of 5, reproduce BANC's statistics at p ≈ 0.35–0.4. Transfer between independently thinned copies isolates density from all other differences (Figure S1).

![Thinning](figures/thinning.png)

*Figure S1. Transfer between thinned FlyWire copies, with prior matching (test split, mean over 3 seeds). Rows: density of the source copy the GNN was trained on; "augmented": one GNN trained with a random density in [0.25, 1] per epoch.*

- **Density explains about half of the drop on BANC**: the full-density GNN reaches 72.9 % on a copy at BANC's density, against 55.7 % on BANC.
- **A sparser target is best served by a source of its density** (83.9 vs. 72.9 % at p = 0.4), but accuracy falls even then (95.1 → 67.7 % on the diagonal): a sparse reconstruction loses information.
- **A denser target is benign with prior matching**: a GNN trained at p = 0.4 reaches 87.0 % on the full-density copy (70.0 % without prior matching).
- **One GNN trained with random thinning approaches the density-matched GNNs on every target** (94.0, 90.9, 83.7, 63.7 % against 95.1, 91.3, 83.9, 67.7 %), without knowing the target's density.
- On BANC itself, neither density-matched nor augmented GNNs beat the full-density GNN (51.2 and 55.0 vs. 55.7 %, exploratory). There, the `density` variant, which lowers the threshold to 2 synapses, recovers strong connections that survive as weak ones; the thinned copies cannot show this, because FlyWire's public connections start at 5 synapses.

**Selection.** Degraded copies of the development connectomes with known labels (FAFB → MCNS at 100, 50 and 20 % of MCNS's synapses, MCNS → FAFB at 100 and 35 %, FAFB → MCNS without R1-6, with and without bounded prior matching) reproduce the BANC failure: at 20 % density, type consistency ranks the candidates inversely to their accuracy (Spearman −0.68 for cell types). Table S2 compares label-free selection rules.

*Table S2. Gap between the chosen and the best candidate, points, mean / worst.*

| Rule | Sandbox (14 cases) | BANC re-scored (6 cases, exploratory) |
|---|---|---|
| Type consistency (original) | 4.1 / 26.7 | 19.9 / 56.6 |
| Type sizes closest to the prior | 4.1 / 26.0 | 0.6 / 1.8 |
| Consistency + seed agreement, collapse guard | 1.1 / 8.9 | 1.5 / 5.8 |
| **Type sizes + seed agreement (revised)** | **1.3 / 4.0** | **1.1 / 6.3** |

*Cases are runs × labels. BANC: the four pre-registered runs (FAFB → BANC and MCNS → BANC, cell types and super classes) and, exploratory, FAFB → BANC with GNNs trained on FlyWire thinned to p = 0.4 (both labels). The worst case of the revised rule (6.3) is the density-matched super-class run.*

Type consistency is the best single signal on complete graphs and fails on sparse ones; the type-size divergence behaves the other way round; seed agreement tracks accuracy closely on complete graphs (Spearman 0.98 for MCNS → FAFB cell types). The revised rule had the smallest worst case on the sandbox and was chosen there, before it was computed on BANC. Re-scored on BANC (the candidates' accuracies were known from the pre-registered runs), it lands within 0.5 points of the best candidate in all four pre-registered runs, including the collapsed one (58.0 instead of 1.4 %). In the two pre-registered rare-type tests on new targets it chose the best candidate in four of five stage-1 runs and missed by 1.9 points in the fifth. Two candidate fixes did not help: bounding prior matching to avoid filling missing types (fewer wrong R1-6, but lower accuracy), and estimating target type sizes by EM (Saerens et al., 2002), which amplified the classifier's errors.

## S3 Estimating accuracy without labels (Section 4.6)

**Can one tell without labels how good a labeling is?** We fitted simple linear predictors of accuracy from the label-free properties of all 1,200 candidate labelings of the development runs and the sandbox, and tested them on BANC and the hemibrain. The pre-specified predictor (seed agreement and type-size divergence) missed by 12.8 points on average (median 8.4). It was accurate on complete targets with a non-collapsed labeling (FAFB → BANC cell types: 0.5 points). It failed when all seeds made the same error: on the collapsed MCNS → BANC labeling (predicted 44.5 %, true 1.4 %) and on hemibrain super classes with the FlyWire prior (seed agreement 0.94, 34.7 % correct). Seed agreement does rank neurons and types where the transfer works: reviewing the 20 % least-agreed neurons finds 77 % of the errors on MCNS, 55 % on FlyWire and 34 % on BANC (random review: 20 %), and per type it correlates with F1 (Spearman 0.54–0.66). On the hemibrain, where 80 % of the labels are wrong, it is no better than random.
