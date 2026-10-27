# Nerve-cord rare-type test protocol

**Fixed 2026-10-27**, approved by the user together with decisions 1, 2 and 5 (pooled passing block; connections between two passing neurons dropped; self-connections dropped).

To be fixed and committed before any model of the two-stage rare-type pipeline is run on a nerve cord other than MANC. After this commit, only data-loading and out-of-memory fixes may change, and every deviation from this protocol is reported together with the results. Analyses not listed here are labeled exploratory.

## Purpose

The two-stage pipeline types the neurons of the rarest source cell types (2–4 source neurons) in another connectome, without that connectome's labels:
- **stage 1:** a GNN transferred label-free from the source selects the candidates;
- **stage 2:** graph matching against the source's rare-type neurons types them.

**History:**
- The pipeline was developed on FAFB → MCNS (brain) and confirmed in a pre-registered BANC brain test: 22.2 → 41.3 % on the 2–4 stratum ([docs/banc_rare_protocol.md](banc_rare_protocol.md)).
- It was then improved on development data only (FAFB → MCNS and a 25 % thinned MCNS sandbox; [docs/candidate_selection_plan.md](candidate_selection_plan.md), `results/candidate_selection/SUMMARY.md`):
  - **`union` candidates:** the stage-1 type before or after NTAC is a rare type;
  - **two-tier fallback:** neurons the matching does not type take the ≥ 10 model's type.
  - Results: sandbox 47.3 → 60.1 %, full MCNS 74.5 → 81.6 % on the 2–4 stratum; +2.5 to +3.7 points over all labeled neurons from the fallback.

This test applies the improved pipeline to new data: **MANC as the source** and **the nerve cords of BANC and MCNS as targets**. No nerve cord was used in developing either version. GNNs are trained on MANC only, and nothing is trained on a target.

## What was looked at before this protocol

- **Metadata only, no graph or model** (planning, 2026-10-26):
  - column contents of MANC v1.2.1 (harmonized), BANC v888 and MCNS v1.0;
  - MANC type sizes;
  - per MANC size bin: how many target neurons and types carry a MANC type, under each reference, by side and passing class;
  - agreement between BANC's `manc_cell_type` and `cell_type`, and between MCNS's `mancType` and `type`.
- **Side agreement of annotated matches:** each target neuron's `side` against the side of the MANC body it is matched to (BANC `manc_match`, MCNS `mancBodyid`), by super class.
  - It showed that ascending neurons agree at chance level (BANC 43.9 %, MCNS 49.0 %), descending neurons 70–74 % of the time, all others 93–99 %.
  - This label-based count is the reason for the passing-neuron block in the method (decision 1, approved 2026-10-26).
- **Label-free graph statistics** of the three cord graphs (`experiments/b_gnn_celltypes/vnc_data.py`, `results/vnc_rare/d2_graph_stats.json`): local MD5s, neurons, connections and synapses per neuron by threshold, synapses per connection, the share of connections between two passing neurons, and self-connections. They set the variants below.
- **MANC-internal results** (T1–T3, 2026-10-27; source data only; `results/size_threshold_manc/m*/within_*`). Test accuracy of the GNNs on MANC itself, with FAFB's for comparison:

  | Min type size | Split | MANC | FAFB |
  |---|---|---|---|
  | 2 (3,901 types) | random, seeds 0–2 | 71.4, 70.5, 70.8 % (mean 70.9) | mean 82.9 % |
  | 2 | left → right, seed 0 | 63.2 % | 85.1 % |
  | 10 (438 types) | random, seeds 0–2 | 77.3, 78.6, 78.3 % (mean 78.1) | mean 94.2 % |
  | 10 | left → right, seed 0 | 68.3 % | 94.0 % |

  - **The rare types are typed about as well as in FAFB** (2–4-neuron types, min size 2, seed 0): 64.4 % on the random split (FAFB 67.2 %) and 69.6 % left → right (FAFB 76.3 %).
  - **The gap is in the large types:** ≥ 20-neuron types reach 63.7 % on the random split (FAFB 85.1 %) and 54.8 % left → right. In MANC, 4,824 of the neurons in ≥ 20-neuron types are sensory; a likely reason (not tested) is that large sensory types differ mainly in where they project, less in their partners.
  - Passing neurons are typed slightly better than the rest (random 73.3 vs 71.1 %; left → right 68.6 vs 62.2 %).
  - Training took 29 min on the CPU in all (1.37 GB peak). The optional MANC-internal two-stage (T4) was not run.
- **Development checks** on the 25 % MCNS sandbox:
  - **Reproduction check (V0, 2026-10-26):** the changed matching code (`union` candidates, barycenter start, 30 iterations) gave the same final type, the same matched source neuron and the same gain for all 20,123 sandbox neurons as the earlier run (R8), and the same metrics (2–4 accuracy 60.1 %).
  - **Passing block (V1, 2026-10-26):** the same run with the pooled passing block (`--passing-block`). 2–4 accuracy rose from 60.1 to 62.2 % (matched candidates 86.2 → 89.1 %), against the acceptance bar of 59.1 % set before the run. All of the gain was in the passing neurons: on 2–4 candidates they went from 73.4 to 92.3 % correct, while the others stayed at 87.7 %. Brain passing neurons have consistent sides, so the gain is unlikely to come from fixing sides. More likely, the separate block makes the passing neurons' cross-side connections easier to match. The run took 24 min and peaked at 5.4 GB.
- Everything earlier in the project is known: the brain results on FAFB, MCNS, the hemibrain and BANC.

No GNN, no label-free transfer and no matching has been run on the BANC or MCNS nerve cord.

## Data

- **MANC v1.2.1**, harmonized by the BANC project. Source `gs://lee-lab_brain-and-nerve-cord-fly-connectome/compiled_data/manc_121/`, CC BY 4.0. Files: `manc_121_meta.feather` (MD5, base64 `Yx8P7LnG+9+pqBcRJPc+NA==`) and `manc_121_simple_edgelist.feather` (`7ko0TfSZiZ1EGPF2O8OaMA==`).
  - All 23,650 neurons (`flybrain/manc.py`): 19,920 in the cord and 3,730 neck-connective fragments of passing neurons.
  - Connections with ≥ 5 synapses (FAFB's convention); exact duplicate edge rows dropped. 55.1 connections and 969 synapses per neuron, 17.6 synapses per connection (after dropping passing pairs, below). No self-connections.
- **BANC v888 nerve cord:** `banc_888_meta.feather` (`jCuTpgjHFj7J2U1o4HVv9A==`) and `banc_888_edgelist_simple_v2.feather` (`OUQG+Km98JPIlfla/09sSQ==`), as in [docs/banc_protocol.md](banc_protocol.md).
  - Neurons (`flybrain/banc.py` `cord_neurons`): proofread or roughly proofread; not glia, trachea or non-neuronal; region `ventral_nerve_cord`, plus descending and sensory descending neurons.
  - 25,929 neurons: 12,374 left, 12,697 right, 858 without a side; 3,697 passing.
  - Connections between them only. At threshold 5: 16.9 connections and 235 synapses per neuron (31 % and 24 % of MANC's), 13.9 synapses per connection (after dropping self-connections and passing pairs, below).
- **MCNS v1.0 nerve cord:** `body-annotations-male-cns-v1.0-minconf-0.5.feather` (`UKdxh3DFciDxYLpPQxq4ng==`) and `connectome-weights-male-cns-v1.0-minconf-0.5.feather` (`8w6dzKJc/QIb8eez2XVZng==`).
  - Neurons (`flybrain/mcns.py` `cord_neurons`): `Traced`; superclass `vnc_*`, or ascending, descending, sensory ascending/descending, efferent ascending/descending.
  - 24,073 neurons: 11,901 left, 11,832 right, 315 center, 25 without a side; 3,723 passing.
  - Connections between them only. At threshold 5: 43.4 connections and 759 synapses per neuron (79 % and 78 % of MANC's), 17.5 synapses per connection. 7 self-connections (dropped, below).
- **Self-connections (a neuron onto itself) are dropped in all three graphs** (decision 5, 2026-10-27; `loading.drop_self_connections`, in the cache builders). MANC has none. The BANC cord has 16,397 at ≥ 5 synapses (4.8 % of its synapses) and the MCNS cord 7, so the targets follow the source's convention. The BANC brain had 7.6 % and both BANC brain protocols kept them; the cord test differs on purpose, because its source has none.
- **Passing neurons:** ascending, descending, sensory ascending, sensory descending and ascending visceral (and the MCNS equivalents).
  - **Connections between two passing neurons are dropped in all three graphs** (decision 2, approved 2026-10-26). In the targets they can lie in the brain, which MANC lacks.
  - At threshold 5 this drops 5.1 % of MANC's connections (5.2 % of its synapses), all of them real cord connections, and 8.1 % of the BANC cord's connections (8.3 % of its synapses) and 8.5 % of the MCNS cord's (9.8 %). The excess in the targets is consistent with brain connections.
- **Target variants (label-free, the BANC rules applied against MANC at threshold 5; thresholds 1–20 considered):**
  - `raw`: threshold 5, nothing aligned;
  - `density`: the threshold whose connections per neuron are closest to MANC's 55.1: **BANC ≥ 2** (47.8 per neuron), **MCNS ≥ 4** (52.8);
  - `scaled`: threshold 5, counts × the factor that matches MANC's 969 synapses per neuron: **BANC × 4.13**, **MCNS × 1.28**.

  A variant that equals `raw` is left out, as for the hemibrain.
- **MANC label space at min size 2:** 3,901 types, 22,883 neurons.
  - Placeholder names ending in `xx` (6 names, 488 neurons), comma-joined names (10 neurons) and singletons (165) are unlabeled.
  - **2,452 types have 2–4 neurons:** 6,304 neurons, of which 3,058 left, 3,102 right and 144 center. 1,897 of them are passing neurons.
  - **At min size 10:** 438 types, 10,069 neurons.
  - A type's size is its number of MANC neurons (`stratify.source_types("manc")`).

## Labels (used only for evaluation)

Target labels are coded in MANC's label space. Any other name is unlabeled.

**BANC cord:**
- **Reference A (BANC's alignment to MANC):** `manc_cell_type`, the type of the matched MANC body.
  - 22,451 neurons, 3,677 types.
  - **2–4 stratum:** 5,989 neurons of 2,278 types, 30 without a side, 1,840 passing.
- **Reference B (BANC's curated typing):** `cell_type`.
  - 20,511 neurons, 3,281 types.
  - **2–4 stratum:** 5,257 neurons of 1,970 types.
  - BANC names its descending neurons after FAFB types, so B covers few of them (513 of 1,316 descending neurons are in MANC's label space under B, 1,247 under A).
- Where both exist, A and B agree on 99.1 % of the neurons and on 98.4 % of the rare ones. B covers 86 % of A's rare neurons.
- **How reference A was made** (Bates et al., Nature 2026, doi 10.1038/s41586-026-10735-w; preprint doi 10.1101/2025.07.31.667571 v2, Methods "Cell-type matching and annotation"):
  - candidate matches to MANC v1.2.1 by NBLAST, then reviewed by hand;
  - for intrinsic cord neurons, connectivity similarity to MANC neurons was computed as well. The type was assigned when the top NBLAST and connectivity matches agreed; conflicts were resolved by viewing the two neurons together in 3D;
  - estimated error about 7 % (1,000 matched neurons sampled), mostly a similar type of the same hemilineage.
  - Agreement therefore measures agreement with that alignment. For intrinsic neurons it partly rests on connectivity similarity to MANC, the kind of information the pipeline uses, so agreement there may be more favorable than against a reference built from morphology alone. The per-group breakdown (Reported) shows intrinsic neurons separately.

**MCNS cord:**
- **Reference A (MCNS mapping to MANC):** `mancType` where it names a single type.
  - 21,402 neurons, 3,670 types.
  - **2–4 stratum:** 5,945 neurons of 2,261 types, 137 without a left/right side, 1,815 passing.
  - 96.6 % of the rare-type neurons have a body-level match (`mancBodyid`).
  - **How it was made** (Berg et al., Cell 189:5504–5526, 2026, doi 10.1016/j.cell.2026.08.015; preprint doi 10.1101/2025.10.09.680999 v2, Methods "Typing in the central brain and the ventral nerve cord"):
    - types were cross-matched to MANC, at first by morphology (spatial transforms + NBLAST);
    - terminal types were mainly assigned by connectivity co-clustering, with each neuron's synapse counts to already cross-matched types compared by direction, not magnitude; NBLAST was used to cross-check; the pipeline was run iteratively;
    - 93.1 % of cord neurons were matched to MANC; 15 % of efferent and sensory and 29 % of sensory ascending neurons could not be matched confidently;
    - a neuron matching several MANC types lists them comma-separated (unlabeled here).
    - The text does not define the body-level `mancBodyid` field; it is only counted here, not used for scoring.
    - As for BANC, reference A rests mainly on connectivity at the level of terminal types, so agreement with it is not independent of the information the pipeline uses.
  - 778 neurons list several types. They are unlabeled, and scored set-based as exploratory only.
- MCNS `type` equals `mancType` on 99.8 % of the rare-type neurons where both exist, so it is not reported as a second reference.

Both cords are typed about as completely as MANC: on the 2–4 stratum, 1.02 target neurons per type relative to MANC's size, for both targets.

## Method (fixed by the development runs; one change, disclosed)

- **GNNs:** trained on MANC with `configs/cell_type.yaml` unchanged (`size_threshold.py --source manc --stages within`):
  - **min size 2:** random split, seeds 0–2, `results/size_threshold_manc/m2/models` (lean training, as FAFB m2);
  - **min size 10:** seeds 0–2, `results/size_threshold_manc/m10/models`;
  - SHA-256 below.
- **Stage 1, label-free MANC → target labeling** (`size_threshold.py` with `lean_label_free.py`; `label_free.py`'s pipeline), at min size 2 and, for the fallback, at min size 10:
  - candidates: target variant × normalization (target, source) × rho (0, 0.25, 0.5, 0.75, 1);
  - type-size prior: the source prior;
  - selection: **`size_agreement`** (`flybrain/selection.py`), as in the BANC rare-type test and the sandbox;
  - refinement: NTAC on the chosen graph, seeded where all three GNNs agree.
- **Candidates: `union`.** The target neurons whose m2 stage-1 type after NTAC (`pred`), or whose prior-matched type before NTAC (`pred_selected`), is a 2–4-neuron MANC type, labeled or not. It is computed from the stage-1 `predictions.parquet` alone.
- **Stage 2, graph matching** (`graph_matching_mcns.py --source manc --target <cord> --candidates union`):
  - **Matching:** the MANC neurons of 2–4-neuron types are matched one-to-one to the candidates by FAQ / Frank–Wolfe. P is block-diagonal and maximizes the agreement of the log1p(synapse) adjacencies among the selected neurons, summed over all ordered pairs of blocks.
  - **Blocks: left, right, and passing** (decision 1).
    - Passing neurons of every side form one pooled block in MANC and in the target, because their `side` follows different anatomy: neck side in MANC, soma side in BANC and MCNS.
    - The other neurons are blocked by `side`.
    - With two blocks this is the frozen method exactly: the 4 ipsilateral and contralateral terms.
  - **Barycenter start, 30 iterations** (stop earlier if the step falls below 1e-4). The first direction pairs the leading singular vectors by rank.
  - The target graph is the variant the m2 stage 1 chose.
  - Each block is padded to a square. Candidates matched to padding are not assigned. Non-passing neurons without a left/right side are not matched.
- **Two-tier fallback:** every target neuron not assigned a MANC neuron by the matching takes the m10 stage-1 type if its m2 stage-1 type has ≥ 10 MANC neurons, and keeps its m2 stage-1 type otherwise. This covers non-candidates, candidates on padding and neurons without a side. The gate is the coarse model's own minimum type size.
- **Final labeling (primary):** assigned candidates get the type of their MANC neuron; every other neuron gets the two-tier type.
- **Not used** (rejected on development data): probability-mass selection, absorb / reject, many-to-one matching, the soft start.
- **Disclosure:** the passing block is the one change to the frozen pipeline. It was decided after the side-agreement count above, before any model output on a cord. On development data it changed the 2–4 sandbox accuracy from 60.1 to 62.2 % (V1).

## Known violations of the pipeline's assumptions (not corrected)

- **Sex:** BANC is female, MANC male. Male-specific cord types have no counterpart in BANC.
  - BANC's `sexually_dimorphic` flag on its 5,989 reference-A rare-type neurons: 5,094 isomorphic, 734 dimorphic (12.3 %), 160 female-specific (2.7 %), 1 without a flag. The female-specific ones nonetheless carry a MANC type.
  - From the other side, MCNS's `dimorphism` field marks 29 of MANC's 2,452 rare types as male-specific (9) or potentially male-specific (20): 74 MANC neurons, 1.2 % of MANC's 6,304 rare-type neurons. In BANC their MANC neurons can only be matched to neurons of other types.
- **Sparsity:** at MANC's threshold the BANC cord has 31 % of MANC's connections per neuron (16.9 vs 55.1), about the ratio of the BANC brain against FAFB (32 %). The MCNS cord has 79 %.
- **Sides:** apart from passing neurons, `side` is taken to mean the same in all three datasets (93–99 % agreement with the matched MANC body). 30 (BANC) and 137 (MCNS) reference-A rare-type neurons without a left/right side cannot be changed by matching.
- **Passing–passing connections** are dropped everywhere, including the real cord connections between them in MANC (5.2 % of MANC's synapses at threshold 5).
- **Serial homologs:** many MANC types repeat across neuromeres with similar connectivity. The blocks do not separate segments.
- **One-to-one matching:** type sizes in the targets can differ from MANC's.

## Runs

**Per target, BANC cord first, then MCNS cord.** For the MCNS cord, replace `banc_cord` with `mcns_cord`.

1. **Stage 1, m2:** `python experiments/b_gnn_celltypes/size_threshold.py --source manc --min-sizes 2 --stages label_free --target banc_cord --select size_agreement` (to `results/size_threshold_manc/m2/label_free_banc_cord_size_agreement`).
2. **Stage 1, m10:** the same with `--min-sizes 10`.
3. **Primary, improved two-stage:** `python experiments/b_gnn_celltypes/graph_matching_mcns.py --source manc --target banc_cord --stage match --candidates union --passing-block --fallback size_threshold_manc/m10/label_free_banc_cord_size_agreement` (barycenter start, 30 iterations, stage 1's variant; to `results/graph_matching_banc_cord`).
4. **Old pipeline, for comparison:** run 3 with `--candidates gnn` (stage-1 type after NTAC is 2–4).
5. **Diagnostic, oracle candidates:** run 3 with `--candidates oracle`, i.e. the reference-A neurons of 2–4-neuron types. This run uses labels.
6. **Sensitivity to the graph variant:** run 3 with `--variant v`, for each target variant v that stage 1 did not choose.
7. **Sensitivity to the passing block:** run 3 without `--passing-block` (sides as given).

Run 3 is the primary result whatever runs 6 and 7 show.

## Reported, whatever the outcome

- **Stage 1 (m2 and m10)**, from `selected.json`:
  - accuracy / macro-F1 on all reference-A neurons: without prior matching, prior matched, after NTAC, and of the best candidate (post hoc);
  - the chosen candidate, the choice of every rule in `flybrain/selection.py`, and the chosen candidate's gap to the best;
  - for BANC, the same against reference B.
- **For runs 3–7, and for stage 1 (after NTAC and prior matched) on the same subsets:** accuracy and macro-F1 on
  - reference A: the 2–4 stratum, the part of it sent to matching, all labeled neurons, and the bins 5–9, 10–19 and ≥ 20;
  - the reference-A 2–4 stratum by MANC super-class group of the true type (intrinsic, ascending, descending, motor, sensory, other) and by passing / not passing;
  - BANC reference B: the 2–4 stratum, its matched part, and all labeled neurons.
- **Rows of runs 3 and 4:** with and without the two-tier fallback.
- Macro-F1 is the mean, over the stratum's true types, of each type's F1 computed over all neurons labeled in that reference (`stratify.per_type`).
- **Counts:**
  - candidates per block and without a side;
  - candidate recall (share of reference-A rare-type neurons selected) for `union` and `gnn`;
  - the shares of candidates that are reference-A rare-type, other-type and unlabeled;
  - candidates on padding; MANC rows unmatched;
  - runtime and peak memory.
- **The gap to oracle candidates, and the share of it that the pipeline closes:** (final − stage 1) / (oracle − stage 1), on the reference-A 2–4 stratum.
- **The MANC-internal results above**, for scale.
- Unlabeled candidates also receive types. They are counted but not scored.

**Context, not targets** (2–4 stratum, stage 1 → final; oracle):

| Pair | Pipeline | Stage 1 → final | Oracle | Share of gap closed |
|---|---|---|---|---|
| FAFB → MCNS brain | improved (`union`) | 54.2 → 81.6 % | 85.3 % | 0.88 |
| 25 % sandbox | improved (`union`) | 19.2 → 60.1 % | 82.0 % | 0.65 |
| FAFB → BANC brain (test) | old (`gnn`) | 22.2 → 41.3 % | 65.9 % | 0.44 |

The two-tier fallback added 2.5 (MCNS) and 3.7 (sandbox) points over all labeled neurons.

## Expectations stated in advance

Each must hold on both cords against reference A; expectation 2 for BANC also against reference B. "Final" means run 3 with the fallback, "stage 1" the m2 labeling after NTAC.

1. **The improvement replicates.** On the 2–4 stratum, run 3 (`union`) beats run 4 (`gnn` candidates) by at least 3 points, and its macro-F1 is not more than 0.01 lower.
   - Development: +7.1 (full MCNS), +12.8 and +13.7 (two sandbox realizations).
2. **Two stages beat one.** On the 2–4 stratum, the final accuracy exceeds stage 1 by at least 15 points.
   - Development: +27.4 (MCNS) and +40.9 (sandbox); BANC brain, old pipeline: +19.1.
3. **Candidate selection no longer dominates the loss.** The pipeline closes at least half of the gap between stage 1 and oracle candidates on the 2–4 stratum: (final − stage 1) / (oracle − stage 1) ≥ 0.5. If oracle ≤ stage 1, this counts as not met.
   - Development: 0.88 and 0.65; BANC brain, old pipeline: 0.44.
4. **The fallback helps without cost to the rare types.** Over all reference-A neurons, adding two-tier to run 3 raises accuracy by at least 0.5 points, lowers macro-F1 by at most 0.01, and leaves the 2–4 stratum unchanged.
   - Development: +2.5 and +3.7.
   - The threshold is lower than on the brain because only 44 % of labeled MANC neurons are in ≥ 10-neuron types (FAFB: 82 %), so the fallback reaches fewer neurons.

## Saved models (SHA-256, first 16 hex digits)

```
size_threshold_manc/m2/models/cell_type/gnn/s0 a39880cf8a052bd6   s1 b5691f0fbaad359d   s2 bb1658e186eae088
size_threshold_manc/m10/models/cell_type/gnn/s0 9081eb4b40ad5252   s1 5a7e85ceae9e10d0   s2 e713ba1699696ead
```
