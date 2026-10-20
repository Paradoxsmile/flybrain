# BANC rare-type test protocol

Fixed and committed before any model of the two-stage rare-type pipeline was run on BANC. After this commit, only data-loading and out-of-memory fixes may change, and every deviation from this protocol is reported together with the results. Analyses not listed here are labeled exploratory.

## Purpose

The two-stage pipeline types the neurons of the rarest FAFB cell types (2–4 FAFB neurons) in another connectome without its labels. A GNN transferred label-free from FAFB selects the candidates (stage 1). Graph matching against FAFB's rare-type neurons, per hemisphere, types them (stage 2). It was developed on FAFB → MCNS, where it raised the accuracy on these types from 54.2 to 74.5 % (`results/graph_matching_mcns/SUMMARY.md`). MCNS is the development pair, and its mapping to FAFB types is partly connectivity-based. BANC (Bates et al., 2026) was the held-out test of the pipeline for types of at least 20 neurons ([docs/banc_protocol.md](banc_protocol.md)). It has never been used for rare types, for a label space below 20 neurons per type, or for graph matching. This test applies both stages unchanged to BANC. Nothing is trained on BANC.

## What was looked at before this protocol

- The BANC v888 metadata, as for [docs/banc_protocol.md](banc_protocol.md) (MD5s re-checked).
- Label counts only, no model output:
  - how many BANC neurons carry a 2–4-neuron FAFB type under references A and B, by region, side, FAFB super class and FAFB type size;
  - how often the two references agree on these neurons;
  - how often reference A equals `fafb_alignment_cell_type` on these neurons.
- Results of the earlier BANC runs at 20 or more neurons per type are known: the pre-registered test, the sandbox re-scoring of selection rules, labeling effort and accuracy estimation (README).

No GNN at a minimum type size below 20 and no graph matching has been run on BANC.

## Data

- BANC v888, the same files and checksums as in [docs/banc_protocol.md](banc_protocol.md): `banc_888_meta.feather` (MD5, base64 `jCuTpgjHFj7J2U1o4HVv9A==`) and `banc_888_edgelist_simple_v2.feather` (`OUQG+Km98JPIlfla/09sSQ==`).
- Neurons and connections as there (`flybrain/banc.py`): 133,549 brain neurons and the alignment variants `raw` (threshold 5), `density` (threshold 2) and `scaled` (threshold 5, counts × 3.63).
  - Sides: 58,425 left, 62,357 right, 12,767 without a side (mostly optic-lobe and sensory neurons).
- FAFB v783 in the label space of minimum type size 2: 8,658 types, 138,213 neurons.
  - 6,494 of these types have 2–4 neurons: 16,414 neurons, of which 8,183 left, 8,182 right and 49 without a left/right side.
  - A type's size is its number of FAFB neurons (`stratify.fafb_types`).

## Labels (used only for evaluation)

- **Reference A (BANC typing):** `cell_type` where it names one of the 8,658 FAFB types.
  - 88,639 neurons and 7,380 types: 53,652 in the optic lobe, 34,973 in the central brain.
  - **2–4-neuron types:** 14,512 neurons of 5,416 types. By region: 13,980 central brain, 523 optic lobe, 9 neither. By FAFB super class of the type: 12,741 central, 941 descending, 453 visual projection, 179 visual centrifugal, 198 other. 448 have no side. FAFB → MCNS had 13,742 such neurons of 5,370 types.
  - These labels come from BANC's alignment to FAFB, which uses morphology and connectivity: on the rare-type neurons, `cell_type` equals `fafb_alignment_cell_type` for 99.9 % of those in the central brain and 99.8 % of those in the optic lobe.
- **Reference B (morphology-matched central-brain types):** central brain only, `fafb_cell_type` (the type of the FAFB neuron matched by NBLAST, with review) where it names one of the 8,658 types.
  - 35,736 neurons and 6,803 types.
  - **2–4-neuron types:** 13,848 neurons of 5,169 types. 387 have no side.
- For rare types the two references nearly coincide. Where both exist, they agree on 98.7 % of the reference-A rare-type neurons and on 99.5 % of the reference-B ones. Unlike at 20 or more neurons per type, reference B therefore covers almost the whole rare stratum.
- The rare types are typed about as completely in BANC as in FAFB: a mean of 2.68 BANC neurons per type, or 1.06 times its FAFB size. The incomplete typing of BANC concerns mainly the optic lobe, especially its left side. That affects the "all labeled" scores, not the rare stratum.

## Method (fixed by the FAFB → MCNS development)

- **GNNs:** the three FAFB GNNs of `results/size_threshold/m2/models` (min type size 2, random split, seeds 0–2; SHA-256 below). They are the models of the FAFB → MCNS run.
- **Stage 1, label-free FAFB → BANC labeling** (`size_threshold.py` with `lean_label_free.py`; `label_free.py`'s pipeline):
  - candidates: BANC variant (`raw`, `density`, `scaled`) × normalization (target, source) × rho (0, 0.25, 0.5, 0.75, 1);
  - type-size prior: the source prior;
  - selection: **`size_agreement`** (`flybrain/selection.py`), the rule chosen on the development sandbox and used in [docs/hemibrain_protocol.md](hemibrain_protocol.md). It takes the best mean rank of two quantities: how close the predicted type sizes are to the reference, and how well the seeds agree;
  - refinement: NTAC on the chosen BANC graph, seeded where all three GNNs agree.
- **Candidates:** the BANC neurons whose stage-1 type (after NTAC) is a 2–4-neuron FAFB type, labeled or not. Every other BANC neuron keeps its stage-1 type.
- **Stage 2, graph matching** (`graph_matching_mcns.py --target banc`, unchanged from FAFB → MCNS apart from the target):
  - The FAFB neurons of 2–4-neuron types are matched one-to-one to the candidates by FAQ / Frank–Wolfe. The matching P = diag(P_L, P_R) is block-diagonal by hemisphere and maximizes ipsilateral plus contralateral agreement of the log1p(synapse) adjacencies among the selected neurons.
  - **Barycenter start, 30 iterations** (stop earlier if the step falls below 1e-4). The first direction pairs the leading singular vectors by rank.
  - The BANC graph is the variant stage 1 chose.
  - Each block is padded to a square. Candidates matched to padding keep their stage-1 type. Neurons without a left/right side are not matched and keep their stage-1 type.
- **Disclosure: the stage-1 rule differs from the MCNS run.** The FAFB → MCNS stage 1 was selected by type consistency, the rule at the time. `size_agreement` would have chosen rho 1 instead of 0.75 there: 78.4 instead of 80.1 % on all labeled MCNS neurons, before NTAC. So the 74.5 % was obtained with a stage 1 that this protocol does not use. `size_agreement` is fixed here because type consistency failed on BANC ([docs/banc_protocol.md](banc_protocol.md)).
- **Reproduction check on MCNS (before this commit):** the generalized script (`--target`, `--stage1`, `--out`, `--variant`), run on MCNS with the same stage 1, reproduces the FAFB → MCNS result exactly. The final type is the same for all 18,479 candidates, giving 74.5 % / 0.78 on the 2–4 stratum, 89.1 % on the rare-type candidates and 86.3 % / 0.73 over all labeled neurons (`results/graph_matching_mcns_check`, 30 min).

## Known violations of the pipeline's assumptions (not corrected)

- **Sparsity.** At FAFB's threshold, BANC has 8.6 connections per neuron (FAFB: 26.8). Stage 1 suffers: at 20 or more neurons per type, FAFB → BANC typed 44.9 % of the neurons of types outside the optic lobe correctly (reference A), against 74.0 % in FAFB → MCNS. Graph matching suffers too: it relies on the edges among a few thousand neurons per hemisphere.
- **Damaged antennal nerves.** Johnston's organ neurons and nearby central-brain neurons are under-represented. Some FAFB rare neurons then have no counterpart, and the one-to-one matching still assigns them to candidates if the hemisphere has enough of them.
- **Sides.** BANC `side` is taken to mean the same as FAFB `side`. 448 reference-A rare-type neurons (3.1 %) have no side and cannot be changed by matching.
- **One-to-one matching.** Type sizes in BANC can differ from FAFB's.

## Runs

1. **Stage 1:** `python experiments/b_gnn_celltypes/size_threshold.py --min-sizes 2 --stages label_free --target banc --select size_agreement` (to `results/size_threshold/m2/label_free_banc_size_agreement`).
2. **Primary, two-stage:** `python experiments/b_gnn_celltypes/graph_matching_mcns.py --target banc --stage match` (GNN candidates, barycenter start, 30 iterations, stage 1's variant; to `results/graph_matching_banc`).
3. **Diagnostic, oracle candidates:** the same with `--candidates oracle`. The candidates are the reference-A neurons of 2–4-neuron types. This run uses labels, so it is not a label-free result. It separates the loss from candidate selection from the loss from matching.
4. **Sensitivity to the graph variant:** run 2 with `--variant v` for each BANC variant v that stage 1 did not choose. Run 2 stays the primary result whichever variant does best.

The soft start and the GNN probabilities as a linear term are not run: on MCNS they gave the same accuracy or less.

## Reported, whatever the outcome

- **Stage 1** (as in [docs/banc_protocol.md](banc_protocol.md), from `selected.json`):
  - accuracy / macro-F1 on all reference-A neurons without prior matching, after prior matching, after NTAC, and of the best candidate (post hoc);
  - the chosen candidate, the choice of every rule in `flybrain/selection.py`, and the chosen candidate's gap to the best;
  - the breakdown by region and against reference B.
- **For runs 2–4, and for stage 1 on the same subsets:** accuracy and macro-F1 on
  - reference A: the 2–4 stratum (14,512 neurons), the part of it sent to matching, and all labeled neurons (88,639);
  - the reference-A 2–4 stratum by region (central brain, optic lobe) and by FAFB super class of the true type;
  - reference B: the 2–4 stratum (13,848), the part of it sent to matching, and all labeled neurons (35,736).
- Macro-F1 is the mean, over the stratum's true types, of each type's F1 computed over all neurons labeled in that reference (`stratify.per_type`, as on MCNS).
- **Counts:**
  - candidates per hemisphere and without a side;
  - candidate recall (the share of reference-A rare-type neurons selected);
  - the shares of candidates that are reference-A rare-type, other-type and unlabeled;
  - candidates left on padding; FAFB neurons left unmatched;
  - runtime and memory.
- Unlabeled candidates also receive types. These are counted but not scored.

Context, not targets:
- FAFB → MCNS, 2–4 stratum: stage 1 54.2 % / 0.49, two-stage 74.5 % / 0.78, oracle candidates 85.3 %, candidate recall 83.8 %. All labeled MCNS neurons: 84.2 % / 0.51 → 86.3 % / 0.73.
- Within FAFB (left → right): 78.9 → 85.8 %, oracle candidates 92.2 %.

## Expectations stated in advance

1. On the 2–4 stratum, the two-stage pipeline beats stage 1 by at least 10 points, against reference A and against reference B (FAFB → MCNS: +20.3).
2. Over all reference-A neurons, macro-F1 rises by at least 0.05 and accuracy does not fall (FAFB → MCNS: +0.22 and +2.1 points).
3. On the 2–4 stratum (reference A), the two-stage accuracy stays below FAFB → MCNS (74.5 %), because BANC is about three times sparser and stage 1 selects candidates less well.
4. Matching withstands the sparsity better than the GNN does. With oracle candidates, matching keeps a larger share of its FAFB → MCNS accuracy than stage 1 keeps of its own, on the 2–4 stratum (reference A): oracle(BANC) / 85.3 > stage 1(BANC) / 54.2.

## Saved models (SHA-256, first 16 hex digits)

```
size_threshold/m2/models/cell_type/gnn/s0 eeb76038cc7c33bb   s1 4b93a898034a388c   s2 029f1f751cd4bd6f
```
