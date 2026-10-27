# Plan: a pre-registered nerve-cord test of the improved two-stage rare-type pipeline

Status: draft for review. Nothing has been trained or run on any nerve cord. The protocol draft is [docs/vnc_rare_protocol.md](vnc_rare_protocol.md). It is committed only after the steps marked "before the protocol" below, and before any run on a target cord.

## Why the nerve cord

- The improved pipeline (`union` candidates + two-tier ≥ 10 fallback) was fixed on development data only: full FAFB → MCNS brain (2–4 stratum 74.5 → 81.6 %) and the 25 % thinned MCNS sandbox (47.3 → 60.1 %). See [docs/candidate_selection_plan.md](candidate_selection_plan.md) and `results/candidate_selection/SUMMARY.md`.
- BANC's brain is spent for rare types (pre-registered test, 22.2 → 41.3 %, [docs/banc_rare_protocol.md](banc_rare_protocol.md)).
- No nerve cord has been used anywhere in the project. MANC → BANC cord and MANC → MCNS cord are a new source, two new targets, a new label space and new tissue. Nothing about them informed the pipeline.
- The two targets differ in useful ways:
  - **MCNS cord:** male like MANC, same lab and pipeline family (Janelia), labels from a body-by-body mapping to MANC;
  - **BANC cord:** female, an independent reconstruction (sparse in the brain, likely sparse in the cord too), with its own MANC alignment.

## What was looked at for this plan (read-only, label counts only)

Everything here comes from metadata. No graph, model or matching was built or run on any cord.

- Column contents of the MANC v1.2.1 harmonized metadata (downloaded with the user's OK, MD5 checked), the BANC v888 metadata and the MCNS v1.0 annotations.
- The counts below: MANC type sizes; shared types and neurons per MANC size bin, side and passing class in both targets; agreement between each target's two type columns.
- **Side agreement of annotated matches:** a target neuron's `side` against the side of the MANC body it is matched to (BANC `manc_match`, MCNS `mancBodyid`), by super class. This used match annotations, which are label information. It led to decision 1 below and is disclosed in the protocol.
- Listings (names, sizes, MD5) of the public buckets: BANC `compiled_data/`, Janelia `flyem-manc-exports/v1.0/`, MCNS `flat-connectome/`.

## 1. Data

### MANC (source)

- **Version:** MANC v1.2.1, in the harmonized form of the BANC project, from the bucket the project already uses for the hemibrain: `gs://lee-lab_brain-and-nerve-cord-fly-connectome/compiled_data/manc_121/`.

  | File | Size | MD5 (base64) | Status |
  |---|---|---|---|
  | `manc_121_meta.feather` | 1.4 MB | `Yx8P7LnG+9+pqBcRJPc+NA==` | downloaded to `data/raw/manc/`, MD5 verified |
  | `manc_121_simple_edgelist.feather` | 87 MB | `7ko0TfSZiZ1EGPF2O8OaMA==` | not downloaded (step D0) |

  The bucket also holds `manc_121_split_edgelist.feather` (336 MB, axon/dendrite compartments) and synapse tables (2.5–3.9 GB). Neither is needed.
- **Licence and citation:** CC BY 4.0, like the other harmonized files. Cite Takemura et al. (2024), Marin et al. (2024) and Cheong et al. (2024) for MANC, and Bates et al. (2026) for the harmonization. The MANC references still have to be added to `docs/literature.md` and checked.
- **Official alternative:** Janelia's `gs://flyem-manc-exports/v1.0/` (MANC v1.0) has `manc-v1.0-neuron-properties.feather` (17 MB, with `somaSide` and `rootSide`) and neuPrint dumps. It is a different version, so it is not used for labels. Optional use: checking the side convention (decision 1).
- **Neurons:** 23,650 in all, 19,920 with region `ventral_nerve_cord` and 3,730 with region `neck_connective` (the passing neurons). By super class: 13,162 intrinsic, 5,925 sensory, 1,862 ascending, 1,322 descending, 721 motor, 535 sensory ascending, 111 visceral/circulatory, 6 ascending visceral and 6 sensory descending.
- **Neuron table (`cell_type`):** 4,075 names on 23,546 neurons.
  - **Unlabeled by rule:** 6 placeholder names ending in `xx` (`SNxxxx`, `SNppxx`, `SNtaxx`, `SAchxx`, `SAppxx`, `SAxxxx`; 488 neurons), 10 comma-joined names, and 165 singleton types.
  - **Label space at min size 2:** 3,901 types, 22,883 neurons. **At min size 10:** 438 types, 10,069 neurons.
- **Side:** `side`, with 11,656 left, 11,430 right, 317 center and 247 missing.

**MANC types per size bin (min size 2, placeholders out):**

| MANC type size | Types | Neurons | Of these left / right / center |
|---|---|---|---|
| **2–4** | **2,452** | **6,304** | 3,058 / 3,102 / 144 |
| 5–9 | 1,011 | 6,510 | |
| 10–19 | 322 | 4,086 | |
| ≥ 20 | 116 | 5,983 | |

- **The rare stratum is relatively much larger than in FAFB:** 27.5 % of labeled MANC neurons, against 11.9 % in FAFB. Of the 2,452 rare types, 1,686 have exactly 2 neurons, mostly a left/right pair.
- **1,897 of the 6,304 rare neurons (30 %) pass through the neck:** 960 ascending, 916 descending, plus a few sensory ascending/descending. This matters for sides (decision 1).

### BANC cord (target 1)

- **Files:** the same v888 files as in [docs/banc_protocol.md](banc_protocol.md): meta `jCuTpgjHFj7J2U1o4HVv9A==` (local file re-checked today, unchanged in the bucket) and simple edge list v2 `OUQG+Km98JPIlfla/09sSQ==`.
- **Cord neurons:**
  - proofread or roughly proofread; not glia, trachea or non-neuronal;
  - `region == "ventral_nerve_cord"`, plus descending neurons (`super_class` descending or sensory_descending), whose somata BANC places in the brain.

  That gives 25,929 neurons: 12,374 left, 12,697 right and 858 without a side. 3,697 are passing neurons.
- **Type columns:**
  - `manc_cell_type`: the type of the matched MANC body. 90.6 % of these neurons have a body-level `manc_match`; status flags such as `MANC_MATCH_MANUALLY_CHECKED` mark reviewed ones.
  - `cell_type`: BANC's curated type. It uses MANC names for cord-intrinsic neurons, but FAFB names for descending neurons (e.g. `DNge149`, MANC `DNxl126`). Only 513 of 1,316 BANC descending neurons have a `cell_type` in MANC's label space, against 1,247 with `manc_cell_type`.
- So, unlike the brain test, **reference A is `manc_cell_type`** (the alignment to the source) and **reference B is `cell_type`**.

### MCNS cord (target 2)

- **Files:** the same v1.0 files as for the brain: annotations `UKdxh3DFciDxYLpPQxq4ng==` and weights `8w6dzKJc/QIb8eez2XVZng==` (bucket MD5s; the local files are verified in D2).
- **Cord neurons:**
  - `status == "Traced"`;
  - `superclass` starting with `vnc_`, or ascending, descending, sensory ascending/descending or efferent ascending/descending.

  That gives 24,073 neurons: 11,901 left, 11,832 right, 315 center and 25 without a side. The brain cache excludes `vnc_*`, so this is a new cache.
- **Type column:** `mancType`, the type of the MANC body matched to this neuron.
  - 96.6 % of the rare-type neurons have a `mancBodyid`.
  - 778 entries list several types and are unlabeled (kept for set-based scoring).
  - MCNS `type` equals `mancType` on 99.8 % of rare-type neurons where both exist, so it is no second reference. MCNS gets reference A only.
  - The matching method (morphology, connectivity or both; Berg et al., 2026) has to be checked and stated in the protocol, as "partly connectivity-based" was for the brain.
- **Side:** `somaSide`, else `rootSide`, as in the brain loader.

### Shared types and neurons per MANC size bin

The neuron counts are target neurons whose reference type is in MANC's min-size-2 label space. "Per type / MANC" is the mean number of target neurons per type, divided by the mean MANC size of the bin's types.

| MANC size | MANC types / neurons | BANC ref A (`manc_cell_type`): types / neurons (no side, passing) | BANC ref B (`cell_type`) | MCNS (`mancType`): types / neurons (no side, passing) |
|---|---|---|---|---|
| **2–4** | 2,452 / 6,304 | **2,278 / 5,989** (30, 1,840) | 1,970 / 5,257 | **2,261 / 5,945** (137, 1,815) |
| 5–9 | 1,011 / 6,510 | 978 / 6,363 | 937 / 6,136 | 989 / 6,340 |
| 10–19 | 322 / 4,086 | 314 / 3,978 | 290 / 3,661 | 311 / 3,904 |
| ≥ 20 | 116 / 5,983 | 107 / 6,121 | 84 / 5,457 | 109 / 5,213 |
| **All** | 3,901 / 22,883 | 3,677 / 22,451 | 3,281 / 20,511 | 3,670 / 21,402 |
| per type / MANC, 2–4 | | 1.02 | 1.04 | 1.02 |

- **Both cords are typed about as completely as MANC itself.** 93 % (BANC) and 92 % (MCNS) of MANC's rare types occur, at MANC's size.
- **BANC references A and B agree** on 99.1 % of the neurons labeled in both, and on 98.4 % of the rare ones. B covers 86 % of A's rare neurons; the gap is mostly the descending neurons.
- **The rare strata are about as large as in the brain tests, with far fewer neurons:** about 6,000 rare-type neurons per target, against 14,512 (BANC brain) and 13,742 (MCNS brain). Matching blocks are therefore small (≈ 3,100 MANC rows per side).

### Left/right blocks

- **Per hemisphere, by `side`, as frozen.** Hemisegment blocks (side × neuromere) would be finer, but MANC's harmonized table has no neuromere, BANC has `neuromere` for half its cord neurons and MCNS has `somaNeuromere` for 69 %. They would also be a new method, so they are not used.
- **Side agreement with the matched MANC body**, as a share of matched neurons (rare-type neurons in brackets):

  | Super class | BANC | MCNS |
  |---|---|---|
  | intrinsic | 95.7 % (94.8) | 97.9 % (96.8) |
  | motor | 97.4 % | 95.1 % |
  | sensory | 99.2 % | 92.6 % |
  | sensory ascending | 94.7 % | 97.9 % |
  | **descending** | **74.4 % (73.2)** | **70.4 % (69.1)** |
  | **ascending** | **43.9 % (46.3)** | **49.0 % (51.9)** |

- **For ascending neurons, side carries no information across datasets, and for descending neurons it is wrong about 30 % of the time.**
  - The likely cause is a convention difference. MANC sides passing neurons by their neck-connective (axon) side. BANC and MCNS side them by the soma: in the cord for ascending neurons, in the brain for descending ones.
  - The frozen method would put about half of the rare ascending neurons into the wrong block. They cannot then be typed correctly by matching. → **Decision 1.**

## Decisions for the user (before the protocol)

1. **Sides of passing neurons.** Options:
   - **B (recommended): a third, pooled block.** All passing neurons, i.e. ascending, descending, sensory ascending/descending and ascending visceral, go into a third block in every graph, whatever their side. P = diag(P_L, P_R, P_pass), and the objective sums all 9 ordered block pairs (with two blocks, these are the current 4 terms).
     - It is a label-free rule stated for any source/target pair: neurons whose `side` is defined by different anatomy in the two datasets are matched without the side constraint.
     - The pooled block has ≈ 1,900 MANC rows.
   - **A: sides as they are**, the frozen method literally. Expected to lose most rare ascending neurons and part of the descending ones. Recommended as a pre-registered sensitivity run either way (one matching per target, ~10 min).
   - **C: harmonize the sides.** For ascending neurons, use MANC's soma side from Janelia's v1.0 table (another 17 MB download, and v1.0 vs v1.2.1 body ids must be checked); descending neurons pooled as in B. More data and more assumptions for the same goal.

   B changes the matching for 30 % of the rare neurons, so it should be checked on development data first (V1 below). The decision was triggered by a label-based count, which the protocol discloses.
2. **Edges between two passing neurons.** In BANC and MCNS, an ascending → descending connection can lie in the brain (ascending axons and descending dendrites meet there). MANC has no brain.
   - **Recommended:** drop every connection whose two partners are both passing neurons, in all three graphs, MANC included, so the rule is symmetric.
   - D2 measures, label-free, the share of synapses this drops in MANC (real cord connections lost) and in each target (brain contamination removed). That is reported.
   - "Keep" can be an exploratory sensitivity run if the user wants it (stage 1 + matching, ~45 min per target).
3. **Where to train:** locally on the CPU (recommended; ~1.5–3 h in all, no upload), or on a Kaggle T4 (~15 min, but the MANC cache has to be uploaded and the models downloaded).
4. **Optional checks before the protocol:** V0 (a reproduction check after the code changes), V1 (the pooled block on the development sandbox) and T4 (MANC-internal two-stage). Each is listed in the run table. V0 is recommended because the matching core is touched.
5. **Self-connections** (raised by D2). The BANC cord has 16,397 at ≥ 5 synapses (4.8 % of its synapses), MCNS 7, MANC none. **Decided 2026-10-27: drop them in all three cord graphs**, so the targets follow the source's convention (the BANC brain protocols kept them; their source FAFB has them too). D2 was rerun: only BANC's scale changes, × 3.99 → × 4.13.

## 2. Code changes

Every change is additive: FAFB-source defaults stay, and all earlier commands reproduce their runs.

| File | Functions | Change |
|---|---|---|
| `flybrain/manc.py` (new) | `cord_neurons`, `build_manc_cache`, `load_manc_tables` | Load the harmonized MANC files in the FAFB table schema (`root_id`, `super_class`, `cell_type`, `side`, `passing`). Placeholder (`…xx`) and comma-joined names become unlabeled. Exact duplicate edge rows are dropped (as in `hemibrain.py`). Threshold ≥ 5 (FAFB's convention). One variant, `raw`. Cache `data/processed/manc_{neurons,connections}.parquet`. |
| `flybrain/loading.py` | `load_graph`, `build_graph`, new `drop_passing_pairs` | `load_graph(source="fafb" \| "manc")`: the label space comes from the source table, not always from FAFB. New datasets `manc`, `banc_cord`, `mcns_cord` (synthetic offsets 4000–6000). `build_graph` passes `alt_type`, `manc_types` and `passing` as attributes. `drop_passing_pairs` implements decision 2, applied in the three cord loaders. |
| `flybrain/banc.py` | new `cord_neurons`, `build_banc_cord_cache`, `load_banc_cord_tables`, `CORD_VARIANTS` | The cord selection above. `cell_type` = `manc_cell_type` (ref A) and `alt_type` = `cell_type` (ref B), both coded with MANC's rules. Cache `banc_cord_*.parquet`. Variants `raw` / `density` / `scaled`, set relative to MANC's statistics by the BANC rules (numbers from D2). The brain functions are unchanged. |
| `flybrain/mcns.py` | new `cord_neurons`, `build_mcns_cord_cache`, `load_mcns_cord_tables`, `CORD_VARIANTS` | The cord selection above. `cell_type` = single `mancType`, `manc_types` = raw `mancType`. Side from `somaSide`, else `rootSide`. Cache `mcns_cord_*.parquet`. Variants by the same MANC-relative rules, replacing the FlyWire factor 0.581, which is specific to FAFB. |
| `experiments/b_gnn_celltypes/stratify.py` | new `source_types(source)`; `fafb_types` becomes `source_types("fafb")` | Size and region of each source type. For MANC, the region is the super-class group: intrinsic, ascending, descending, motor, sensory, other. |
| `experiments/b_gnn_celltypes/label_free.py` | `variants`, `load`, `setups`, `breakdown`, `main` | `--source manc`, with `{fafb, manc}` as single-graph sources. Targets are loaded with `source=args.source`. `DATASET_VARIANTS` / `SCORING` get `banc_cord` / `mcns_cord` (scoring graph `raw`). `breakdown` scores ref B from `alt_type` (as `morph_*` in `selected.json`). `selected.json` records the source. |
| `experiments/b_gnn_celltypes/size_threshold.py` | `run_within`, `run_label_free`, `label_free_dir`, `main` | `--source manc` writes to `results/size_threshold_manc/m<size>/…`. The `within` stage trains on MANC. `--target banc_cord \| mcns_cord`, so a run is e.g. `m2/label_free_banc_cord_size_agreement`. Size bins and type tables come from `stratify.source_types`. |
| `experiments/b_gnn_celltypes/lean_label_free.py` | – | No FAFB assumption found (`seed_outputs` and `LazyProbs` are generic). Only the tests are extended. |
| `experiments/b_gnn_celltypes/graph_matching_mcns.py` | `graphs`, `setup`, `candidates`, `score_rows`, `match`, `faq_blocks` / `TERMS`, `soft_probabilities`, `main` | **Source:** `--source manc`; `graphs()` loads the source by name. `STAGE1` / `STAGE1_ROOT` / `MODELS` are keyed by source; `sizes` come from `source_types`. **Candidates:** a built-in `union` (`rare(pred) \| rare(pred_selected)`, computed from `predictions.parquet`, label-free), so the test does not need `candidate_selection.py`, which keeps its development-only guard. **Blocks:** `--passing-block` (decision 1) puts passing neurons into a third block, and `TERMS` becomes all ordered block pairs. **Scoring:** reference names for the cord targets; ref B from `alt_type`; strata by MANC super-class group; size bins 5–9 and 10–19 reported separately (the 5–19 row stays). **Naming:** `info_*.json` uses `source_rare` (alias of `fafb_rare`). |
| `experiments/b_gnn_celltypes/candidate_selection.py` | – | Not used for the test, and unchanged. |
| `data/README.md`, `README.md` | – | A MANC section (files, MD5, licence) and the two cord caches. |

**Unit tests (synthetic, seconds each):**
- `tests/test_manc.py` (new):
  - `cord_neurons` coding: placeholder and comma names unlabeled; side and `passing`;
  - the cache from tiny feather files with duplicated edge rows, which are dropped;
  - threshold 5;
  - `drop_passing_pairs` removes exactly the passing–passing edges;
  - `load_graph(source="manc", synthetic=True)` codes the target in MANC's label space.
- `tests/test_banc.py`: cord selection (cord neurons and descending neurons kept; glia and brain intrinsic out); ref A = `manc_cell_type`, ref B = `cell_type`; cord variants.
- `tests/test_mcns.py`: cord selection; comma `mancType` unlabeled; side from `somaSide`, else `rootSide`.
- `tests/test_graph_matching.py`:
  - `union` equals `candidate_selection`'s union on a synthetic stage-1 table;
  - FAQ with three blocks recovers a planted matching;
  - the generalized `TERMS` for two blocks equal the old four (regression);
  - end-to-end `match()` with `source="manc"` and `--passing-block`.
- `tests/test_lean.py` / `test_selection.py`: a synthetic `label_free` run with `--source manc --target banc_cord`.
- All existing test files pass unchanged.

## 3. Training (MANC only)

- **Graph:** MANC, 23,650 neurons, connections ≥ 5 synapses, with passing–passing connections dropped (decision 2). Degree features as for FAFB.
- **Labels:** min size 2 (3,901 types, 22,883 neurons) and min size 10 (438 types, 10,069 neurons).
- **Config:** `configs/cell_type.yaml` unchanged: 3 bidirectional layers, 64 hidden, count-weighted, balanced class weights, 300 epochs, model chosen by validation macro-F1. Below min size 5 the code uses `lean_training.py`, as for FAFB m2.
- **Runs:**
  - random split, seeds 0–2, at min size 2 and 10: these 6 models form the stage-1 ensembles;
  - left → right, seed 0, at both sizes: a sanity check.

  `size_threshold.py --source manc --min-sizes 2 10 --stages within --seeds 0 1 2 --lr-seeds 0`.
- **Time:**
  - **FAFB references:** m2 random split 225 s per seed on the GPU (139k neurons, 8,658 types); FAFB m20 left → right 908 s on this laptop's CPU against about 112 s on the GPU, so the CPU is about 8× slower.
  - **MANC:** 1/6 of the neurons and 45 % of the types. Its number of connections is not yet known (D2), so the estimate is a range:
    - m2: about 6–18 min per model on the CPU;
    - m10: about 3–8 min.
  - **Total, 8 models:** about 1–2.5 h on the CPU, or under 15 min on a T4.
  - **Memory:** under 3 GB (FAFB-size runs fit in 15 GB).
- **Sanity checks are allowed before the protocol.** The random-split test accuracy, the left → right accuracy and (optionally) a MANC-internal two-stage left → right use only source data. They cannot leak target labels.
  - They are listed in the protocol under "what was looked at", with their numbers.
  - They may reveal bugs, and a bug may be fixed before the protocol commit, with a note.
  - They do not change the config, the thresholds or any rule.
  - FAFB values for scale: test accuracy m2 82–84 %, m10 94 %; left → right m2 83–85 %, m10 92–94 %.
- **Stop rule (pre-stated):** if a MANC m2 model reaches under 30 % test accuracy, or under half of its own m10 accuracy, training is treated as broken. It is investigated as a bug before anything else.

## 4. Protocol

Drafted in [docs/vnc_rare_protocol.md](vnc_rare_protocol.md). It has placeholders, marked **[D2]** / **[T]** / **[V]**, for the numbers that come from the steps before the protocol: variant thresholds and scales, edge counts and the passing-pair shares, model SHA-256s, the MANC-internal results and the V-checks. They are filled in, and the draft reviewed, before the commit.

## 5. Runs, in order (each needs an explicit OK)

The time and memory estimates scale from logged runs:
- **Stage 1:** FAFB → MCNS m2 took 161 min at ~7 GB, about 25 min per (variant, norm) for the prior matching, which scales with N_target × K. BANC cord gives 26k × 3.9k, 1/12 of MCNS's 144k × 8.7k.
- **Matching:** 27 min / 4.5 GB at n = 9.3k, about n^2.25 in time and n² in memory. With `union`, n ≈ 4–6k per block here.

Peak memory is sampled for every run, as in the candidate-selection runs.

| # | What | Data touched | Est. time | Est. peak RAM | Before / after protocol |
|---|---|---|---|---|---|
| D0 | Download `manc_121_simple_edgelist.feather` (87 MB, MD5 above) | – | 1 min | < 0.5 GB | before |
| C | Code changes + unit tests (`pytest tests/`) | synthetic | < 2 min compute | < 1 GB | before |
| D2 | Build the three cord caches; label-free graph statistics: neurons, connections and synapses per neuron, synapses per connection, passing-pair shares; variant thresholds and scales by the BANC rules relative to MANC; local MD5s | MANC, BANC and MCNS edges, no labels | **done:** 40 s, 4.72 GB peak; rerun 2026-10-27 without self-connections, 26 s, 4.81 GB (`results/vnc_rare/d2_graph_stats.json`; the first run's output is in `d2.log`) | – | before |
| V0 | (recommended) Reproduction check: sandbox 25 % `union` matching with the changed code must give the same final types as R8. **Done 2026-10-26: identical** (types, matched source neurons, gains and metrics; 21 min, 5.23 GB) | dev sandbox | 23 min | 4.9 GB | before |
| V1 | (recommended with decision 1 = B) Sandbox 25 % `union` with `--passing-block`. Accepted if the 2–4 accuracy is not more than 1 point below 60.1 %. In the brain, passing neurons' sides agree, so B should cost little there. **Done 2026-10-26: accepted, 62.2 %** (passing 2–4 candidates 73.4 → 92.3 %, others unchanged; 24 min, 5.41 GB) | dev sandbox | 30–40 min (a third n × n block) | ~7 GB | before |
| T1 | MANC GNNs, min size 2, random split, seeds 0–2. **Done 2026-10-27:** 70.9 % mean (FAFB 82.9 %) | MANC | 15–35 min CPU (MANC: 1.30 M connections at ≥ 5, 35 % of FAFB's) | < 3 GB | before |
| T2 | MANC GNNs, min size 10, seeds 0–2. **Done 2026-10-27:** 78.1 % mean (FAFB 94.2 %) | MANC | 10–25 min CPU | < 2 GB | before |
| T3 | MANC left → right, seed 0, both sizes (sanity). **Done 2026-10-27:** 63.2 % (m2), 68.3 % (m10); FAFB 85.1 / 94.0 %. T1–T3 together 29 min, 1.37 GB | MANC | 10–25 min CPU | < 3 GB | before |
| T4 | (optional) MANC-internal two-stage, left → right: GNN candidates on the right, matched against the left (`graph_matching.py` with `--source manc`) | MANC | 5–10 min + code | 1–2 GB | before |
| P | Fill in the protocol, user review, **commit the protocol** | – | – | – | **gate** |
| S1b | Stage 1, MANC → BANC cord, m2 (`size_threshold.py --source manc --min-sizes 2 --stages label_free --target banc_cord --select size_agreement`) | BANC cord | 15–30 min | 2–4 GB | after |
| S2b | Stage 1, m10, same target | BANC cord | 5–12 min | 1.5–3 GB | after |
| M1b | **Primary:** `union` + two-tier m10, barycenter, 30 iterations, chosen variant (`graph_matching_mcns.py --source manc --target banc_cord --stage match --candidates union --fallback …m10… --passing-block`) | BANC cord | 5–12 min | 1.5–3 GB | after |
| M2b | Old pipeline: `--candidates gnn` (also reported + two-tier) | BANC cord | 4–10 min | 1.5–2.5 GB | after |
| M3b | Oracle candidates (diagnostic, uses labels) | BANC cord | 4–10 min | 1.5–2.5 GB | after |
| M4b | Sensitivity: M1 on the two variants stage 1 did not choose | BANC cord | 2 × 5–12 min | 1.5–3 GB | after |
| M5b | Sensitivity: M1 without `--passing-block` (sides as given; only if decision 1 = B) | BANC cord | 5–12 min | 1.5–3 GB | after |
| S1m–M5m | The same seven runs on the MCNS cord | MCNS cord | ~1–1.5 h together | ≤ 4 GB | after |
| R | Report: `results/vnc_rare/SUMMARY.md`, README section, expectations met or not | saved outputs | < 1 min | < 1 GB | after |

**Totals:**
- **Before the protocol:** about 1.5–3 h of compute (training dominates; V0/V1 add 45–55 min).
- **After the protocol:** about 1–1.5 h per target, 2–3 h for both.
- Nothing is run concurrently, so the peak is about 8 GB (D2) and otherwise ≤ 5 GB.
- **Order inside each target:** S1 → S2 → M1 → M2 → M3 → M4 → M5. The BANC cord goes first; its result cannot change anything for MCNS, because the protocol is already fixed.

## Risks and open points

- **Sparsity of the BANC cord** is unknown until D2. Its brain had 32 % of FAFB's connections per neuron. The density variant handles this as in the brain test.
- **Sex:** BANC is female, MANC male. Male-specific cord types have no counterpart. BANC's `sexually_dimorphic` column lets the protocol report how many rare-type neurons that concerns (a label count, done before the commit).
- **Serial homologs:** many MANC types repeat across the T1–T3 neuromeres with similar connectivity. Stage 1 may confuse them; matching can separate them by their segment-local partners. The per-hemisphere blocks are kept, as frozen.
- **Reference provenance:** the MCNS `mancType` and BANC `manc_cell_type` matching methods have to be described from Berg et al. (2026) and Bates et al. (2026) before the commit. They are what agreement measures.
- **MANC's own types** were defined from morphology and connectivity (Marin et al., 2024; to be checked). If connectivity weighed heavily, the cord types are easier for connectivity-only methods than the brain's partly morphological ones. That affects the absolute levels, not the comparisons the expectations make.
