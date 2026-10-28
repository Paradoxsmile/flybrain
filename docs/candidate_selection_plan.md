# Plan: better stage-1 candidate selection for the two-stage rare-type pipeline

Status: R0–R11 and R14 done (results and conclusions: `results/candidate_selection/SUMMARY.md`). The 0.35 sandbox missed the recall criterion, so the 0.25 sandbox (recall 54.4 %, BANC 54.3 %) was used. R12 was skipped after R11 failed, R13 was not needed, and R14 (seed 3) replicates the findings. The user approved running the plan in the order below. The findings in "What the existing outputs already show" came from read-only checks on saved outputs (pandas, seconds each).

## Why candidate selection

| | Candidate recall | Stage 1 (2–4) | Two-stage (2–4) | Oracle candidates | Gap to oracle |
|---|---|---|---|---|---|
| FAFB → MCNS (development) | 83.8 % | 54.2 | 74.5 | 85.3 | 10.8 |
| FAFB → BANC (pre-registered test) | 54.3 % | 22.2 | 41.3 | 65.9 | 24.6 |

On BANC, matching types 77.1 % of the rare-type candidates it gets. The bigger loss is the 46 % of rare-type neurons that never become candidates. BANC also had fewer candidates than FAFB has rare neurons per hemisphere (5,932 / 6,049 vs 8,183 / 8,182). So 4,384 FAFB rows stayed unmatched, and adding candidates costs no matching capacity.

## Ground rules

- **Development data only:** FAFB → MCNS, plus degraded MCNS copies (`load_graph` retention / exclude_types / crop_fraction) with known labels. BANC is spent for rare types. Any BANC run would need a new protocol, and none is planned here.
- **Fresh test later:** the improved pipeline will be tested in a new pre-registered test, probably on the nerve cord (MANC → BANC / MCNS VNC). Every new hyperparameter therefore gets a label-free rule that carries over to another source/target pair, not a value tuned on MCNS.
- **One run per setting**, as before. The barycenter start is deterministic. The thinning realization is not, so an idea is adopted only if it helps on full MCNS *and* on the sandbox (same sign on both).
- **Machine:** Ryzen 9 5900HX (16 threads), 15.4 GB RAM, CPU only. Stage 1 (~7–8 GB) and matching (~4.5 GB at n = 9.3k) run one at a time. Peak memory is recorded for every run this time, by a background sampler of the process working set (the BANC test missed this).

## What the existing outputs already show (read-only, FAFB → MCNS)

From `results/size_threshold/m2/label_free/cell_type/predictions.parquet`, `results/size_threshold/m10/...`, `results/label_free/...` and `results/graph_matching_mcns/matches_gnn_gm_bary.parquet`:

1. **Most missed rare neurons have a near-rare stage-1 type.** Of the 2,221 labeled rare-type MCNS neurons that are not candidates, 1,497 (67 %) are predicted as a 5–9-neuron type, 380 as a 10–19-neuron type and 344 as a type of ≥ 20.
2. **Candidate sets that are already saved** (recall over the 13,742 labeled rare-type neurons; precision = share of candidates that are labeled rare-type):

   | Candidates | Count | Recall | Precision |
   |---|---|---|---|
   | stage-1 type after NTAC is 2–4 (current) | 18,479 | 83.8 % | 62.3 % |
   | prior-matched type before NTAC (`pred_selected`) is 2–4 | 19,029 | 85.9 % | 62.0 % |
   | union of the two | 22,754 | 91.8 % | 55.5 % |
   | stage-1 type after NTAC is 2–9 | ~28,000 | 94.7 % | – |

3. **Expected number of rare-type neurons:** FAFB has 8,183 / 69,395 (left) and 8,182 / 68,659 (right) rare-type neurons. Scaled to MCNS's 71,325 left and 72,741 right neurons, that gives about 8,410 and 8,670 per side. The current candidates are 8,872 left and 9,342 right.
4. **The size mismatch caps one-to-one matching.** Count per (type, side) how many labeled MCNS rare neurons exceed the number of FAFB neurons. 979 of 13,688 (7.2 %) have no FAFB partner of their own type, so even oracle candidates must be mistyped there.
5. **Two-tier fallback (idea 2), previewed on the saved outputs:**

   | Final labeling (candidates: two-stage, barycenter) | All labeled | 2–4 | 5–19 | ≥ 20 |
   |---|---|---|---|---|
   | non-candidates keep the m2 type (current) | 86.3 / 0.735 | 74.5 / 0.78 | 59.8 | 90.2 |
   | non-candidates: m10 type if the m2 type has ≥ 10 neurons, else m2 | **88.8 / 0.738** | 74.5 / 0.78 | 60.1 | 93.2 |
   | same with m20 (m2 type ≥ 20) | 88.0 / 0.735 | 74.5 / 0.78 | 59.8 | 92.1 |
   | m10 for all non-candidates | 86.2 / 0.657 | 74.5 / 0.78 | 21.7 | 93.3 |

   These numbers are post hoc on the development pair. They have to be repeated on the sandbox before anything is adopted.

6. **Reusable caches:**
   - `results/graph_matching_mcns/soft_gnn.npz` holds the three seeds' prior-matching weights `weights` (3 × 8,658) for the chosen variant and normalization (`raw`, source). With those and the saved `model.pt` files, the prior-matched probabilities of *every* MCNS neuron can be recomputed with a forward pass, without the 25-min Sinkhorn.
   - Its `probs` only cover the 18,479 current candidates, so it does not by itself give the rare-type mass of all neurons.

## The sandbox (BANC proxy)

- **Setup:** MCNS thinned to retention 0.35 (`retention_seed` 2, label_free's default), stage 1 by the protocol's rule `size_agreement`. Then stage 2 on the same thinned copy, with GNN and oracle candidates.
- **Why 0.35:** at ≥ 20 neurons per type, the best label-free candidate reaches 79.8 % at retention 0.5 and 27.5 % at 0.2 (`results/label_free_fafb_mcns_r0.5`, `_r0.2`), against 55.7 % on BANC. Linear interpolation puts BANC's level at a retention of about 0.36.
- **Reproduction criterion, checked before any fix:** the sandbox is accepted as a BANC proxy if
  - candidate recall ≤ 65 % (BANC 54.3 %), and
  - the gap between oracle and two-stage on the 2–4 stratum is ≥ 15 points (BANC 24.6).

  If recall stays above 70 %, retention 0.25 is run instead (S1b). If the copy collapses (stage 1 below 10 % on all labeled), that is reported, and 0.45 is tried.
- **Optional second axis (only if thinning alone does not reproduce BANC):** remove the antennal Johnston's-organ types (`--target-exclude`), as a stand-in for BANC's damaged antennal nerves.

## Idea 1: probability-mass selection

**Score.** For neuron i: m_i = Σ_{t: 2–4 FAFB neurons} q̄_i(t). Here q̄ is the stage-1 ensemble: the mean over seeds of the prior-matched probabilities at the chosen variant, normalization and ρ, before NTAC. It is the same ensemble whose argmax is `pred_selected`.

**Label-free count.** Per hemisphere, K_side = N_target,side × (FAFB rare-type neurons on that side / FAFB neurons on that side): the number of rare-type neurons expected if the target has FAFB's composition. The threshold τ_side is the value at which that many neurons with that side have m_i > τ, which amounts to taking the top K_side by mass. MCNS: 8,410 / 8,670. On a new pair, this uses only the source labels and the target's neuron count.

- At ρ = 1, prior matching makes the soft rare-type count Σ_i m_i equal to K by construction. The rule is the user's "expected number of candidates = expected number of rare-type neurons". It is rank-based, so it does not depend on how well the masses are calibrated on a sparse target.
- Neurons without a side cannot be matched. They are excluded from selection and keep their labeling.

**Variants (no further hyperparameters):**
- `mass`: top K_side by m_i.
- `mass_fill`: the current candidates (NTAC type 2–4) plus the highest-mass non-candidates, up to K_side. If there are already K_side or more, the set is unchanged.
- `union`: NTAC type 2–4 ∪ prior-matched type 2–4. Free, with no hyperparameter at all. It is 22,754 candidates on MCNS, more than the matching capacity, so the surplus goes to dummies.
- `band9` (follow-up only): the stage-1 type has 2–9 neurons, *and* FAFB's 5–9-neuron types join the matching rows (+4,379 / +4,424). Finding 1 motivates it. It doubles the matching size, so it runs only if R2 shows that mass selection still misses the 5–9 predictions.

**Code changes:**
- `experiments/b_gnn_celltypes/candidate_selection.py` (new):
  - `--stage mass --stage1 <run>`:
    - loads the run's `selected.json`, the m2 `model.pt` files and the cached prior-matching weights: `prior_weights.npz` for new runs, `soft_gnn.npz["weights"]` for the existing MCNS run;
    - builds `lean_label_free.LazyProbs` on the chosen target graph (degraded as recorded in `selected.json`);
    - computes, chunk by chunk, the ensemble mass on 2–4 types and on 5–9 types, per-seed masses, the ensemble max probability and the top rare type;
    - asserts that its argmax equals `pred_selected` (as `soft_probabilities` does);
    - writes `results/size_threshold/m2/<run>/cell_type/rare_mass.parquet`.
  - `--stage select`: builds the candidate sets above. It writes `candidates_<name>.parquet` (root ids) and `selection.csv`, with the count per side, K_side, recall, precision and the other-type and unlabeled shares. Labels are used only for these columns.
- `experiments/b_gnn_celltypes/graph_matching_mcns.py`:
  - `candidates()` / `--candidates` accept `file:<name>`, which reads `candidates_<name>.parquet` from the stage-1 directory. `gnn` and `oracle` are unchanged.
  - `graphs()` passes the stage-1 run's degradation (`target_retention`, `retention_seed`, `target_exclude`, `target_crop` from `selected.json`) to `load_graph`, so the sandbox matching sees the same thinned copy. The `root_id` assertion in `setup` still guards the node order.
  - `--candidates file:*` runs skip `soft_start` (barycenter only).
- `experiments/b_gnn_celltypes/label_free.py` `run_label` (additive):
  - saves `prior_weights.npz`: the 3 × K weights of every (variant, norm), about 1 MB;
  - writes `retention_seed` into `selected.json`.

  New stage-1 runs then never need the Sinkhorn again for mass selection.
- `experiments/b_gnn_celltypes/size_threshold.py`:
  - `--target-retention`, `--retention-seed`, `--target-crop`, `--target-exclude` for the `label_free` stage. They are currently hard-coded to the full graph in `run_label_free`;
  - `label_free_dir` names degraded runs `label_free_mcns_r0.35_size_agreement`.

  Defaults are unchanged, so existing runs are skipped as before.

**Reuse:**
- full MCNS: the stage-1 run is reused as is (consistency rule, ρ = 0.75, the run behind the 74.5 %), together with the `soft_gnn.npz` weights;
- sandbox: the new stage-1 run and its `prior_weights.npz`.

No retraining.

**Evaluation:**
- *Selection only (R2, R5):* recall and precision at equal count, compared with the current set and with each other.
- *Two-stage (R8, R9):* graph matching with the chosen set, on the sandbox and on full MCNS.

**Adoption:**
- Sandbox 2–4 accuracy (two-stage) rises ≥ 3 points over the sandbox baseline, and its macro-F1 does not fall.
- On full MCNS, the 2–4 accuracy does not fall by more than 1 point, and the all-labeled accuracy not by more than 0.5 points.
- Candidate recall rises on both.
- Also reported: the gap to oracle candidates.

## Idea 2: two-tier

**Final labeling:**
- **Candidates assigned to a FAFB neuron** get its type, as now.
- **Every other neuron** gets the coarse model's type if its m2 stage-1 type lies in the coarse label space (≥ 10 FAFB neurons for m10), and otherwise its m2 type. "Every other neuron" means non-candidates, candidates left on dummies, candidates without a side and, with idea 3, rejected candidates.
  - The coarse model cannot output 5–9 types, which is why "m10 for all non-candidates" loses the 5–19 stratum (21.7 vs 59.8 %).
  - The gate is the coarse model's own minimum type size, not a tuned value.
- **The m2 model only flags candidates**, by idea 1's rule or the current one.

  A side effect: a falsely flagged common-type neuron that the matching rejects or leaves on a dummy now falls back to the better coarse type. That lowers the cost of a larger candidate set.

**Coarse model:** m10. In the preview on full MCNS it beats m20 (88.8 vs 88.0 % on all labeled). m20 is reported alongside, and the choice between them is made on full MCNS and the sandbox together. For the VNC test, the coarse model is the same pipeline at that pair's ≥ 10 label space.

**Code changes:**
- `graph_matching_mcns.py`: `--fallback <coarse stage-1 run>` (e.g. `size_threshold/m10/label_free`). Every `evaluate` call then adds a second row, "<method> + two-tier m10", computed from the same matches in seconds. The assertion is that the coarse predictions have the same `root_id` order.
- `candidate_selection.py --stage report` collects both rows.

**Reuse:**
- full MCNS: `results/size_threshold/m10/label_free` (all 144,725 neurons) and `results/label_free` (m20) are already saved;
- sandbox: needs one m10 stage-1 run on the same thinned copy (R10, 1 h). The m10 models exist (`results/size_threshold/m10/models`).

**Evaluation and adoption:**
- All-labeled accuracy rises ≥ 1 point on full MCNS *and* on the sandbox.
- All-labeled macro-F1 does not fall by more than 0.01.
- The 2–4 stratum is unchanged, as it is by construction for a fixed candidate set.
- Reported by size bin and by region.

## Idea 3: many-to-one matching

The objective stays f(Q) = Σ_terms ⟨F_k, Q_a M_k Q_cᵀ⟩, with Q the FAFB × candidate assignment per hemisphere. Only the feasible set changes. Three steps, cheapest first:

- **3a. Absorb and reject (post hoc, no new optimization):**
  - Start from the 1:1 FAQ result. Then take the gain of putting candidate m on FAFB neuron f, with all other assignments fixed: g(f, m) = the partial derivative of f along that assignment.
  - Compute it once against a degree-preserving null, which leaves no free hyperparameter: F̃ = F − k_out k_inᵀ / W per term block, likewise for M. A positive centered gain means more agreement than chance.
  - *Absorb:* each candidate on a dummy goes to argmax_f g̃(f, m), and many-to-one is allowed, if that gain is > 0.
  - *Reject:* a matched candidate whose own centered gain is ≤ 0 reverts to the fallback labeling (idea 2).
  - Both are sparse products, seconds per block.
- **3b. Capacity-relaxed Frank–Wolfe:**
  - The feasible set is {Q ≥ 0 : every candidate column sums to 1 (or 0 on a dummy row), every FAFB row sums to ≤ c}, with c = 2.
  - The FW linear step is a transportation problem. It is solved exactly as `linear_sum_assignment` on the gradient with every FAFB row repeated c times, a rectangular (c·R + dummies) × K problem. The objective and the gradient are computed on the unreplicated Q, so F and M do not grow.
  - It warm-starts from the 1:1 solution, which is feasible, with 10 iterations. The result is projected onto the capacity polytope's vertices by the same LAP.
  - c = 2 is fixed a priori, as the smallest relaxation (a type of s FAFB neurons may absorb up to 2s candidates). It is not tuned.
- **3c (not planned unless 3b fails on memory):** matching on the type quotient graph, with rows = rare types, F̄ = the mean FAFB type-to-type adjacency and per-type capacity. It is smaller, but it changes the objective, so it is listed only as a fallback.

**Code changes:**
- `graph_matching_mcns.py`:
  - `match()` stores `fafb_root_id` (−1 on a dummy) and the raw and centered gains of every candidate in `matches_*.parquet`, which today holds only types;
  - a function `absorb_reject(F, M, blocks, matches)` adds rows "<method> + absorb" and "+ absorb/reject" to every run;
  - `--capacity c` runs 3b after the 1:1 FAQ (`faq_blocks` gets an `lmo` argument; a new `_lap_capacity(g, c)` does the replicated-row assignment), as method `gm_bary_cap2`.
- `tests/test_graph_matching.py`: tiny cases for `_lap_capacity` (it reduces to the 1:1 LAP at c = 1), the centered gain, and absorb/reject on a planted 6-node graph. They run in seconds.

**Reuse:** 3a needs the 1:1 indices, which no saved run has. Any new matching run (R6–R9) produces them, so 3a costs no extra run. 3b needs its own run.

**Evaluation:**
- Finding 4 bounds the matching-side gain on full MCNS at about 7 points with oracle candidates. So the deciding run is **3b on full-MCNS oracle candidates** (R11), against 85.3 %.
- Then 3a/3b on the sandbox with the adopted selection.

**Adoption:**
- Oracle-candidate 2–4 accuracy rises ≥ 2 points on full MCNS, and
- with GNN candidates the 2–4 accuracy rises ≥ 1 point on full MCNS and the sandbox, and
- all-labeled accuracy does not fall.

3a is adopted on the same rule, independently of 3b.

## Metrics, reported for every run

- **Candidates:** count per side and without a side; recall (labeled rare-type neurons selected); composition (rare-type / other type / unlabeled); candidates on dummies; FAFB rows unmatched.
- **Accuracy % and macro-F1** (macro-F1 as `stratify.per_type`, as in `metrics.csv`) for stage 1, two-stage, and two-stage + two-tier, on:
  - the 2–4 stratum;
  - its matched part;
  - all labeled neurons;
  - the 2–4 stratum by region;
  - size bins 5–19 and ≥ 20.
- **Gap to oracle candidates** on the same target copy.
- **Runtime and peak memory.**

## Runs (in order; each needs an explicit OK)

Times come from the logs:
- stage 1 at m2 on full MCNS: 161 min, of which the prior matching is about 25 min per (variant, norm), × 6;
- matching: 27 min at n = 9,342 (4.5 GB), 26 min at n = 8,183 on BANC.

Matching scales as roughly n²–n^2.5 in time and n² in memory.

| Run | What | Inputs (reused) | Est. time | Est. peak RAM | Decision it informs |
|---|---|---|---|---|---|
| R0 | Code changes above, plus unit tests (`pytest tests/test_graph_matching.py tests/test_lean.py`) | – | < 1 min compute | < 1 GB | – |
| R1 | `candidate_selection.py --stage mass` on full MCNS | m2 models, `soft_gnn.npz` weights, m2 stage 1 | 5–10 min | ~3 GB | – |
| R2 | `--stage select` on full MCNS: recall/precision of `mass`, `mass_fill`, `union`, `band9` at their counts, and recall vs K curves | R1, predictions.parquet | < 1 min | < 1 GB | **Is mass better than argmax at equal count?** Which ≤ 2 sets go to matching; whether `band9` is needed |
| R3 | Two-tier merge (m10, m20) on full MCNS from saved outputs (formalizes the preview) | m2/m10/m20 predictions, `matches_gnn_gm_bary` | < 1 min | < 1 GB | Idea 2 on the development pair |
| R4 | **Sandbox stage 1**: `size_threshold.py --min-sizes 2 --stages label_free --select size_agreement --target-retention 0.35` | m2 models | 150–165 min | ~7–8 GB | – |
| R5 | `--stage mass` + `--stage select` on the sandbox | R4 incl. `prior_weights.npz` | 5–10 min | ~3 GB | **Does the sandbox reproduce BANC's low recall** (≤ 65 %)? If not → R4b |
| R4b | (only if R5 fails) R4 at retention 0.25, then R5 again | m2 models | 155–175 min | ~7–8 GB | Sandbox choice |
| R6 | Sandbox two-stage baseline (`--candidates gnn`, barycenter) | R4 | 20–27 min | ~4–4.5 GB | Sandbox baseline; 3a rows for free |
| R7 | Sandbox oracle candidates | R4 | 18–25 min | ~3.5–4 GB | Gap to oracle; with R6 completes the reproduction check |
| R8 | Sandbox two-stage with the best set from R5 (`file:mass_fill` or `file:union`) | R4, R5 | 25–45 min | 4–7 GB | **Settles idea 1** (adoption rule above); 3a on a larger set |
| R9 | Full MCNS two-stage with the same set | R1, R2 | 25–45 min | 4–7 GB | Idea 1 on the development pair (guard: no loss) |
| R10 | Sandbox m10 stage 1 (`--min-sizes 10`, same degradation), then the two-tier rows for R6/R8 | m10 models, R4 copy | 55–65 min + < 1 min | ~4 GB | **Settles idea 2** (with R3) |
| R11 | Full MCNS **oracle candidates with `--capacity 2`** | MCNS labels (diagnostic) | 35–55 min | ~5–6 GB | **Settles idea 3b** |
| R12 | Sandbox `--capacity 2` with the adopted set | R4, R5 | 35–55 min | ~5–7 GB | Idea 3b on the BANC proxy |
| R13 | (optional) `band9` on full MCNS: candidates with a 2–9 type, rows of the 2–9 FAFB types (n ≈ 14k) | R1 | 70–90 min | ~10 GB (tight) | Only if R2 shows that mass selection still misses the 5–9 predictions |
| R14 | (optional) Second sandbox realization (`--retention-seed 3`) of the final combination: stage 1 + baseline + final | m2 models | 150–165 + 2 × 25–45 min | ~7–8 GB | Robustness before freezing the protocol |

**Total:**
- **R0–R12:** about 7.5 h of compute (R4 alone 2.6 h), at most 8 GB at a time;
- **with the R4b fallback:** about 11 h;
- **R13 and R14:** another 1.5 h and 3.5 h.

R1–R3 take about 15 min together and already decide which selection sets are worth matching.

**Order and what each step settles:**
- **R1–R3 first** (15 min). If no new set beats the current one on recall at equal count by ≥ 3 points, idea 1 drops to `union` alone, which is free.
- **R4 overnight**, then R5. Nothing is fixed on the sandbox until R5–R7 show that it reproduces BANC's low recall and its oracle gap.
- **R8 settles idea 1, R10 idea 2, R11 idea 3.**
- **R6–R9 also give 3a's rows** with no extra run.
- **Order by information per hour:** R10 is independent of R6–R9 and could follow R4 directly. R11 needs only the full-MCNS setup and can run any time after R0.

## After the runs

- The adopted combination is frozen in a new protocol before the nerve-cord test: selection rule, fallback, matching variant, every rule for K, c, gates and the acceptance criterion.
- Its label-free rules are stated for a general source/target pair:
  - K from the source's rare share and the target's neuron count per side;
  - the gate = the coarse model's own minimum type size;
  - c = 2;
  - acceptance by centered gain > 0.

  Then they apply to MANC → BANC / MCNS VNC unchanged.
- The disclosure that full MCNS uses the consistency-rule stage 1 (ρ = 0.75), and the sandbox uses `size_agreement`, carries over.
