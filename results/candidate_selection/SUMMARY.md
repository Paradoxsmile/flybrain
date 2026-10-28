# Candidate selection: first runs (R1–R3, full FAFB → MCNS)

Plan: `docs/candidate_selection_plan.md`. Script: `experiments/b_gnn_celltypes/candidate_selection.py`.
Logs: `r1_mass_mcns.log`, `r2_select_mcns.log`, `r3_twotier_mcns.log` (here). Outputs next to the
stage-1 run `results/size_threshold/m2/label_free/cell_type/`: `rare_mass.parquet`,
`selection.csv`, `selection_curve.csv`, `candidates_{union,mass,mass_fill,mass_at_gnn}.parquet`;
`twotier_matches_gnn_gm_bary.csv` here. Development data only, no BANC.

## R1: rare-type mass of every MCNS neuron (2 min, 2.2 GB peak)

- Three GNN forward passes on the chosen graph (`raw`, source normalization, ρ = 0.75), with the
  prior-matching weights cached in `results/graph_matching_mcns/soft_gnn.npz`. No Sinkhorn.
- The recomputed ensemble's argmax equals `pred_selected` on 100.0000 % of the 144,725 neurons.
- The mean mass on 2–4-neuron types is 0.160, i.e. a soft count of 23,200 rare-type neurons. That
  is more than the label-free expectation K = 17,079 (8,411 left, 8,668 right) because ρ < 1.

## R2: candidate sets (recall over the 13,742 labeled rare-type neurons)

| Set | Candidates (left / right / no side) | Recall | Precision | Other type | Unlabeled | Rare gained / lost vs current |
|---|---|---|---|---|---|---|
| `gnn` (current: NTAC type 2–4) | 18,479 (8,872 / 9,342 / 265) | 83.8 % | 62.3 % | 14.0 % | 23.6 % | – |
| `selected` (prior-matched type 2–4) | 19,029 | 85.9 % | 62.0 % | 16.4 % | 21.5 % | 1,100 / 817 |
| `union` | 22,754 (10,943 / 11,446 / 365) | **91.8 %** | 55.5 % | 19.8 % | 24.7 % | 1,100 / 0 |
| `mass` (top K per side) | 17,079 (8,411 / 8,668 / 0) | 81.6 % | 65.6 % | 10.4 % | 24.0 % | 888 / 1,198 |
| `mass_fill` (current + mass up to K) | = current | 83.8 % | | | | 0 / 0 |
| `mass_at_gnn` (mass, current count per side) | 18,214 (8,872 / 9,342 / 0) | 84.4 % | 63.7 % | 11.7 % | 24.6 % | 1,034 / 952 |
| `band9` (NTAC type 2–9, diagnostic) | 28,135 | 94.7 % | 46.3 % | 29.1 % | 24.7 % | 1,497 / 0 |

Mass ranking against the count (per side, multiple of K): 0.5 K 53.2 % recall (85.6 % precision),
0.9 K 79.1 %, 1.0 K 81.6 %, 1.1 K 85.7 %, 1.25 K 90.4 %, 1.5 K 95.5 %, 2 K 98.5 %.

- **At equal count, mass beats the argmax by only 0.6 points** (84.4 vs 83.8 %, precision 63.7 vs
  62.3 %). The plan's rule (≥ 3 points at equal count) is not met on full MCNS.
- **On full MCNS the current set already exceeds K on both sides**, so `mass_fill` is the current
  set. Mass selection can only matter where stage 1 selects fewer than K. That was BANC's case
  (5,932 / 6,049 candidates against K ≈ 6,900 / 7,400), and the sandbox (R5) has to test it.
- **`union` is free and reaches 91.8 %**, about what the mass ranking reaches at the same count
  (~92 %, interpolated). It also takes 52 % of the missed neurons with a 5–9-neuron stage-1 type
  (`band9` would take all of them, at 28,135 candidates and twice the matching size).

## R3: two-tier fallback on the existing two-stage matches (seconds)

Accuracy % / macro-F1. Candidates: `gnn`, barycenter matching (`matches_gnn_gm_bary.parquet`,
which predates `fafb_root_id`, so every candidate counts as assigned). Neurons not assigned by the
matching take the coarse type where their m2 type has ≥ 10 (m10) or ≥ 20 (m20) FAFB neurons.

| Subset | Two-stage | + two-tier m10 | + two-tier m20 |
|---|---|---|---|
| 2–4 stratum (13,742) | 74.5 / 0.782 | 74.5 / 0.782 | 74.5 / 0.782 |
| 5–19 | 59.8 / 0.526 | 60.1 / 0.529 | 59.8 / 0.526 |
| ≥ 20 | 90.2 / 0.779 | **93.2 / 0.822** | 92.1 / 0.779 |
| **All labeled (133,356)** | 86.3 / 0.735 | **88.8 / 0.738** | 88.0 / 0.735 |
| all labeled, optic-lobe intrinsic | 93.1 / 0.925 | 96.7 / 0.951 | 96.9 / 0.946 |
| all labeled, visual projection | 94.0 / 0.912 | 96.4 / 0.929 | 94.2 / 0.918 |
| all labeled, sensory | 59.5 / 0.447 | 62.0 / 0.480 | 58.4 / 0.419 |
| all labeled, other | 58.8 / 0.643 | 59.1 / 0.643 | 58.8 / 0.643 |
| all labeled, central brain | 71.9 / 0.741 | **70.9** / 0.742 | 67.5 / 0.740 |

(Region = FAFB super class of the true type. "2–4, matched candidates" is 88.9 % here: it counts
all candidates in the parquet, including the 265 without a side, against 89.1 % in
`results/graph_matching_mcns/SUMMARY.md`.)

- On full MCNS, m10 meets the adoption rule for all labeled neurons (+2.5 points, macro-F1
  +0.003), with the 2–4 stratum unchanged.
- The gain comes from the optic lobe, visual projection and sensory neurons. **Central-brain
  types lose 1.0 point** (m20: 4.4). This is the m10 model being worse than m2 on central-brain
  types of ≥ 10 neurons, and is reported with the sandbox result (R10).

## R4: sandbox stage 1, MCNS thinned to 35 % (`retention_seed` 2, `size_agreement`)

`results/size_threshold/m2/label_free_mcns_r0.35_size_agreement` (log
`results/size_threshold/label_free_mcns_r0.35_size_agreement.log`). Peak RAM 7.07 GB. Wall time
50 h only because the laptop slept twice. It computed about 2.5 h; `seconds` in `selected.json`
includes the sleep.

| | No prior matching | Prior matched (selected) | + NTAC | 2–4 stratum (13,742): selected / + NTAC |
|---|---|---|---|---|
| Sandbox r0.35 | 35.4 | 52.7 / 0.38 | 61.7 / 0.34 | 45.6 / 39.3 |
| Full MCNS (consistency, ρ 0.75) | – | 80.1 | 84.2 / 0.51 | – / 54.2 |
| BANC (test, ref A) | 24.8 | 43.2 / 0.25 | 47.9 / 0.19 | – / 22.2 |

- Every rule in `flybrain/selection.py` picks `raw`, target normalization, ρ 1, which is also the
  best of the 30 candidates (BANC: `density`, target, ρ 1, also the best).
- On the 2–4 stratum, NTAC *lowers* the sandbox accuracy (45.6 → 39.3 %).

## R5: candidate sets on the sandbox (2 min, 2.5 GB; argmax == `pred_selected` 100 %)

| Set | Candidates (left / right / no side) | Recall | Precision | Other type | Unlabeled | Rare gained / lost |
|---|---|---|---|---|---|---|
| `gnn` (current) | 14,625 (7,255 / 7,283 / 87) | **70.2 %** | 66.0 % | 14.9 % | 19.1 % | – |
| `selected` | 17,493 | 71.9 % | 56.5 % | 25.3 % | 18.2 % | 1,592 / 1,361 |
| `union` | 21,174 (10,696 / 10,339 / 139) | 81.8 % | 53.1 % | 26.9 % | 20.0 % | 1,592 / 0 |
| `mass` | 17,079 (8,411 / 8,668 / 0) | 75.1 % | 60.4 % | 20.0 % | 19.5 % | 1,686 / 1,010 |
| `mass_fill` | 17,166 (8,411 / 8,668 / 87) | 77.3 % | 61.8 % | 18.3 % | 19.9 % | 969 / 0 |
| `mass_at_gnn` | 14,538 (7,255 / 7,283 / 0) | 69.6 % | 65.7 % | 16.3 % | 18.0 % | 1,293 / 1,383 |
| `band9` (diagnostic) | 23,243 | 82.3 % | 48.7 % | 29.8 % | 21.5 % | 1,667 / 0 |

Mass ranking against the count: 0.9 K 71.5 %, 1.0 K 75.1 %, 1.1 K 78.5 %, 1.25 K 82.9 %, 1.5 K 88.3 %.

- **The sandbox at 35 % misses the reproduction criterion.** Recall is 70.2 %: plan ≤ 65 %,
  BANC 54.3 %, full MCNS 83.8 %. As the plan says for recall above 70 %, R4b (retention 0.25)
  runs instead, and R6/R7 wait for it.
- Unlike full MCNS, the current set here is below K on both sides (7,255 / 7,283 vs 8,411 /
  8,668), as on BANC. `mass_fill` raises recall by 7.1 points with 2,541 more candidates. At equal
  count, mass is again no better than the argmax (69.6 vs 70.2 %). So the gain comes from
  selecting *more* candidates, up to the label-free K, not from ranking them better.
- At about the same count, `union` (81.8 % at 21,174) and the mass ranking (82.9 % at 21,349) are
  close, as on full MCNS.

## R4b / R5b: sandbox at 25 % (`label_free_mcns_r0.25_size_agreement`; 2 h 39 min, 7.05 GB)

Stage 1:
- Every rule but two picks `raw`, target normalization, ρ 1, the best candidate.
- Type consistency and plain agreement pick a collapsed labeling (5.7 %), as type consistency did
  on BANC at ≥ 20.

| | No prior matching | Prior matched (selected) | + NTAC | 2–4 stratum: selected / + NTAC |
|---|---|---|---|---|
| Sandbox r0.25 | 17.5 | 34.3 / 0.22 | 40.2 / 0.15 | 26.1 / 19.2 |
| BANC (test, ref A) | 24.8 | 43.2 / 0.25 | 47.9 / 0.19 | – / 22.2 |

Candidate sets (R5b: mass 85 s, 2.15 GB, argmax == `pred_selected` 100 %):

| Set | Candidates (left / right / no side) | Recall | Precision | Other type | Unlabeled | Rare gained / lost |
|---|---|---|---|---|---|---|
| `gnn` (current) | 12,167 (6,110 / 5,986 / 71) | **54.4 %** | 61.5 % | 20.6 % | 17.9 % | – |
| `selected` | 15,825 | 57.5 % | 49.9 % | 33.4 % | 16.7 % | 2,148 / 1,729 |
| `union` | 20,123 (10,086 / 9,916 / 121) | 70.0 % | 47.8 % | 34.0 % | 18.2 % | 2,148 / 0 |
| `mass` | 17,079 (8,411 / 8,668 / 0) | 67.1 % | 54.0 % | 28.1 % | 17.9 % | 2,686 / 939 |
| `mass_fill` | 17,150 (8,411 / 8,668 / 71) | **68.4 %** | 54.8 % | 26.7 % | 18.5 % | 1,926 / 0 |
| `mass_at_gnn` | 12,096 (6,110 / 5,986 / 0) | 55.2 % | 62.7 % | 21.5 % | 15.7 % | 1,790 / 1,677 |
| `band9` (diagnostic) | 19,083 | 66.1 % | 47.6 % | 33.1 % | 19.3 % | 1,607 / 0 |

Mass ranking against the count: 0.9 K 63.5 %, 1.0 K 67.1 %, 1.1 K 70.6 %, 1.25 K 74.5 %, 1.5 K 80.1 %.

- **The 25 % sandbox reproduces BANC's candidate selection.**
  - Recall is 54.4 % (BANC 54.3 %), with 6,110 / 5,986 candidates per side (BANC 5,932 / 6,049).
  - Precision is 61.5 % (BANC 62.5 %).
  - Stage 1 on the 2–4 stratum reaches 19.2 % (BANC 22.2 %).
  - The second criterion, an oracle gap of ≥ 15 points, needs R6/R7.
- **Filling to the label-free K adds 14.0 recall points** (`mass_fill` 68.4 %). The gain again
  comes from the count: at equal count, mass is +0.8 points (55.2 %).
- **`union` gives 70.0 % at 20,123 candidates**; the mass ranking at that count gives about 72 %
  (interpolated).

## R6 / R7: matching on the 25 % sandbox (`results/graph_matching_mcns_r0.25`)

Barycenter start, 30 iterations, `raw` graph. R6 21 min, R7 19 min, 3.4 GB each. Accuracy % /
macro-F1:

| Method | 2–4 stratum (13,742) | 2–4, matched candidates | All labeled (133,356) |
|---|---|---|---|
| Stage 1 | 19.2 / 0.156 | 35.4 / 0.231 (7,445) | 40.2 / 0.162 |
| **Two-stage, current candidates** (R6) | **47.3 / 0.531** | 87.2 / 0.789 | 43.1 / 0.447 |
| + absorb / reject | 47.3 / 0.531 | 87.3 / 0.788 | 43.1 / 0.447 |
| Oracle candidates (R7) | 82.0 / 0.838 | 82.3 / 0.842 | 46.7 / 0.686 |
| BANC (test): stage 1 → two-stage → oracle | 22.2 → 41.3 → 65.9 | 41.3 → 77.1 | 47.9 → 51.0 |

- **Reproduction check passed.** Candidate recall is 54.4 % (≤ 65 %), and the gap between
  two-stage and oracle is 34.7 points (≥ 15; BANC 24.6). The 25 % copy is the sandbox from here on.
- **The sandbox is harsher than BANC for stage 1 and milder for matching.** With oracle
  candidates it reaches 82.0 % (full MCNS 85.3 %, BANC 65.9 %). Matching a thinned copy of
  another animal is easier than matching BANC, where the reconstruction, the animal and the
  damaged nerves differ too. So the sandbox measures what candidate selection can win, and it
  overstates matching accuracy.
- **3a does nothing here.** There are fewer candidates than FAFB rows on both sides (4,269 FAFB
  rows stay unmatched), so no candidate sits on a dummy and absorb has nothing to do. Reject
  unassigns 763 candidates with centered gain ≤ 0 (oracle: 124). The 2–4 accuracy is unchanged,
  so the gain does not separate good from bad assignments.

## R8: matching with the larger candidate sets on the 25 % sandbox

`--candidates file:mass_fill` (n = 8,668, 21 min, 3.8 GB) and `file:union` (n = 10,086, 23 min,
4.9 GB). Accuracy % / macro-F1:

| Candidates | Count | Recall | 2–4 stratum | 2–4, matched | All labeled | 5–19 | ≥ 20 |
|---|---|---|---|---|---|---|---|
| current (`gnn`, R6) | 12,167 | 54.4 % | 47.3 / 0.531 | 87.2 / 0.789 | 43.1 / 0.447 | 18.0 | 44.8 |
| **`mass_fill`** | 17,150 | 68.4 % | **59.4 / 0.648** | 87.1 / 0.784 | **44.0 / 0.537** | 16.4 | 44.5 |
| `union` | 20,123 | 70.0 % | 60.1 / 0.649 | 86.2 / 0.760 | 44.0 / 0.537 | 16.1 | 44.5 |
| oracle (R7) | 13,702 | – | 82.0 / 0.838 | 82.3 / 0.842 | 46.7 / 0.686 | 18.0 | 44.8 |

With absorb: `mass_fill` 59.4 / 0.644, `union` 60.1 / 0.632 (macro-F1 −0.017). With absorb +
reject: 59.5 / 0.644 and 60.1 / 0.632.

- **Idea 1 meets its sandbox adoption rule by a wide margin.** `mass_fill` gives +12.1 points on
  the 2–4 stratum (≥ 3 required), macro-F1 +0.117, all labeled +0.9. It closes 35 % of the gap
  to oracle candidates (34.7 → 22.6 points).
- **On the rare-type candidates it adds, matching is as accurate as on the old ones.** Of the
  4,983 added candidates, 1,926 are labeled rare-type, and matching types 80.7 % of them (stage 1:
  0 %, by construction). 2,066 are labeled with another type: 495 of them were right at stage 1,
  38 still are, and 272 end on dummies. 991 are unlabeled.
- **`union` is 0.7 points better on the 2–4 stratum**, with 3,000 more candidates, a larger
  matching (n = 10,086) and slightly lower 5–19 accuracy.
- The full-MCNS guard (R9) is trivially met for `mass_fill`: there it is the current set (R2), so
  the 74.5 % result is unchanged. R9 runs `union` on full MCNS.
- **3a is not adopted.** Absorb lowers macro-F1 when there are many dummies (`union`: 3,637), and
  reject changes nothing.

## R9: `union` on full MCNS (`results/graph_matching_mcns_cand`; n = 11,446, 34 min, 6.2 GB)

| Method | 2–4 stratum | 2–4, matched | All labeled | 5–19 | ≥ 20 |
|---|---|---|---|---|---|
| Stage 1 | 54.2 / 0.490 | – | 84.2 / 0.513 | 59.8 | 90.2 |
| Two-stage, current candidates (published) | 74.5 / 0.782 | 88.9 / 0.875 | 86.3 / 0.735 | 59.8 | 90.2 |
| **Two-stage, `union`** | **81.6 / 0.834** | 89.1 / 0.863 | 86.9 / 0.776 | 57.6 | 90.1 |
| `union` + two-tier m10 | 81.6 / 0.834 | 89.1 / 0.863 | **89.4 / 0.779** | 57.9 | 93.1 |
| `union` + absorb | 81.2 / 0.803 | 88.6 / 0.830 | 86.5 / 0.750 | 55.6 | 89.9 |
| Oracle candidates | 85.3 / 0.868 | – | 87.4 / 0.809 | 59.8 | 90.2 |

- **The full-MCNS guard is passed, and `union` also helps there:** +7.1 points on the 2–4
  stratum. It closes 66 % of the gap to oracle candidates (10.8 → 3.7 points), with 6,024
  candidates left on dummies. The 5–19 stratum loses 2.2 points (neurons of 5–19 types
  sent to matching).
- Absorb again lowers macro-F1 (−0.031).

## R11: many-to-one matching, full-MCNS oracle candidates (`results/graph_matching_mcns_cap`; 39 min, 5.5 GB)

Frank–Wolfe from the one-to-one result, each FAFB neuron taking up to 2 candidates, 10 iterations
(13 min):
- **85.3 → 76.1 % on the 2–4 stratum** (macro-F1 0.868 → 0.713).
- FAFB rows left unmatched rise from 2,677 to 6,429: the relaxed objective piles candidates onto
  fewer, well-connected FAFB neurons.
- **Idea 3b fails its deciding run by a wide margin.** R12 (3b on the sandbox) is therefore
  skipped.
- The one-to-one oracle run reproduces 85.3 % exactly.

## R10: two-tier on the 25 % sandbox (m10 stage 1 on the same copy; 48 min, 7.3 GB)

The m10 stage 1 (`results/size_threshold/m10/label_free_mcns_r0.25_size_agreement`) chose `raw`,
target, ρ 1 (best candidate): 44.0 % prior matched, 49.2 % after NTAC, against 40.2 % for m2.
Neurons not assigned a FAFB neuron take the m10 type where their m2 type has ≥ 10 FAFB neurons:

| Candidates | All labeled: two-stage → + two-tier m10 | ≥ 20 | 5–19 | Central brain | Sensory |
|---|---|---|---|---|---|
| current | 43.1 / 0.447 → **46.9 / 0.452** | 44.8 → 49.2 | 18.0 → 20.1 | 37.2 → 36.8 | 38.4 → 34.1 |
| `mass_fill` | 44.0 / 0.537 → **47.7 / 0.541** | 44.5 → 48.9 | 16.4 → 18.2 | 41.5 → 41.0 | 38.1 → 33.8 |
| `union` | 44.0 / 0.537 → **47.7 / 0.541** | 44.5 → 48.9 | 16.1 → 17.7 | 42.0 → 41.5 | 38.1 → 33.9 |

- **Idea 2 meets its adoption rule on both targets:** all labeled +3.7 (sandbox) and +2.5 (full
  MCNS) points, macro-F1 not lower, 2–4 stratum unchanged.
- **The cost is consistent:** central-brain types −0.4 to −1.0 points on both targets, sensory −4.3
  on the sandbox (+2.5 on full MCNS).
- Coarse model: m10. m20 was worse on full MCNS (88.5 vs 89.4 with `union`) and was not run on the
  sandbox.

## R14: second realization of the 25 % sandbox (`--retention-seed 3`)

Chain `results/candidate_selection/r14/` (logs), stage 1 runs `…/m2|m10/label_free_mcns_r0.25_seed3_size_agreement`,
matching `results/graph_matching_mcns_r0.25_seed3`.

**Runtimes:**
- m2 stage 1: 2 h 37 min, 7.1 GB;
- m10 stage 1: 40 min, 8.05 GB;
- mass: 81 s, 2.3 GB;
- matchings: 20–22 min, 3.5–4.7 GB.

**Stage 1:** both stage-1 runs again chose `raw`, target, ρ 1, the best candidate. m2: 39.9 %
after NTAC, 19.1 % on the 2–4 stratum. m10: 48.8 %.

| | Seed 2 (R4b–R10) | Seed 3 (R14) |
|---|---|---|
| Candidate recall: current / `mass_fill` / `union` | 54.4 / 68.4 / 70.0 % | 52.6 / 68.6 / 69.5 % |
| Mass vs argmax at equal count | 55.2 vs 54.4 % | 53.2 vs 52.6 % |
| 2–4 stratum: stage 1 → two-stage, current candidates | 19.2 → 47.3 / 0.531 | 19.1 → 45.6 / 0.511 |
| 2–4 stratum, `mass_fill` | 59.4 / 0.648 | 59.3 / 0.646 |
| 2–4 stratum, `union` | 60.1 / 0.649 | 59.3 / 0.641 |
| All labeled, `union`: two-stage → + two-tier m10 | 44.0 → 47.7 | 43.6 → 47.3 |
| Two-tier m10 cost (current candidates): central brain / sensory | −0.4 / −4.3 | −0.5 / −4.7 |

- **Every finding replicates.** Selecting more candidates adds 12–14 points on the 2–4 stratum.
  Ranking by mass does not beat the argmax at equal count. Two-tier adds 3.7 points over all
  labeled neurons, with the same small losses on central-brain and sensory types.
- **`union` and `mass_fill` are tied on seed 3** (59.3 %; macro-F1 0.641 vs 0.646). Over the two
  realizations, `union` is ahead by 0.35 points on accuracy and behind by 0.002 on macro-F1. The
  case for `union` rests on full MCNS (+7.1 points, where `mass_fill` cannot act) and on its
  having no hyperparameter. `mass_fill` gives a smaller matching.

## Conclusions (development only; to be frozen in a new protocol)

| Pipeline | Sandbox 25 %: 2–4 / all labeled | Full MCNS: 2–4 / all labeled |
|---|---|---|
| Published (current candidates, m2 for the rest) | 47.3 / 0.531 · 43.1 / 0.447 | 74.5 / 0.782 · 86.3 / 0.735 |
| `mass_fill` + two-tier m10 | 59.4 / 0.648 · 47.7 / 0.541 | = published + two-tier: 74.5 / 0.782 · 88.8 / 0.738 |
| **`union` + two-tier m10** | **60.1 / 0.649 · 47.7 / 0.541** | **81.6 / 0.834 · 89.4 / 0.779** |
| Oracle candidates (diagnostic) | 82.0 / 0.838 · 46.7 / 0.686 | 85.3 / 0.868 · 87.4 / 0.809 |

1. **Candidate selection was the lever, and the fix is to select more candidates, not to rank
   them better.**
   - At equal count, rare-type probability mass beats the argmax by under 1 recall point
     everywhere.
   - What helps is selecting more neurons:
     - up to the label-free expected count K (`mass_fill`: recall 54 → 68 % on the sandbox), or
     - the union of the rare-type argmax before and after NTAC (`union`: 54 → 70 % on the sandbox,
       84 → 92 % on full MCNS).

   Matching keeps its accuracy on the added rare-type candidates (81 % on the sandbox), and the
   one-to-one matching with dummies absorbs most of the false positives.
2. **`union` is the better rule on the development data:**
   - +0.7 points on the sandbox and +7.1 on full MCNS, where `mass_fill` can do nothing because
     the current set already exceeds K;
   - it has no hyperparameter;
   - the cost is a larger matching (n up to 11.4k, 34 min, 6.2 GB).
3. **The two-tier fallback (m10) adds 2.5–3.7 points over all labeled neurons** at no cost to the
   rare types, with a small loss on central-brain types.
4. **Many-to-one matching does not help:**
   - absorb lowers macro-F1, and reject changes nothing;
   - the capacity relaxation loses 9 points with oracle candidates.

   The one-to-one objective stays.
5. **Open before a protocol:**
   - two thinning realizations of one MCNS copy (R14 replicated seed 2);
   - the sandbox overstates matching accuracy relative to BANC (oracle 82 vs 66 %);
   - the full-MCNS stage 1 used type consistency, the sandbox `size_agreement`.

## Status

- Code (uncommitted):
  - `candidate_selection.py` (new);
  - `graph_matching_mcns.py`: degraded targets, `--candidates file:<name>`, `--fallback`,
    absorb / reject rows on every matching, `--capacity`;
  - `size_threshold.py`: `--target-retention` / `--retention-seed` / `--target-crop` /
    `--target-exclude`;
  - `label_free.py`: `prior_weights.npz`, `retention_seed` in `selected.json`.
- Tests: `tests/test_graph_matching.py` +7, including an end-to-end `match()` on synthetic
  brains. The test files `test_graph_matching`, `test_lean`, `test_analysis_helpers`,
  `test_selection`, `test_mcns`, `test_thinning`, `test_banc`, `test_coverage` and
  `test_hemibrain` pass.
- R0–R11 done as planned:
  - R4b replaced R4;
  - R12 was skipped after R11;
  - R13 was not needed: `union` and `mass_fill` beat `band9`.
- R14 (second sandbox realization, seed 3) done: the findings replicate.
