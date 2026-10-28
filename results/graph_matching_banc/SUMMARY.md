# Two-stage typing of the rarest cell types on BANC (pre-registered test)

Protocol: `docs/banc_rare_protocol.md`, committed with the code that ran here in `0d1ee67` (branch
`claude/reverse-transfer`) before any run on BANC. Runs 1–4 executed exactly as written, 2026-10-19
20:09–23:54. Logs: `results/size_threshold/label_free_banc_size_agreement.log`, `run_*.log` here,
`results/graph_matching_banc_chain.log`.

**Deviation:** peak memory was not recorded (the protocol lists "runtime and memory"). There were no other
deviations and no code changes after the protocol commit.

## Stage 1: label-free FAFB → BANC at minimum type size 2 (run 1, 2 h 21 min)

- Chosen by `size_agreement`: `density`, target normalization, rho 1. This is also the best of the
  30 candidates (post hoc), a gap of 0.0 points.
- Every rule in `flybrain/selection.py` picks the same candidate except `agreement`, which picks a
  collapsed labeling (1.2 %).

Accuracy % / macro-F1 on all reference-A neurons (88,639):

| No prior matching | Prior matched (selected) | + NTAC | Best candidate |
|---|---|---|---|
| 24.8 / 0.05 | 43.2 / 0.25 | 47.9 / 0.19 | 43.2 / 0.25 |

After NTAC: central brain 26.0 % / 0.17, optic lobe 62.2 % / 0.34, reference B 25.8 % / 0.17.
Reference B agrees with reference A on 95.6 % of the neurons that have both labels.

## Stage 2: graph matching (accuracy % / macro-F1)

| Method | 2–4, ref A (14,512) | 2–4 matched, ref A | All ref A (88,639) | 2–4, ref B (13,848) | 2–4 matched, ref B | All ref B (35,736) |
|---|---|---|---|---|---|---|
| Stage 1 (label-free GNN) | 22.2 / 0.19 | 41.3 / 0.27 (7,732) | 47.9 / 0.19 | 21.3 / 0.18 | 40.1 / 0.27 (7,309) | 25.8 / 0.18 |
| **Two-stage, `density` (run 2, primary)** | **41.3 / 0.47** | **77.1 / 0.68** | **51.0 / 0.40** | **40.6 / 0.47** | **76.7 / 0.68** | **33.2 / 0.39** |
| Two-stage, `raw` (run 4) | 40.4 / 0.46 | 75.5 / 0.67 | 50.9 / 0.40 | 39.7 / 0.46 | 74.8 / 0.67 | 32.9 / 0.39 |
| Two-stage, `scaled` (run 4) | 40.3 / 0.46 | 75.3 / 0.67 | 50.8 / 0.39 | 39.6 / 0.45 | 74.7 / 0.66 | 32.8 / 0.39 |
| Oracle candidates (run 3, uses labels) | 65.9 / 0.70 | 67.8 / 0.70 (14,064) | 55.0 / 0.57 | 65.5 / 0.70 | 67.4 / 0.70 (13,406) | 42.9 / 0.58 |

Reference A, 2–4 stratum by BANC region:

| Region | Neurons | Stage 1 | Two-stage | Oracle |
|---|---|---|---|---|
| Central brain | 13,980 | 21.3 | 40.6 | 65.4 |
| Optic lobe | 523 | 45.9 | 61.2 | 82.0 |

Reference A, 2–4 stratum by FAFB super class of the true type:

| Region | Neurons | Stage 1 | Two-stage | Oracle |
|---|---|---|---|---|
| Central brain | 12,741 | 21.1 | 38.7 | 63.5 |
| Other (descending, ascending, …) | 1,163 | 24.2 | 61.8 | 85.5 |
| Visual projection | 453 | 40.6 | 58.7 | 81.5 |
| Sensory | 81 | 23.5 | 42.0 | 63.0 |
| Optic-lobe intrinsic | 74 | 58.1 | 62.2 | 79.7 |

### Counts

- **Candidates:** 12,604 (5,932 left, 6,049 right, 623 without a side).
  - Recall: 54.3 % of the 14,512 reference-A rare-type neurons are candidates (MCNS: 83.8 %).
  - Composition: 62.5 % rare-type, 19.9 % another type, 17.6 % unlabeled (2,219 neurons, typed
    but not scored).
- **Matching:**
  - BANC has fewer candidates than FAFB has rare neurons in each hemisphere (FAFB: 8,183 / 8,182),
    so no candidate is left on padding, and 4,384 FAFB neurons stay unmatched (oracle: 2,301).
  - On rare-type candidates, matching fixes 3,100 stage-1 errors and breaks 330 correct types.
- **Edges among candidates:** in the `density` graph LL 163k, RR 189k, LR 90k, RL 83k; in
  `raw`/`scaled` 40 % of that. The FAFB rare subgraph has LL 236k, RR 250k.
- **Runtime per matching:** 26 min (`density`), 17 min (`raw`, `scaled`), 23 min (oracle).

## Expectations stated in advance

1. **Two-stage beats stage 1 by ≥ 10 points on the 2–4 stratum, against both references: met.**
   +19.1 (ref A, 22.2 → 41.3) and +19.3 (ref B, 21.3 → 40.6). FAFB → MCNS: +20.3.
2. **Over all ref-A neurons, macro-F1 rises by ≥ 0.05 and accuracy does not fall: met.** Macro-F1
   0.194 → 0.400 (+0.21), accuracy 47.9 → 51.0 % (+3.1).
3. **The 2–4 stratum (ref A) stays below FAFB → MCNS (74.5 %): met.** 41.3 %.
4. **Matching withstands sparsity better than the GNN: met.** With oracle candidates, matching keeps
   65.9 / 85.3 = 0.77 of its MCNS accuracy, against 22.2 / 54.2 = 0.41 for stage 1.

## Findings

- **The gain carries over to a held-out connectome almost unchanged in size.** Accuracy roughly
  doubles on the rarest types (+19 points, macro-F1 0.19 → 0.47) on a brain about three times
  sparser than FAFB, with all choices fixed in advance. The gain is the same against the
  morphology-matched reference B as against BANC's own typing.
- **Candidate selection is the bottleneck, more so than on MCNS.** The GNN finds only 54 % of the
  rare-type neurons, against 84 % on MCNS. On the candidates it does find, matching reaches 77.1 %
  (MCNS 89.1 %).
  - Oracle candidates would add 24.6 points (65.9 vs 41.3 %).
  - Matching across animals and densities costs 19.4 points relative to MCNS (85.3 → 65.9 % with
    oracle candidates).
- **Matching is insensitive to the graph variant.** `raw` and `scaled` keep only 40 % of
  `density`'s edges among the candidates, yet lose only 0.9–1.0 points.
- **The largest gains are outside the central brain's intrinsic neurons:** descending and other
  neurons 24.2 → 61.8 %, visual projection 40.6 → 58.7 %. Central-brain types rise 21.1 → 38.7 %.

## Caveats

- One run per setting. The barycenter start is deterministic.
- Reference A comes from BANC's FAFB alignment (morphology + connectivity). On the rare stratum,
  reference B (NBLAST with review) gives the same numbers within 1 point.
- Stage 1 used the protocol's `size_agreement` rule. FAFB → MCNS used type consistency; on BANC
  both rules pick the same candidate.
