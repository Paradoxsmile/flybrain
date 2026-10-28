# Nerve-cord test of the two-stage rare-type pipeline: results

Pre-registered in [docs/vnc_rare_protocol.md](../../docs/vnc_rare_protocol.md), fixed in commit **5119069** (2026-10-27) before any run on a target cord. Source: MANC v1.2.1. Targets: the BANC v888 nerve cord and the MCNS v1.0 nerve cord. All runs on the CPU, 2026-10-27, in the order of the protocol (BANC first). Tables and checks: `experiments/b_gnn_celltypes/vnc_report.py` → `results/vnc_rare/report.json`; logs in `results/vnc_rare/logs/`.

## Outcome

| Expectation (reference A, 2–4 stratum unless stated) | BANC cord | MCNS cord |
|---|---|---|
| 1. `union` beats `gnn` by ≥ 3 points, macro-F1 not > 0.01 lower | **met** (+8.9; F1 +0.061) | **met** (+5.1; F1 +0.029) |
| 2. Final ≥ stage 1 + 15 points | **met** (+21.4; ref B +20.3) | **not met** (+10.2) |
| 3. Pipeline closes ≥ half of the gap to oracle candidates | **met** (0.60) | **met** (0.68) |
| 4. Two-tier fallback: ≥ +0.5 points over all labeled, F1 ≥ −0.01, 2–4 unchanged | **not met** (−0.4; F1 −0.002) | **not met** (−1.2; F1 −0.003) |

- **The two-stage idea and the candidate improvement replicate** on a new source and two new targets: matching lifts the rare types by 21 (BANC) and 10 (MCNS) points, `union` beats the old `gnn` candidates on both cords, and the pipeline closes 60–68 % of the gap to oracle candidates (development: 65–88 %; BANC brain, old pipeline: 44 %).
- **Expectation 2 fails on MCNS because stage 1 is already strong there** (72.0 % on the 2–4 stratum, against 19–54 % in the earlier tests). The oracle ceiling is 87.0 %, only 15.0 points above stage 1, so +15 would have required closing the whole gap to oracle candidates. The pipeline reached 82.2 % (0.68 of the gap).
- **The two-tier fallback does not transfer to the cord.** On both cords the min-size-10 stage 1 types the neurons it covers worse than the min-size-2 stage 1 does (10–19 bin: BANC 55.7 → 52.8 %, MCNS 76.1 → 72.5 %), so replacing the non-assigned neurons' types costs accuracy. The rare types are unchanged, as required. On the brain the opposite held. A likely reason (not tested): in MANC the min-size-10 model has no advantage within the source either (78.1 % on its own 438 types vs 70.9 % for the min-size-2 model on 3,901 types, a much smaller gap than FAFB's 94.2 vs 82.9 %).

## Primary result (run 3: `union` + passing block + two-tier m10, stage 1's variant)

| | BANC cord, ref A | BANC cord, ref B | MCNS cord, ref A |
|---|---|---|---|
| 2–4 stratum: stage 1 → final | 39.5 → **61.0 %** (n = 5,989) | 40.7 → **61.0 %** (n = 5,257) | 72.0 → **82.2 %** (n = 5,945) |
| 2–4 macro-F1 | 0.357 → 0.611 | 0.373 → 0.611 | 0.703 → 0.820 |
| 2–4, part sent to matching | 49.1 → 75.9 % (n = 4,799) | 50.6 → 76.0 % (n = 4,206) | 78.4 → 89.7 % |
| All labeled: stage 1 → final | 44.8 → 49.0 % (n = 22,451) | 45.1 → 48.7 % (n = 20,511) | 73.1 → 73.8 % |
| All labeled, without two-tier | 49.4 % | 49.2 % | 75.0 % |

"Stage 1" is the min-size-2 labeling after NTAC. Final = run 3 with the fallback, as defined in the protocol.

## All runs, reference A

| Run | BANC 2–4 acc / F1 | BANC all | MCNS 2–4 acc / F1 | MCNS all |
|---|---|---|---|---|
| Stage 1 (m2, after NTAC) | 39.5 % / 0.357 | 44.8 % | 72.0 % / 0.703 | 73.1 % |
| Stage 1 + two-tier | 39.5 % / 0.357 | 44.4 % | 72.0 % / 0.703 | 71.8 % |
| **3 `union` (primary)** | **61.0 % / 0.611** | 49.4 % | **82.2 % / 0.820** | 75.0 % |
| **3 `union` + two-tier (final)** | **61.0 % / 0.611** | 49.0 % | **82.2 % / 0.820** | 73.8 % |
| 4 `gnn` (old pipeline) | 52.1 % / 0.550 | 48.1 % | 77.1 % / 0.791 | 74.5 % |
| 4 `gnn` + two-tier | 52.1 % / 0.550 | 47.7 % | 77.1 % / 0.791 | 73.3 % |
| 5 oracle candidates (uses labels) | 75.5 % / 0.729 | 54.4 % | 87.0 % / 0.856 | 77.2 % |
| 6 `union`, variant `raw` | 56.1 % / 0.562 | 48.0 % | 81.9 % / 0.817 | 74.9 % |
| 6 `union`, variant `scaled` | 57.1 % / 0.573 | 48.3 % | 81.9 % / 0.816 | 74.9 % |
| 7 `union`, no passing block | 55.3 % / 0.547 | 47.8 % | 76.6 % / 0.755 | 73.4 % |

Runs 6 and 7 are shown without the fallback; with it, their 2–4 rows are the same and "all" is 0.3–1.2 points lower. BANC reference B gives the same picture (run 4: 51.7 %, run 5: 75.5 %, run 6: 55.9 / 56.9 %, run 7: 55.5 %).

- **Passing block (run 7):** without it the rare types lose 5.7 (BANC) and 5.6 (MCNS) points. The block is worth more on the cords than on the development sandbox (+2.1).
- **Graph variant (run 6):** stage 1 chose the density variant on both cords. On BANC the other variants would have cost 4–5 points; on MCNS 0.3.

## Reference-A 2–4 stratum by group (stage 1 → run 3)

| Group | BANC n | BANC | MCNS n | MCNS |
|---|---|---|---|---|
| intrinsic | 3,837 | 37.3 → 56.7 % | 3,827 | 70.5 → 80.0 % |
| ascending | 919 | 54.1 → 71.2 % | 939 | 76.6 → 84.1 % |
| descending | 891 | 41.4 → 74.2 % | 864 | 77.7 → 90.2 % |
| motor | 225 | 16.0 → 48.4 % | 203 | 56.2 → 85.7 % |
| sensory | 74 | 29.7 → 48.6 % | 71 | 83.1 → 85.9 % |
| other | 43 | 23.3 → 34.9 % | 41 | 46.3 → 48.8 % |
| passing | 1,840 | 47.3 → 71.9 % | 1,815 | 77.4 → 87.0 % |
| not passing | 4,149 | 36.1 → 56.1 % | 4,130 | 69.6 → 80.1 % |

Every group gains on both cords. Intrinsic neurons, whose references rest partly (BANC) or mainly (MCNS) on connectivity, gain no more than the others.

## Larger types (reference A; stage 1 → run 3 → run 3 + two-tier)

| Bin | BANC | MCNS |
|---|---|---|
| 5–9 | 54.5 → 52.5 → 52.5 % (n = 6,363) | 78.7 → 77.2 → 77.2 % (n = 6,340) |
| 10–19 | 55.7 → 53.2 → 50.7 % (n = 3,978) | 76.1 → 74.4 → 71.1 % (n = 3,904) |
| ≥ 20 | 32.7 → 32.2 → 32.6 % (n = 6,121) | 65.3 → 64.7 → 62.4 % (n = 5,213) |

Matching costs the 5–19 bins 1.5–2.5 points. A likely reason: candidates of other types receive a rare type (30.6 % of BANC's and 17.4 % of MCNS's `union` candidates carry a reference-A type that is not a 2–4 type).

## Stage 1

| | BANC m2 | BANC m10 | MCNS m2 | MCNS m10 |
|---|---|---|---|---|
| Chosen candidate (`size_agreement`) | density, target, ρ = 1 | density, target, ρ = 1 | density, target, ρ = 1 | raw, source, ρ = 1 |
| Without prior matching | 31.0 % | 26.5 % | 67.1 % | 58.4 % |
| Prior matched (chosen) / macro-F1 | 43.1 % / 0.441 | 45.0 % / 0.516 | 72.0 % / 0.723 | 70.4 % / 0.728 |
| Best candidate (post hoc) | 43.1 % (the chosen) | 45.0 % (the chosen) | 72.0 % (the chosen) | 72.3 % (density, target, ρ = 1) |
| After NTAC / macro-F1 | 44.8 % / 0.395 | 47.5 % / 0.542 | 73.1 % / 0.698 | 71.7 % / 0.755 |

Accuracy over all reference-A neurons of each model's label space. The choice made by each rule is in `report.json` (`choice_by_rule`). On BANC m2 and m10 and MCNS m2, `size_agreement` picked the best candidate. On MCNS m10 it missed by 1.9 points (`rank_mean` would have picked the best).

- **BANC reference B (m2):** without adaptation 32.0 %, chosen 43.4 %, after NTAC 45.1 %. A and B agree on 99.1 % of the neurons labeled in both.
- In three BANC m2 candidates (source normalization, high ρ), the computation overflowed (RuntimeWarnings in `lean_label_free.py`), and they collapsed to two types. Of the eight selection rules only `agreement` chose one of them (scaled, source, ρ = 1); `size_agreement` did not. Reported as observed; nothing was changed.

## Counts (run 3, `union`; run 4, `gnn`; run 5, oracle)

| | BANC `union` | BANC `gnn` | BANC oracle | MCNS `union` | MCNS `gnn` | MCNS oracle |
|---|---|---|---|---|---|---|
| Candidates | 8,504 | 6,364 | 5,989 | 7,594 | 6,606 | 5,945 |
| … per block (left / right / passing) | 3,067 / 3,083 / 2,188 | 2,227 / 2,215 / 1,817 | 2,044 / 2,089 / 1,840 | 2,704 / 2,678 / 2,069 | 2,283 / 2,301 / 1,905 | 2,007 / 1,995 / 1,815 |
| … without a side | 166 | 105 | 16 | 143 | 117 | 128 |
| Recall of reference-A rare neurons | 80.4 % | 67.3 % | 100 % | 92.1 % | 86.8 % | 100 % |
| Share rare-type / other-type / unlabeled | 56.6 / 30.6 / 12.8 % | 63.4 / 24.2 / 12.5 % | 100 / 0 / 0 % | 72.1 / 17.4 / 10.5 % | 78.1 / 12.0 / 9.9 % | 100 / 0 / 0 % |
| Candidates on padding | 2,176 | 177 | 0 | 1,289 | 327 | 0 |
| MANC rows unmatched | 0 | 80 | 189 | 0 | 0 | 345 |

MANC rare-type rows per block: 2,133 left, 2,132 right, 1,897 passing. Unlabeled candidates also receive types (BANC: about 1,090, MCNS: about 800). They are counted but not scored.

## Runtime and memory

| Run | BANC cord | MCNS cord |
|---|---|---|
| 1. Stage 1, m2 | 15.3 min, 1.96 GB | 14.1 min, 2.13 GB |
| 2. Stage 1, m10 | 3.9 min, 1.80 GB | 4.1 min, 1.46 GB |
| 3. `union` | 2.7 min, 1.23 GB | 2.0 min, 1.08 GB |
| 4. `gnn` | 1.8 min, 0.96 GB | 1.7 min, 0.95 GB |
| 5. oracle | 1.7 min, 0.92 GB | 1.2 min, 0.89 GB |
| 6. `raw` / `scaled` | 2.6 / 2.6 min, 1.21 / 1.20 GB | 1.4 / 1.9 min, 1.06 / 1.06 GB |
| 7. no passing block | 3.0 min, 1.41 GB | 2.3 min, 1.24 GB |

## MANC-internal results, for scale

The GNNs typed MANC itself at 70.9 % (min size 2, random split, mean of 3 seeds) and 78.1 % (min size 10), and left → right at 63.2 % and 68.3 % (FAFB: 82.9, 94.2, 85.1, 94.0 %). On rare (2–4) types: 64.4 % random, 69.6 % left → right. Stage 1 on the MCNS cord (73.1 % over all reference-A neurons) is above MANC's own random-split accuracy. This is possible because the transfer models are trained on all MANC neurons, and the MCNS cord is a second male animal.

## Deviations and additions (none to the method)

- **Run 6 output folders:** the variant is not part of the output file names, so the two variant runs were written to `results/graph_matching_<target>_<variant>/` instead of the run-3 folder, so that run 3's files would not be overwritten.
- **`experiments/b_gnn_celltypes/vnc_report.py`** was written after the commit to collect the protocol's tables. It reads the saved outputs only.
- **T4** (the optional MANC-internal two-stage) was not run, by decision before the commit.
- No data-loading or memory fixes were needed. Every run exited normally.
