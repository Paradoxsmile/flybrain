# Label-free accuracy estimation and error flagging

Script: `experiments/b_gnn_celltypes/estimate_accuracy.py` (analysis only; the neuron-level part
re-applies the saved GNNs of the chosen candidate, 3 seeds, with no training, cached in
`neurons_<run>.parquet` and checked against `stages.parquet` / `selected.json`). All tables:
`tables.md`. Raw: `candidates.csv` (1,200 candidates: 17 runs per label, 18 run directories),
`run_level.csv`, `types.csv`, `flagging.csv`.

**Bottom line:** No. A few-parameter run-level predictor does **not** estimate accuracy to within
±3–5 points on held-out connectomes in general. It gets within ±5 points only on full-brain
targets whose labeling has not collapsed, and it fails silently (errors of 40–50 points) when all
seeds make the same mistake. Neuron-level flagging is the more solid result. On FAFB↔MCNS,
reviewing the 20 % least-agreed neurons catches 55–77 % of the errors (random review: 20 %). On
FAFB→BANC the ranking is better than random but well below the oracle. On the hemibrain the
labeling is mostly wrong, so flagging is no better than random.

## 1. Run level

Candidates are grouped by target connectome: **MCNS** (FAFB→MCNS plus the sandbox: thinned to
0.5 and 0.2, type-excluded, prior bounded, cropped central-brain copy with three priors and
coverage normalization; 9 runs), **FAFB** (MCNS→FAFB, full and thinned to 0.35), **BANC**
(FAFB→BANC, MCNS→BANC, augmented-model and thinned-source variants), and **hemibrain**
(coverage and source prior). Each label gets its own linear fit, with every run weighted equally.
`agree_js` (accuracy from pairwise seed agreement and size_js) was fixed as the main predictor
before looking at held-out errors. `all` adds consistency and top_share / ref_top_share.
Baselines: `mean` is the mean training accuracy; `agree0` takes accuracy to be the agreement
itself.

Absolute error of the predicted accuracy of each run's chosen candidate, in points, as
mean / worst (in brackets: runs within 5 points):

| fit → test | label | runs | mean | agree0 | agree | **agree_js** | all |
|---|---|---|---|---|---|---|---|
| dev (MCNS+FAFB) → BANC | cell type | 4 | 23.9 / 37.5 (0) | 31.4 / 72.9 (0) | 20.0 / 70.5 (3) | **12.9 / 43.1 (2)** | 4.8 / 12.7 (3) |
| dev → hemibrain | cell type | 2 | 21.5 / 23.1 (0) | 30.0 / 30.4 (0) | 6.7 / 7.4 (0) | **12.9 / 13.2 (0)** | 6.0 / 7.2 (1) |
| dev → BANC | super class | 4 | 6.1 / 12.3 (1) | 12.9 / 13.6 (0) | 10.5 / 19.6 (1) | **6.8 / 14.7 (2)** | 1.8 / 2.7 (4) |
| dev → hemibrain | super class | 2 | 19.8 / 37.4 (1) | 40.1 / 59.6 (0) | 23.8 / 43.0 (1) | **24.6 / 46.2 (1)** | 26.7 / 48.7 (1) |
| LOCO → MCNS | cell type | 9 | 35.8 / 57.5 (0) | 27.6 / 65.1 (3) | 19.4 / 57.3 (1) | **16.0 / 43.9 (1)** | 10.9 / 40.2 (3) |
| LOCO → FAFB | cell type | 2 | 41.5 / 50.9 (0) | 3.1 / 3.5 (2) | 8.4 / 8.7 (0) | **8.9 / 9.7 (0)** | 6.7 / 12.7 (1) |
| LOCO → BANC | cell type | 4 | 23.9 / 33.7 (0) | 31.4 / 72.9 (0) | 19.9 / 70.6 (3) | **13.4 / 46.2 (2)** | 4.1 / 10.1 (3) |
| LOCO → hemibrain | cell type | 2 | 19.1 / 20.8 (0) | 30.0 / 30.4 (0) | 9.2 / 9.6 (0) | **16.3 / 16.4 (0)** | 6.5 / 7.5 (0) |

LOCO rows for super class, plus per-run errors, are in `tables.md`. Pooled over the held-out runs:

| split | mean | agree0 | agree | **agree_js** | all |
|---|---|---|---|---|---|
| dev → BANC + hemibrain (12 runs), mean / worst | 16.9 / 37.5 | 26.4 / 72.9 | 15.2 / 70.5 | **12.8 / 46.2** | 7.6 / 48.7 |
| median; within 3 / 5 points | 14.5; 2 / 2 | 17.1; 0 / 0 | 5.7; 2 / 5 | **8.4; 4 / 5** | 3.6; 6 / 9 |
| leave one out (34 runs), mean / worst | 23.8 / 57.5 | 22.4 / 72.9 | 15.1 / 70.6 | **13.1 / 46.7** | 9.1 / 49.5 |
| mean over all candidates of a run | 14.3 | 21.0 | 9.0 | **7.3** | 6.2 |

Error at the candidate the current size_agreement rule chooses (dev → held-out): agree_js
8.4 / 46.2, all 7.8 / 48.7. Coefficients fitted on dev, cell type: acc = −0.42 + 1.55 · agreement
− 0.32 · size_js.

Per-run errors from dev → held-out (agree_js / all, points):

| run | label | true | agree_js | all |
|---|---|---|---|---|
| FAFB → BANC | cell type | 55.7 | +0.5 | −0.6 |
| FAFB → BANC, source thinned 0.4 | cell type | 48.6 | +0.8 | +4.4 |
| FAFB → BANC, aug models (collapsed choice) | cell type | 7.3 | −7.3 | +12.7 |
| MCNS → BANC (collapsed choice) | cell type | 1.4 | **+43.1** | −1.4 |
| FAFB → hemibrain | cell type | 19.1 | +13.2 | +7.2 |
| FAFB → BANC | super class | 72.3 | −9.5 | −1.4 |
| MCNS → BANC | super class | 78.7 | −1.9 | +2.7 |
| FAFB → hemibrain | super class | 74.3 | +3.1 | −4.7 |
| FAFB → hemibrain, source prior | super class | 34.7 | **+46.2** | +48.7 |

## 2. Type level (cell types, + NTAC labeling)

For each predicted type: the mean share of seeds agreeing with the final label over all neurons
given that type (label-free), compared with the type's F1 on the labeled neurons. Types with
≥ 10 neurons.

| transfer | types | Spearman ρ | mean F1, least / most agreed quarter |
|---|---|---|---|
| FAFB → MCNS | 389 | **0.63** | 0.48 / 0.99 |
| MCNS → FAFB | 378 | **0.66** | 0.53 / 0.96 |
| FAFB → BANC | 386 | **0.54** | 0.22 / 0.68 |
| MCNS → BANC (collapsed) | 18 | −0.02 | 0.03 / 0.04 |
| FAFB → hemibrain | 197 | 0.25 | 0.07 / 0.31 |

## 3. Neuron level: errors caught by reviewing the least-confident neurons

Confidence is the share of seeds that agree with the final label, with ties broken by the
ensemble probability. Ranking by ensemble probability alone gives the same results within
±2 points. The ranking covers the labeled neurons. Cells show % of all errors caught
(precision in brackets). Random review catches 10 / 20 / 30 %.

| transfer | labeled | error % | 10 % | 20 % | 30 % | oracle 10 / 20 / 30 % |
|---|---|---|---|---|---|---|
| FAFB → MCNS | 109,901 | 7.4 | **52** (38) | **77** (29) | **87** (21) | 100 / 100 / 100 |
| MCNS → FAFB | 108,647 | 16.6 | 35 (57) | 55 (45) | 72 (40) | 60 / 100 / 100 |
| FAFB → BANC | 63,332 | 38.3 | 20 (75) | 34 (66) | 47 (60) | 26 / 52 / 78 |
| MCNS → BANC | 63,332 | 98.6 | 10 | 20 | 30 | 10 / 20 / 30 |
| FAFB → hemibrain | 7,639 | 80.4 | 12 (99) | 23 (94) | 35 (93) | 12 / 25 / 37 |

The prior-matched labeling, before NTAC, gives similar numbers (FAFB → MCNS: 60 / 81 / 89 %;
see `tables.md`).

## Findings

1. **No ±3–5-point accuracy estimate on held-out connectomes.** The pre-specified `agree_js`
   predictor, fitted on MCNS + FAFB, misses BANC and hemibrain by 12.8 points on average (median
   8.4). Only 5 of 12 held-out runs land within 5 points, and 4 of 12 within 3. The 5-parameter
   `all` model does better (median 3.6, 9 of 12 within 5 points), but its worst case is no
   better (48.7). Leave-one-out gives the same picture (13.1 and 9.1 points mean error).
2. **When the estimate works, and when it fails.** On full-brain targets with a non-collapsed
   labeling (FAFB → BANC cell type +0.5, thinned source +0.8, MCNS → BANC super class −1.9), the
   estimate is within ±2 points. It fails in two ways:
   - *Collapse.* MCNS → BANC predicts one class for most neurons, so the seeds agree on a
     labeling that is 1.4 % correct. The agreement-based estimate says 44.5 %. The top_share
     column picks this up: the `all` model's error falls to −1.4. The coefficient is fitted on a
     single collapsed development run, though, so this fix is fragile.
   - *Shared systematic error in partial volumes.* With the source prior, the hemibrain super
     classes reach 0.94 agreement and a near-perfect size match, yet are 34.7 % correct. The
     cropped MCNS copy fails the same way (32.8 %). Every label-free column looks healthy, so no
     predictor built on them can catch this.
3. **Seed agreement ranks types and neurons well wherever the transfer basically works.** Per
   type, Spearman ρ is 0.54–0.66 on FAFB↔MCNS and FAFB → BANC: the least-agreed quarter of types
   has a mean F1 of 0.22–0.53, the most-agreed quarter 0.68–0.99. Per neuron, on FAFB → MCNS,
   **reviewing 20 % of neurons catches 77 % of errors (30 %: 87 %)**. MCNS → FAFB: 55 % and 72 %.
   FAFB → BANC: 34 % and 47 %, compared with an oracle at 52 % and 78 %. That is about 1.7×
   random, with precision of 60–75 % among the reviewed. On the hemibrain (80 % errors) and on
   collapsed MCNS → BANC, flagging is essentially random: when most labels are wrong, ranking
   cannot help.
4. **A zero-parameter per-neuron estimate tracks accuracy, but only on labeled neurons.** The
   share of labeled neurons where all three seeds agree with the final label is 90.2 % on MCNS
   (true accuracy 92.6 %), 82.6 % on FAFB (83.4 %) and 60.5 % on BANC (61.7 %). That is within
   2.4 points on all three full-brain targets, but it is not fully label-free: which neurons are
   labeled (their type is in the shared vocabulary) comes from the target annotation. Over all
   neurons the same share is 78.6 %, 71.6 % and 52.4 %, i.e. 7–14 points too low. It fails on
   collapse (72.1 % vs 1.4 %) and on the hemibrain (35.5 % vs 19.6 %).
5. **The fitted predictor helps with selection, but no more than the existing rule.** Choosing
   the candidate with the highest `agree_js` prediction leaves a gap to the best candidate of
   4.3 / 9.1 points (mean / worst, cell type, dev → held-out). size_agreement gives 2.6 / 9.1,
   and the runs' own recorded choices gave 20.4 / 56.6. For super class all are about 8 / 38,
   with the worst case being the hemibrain source prior. The fitted predictor adds no selection
   power beyond size_agreement.

**What this supports for the paper:** a practical-impact claim, stated per target ("on
FAFB → MCNS, reviewing 20 % of neurons catches 77 % of errors; on BANC 34 %"). It does not
support a claim of a validated label-free accuracy estimate. The honest framing is that
label-free signals detect *disagreement* and *collapse*, not shared systematic errors, which are
exactly the failure mode of partial-volume targets.

Caveats: these are few runs (12 held-out, from 2 connectomes) and they are not independent
(variants share models and targets). The sandbox runs dominate the MCNS group. Hemibrain
accuracy is low partly because of vocabulary mismatch, which seed agreement cannot see.
