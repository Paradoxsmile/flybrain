# Labeling effort: how many hand labels does the label-free GNN save?

Script: `experiments/b_gnn_celltypes/label_effort.py` (no retraining). Full tables, including the
per-region breakdown: `tables.md`; per-run rows: `metrics.csv` (overall) and `strata.csv` (by region);
figure: `label_effort.png` / `.pdf`.

## Setup

- Targets: FAFB → MCNS (`results/label_free`), and FAFB → BANC (`results/label_free_fafb_banc`)
  scored against reference A (BANC typing, 63,332 neurons) and reference B (morphology match,
  central brain, 11,656 neurons). For reference B the hand labels also come from reference B.
- Hand labels: k ∈ {1, 2, 5, 10, 20, 50} % of each type's reference neurons, drawn at random
  (round(k % × type size) per type), 3 draws. Scored on the reference neurons without a hand label.
- Labelings: **NTAC** seeded with the hand labels only; **GNN + NTAC** seeded with the hand labels
  plus the unanimous predictions of the 3 label-free GNNs, with hand labels taking precedence (as in
  `hybrid.py`); **NTAC + GNN fallback** (added at the parent session's request, no extra run): the NTAC
  labeling, except where the label-free pipeline predicts a type that has no hand label (NTAC cannot
  output such a type), in which case the pipeline's label is kept; flat lines for the **GNN**
  ensemble and the **GNN pipeline + NTAC** without hand labels.
- NTAC: 15 steps, learning rate 0.3, on the target graph of the chosen candidate (MCNS `raw`, BANC
  `density`). Check: GNN + NTAC at k = 0 reproduces the saved pipeline labeling neuron for neuron
  (MCNS 92.6 %, BANC-A 61.7 %).

## Overall accuracy % / macro-F1 on unlabeled neurons (mean of 3 draws)

| target | method | k=1 | 2 | 5 | 10 | 20 | 50 |
|---|---|---|---|---|---|---|---|
| MCNS (pipeline 92.6 / 0.81) | NTAC | 91.7 / 0.51 | 95.5 / 0.74 | 97.2 / 0.87 | 97.7 / 0.90 | 98.1 / 0.93 | 98.5 / 0.95 |
| | GNN + NTAC | 92.7 / 0.81 | 92.9 / 0.82 | 93.2 / 0.83 | 93.6 / 0.84 | 94.1 / 0.86 | 95.1 / 0.89 |
| | NTAC + GNN fallback | **94.1 / 0.81** | 95.3 / 0.83 | 96.7 / 0.87 | 97.2 / 0.90 | 97.5 / 0.92 | 97.9 / 0.94 |
| BANC ref A (pipeline 61.7 / 0.42) | NTAC | 58.5 / 0.17 | 68.4 / 0.30 | 73.9 / 0.45 | 76.0 / 0.51 | 78.2 / 0.55 | 80.0 / 0.59 |
| | GNN + NTAC | 62.1 / 0.43 | 62.5 / 0.44 | 63.5 / 0.46 | 65.0 / 0.48 | 67.1 / 0.51 | 70.3 / 0.55 |
| | NTAC + GNN fallback | 57.7 / 0.37 | 64.3 / 0.38 | 68.7 / 0.42 | 70.6 / 0.47 | 72.4 / 0.50 | 73.9 / 0.53 |
| BANC ref B (pipeline 32.3 / 0.27) | NTAC | 25.8 / 0.04 | 44.9 / 0.25 | 51.0 / 0.42 | 52.6 / 0.45 | 56.0 / 0.50 | 61.2 / 0.56 |
| | GNN + NTAC | 32.8 / 0.27 | 34.7 / 0.31 | 37.1 / 0.35 | 40.5 / 0.40 | 45.0 / 0.45 | 52.1 / 0.51 |
| | NTAC + GNN fallback | 25.9 / 0.25 | 33.6 / 0.28 | 40.2 / 0.34 | 41.5 / 0.36 | 43.9 / 0.40 | 47.5 / 0.44 |

## Headline: hand labels plain NTAC needs to match GNN + NTAC

k\* = % of each type that NTAC alone needs (log-linear interpolation) to reach GNN + NTAC at k.

| target | k | GNN + NTAC | hand labels at k | k\* by accuracy (labels) | k\* by macro-F1 (labels) |
|---|---|---|---|---|---|
| MCNS | 1 | 92.7 / 0.81 | 1,075 | 1.2 (~1,330) | 3.4 (~3,720) |
| MCNS | 2 | 92.9 / 0.82 | 2,202 | 1.3 (~1,380) | 3.6 (~3,950) |
| MCNS | 5 | 93.2 / 0.83 | 5,502 | 1.3 (~1,460) | 4.0 (~4,360) |
| BANC ref A | 1 | 62.1 / 0.43 | 591 | 1.3 (~820) | 4.4 (~2,790) |
| BANC ref A | 2 | 62.5 / 0.44 | 1,255 | 1.3 (~840) | 4.8 (~3,020) |
| BANC ref A | 5 | 63.5 / 0.46 | 3,163 | 1.4 (~900) | 5.7 (~3,630) |
| BANC ref B | 1 | 32.8 / 0.27 | 91 | 1.3 (~150) | 2.3 (~260) |
| BANC ref B | 2 | 34.7 / 0.31 | 229 | 1.4 (~160) | 2.7 (~320) |
| BANC ref B | 5 | 37.1 / 0.35 | 580 | 1.5 (~180) | 3.5 (~410) |

## Findings

1. **By accuracy, GNN seeding saves hand labels only below about 1.3 % hand labeling.** On all three
   references, plain NTAC with 1.2–1.5 % of each type labeled matches GNN + NTAC at k = 1, 2 or 5 %.
   At k = 2–5 %, GNN seeding *costs* labels: for example, on MCNS, GNN + NTAC at 5 % (5,502
   labels, 93.2 %) is matched by NTAC with about 1,460 labels, and NTAC at 5 % reaches 97.2 %. At
   k = 1 %, GNN seeding is ahead by 1.0 (MCNS), 3.6 (BANC-A) and 7.0 (BANC-B) points.
2. **The saving is real for macro-F1, i.e. for small types.** With random labeling, types under 50
   neurons get no hand label at 1 %, so NTAC cannot assign them at all (NTAC-alone macro-F1 at 1 %:
   0.51 / 0.17 / 0.04). GNN + NTAC at 1 % matches the macro-F1 of NTAC with 3.4 % (MCNS), 4.4 %
   (BANC-A) and 2.3 % (BANC-B) of neurons labeled, which is 2.3–4.4× as many hand labels. The
   advantage shrinks with k and is gone by 5–10 %.
3. **Why GNN + NTAC falls behind: the GNN seeds are frozen.** The unanimous GNN predictions (79 %
   of MCNS neurons, 52 % of BANC) are fixed seeds that NTAC never revises, and each hand label
   corrects only its own neuron. The biggest gap is in the MCNS central brain: from 1 to 50 % hand
   labels, GNN + NTAC goes from 65.8 % to 75.0 % there, while NTAC alone reaches 85.0 % already at
   2 % and 96.6 % at 50 %. On the optic lobe both are at ceiling (97–99 %).
4. **The GNN fallback is best on MCNS and harmful on BANC.** Here NTAC labels everything it can,
   and the pipeline's label is kept only for types that have no hand label. On MCNS at 1 % this
   gives the best of both: 94.1 % with macro-F1 0.81, better than either at that budget (central
   brain 88.1 % vs 69.9 / 65.8). On BANC the transferred labels are too unreliable: the fallback
   sits below NTAC alone in accuracy from 2 % on (sensory types especially), and below GNN + NTAC
   in macro-F1. How much the GNN can contribute follows its transfer accuracy.
5. **What this means in practice:** GNN-seeded NTAC is useful with no labels at all, or with a very
   small budget (≲ 1 % per type) when every type should be covered, which is when random labeling
   misses small types. As soon as a team can label 2 % of each type, plain NTAC, or NTAC with the GNN
   used only as a fallback for types it has not seen, is the better use of the GNN. Freezing GNN
   seeds is the wrong way to combine them once hand labels exist.

## Caveats

- Hand labels are random within each type, with per-type rounding. A real team would label at
  least one neuron of each type, which would raise NTAC's macro-F1 at 1–2 % and shrink the GNN's
  advantage from finding 2.
- The flat lines are scored on all reference neurons, while the curves are scored on the neurons
  without a hand label (a random subset at each k).
- NTAC runs with the package defaults on the graph of the pipeline's chosen candidate. MCNS is a
  single target and BANC is a single run, so there are no confidence intervals beyond the 3 draws
  (spread ≤ 1.5 points on MCNS and BANC-A, up to 3.5 points on BANC-B).
- Not tested: the GNN labels as NTAC's *initialization* or as revisable seeds, with only the hand
  labels fixed. That needs a small change to `flybrain/ntac.py` and about 3 h of NTAC runs.
