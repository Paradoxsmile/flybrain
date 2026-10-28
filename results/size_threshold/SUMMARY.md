# Cell-type accuracy vs. minimum type size

Scripts: `experiments/b_gnn_celltypes/size_threshold.py` (the sweep), `lean_training.py` and
`lean_label_free.py` (memory-lean training and label-free mapping for thresholds below 5).
`size_threshold.py` has since been committed by another session; the two lean files are new and
uncommitted. No change to `flybrain/loading.py` was needed, because
`load_graph(min_class_size=...)` already makes the threshold configurable. The 20 was hard-coded
only in `configs/cell_type.yaml` and in `label_free.LABELS`, and the script sets both per threshold.
The full tables, regenerated from every finished run, are in `table.md`. Per-stratum and per-type
numbers are in `strata.csv` and `types.csv`.

## Status: every threshold has been run

| min size | FAFB types (labeled neurons) | FAFB random (3 seeds) | FAFB left→right (3 seeds) | FAFB → MCNS label-free |
|---|---|---|---|---|
| 20 | 434 (108,647; 78 %) | reused `transfer2` | CPU + Kaggle | reused `label_free` |
| 10 | 779 (112,996; 81 %) | Kaggle T4, 54 s/run | Kaggle | 58 min CPU |
| 5 | 2,164 (121,799; 87 %) | Kaggle T4, 112 s/run | Kaggle | 145 min CPU |
| 3 | 4,175 (129,247; 93 %) | Kaggle T4, lean, 115 s/run | Kaggle, lean | 88 min CPU, lean |
| 2 | 8,658 (138,213; 99 %) | Kaggle T4, lean, 225 s/run | Kaggle, lean | 161 min CPU, lean |

How the lean code was checked:
- **`lean_training.py`** computes the output layer in row chunks with checkpointing. At ≥ 20 it is
  identical to `fit_gnn`: same losses and predictions, parameters within 1e-5. Peak RAM at ≥ 2 is
  2.5 GB instead of more than 20 GB.
- **`lean_label_free.py`** computes probabilities lazily in chunks and runs prior matching as a
  chunked float64 Sinkhorn. At ≥ 10 it reproduces the standard run: same choice, identical
  accuracies before NTAC, and 93.02 vs 93.04 % after NTAC.

Logs: `label_free_run.log` (20, 10, 5), `label_free_lean_run.log` (3, 2), `lr20_run.log`. Kaggle steps
are in `KAGGLE.md`.

In the random split, types with 2 neurons go entirely to training, so the ≥ 2 random row tests the
same 4,175 types as ≥ 3. Only the left→right split tests 2-neuron types. Those are typically one
neuron per hemisphere, so the model has seen a single example.

## Within FAFB (accuracy % / macro-F1, mean over 3 seeds)

By region of the type (central-brain type counts in brackets):

| split | min size | types | all | optic-lobe | visual proj. | central brain | sensory | other |
|---|---|---|---|---|---|---|---|---|
| random | 20 | 434 | 95.1 ± 0.1 / 0.93 | 96.1 / 0.96 | 97.1 / 0.95 | 94.2 / 0.92 (93) | 90.5 / 0.88 | 86.2 / 0.82 |
| random | 10 | 779 | 94.2 ± 0.5 / 0.90 | 95.7 / 0.95 | 96.1 / 0.93 | 91.7 / 0.88 (332) | 88.6 / 0.85 | 81.7 / 0.76 |
| random | 5 | 2,164 | 90.9 ± 0.8 / 0.82 | 93.0 / 0.93 | 95.9 / 0.94 | 86.2 / 0.81 (1,567) | 86.5 / 0.83 | 73.7 / 0.63 |
| random | 3 | 4,175 | 87.5 ± 0.4 / 0.72 | 91.7 / 0.91 | 93.7 / 0.90 | 78.7 / 0.71 (3,382) | 85.8 / 0.74 | 65.5 / 0.56 |
| random | 2 | (4,175 tested) | 82.9 ± 0.9 / 0.68 | 88.5 / 0.89 | 91.8 / 0.89 | 73.5 / 0.67 (3,382) | 74.5 / 0.69 | 59.2 / 0.53 |
| left→right | 20 | 428 | 94.3 ± 0.7 / 0.92 | 95.1 / 0.95 | 96.6 / 0.95 | 91.5 / 0.89 (93) | 91.6 / 0.87 | 72.6 / 0.71 |
| left→right | 10 | 766 | 93.1 ± 1.1 / 0.87 | 94.6 / 0.94 | 95.6 / 0.93 | 88.6 / 0.86 (331) | 89.2 / 0.86 | 73.6 / 0.67 |
| left→right | 5 | 2,144 | 89.1 ± 0.9 / 0.77 | 91.8 / 0.92 | 93.5 / 0.90 | 81.9 / 0.76 (1,565) | 85.1 / 0.80 | 60.0 / 0.53 |
| left→right | 3 | 4,141 | 87.2 ± 0.8 / 0.72 | 91.5 / 0.90 | 92.1 / 0.89 | 77.1 / 0.71 (3,375) | 84.5 / 0.75 | 56.5 / 0.54 |
| left→right | 2 | 8,582 | 83.8 ± 1.1 / 0.71 | 89.6 / 0.88 | 89.9 / 0.86 | 74.6 / 0.72 (6,668) | 77.9 / 0.70 | 59.5 / 0.60 |

By FAFB type size:

| split | min size | 2–4 | 5–9 | 10–19 | 20–49 | 50–199 | 200–999 | ≥ 1000 |
|---|---|---|---|---|---|---|---|---|
| random | 20 | – | – | – | 93.7 / 0.91 | 94.8 / 0.94 | 93.6 / 0.94 | 95.9 / 0.97 |
| random | 10 | – | – | 89.7 / 0.87 | 90.7 / 0.89 | 94.1 / 0.94 | 93.1 / 0.93 | 95.3 / 0.96 |
| random | 5 | – | 83.9 / 0.79 | 83.3 / 0.81 | 89.4 / 0.87 | 92.8 / 0.93 | 91.1 / 0.91 | 92.2 / 0.93 |
| random | 3 | 72.6 / 0.68 | 78.5 / 0.71 | 79.4 / 0.76 | 84.6 / 0.82 | 90.0 / 0.89 | 88.0 / 0.89 | 92.1 / 0.93 |
| random | 2 | 67.5 / 0.64 | 75.0 / 0.68 | 74.0 / 0.71 | 81.0 / 0.78 | 87.5 / 0.87 | 85.8 / 0.85 | 86.1 / 0.89 |
| left→right | 20 | – | – | – | 90.7 / 0.89 | 92.8 / 0.94 | 93.1 / 0.93 | 95.4 / 0.95 |
| left→right | 10 | – | – | 86.6 / 0.84 | 87.5 / 0.86 | 91.8 / 0.93 | 92.5 / 0.92 | 94.6 / 0.94 |
| left→right | 5 | – | 76.4 / 0.74 | 80.4 / 0.78 | 84.7 / 0.83 | 89.5 / 0.91 | 90.2 / 0.91 | 91.3 / 0.91 |
| left→right | 3 | 76.8 / 0.71 (1,997 t) | 70.2 / 0.69 | 75.5 / 0.74 | 81.1 / 0.80 | 87.7 / 0.89 | 87.3 / 0.88 | 92.0 / 0.91 |
| left→right | 2 | 75.9 / 0.72 (6,438 t) | 66.2 / 0.65 | 70.3 / 0.69 | 76.3 / 0.75 | 84.6 / 0.85 | 85.6 / 0.87 | 89.0 / 0.89 |

## FAFB → MCNS, label-free (accuracy % / macro-F1 on labeled MCNS neurons, ensemble of 3 seeds)

At ≥ 10, 5, 3 and 2 the selection chose raw alignment, source normalization and rho 0.75, which
was also the best candidate in hindsight. At ≥ 20 (the paper's run) it chose rho 0.5: 91.2 %
before NTAC, against 91.4 % for the best candidate.

| min size | MCNS types / neurons | no adaptation | prior matched | + NTAC |
|---|---|---|---|---|
| 20 | 394 / 109,901 | 87.6 / 0.73 | 91.2 / 0.79 | 92.6 / 0.81 |
| 10 | 649 / 112,997 | 85.7 / 0.64 | 91.1 / 0.78 | 93.0 / 0.80 |
| 5 | 1,694 / 119,614 | 78.9 / 0.50 | 87.5 / 0.70 | 90.3 / 0.71 |
| 3 | 3,206 / 125,360 | 74.8 / 0.40 | 84.2 / 0.62 | 87.6 / 0.57 |
| 2 | 7,064 / 133,356 | 66.9 / 0.33 | 80.1 / 0.59 | 84.2 / 0.51 |

After NTAC, by region of the type:

| min size | optic-lobe | visual proj. | central brain | sensory | other |
|---|---|---|---|---|---|
| 20 | 97.2 / 0.96 | 95.7 / 0.96 | 65.1 / 0.80 (71 t, 6,231 n) | 59.1 / 0.26 | 95.4 / 0.73 (5 t) |
| 10 | 96.9 / 0.97 | 97.5 / 0.96 | 77.7 / 0.79 (243 t, 8,197 n) | 63.4 / 0.39 | 66.0 / 0.49 (34 t) |
| 5 | 95.5 / 0.94 | 96.5 / 0.95 | 72.3 / 0.70 (1,181 t, 14,063 n) | 63.5 / 0.43 | 48.8 / 0.36 (96 t) |
| 3 | 95.3 / 0.94 | 94.6 / 0.88 | 62.5 / 0.54 (2,542 t, 19,257 n) | 62.6 / 0.41 | 43.6 / 0.31 (192 t) |
| 2 | 93.1 / 0.92 | 93.7 / 0.85 | 64.3 / 0.54 (5,371 t, 25,157 n) | 59.3 / 0.40 | 31.6 / 0.22 (1,027 t) |

After NTAC, by FAFB type size:

| min size | 2–4 | 5–9 | 10–19 | 20–49 | 50–199 | 200–999 | ≥ 1000 |
|---|---|---|---|---|---|---|---|
| 20 | – | – | – | 77.0 / 0.74 | 88.2 / 0.83 | 92.4 / 0.88 | 95.1 / 0.93 |
| 10 | – | – | 75.5 / 0.74 (255 t) | 80.4 / 0.76 | 90.8 / 0.87 | 93.7 / 0.90 | 95.2 / 0.93 |
| 5 | – | 71.6 / 0.68 (1,045 t) | 68.6 / 0.66 | 79.6 / 0.75 | 90.8 / 0.87 | 90.4 / 0.88 | 94.0 / 0.92 |
| 3 | 54.2 / 0.48 (1,512 t) | 66.5 / 0.59 | 63.6 / 0.59 | 78.3 / 0.72 | 89.8 / 0.85 | 89.3 / 0.87 | 93.5 / 0.91 |
| 2 | 54.2 / 0.49 (5,370 t, 13,742 n) | 58.9 / 0.51 | 61.8 / 0.57 | 73.7 / 0.68 | 88.2 / 0.83 | 88.8 / 0.85 | 92.6 / 0.91 |

The overall rows cover different neuron sets, because lower thresholds label more MCNS neurons. The
size bins compare the same types across thresholds. Macro-F1 here is stratify.py's: the mean over the
true types present. label_free.py's own macro-F1, which also counts predicted types that are absent
from the target, is 0.75 / 0.65 / 0.52 / 0.49 at ≥ 10 / 5 / 3 / 2 (`selected.json`).

## Findings

- **Wiring identifies even the rarest types.** Within FAFB across hemispheres, a model that has seen
  only 1–2 neurons of a type on the left types its right-hemisphere neurons at 75.9 % (macro-F1 0.72,
  6,438 types of 2–4 neurons, at ≥ 2). Chance is about 1 / 8,582.
- **Rare types transfer label-free, but the gap widens below 10 neurons.**
  - FAFB → MCNS without any MCNS label: 10–19-neuron types reach 75.5 % (at ≥ 10), 5–9-neuron types
    71.6 % (at ≥ 5), and 2–4-neuron types 54.2 % (5,370 types, 13,742 neurons, at ≥ 2).
  - Against within-FAFB left→right, the gap is about 11 points at 10–19, 5 at 5–9 and 22 at 2–4.
  - Matching rare types across animals is therefore feasible: about half of the 2–4-neuron types
    are recovered. This is the main open margin.
- **≥ 10 is a free improvement over ≥ 20. Below that, every threshold trades common-type accuracy for
  coverage.**
  - At ≥ 10, every size bin of 20 or more neurons is equal or better than at ≥ 20.
  - At ≥ 2, the 20–49 bin falls from 77.0 to 73.7 % and the ≥ 1000 bin from 95.1 to 92.6 %.
  - Within FAFB the cost is larger: ≥ 1000 falls from 95.4 to 89.0 % (left→right).
  - A two-tier model (a ≥ 10 classifier for common types, a separate stage for rare ones) may avoid
    the trade-off.
- **Central brain and "other" carry the rare types and their errors.**
  - Central-brain accuracy after transfer peaks at 77.7 % (243 types, ≥ 10).
  - With thousands of rare central types it settles at 62–64 % (5,371 types at ≥ 2).
  - "Other" (ascending, descending etc.) falls to 31.6 % at ≥ 2.
  - Optic-lobe and visual-projection types stay at 93–97 % at every threshold.
- **The pipeline steps behave differently at large K.**
  - Prior matching adds more as K grows: +13.2 points at ≥ 2, against +3.6 at ≥ 20.
  - NTAC still raises accuracy (+3–4 points) but lowers macro-F1 at ≥ 3 and ≥ 2 (0.62 → 0.57,
    0.59 → 0.51). It propagates common types into rare ones, so for rare-type work the prior-matched
    labeling, or a rare-type-aware refinement, is the better endpoint.
