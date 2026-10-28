# 95 % intervals for the pre-registered rare-type tests

Accuracy % on the 2-4 stratum (reference A) and gain over stage 1, points; 95 % intervals from resampling true types (2000 draws, seed 0). Command: `python experiments/b_gnn_celltypes/bootstrap_ci.py`.

| Test | Labeling | Accuracy [95 %] | Gain over stage 1 [95 %] |
|---|---|---|---|
| BANC brain (Table 6) (14,512 neurons, 5,416 types) | stage 1 | 22.2 [21.1, 23.3] | – |
| BANC brain (Table 6) (14,512 neurons, 5,416 types) | two-stage, gnn | 41.3 [40.2, 42.4] | 19.1 [18.0, 20.1] |
| BANC brain (Table 6) (14,512 neurons, 5,416 types) | oracle candidates | 65.9 [64.8, 67.1] | 43.7 [42.5, 44.9] |
| BANC cord (Table 8) (5,989 neurons, 2,278 types) | stage 1 | 39.5 [37.5, 41.4] | – |
| BANC cord (Table 8) (5,989 neurons, 2,278 types) | two-stage, gnn | 52.1 [50.3, 53.9] | 12.6 [11.1, 14.1] |
| BANC cord (Table 8) (5,989 neurons, 2,278 types) | two-stage, union | 61.0 [59.2, 62.7] | 21.4 [19.8, 23.1] |
| BANC cord (Table 8) (5,989 neurons, 2,278 types) | oracle candidates | 75.5 [73.9, 77.1] | 36.0 [34.0, 38.0] |
| MCNS cord (Table 8) (5,945 neurons, 2,261 types) | stage 1 | 72.0 [70.3, 73.7] | – |
| MCNS cord (Table 8) (5,945 neurons, 2,261 types) | two-stage, gnn | 77.1 [75.6, 78.6] | 5.1 [3.9, 6.4] |
| MCNS cord (Table 8) (5,945 neurons, 2,261 types) | two-stage, union | 82.2 [80.9, 83.6] | 10.2 [8.9, 11.5] |
| MCNS cord (Table 8) (5,945 neurons, 2,261 types) | oracle candidates | 87.0 [85.8, 88.2] | 15.0 [13.4, 16.7] |
