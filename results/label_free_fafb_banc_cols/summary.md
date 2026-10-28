Label-free FAFB -> BANC mapping (accuracy % / macro-F1 on labeled target neurons).

| label | chosen (variant, norm., rho) | no adaptation | prior matched | + NTAC | best candidate (post hoc) |
|---|---|---|---|---|---|
| super_class | scaled, source, 1.0 | 42.2 | 72.3 / 0.24 | **73.3 / 0.22** | 92.6 |
| cell_type | density, target, 1.0 | 38.2 | 55.7 / 0.37 | **61.7 / 0.40** | 55.7 |

super_class, accuracy % by subset (n labeled neurons):

| labeling | central_brain | optic_lobe |
|---|---|---|
| no_adaptation | 19.6 (39120) | 54.3 (81151) |
| selected | 67.6 (39120) | 76.6 (81151) |
| refined | 67.9 (39120) | 78.0 (81151) |
| best_possible | 93.5 (39120) | 94.5 (81151) |

cell_type, accuracy % by subset (n labeled neurons):

| labeling | central_brain | optic_lobe | morph |
|---|---|---|---|
| no_adaptation | 9.0 (10728) | 44.1 (52603) | 8.5 (11656) |
| selected | 28.8 (10728) | 61.1 (52603) | 27.4 (11656) |
| refined | 33.7 (10728) | 67.4 (52603) | 32.3 (11656) |
| best_possible | 28.8 (10728) | 61.1 (52603) | 27.4 (11656) |