Label-free FAFB -> BANC mapping (accuracy % / macro-F1 on labeled target neurons).

| label | chosen (variant, norm., rho) | no adaptation | prior matched | + NTAC | best candidate (post hoc) |
|---|---|---|---|---|---|
| super_class | scaled, source, 1.0 | 58.0 | 66.6 / 0.20 | **65.4 / 0.17** | 90.7 |
| cell_type | scaled, target, 1.0 | 35.6 | 48.6 / 0.30 | **51.5 / 0.31** | 51.2 |

super_class, accuracy % by subset (n labeled neurons):

| labeling | central_brain | optic_lobe |
|---|---|---|
| no_adaptation | 14.1 (39120) | 81.0 (81151) |
| selected | 58.5 (39120) | 72.4 (81151) |
| refined | 58.3 (39120) | 70.7 (81151) |
| best_possible | 86.9 (39120) | 95.1 (81151) |

cell_type, accuracy % by subset (n labeled neurons):

| labeling | central_brain | optic_lobe | morph |
|---|---|---|---|
| no_adaptation | 17.2 (10728) | 39.4 (52603) | 16.1 (11656) |
| selected | 29.8 (10728) | 52.5 (52603) | 29.0 (11656) |
| refined | 31.7 (10728) | 55.6 (52603) | 31.3 (11656) |
| best_possible | 28.8 (10728) | 55.7 (52603) | 27.9 (11656) |