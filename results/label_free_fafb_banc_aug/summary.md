Label-free FAFB -> BANC mapping (accuracy % / macro-F1 on labeled target neurons).

| label | chosen (variant, norm., rho) | no adaptation | prior matched | + NTAC | best candidate (post hoc) |
|---|---|---|---|---|---|
| super_class | raw, target, 0.0 | 84.4 | 84.4 / 0.43 | **85.1 / 0.41** | 92.9 |
| cell_type | scaled, source, 0.0 | 7.3 | 7.3 / 0.01 | **5.7 / 0.00** | 55.0 |

super_class, accuracy % by subset (n labeled neurons):

| labeling | central_brain | optic_lobe |
|---|---|---|
| no_adaptation | 85.8 (39120) | 85.9 (81151) |
| selected | 85.8 (39120) | 85.9 (81151) |
| refined | 85.0 (39120) | 87.4 (81151) |
| best_possible | 92.9 (39120) | 95.2 (81151) |

cell_type, accuracy % by subset (n labeled neurons):

| labeling | central_brain | optic_lobe | morph |
|---|---|---|---|
| no_adaptation | 2.2 (10728) | 8.3 (52603) | 2.8 (11656) |
| selected | 2.2 (10728) | 8.3 (52603) | 2.8 (11656) |
| refined | 0.8 (10728) | 6.7 (52603) | 1.2 (11656) |
| best_possible | 23.9 (10728) | 61.3 (52603) | 23.1 (11656) |