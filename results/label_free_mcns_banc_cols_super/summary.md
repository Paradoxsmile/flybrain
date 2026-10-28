Label-free MCNS -> BANC mapping (accuracy % / macro-F1 on labeled target neurons).

| label | chosen (variant, norm., rho) | no adaptation | prior matched | + NTAC | best candidate (post hoc) |
|---|---|---|---|---|---|
| super_class | raw/raw, source, 1.0 | 69.0 | 78.7 / 0.33 | **79.0 / 0.31** | 94.3 |

super_class, accuracy % by subset (n labeled neurons):

| labeling | central_brain | optic_lobe |
|---|---|---|
| no_adaptation | 50.2 (39120) | 79.8 (81151) |
| selected | 68.6 (39120) | 85.0 (81151) |
| refined | 67.5 (39120) | 86.2 (81151) |
| best_possible | 93.4 (39120) | 95.6 (81151) |