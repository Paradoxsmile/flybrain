Label-free MCNS -> BANC mapping (accuracy % / macro-F1 on labeled target neurons).

| label | chosen (variant, norm., rho) | no adaptation | prior matched | + NTAC | best candidate (post hoc) |
|---|---|---|---|---|---|
| cell_type | density/scaled, source, 0.0 | 1.4 | 1.4 / 0.00 | **1.4 / 0.00** | 58.0 |

cell_type, accuracy % by subset (n labeled neurons):

| labeling | central_brain | optic_lobe | morph |
|---|---|---|---|
| no_adaptation | 0.3 (10728) | 1.6 (52603) | 0.4 (11656) |
| selected | 0.3 (10728) | 1.6 (52603) | 0.4 (11656) |
| refined | 0.4 (10728) | 1.6 (52603) | 0.4 (11656) |
| best_possible | 27.3 (10728) | 64.3 (52603) | 24.6 (11656) |