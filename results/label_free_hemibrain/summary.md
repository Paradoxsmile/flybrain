Label-free FAFB -> HEMIBRAIN mapping (accuracy % / macro-F1 on labeled target neurons).

| label | chosen (variant, norm., rho) | no adaptation | prior matched | + NTAC | best candidate (post hoc) |
|---|---|---|---|---|---|
| super_class | raw, source, 0.5 | 72.8 | 74.3 / 0.20 | **74.4 / 0.19** | 78.4 |
| cell_type | raw, source, 1.0 | 17.3 | 19.1 / 0.09 | **19.6 / 0.09** | 25.5 |

super_class, accuracy % by subset (n labeled neurons):

| labeling | cropped | complete | central_brain | neck_connective | optic_lobe |
|---|---|---|---|---|---|
| no_adaptation | 68.3 (4039) | 73.7 (21303) | 84.3 (16398) | 23.5 (115) | 1.9 (3698) |
| selected | 65.9 (4039) | 75.9 (21303) | 84.8 (16398) | 40.9 (115) | 6.8 (3698) |
| refined | 65.5 (4039) | 76.1 (21303) | 85.4 (16398) | 43.5 (115) | 3.7 (3698) |
| best_possible | 85.1 (4039) | 77.1 (21303) | 89.8 (16398) | 47.8 (115) | 13.5 (3698) |

cell_type, accuracy % by subset (n labeled neurons):

| labeling | cropped | complete | central_brain | optic_lobe |
|---|---|---|---|---|
| no_adaptation | 18.3 (2322) | 16.9 (5317) | 25.5 (4867) | 3.0 (2772) |
| selected | 11.9 (2322) | 22.3 (5317) | 28.6 (4867) | 2.5 (2772) |
| refined | 12.2 (2322) | 22.8 (5317) | 28.0 (4867) | 4.8 (2772) |
| best_possible | 22.6 (2322) | 26.7 (5317) | 38.3 (4867) | 3.0 (2772) |