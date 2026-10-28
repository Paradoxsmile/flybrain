Label-free FAFB -> HEMIBRAIN mapping (accuracy % / macro-F1 on labeled target neurons).

| label | chosen (variant, norm., rho) | no adaptation | prior matched | + NTAC | best candidate (post hoc) |
|---|---|---|---|---|---|
| super_class | raw, source, 1.0 | 72.8 | 34.7 / 0.14 | **34.5 / 0.13** | 73.1 |
| cell_type | raw, source, 1.0 | 17.3 | 15.8 / 0.07 | **17.5 / 0.08** | 24.9 |

super_class, accuracy % by subset (n labeled neurons):

| labeling | cropped | complete | central_brain | neck_connective | optic_lobe |
|---|---|---|---|---|---|
| no_adaptation | 68.3 (4039) | 73.7 (21303) | 84.3 (16398) | 23.5 (115) | 1.9 (3698) |
| selected | 59.2 (4039) | 30.1 (21303) | 46.1 (16398) | 0.9 (115) | 4.0 (3698) |
| refined | 58.5 (4039) | 30.0 (21303) | 46.1 (16398) | 0.9 (115) | 2.8 (3698) |
| best_possible | 64.0 (4039) | 74.8 (21303) | 84.3 (16398) | 32.2 (115) | 1.4 (3698) |

cell_type, accuracy % by subset (n labeled neurons):

| labeling | cropped | complete | central_brain | optic_lobe |
|---|---|---|---|---|
| no_adaptation | 18.3 (2322) | 16.9 (5317) | 25.5 (4867) | 3.0 (2772) |
| selected | 6.4 (2322) | 19.9 (5317) | 24.6 (4867) | 0.3 (2772) |
| refined | 8.2 (2322) | 21.6 (5317) | 27.4 (4867) | 0.0 (2772) |
| best_possible | 20.2 (2322) | 27.0 (5317) | 37.9 (4867) | 2.1 (2772) |