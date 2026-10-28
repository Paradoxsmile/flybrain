Accuracy (%) / macro-F1 on the scored neurons. GNN: mean of single models.

| scenario | label | GNN | GNN vote | seeds (share, acc) | GNN + NTAC |
|---|---|---|---|---|---|
| transfer | super_class | 94.0 / 0.68 | 94.1 / 0.69 | 96 %, 96.1 | **94.4 / 0.66** |
| transfer | cell_type | 73.4 / 0.50 | 75.5 / 0.53 | 71 %, 89.0 | **79.5 / 0.57** |
| fafb-left_right | super_class | 97.9 / 0.84 | 98.0 / 0.85 | 99 %, 98.5 | **98.0 / 0.85** |
| fafb-left_right | cell_type | 94.5 / 0.90 | 94.8 / 0.91 | 94 %, 98.0 | **96.5 / 0.93** |
| fafb-right_left | super_class | 96.6 / 0.76 | 96.6 / 0.75 | 98 %, 97.6 | **96.9 / 0.76** |
| fafb-right_left | cell_type | 86.7 / 0.87 | 87.8 / 0.89 | 85 %, 95.7 | **90.8 / 0.92** |
| mcns-left_right | super_class | 97.6 / 0.86 | 97.7 / 0.87 | 99 %, 98.0 | **97.7 / 0.87** |
| mcns-left_right | cell_type | 95.4 / 0.93 | 96.0 / 0.94 | 95 %, 98.1 | **97.1 / 0.95** |
| mcns-right_left | super_class | 97.9 / 0.86 | 98.0 / 0.86 | 99 %, 98.4 | **98.0 / 0.87** |
| mcns-right_left | cell_type | 93.7 / 0.91 | 94.6 / 0.92 | 92 %, 97.7 | **96.2 / 0.93** |