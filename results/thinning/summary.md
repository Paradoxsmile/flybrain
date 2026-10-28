Thinned FAFB copies: accuracy % on the test split, mean over seeds.

cell_type, rho 0 (rows: source density, columns: target density):

| p_s \ p_t | 1 | 0.6 | 0.4 | 0.25 |
|---|---|---|---|---|
| 1 | 94.1 | 85.0 | 62.4 | 28.4 |
| 0.6 | 88.3 | 89.6 | 77.2 | 42.5 |
| 0.4 | 70.0 | 83.2 | 81.7 | 55.6 |
| 0.25 | 41.4 | 59.9 | 68.9 | 64.0 |

cell_type, rho 1 (rows: source density, columns: target density):

| p_s \ p_t | 1 | 0.6 | 0.4 | 0.25 |
|---|---|---|---|---|
| 1 | 95.1 | 89.4 | 72.9 | 41.7 |
| 0.6 | 93.2 | 91.3 | 82.4 | 54.0 |
| 0.4 | 87.0 | 88.8 | 83.9 | 64.5 |
| 0.25 | 56.3 | 72.6 | 76.5 | 67.7 |

cell_type, denser target thinned to the source density:

| p_s | p_t | as is, rho 0 | thinned, rho 0 | as is, rho 1 | thinned, rho 1 |
|---|---|---|---|---|---|
| 0.25 | 0.4 | 68.9 | 64.0 | 76.5 | 67.6 |
| 0.25 | 0.6 | 59.9 | 63.7 | 72.6 | 67.5 |
| 0.25 | 1 | 41.4 | 63.6 | 56.3 | 67.3 |
| 0.4 | 0.6 | 83.2 | 82.0 | 88.8 | 84.2 |
| 0.4 | 1 | 70.0 | 82.1 | 87.0 | 84.3 |
| 0.6 | 1 | 88.3 | 89.7 | 93.2 | 91.4 |

super_class, rho 0 (rows: source density, columns: target density):

| p_s \ p_t | 1 | 0.6 | 0.4 | 0.25 |
|---|---|---|---|---|
| 1 | 97.8 | 95.2 | 83.6 | 46.6 |
| 0.6 | 97.2 | 97.1 | 93.4 | 72.9 |
| 0.4 | 92.9 | 95.7 | 95.1 | 86.8 |
| 0.25 | 89.6 | 93.1 | 93.7 | 91.8 |

super_class, rho 1 (rows: source density, columns: target density):

| p_s \ p_t | 1 | 0.6 | 0.4 | 0.25 |
|---|---|---|---|---|
| 1 | 97.8 | 96.2 | 91.5 | 74.7 |
| 0.6 | 97.6 | 97.1 | 94.5 | 86.4 |
| 0.4 | 96.1 | 96.6 | 95.1 | 90.8 |
| 0.25 | 93.1 | 95.0 | 94.3 | 91.8 |

super_class, denser target thinned to the source density:

| p_s | p_t | as is, rho 0 | thinned, rho 0 | as is, rho 1 | thinned, rho 1 |
|---|---|---|---|---|---|
| 0.25 | 0.4 | 93.7 | 91.9 | 94.3 | 91.9 |
| 0.25 | 0.6 | 93.1 | 91.9 | 95.0 | 91.9 |
| 0.25 | 1 | 89.6 | 91.8 | 93.1 | 91.8 |
| 0.4 | 0.6 | 95.7 | 95.1 | 96.6 | 95.1 |
| 0.4 | 1 | 92.9 | 95.0 | 96.1 | 95.0 |
| 0.6 | 1 | 97.2 | 97.0 | 97.6 | 97.0 |

Graph statistics of the copies:

| label | copy | p | connections_per_neuron | synapses_per_neuron | synapses_per_connection | median_input_synapses | median_input_partners | isolated |
|---|---|---|---|---|---|---|---|---|
| cell_type | source | 1 | 26.8 | 364 | 13.6 | 152 | 14 | 0.00482 |
| cell_type | source | 0.6 | 15.7 | 184 | 11.7 | 69 | 7 | 0.00989 |
| cell_type | source | 0.4 | 9.79 | 104 | 10.6 | 34 | 4 | 0.0269 |
| cell_type | source | 0.25 | 5.36 | 50.8 | 9.48 | 14 | 2 | 0.0874 |
| cell_type | target | 1 | 26.8 | 364 | 13.6 | 152 | 14 | 0.00482 |
| cell_type | target | 0.6 | 15.7 | 184 | 11.7 | 69 | 7 | 0.00979 |
| cell_type | target | 0.4 | 9.8 | 104 | 10.6 | 34 | 4 | 0.0264 |
| cell_type | target | 0.25 | 5.37 | 50.9 | 9.48 | 14 | 2 | 0.0874 |
| super_class | source | 1 | 26.8 | 364 | 13.6 | 152 | 14 | 0.00482 |
| super_class | source | 0.6 | 15.7 | 184 | 11.7 | 69 | 7 | 0.00989 |
| super_class | source | 0.4 | 9.79 | 104 | 10.6 | 34 | 4 | 0.0269 |
| super_class | source | 0.25 | 5.36 | 50.8 | 9.48 | 14 | 2 | 0.0874 |
| super_class | target | 1 | 26.8 | 364 | 13.6 | 152 | 14 | 0.00482 |
| super_class | target | 0.6 | 15.7 | 184 | 11.7 | 69 | 7 | 0.00979 |
| super_class | target | 0.4 | 9.8 | 104 | 10.6 | 34 | 4 | 0.0264 |
| super_class | target | 0.25 | 5.37 | 50.9 | 9.48 | 14 | 2 | 0.0874 |