Thinned FAFB copies: accuracy % on the test split, mean over seeds.

cell_type, rho 0 (rows: source density, columns: target density):

| p_s \ p_t | 1 | 0.6 | 0.4 | 0.25 |
|---|---|---|---|---|
| 1 | 92.8 | 89.5 | 80.5 | 55.6 |

cell_type, rho 1 (rows: source density, columns: target density):

| p_s \ p_t | 1 | 0.6 | 0.4 | 0.25 |
|---|---|---|---|---|
| 1 | 94.0 | 90.9 | 83.7 | 63.7 |

super_class, rho 0 (rows: source density, columns: target density):

| p_s \ p_t | 1 | 0.6 | 0.4 | 0.25 |
|---|---|---|---|---|
| 1 | 97.4 | 96.7 | 94.4 | 86.9 |

super_class, rho 1 (rows: source density, columns: target density):

| p_s \ p_t | 1 | 0.6 | 0.4 | 0.25 |
|---|---|---|---|---|
| 1 | 97.4 | 96.7 | 94.5 | 90.3 |

Graph statistics of the copies:

| label | copy | p | connections_per_neuron | synapses_per_neuron | synapses_per_connection | median_input_synapses | median_input_partners | isolated |
|---|---|---|---|---|---|---|---|---|
| cell_type | source | 1 | 26.8 | 364 | 13.6 | 152 | 14 | 0.00482 |
| cell_type | target | 1 | 26.8 | 364 | 13.6 | 152 | 14 | 0.00482 |
| cell_type | target | 0.6 | 15.7 | 184 | 11.7 | 69 | 7 | 0.00979 |
| cell_type | target | 0.4 | 9.8 | 104 | 10.6 | 34 | 4 | 0.0264 |
| cell_type | target | 0.25 | 5.37 | 50.9 | 9.48 | 14 | 2 | 0.0874 |
| super_class | source | 1 | 26.8 | 364 | 13.6 | 152 | 14 | 0.00482 |
| super_class | target | 1 | 26.8 | 364 | 13.6 | 152 | 14 | 0.00482 |
| super_class | target | 0.6 | 15.7 | 184 | 11.7 | 69 | 7 | 0.00979 |
| super_class | target | 0.4 | 9.8 | 104 | 10.6 | 34 | 4 | 0.0264 |
| super_class | target | 0.25 | 5.37 | 50.9 | 9.48 | 14 | 2 | 0.0874 |