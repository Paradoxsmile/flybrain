Cell types by region of the type (FAFB super class), accuracy % / macro-F1.

| Run | Stage | all | optic-lobe intrinsic | visual projection | central brain | sensory | other |
|---|---|---|---|---|---|---|---|
| **FAFB → MCNS** | share of labeled neurons | 109,901 neurons | 80 % | 7 % | 6 % | 7 % | 0.2 % |
| | without prior matching | 87.6 / 0.73 | 92.9 / 0.91 | 86.4 / 0.84 | 52.6 / 0.63 | 55.4 / 0.21 | 88.2 / 0.74 |
| | with prior matching | 91.2 / 0.79 | 95.6 / 0.94 | 94.0 / 0.92 | 63.9 / 0.77 | 59.8 / 0.29 | 90.7 / 0.76 |
| | + NTAC | 92.6 / 0.81 | 97.2 / 0.96 | 95.7 / 0.96 | 65.1 / 0.80 | 59.1 / 0.26 | 95.4 / 0.73 |
| **MCNS → FAFB** | share of labeled neurons | 108,647 neurons | 71 % | 6 % | 7 % | 15 % | 0.4 % |
| | without prior matching | 72.0 / 0.57 | 78.5 / 0.78 | 73.2 / 0.73 | 35.5 / 0.38 | 59.5 / 0.20 | 46.7 / 0.40 |
| | with prior matching | 81.3 / 0.68 | 90.2 / 0.86 | 85.7 / 0.83 | 51.5 / 0.51 | 52.7 / 0.36 | 48.3 / 0.38 |
| | + NTAC | 83.4 / 0.69 | 92.6 / 0.90 | 87.8 / 0.86 | 54.0 / 0.52 | 53.6 / 0.31 | 52.8 / 0.41 |
| **FAFB → BANC** | share of labeled neurons | 63,332 neurons | 71 % | 9 % | 8 % | 12 % | 0.3 % |
| | without prior matching | 38.2 / 0.22 | 44.0 / 0.32 | 31.5 / 0.22 | 10.7 / 0.16 | 26.1 / 0.03 | 57.8 / 0.42 |
| | with prior matching | 55.7 / 0.39 | 62.0 / 0.47 | 56.8 / 0.45 | 36.9 / 0.37 | 28.2 / 0.15 | 70.2 / 0.48 |
| | + NTAC | 61.7 / 0.42 | 68.6 / 0.52 | 63.1 / 0.51 | 44.2 / 0.38 | 30.3 / 0.14 | 77.6 / 0.50 |
| **MCNS → BANC** | share of labeled neurons | 63,332 neurons | 71 % | 9 % | 8 % | 12 % | 0.3 % |
| | without prior matching | 1.4 / 0.00 | 1.8 / 0.00 | 0.0 / 0.00 | 0.3 / 0.00 | 1.1 / 0.00 | 0.0 / 0.00 |
| | with prior matching | 1.4 / 0.00 | 1.8 / 0.00 | 0.0 / 0.00 | 0.3 / 0.00 | 1.1 / 0.00 | 0.0 / 0.00 |
| | + NTAC | 1.4 / 0.00 | 1.8 / 0.00 | 0.0 / 0.00 | 0.5 / 0.00 | 0.7 / 0.00 | 0.0 / 0.00 |
| **FAFB → hemibrain** | share of labeled neurons | 7,639 neurons | 2 % | 35 % | 32 % | 30 % | 0.2 % |
| | without prior matching | 17.3 / 0.15 | 0.0 / 0.00 | 3.1 / 0.01 | 33.6 / 0.32 | 18.5 / 0.15 | 0.0 / 0.00 |
| | with prior matching | 19.1 / 0.13 | 0.0 / 0.00 | 2.6 / 0.02 | 46.0 / 0.25 | 11.9 / 0.15 | 0.0 / 0.00 |
| | + NTAC | 19.6 / 0.12 | 0.0 / 0.00 | 5.0 / 0.03 | 44.5 / 0.24 | 12.3 / 0.13 | 0.0 / 0.00 |

Cell types by FAFB type size (neurons), accuracy % / macro-F1.

| Run | Stage | all | 20–49 | 50–199 | 200–999 | ≥ 1000 |
|---|---|---|---|---|---|---|
| **FAFB → MCNS** | share of labeled neurons | 109,901 neurons | 5 % | 13 % | 23 % | 58 % |
| | without prior matching | 87.6 / 0.73 | 69.2 / 0.62 | 83.2 / 0.77 | 85.2 / 0.81 | 91.1 / 0.90 |
| | with prior matching | 91.2 / 0.79 | 75.8 / 0.72 | 87.4 / 0.83 | 90.0 / 0.86 | 93.9 / 0.92 |
| | + NTAC | 92.6 / 0.81 | 77.0 / 0.74 | 88.2 / 0.83 | 92.4 / 0.88 | 95.1 / 0.93 |
| **MCNS → FAFB** | share of labeled neurons | 108,647 neurons | 6 % | 12 % | 22 % | 60 % |
| | without prior matching | 72.0 / 0.57 | 51.0 / 0.44 | 74.6 / 0.67 | 66.7 / 0.65 | 75.4 / 0.76 |
| | with prior matching | 81.3 / 0.68 | 64.4 / 0.56 | 81.0 / 0.77 | 78.1 / 0.76 | 84.1 / 0.85 |
| | + NTAC | 83.4 / 0.69 | 64.6 / 0.57 | 82.9 / 0.78 | 80.7 / 0.79 | 86.3 / 0.87 |
| **FAFB → BANC** | share of labeled neurons | 63,332 neurons | 7 % | 15 % | 16 % | 62 % |
| | without prior matching | 38.2 / 0.22 | 18.5 / 0.14 | 22.6 / 0.23 | 45.4 / 0.32 | 42.5 / 0.41 |
| | with prior matching | 55.7 / 0.39 | 33.7 / 0.32 | 42.4 / 0.39 | 60.7 / 0.43 | 60.2 / 0.63 |
| | + NTAC | 61.7 / 0.42 | 35.3 / 0.34 | 46.5 / 0.43 | 66.9 / 0.49 | 67.3 / 0.69 |
| **MCNS → BANC** | share of labeled neurons | 63,332 neurons | 7 % | 15 % | 16 % | 62 % |
| | without prior matching | 1.4 / 0.00 | 0.5 / 0.00 | 0.0 / 0.00 | 0.1 / 0.00 | 2.2 / 0.01 |
| | with prior matching | 1.4 / 0.00 | 0.5 / 0.00 | 0.0 / 0.00 | 0.1 / 0.00 | 2.2 / 0.01 |
| | + NTAC | 1.4 / 0.00 | 0.8 / 0.00 | 0.0 / 0.00 | 0.0 / 0.00 | 2.1 / 0.01 |
| **FAFB → hemibrain** | share of labeled neurons | 7,639 neurons | 34 % | 42 % | 12 % | 11 % |
| | without prior matching | 17.3 / 0.15 | 24.7 / 0.20 | 18.7 / 0.11 | 1.0 / 0.00 | 7.8 / 0.03 |
| | with prior matching | 19.1 / 0.13 | 20.1 / 0.17 | 10.2 / 0.08 | 1.1 / 0.02 | 68.8 / 0.12 |
| | + NTAC | 19.6 / 0.12 | 19.3 / 0.16 | 12.1 / 0.08 | 0.3 / 0.01 | 69.2 / 0.12 |

Points of the overall accuracy contributed by each region (+ NTAC).

| Run | all | optic-lobe intrinsic | visual projection | central brain | sensory | other |
|---|---|---|---|---|---|---|
| FAFB → MCNS | 92.6 | 78.0 | 6.6 | 3.7 | 4.1 | 0.2 |
| MCNS → FAFB | 83.4 | 65.8 | 5.4 | 4.0 | 8.1 | 0.2 |
| FAFB → BANC | 61.7 | 48.8 | 5.7 | 3.6 | 3.5 | 0.2 |
| MCNS → BANC | 1.4 | 1.3 | 0.0 | 0.0 | 0.1 | 0.0 |
| FAFB → hemibrain | 19.6 | 0.0 | 1.8 | 14.2 | 3.7 | 0.0 |
