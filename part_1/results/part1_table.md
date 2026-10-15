| Model | Params | Test PPL | Val PPL | Train time (min) | Test-scoring tok/s | CPU tok/s | Gen ms/token | distinct-2 (sample) | repeat-3 (greedy) |
|---|---|---|---|---|---|---|---|---|---|
| Trigram (interp. KN) | 1.85M | 70.9 | 70.0 | 0.2 | 6,898,916 (cpu) | 3,800,385 | 1.7 | 0.95 | 0.64 |
| RNN | 8.02M | 71.6 | 70.9 | 13.9 | 6,113 (mps) | 31,441 | 1.8 | 0.95 | 0.26 |
| LSTM | 8.69M | 75.6 | 74.9 | 11.9 | 27,091 (mps) | 21,541 | 1.1 | 0.95 | 0.28 |
| Transformer | 11.02M | 45.3 | 44.7 | 22.6 | 41,525 (mps) | 14,141 | 9.2 | 0.91 | 0.27 |
