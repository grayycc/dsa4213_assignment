| Setting | Trainable params | Test PPL | Best val PPL (epoch) | Val PPL ep.1 | Epochs to ≤5% of best | LM train time (min) | Grad-norm median / p95 | Clipped steps | Final val−train gap | distinct-2 (sample) | repeat-3 (greedy) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A: random, trainable | 8.69M / 8.69M | 75.6 | 74.9 (10) | 206.7 | 7 | 11.9 | 0.21 / 0.27 | 1% | -0.193 | 0.95 | 0.28 |
| B: own skip-gram, frozen | 6.51M / 8.69M | 73.1 | 72.7 (10) | 260.9 | 7 | 11.8 | 0.23 / 0.30 | 1% | -0.255 | 0.88 | 0.44 |
| C: GloVe, frozen | 6.51M / 8.69M | 81.8 | 81.1 (10) | 239.4 | 7 | 11.1 | 0.23 / 0.29 | 1% | -0.245 | 0.93 | 0.38 |
| random, frozen (control) | 6.51M / 8.69M | 89.4 | 88.6 (10) | 232.6 | 7 | 11.4 | 0.22 / 0.28 | 1% | -0.234 | 0.92 | 0.47 |
| GloVe, trainable (control) | 8.69M / 8.69M | 69.9 | 69.3 (10) | 200.7 | 7 | 11.7 | 0.22 / 0.29 | 1% | -0.184 | 0.90 | 0.22 |

Own skip-gram embedding cost (separate from LM training): 32.3 s on arm64 CPU (macOS-26.6.2-arm64-arm-64bit), 1 worker thread(s), 2,112,630 training tokens.
GloVe: no training on our side; pre-trained by its authors on ~6B tokens (about 2,800x our 2.1M-token LM training split).
