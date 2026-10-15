
### Test PPL by training frequency of the TARGET word

| Setting | <unk> (3.1%) | 3-10 (2.3%) | 11-100 (9.5%) | 101-1k (19.3%) | >1k (65.7%) |
|---|---|---|---|---|---|
| A: random, trainable | 50.0 | 146755.8 | 12102.1 | 876.8 | 13.8 |
| B: own skip-gram, frozen | 50.4 | 115882.5 | 9203.7 | 707.3 | 14.6 |
| C: GloVe, frozen | 52.4 | 143576.8 | 11678.5 | 867.8 | 15.6 |
| random, frozen (control) | 51.8 | 162861.5 | 14543.2 | 1067.9 | 16.2 |
| GloVe, trainable (control) | 49.1 | 127024.0 | 10072.4 | 734.5 | 13.3 |

### Test PPL by training frequency of the INPUT (previous) word

| Setting | <unk> (3.1%) | 3-10 (2.3%) | 11-100 (9.5%) | 101-1k (19.3%) | >1k (65.7%) |
|---|---|---|---|---|---|
| A: random, trainable | 179.5 | 227.4 | 155.9 | 85.6 | 60.6 |
| B: own skip-gram, frozen | 168.1 | 185.0 | 136.2 | 82.3 | 60.0 |
| C: GloVe, frozen | 182.0 | 202.5 | 157.9 | 100.2 | 65.4 |
| random, frozen (control) | 195.8 | 293.0 | 237.1 | 126.2 | 64.8 |
| GloVe, trainable (control) | 177.8 | 157.5 | 112.1 | 70.2 | 60.7 |

### Test PPL by how the INPUT word was initialised in the GloVe table

| Setting | exact (90.5%) | digit (0.6%) | alias (0.6%) | compound (0.8%) | random (0.8%) | special (6.8%) |
|---|---|---|---|---|---|---|
| A: random, trainable | 70.8 | 17.5 | 30.4 | 191.6 | 90.4 | 196.8 |
| B: own skip-gram, frozen | 68.8 | 19.6 | 31.7 | 137.8 | 79.3 | 184.0 |
| C: GloVe, frozen | 76.9 | 22.5 | 36.8 | 211.7 | 107.3 | 196.6 |
| random, frozen (control) | 84.3 | 20.7 | 35.0 | 267.9 | 129.2 | 205.3 |
| GloVe, trainable (control) | 65.4 | 16.8 | 27.4 | 131.3 | 74.6 | 197.2 |

Percentages are the share of the 261,105 test targets in each bucket.
