# Part II — Embedding variants and ablation

We compare three input-embedding settings for one architecture: the Part I **LSTM** (embedding 100 → 2-layer
LSTM, hidden size 256, dropout 0.3 → linear to vocab). Only the input-embedding table changes between runs.

| Setting | Input embeddings | Trainable? | Run tag |
|---|---|---|---|
| A | random init `N(0,1)` | yes | `lstm` (the Part I run, reused) |
| B | own Skip-gram, trained on the LM training split | **frozen** | `lstm_w2v` |
| C | public GloVe 6B-100d | **frozen** | `lstm_glove` |
| control (optional) | random init | frozen | `lstm_randfrozen` |
| control (optional) | GloVe | trainable | `lstm_glove_ft` |

## Why the LSTM

The Transformer passes its embeddings through a trainable 100→256 projection. A "frozen" table there would
still get a learned transform. The LSTM consumes the table directly, so freezing it really fixes the input
representation.

## What is held constant

The following are identical across settings, as in Part I:
- data, vocabulary (21,844 types) and the order of training batches;
- embedding dimension (**100**, chosen to match GloVe);
- architecture, AdamW (lr 1e-3, weight decay 0.01), warm-up + cosine schedule, gradient clipping at 1.0;
- 10 epochs and best-validation checkpointing;
- seed 42.

Because the seed is fixed and the embedding table is built first, the LSTM and output-layer weights start
identical in every run. In B and C, `set_input_embeddings` then overwrites the table and sets
`requires_grad=False` on the **whole** table, including fallback/random rows. The optimiser only receives
trainable parameters. `evaluate_part2.py` asserts that each frozen table is bit-identical after training.

Freezing removes 2.18M (21,844 × 100) trainable parameters, leaving 6.51M of 8.69M trainable.

## Embedding construction

**B: Skip-gram** ([train_w2v.py](train_w2v.py))
- gensim 4.4 Word2Vec with skip-gram and negative sampling: dim 100, window 5, 5 negatives,
  sub-sampling 1e-3, 10 epochs, seed 42, 1 worker, `PYTHONHASHSEED=0`.
- Trained on the exact LM training id stream, split into sentences at `<eos>`. Rare words are therefore
  `<unk>`, as in LM training, and every vocab type except `<pad>` (zero row) gets a vector.
- Validation and test text are never used.
- Its cost is timed separately from LM training and stored in `emb/w2v100_meta.json`.

**C: GloVe** ([build_glove.py](build_glove.py))
- Vectors: `glove-wiki-gigaword-100` from gensim-data, which is Stanford's `glove.6B.100d`
  (Pennington et al., 2014): Wikipedia 2014 + Gigaword 5, ~6B tokens, 400k uncased words.
- Missing words are initialised by the first rule that applies:

| Rule | Example | Types | Train tokens | Test tokens |
|---|---|---|---|---|
| exact match | `placebo` | 15,323 (70.2%) | 91.3% | 90.5% |
| digit: `@` → `0` | `@` → `0` | 40 | 0.67% | 0.65% |
| alias: PTB brackets, or `@` glued to letters | `-lsb-` → `[`, `@weeks` → `weeks` | 289 | 0.58% | 0.55% |
| compound: mean of covered parts | `placebo-controlled` | 2,879 | 1.20% | 0.83% |
| random `N(μ_d, σ_d²)` from GloVe's per-dimension stats, seed 42 | `bupivacaine`, `hrqol`, `randomisation` | 3,310 (15.2%) | 1.33% | 0.75% |

Special tokens:
- `<pad>`: zeros.
- `<unk>`: mean of the exact-match vectors.
- `<eos>`: random. It is not set to `.`, because `.` always precedes `<eos>` and the two inputs would then
  be identical.

Token shares do not sum to 100% because special tokens make up the rest. The per-word rule is saved in
`emb/glove100_coverage.json`.

## Evaluation ([evaluate_part2.py](evaluate_part2.py))

- **Test quality:** the Part I protocol (same 261,105 targets, batch 1, hidden state carried through).
- **Convergence:** validation perplexity per epoch, and the first epoch within 5% of the run's best.
- **Stability:**
  - pre-clip gradient norm, every 50 steps (median / p95 / max);
  - fraction of steps clipped;
  - jitter in the step-to-step training loss;
  - number of epochs where validation loss went up;
  - final validation−train gap.
- **Generated text:** the Part I prompts and decoding settings, plus repetition statistics.
- **Extra analyses:**
  - test perplexity bucketed by the training frequency of the target word and of the input word;
  - test perplexity by the GloVe initialisation rule of the input word;
  - cosine nearest neighbours of probe words in each table.

Outputs:
- `results/part2_table.md`, `part2_results.json`
- `results/part2_freq_buckets.md`, `part2_neighbours.md`, `part2_generations.md`
- `figures/part2_curves.png`

## Reproduce

From the repository root, after `bash part_1/run_part1.sh`:

```bash
pip install -r requirements.txt     # adds gensim==4.4.0
bash part_2/run_part2.sh            # ~35 min on an M4: GloVe download, skip-gram, 2 LSTM runs, evaluation
CONTROLS=1 bash part_2/run_part2.sh # optionally also the two 2x2 control runs
```

GloVe is downloaded with `curl` because the python.org macOS Python fails SSL verification in
`gensim.downloader`. gensim prints a harmless `Exception ignored in: ... our_dot_float` message from its BLAS
self-check on import.

**Hardware:** Apple M4 (10-core CPU and GPU, 16 GB RAM). LMs train on MPS; Skip-gram trains on the CPU with
1 thread.
