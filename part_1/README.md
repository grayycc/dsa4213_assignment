# Part I — Language model training and comparison

Four next-token language models trained on the same corpus and scored on the same test targets:
interpolated Kneser–Ney trigram, vanilla RNN, LSTM, and a causal Transformer.

## Data

- **Corpus:** PubMed 20k RCT, `armanc/pubmed-rct20k` on the Hugging Face Hub (Dernoncourt & Lee, 2017).
  It contains sentences from abstracts of randomised controlled trials. It is downloaded automatically by
  `data.py`. No data is redistributed in this repo.
- **Format:** the corpus is already tokenised, and every digit is replaced by `@`. Each sentence also has a
  rhetorical-role label (background, objective, methods, results or conclusions). The labels are not used
  for language modelling; Part III uses them.
- **Subset:** the official splits are disjoint by abstract. From each one we take a seeded random sample
  of whole abstracts (`SUBSET_SEED = 0`): 6,500 of 15,000 for training, 700 of 2,500 for validation and
  800 of 2,500 for test. The chosen abstract ids are saved to `cache/lm_abstract_ids.json`.
- **Preprocessing** ([data.py](data.py)):
  - lowercase the text and split on whitespace;
  - keep each abstract's sentences in their original order;
  - append `<eos>` after every sentence.
- **Vocabulary:** built from the training subset only, keeping words with frequency ≥ 3. This gives
  21,844 types, including `<pad>`, `<unk>` and `<eos>`. Other words become `<unk>`.

| split | abstracts | tokens | `<unk>` rate |
|---|---|---|---|
| train | 6,500 | 2,112,630 | 1.30% |
| validation | 700 | 230,639 | 2.99% |
| test | 800 | 261,106 | 3.12% |

## Models (defaults in [train.py](train.py) `DEFAULTS`)

| Model | Architecture |
|---|---|
| Trigram | Interpolated Kneser–Ney with three discounts, grid-searched on validation ([ngram.py](ngram.py)) |
| RNN | Embedding 100 → 2-layer Elman RNN (tanh), hidden size 256, dropout 0.3 → linear to vocab |
| LSTM | Embedding 100 → 2-layer LSTM, hidden size 256, dropout 0.3 → linear to vocab |
| Transformer | Embedding 100 → linear to 256 + learned positions; 4 pre-norm decoder blocks (4 heads, FFN 1024, GELU, dropout 0.2); causal mask; context 128 |

All neural models share these settings:
- input embeddings that are randomly initialised and trainable;
- output layer not tied to the input embeddings;
- the same batches in the same order: 32 contiguous streams × 128-token chunks;
- AdamW (lr 1e-3, weight decay 0.01), 300 linear warm-up steps then cosine decay, gradient clipping at 1.0;
- 10 epochs; the checkpoint with the lowest validation loss is kept;
- seed 42.

The RNN and LSTM carry their hidden state across chunks (truncated backpropagation through time).

## Evaluation protocol ([lm_eval.py](lm_eval.py))

Every model scores the same 261,105 test targets: each token `x_t` for t ≥ 1 is scored using only `x_<t`. The result is reported as average cross-entropy and perplexity.

How each model sees context:
- **RNN/LSTM:** a single stream (batch size 1) with the hidden state carried through, so the whole history is available.
- **Transformer:** 128-token sliding windows with stride 64. Each target is scored once, with ≥ 64 tokens of context.
- **Trigram:** the previous two tokens.

The loss uses `ignore_index=<pad>`. The only padding is in the last, partial Transformer window.

Timing:
- **Training time:** wall-clock time of the training loop only (validation excluded), with the GPU synchronised.
- **Inference:** a common workload of scoring the full test set, timed as the median of 3 runs after a warm-up.
- **CPU throughput:** a second measurement for all four models on the first 20k test tokens, so every model is compared on the same hardware.
- **Generation latency:** ms per token for greedy decoding with batch size 1.

Generation:
- the same 5 prompts and 40 new tokens for every model;
- greedy decoding, plus sampling with temperature 0.8 and top-k 40 (seeds 0, 1, 2);
- repetition is measured with distinct-1/2 and the repeated-trigram rate.

## Reproduce

One-time setup, run from the repository root (`dsa4213_assignment1/`, not `part_1/`):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then run the full Part I pipeline. It downloads the data, trains the RNN, LSTM and Transformer one after another (about 50 min on an M4), then runs evaluation and plotting:

```bash
bash part_1/run_part1.sh
```

Outputs:
- `results/part1_table.md` and `results/part1_results.json`: metrics
- `results/generations.md`: generated text
- `results/history_*.json` and `figures/part1_loss_curves.png`: learning curves
- `checkpoints/*.pt`: best-validation weights

**Hardware:** Apple M4 (10-core CPU and GPU, 16 GB RAM), PyTorch MPS backend. The trigram runs on the CPU with numpy.
