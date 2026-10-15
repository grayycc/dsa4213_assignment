# Part III — Downstream representation analysis

**Question:** does better next-token prediction give better sentence representations for a downstream task?

The language models from Parts I–II are kept **frozen**. Each one turns a sentence into a fixed-length vector,
and the same classifier is trained on every representation. The task is **rhetorical-role classification** of
PubMed 20k RCT sentences into five classes: background, objective, methods, results and conclusions.

## Data ([data_cls.py](data_cls.py))

- The labels come from the same corpus as Part I (`armanc/pubmed-rct20k`). Vocabulary and tokenisation are
  unchanged.
- **No overlap with LM text:** the classifier only uses abstracts that are *not* in the LM subsets listed in
  `part_1/cache/lm_abstract_ids.json`. Splitting is by abstract, as in the official splits:

| Classifier split | Source | Abstracts |
|---|---|---|
| train | unused official-train abstracts (8,500 available) | 2,000, sampled with `CLS_SEED = 0` |
| validation | unused official-validation abstracts (1,800 available) | 500, sampled with `CLS_SEED = 0` |
| test | unused official-test abstracts | all 1,700 |

- `data_cls.py` asserts that no classifier abstract appears in any LM split and that the classifier splits do
  not overlap. The chosen ids are saved to `cache/cls_abstract_ids.json`.
- Run `python data_cls.py` to print the sentence counts and label distribution for each split.

## From a sentence to a fixed-length vector ([extract.py](extract.py))

1. **Input:** `<eos> w_1 … w_n`, mapped with the Part I vocabulary (OOV → `<unk>`) and truncated to 128
   tokens. In the LM training stream every sentence followed an `<eos>`, so it acts as a start token.
2. **Encoding:** each sentence is encoded on its own, without the rest of the abstract.
   - The recurrent state starts at zero; Transformer positions start at 0.
   - Models run in `eval()` mode (no dropout) under `no_grad`.
   - Batches are right-padded. Every model is causal, so padding never affects the states at real tokens.
3. **Hidden states:** taken from the top layer, i.e. the input to the vocabulary projection:
   - RNN/LSTM: output of the top recurrent layer, size 256;
   - Transformer: output of the final LayerNorm, size 256.
4. **Pooling:**

| Pool | Definition | Dim |
|---|---|---|
| `last` | top-layer state at the last real token. The models read left to right, so this state has seen the whole sentence. | 256 |
| `mean` | average of top-layer states over `w_1 … w_n`. Excludes the leading `<eos>` and padding. | 256 |
| `emb` | average of the model's **input embedding** rows over `w_1 … w_n`. Uses no context: a bag-of-vectors control. | 100 |

Features are stored as float16 in `feats/<tag>.npz`. Extraction time is recorded in `feats/<tag>_meta.json`.

## Representations compared

| Group | Representations | Question |
|---|---|---|
| A: model family | `rnn`, `lstm`, `transformer` × {last, mean} | Does the perplexity ranking match the downstream ranking? |
| B: embedding variant | `lstm` (random, trainable), `lstm_w2v`, `lstm_glove` (frozen); optional controls `lstm_randfrozen`, `lstm_glove_ft` × {last, mean} | Do Part II's embedding choices carry over downstream? |
| C: no context | `emb` pooling for every model | How much do the LM's contextual layers add on top of its word vectors? |
| baselines | majority class; TF-IDF word 1–2-grams (fitted on classifier train only) | floor / non-neural reference |

Any checkpoint missing from `part_1/checkpoints/` is skipped. This applies, for example, to the optional
controls.

## Classifier and evaluation ([classify.py](classify.py))

- **Classifier:** the same for every representation.
  - `StandardScaler` → multinomial `LogisticRegression` (lbfgs, `max_iter=3000`, `random_state=0`).
  - `C` ∈ {0.001, 0.01, 0.1, 1, 10, 100} is chosen by **validation macro-F1**, then the model is scored once on test.
  - TF-IDF uses the same LR and grid without scaling, because its features are sparse.
- **Metrics:**
  - macro-F1 is the main metric, because the classes are imbalanced;
  - accuracy and per-class F1 are also reported.
- **Uncertainty:** 95% percentile bootstrap over test **abstracts** (1,000 resamples, seed 0).
  - Sentences within an abstract are not independent, which is why abstracts are resampled.
  - Every representation uses the same resamples, so pairwise differences in `part3_pairs.md` are paired
    bootstrap tests.
- **LM vs downstream ranking:**
  - Spearman ρ between test perplexity (Part I protocol) and macro-F1, computed for each pooling method;
  - the scatter plot is `figures/part3_ppl_vs_f1.png`.
  - With ≤ 7 models this is only suggestive.
- **Error analysis** (`part3_errors.md`):
  - the most common confusions for the best representation, which is chosen on *validation*;
  - accuracy by sentence position in the abstract (first / middle / last);
  - sampled misclassified sentences;
  - sentences that the Transformer gets right and the LSTM gets wrong, and the reverse;
  - row-normalised confusion matrices in `figures/part3_confusion.png`.

## Outputs

- `results/part3_table.md`, `results/part3_results.json`: all metrics, chosen `C`, confusion matrices,
  Spearman ρ
- `results/part3_pairs.md`: paired bootstrap differences
- `results/part3_errors.md`: error analysis
- `figures/part3_ppl_vs_f1.png`, `figures/part3_confusion.png`

## Reproduce

From the repository root, after `bash part_1/run_part1.sh` and `bash part_2/run_part2.sh`:

```bash
pip install -r requirements.txt    # adds scikit-learn
bash part_3/run_part3.sh
```

For a quick check of the pipeline, use `python part_3/extract.py --limit 200`. Then delete `part_3/feats/`
before the real run.

**Seeds:** `CLS_SEED = 0` for split sampling, 42 for torch in extraction, 0 for LR and bootstrap.

**Hardware:** Apple M4 (10-core CPU and GPU, 16 GB RAM). Feature extraction runs on MPS and the classifiers on
the CPU.

## Limitations

- Sentences are classified **without their position or neighbouring sentences**, although the role of a
  sentence depends heavily on where it sits in the abstract. This is a deliberate choice, so the comparison
  isolates what each LM encodes about a single sentence. Absolute scores are therefore below context-aware
  systems (Dernoncourt & Lee, 2017).
- The LMs are frozen and the classifier is linear, so the results measure what is *linearly accessible* in each
  representation. They do not measure what the LM could learn with fine-tuning.
