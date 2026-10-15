# DSA4213 Assignment 1: Language models, embeddings and representations

All three parts use one corpus (PubMed 20k RCT), one vocabulary and one test protocol, so their results can be
compared directly.

| Part | What it does | Details |
|---|---|---|
| I | Trains a Kneser–Ney trigram, an RNN, an LSTM and a Transformer LM, then compares them | [part_1/README.md](part_1/README.md) |
| II | Trains the LSTM with three input embeddings: random (trainable), own skip-gram (frozen), GloVe (frozen) | [part_2/README.md](part_2/README.md) |
| III | Freezes the LMs and uses their sentence vectors for rhetorical-role classification | [part_3/README.md](part_3/README.md) |

## Setup

Python 3.12. Run everything from the repository root.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Reproduce

Run the parts in this order, because each part reuses checkpoints from the previous ones:

```bash
bash part_1/run_part1.sh                   # ~50 min: data download, RNN/LSTM/Transformer training, evaluation
bash part_2/run_part2.sh                   # ~35 min: GloVe download, skip-gram, 2 LSTM runs, evaluation
CONTROLS=1 bash part_2/run_part2.sh        # optional: also the two 2x2 control runs (~25 min more)
bash part_3/run_part3.sh                   # feature extraction + classifiers
```

Runtimes were measured on the hardware below. Training runs one model at a time, so the reported training
times are not distorted by sharing the GPU.

- **Hardware:** Apple M4 (10-core CPU and GPU, 16 GB RAM).
  - LMs train on the PyTorch MPS backend.
  - The trigram, skip-gram and classifiers run on the CPU.
- **Seeds:**

| Purpose | Seed |
|---|---|
| LM abstract subset (`SUBSET_SEED`) | 0 |
| LM training (torch / numpy / random) | 42 |
| Skip-gram (gensim, 1 worker) | 42, with `PYTHONHASHSEED=0` (set by `run_part2.sh`) |
| GloVe fallback vectors | 42 |
| Classifier splits (`CLS_SEED`), logistic regression, bootstrap | 0 |
| Part III feature extraction | 42 |

Results are written to `part_*/results/` and figures to `part_*/figures/`. Model checkpoints
(`part_1/checkpoints/`), embedding files (`part_2/emb/`) and data caches are produced by the scripts and are not
included in the submission.

## Data and external resources

No data is redistributed in this repository. Everything is downloaded by the scripts.

- **PubMed 20k RCT.** Dernoncourt, F. & Lee, J. Y. (2017). *PubMed 200k RCT: a Dataset for Sequential Sentence
  Classification in Medical Abstracts.* IJCNLP. It is loaded from the Hugging Face Hub as `armanc/pubmed-rct20k`.
  The original release is at https://github.com/Franck-Dernoncourt/pubmed-rct. Use is subject to that source's
  terms.
- **GloVe 6B-100d.** Pennington, J., Socher, R. & Manning, C. D. (2014). *GloVe: Global Vectors for Word
  Representation.* EMNLP. It is downloaded from the gensim-data mirror (`glove-wiki-gigaword-100`).
- **Libraries.** PyTorch, gensim (Word2Vec), scikit-learn, Hugging Face `datasets`, numpy, pandas and
  matplotlib. Exact versions are pinned in [requirements.txt](requirements.txt).
