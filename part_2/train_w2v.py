"""Part II setting B: train our own Skip-gram embeddings on the LM training split, aligned to the LM vocab.

Input text is the exact training id stream the language models see (part_1/cache/ids.pt), split back into
sentences at <eos>, so rare words appear as <unk> exactly as in LM training and every vocab type has a vector.
Validation/test text is never used. <eos> is kept as the last token of each sentence so it gets a vector too.

Output: emb/w2v100.npy (V x 100, row i = vocab.itos[i], <pad> row = 0) and emb/w2v100_meta.json (cost,
hyperparameters, coverage). Reproducible with workers=1 and PYTHONHASHSEED=0 (gensim seeds each word's
initial vector with Python's string hash).

Example:
    PYTHONHASHSEED=0 python train_w2v.py
"""
import argparse
import json
import os
import platform
import sys
import time

import numpy as np
from gensim.models import Word2Vec

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "part_1"))
from data import PAD, load_corpus  # noqa: E402


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--dim", type=int, default=100, help="must equal the LM emb_dim (and the GloVe dim)")
    p.add_argument("--window", type=int, default=5)
    p.add_argument("--negative", type=int, default=5)
    p.add_argument("--sample", type=float, default=1e-3, help="frequent-word downsampling threshold")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=1, help="1 = deterministic")
    p.add_argument("--out", default=os.path.join(HERE, "emb", "w2v100.npy"))
    return p.parse_args()


def train_sentences(vocab, ids):
    """Split the training id stream at <eos> into token lists, keeping <eos> at the end of each sentence."""
    words, sents, cur = vocab.itos, [], []
    for i in ids.tolist():
        cur.append(words[i])
        if i == vocab.eos_id:
            sents.append(cur)
            cur = []
    if cur:
        sents.append(cur)
    return sents


def main():
    args = parse()
    if os.environ.get("PYTHONHASHSEED") != "0":
        print("warning: PYTHONHASHSEED is not 0, so the vectors are not bit-for-bit reproducible")
    vocab, ids = load_corpus()
    sents = train_sentences(vocab, ids["train"])
    n_tokens = sum(len(s) for s in sents)
    print(f"{len(sents):,} training sentences, {n_tokens:,} tokens")

    # min_count=1: the stream is already restricted to the LM vocab (train freq >= 3, others -> <unk>)
    model = Word2Vec(vector_size=args.dim, window=args.window, min_count=1, sg=1, negative=args.negative,
                     sample=args.sample, seed=args.seed, workers=args.workers)
    t0 = time.perf_counter()
    model.build_vocab(sents)
    t_vocab = time.perf_counter() - t0
    t0 = time.perf_counter()
    model.train(sents, total_examples=len(sents), epochs=args.epochs)
    t_train = time.perf_counter() - t0
    print(f"skip-gram: build_vocab {t_vocab:.1f}s, train {t_train:.1f}s ({args.epochs} epochs, "
          f"{args.workers} worker)")

    E = np.zeros((len(vocab), args.dim), dtype=np.float32)
    missing = []
    for i, w in enumerate(vocab.itos):
        if w == PAD:
            continue  # <pad> never appears in the stream; keep its row at zero
        if w in model.wv.key_to_index:
            E[i] = model.wv[w]
        else:
            missing.append(w)
    assert not missing, f"vocab words without a skip-gram vector: {missing[:10]}"

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.save(args.out, E)
    norms = np.linalg.norm(E[1:], axis=1)
    meta = dict(
        method="word2vec skip-gram with negative sampling (gensim)", gensim_version=__import__("gensim").__version__,
        hyperparameters=vars(args) | {"min_count": 1, "sg": 1}, corpus="LM training split only",
        train_sentences=len(sents), train_tokens=n_tokens, vocab_size=len(vocab),
        coverage_types=f"{len(vocab) - 1}/{len(vocab)} (all except <pad>, which is set to zero)",
        build_vocab_time_s=t_vocab, train_time_s=t_train, total_time_s=t_vocab + t_train,
        hardware=f"{platform.machine()} CPU ({platform.platform()}), {args.workers} worker thread(s)",
        pythonhashseed=os.environ.get("PYTHONHASHSEED"),
        row_norm_mean=float(norms.mean()), row_norm_std=float(norms.std()), value_std=float(E[1:].std()))
    with open(args.out.replace(".npy", "_meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"saved {args.out}  shape={E.shape}  mean row norm {norms.mean():.2f}")


if __name__ == "__main__":
    main()
