"""Part II setting C: align public GloVe vectors (6B tokens, Wikipedia 2014 + Gigaword 5, 100d, uncased;
Pennington et al., 2014) to the LM vocabulary and record coverage.

The corpus replaces every digit with '@' and has many hyphenated clinical compounds, which GloVe lacks.
Each vocab word is initialised by the first rule that applies (the rule is recorded per word):
  1. exact      the word is in GloVe
  2. digit      '@' -> '0' (e.g. '@' -> '0', '@-week' -> '0-week') and the result is in GloVe
  3. alias      Penn-Treebank bracket tokens ('-lsb-' -> '['), or '@' dropped where it is glued to letters
                ('@months' -> 'months', 'cd@' -> 'cd')
  4. compound   split on - / + : , ; = ( ) (after '@' -> '0'); mean of the parts that are in GloVe
  5. random     N(mu_d, sigma_d^2) with GloVe's per-dimension mean/std, seeded
Special tokens: <pad> = zeros (never an input), <unk> = mean of the exact-match vectors, <eos> = random as in
rule 5. Not '.': every sentence already ends with '.', which <eos> follows, so the two inputs would be
identical. The entire table, including rules 2-4, is kept frozen during LM training.

GloVe source: gensim-data 'glove-wiki-gigaword-100' (the nlp.stanford.edu glove.6B.100d.txt vectors with a
word2vec header line; 128 MB). Download once (see README):
    curl -L -o emb/glove-wiki-gigaword-100.gz \
      https://github.com/RaRe-Technologies/gensim-data/releases/download/glove-wiki-gigaword-100/glove-wiki-gigaword-100.gz
--glove also accepts the Stanford glove.6B.100d.txt (no header line).

Output: emb/glove100.npy, emb/glove100_coverage.json
"""
import argparse
import json
import os
import re
import sys
import time
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "part_1"))
from data import EOS, PAD, UNK, load_corpus  # noqa: E402

RULES = ["exact", "digit", "alias", "compound", "random"]
PTB = {"-lrb-": "(", "-rrb-": ")", "-lsb-": "[", "-rsb-": "]", "-lcb-": "{", "-rcb-": "}"}
SPLIT = re.compile(r"[-/+:,;=()]+")


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--glove", default=os.path.join(HERE, "emb", "glove-wiki-gigaword-100.gz"),
                   help="GloVe vectors in text format (.gz or plain; with or without a word2vec header)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=os.path.join(HERE, "emb", "glove100.npy"))
    return p.parse_args()


def load_glove(path):
    import gzip

    from gensim.models import KeyedVectors
    with (gzip.open(path, "rt") if path.endswith(".gz") else open(path)) as f:
        has_header = len(f.readline().split()) == 2  # word2vec header: "<count> <dim>"
    return KeyedVectors.load_word2vec_format(path, binary=False, no_header=not has_header), os.path.basename(path)


def lookup(w, kv):
    """Return (vector, rule) for rules 1-4, or (None, 'random')."""
    if w in kv.key_to_index:
        return kv[w], "exact"
    w0 = w.replace("@", "0")
    if w0 != w and w0 in kv.key_to_index:
        return kv[w0], "digit"
    alias = PTB.get(w, w.replace("@", "") if re.fullmatch(r"[a-z@]+", w) else None)
    if alias and alias != w and alias in kv.key_to_index:
        return kv[alias], "alias"
    parts = [p for p in SPLIT.split(w0) if p]
    hits = [kv[p] for p in parts if p in kv.key_to_index]
    if len(parts) > 1 and hits:
        return np.mean(hits, axis=0), "compound"
    return None, "random"


def main():
    args = parse()
    vocab, ids = load_corpus()
    t0 = time.perf_counter()
    kv, source = load_glove(args.glove)
    t_load = time.perf_counter() - t0
    G = kv.vectors
    dim = G.shape[1]
    mu, sd = G.mean(0), G.std(0)
    rng = np.random.default_rng(args.seed)

    E = np.zeros((len(vocab), dim), dtype=np.float32)
    rule = {}
    for i, w in enumerate(vocab.itos):
        if w in (PAD, UNK, EOS):
            continue
        v, r = lookup(w, kv)
        rule[w] = r
        E[i] = v if v is not None else rng.normal(mu, sd)
    exact_rows = [vocab.stoi[w] for w, r in rule.items() if r == "exact"]
    E[vocab.unk_id] = E[exact_rows].mean(0)
    E[vocab.eos_id] = rng.normal(mu, sd)
    rule.update({PAD: "special (zeros)", UNK: "special (mean of exact)", EOS: "special (random)"})

    # coverage by type and by token (token = frequency-weighted over the id streams the LM is scored on)
    by_type = Counter(rule.values())
    by_token = {}
    for split in ["train", "test"]:
        counts = np.bincount(ids[split].numpy(), minlength=len(vocab))
        c = Counter()
        for w, r in rule.items():
            c[r] += int(counts[vocab.stoi[w]])
        by_token[split] = {r: n / counts.sum() for r, n in c.items()}
    examples = {r: [w for w in vocab.itos if rule.get(w) == r][:25] for r in RULES[1:]}

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.save(args.out, E)
    norms = np.linalg.norm(E[[i for i in range(len(vocab)) if i != vocab.pad_id]], axis=1)
    report = dict(
        source=source, glove_vocab=len(kv.key_to_index), dim=dim, load_time_s=t_load, seed=args.seed,
        lm_vocab=len(vocab), coverage_by_type={k: by_type[k] for k in RULES} | {
            "special": sum(v for k, v in by_type.items() if k.startswith("special"))},
        coverage_by_token=by_token, examples=examples,
        random_init=f"N(mu_d, sd_d^2) per dimension from all {len(kv.key_to_index):,} GloVe vectors, seed {args.seed}",
        row_norm_mean=float(norms.mean()), row_norm_std=float(norms.std()), value_std=float(E.std()),
        rule_per_word=rule)
    with open(args.out.replace(".npy", "_coverage.json"), "w") as f:
        json.dump(report, f, indent=1)

    V = len(vocab) - 3
    print(f"GloVe ({source}, {len(kv.key_to_index):,} words) -> LM vocab {len(vocab):,}")
    for r in RULES:
        print(f"  {r:>8}: {by_type[r]:>6,} types ({by_type[r] / V:6.2%})  | tokens: train "
              f"{by_token['train'].get(r, 0):6.2%}, test {by_token['test'].get(r, 0):6.2%}")
    print(f"  random examples: {examples['random'][:12]}")
    print(f"saved {args.out}  mean row norm {norms.mean():.2f}")


if __name__ == "__main__":
    main()
