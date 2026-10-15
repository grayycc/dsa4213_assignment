"""Shared data pipeline: PubMed 20k RCT abstracts -> lowercased word tokens -> integer ids.

Corpus: armanc/pubmed-rct20k (Dernoncourt & Lee, 2017). Each row is one sentence of a randomised-
controlled-trial abstract, already tokenised, with digits replaced by '@' and a rhetorical-role label
(background / objective / methods / results / conclusions) that is NOT used for language modelling.

For language modelling we use a seeded random subset of abstracts from each official split
(official splits are disjoint by abstract). Each abstract's sentences are kept in order and every
sentence is followed by <eos>. Every model in Part I (and Parts II/III) uses this exact vocabulary and
these exact id streams, so test perplexities are computed over identical targets.
"""
import json
import os
import random
from collections import Counter

import numpy as np
import torch

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
DATASET = "armanc/pubmed-rct20k"
MIN_FREQ = 3
SUBSET_SEED = 0
N_ABSTRACTS = {"train": 6500, "validation": 700, "test": 800}  # of 15000 / 2500 / 2500
PAD, UNK, EOS = "<pad>", "<unk>", "<eos>"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device(name=None):
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync(device):
    """Wait for queued GPU work so wall-clock timings are accurate."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def tokenize(sentence):
    """The corpus is already tokenised (and digits are '@'); we only lowercase and split on spaces."""
    return sentence.lower().split()


def load_abstracts(split):
    """Return an ordered dict {abstract_id: [(tokens, label), ...]} for one official split."""
    from datasets import load_dataset

    ds = load_dataset(DATASET, split=split)
    abstracts = {}
    for aid, sid, text, label in zip(ds["abstract_id"], ds["sentence_id"], ds["text"], ds["label"]):
        abstracts.setdefault(aid, []).append((sid, tokenize(text), label))
    return {a: [(t, l) for _, t, l in sorted(s, key=lambda x: x[0])] for a, s in abstracts.items()}


def lm_abstract_ids():
    """Seeded random subset of abstract ids per split used for language modelling (cached to disk so
    Parts II/III can see exactly which abstracts the LMs were trained/evaluated on)."""
    path = os.path.join(DATA_DIR, "lm_abstract_ids.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    rng = random.Random(SUBSET_SEED)
    chosen = {s: sorted(rng.sample(sorted(load_abstracts(s)), n)) for s, n in N_ABSTRACTS.items()}
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(path, "w") as f:
        json.dump(chosen, f)
    return chosen


def load_tokens():
    """Return {'train'|'validation'|'test': list of token lists (one per sentence, abstracts in order)}."""
    chosen = lm_abstract_ids()
    out = {}
    for split in ["train", "validation", "test"]:
        abstracts = load_abstracts(split)
        out[split] = [toks for aid in chosen[split] for toks, _ in abstracts[aid]]
    return out


class Vocab:
    def __init__(self, itos):
        self.itos = itos
        self.stoi = {w: i for i, w in enumerate(itos)}
        self.pad_id, self.unk_id, self.eos_id = self.stoi[PAD], self.stoi[UNK], self.stoi[EOS]

    def __len__(self):
        return len(self.itos)

    def encode(self, words):
        return [self.stoi.get(w, self.unk_id) for w in words]

    def decode(self, ids):
        return " ".join(self.itos[i] for i in ids)

    @classmethod
    def build(cls, train_paragraphs, min_freq=MIN_FREQ):
        counts = Counter(w for p in train_paragraphs for w in p)
        words = sorted((w for w, c in counts.items() if c >= min_freq), key=lambda w: (-counts[w], w))
        return cls([PAD, UNK, EOS] + words)


def flatten(paragraphs, vocab):
    """Concatenate sentences into one id stream, appending <eos> after each sentence."""
    ids = []
    for p in paragraphs:
        ids.extend(vocab.encode(p))
        ids.append(vocab.eos_id)
    return torch.tensor(ids, dtype=torch.long)


def load_corpus():
    """Build (or load cached) vocab + id streams. Returns vocab, {split: LongTensor}."""
    os.makedirs(DATA_DIR, exist_ok=True)
    vpath, tpath = os.path.join(DATA_DIR, "vocab.json"), os.path.join(DATA_DIR, "ids.pt")
    if os.path.exists(vpath) and os.path.exists(tpath):
        with open(vpath) as f:
            vocab = Vocab(json.load(f))
        return vocab, torch.load(tpath)
    toks = load_tokens()
    vocab = Vocab.build(toks["train"])  # vocabulary from TRAIN only (no leakage)
    ids = {s: flatten(p, vocab) for s, p in toks.items()}
    with open(vpath, "w") as f:
        json.dump(vocab.itos, f)
    torch.save(ids, tpath)
    return vocab, ids


def batchify(ids, bsz):
    """Reshape a 1-D stream into (n_steps, bsz) columns of contiguous text (dropping the remainder)."""
    n = ids.size(0) // bsz
    return ids[: n * bsz].view(bsz, n).t().contiguous()


if __name__ == "__main__":
    vocab, ids = load_corpus()
    print(f"vocab size: {len(vocab)} (min_freq={MIN_FREQ})")
    for s, t in ids.items():
        unk = (t == vocab.unk_id).float().mean().item()
        print(f"{s:>10}: {t.numel():>9,} tokens, <unk> rate {unk:.2%}")
