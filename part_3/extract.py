"""Part III: turn every labelled sentence into fixed-length vectors with a FROZEN language model.

Input: `<eos> w_1 ... w_n` (truncated to 128 tokens). Every training sentence followed an <eos> in the LM
stream, so the leading <eos> plays the role of a start token. Each sentence is encoded on its own (fresh
recurrent state; for the Transformer, positions start at 0). Models run in eval() mode (no dropout) under
no_grad. Sentences are right-padded within a batch. All models are causal, so the states at real positions do not
depend on padding.

Hidden states h_t come from the top layer: the top RNN/LSTM layer output, or the Transformer's final
LayerNorm output (the input to the vocabulary projection). From these we build:
  last  h at the last real token (each model reads left to right, so this state has seen the whole sentence)
  mean  average of h over the real tokens w_1..w_n (excludes the leading <eos> and padding)
  emb   average of the model's INPUT embedding rows over w_1..w_n (no context; a bag-of-vectors baseline)

Output: feats/<tag>.npz with arrays <split>_<pool> (float16) + y_<split>; timing in feats/<tag>_meta.json.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
P1 = os.path.join(HERE, "..", "part_1")
sys.path.insert(0, P1)
from data import get_device, load_corpus, set_seed, sync  # noqa: E402
from evaluate import load_neural  # noqa: E402
from data_cls import load_cls_data  # noqa: E402

FEATS = os.path.join(HERE, "feats")
TAGS = ["rnn", "lstm", "transformer", "lstm_w2v", "lstm_glove", "lstm_randfrozen", "lstm_glove_ft"]
MAX_LEN = 128  # Transformer context window (also used for the recurrent models, so inputs are identical)


@torch.no_grad()
def hidden_states(model, x):
    """Top-layer states h[B,T,H] (forward() without the output projection)."""
    if model.is_recurrent:
        h, _ = model.rnn(model.embed(x))
        return h
    T = x.size(1)
    h = model.proj(model.embed(x)) + model.pos(torch.arange(T, device=x.device))
    h = model.blocks(h, mask=model.causal_mask[:T, :T], is_causal=True)
    return model.norm(h)


@torch.no_grad()
def encode(model, rows, vocab, device, bsz=256):
    seqs = [[vocab.eos_id] + vocab.encode(r["tokens"])[: MAX_LEN - 1] for r in rows]
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))  # length-sorted batches = little padding
    out = {p: [None] * len(seqs) for p in ("last", "mean", "emb")}
    for s in range(0, len(order), bsz):
        idx = order[s: s + bsz]
        lens = torch.tensor([len(seqs[i]) for i in idx], device=device)
        x = torch.full((len(idx), int(lens.max())), vocab.pad_id, dtype=torch.long)
        for j, i in enumerate(idx):
            x[j, : len(seqs[i])] = torch.tensor(seqs[i])
        x = x.to(device)
        h = hidden_states(model, x)
        pos = torch.arange(x.size(1), device=device)
        word = ((pos[None] >= 1) & (pos[None] < lens[:, None])).float()  # real words, not <eos>/padding
        if (lens == 1).any():  # an empty sentence would only contain <eos>; fall back to it
            word[lens == 1, 0] = 1.0
        n = word.sum(1, keepdim=True)
        pooled = {
            "last": h[torch.arange(len(idx), device=device), lens - 1],
            "mean": (h * word[..., None]).sum(1) / n,
            "emb": (model.embed(x) * word[..., None]).sum(1) / n,
        }
        for p, v in pooled.items():
            v = v.float().cpu().numpy()
            for j, i in enumerate(idx):
                out[p][i] = v[j]
    return {p: np.stack(v) for p, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=TAGS, help="checkpoint tags; missing checkpoints are skipped")
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=None, help="debug: only the first N sentences per split")
    args = ap.parse_args()

    set_seed(42)
    device = get_device(args.device)
    vocab, _ = load_corpus()
    data = load_cls_data()
    if args.limit:
        data = {s: rows[: args.limit] for s, rows in data.items()}
    os.makedirs(FEATS, exist_ok=True)
    for tag in args.tags:
        if not os.path.exists(os.path.join(P1, "checkpoints", f"{tag}.pt")):
            print(f"[{tag}] no checkpoint, skipped")
            continue
        model = load_neural(tag, vocab, device)
        arrays, meta = {}, {"device": str(device), "n_sentences": {}, "time_s": {}}
        for split, rows in data.items():
            sync(device)
            t0 = time.perf_counter()
            f = encode(model, rows, vocab, device)
            sync(device)
            meta["time_s"][split] = time.perf_counter() - t0
            meta["n_sentences"][split] = len(rows)
            for p, v in f.items():
                arrays[f"{split}_{p}"] = v.astype(np.float16)
            arrays[f"y_{split}"] = np.array([r["label"] for r in rows], dtype=np.int64)
        np.savez(os.path.join(FEATS, f"{tag}.npz"), **arrays)
        meta["dims"] = {p: int(arrays[f"train_{p}"].shape[1]) for p in ("last", "mean", "emb")}
        with open(os.path.join(FEATS, f"{tag}_meta.json"), "w") as fh:
            json.dump(meta, fh, indent=1)
        tot = sum(meta["n_sentences"].values()) / sum(meta["time_s"].values())
        print(f"[{tag}] dims {meta['dims']} | {tot:,.0f} sentences/s on {device}")


if __name__ == "__main__":
    main()
