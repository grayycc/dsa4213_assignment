"""Common evaluation + decoding for all language models.

Evaluation protocol (identical targets for every model): given a token stream x_0..x_{N-1},
score every target x_t for t = 1..N-1 conditioned only on tokens before it.
  * RNN/LSTM: one stream (batch size 1), chunks of `bptt` tokens, hidden state carried across chunks,
    so each prediction sees the full preceding history.
  * Transformer: sliding windows of `context` tokens with stride context/2; each target is scored once,
    in the window where it has at least context/2 preceding tokens (except at the very start).
  * n-gram: the two preceding tokens (see ngram.py).
Loss is summed with ignore_index=<pad>, so right-padding of the last (short) window never counts.
"""
import math

import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def neural_nll(model, ids, pad_id, device, bptt=128, win_batch=32):
    """Mean next-token NLL (nats) over targets ids[1:]. Returns (mean_nll, n_targets)."""
    model.eval()
    x_all, y_all = ids[:-1], ids[1:]
    M = x_all.numel()
    total, count = 0.0, 0

    if model.is_recurrent:
        state = None
        for i in range(0, M, bptt):
            x = x_all[i:i + bptt].unsqueeze(0).to(device)
            y = y_all[i:i + bptt].unsqueeze(0).to(device)
            logits, state = model(x, state)
            total += F.cross_entropy(logits.flatten(0, 1), y.flatten(), ignore_index=pad_id,
                                     reduction="sum").item()
            count += (y != pad_id).sum().item()
        return total / count, count

    L, stride = model.context, model.context // 2
    windows, pos = [], 0  # (begin, end, first_scored_index)
    while pos < M:
        end = min(M, max(L, pos + stride))
        windows.append((max(0, end - L), end, pos))
        pos = end
    for b in range(0, len(windows), win_batch):
        group = windows[b:b + win_batch]
        x = torch.full((len(group), L), pad_id, dtype=torch.long)
        y = torch.full((len(group), L), pad_id, dtype=torch.long)
        for j, (beg, end, first) in enumerate(group):
            n = end - beg
            x[j, :n] = x_all[beg:end]
            # only targets not already scored by a previous window; everything else -> pad (ignored)
            k = first - beg
            y[j, k:n] = y_all[first:end]
        logits, _ = model(x.to(device))
        y = y.to(device)
        total += F.cross_entropy(logits.flatten(0, 1), y.flatten(), ignore_index=pad_id,
                                 reduction="sum").item()
        count += (y != pad_id).sum().item()
    assert count == M, (count, M)
    return total / count, count


def sample_from_logits(logits, temperature, top_k, generator, banned=()):
    """Greedy if temperature == 0, otherwise temperature + top-k sampling. logits: 1-D tensor (CPU)."""
    logits = logits.clone().float()
    for b in banned:
        logits[b] = -float("inf")
    if temperature == 0:
        return int(logits.argmax())
    logits = logits / temperature
    if top_k:
        kth = torch.topk(logits, top_k).values[-1]
        logits[logits < kth] = -float("inf")
    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, 1, generator=generator))


@torch.no_grad()
def neural_generate(model, prompt_ids, n_new, device, temperature=0.8, top_k=40, seed=0, banned=()):
    model.eval()
    g = torch.Generator().manual_seed(seed)
    out = list(prompt_ids)
    if model.is_recurrent:
        logits, state = model(torch.tensor([out], device=device))
        for _ in range(n_new):
            nxt = sample_from_logits(logits[0, -1].cpu(), temperature, top_k, g, banned)
            out.append(nxt)
            logits, state = model(torch.tensor([[nxt]], device=device), state)
    else:
        for _ in range(n_new):
            ctx = torch.tensor([out[-model.context:]], device=device)
            logits, _ = model(ctx)
            out.append(sample_from_logits(logits[0, -1].cpu(), temperature, top_k, g, banned))
    return out[len(prompt_ids):]


def ngram_generate(lm, prompt_ids, n_new, eos_id, temperature=0.8, top_k=40, seed=0, banned=()):
    g = torch.Generator().manual_seed(seed)
    out = [eos_id] + list(prompt_ids)  # <eos> = paragraph boundary, used if the prompt is 1 token
    for _ in range(n_new):
        p = torch.from_numpy(lm.distribution(out[-2], out[-1]))
        out.append(sample_from_logits(torch.log(p), temperature, top_k, g, banned))
    return out[len(prompt_ids) + 1:]


def repetition_stats(token_lists):
    """distinct-1 / distinct-2 (unique n-grams / total n-grams) and fraction of repeated trigrams,
    averaged over generations. Higher distinct and lower repeat-3 = less repetitive."""
    d1, d2, r3 = [], [], []
    for t in token_lists:
        bi = list(zip(t, t[1:]))
        tri = list(zip(t, t[1:], t[2:]))
        d1.append(len(set(t)) / max(len(t), 1))
        d2.append(len(set(bi)) / max(len(bi), 1))
        r3.append(1 - len(set(tri)) / max(len(tri), 1))
    return {"distinct_1": float(np.mean(d1)), "distinct_2": float(np.mean(d2)), "repeat_3": float(np.mean(r3))}


def ppl(nll):
    return math.exp(nll)
