"""Train a neural language model (rnn | lstm | transformer) on the PubMed 20k RCT subset (see data.py).

Every model sees exactly the same training batches in the same order: the training stream is split
into `batch_size` contiguous columns and consumed in chunks of `bptt` tokens. The RNN/LSTM carry their
hidden state across chunks (truncated BPTT); the Transformer treats each chunk as an independent
`bptt`-token context window.

Example:
    python train.py --model lstm
    python train.py --model lstm --emb ../part_2/emb/glove100.npy --freeze_emb --tag lstm_glove   # Part II
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from data import batchify, get_device, load_corpus, set_seed, sync
from lm_eval import neural_nll
from models import build_model

HERE = os.path.dirname(os.path.abspath(__file__))

DEFAULTS = {  # shared hyperparameters (document these in the report)
    "rnn": dict(emb_dim=100, hidden=256, layers=2, dropout=0.3),
    "lstm": dict(emb_dim=100, hidden=256, layers=2, dropout=0.3),
    "transformer": dict(emb_dim=100, d_model=256, layers=4, heads=4, ff=1024, dropout=0.2, context=128),
}


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=list(DEFAULTS))
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--bptt", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup", type=int, default=300, help="linear warm-up steps before cosine decay")
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--emb", default=None, help="Part II: .npy (V x emb_dim) input embedding matrix")
    p.add_argument("--freeze_emb", action="store_true", help="Part II: keep the input embedding table fixed")
    p.add_argument("--tag", default=None, help="run name (default: model name)")
    p.add_argument("--device", default=None)
    p.add_argument("--max_steps", type=int, default=None, help="debug: stop each epoch early")
    return p.parse_args()


def main():
    args = parse()
    tag = args.tag or args.model
    set_seed(args.seed)
    device = get_device(args.device)
    vocab, ids = load_corpus()

    cfg = DEFAULTS[args.model]
    model = build_model(args.model, len(vocab), vocab.pad_id, **cfg)
    if args.emb:
        model.set_input_embeddings(np.load(args.emb), freeze=args.freeze_emb)
    elif args.freeze_emb:  # Part II control: keep the random initial table fixed
        model.embed.weight.requires_grad_(False)
    model.to(device)
    print(f"[{tag}] device={device} params={model.n_params():,} trainable={model.n_params(True):,}")

    train = batchify(ids["train"], args.batch_size)  # (steps, B)
    n_chunks = math.ceil((train.size(0) - 1) / args.bptt)
    total_steps = n_chunks * args.epochs
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / args.warmup) *
                                              0.5 * (1 + math.cos(math.pi * min(s, total_steps) / total_steps)))

    ckpt_dir, res_dir = os.path.join(HERE, "checkpoints"), os.path.join(HERE, "results")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)
    hist = {"tag": tag, "model": args.model, "config": cfg, "args": vars(args), "params": model.n_params(),
            "trainable_params": model.n_params(True), "device": str(device),
            "epoch": [], "train_loss": [], "val_loss": [], "epoch_time_s": [], "step_loss": [],
            "grad_norm": [], "clip_frac": []}
    best_val, train_time = float("inf"), 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        sync(device)
        t0 = time.perf_counter()
        state, tot, n, n_clipped = None, 0.0, 0, 0
        for step, i in enumerate(range(0, train.size(0) - 1, args.bptt)):
            if args.max_steps and step >= args.max_steps:
                break
            seq = min(args.bptt, train.size(0) - 1 - i)
            x = train[i:i + seq].t().to(device)          # (B, T) inputs
            y = train[i + 1:i + 1 + seq].t().to(device)  # (B, T) next-token targets
            state = model.detach_state(state)
            logits, state = model(x, state)
            loss = F.cross_entropy(logits.flatten(0, 1), y.flatten(), ignore_index=vocab.pad_id)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip).item()  # pre-clip norm
            n_clipped += gnorm > args.clip
            opt.step()
            sched.step()
            tot, n = tot + loss.item(), n + 1
            if step % 50 == 0:
                hist["step_loss"].append([(epoch - 1) * n_chunks + step, loss.item()])
                hist["grad_norm"].append([(epoch - 1) * n_chunks + step, gnorm])
        sync(device)
        ep_time = time.perf_counter() - t0  # training time only (validation excluded)
        train_time += ep_time

        val, _ = neural_nll(model, ids["validation"], vocab.pad_id, device, bptt=args.bptt)
        hist["epoch"].append(epoch)
        hist["train_loss"].append(tot / n)
        hist["val_loss"].append(val)
        hist["epoch_time_s"].append(ep_time)
        hist["clip_frac"].append(n_clipped / n)  # fraction of steps whose gradient norm was clipped
        flag = ""
        if val < best_val:  # model selection on VALIDATION only
            best_val, flag = val, " *"
            torch.save({"state_dict": model.state_dict(), "model": args.model, "config": cfg,
                        "epoch": epoch}, os.path.join(ckpt_dir, f"{tag}.pt"))
        print(f"[{tag}] epoch {epoch:2d} train {tot / n:.3f} (ppl {math.exp(tot / n):7.1f}) | "
              f"val {val:.3f} (ppl {math.exp(val):7.1f}) | {ep_time:5.1f}s{flag}", flush=True)

        # save the history after every epoch so curves survive an interrupted run
        hist.update(best_val_loss=best_val, best_epoch=hist["epoch"][int(np.argmin(hist["val_loss"]))],
                    train_time_s=train_time, completed=epoch == args.epochs)
        with open(os.path.join(res_dir, f"history_{tag}.json"), "w") as f:
            json.dump(hist, f, indent=1)

    print(f"[{tag}] done: best val ppl {math.exp(best_val):.1f} at epoch {hist['best_epoch']}, "
          f"train time {train_time / 60:.1f} min")


if __name__ == "__main__":
    main()
