"""Part I evaluation: n-gram training + common test evaluation, cost/timing, and generated text.

Outputs (in results/):
  part1_results.json   all numbers
  part1_table.md       summary table for the report
  generations.md       continuations from the same prompts with matched decoding settings
Run after train.py has produced checkpoints/{rnn,lstm,transformer}.pt.
"""
import argparse
import json
import os
import statistics
import time

import torch

from data import get_device, load_corpus, set_seed, sync
from lm_eval import neural_generate, neural_nll, ngram_generate, ppl, repetition_stats
from models import build_model
from ngram import KNTrigram

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
NEURAL = ["rnn", "lstm", "transformer"]
PROMPTS = [  # one per rhetorical role of an RCT abstract; all words are in-vocabulary
    "this study aimed to evaluate the",       # objective
    "patients were randomly assigned to",     # methods
    "the primary outcome was",                # methods
    "there was no significant difference in", # results
    "these findings suggest that",            # conclusions
]
GEN_LEN = 40
SAMPLE = dict(temperature=0.8, top_k=40)
SAMPLE_SEEDS = [0, 1, 2]
TIMING_RUNS = 3


def timed(fn, device, runs=TIMING_RUNS):
    """Median wall-clock seconds over `runs` calls (after one warm-up call made by the caller)."""
    times = []
    for _ in range(runs):
        sync(device)
        t0 = time.perf_counter()
        fn()
        sync(device)
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def load_neural(tag, vocab, device):
    ck = torch.load(os.path.join(HERE, "checkpoints", f"{tag}.pt"), map_location="cpu")
    model = build_model(ck["model"], len(vocab), vocab.pad_id, **ck["config"])
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=NEURAL, help="checkpoint tags to evaluate")
    ap.add_argument("--cpu_tokens", type=int, default=20000, help="test tokens for the CPU timing workload")
    args = ap.parse_args()

    set_seed(42)
    device, cpu = get_device(), torch.device("cpu")
    vocab, ids = load_corpus()
    test = ids["test"]
    n_targets = test.numel() - 1
    banned = [vocab.pad_id]
    prompts = [vocab.encode(p.split()) for p in PROMPTS]
    for p, enc in zip(PROMPTS, prompts):
        assert vocab.unk_id not in enc, f"prompt has OOV word: {p}"
    results, gens = {}, {}

    # ---------------- n-gram ----------------
    t0 = time.perf_counter()
    lm = KNTrigram(len(vocab)).fit(ids["train"].numpy())
    fit_time = time.perf_counter() - t0
    t0 = time.perf_counter()
    best_val, discounts = lm.tune(ids["validation"].numpy(), vocab.eos_id)
    tune_time = time.perf_counter() - t0
    test_np = test.numpy()
    lm.nll(test_np[:5000], vocab.eos_id)  # warm-up
    nll = lm.nll(test_np, vocab.eos_id)
    t_full = timed(lambda: lm.nll(test_np, vocab.eos_id), cpu)
    t_cpu = timed(lambda: lm.nll(test_np[:args.cpu_tokens + 1], vocab.eos_id), cpu)
    results["trigram_kn"] = dict(
        test_nll=nll, test_ppl=ppl(nll), val_ppl=ppl(best_val), discounts=discounts,
        params=int(lm.n_params()), param_note="stored n-gram statistics, not trainable weights",
        train_time_s=fit_time + tune_time, fit_time_s=fit_time, tune_time_s=tune_time,
        infer_device="cpu", infer_time_s=t_full, infer_tok_per_s=n_targets / t_full,
        cpu_tok_per_s=args.cpu_tokens / t_cpu)
    print(f"trigram-KN: test ppl {ppl(nll):.1f}  discounts={discounts}")

    gens["trigram_kn"] = {"greedy": [], "sample": []}
    t0 = time.perf_counter()
    for p in prompts:
        gens["trigram_kn"]["greedy"].append(
            ngram_generate(lm, p, GEN_LEN, vocab.eos_id, temperature=0, banned=banned))
    results["trigram_kn"]["gen_ms_per_token"] = 1000 * (time.perf_counter() - t0) / (len(prompts) * GEN_LEN)
    for p in prompts:
        for s in SAMPLE_SEEDS:
            gens["trigram_kn"]["sample"].append(
                ngram_generate(lm, p, GEN_LEN, vocab.eos_id, seed=s, banned=banned, **SAMPLE))

    # ---------------- neural ----------------
    for tag in args.models:
        model = load_neural(tag, vocab, device)
        with open(os.path.join(RES, f"history_{tag}.json")) as f:
            hist = json.load(f)
        neural_nll(model, test[:5001], vocab.pad_id, device)  # warm-up
        nll, count = neural_nll(model, test, vocab.pad_id, device)
        assert count == n_targets
        t_full = timed(lambda: neural_nll(model, test, vocab.pad_id, device), device)

        # generation latency (greedy, batch 1) on the main device
        gens[tag] = {"greedy": [], "sample": []}
        neural_generate(model, prompts[0], 5, device, temperature=0)  # warm-up
        sync(device)
        t0 = time.perf_counter()
        for p in prompts:
            gens[tag]["greedy"].append(neural_generate(model, p, GEN_LEN, device, temperature=0, banned=banned))
        sync(device)
        gen_ms = 1000 * (time.perf_counter() - t0) / (len(prompts) * GEN_LEN)
        for p in prompts:
            for s in SAMPLE_SEEDS:
                gens[tag]["sample"].append(neural_generate(model, p, GEN_LEN, device, seed=s, banned=banned,
                                                           **SAMPLE))

        # same hardware as the n-gram: CPU, first cpu_tokens test tokens
        model_cpu = load_neural(tag, vocab, cpu)
        sub = test[:args.cpu_tokens + 1]
        neural_nll(model_cpu, test[:2001], vocab.pad_id, cpu)  # warm-up
        t_cpu = timed(lambda: neural_nll(model_cpu, sub, vocab.pad_id, cpu), cpu)

        results[tag] = dict(
            test_nll=nll, test_ppl=ppl(nll), val_ppl=ppl(hist["best_val_loss"]), best_epoch=hist["best_epoch"],
            epochs=len(hist["epoch"]), params=hist["params"], train_time_s=hist["train_time_s"],
            train_s_per_epoch=hist["train_time_s"] / len(hist["epoch"]), infer_device=str(device),
            infer_time_s=t_full, infer_tok_per_s=n_targets / t_full, cpu_tok_per_s=args.cpu_tokens / t_cpu,
            gen_ms_per_token=gen_ms, config=hist["config"],
            final_train_loss=hist["train_loss"][-1], final_val_loss=hist["val_loss"][-1])
        print(f"{tag}: test ppl {ppl(nll):.1f}")

    # ---------------- repetition statistics ----------------
    for tag, g in gens.items():
        results[tag]["gen_greedy"] = repetition_stats(g["greedy"])
        results[tag]["gen_sample"] = repetition_stats(g["sample"])
        results[tag]["gen_sample_unk_rate"] = sum(t.count(vocab.unk_id) for t in g["sample"]) / sum(
            len(t) for t in g["sample"])

    results["_setup"] = dict(
        test_targets=n_targets, device=str(device), prompts=PROMPTS, gen_len=GEN_LEN, sampling=SAMPLE,
        sample_seeds=SAMPLE_SEEDS, timing=f"median of {TIMING_RUNS} runs after 1 warm-up; GPU synchronised",
        torch=torch.__version__)
    os.makedirs(RES, exist_ok=True)
    with open(os.path.join(RES, "part1_results.json"), "w") as f:
        json.dump(results, f, indent=1)
    write_table(results)
    write_generations(gens, vocab, prompts)


def write_table(r):
    names = {"trigram_kn": "Trigram (interp. KN)", "rnn": "RNN", "lstm": "LSTM", "transformer": "Transformer"}
    rows = ["| Model | Params | Test PPL | Val PPL | Train time (min) | Test-scoring tok/s | CPU tok/s | "
            "Gen ms/token | distinct-2 (sample) | repeat-3 (greedy) |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for k in [k for k in names if k in r]:
        x = r[k]
        rows.append(f"| {names[k]} | {x['params'] / 1e6:.2f}M | {x['test_ppl']:.1f} | {x['val_ppl']:.1f} | "
                    f"{x['train_time_s'] / 60:.1f} | {x['infer_tok_per_s']:,.0f} ({x['infer_device']}) | "
                    f"{x['cpu_tok_per_s']:,.0f} | {x['gen_ms_per_token']:.1f} | "
                    f"{x['gen_sample']['distinct_2']:.2f} | {x['gen_greedy']['repeat_3']:.2f} |")
    table = "\n".join(rows)
    with open(os.path.join(RES, "part1_table.md"), "w") as f:
        f.write(table + "\n")
    print(table)


def write_generations(gens, vocab, prompts):
    lines = [f"# Generated continuations ({GEN_LEN} tokens)\n",
             f"Greedy decoding, and sampling with temperature={SAMPLE['temperature']}, top-k={SAMPLE['top_k']}, "
             f"seeds {SAMPLE_SEEDS}. `<pad>` is never generated; `<eos>` marks a paragraph break.\n"]
    for i, p in enumerate(PROMPTS):
        lines.append(f"\n## Prompt {i + 1}: \"{p}\"\n")
        for tag, g in gens.items():
            lines.append(f"**{tag}** (greedy): {vocab.decode(g['greedy'][i])}\n")
            for j, s in enumerate(SAMPLE_SEEDS):
                lines.append(f"- {tag} (sample, seed {s}): {vocab.decode(g['sample'][i * len(SAMPLE_SEEDS) + j])}")
            lines.append("")
    with open(os.path.join(RES, "generations.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
