"""Part II evaluation: compare LSTM language models that differ only in their input-embedding table.

  A  lstm             random init, trainable      (the Part I LSTM, reused)
  B  lstm_w2v         own skip-gram, frozen
  C  lstm_glove       public GloVe 6B-100d, frozen
optional 2x2 controls (evaluated if their checkpoints exist):
     lstm_randfrozen  random init, frozen         (fixed table with no pre-trained information)
     lstm_glove_ft    GloVe init, trainable       (GloVe's extra data + trainability)

Everything else is identical (architecture, data, batches, optimiser, schedule, epochs, seed), and checkpoints
and training histories come from part_1/train.py. Uses the Part I test protocol, prompts and decoding settings.

Outputs (part_2/results/): part2_results.json, part2_table.md, part2_freq_buckets.md, part2_neighbours.md,
part2_generations.md; figure: part_2/figures/part2_curves.png
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
P1 = os.path.join(HERE, "..", "part_1")
sys.path.insert(0, P1)
from data import get_device, load_corpus, set_seed  # noqa: E402
from evaluate import GEN_LEN, PROMPTS, SAMPLE, SAMPLE_SEEDS, load_neural  # noqa: E402
from lm_eval import neural_generate, neural_nll, ppl, repetition_stats  # noqa: E402

RES, FIG, EMB = (os.path.join(HERE, d) for d in ("results", "figures", "emb"))
VARIANTS = {  # tag -> (label, embedding source, frozen)
    "lstm": ("A: random, trainable", "random", False),
    "lstm_w2v": ("B: own skip-gram, frozen", "skip-gram (train split)", True),
    "lstm_glove": ("C: GloVe, frozen", "GloVe 6B-100d", True),
    "lstm_randfrozen": ("random, frozen (control)", "random", True),
    "lstm_glove_ft": ("GloVe, trainable (control)", "GloVe 6B-100d", False),
}
COLORS = {"lstm": "#2a78d6", "lstm_w2v": "#eb6834", "lstm_glove": "#1baf7a",
          "lstm_randfrozen": "#eda100", "lstm_glove_ft": "#e87ba4"}
FREQ_BINS = [(0, 0, "<unk>"), (3, 10, "3-10"), (11, 100, "11-100"), (101, 1000, "101-1k"), (1001, None, ">1k")]
PROBES = ["placebo", "mg", "randomized", "efficacy", "pain", "insulin", "children", "surgery",
          "significantly", "@-week", "bupivacaine", "hrqol"]
VAL_PPL_TOL = 1.05  # convergence: first epoch whose val ppl is within 5% of the run's best


@torch.no_grad()
def token_nll(model, ids, device, bptt=128):
    """Per-target NLL for a recurrent LM, same protocol as lm_eval.neural_nll (batch 1, state carried)."""
    model.eval()
    x_all, y_all, out, state = ids[:-1], ids[1:], [], None
    for i in range(0, x_all.numel(), bptt):
        logits, state = model(x_all[i:i + bptt].unsqueeze(0).to(device), state)
        out.append(F.cross_entropy(logits[0], y_all[i:i + bptt].to(device), reduction="none").cpu())
    return torch.cat(out).numpy()


def freq_bucket(counts, unk_id):
    """Map each vocab id to a FREQ_BINS index by its training count (<unk> gets its own bucket)."""
    b = np.zeros(len(counts), dtype=int)
    for k, (lo, hi, _) in enumerate(FREQ_BINS[1:], start=1):
        b[(counts >= lo) & (counts <= (hi if hi else counts.max()))] = k
    b[unk_id] = 0
    return b


def stability(h):
    s = np.array(h["step_loss"])
    first = s[:, 0] >= s[-1, 0] / len(h["epoch"])  # skip epoch 1 (warm-up transient)
    out = dict(step_loss_jitter=float(np.std(np.diff(s[first, 1]))),
               val_increases=int(np.sum(np.diff(h["val_loss"]) > 0)),
               train_val_gap_final=h["val_loss"][-1] - h["train_loss"][-1])
    if h.get("grad_norm"):
        g = np.array(h["grad_norm"])[:, 1]
        out.update(grad_norm_median=float(np.median(g)), grad_norm_p95=float(np.percentile(g, 95)),
                   grad_norm_max=float(g.max()), clip_frac_mean=float(np.mean(h["clip_frac"])))
    return out


def convergence(h):
    v = np.exp(h["val_loss"])
    return dict(val_ppl_by_epoch=v.round(2).tolist(), val_ppl_epoch1=float(v[0]),
                epochs_to_within_5pct=int(np.argmax(v <= VAL_PPL_TOL * v.min())) + 1)


def neighbours(E, vocab, words, k=5):
    N = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    out = {}
    for w in words:
        if w not in vocab.stoi:
            continue
        sims = N @ N[vocab.stoi[w]]
        sims[[vocab.stoi[w], vocab.pad_id]] = -np.inf
        out[w] = [vocab.itos[i] for i in np.argsort(-sims)[:k]]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None)
    ap.add_argument("--no_plot", action="store_true")
    args = ap.parse_args()
    set_seed(42)
    device = get_device(args.device)
    vocab, ids = load_corpus()
    test = ids["test"]
    tags = [t for t in VARIANTS if os.path.exists(os.path.join(P1, "checkpoints", f"{t}.pt"))]
    print("evaluating:", tags)
    counts = np.bincount(ids["train"].numpy(), minlength=len(vocab))
    fb = freq_bucket(counts, vocab.unk_id)
    x_ids, y_ids = test[:-1].numpy(), test[1:].numpy()
    glove_rule = json.load(open(os.path.join(EMB, "glove100_coverage.json")))["rule_per_word"]
    rule_of_id = np.array([glove_rule.get(w, "special").split(" ")[0] for w in vocab.itos])

    prompts = [vocab.encode(p.split()) for p in PROMPTS]
    banned = [vocab.pad_id]
    results, gens, tables, buckets = {}, {}, {}, {}
    for tag in tags:
        model = load_neural(tag, vocab, device)
        h = json.load(open(os.path.join(P1, "results", f"history_{tag}.json")))
        nll = token_nll(model, test, device)
        ref, n = neural_nll(model, test, vocab.pad_id, device)
        assert n == len(nll) and abs(ref - nll.mean()) < 1e-3, (ref, nll.mean())  # same protocol as Part I

        buckets[tag] = {
            "target_freq": {FREQ_BINS[k][2]: float(nll[fb[y_ids] == k].mean()) for k in range(len(FREQ_BINS))},
            "input_freq": {FREQ_BINS[k][2]: float(nll[fb[x_ids] == k].mean()) for k in range(len(FREQ_BINS))},
            "input_glove_rule": {r: float(nll[rule_of_id[x_ids] == r].mean())
                                 for r in ["exact", "digit", "alias", "compound", "random", "special"]
                                 if (rule_of_id[x_ids] == r).any()}}
        E = model.embed.weight.detach().cpu().numpy()
        tables[tag] = E

        gens[tag] = {"greedy": [neural_generate(model, p, GEN_LEN, device, temperature=0, banned=banned)
                                for p in prompts],
                     "sample": [neural_generate(model, p, GEN_LEN, device, seed=s, banned=banned, **SAMPLE)
                                for p in prompts for s in SAMPLE_SEEDS]}
        label, source, frozen = VARIANTS[tag]
        results[tag] = dict(
            label=label, embedding=source, frozen=frozen, test_nll=float(nll.mean()), test_ppl=ppl(nll.mean()),
            val_ppl=ppl(h["best_val_loss"]), best_epoch=h["best_epoch"], params=h["params"],
            trainable_params=h["trainable_params"], train_time_s=h["train_time_s"],
            emb_row_norm_mean=float(np.linalg.norm(E[1:], axis=1).mean()),
            **convergence(h), **stability(h),
            gen_greedy=repetition_stats(gens[tag]["greedy"]), gen_sample=repetition_stats(gens[tag]["sample"]))
        print(f"{tag:>16}: test ppl {results[tag]['test_ppl']:.1f}")

    # sanity: frozen tables must be bit-identical to the matrices they were loaded from
    for tag, f in [("lstm_w2v", "w2v100.npy"), ("lstm_glove", "glove100.npy")]:
        if tag in tables:
            assert np.array_equal(tables[tag], np.load(os.path.join(EMB, f))), f"{tag} table changed in training"
            results[tag]["table_unchanged_after_training"] = True

    emb_cost = {}
    w2v_meta = os.path.join(EMB, "w2v100_meta.json")
    if os.path.exists(w2v_meta):
        m = json.load(open(w2v_meta))
        emb_cost["skip-gram"] = {k: m[k] for k in ("total_time_s", "train_tokens", "hardware", "hyperparameters")}
    g = json.load(open(os.path.join(EMB, "glove100_coverage.json")))
    emb_cost["glove"] = dict(our_cost="download + vocab alignment only (no training)",
                             pretraining_tokens=6_000_000_000, source=g["source"],
                             coverage_by_type=g["coverage_by_type"], coverage_by_token=g["coverage_by_token"])

    results["_setup"] = dict(test_targets=int(len(y_ids)), embedding_cost=emb_cost, freq_buckets=buckets,
                             convergence_rule=f"first epoch with val ppl <= {VAL_PPL_TOL} x best val ppl",
                             device=str(device))
    os.makedirs(RES, exist_ok=True)
    json.dump(results, open(os.path.join(RES, "part2_results.json"), "w"), indent=1)

    write_table(results, tags)
    write_buckets(buckets, tags, fb, rule_of_id, x_ids, y_ids)
    # neighbours: the learned table of each run (for frozen runs this is the loaded table itself)
    nb = {t: neighbours(tables[t], vocab, PROBES) for t in tags}
    write_neighbours(nb, tags)
    write_generations(gens, vocab, tags)
    if not args.no_plot:
        plot(tags)


def write_table(r, tags):
    rows = ["| Setting | Trainable params | Test PPL | Best val PPL (epoch) | Val PPL ep.1 | Epochs to ≤5% of best | "
            "LM train time (min) | Grad-norm median / p95 | Clipped steps | Final val−train gap | "
            "distinct-2 (sample) | repeat-3 (greedy) |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for t in tags:
        x = r[t]
        gn = f"{x['grad_norm_median']:.2f} / {x['grad_norm_p95']:.2f}" if "grad_norm_median" in x else "n/a"
        cf = f"{x['clip_frac_mean']:.0%}" if "clip_frac_mean" in x else "n/a"
        rows.append(f"| {x['label']} | {x['trainable_params'] / 1e6:.2f}M / {x['params'] / 1e6:.2f}M | "
                    f"{x['test_ppl']:.1f} | {x['val_ppl']:.1f} ({x['best_epoch']}) | {x['val_ppl_epoch1']:.1f} | "
                    f"{x['epochs_to_within_5pct']} | {x['train_time_s'] / 60:.1f} | {gn} | {cf} | "
                    f"{x['train_val_gap_final']:+.3f} | {x['gen_sample']['distinct_2']:.2f} | "
                    f"{x['gen_greedy']['repeat_3']:.2f} |")
    c = r["_setup"]["embedding_cost"]
    rows.append("")
    if "skip-gram" in c:
        rows.append(f"Own skip-gram embedding cost (separate from LM training): {c['skip-gram']['total_time_s']:.1f} s "
                    f"on {c['skip-gram']['hardware']}, {c['skip-gram']['train_tokens']:,} training tokens.")
    rows.append("GloVe: no training on our side; pre-trained by its authors on ~6B tokens "
                "(about 2,800x our 2.1M-token LM training split).")
    open(os.path.join(RES, "part2_table.md"), "w").write("\n".join(rows) + "\n")
    print("\n".join(rows))


def write_buckets(b, tags, fb, rule_of_id, x_ids, y_ids):
    lines = []
    for key, title, counter in [("target_freq", "Test PPL by training frequency of the TARGET word", fb[y_ids]),
                                ("input_freq", "Test PPL by training frequency of the INPUT (previous) word",
                                 fb[x_ids]),
                                ("input_glove_rule", "Test PPL by how the INPUT word was initialised in the GloVe "
                                 "table", rule_of_id[x_ids])]:
        cols = list(b[tags[0]][key])
        share = {c: (counter == (cols.index(c) if key != "input_glove_rule" else c)).mean() for c in cols}
        lines += [f"\n### {title}\n", "| Setting | " + " | ".join(f"{c} ({share[c]:.1%})" for c in cols) + " |",
                  "|---" * (len(cols) + 1) + "|"]
        for t in tags:
            lines.append(f"| {VARIANTS[t][0]} | " + " | ".join(f"{np.exp(b[t][key][c]):.1f}" for c in cols) + " |")
    lines.append("\nPercentages are the share of the 261,105 test targets in each bucket.")
    open(os.path.join(RES, "part2_freq_buckets.md"), "w").write("\n".join(lines) + "\n")


def write_neighbours(nb, tags):
    lines = ["# Nearest neighbours (cosine, within the LM vocabulary) in each input-embedding table\n",
             "| word | " + " | ".join(VARIANTS[t][0] for t in tags) + " |", "|---" * (len(tags) + 1) + "|"]
    for w in nb[tags[0]]:
        lines.append(f"| {w} | " + " | ".join(", ".join(nb[t][w]) for t in tags) + " |")
    open(os.path.join(RES, "part2_neighbours.md"), "w").write("\n".join(lines) + "\n")


def write_generations(gens, vocab, tags):
    lines = [f"# Part II continuations ({GEN_LEN} tokens)\n",
             f"Greedy, and sampling with temperature={SAMPLE['temperature']}, top-k={SAMPLE['top_k']}, seeds "
             f"{SAMPLE_SEEDS} (same settings as Part I).\n"]
    for i, p in enumerate(PROMPTS):
        lines.append(f"\n## Prompt {i + 1}: \"{p}\"\n")
        for t in tags:
            lines.append(f"**{VARIANTS[t][0]}** (greedy): {vocab.decode(gens[t]['greedy'][i])}\n")
            for j, s in enumerate(SAMPLE_SEEDS):
                lines.append(f"- sample seed {s}: {vocab.decode(gens[t]['sample'][i * len(SAMPLE_SEEDS) + j])}")
            lines.append("")
    open(os.path.join(RES, "part2_generations.md"), "w").write("\n".join(lines) + "\n")


def plot(tags):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hs = {t: json.load(open(os.path.join(P1, "results", f"history_{t}.json"))) for t in tags}
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for t, h in hs.items():
        c, lab = COLORS[t], VARIANTS[t][0]
        ax[0].plot(h["epoch"], h["train_loss"], "--", color=c, lw=2, marker="o", ms=4, alpha=0.7)
        ax[0].plot(h["epoch"], h["val_loss"], "-", color=c, lw=2, marker="o", ms=4, label=lab)
        s = np.array(h["step_loss"])
        spe = (s[-1, 0] + 50) / len(h["epoch"])
        ax[1].plot(s[:, 0] / spe, s[:, 1], color=c, lw=1, alpha=0.85, label=lab)
        if h.get("grad_norm"):
            g = np.array(h["grad_norm"])
            ax[2].plot(g[:, 0] / spe, g[:, 1], color=c, lw=1, alpha=0.85, label=lab)
    ax[2].axhline(1.0, color="#888888", ls=":", lw=1, label="clip threshold")
    ax[0].set(xlabel="epoch", ylabel="cross-entropy (nats/token)",
              title="Validation (solid) and train (dashed) loss")
    ax[1].set(xlabel="epoch", ylabel="cross-entropy (nats/token)", title="Training loss every 50 steps",
              ylim=(None, 7))
    ax[2].set(xlabel="epoch", ylabel="pre-clip gradient norm", title="Gradient norm every 50 steps", yscale="log")
    for a in ax:
        a.grid(alpha=0.25)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
        a.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, "part2_curves.png"), dpi=200)
    print("saved", os.path.join(FIG, "part2_curves.png"))


if __name__ == "__main__":
    main()
