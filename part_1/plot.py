"""Training/validation loss curves for the neural language models (results/history_*.json)."""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
TAGS = sys.argv[1:] or ["rnn", "lstm", "transformer"]
COLORS = {"rnn": "#d1495b", "lstm": "#2e86ab", "transformer": "#3b8c4a"}


def main():
    hists = {}
    for t in TAGS:
        p = os.path.join(RES, f"history_{t}.json")
        if os.path.exists(p):
            with open(p) as f:
                hists[t] = json.load(f)
    ngram = None
    rp = os.path.join(RES, "part1_results.json")
    if os.path.exists(rp):
        with open(rp) as f:
            ngram = json.load(f).get("trigram_kn")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for t, h in hists.items():
        c = COLORS.get(t)
        axes[0].plot(h["epoch"], h["train_loss"], "--o", color=c, ms=3, label=f"{t} train")
        axes[0].plot(h["epoch"], h["val_loss"], "-o", color=c, ms=3, label=f"{t} val")
        s = np.array(h["step_loss"])
        steps_per_epoch = (s[-1, 0] + 50) / len(h["epoch"])
        axes[1].plot(s[:, 0] / steps_per_epoch, s[:, 1], color=c, lw=0.8, alpha=0.8, label=t)
    if ngram:
        axes[0].axhline(ngram["test_nll"], color="grey", ls=":", label="trigram KN (test)")
    axes[0].set(xlabel="epoch", ylabel="cross-entropy (nats/token)", title="Epoch-mean train vs validation loss")
    axes[1].set(xlabel="epoch", ylabel="cross-entropy (nats/token)", title="Training loss (every 50 steps)",
                ylim=(None, 8))
    for a in axes:
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    out = os.path.join(HERE, "figures", "part1_loss_curves.png")
    fig.savefig(out, dpi=200)
    print("saved", out)


if __name__ == "__main__":
    main()
