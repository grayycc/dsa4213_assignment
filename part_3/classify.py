"""Part III: train the SAME classifier on every frozen representation and compare them on held-out abstracts.

Classifier (identical for every representation): StandardScaler -> multinomial logistic regression (lbfgs).
C is chosen from C_GRID by classifier-validation macro-F1, then the model is scored once on the classifier test
set. TF-IDF uses the same LR and grid, but without scaling (sparse features).

Representations: <tag>:<pool> for every feats/<tag>.npz (pools last / mean / emb, see extract.py), plus baselines
majority-class and tfidf (word 1-2 grams fitted on classifier train only).

Metrics: macro-F1 (main; the classes are imbalanced), accuracy and per-class F1. Uncertainty: 95% percentile
bootstrap over test ABSTRACTS (sentences within one abstract are not independent). The same resamples are used
for every representation, so pairwise differences are paired.

Outputs: results/part3_results.json, part3_table.md, part3_pairs.md, part3_errors.md;
figures/part3_ppl_vs_f1.png, part3_confusion.png
"""
import argparse
import json
import os
import time
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
P1, P2 = os.path.join(HERE, "..", "part_1"), os.path.join(HERE, "..", "part_2")
from data_cls import LABELS, load_cls_data  # noqa: E402
from extract import FEATS, TAGS  # noqa: E402

RES, FIG = os.path.join(HERE, "results"), os.path.join(HERE, "figures")
C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
POOLS = ["last", "mean", "emb"]
N_BOOT, SEED = 1000, 0
K = len(LABELS)
GROUP = {"rnn": "A: model family", "lstm": "A: model family", "transformer": "A: model family",
         "lstm_w2v": "B: embedding variant", "lstm_glove": "B: embedding variant",
         "lstm_randfrozen": "B: embedding variant (control)", "lstm_glove_ft": "B: embedding variant (control)"}
LSTM_VARIANTS = ["lstm", "lstm_w2v", "lstm_glove", "lstm_randfrozen", "lstm_glove_ft"]
SERIES = {"last": "#2a78d6", "mean": "#eb6834"}  # categorical slots 1-2 (same palette as Part II)


def lm_ppl():
    """Test perplexity of each LM on the Part I protocol (Part I + Part II result files)."""
    out = {}
    for path in (os.path.join(P1, "results", "part1_results.json"), os.path.join(P2, "results", "part2_results.json")):
        if os.path.exists(path):
            for tag, r in json.load(open(path)).items():
                if isinstance(r, dict) and "test_ppl" in r and tag in TAGS:
                    out[tag] = r["test_ppl"]
    return out


def fit_lr(Xtr, ytr, Xva, yva, scale=True):
    best = None
    for C in C_GRID:
        lr = LogisticRegression(C=C, max_iter=3000, random_state=SEED)
        clf = make_pipeline(StandardScaler(), lr) if scale else lr
        clf.fit(Xtr, ytr)
        f = macro_f1(*counts(yva, clf.predict(Xva)))
        if best is None or f > best[1]:
            best = (C, f, clf)
    return best


def counts(y, p):
    tp = np.bincount(y[p == y], minlength=K)
    return tp, np.bincount(p, minlength=K) - tp, np.bincount(y, minlength=K) - tp


def macro_f1(tp, fp, fn, per_class=False):
    f = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros(np.shape(tp), float), where=(2 * tp + fp + fn) > 0)
    return f if per_class else f.mean(-1)


def per_abstract_counts(y, p, abs_idx, n_abs):
    """tp/fp/fn per (abstract, class) so bootstrap resamples are just weighted sums."""
    tp, fp, fn = (np.zeros((n_abs, K)) for _ in range(3))
    np.add.at(tp, (abs_idx, y), p == y)
    np.add.at(fp, (abs_idx, p), p != y)
    np.add.at(fn, (abs_idx, y), p != y)
    return tp, fp, fn


def confusion(y, p):
    m = np.zeros((K, K), int)
    np.add.at(m, (y, p), 1)
    return m


def spearman(a, b):
    ra, rb = (np.argsort(np.argsort(v)).astype(float) for v in (a, b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="debug: match `extract.py --limit N`")
    args = ap.parse_args()
    os.makedirs(RES, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    data = load_cls_data()
    if args.limit:
        data = {s: rows[: args.limit] for s, rows in data.items()}
    y = {s: np.array([r["label"] for r in rows]) for s, rows in data.items()}
    test = data["test"]
    abs_pos = {a: i for i, a in enumerate(sorted({r["abstract_id"] for r in test}))}
    abs_idx = np.array([abs_pos[r["abstract_id"]] for r in test])
    n_abs = len(abs_pos)
    rng = np.random.default_rng(SEED)
    W = np.stack([np.bincount(rng.integers(0, n_abs, n_abs), minlength=n_abs) for _ in range(N_BOOT)])

    reps = {}  # name -> dict(group, dim, train/val/test features or None)
    for tag in TAGS:
        path = os.path.join(FEATS, f"{tag}.npz")
        if not os.path.exists(path):
            continue
        z = np.load(path)
        assert all(np.array_equal(z[f"y_{s}"], y[s]) for s in y), f"{tag}: features out of sync with cls data"
        for pool in POOLS:
            reps[f"{tag}:{pool}"] = dict(group=GROUP[tag] if pool != "emb" else "C: input embeddings only",
                                         tag=tag, pool=pool,
                                         X={s: z[f"{s}_{pool}"].astype(np.float32) for s in y})

    results, preds = {}, {}
    # baselines
    maj = Counter(y["train"].tolist()).most_common(1)[0][0]
    preds["majority"] = np.full_like(y["test"], maj)
    results["majority"] = dict(group="baseline", dim=0, C=None)
    t0 = time.perf_counter()
    tfidf = TfidfVectorizer(analyzer=lambda toks: toks + [f"{a} {b}" for a, b in zip(toks, toks[1:])],
                            min_df=2, sublinear_tf=True)
    Xs = {"train": tfidf.fit_transform([r["tokens"] for r in data["train"]])}
    for s in ("validation", "test"):
        Xs[s] = tfidf.transform([r["tokens"] for r in data[s]])
    C, fva, clf = fit_lr(Xs["train"], y["train"], Xs["validation"], y["validation"], scale=False)
    preds["tfidf"] = clf.predict(Xs["test"])
    results["tfidf"] = dict(group="baseline", dim=Xs["train"].shape[1], C=C, val_macro_f1=fva,
                            fit_time_s=time.perf_counter() - t0)
    print(f"{'tfidf':>22}: val {fva:.3f} (C={C})")

    for name, r in reps.items():
        t0 = time.perf_counter()
        C, fva, clf = fit_lr(r["X"]["train"], y["train"], r["X"]["validation"], y["validation"])
        preds[name] = clf.predict(r["X"]["test"])
        results[name] = dict(group=r["group"], tag=r["tag"], pool=r["pool"], dim=r["X"]["train"].shape[1], C=C,
                             val_macro_f1=fva, fit_time_s=time.perf_counter() - t0)
        print(f"{name:>22}: val {fva:.3f} (C={C})")

    # test metrics + bootstrap
    boot = {}
    for name, p in preds.items():
        tp, fp, fn = counts(y["test"], p)
        a_tp, a_fp, a_fn = per_abstract_counts(y["test"], p, abs_idx, n_abs)
        boot[name] = macro_f1(W @ a_tp, W @ a_fp, W @ a_fn)
        results[name].update(
            test_macro_f1=float(macro_f1(tp, fp, fn)), test_acc=float((p == y["test"]).mean()),
            test_f1_ci95=[float(v) for v in np.percentile(boot[name], [2.5, 97.5])],
            per_class_f1=dict(zip(LABELS, macro_f1(tp, fp, fn, per_class=True).round(4).tolist())),
            confusion=confusion(y["test"], p).tolist())

    # LM ranking vs downstream ranking
    ppl = lm_ppl()
    for name, r in results.items():
        if r.get("tag") in ppl:
            r["lm_test_ppl"] = ppl[r["tag"]]
    ranking = {}
    for pool in ("last", "mean"):
        tags = [t for t in TAGS if t in ppl and f"{t}:{pool}" in results]
        if len(tags) >= 3:
            ranking[pool] = dict(tags=tags, spearman_negppl_vs_f1=spearman(
                [-ppl[t] for t in tags], [results[f"{t}:{pool}"]["test_macro_f1"] for t in tags]))

    pairs = paired_tests(results, boot)
    json.dump(dict(results=results, ranking=ranking, pairs=pairs,
                   setup=dict(C_grid=C_GRID, n_boot=N_BOOT, seed=SEED, labels=LABELS,
                              n_sentences={s: len(v) for s, v in y.items()},
                              n_abstracts={s: len({r["abstract_id"] for r in rows}) for s, rows in data.items()})),
              open(os.path.join(RES, "part3_results.json"), "w"), indent=1)
    write_table(results, ranking)
    write_pairs(pairs)
    write_errors(results, preds, test, y["test"])
    plot(results, ppl)


def paired_tests(results, boot):
    cand = [("transformer:mean", "lstm:mean"), ("transformer:mean", "rnn:mean"), ("rnn:mean", "lstm:mean"),
            ("transformer:last", "lstm:last"), ("transformer:last", "rnn:last"), ("rnn:last", "lstm:last")]
    cand += [(f"{t}:mean", f"{t}:last") for t in TAGS] + [(f"{t}:mean", f"{t}:emb") for t in TAGS]
    cand += [(f"{t}:mean", "lstm:mean") for t in LSTM_VARIANTS[1:]] + [("lstm_glove:mean", "lstm_w2v:mean"), ("lstm_w2v:mean", "transformer:mean")]
    cand += [(f"{t}:mean", "tfidf") for t in ("rnn", "lstm", "transformer")]
    out = []
    for a, b in cand:
        if a in boot and b in boot:
            d = boot[a] - boot[b]
            out.append(dict(a=a, b=b, diff=results[a]["test_macro_f1"] - results[b]["test_macro_f1"],
                            ci95=[float(v) for v in np.percentile(d, [2.5, 97.5])],
                            p_two_sided=float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))))
    return out


def write_table(results, ranking):
    lines = ["| Representation | Group | Dim | C | Val macro-F1 | Test macro-F1 [95% CI] | Test acc | "
             + " | ".join(f"F1 {l[:4]}" for l in LABELS) + " | LM test PPL |",
             "|---|---|---|---|---|---|---|" + "---|" * K + "---|"]
    order = sorted(results, key=lambda n: (results[n]["group"], n))
    for n in order:
        r = results[n]
        lo, hi = r["test_f1_ci95"]
        C = r["C"] if r["C"] is not None else "-"
        val = f"{r['val_macro_f1']:.3f}" if "val_macro_f1" in r else "-"
        ppl = f"{r['lm_test_ppl']:.1f}" if "lm_test_ppl" in r else "-"
        per_class = " | ".join(f"{v:.3f}" for v in r["per_class_f1"].values())
        lines.append(f"| {n} | {r['group']} | {r['dim']:,} | {C} | {val} | {r['test_macro_f1']:.3f} "
                     f"[{lo:.3f}, {hi:.3f}] | {r['test_acc']:.3f} | {per_class} | {ppl} |")
    lines.append("")
    for pool, rk in ranking.items():
        lines.append(f"Spearman rho (lower test PPL vs higher macro-F1, pool={pool}, n={len(rk['tags'])}): "
                     f"{rk['spearman_negppl_vs_f1']:+.2f}")
    open(os.path.join(RES, "part3_table.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


def write_pairs(pairs):
    lines = ["Paired bootstrap over test abstracts (same resamples for both sides).", "",
             "| A | B | macro-F1 A-B | 95% CI | p (two-sided) |", "|---|---|---|---|---|"]
    lines += [f"| {p['a']} | {p['b']} | {p['diff']:+.3f} | [{p['ci95'][0]:+.3f}, {p['ci95'][1]:+.3f}] | "
              f"{p['p_two_sided']:.3f} |" for p in pairs]
    open(os.path.join(RES, "part3_pairs.md"), "w").write("\n".join(lines) + "\n")


def position_bucket(r):
    return "first" if r["pos"] == 0 else "last" if r["pos"] == r["n_sent"] - 1 else "middle"


def write_errors(results, preds, test, y, n_examples=8):
    lm_reps = [n for n, r in results.items() if r["group"].startswith(("A", "B")) and "val_macro_f1" in r]
    best = max(lm_reps, key=lambda n: results[n]["val_macro_f1"])  # chosen on validation, not test
    rng = np.random.default_rng(SEED)
    text = lambda r: " ".join(r["tokens"])  # noqa: E731
    lines = [f"# Part III error analysis", "", f"Best LM representation by validation macro-F1: **{best}**", ""]

    p = preds[best]
    conf = Counter((LABELS[a], LABELS[b]) for a, b in zip(y, p) if a != b)
    lines += ["## Most common confusions (true -> predicted)", "", "| true | predicted | count | share of errors |",
              "|---|---|---|---|"]
    n_err = sum(conf.values())
    lines += [f"| {a} | {b} | {c} | {c / n_err:.1%} |" for (a, b), c in conf.most_common(6)]

    lines += ["", "## Accuracy by sentence position in the abstract", "",
              "| representation | first | middle | last |", "|---|---|---|---|"]
    buckets = np.array([position_bucket(r) for r in test])
    for n in [best, "transformer:mean", "lstm:mean", "rnn:mean", "tfidf"]:
        if n in preds:
            acc = [(preds[n][buckets == b] == y[buckets == b]).mean() for b in ("first", "middle", "last")]
            lines.append(f"| {n} | " + " | ".join(f"{a:.3f}" for a in acc) + " |")

    lines += ["", f"## Misclassified examples ({best}), from the top confusions", ""]
    for (a, b), _ in conf.most_common(4):
        idx = [i for i in range(len(y)) if LABELS[y[i]] == a and LABELS[p[i]] == b]
        for i in rng.choice(idx, size=min(2, len(idx)), replace=False):
            r = test[i]
            lines.append(f"- true **{a}**, predicted **{b}** (sentence {r['pos'] + 1}/{r['n_sent']}): {text(r)}")

    for a, b in [("transformer:mean", "lstm:mean"), ("lstm:mean", "transformer:mean")]:
        if a in preds and b in preds:
            idx = np.flatnonzero((preds[a] == y) & (preds[b] != y))
            lines += ["", f"## {a} right, {b} wrong ({len(idx)} sentences); examples", ""]
            for i in rng.choice(idx, size=min(n_examples // 2, len(idx)), replace=False):
                r = test[i]
                lines.append(f"- true **{LABELS[y[i]]}**, {b} said **{LABELS[preds[b][i]]}** "
                             f"(sentence {r['pos'] + 1}/{r['n_sent']}): {text(r)}")
    open(os.path.join(RES, "part3_errors.md"), "w").write("\n".join(lines) + "\n")


def plot(results, ppl):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    ink, muted, grid = "#1f1f1e", "#6b6a63", "#e6e5e0"
    plt.rcParams.update({"axes.edgecolor": muted, "axes.labelcolor": ink, "xtick.color": muted,
                         "ytick.color": muted, "font.size": 9})

    # LM test perplexity vs downstream macro-F1: both pools of one LM share its x, joined by a thin line
    fig, ax = plt.subplots(figsize=(7, 4.2))
    tags = [t for t in TAGS if t in ppl and all(f"{t}:{p}" in results for p in SERIES)]
    for t in tags:
        ys = [results[f"{t}:{p}"]["test_macro_f1"] for p in SERIES]
        ax.plot([ppl[t]] * 2, ys, color=muted, linewidth=1, zorder=2)
        ax.annotate(t, (ppl[t], min(ys)), xytext=(0, -8), textcoords="offset points", rotation=90,
                    ha="center", va="top", fontsize=8, color=ink)
    for pool, col in SERIES.items():
        ax.scatter([ppl[t] for t in tags], [results[f"{t}:{pool}"]["test_macro_f1"] for t in tags], s=64,
                   color=col, edgecolor="white", linewidth=2, zorder=3, label=f"{pool} pooling")
    ax.margins(x=0.06)
    lo = min(results[f"{t}:{p}"]["test_macro_f1"] for t in tags for p in SERIES)
    hi = max([results[f"{t}:{p}"]["test_macro_f1"] for t in tags for p in SERIES]
             + [results["tfidf"]["test_macro_f1"]] if "tfidf" in results else [])
    ax.set_ylim(lo - 0.06, hi + 0.012)
    if "tfidf" in results:
        ax.axhline(results["tfidf"]["test_macro_f1"], color=muted, linewidth=1, linestyle="--", zorder=1)
        ax.annotate("TF-IDF baseline", (ax.get_xlim()[1], results["tfidf"]["test_macro_f1"]), xytext=(-4, 4),
                    textcoords="offset points", ha="right", fontsize=8, color=muted)
    ax.set_xlabel("LM test perplexity (lower is better)")
    ax.set_ylabel("Classifier test macro-F1")
    ax.set_title("Next-token quality vs downstream role classification", color=ink, loc="left")
    ax.grid(color=grid, linewidth=0.8, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(1, 0.95))
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "part3_ppl_vs_f1.png"), dpi=200)
    plt.close(fig)

    # row-normalised confusion matrices
    names = [n for n in ("transformer:mean", "lstm:mean", "tfidf") if n in results]
    cmap = LinearSegmentedColormap.from_list("blue", ["#ffffff", "#9ec5f4", "#2a78d6", "#0d366b"])
    fig, axes = plt.subplots(1, len(names), figsize=(3.6 * len(names), 3.6), squeeze=False)
    short = [l[:4] for l in LABELS]
    for ax, n in zip(axes[0], names):
        m = np.array(results[n]["confusion"], float)
        m = m / m.sum(1, keepdims=True)
        ax.imshow(m, cmap=cmap, vmin=0, vmax=1)
        for i in range(K):
            for j in range(K):
                ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if m[i, j] > 0.55 else ink)
        ax.set_xticks(range(K), short)
        ax.set_yticks(range(K), short)
        ax.set_xlabel("predicted")
        ax.set_title(f"{n}  (macro-F1 {results[n]['test_macro_f1']:.3f})", fontsize=9, color=ink)
    axes[0][0].set_ylabel("true")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "part3_confusion.png"), dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
