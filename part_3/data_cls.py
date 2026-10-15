"""Part III labelled data: rhetorical-role classification of PubMed 20k RCT sentences.

Each sentence carries one of 5 labels (background / objective / methods / results / conclusions). To avoid any
overlap with language-model text, the classifier only uses abstracts that are NOT in the LM subsets
(part_1/cache/lm_abstract_ids.json). The official splits are disjoint by abstract and we split by abstract too:

  classifier train       N_CLS["train"] abstracts sampled from the 8,500 unused official-train abstracts
  classifier validation  N_CLS["validation"] abstracts sampled from the 1,800 unused official-validation abstracts
  classifier test        all 1,700 unused official-test abstracts

Tokenisation is the Part I one (lowercase + whitespace split) and words are mapped with the Part I vocabulary
(built from the LM training split only), so rare/unseen words become <unk> exactly as in LM training.
"""
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
P1 = os.path.join(HERE, "..", "part_1")
sys.path.insert(0, P1)
from data import load_abstracts, lm_abstract_ids  # noqa: E402

CACHE = os.path.join(HERE, "cache")
LABELS = ["background", "objective", "methods", "results", "conclusions"]  # order of the abstract
N_CLS = {"train": 2000, "validation": 500, "test": None}  # None = all unused abstracts
CLS_SEED = 0


def load_cls_data():
    """Return {split: list of dicts(abstract_id, pos, n_sent, tokens, label)}; cached to cache/cls_data.json."""
    path = os.path.join(CACHE, "cls_data.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    used = lm_abstract_ids()
    rng = random.Random(CLS_SEED)
    out, chosen = {}, {}
    for split, n in N_CLS.items():
        abstracts = load_abstracts(split)
        free = sorted(set(abstracts) - set(used[split]))
        chosen[split] = sorted(rng.sample(free, n)) if n else free
        out[split] = [dict(abstract_id=aid, pos=i, n_sent=len(abstracts[aid]), tokens=toks, label=LABELS.index(lab))
                      for aid in chosen[split] for i, (toks, lab) in enumerate(abstracts[aid])]
    # leakage checks: no abstract shared with the LM subsets or between classifier splits
    all_lm = {a for ids in used.values() for a in ids}
    assert not all_lm & {a for ids in chosen.values() for a in ids}, "classifier abstract used in LM training/eval"
    assert not (set(chosen["train"]) & set(chosen["validation"]) | set(chosen["train"]) & set(chosen["test"]))
    os.makedirs(CACHE, exist_ok=True)
    with open(os.path.join(CACHE, "cls_abstract_ids.json"), "w") as f:
        json.dump(chosen, f)
    with open(path, "w") as f:
        json.dump(out, f)
    return out


if __name__ == "__main__":
    from collections import Counter

    data = load_cls_data()
    for split, rows in data.items():
        c = Counter(r["label"] for r in rows)
        n_abs = len({r["abstract_id"] for r in rows})
        dist = ", ".join(f"{LABELS[k]} {c[k] / len(rows):.1%}" for k in range(len(LABELS)))
        print(f"{split:>10}: {n_abs:>5} abstracts, {len(rows):>6} sentences | {dist}")
