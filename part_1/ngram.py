"""Interpolated Kneser-Ney trigram language model (implemented from scratch with numpy).

P3(w|u,v) = max(c(uvw)-D3, 0)/c(uv.) + D3 * N1+(uv.)/c(uv.) * P2(w|v)        [falls back to P2 if c(uv.)=0]
P2(w|v)   = max(N1+(.vw)-D2, 0)/N1+(.v.) + D2 * N1+(v.)/N1+(.v.) * P1(w)      [continuation counts]
P1(w)     = max(N1+(.w)-D1, 0)/N1+(..) + D1 * |{w: N1+(.w)>0}|/N1+(..) * 1/V  [interpolated with uniform]

Each level sums to 1 over the vocabulary, so the model is a proper distribution
(every vocabulary item, including <unk>, gets non-zero probability).
"""
import numpy as np


def _lookup(keys, values, query):
    """Vectorised dictionary lookup: values[keys == query], 0 if absent. `keys` must be sorted."""
    idx = np.searchsorted(keys, query)
    idx_c = np.minimum(idx, len(keys) - 1)
    found = keys[idx_c] == query
    return np.where(found, values[idx_c], 0)


class KNTrigram:
    def __init__(self, vocab_size, discounts=(0.75, 0.75, 0.75)):
        self.V = vocab_size
        self.D1, self.D2, self.D3 = discounts

    def fit(self, ids):
        V = np.int64(self.V)
        ids = np.asarray(ids, dtype=np.int64)
        u, v, w = ids[:-2], ids[1:-1], ids[2:]
        # trigram counts c(uvw) and context totals c(uv.), N1+(uv.)
        self.k3, self.c3 = np.unique(u * V * V + v * V + w, return_counts=True)
        self.ctx3, inv = np.unique(self.k3 // V, return_inverse=True)
        self.C3 = np.bincount(inv, weights=self.c3)
        self.N3 = np.bincount(inv)
        # continuation bigram counts N1+(.vw) = #distinct u preceding (v,w)
        self.k2, self.c2 = np.unique(self.k3 % (V * V), return_counts=True)
        self.ctx2, inv = np.unique(self.k2 // V, return_inverse=True)
        self.C2 = np.bincount(inv, weights=self.c2)
        self.N2 = np.bincount(inv)
        # continuation unigram counts N1+(.w) = #distinct v preceding w
        self.c1 = np.bincount(self.k2 % V, minlength=self.V).astype(np.float64)
        self.T1 = self.c1.sum()
        self.n1 = (self.c1 > 0).sum()
        return self

    def n_params(self):
        """Number of stored statistics (n-gram types with counts) — the model's 'size'."""
        return len(self.k3) + len(self.k2) + len(self.ctx3) + len(self.ctx2) + self.V

    def prob(self, u, v, w):
        """P(w | u, v) for aligned integer arrays u, v, w."""
        V = np.int64(self.V)
        u, v, w = (np.asarray(a, dtype=np.int64) for a in (u, v, w))
        p1 = (np.maximum(self.c1[w] - self.D1, 0) + self.D1 * self.n1 / self.V) / self.T1

        c2 = _lookup(self.k2, self.c2, v * V + w)
        C2 = _lookup(self.ctx2, self.C2, v)
        N2 = _lookup(self.ctx2, self.N2, v)
        p2 = np.where(C2 > 0, (np.maximum(c2 - self.D2, 0) + self.D2 * N2 * p1) / np.maximum(C2, 1), p1)

        c3 = _lookup(self.k3, self.c3, u * V * V + v * V + w)
        C3 = _lookup(self.ctx3, self.C3, u * V + v)
        N3 = _lookup(self.ctx3, self.N3, u * V + v)
        return np.where(C3 > 0, (np.maximum(c3 - self.D3, 0) + self.D3 * N3 * p2) / np.maximum(C3, 1), p2)

    def distribution(self, u, v):
        """Full next-token distribution over the vocabulary given context (u, v)."""
        w = np.arange(self.V)
        return self.prob(np.full(self.V, u), np.full(self.V, v), w)

    def nll(self, ids, eos_id):
        """Mean negative log-likelihood of targets ids[1:], each conditioned on its two preceding
        tokens (the first target uses <eos> as the missing left context). Same targets as the neural
        models' evaluation."""
        ids = np.asarray(ids, dtype=np.int64)
        prev2 = np.concatenate([[eos_id], ids[:-2]])
        return float(-np.log(self.prob(prev2, ids[:-1], ids[1:])).mean())

    def tune(self, valid_ids, eos_id, grid=(0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)):
        """Grid-search the three discounts on VALIDATION data only."""
        best = (np.inf, None)
        for d1 in grid:
            for d2 in grid:
                for d3 in grid:
                    self.D1, self.D2, self.D3 = d1, d2, d3
                    loss = self.nll(valid_ids, eos_id)
                    if loss < best[0]:
                        best = (loss, (d1, d2, d3))
        self.D1, self.D2, self.D3 = best[1]
        return best
