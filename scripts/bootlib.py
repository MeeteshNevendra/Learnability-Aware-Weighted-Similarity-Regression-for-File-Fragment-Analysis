"""File-cluster bootstrap machinery (used by 60_bootstrap.py and 02_reevaluate_archived.py).

AUC: for unordered fragment pairs grouped by source-file pair g = (A, B), a bootstrap draw with file multiplicities c
gives every pair of g the weight c_A*c_B (A != B) or c_A (A == B; pairs between two copies of one file are never
formed). With scores binned into 32,768 equal-width bins, K[g_pos, g_neg] = sum over (p in g_pos, q in g_neg) of
1[bin_q < bin_p] + 0.5*1[bin_q == bin_p] is computed once, and a replicate's AUC is
  w_pos' K w_neg / ((w_pos' n_pos) (w_neg' n_neg)).
The binned AUC is compared with the exact AUC (evallib.PairAUC) and the absolute difference is reported.
Query-level metrics: per-file sums of per-query values; a replicate averages over the queries of the drawn files
with multiplicity (ranks are those computed on the full pool).
"""
import numpy as np

NBINS = 32768


def draws(n_files, reps, seed):
    rng = np.random.default_rng(seed)
    return np.stack([np.bincount(rng.integers(0, n_files, n_files), minlength=n_files) for _ in range(reps)]).astype(np.float64)


class GroupedAUC:
    def __init__(self, S, file_idx, pos, subset=None, nbins=NBINS, chunk=400):
        n = len(S)
        nf = int(file_idx.max()) + 1
        iu, ju = np.triu_indices(n, 1)
        keep = np.ones(len(iu), dtype=bool) if subset is None else subset[iu, ju]
        iu, ju = iu[keep], ju[keep]
        s = S[iu, ju].astype(np.float64)
        lo, hi = s.min(), s.max()
        b = np.minimum(((s - lo) / max(hi - lo, 1e-12) * nbins).astype(np.int64), nbins - 1)
        fa, fb = file_idx[iu], file_idx[ju]
        g = np.minimum(fa, fb).astype(np.int64) * nf + np.maximum(fa, fb)
        p = pos[iu, ju]
        self.nf = nf
        self.gp, gp_inv = np.unique(g[p], return_inverse=True)
        self.gn, gn_inv = np.unique(g[~p], return_inverse=True)
        P = np.bincount(gp_inv * nbins + b[p], minlength=len(self.gp) * nbins).reshape(len(self.gp), nbins).astype(np.float64)
        self.npos = P.sum(1)
        bn, gnn = b[~p], gn_inv
        order = np.argsort(gnn, kind="stable")
        bn, gnn = bn[order], gnn[order]
        bounds = np.searchsorted(gnn, np.arange(len(self.gn) + 1))
        K = np.zeros((len(self.gp), len(self.gn)))
        self.nneg = np.zeros(len(self.gn))
        for c0 in range(0, len(self.gn), chunk):
            c1 = min(c0 + chunk, len(self.gn))
            sl = slice(bounds[c0], bounds[c1])
            H = np.bincount((gnn[sl] - c0) * nbins + bn[sl], minlength=(c1 - c0) * nbins).reshape(c1 - c0, nbins).astype(np.float64)
            self.nneg[c0:c1] = H.sum(1)
            Ncum = np.cumsum(H, axis=1) - 0.5 * H
            K[:, c0:c1] = P @ Ncum.T
        self.K = K

    def _w(self, gids, c):
        a, b = np.divmod(gids, self.nf)
        return np.where(a == b, c[a], c[a] * c[b])

    def auc(self, c=None):
        c = np.ones(self.nf) if c is None else c
        wp, wn = self._w(self.gp, c), self._w(self.gn, c)
        den = (wp @ self.npos) * (wn @ self.nneg)
        return float(wp @ self.K @ wn / den) if den > 0 else float("nan")

    def auc_many(self, C):
        """C: [reps, n_files] -> AUC per replicate (vectorised)."""
        a_p, b_p = np.divmod(self.gp, self.nf)
        a_n, b_n = np.divmod(self.gn, self.nf)
        WP = np.where(a_p == b_p, C[:, a_p], C[:, a_p] * C[:, b_p])
        WN = np.where(a_n == b_n, C[:, a_n], C[:, a_n] * C[:, b_n])
        num = np.einsum("rp,pn,rn->r", WP, self.K, WN, optimize=True)
        den = (WP @ self.npos) * (WN @ self.nneg)
        return np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)


def query_means(values, file_of_query, C, n_files):
    """values: per-query metric (NaN = not a query). Returns replicate means [reps]."""
    ok = ~np.isnan(values)
    sums = np.bincount(file_of_query[ok], weights=values[ok], minlength=n_files)
    cnts = np.bincount(file_of_query[ok], minlength=n_files).astype(np.float64)
    den = C @ cnts
    return np.where(den > 0, (C @ sums) / np.where(den > 0, den, 1), np.nan)


def chaining_file_stats(links, links_stop, file_idx, offset, block, n_files):
    """Per-file contributions of a fixed (full-pool) chaining, for file-reweighting bootstrap replicates:
    recall = sum c_f correct_f / sum c_f true_f; precision = sum c_f correct_f / sum c_f accepted_f (each accepted link
    counts 1/2 for each endpoint's file); purity (oracle stop) = sum c_f major_f / sum c_f inchain_f, where major_f
    counts fragments of f lying in chains (>= 2 members) whose majority file is f and inchain_f counts fragments of f in
    chains with >= 2 members."""
    file_idx = np.asarray(file_idx)
    offset = np.asarray(offset)
    correct, accepted, true_adj = (np.zeros(n_files) for _ in range(3))
    for a, b in links:
        fa, fb = file_idx[a], file_idx[b]
        if fa == fb and abs(offset[a] - offset[b]) == block:
            correct[fa] += 1
        accepted[fa] += 0.5
        accepted[fb] += 0.5
    offs = {}
    for f, o in zip(file_idx, offset):
        offs.setdefault(f, set()).add(int(o))
    for f, s in offs.items():
        true_adj[f] = sum(1 for o in s if o + block in s)
    n = len(file_idx)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b in links_stop:
        parent[find(a)] = find(b)
    comps = {}
    for i in range(n):
        comps.setdefault(find(i), []).append(i)
    major, inchain = np.zeros(n_files), np.zeros(n_files)
    for c in comps.values():
        if len(c) < 2:
            continue
        fs = file_idx[c]
        vals, cnt = np.unique(fs, return_counts=True)
        top = vals[np.argmax(cnt)]
        for v, k in zip(vals, cnt):
            inchain[v] += k
            if v == top:
                major[v] += k
    return {"correct": correct, "accepted": accepted, "true": true_adj, "major": major, "inchain": inchain}


def chaining_replicates(st, C):
    def ratio(num, den):
        d = C @ den
        return np.where(d > 0, (C @ num) / np.where(d > 0, d, 1), np.nan)
    return {"adjacency_recall": ratio(st["correct"], st["true"]), "join_precision": ratio(st["correct"], st["accepted"]),
            "chain_purity_weighted": ratio(st["major"], st["inchain"])}


def ci(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return [float("nan"), float("nan")]
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def rank_metrics_per_query(ranks):
    """Per-query indicator arrays from 0-based first-relevant ranks (NaN preserved)."""
    out = {}
    for k in (1, 5, 10):
        out[f"hit@{k}"] = np.where(np.isnan(ranks), np.nan, (ranks < k).astype(float))
    out["mrr"] = np.where(np.isnan(ranks), np.nan, 1.0 / (ranks + 1.0))
    return out
