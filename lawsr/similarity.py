"""Byte-level similarity components (manuscript Section 3.2) and fixed-score fusions.

    s_hist   Eq. (7)   cosine similarity of the normalized 256-bin byte histograms
    s_ent    Eq. (8)   1 - |E_i - E_j| / 8, with Shannon entropy E in bits per byte
    s_ngram  Eq. (9)   Jaccard index of the sets of distinct adjacent byte pairs
    s_comp   Eqs. (10)-(11)  max(0, 1 - NCD); zlib (DEFLATE level 6); concatenation f_i || f_j
                       in the recorded pair order (in retrieval the query is always f_i)
    uniform_fusion   Eq. (15)
    weighted_fusion  Eq. (16)

`pair_similarity` is a direct, unoptimized transcription of the equations.  `FragmentStore` gives
identical values faster, by caching per-fragment quantities, storing the 2-gram sets as a sparse
0/1 matrix (so that set intersections become sparse matrix products), and computing concatenated
compressed lengths in parallel worker processes.
"""
import os
import zlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy import sparse

from .config import ZLIB_LEVEL


# ----------------------------------------------------------------------------- reference transcription
def byte_histogram(frag):
    """H_i(b) = c_i(b) / L_i for b = 0..255."""
    frag = np.asarray(frag, dtype=np.uint8)
    return np.bincount(frag, minlength=256).astype(np.float64) / frag.size


def shannon_entropy(hist):
    """E_i = -sum_{b: H_i(b) > 0} H_i(b) log2 H_i(b), in bits per byte."""
    nz = hist[hist > 0]
    return float(-(nz * np.log2(nz)).sum())


def bigram_set(frag):
    """G_i = {(b_t, b_{t+1}) : 1 <= t < L_i}, encoded as b_t * 256 + b_{t+1}."""
    a = np.asarray(frag, dtype=np.int64)
    return set((a[:-1] * 256 + a[1:]).tolist())


def compressed_length(data, level=ZLIB_LEVEL):
    return len(zlib.compress(bytes(data), level))


def ncd(x, y, level=ZLIB_LEVEL):
    """Eq. (10): (C(x||y) - min(C(x), C(y))) / max(C(x), C(y))."""
    cx, cy = compressed_length(x, level), compressed_length(y, level)
    cxy = compressed_length(bytes(x) + bytes(y), level)
    return (cxy - min(cx, cy)) / max(cx, cy)


def pair_similarity(x, y, level=ZLIB_LEVEL):
    """The four components of Eq. (14) for the ordered pair (x, y)."""
    x = np.asarray(x, dtype=np.uint8)
    y = np.asarray(y, dtype=np.uint8)
    hx, hy = byte_histogram(x), byte_histogram(y)
    s_hist = float(hx @ hy / (np.linalg.norm(hx) * np.linalg.norm(hy)))
    s_ent = 1.0 - abs(shannon_entropy(hx) - shannon_entropy(hy)) / 8.0
    gx, gy = bigram_set(x), bigram_set(y)
    s_ngram = len(gx & gy) / len(gx | gy)
    s_comp = max(0.0, 1.0 - ncd(x.tobytes(), y.tobytes(), level))
    return np.array([s_hist, s_ent, s_ngram, s_comp])


def uniform_fusion(S):
    """Eq. (15): mean of the K components."""
    return np.asarray(S, dtype=np.float64).mean(axis=-1)


def weighted_fusion(S, w):
    """Eq. (16): sum_k w_k s^(k)."""
    return np.asarray(S, dtype=np.float64) @ np.asarray(w, dtype=np.float64)


# ----------------------------------------------------------------------------- memory-light pair scoring
class PairFeaturizer:
    """Same values as `FragmentStore.pair_features`, for large pools of long blocks: per-fragment
    histograms, entropies and compressed lengths are cached, 2-gram sets are built only for the
    fragments that occur in a pair (and cached when cache_bigrams is True); single process."""

    def __init__(self, frags, level=ZLIB_LEVEL, cache_bigrams=True):
        self.frags = np.ascontiguousarray(np.asarray(frags, dtype=np.uint8))
        self.level = level
        n, L = self.frags.shape
        counts = np.zeros((n, 256), dtype=np.float64)
        for i in range(n):
            counts[i] = np.bincount(self.frags[i], minlength=256)
        self.hist = counts / L
        self.hnorm = np.linalg.norm(self.hist, axis=1)
        safe = np.where(self.hist > 0, self.hist, 1.0)
        self.ent = -(self.hist * np.log2(safe)).sum(axis=1)
        self.clen = np.array([len(zlib.compress(self.frags[i].tobytes(), level)) for i in range(n)], dtype=np.int64)
        self.cache_bigrams = cache_bigrams
        self._big = [None] * n

    def bigrams(self, i):
        g = self._big[i]
        if g is None:
            a = self.frags[i].astype(np.int32)
            g = np.unique(a[:-1] * 256 + a[1:])
            if self.cache_bigrams:
                self._big[i] = g
        return g

    def features(self, a, b):
        a = np.asarray(a, dtype=np.int64)
        b = np.asarray(b, dtype=np.int64)
        s_hist = (self.hist[a] * self.hist[b]).sum(axis=1) / (self.hnorm[a] * self.hnorm[b])
        s_ent = 1.0 - np.abs(self.ent[a] - self.ent[b]) / 8.0
        s_ngram = np.empty(len(a))
        s_comp = np.empty(len(a))
        frags, level, clen = self.frags, self.level, self.clen
        for k in range(len(a)):
            i, j = int(a[k]), int(b[k])
            gi, gj = self.bigrams(i), self.bigrams(j)
            inter = np.intersect1d(gi, gj, assume_unique=True).size
            s_ngram[k] = inter / (gi.size + gj.size - inter)
            cij = len(zlib.compress(frags[i].tobytes() + frags[j].tobytes(), level))
            ci, cj = clen[i], clen[j]
            s_comp[k] = max(0.0, 1.0 - (cij - min(ci, cj)) / max(ci, cj))
        return np.column_stack([s_hist, s_ent, s_ngram, s_comp])


# ----------------------------------------------------------------------------- parallel compression
_W = {}


def _init_worker(frags, level):
    _W['bytes'] = [f.tobytes() for f in frags]
    _W['level'] = level


def _lengths_query_vs_all(query_rows):
    pool, level = _W['bytes'], _W['level']
    out = np.empty((len(query_rows), len(pool)), dtype=np.int32)
    for i, q in enumerate(query_rows):
        qb = pool[q]
        out[i] = [len(zlib.compress(qb + pb, level)) for pb in pool]
    return out


def _lengths_pairs(ab):
    a, b = ab
    pool, level = _W['bytes'], _W['level']
    return np.array([len(zlib.compress(pool[i] + pool[j], level)) for i, j in zip(a, b)], dtype=np.int32)


def default_workers():
    return max(1, min(32, (os.cpu_count() or 2) - 2))


class FragmentStore:
    """Per-fragment quantities for a set of fragments of equal length (uint8 array, n x L)."""

    def __init__(self, frags, level=ZLIB_LEVEL, workers=None):
        self.frags = np.ascontiguousarray(np.asarray(frags, dtype=np.uint8))
        self.level = level
        self.workers = default_workers() if workers is None else workers
        n, L = self.frags.shape
        counts = np.empty((n, 256), dtype=np.float64)
        for i in range(n):
            counts[i] = np.bincount(self.frags[i], minlength=256)
        self.hist = counts / L
        self.norm = np.linalg.norm(self.hist, axis=1)
        safe = np.where(self.hist > 0, self.hist, 1.0)
        self.entropy = -(self.hist * np.log2(safe)).sum(axis=1)
        self.clen = np.array([len(zlib.compress(self.frags[i].tobytes(), level)) for i in range(n)], dtype=np.int64)
        rows, cols = [], []
        for i in range(n):                                          # distinct 2-grams of each fragment
            codes = np.unique(self.frags[i, :-1].astype(np.int64) * 256 + self.frags[i, 1:])
            rows.append(np.full(codes.size, i, dtype=np.int64))
            cols.append(codes)
        rows, cols = np.concatenate(rows), np.concatenate(cols)
        self.bigrams = sparse.csr_matrix((np.ones(rows.size, dtype=np.int32), (rows, cols)), shape=(n, 65536))
        self.n_bigrams = np.diff(self.bigrams.indptr).astype(np.int64)

    def __len__(self):
        return len(self.frags)

    # -- helpers -----------------------------------------------------------------------------
    def _pool(self):
        return ProcessPoolExecutor(max_workers=self.workers, initializer=_init_worker,
                                   initargs=(self.frags, self.level))

    def _comp_from_lengths(self, cab, ca, cb):
        ncd_ = (cab - np.minimum(ca, cb)) / np.maximum(ca, cb)
        return np.maximum(0.0, 1.0 - ncd_)

    # -- ordered pairs (a_k, b_k) --------------------------------------------------------------
    def pair_features(self, a, b, chunk=50_000):
        a = np.asarray(a, dtype=np.int64)
        b = np.asarray(b, dtype=np.int64)
        out = np.empty((len(a), 4), dtype=np.float64)
        out[:, 0] = (self.hist[a] * self.hist[b]).sum(axis=1) / (self.norm[a] * self.norm[b])
        out[:, 1] = 1.0 - np.abs(self.entropy[a] - self.entropy[b]) / 8.0
        for s in range(0, len(a), chunk):
            aa, bb = a[s:s + chunk], b[s:s + chunk]
            inter = np.asarray(self.bigrams[aa].multiply(self.bigrams[bb]).sum(axis=1)).ravel()
            out[s:s + chunk, 2] = inter / (self.n_bigrams[aa] + self.n_bigrams[bb] - inter)
        if self.workers <= 1 or len(a) < 2000:
            _init_worker(self.frags, self.level)
            cab = _lengths_pairs((a, b))
        else:
            parts = np.array_split(np.arange(len(a)), self.workers * 4)
            with self._pool() as ex:
                cab = np.concatenate(list(ex.map(_lengths_pairs, [(a[p], b[p]) for p in parts])))
        out[:, 3] = self._comp_from_lengths(cab, self.clen[a], self.clen[b])
        return out

    # -- every query against every fragment of the store ------------------------------------------
    def query_features(self, query_rows, ex=None):
        """Features of (query, candidate) for every query row and every fragment of the store, with the
        query as the first argument.  Returns float64 array (len(query_rows), n, 4); the column of the
        query itself is included and must be excluded by the caller."""
        q = np.asarray(query_rows, dtype=np.int64)
        n = len(self)
        out = np.empty((len(q), n, 4), dtype=np.float64)
        out[:, :, 0] = (self.hist[q] @ self.hist.T) / np.outer(self.norm[q], self.norm)
        out[:, :, 1] = 1.0 - np.abs(self.entropy[q][:, None] - self.entropy[None, :]) / 8.0
        inter = (self.bigrams[q] @ self.bigrams.T).toarray()            # |G_q intersect G_c| for all pairs
        out[:, :, 2] = inter / (self.n_bigrams[q][:, None] + self.n_bigrams[None, :] - inter)
        if ex is None:
            _init_worker(self.frags, self.level)
            cab = _lengths_query_vs_all(q)
        else:
            parts = [p for p in np.array_split(q, max(1, min(len(q), self.workers * 2))) if len(p)]
            cab = np.concatenate(list(ex.map(_lengths_query_vs_all, parts)))
        out[:, :, 3] = self._comp_from_lengths(cab, self.clen[q][:, None], self.clen[None, :])
        return out
