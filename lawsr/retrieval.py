"""Exact-successor retrieval (manuscript Section 3.5, Eq. 25).

For query q the candidates are all other fragments of the pool.  Candidates are sorted by decreasing
score; equal scores are ordered by a fixed permutation of fragment identifiers (seed 42) shared by
all methods (Algorithm 1, steps 21-25).  r_q is the one-based rank of the exact successor.

    Top(k) = (1/Q) sum_q 1{r_q <= k}        MRR = (1/Q) sum_q 1 / r_q
    same-file P@10 = share of the first ten candidates that come from the query's source file
"""
import numpy as np


def successor_ranks(scores, self_cols, target_cols, tie_rank):
    """scores: (n_queries, pool) array, the query's own column is ignored.  Returns 1-based ranks."""
    sc = np.array(scores, dtype=np.float64, copy=True)
    rows = np.arange(len(sc))
    sc[rows, self_cols] = -np.inf
    s_t = sc[rows, target_cols][:, None]
    t_t = tie_rank[target_cols][:, None]
    higher = (sc > s_t).sum(axis=1)
    tied_before = ((sc == s_t) & (tie_rank[None, :] < t_t)).sum(axis=1)
    return 1 + higher + tied_before


def top_k_same_source(scores, self_cols, query_source, pool_source, tie_rank, k=10):
    """Share of the first k ranked candidates that come from the query's source file."""
    sc = np.array(scores, dtype=np.float64, copy=True)
    sc[np.arange(len(sc)), self_cols] = -np.inf
    perm = np.argsort(tie_rank, kind='stable')                 # columns in tie order
    top = perm[np.argsort(-sc[:, perm], axis=1, kind='stable')[:, :k]]
    return (pool_source[top] == np.asarray(query_source)[:, None]).mean(axis=1)


def summarize_ranks(ranks, p_at_10=None):
    ranks = np.asarray(ranks, dtype=np.float64)
    out = {'Top-1 (%)': 100.0 * float((ranks <= 1).mean()), 'Top-10 (%)': 100.0 * float((ranks <= 10).mean()),
           'MRR': float((1.0 / ranks).mean()), 'queries': int(len(ranks))}
    if p_at_10 is not None:
        out['Same-file P@10 (%)'] = 100.0 * float(np.mean(p_at_10))
    return out


def random_order_reference(n_candidates, same_source_candidates=None):
    """Expected values under uniform random ordering of C candidates."""
    C = int(n_candidates)
    harmonic = float(np.sum(1.0 / np.arange(1, C + 1)))
    out = {'Top-1 (%)': 100.0 / C, 'Top-10 (%)': 100.0 * min(10, C) / C, 'MRR': harmonic / C}
    if same_source_candidates is not None:
        out['Same-file P@10 (%)'] = 100.0 * float(np.mean(np.asarray(same_source_candidates) / C))
    return out
