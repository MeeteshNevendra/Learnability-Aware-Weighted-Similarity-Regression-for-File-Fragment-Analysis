"""Evaluation primitives shared by the local, reconstruction and Govdocs1 evaluations.

Conventions fixed in configs/frozen_protocol.json before any test evaluation:
  * model scores for provenance tasks: S = 0.5 * (P + P^T), P[i, j] = head(z_i, z_j) (archived convention);
    regression metrics use the ordered test pairs as sampled; order-reversal statistics are reported separately
  * ranking ties are broken by a fixed pseudo-random key per candidate (numpy default_rng(12345) permutation), i.e.
    random tie-breaking that is reproducible and gives no candidate an index-order advantage
  * per-query output: 0-based rank of the first relevant candidate (NaN when the query has no relevant candidate)
  * AUCs: Mann-Whitney with ties counted 1/2 over unordered fragment pairs i < j
"""
import math

import numpy as np
import torch

import components as C
import model as M

TIE_SEED = 12345


# ------------------------------------------------------------------ score matrices
@torch.no_grad()
def embed(model, frags, device, window=512, batch=2048, fp32=False):
    """L2-normalised embeddings; fragments longer than 512 bytes are split into 512-byte windows whose embeddings are
    averaged and re-normalised (archived 4096-byte procedure). fp32=True disables autocast (sensitivity analysis)."""
    n, L = frags.shape
    w = L // window
    x = torch.from_numpy(np.ascontiguousarray(frags).reshape(n * w, window)).to(device)
    Z = []
    for i in range(0, len(x), batch):
        with torch.amp.autocast("cuda", enabled=device.type == "cuda" and not fp32):
            Z.append(model.encoder(x[i:i + batch].float() / 255.0).float())
    Z = torch.cat(Z)
    if w > 1:
        Z = torch.nn.functional.normalize(Z.view(n, w, -1).mean(1), dim=1)
    return Z


@torch.no_grad()
def head_matrix(model, Z, device, rows=32, fp32=False):
    """P[i, j] = head(z_i, z_j) for every ordered pair (float32, on CPU). Under autocast (archived convention) the
    sigmoid output is float16, so scores are quantised (~5e-4 near 0.5) and ties are common."""
    n = len(Z)
    P = torch.empty(n, n, device=device)
    for i in range(0, n, rows):
        zi = Z[i:i + rows]
        r = len(zi)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda" and not fp32):
            pred, _ = model.head(zi.unsqueeze(1).expand(r, n, -1).reshape(r * n, -1),
                                 Z.unsqueeze(0).expand(r, n, -1).reshape(r * n, -1))
        P[i:i + r] = pred.float().view(r, n)
    return P.cpu().numpy()


def model_scores(ckpt_path, frags, device, fp32=False):
    model, ck = M.load_model(ckpt_path, device)
    if fp32:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    Z = embed(model, frags, device, fp32=fp32)
    P = head_matrix(model, Z, device, fp32=fp32)
    S = 0.5 * (P + P.T)
    asym = np.abs(P - P.T)[np.triu_indices(len(P), 1)]
    return S.astype(np.float32), {"mean_abs_order_difference": float(asym.mean()),
                                  "p99_abs_order_difference": float(np.quantile(asym, 0.99)),
                                  "max_abs_order_difference": float(asym.max())}


def baseline_scores(frags, random_seed=0):
    feat = C.fragment_features(frags)
    s = C.baseline_matrices(feat)
    rng = np.random.default_rng(random_seed)
    r = rng.random((len(frags), len(frags)))
    s["random"] = 0.5 * (r + r.T)
    return {k: v.astype(np.float32) for k, v in s.items()}


# ------------------------------------------------------------------ ranking
def first_relevant_rank(S, relevant, candidate, device, tie_seed=TIE_SEED, chunk=256):
    """0-based rank of the best relevant candidate under descending score with fixed random tie-breaking.
    S, relevant, candidate: [n, n] (numpy). Returns float array [n] (NaN if no relevant candidate) and the number of
    candidates per query."""
    n = len(S)
    tb = torch.from_numpy(np.random.default_rng(tie_seed).permutation(n).astype(np.int64)).to(device)
    St = torch.from_numpy(np.ascontiguousarray(S, dtype=np.float32)).to(device)
    R = torch.from_numpy(np.ascontiguousarray(relevant)).to(device)
    Cn = torch.from_numpy(np.ascontiguousarray(candidate)).to(device)
    ranks = np.full(n, np.nan)
    ncand = Cn.sum(1).cpu().numpy()
    for i in range(0, n, chunk):
        s, rel, cand = St[i:i + chunk], R[i:i + chunk] & Cn[i:i + chunk], Cn[i:i + chunk]
        neg_inf = torch.tensor(float("-inf"), device=device)
        best = torch.where(rel, s, neg_inf).max(1).values                       # best relevant score
        has = rel.any(1)
        nonrel = cand & ~rel
        above = (nonrel & (s > best[:, None])).sum(1)
        tied_rel = rel & (s == best[:, None])
        big = torch.tensor(n + 1, device=device)
        first_tb = torch.where(tied_rel, tb[None, :], big).min(1).values        # tie key of first tied relevant
        tied_before = (nonrel & (s == best[:, None]) & (tb[None, :] < first_tb[:, None])).sum(1)
        r = (above + tied_before).float().cpu().numpy()
        h = has.cpu().numpy()
        ranks[i:i + chunk][h] = r[h]
    return ranks, ncand


def summarize_ranks(ranks, ks=(1, 5, 10)):
    q = ranks[~np.isnan(ranks)]
    out = {"queries": int(len(q))}
    for k in ks:
        out[f"hit@{k}"] = float(np.mean(q < k)) if len(q) else float("nan")
    out["mrr"] = float(np.mean(1.0 / (q + 1.0))) if len(q) else float("nan")
    out["mean_rank"] = float(np.mean(q + 1)) if len(q) else float("nan")
    out["median_rank"] = float(np.median(q + 1)) if len(q) else float("nan")
    return out


def recall_at_k_curve(ranks, ks):
    q = ranks[~np.isnan(ranks)]
    return {int(k): float(np.mean(q < k)) for k in ks}


def candidates_for_recall(ranks, levels=(0.5, 0.8, 0.9, 0.95)):
    """Smallest k such that a fraction >= level of queries has a relevant candidate within the top k."""
    q = np.sort(ranks[~np.isnan(ranks)])
    out = {}
    for lv in levels:
        idx = int(math.ceil(lv * len(q))) - 1
        out[str(lv)] = int(q[idx] + 1) if len(q) else None
    return out


# ------------------------------------------------------------------ AUC with file-cluster bootstrap support
class PairAUC:
    """Exact Mann-Whitney AUC over unordered fragment pairs with optional file-cluster bootstrap weights.

    Pair weights for a bootstrap draw with file multiplicities c: c[a]*c[b] for pairs of different files and c[a]
    for pairs within one file (pairs between two copies of the same file are not formed), as in the archived
    bootstrap. Scores are sorted once; each replicate is a weighted cumulative sum."""

    def __init__(self, S, file_idx, pos, subset=None):
        n = len(S)
        iu, ju = np.triu_indices(n, 1)
        keep = np.ones(len(iu), dtype=bool) if subset is None else subset[iu, ju]
        iu, ju = iu[keep], ju[keep]
        s = S[iu, ju]
        order = np.argsort(s, kind="stable")
        self.s = s[order]
        self.fa, self.fb = file_idx[iu][order].astype(np.int32), file_idx[ju][order].astype(np.int32)
        self.pos = pos[iu, ju][order]
        # tie groups: first and last index of the equal-score run containing each element
        new = np.r_[True, self.s[1:] != self.s[:-1]]
        gid = np.cumsum(new) - 1
        starts = np.flatnonzero(new)
        ends = np.r_[starts[1:], len(self.s)] - 1
        self.first, self.last = starts[gid], ends[gid]
        self.same_file = self.fa == self.fb

    def auc(self, c=None):
        if c is None:
            w = np.ones(len(self.s))
        else:
            w = np.where(self.same_file, c[self.fa], c[self.fa] * c[self.fb]).astype(np.float64)
        wn = np.where(self.pos, 0.0, w)
        wp = np.where(self.pos, w, 0.0)
        cn = np.r_[0.0, np.cumsum(wn)]
        below = cn[self.first]
        equal = cn[self.last + 1] - cn[self.first]
        tp, tn = wp.sum(), wn.sum()
        if tp == 0 or tn == 0:
            return float("nan")
        return float((wp * (below + 0.5 * equal)).sum() / (tp * tn))


# ------------------------------------------------------------------ clustering and chaining
def cluster_ari(S, file_idx, n_clusters=None):
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    D = (S.max() - S).astype(np.float64)
    np.fill_diagonal(D, 0.0)
    k = n_clusters or len(np.unique(file_idx))
    lab = AgglomerativeClustering(n_clusters=k, metric="precomputed", linkage="average").fit_predict(D)
    return {"ari": float(adjusted_rand_score(file_idx, lab)), "nmi": float(normalized_mutual_info_score(file_idx, lab)),
            "clusters": int(k)}


def greedy_chains(S):
    """Links accepted in decreasing score order (stable for ties), max degree 2, no cycles. Uses scores only."""
    n = len(S)
    iu, ju = np.triu_indices(n, 1)
    order = np.argsort(-S[iu, ju], kind="stable")
    a_idx, b_idx = iu[order], ju[order]
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    deg = [0] * n
    links = []
    for a, b in zip(a_idx.tolist(), b_idx.tolist()):
        if deg[a] >= 2 or deg[b] >= 2:
            continue
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        parent[ra] = rb
        deg[a] += 1
        deg[b] += 1
        links.append((a, b))
        if len(links) == n - 1:
            break
    return links


def chaining_report(S, file_idx, offset, block):
    """Greedy chaining evaluated two ways from ONE link order:
      full        archived procedure: links accepted until n-1 links (a single path once complete) -> adjacency recall
                  and join precision over all accepted links
      oracle_stop the same link order stopped after n - F links, F = number of source files in the pool (ORACLE,
                  like the known cluster count for ARI) -> chain purity, pure-chain share, exact ordered runs
    Link selection never uses source identities or offsets."""
    links = greedy_chains(S)
    n_files = len(set(file_idx))
    k = max(len(file_idx) - n_files, 0)
    return links, {"full": chain_metrics(links, file_idx, offset, block),
                   "oracle_stop": {**chain_metrics(links[:k], file_idx, offset, block), "links_allowed": k}}


def chain_metrics(links, file_idx, offset, block):
    n = len(file_idx)
    adj_true = 0
    by_file = {}
    for i, (f, o) in enumerate(zip(file_idx, offset)):
        by_file.setdefault(f, set()).add(o)
    for f, offs in by_file.items():
        adj_true += sum(1 for o in offs if o + block in offs)
    correct = sum(1 for a, b in links if file_idx[a] == file_idx[b] and abs(offset[a] - offset[b]) == block)
    # chains = connected components of the accepted links
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    nbrs = {i: [] for i in range(n)}
    for a, b in links:
        parent[find(a)] = find(b)
        nbrs[a].append(b)
        nbrs[b].append(a)
    comps = {}
    for i in range(n):
        comps.setdefault(find(i), []).append(i)
    chains = [c for c in comps.values() if len(c) >= 2]
    pure_frags, tot_frags, purity_weighted = 0, 0, 0.0
    exact_files = 0
    for c in chains:
        files = [file_idx[i] for i in c]
        top = max(set(files), key=files.count)
        purity_weighted += files.count(top)
        tot_frags += len(c)
        if files.count(top) == len(c):
            pure_frags += len(c)
            # exact sequence: chain holds every retained block of that file, consecutive, as one path
            offs = sorted(offset[i] for i in c)
            if len(c) == len(by_file[top]) and all(offs[k + 1] - offs[k] == block for k in range(len(offs) - 1)):
                ends = [i for i in c if len(nbrs[i]) == 1]
                if len(ends) == 2:
                    path, prev, cur = [ends[0]], None, ends[0]
                    while len(path) < len(c):
                        nxt = [x for x in nbrs[cur] if x != prev][0]
                        path.append(nxt)
                        prev, cur = cur, nxt
                    po = [offset[i] for i in path]
                    if all(abs(po[k + 1] - po[k]) == block for k in range(len(po) - 1)):
                        exact_files += 1
    return {"fragments": int(n), "true_adjacent_pairs": int(adj_true), "links": len(links),
            "correct_links": int(correct), "adjacency_recall": correct / adj_true if adj_true else float("nan"),
            "join_precision": correct / len(links) if links else float("nan"),
            "chains_ge2": len(chains), "chain_purity_weighted": purity_weighted / tot_frags if tot_frags else float("nan"),
            "share_fragments_in_pure_chains": pure_frags / tot_frags if tot_frags else float("nan"),
            "files_recovered_as_exact_ordered_sequence": int(exact_files), "files": len(by_file)}
