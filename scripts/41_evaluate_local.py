"""Test evaluation on the replacement collection (requires configs/frozen_protocol.json).

usage: python scripts/41_evaluate_local.py [--only regression|provenance|shortlist|reconstruction|loto|cost]
Outputs under results/local/: regression/*.json + predictions/*.npz, scores/*.npy (float32 provenance matrices),
per_query/*.csv, provenance/*.json, shortlist.json, reconstruction/*.json, loto_heic.json, cost.json
Methods are evaluated only if their checkpoint exists; finished outputs are skipped (resumable).
"""
import argparse
import importlib
import json
import os
import random
import time

import numpy as np
import pandas as pd
import torch

import components as C
import evallib as E
import model as M
from paths import FRAGMENTS, PAIRS, RESULTS, RUNS

freeze = importlib.import_module("40_freeze_protocol")
EVAL_SPLIT = os.environ.get("FSR_EVAL_SPLIT", "test")      # "val" only for the pre-freeze dry run
OUT = os.environ.get("FSR_LOCAL_OUT", os.path.join(RESULTS, "local"))
for sub in ("regression", "predictions", "scores", "per_query", "provenance", "reconstruction"):
    os.makedirs(os.path.join(OUT, sub), exist_ok=True)
DEV = torch.device("cuda")
BASELINES = ["hist_cosine", "entropy_sim", "ngram_cosine", "compress_sim", "uniform_fusion_4", "random"]


def all_models(fz):
    names = [n for group in fz["models"].values() for n in group]
    seen, out = set(), []
    for n in names:
        if n not in seen and os.path.exists(os.path.join(RUNS, n, "best_model.pt")) and \
                os.path.exists(os.path.join(RUNS, n, "summary_val.json")):
            seen.add(n)
            out.append(n)
    return out


ALIAS = dict(kv.split("=") for kv in os.environ.get("FSR_MODEL_ALIAS", "").split(",") if "=" in kv)   # dry run only


def ckpt(name):
    return os.path.join(RUNS, ALIAS.get(name, name), "best_model.pt")


def dump(obj, *path):
    p = os.path.join(OUT, *path)
    json.dump(obj, open(p, "w", encoding="utf-8"), indent=1, default=float)


# ------------------------------------------------------------------ pools
def split_tables(split):
    frags = np.load(os.path.join(FRAGMENTS, f"{split}_frags.npy"))
    meta = pd.read_csv(os.path.join(FRAGMENTS, f"{split}_fragments.csv"))
    pairs = np.load(os.path.join(PAIRS, f"{split}_pairs.npz"))
    return frags, meta, pairs


def provenance_pool(split):
    frags, meta, pairs = split_tables(split)
    used = np.union1d(pairs["a"], pairs["b"])
    m = meta.iloc[used].reset_index(drop=True)
    files = {f: i for i, f in enumerate(sorted(m.collection_id.unique()))}
    types = {t: i for i, t in enumerate(sorted(m.ext.unique()))}
    merged = m.ext.replace({"jpeg": "jpg"})
    mtypes = {t: i for i, t in enumerate(sorted(merged.unique()))}
    return {"frags": frags[used], "frag_index": used, "file": m.collection_id.to_numpy(), "ext": m.ext.to_numpy(),
            "file_idx": m.collection_id.map(files).to_numpy(), "type_idx": m.ext.map(types).to_numpy(),
            "mtype_idx": merged.map(mtypes).to_numpy(), "offset": m.offset.to_numpy().astype(np.int64),
            "n_split_fragments": int(len(meta))}


def masks(pool, block=512):
    f, t, mt, o = pool["file_idx"], pool["type_idx"], pool["mtype_idx"], pool["offset"]
    n = len(f)
    same_file = f[:, None] == f[None, :]
    not_self = ~np.eye(n, dtype=bool)
    return {"same_file": same_file, "not_self": not_self, "same_type": t[:, None] == t[None, :],
            "same_mtype": mt[:, None] == mt[None, :],
            "adjacent": same_file & (np.abs(o[:, None] - o[None, :]) == block)}


def score_matrices(pool, methods, tag, save=True):
    """Baselines once, then each model. Returns dict name -> S (float32)."""
    out = {}
    bl_path = {b: os.path.join(OUT, "scores", f"{tag}__{b}.npy") for b in BASELINES}
    if all(os.path.exists(p) for p in bl_path.values()):
        out.update({b: np.load(p) for b, p in bl_path.items()})
    else:
        bs = E.baseline_scores(pool["frags"])
        for b in BASELINES:
            out[b] = bs[b]
            if save:
                np.save(bl_path[b], bs[b])
    asym = {}
    for mname in methods:
        p = os.path.join(OUT, "scores", f"{tag}__{mname}.npy")
        if os.path.exists(p):
            out[mname] = np.load(p)
            continue
        S, a = E.model_scores(ckpt(mname), pool["frags"], DEV)
        out[mname] = S
        asym[mname] = a
        if save:
            np.save(p, S)
    if asym:
        prev = os.path.join(OUT, "provenance", f"{tag}__order_asymmetry.json")
        old = json.load(open(prev)) if os.path.exists(prev) else {}
        old.update(asym)
        json.dump(old, open(prev, "w"), indent=1)
    return out


# ------------------------------------------------------------------ regression
@torch.no_grad()
def predict_pairs(name, frags, a, b):
    model, _ = M.load_model(ckpt(name), DEV)
    F = torch.from_numpy(frags).to(DEV)
    A, B = torch.from_numpy(a.astype(np.int64)).to(DEV), torch.from_numpy(b.astype(np.int64)).to(DEV)
    pab, pba = [], []
    for i in range(0, len(A), 2048):
        x1, x2 = F[A[i:i + 2048]].float() / 255, F[B[i:i + 2048]].float() / 255
        with torch.amp.autocast("cuda"):
            pab.append(model(x1, x2)[2].float().cpu())
            pba.append(model(x2, x1)[2].float().cpu())
    return torch.cat(pab).numpy(), torch.cat(pba).numpy()


def run_regression(models):
    T = importlib.import_module("30_train")
    import importlib as il
    T = il.import_module("30_train")
    frags, meta, pairs = split_tables(EVAL_SPLIT)
    rel = pd.read_csv(os.path.join(PAIRS, f"{EVAL_SPLIT}_pairs.csv"), usecols=["pair_mode", "relation", "a_ext", "b_ext"])
    for name in models:
        dst = os.path.join(OUT, "regression", f"{name}.json")
        if os.path.exists(dst):
            continue
        ck = torch.load(ckpt(name), map_location="cpu", weights_only=False)
        w = ck["weights"]
        y = C.composite({k: pairs[k] for k in C.COMPONENTS}, w)
        pab, pba = predict_pairs(name, frags, pairs["a"], pairs["b"])
        np.savez_compressed(os.path.join(OUT, "predictions", f"{name}.npz"), p_ab=pab, p_ba=pba, y=y.astype(np.float32))
        res = {"name": name, "weights": w, "test": T.regression_metrics(y, pab),
               "test_symmetrised_secondary": T.regression_metrics(y, 0.5 * (pab + pba)),
               "order_reversal": {"mean_abs_diff": float(np.mean(np.abs(pab - pba))),
                                  "p99_abs_diff": float(np.quantile(np.abs(pab - pba), 0.99)),
                                  "max_abs_diff": float(np.max(np.abs(pab - pba))),
                                  "corr_pab_pba": float(np.corrcoef(pab, pba)[0, 1])},
               "by_pair_mode": {m: {**T.regression_metrics(y[g.index], pab[g.index]), "n": int(len(g))}
                                for m, g in rel.groupby("pair_mode")},
               "by_relation": {m: {**T.regression_metrics(y[g.index], pab[g.index]), "n": int(len(g))}
                               for m, g in rel.groupby("relation")},
               "target_mean": float(y.mean()), "target_sd": float(y.std()), "pred_mean": float(pab.mean()),
               "pred_sd": float(pab.std())}
        heic = ((rel.a_ext == "heic") | (rel.b_ext == "heic")).to_numpy()
        res["pairs_with_heic"] = T.regression_metrics(y[heic], pab[heic])
        res["pairs_without_heic"] = T.regression_metrics(y[~heic], pab[~heic])
        dump(res, "regression", f"{name}.json")
        print("regression", name, {k: round(v, 4) for k, v in res["test"].items()}, flush=True)


# ------------------------------------------------------------------ provenance
def per_query_table(S, pool, mk):
    out = {}
    for key, rel, cand in (("retrieval_all", mk["same_file"] & mk["not_self"], mk["not_self"]),
                           ("retrieval_type", mk["same_file"] & mk["not_self"], mk["not_self"] & mk["same_type"]),
                           ("retrieval_type_merged", mk["same_file"] & mk["not_self"], mk["not_self"] & mk["same_mtype"]),
                           ("adjacency_within_file", mk["adjacent"], mk["same_file"] & mk["not_self"]),
                           ("adjacency_pooled", mk["adjacent"], mk["not_self"])):
        r, nc = E.first_relevant_rank(S, rel, cand, DEV)
        out[f"rank_{key}"] = r
        out[f"ncand_{key}"] = nc
        out[f"nrel_{key}"] = (rel & cand).sum(1)
    return pd.DataFrame({"pool_index": np.arange(len(S)), "frag_index": pool["frag_index"], "file": pool["file"],
                         "ext": pool["ext"], "offset": pool["offset"], **out})


def provenance_metrics(S, pool, mk, pq):
    res = {}
    for key, pos, sub in (("auc_same_file_all", mk["same_file"], None),
                          ("auc_same_file_within_type", mk["same_file"], mk["same_type"]),
                          ("auc_same_file_within_type_jpg_jpeg_merged", mk["same_file"], mk["same_mtype"]),
                          ("auc_same_type", mk["same_type"], None),
                          ("auc_same_type_jpg_jpeg_merged", mk["same_mtype"], None)):
        res[key] = E.PairAUC(S, pool["file_idx"], pos, sub).auc()
    for key in ("retrieval_all", "retrieval_type", "retrieval_type_merged", "adjacency_within_file", "adjacency_pooled"):
        r = pq[f"rank_{key}"].to_numpy()
        res[key] = E.summarize_ranks(r)
        has = ~np.isnan(r)
        res[key]["chance_hit@1"] = float(np.mean(pq[f"nrel_{key}"].to_numpy()[has] / pq[f"ncand_{key}"].to_numpy()[has]))
        res[key]["candidates_for_recall"] = E.candidates_for_recall(r)
    res.update(E.cluster_ari(S, pool["file_idx"]))
    return res


def run_provenance(models):
    pool = provenance_pool(EVAL_SPLIT)
    mk = masks(pool)
    dump({"pool_fragments": int(len(pool["frags"])), "split_fragments": pool["n_split_fragments"],
          "files": int(len(np.unique(pool["file"]))),
          "retrieval_queries": int((mk["same_file"] & mk["not_self"]).any(1).sum()),
          "adjacency_queries": int(mk["adjacent"].any(1).sum()),
          "pairs": int(len(pool["frags"]) * (len(pool["frags"]) - 1) // 2)}, "provenance", "pool.json")
    todo = [m for m in BASELINES + models if not os.path.exists(os.path.join(OUT, "provenance", f"{EVAL_SPLIT}__{m}.json"))]
    if not todo:
        return
    scores = score_matrices(pool, models, EVAL_SPLIT)
    for name in todo:
        S = scores[name]
        pq = per_query_table(S, pool, mk)
        pq.to_csv(os.path.join(OUT, "per_query", f"{EVAL_SPLIT}__{name}.csv"), index=False)
        res = provenance_metrics(S, pool, mk, pq)
        dump({"method": name, **res}, "provenance", f"{EVAL_SPLIT}__{name}.json")
        print("provenance", name, round(res["auc_same_file_within_type"], 3), round(res["retrieval_all"]["hit@1"], 3),
              round(res["adjacency_within_file"]["hit@1"], 3), round(res["ari"], 3), flush=True)


# ------------------------------------------------------------------ shortlist workflow
def two_stage_ranks(S_short, S_rank, rel, cand, M):
    """Shortlist top-M candidates by S_short, re-rank them by S_rank; 0-based rank of the first relevant candidate in
    the re-ranked shortlist, NaN if the query has relevant candidates but none in the shortlist (miss), and -1 for
    queries without relevant candidates."""
    n = len(S_short)
    Ss = torch.from_numpy(S_short).to(DEV)
    Sr = torch.from_numpy(S_rank).to(DEV)
    R = torch.from_numpy(rel & cand).to(DEV)
    Cn = torch.from_numpy(cand).to(DEV)
    out = np.full(n, -1.0)
    for i in range(0, n, 256):
        s = Ss[i:i + 256].masked_fill(~Cn[i:i + 256], float("-inf"))
        k = min(M, s.shape[1])
        top = torch.topk(s, k, dim=1).indices
        r2 = torch.gather(Sr[i:i + 256], 1, top)
        order = torch.argsort(r2, dim=1, descending=True, stable=True)
        ranked = torch.gather(top, 1, order)
        relv = torch.gather(R[i:i + 256], 1, ranked)
        has_any = R[i:i + 256].any(1)
        in_sl = relv.any(1)
        first = relv.float().argmax(1).float()
        res = torch.where(in_sl, first, torch.full_like(first, float("nan")))
        res = torch.where(has_any, res, torch.full_like(first, -1.0))
        out[i:i + 256] = res.cpu().numpy()
    return out


def summarize_two_stage(r):
    q = r[r >= 0] if not np.isnan(r).all() else r
    valid = r[~(r == -1)]
    hits = np.nan_to_num(valid, nan=np.inf)
    return {"queries": int(len(valid)), "hit@1": float(np.mean(hits < 1)), "hit@5": float(np.mean(hits < 5)),
            "hit@10": float(np.mean(hits < 10)), "mrr": float(np.mean(np.where(np.isfinite(hits), 1 / (hits + 1), 0.0))),
            "shortlist_recall": float(np.mean(np.isfinite(hits)))}


def run_shortlist(models):
    cfg = (freeze.check_frozen() if EVAL_SPLIT == "test" else freeze.PROTOCOL)["shortlist_workflow"]
    grid = cfg["grid"]
    res = {"grid": grid, "rule": cfg["selection"]}
    vpool, tpool = provenance_pool("val"), provenance_pool(EVAL_SPLIT)
    vmk, tmk = masks(vpool), masks(tpool)
    vs = score_matrices(vpool, [m for m in ("heuristic_s42", "uniform_s42", "learnability_s42") if m in models], "val")
    ts = score_matrices(tpool, [m for m in ("heuristic_s42", "uniform_s42", "learnability_s42") if m in models], EVAL_SPLIT)
    tasks = {"retrieval": ("same_file", "not_self"), "adjacency_pooled": ("adjacent", "not_self")}
    for short in ("heuristic_s42", "uniform_s42", "learnability_s42"):
        if short not in vs:
            continue
        res[short] = {}
        for task, (relk, candk) in tasks.items():
            vrel = vmk[relk] & vmk["not_self"]
            vgrid = {M: summarize_two_stage(two_stage_ranks(vs[short], vs["hist_cosine"], vrel, vmk[candk], M)) for M in grid}
            best = max(grid, key=lambda M: (vgrid[M]["mrr"], -M))
            trel = tmk[relk] & tmk["not_self"]
            ts_r = two_stage_ranks(ts[short], ts["hist_cosine"], trel, tmk[candk], best)
            direct_h, _ = E.first_relevant_rank(ts["hist_cosine"], trel, tmk[candk], DEV)
            direct_m, _ = E.first_relevant_rank(ts[short], trel, tmk[candk], DEV)
            n_cand = int(tmk[candk].sum(1).mean())
            effort = {}
            for lab, r in (("direct_hist_cosine", direct_h), ("direct_learned", direct_m)):
                effort[lab] = {"summary": E.summarize_ranks(r), "candidates_for_recall": E.candidates_for_recall(r)}
            q = ts_r[ts_r != -1]
            fin = np.sort(q[np.isfinite(q)])
            effort["two_stage"] = {"summary": summarize_two_stage(ts_r), "shortlist_size": best,
                                   "candidates_for_recall": {str(lv): (int(fin[int(np.ceil(lv * len(q))) - 1] + 1)
                                                                       if np.ceil(lv * len(q)) <= len(fin) else None)
                                                             for lv in (0.5, 0.8, 0.9, 0.95)}}
            res[short][task] = {"validation_grid": vgrid, "selected_M": best, "test": effort,
                                "mean_candidates_per_query": n_cand,
                                "shortlist_fraction_of_candidates": best / max(n_cand, 1)}
            pd.DataFrame({"frag_index": tpool["frag_index"], "two_stage_rank": ts_r, "direct_hist_rank": direct_h,
                          "direct_learned_rank": direct_m}).to_csv(
                os.path.join(OUT, "per_query", f"shortlist__{short}__{task}.csv"), index=False)
            print("shortlist", short, task, "M", best, effort["two_stage"]["summary"], flush=True)
    dump(res, "shortlist.json")


# ------------------------------------------------------------------ reconstruction (test-only consecutive blocks)
def reconstruction_pool(run=24):
    import paths
    man = pd.read_csv(os.path.join(paths.MANIFESTS, "source_manifest.csv"))
    test = man[man.partition == EVAL_SPLIT].sort_values("collection_id")
    frags, rows, excl = [], [], []
    for _, r in test.iterrows():
        n_blocks = r.size_bytes // 512
        if n_blocks < run:
            excl.append({"collection_id": r.collection_id, "ext": r.recorded_extension, "complete_blocks": int(n_blocks)})
            continue
        start = random.Random("recon-v1|" + r.collection_id).randrange(0, n_blocks - run + 1)
        data = open(os.path.join(paths.DATA, r.stored_path), "rb").read()
        for k in range(run):
            o = (start + k) * 512
            frags.append(np.frombuffer(data[o:o + 512], dtype=np.uint8))
            rows.append({"collection_id": r.collection_id, "ext": r.recorded_extension, "offset": o, "block": k})
    meta = pd.DataFrame(rows)
    files = {f: i for i, f in enumerate(sorted(meta.collection_id.unique()))}
    return {"frags": np.stack(frags), "file": meta.collection_id.to_numpy(), "ext": meta.ext.to_numpy(),
            "file_idx": meta.collection_id.map(files).to_numpy(), "offset": meta.offset.to_numpy(),
            "meta": meta, "excluded": excl}


def run_reconstruction(models):
    pool = reconstruction_pool()
    pool["meta"].to_csv(os.path.join(OUT, "reconstruction", "pool_fragments.csv"), index=False)
    dump({"files": int(pool["meta"].collection_id.nunique()), "fragments": int(len(pool["frags"])),
          "run_length": 24, "excluded_files_fewer_than_24_blocks": pool["excluded"],
          "files_by_group": pool["meta"].groupby("ext").collection_id.nunique().to_dict(),
          "true_adjacent_pairs": int(pool["meta"].collection_id.nunique() * 23)}, "reconstruction", "pool.json")
    n = len(pool["frags"])
    f, o = pool["file_idx"], pool["offset"]
    same = f[:, None] == f[None, :]
    adj = same & (np.abs(o[:, None] - o[None, :]) == 512)
    not_self = ~np.eye(n, dtype=bool)
    mk = {"adjacent": adj, "same_file": same, "not_self": not_self}
    rp = {"frags": pool["frags"], "file_idx": f}
    sel = [m for m in models if m in ("heuristic_s42", "heuristic_s43", "heuristic_s44", "uniform_s42", "learnability_s42",
                                      "uniform_s43", "uniform_s44", "learnability_s43", "learnability_s44")]
    scores = score_matrices(rp, sel, "recon")
    for name in BASELINES + sel:
        dst = os.path.join(OUT, "reconstruction", f"{name}.json")
        if os.path.exists(dst):
            continue
        S = scores[name]
        links, rep = E.chaining_report(S, f.tolist(), o.tolist(), 512)
        cm = rep["full"]
        r_pool, _ = E.first_relevant_rank(S, adj, not_self, DEV)
        r_within, _ = E.first_relevant_rank(S, adj, same & not_self, DEV)
        res = {"method": name, **cm, "oracle_stop": rep["oracle_stop"], "adjacency_pooled": E.summarize_ranks(r_pool),
               "adjacency_within_file_oracle": E.summarize_ranks(r_within),
               "adjacency_pooled_chance_hit@1": float(np.mean(adj.sum(1) / (n - 1))),
               "adjacency_within_file_chance_hit@1": float(np.mean(adj.sum(1) / (same & not_self).sum(1)))}
        pd.DataFrame(links, columns=["a", "b"]).to_csv(os.path.join(OUT, "reconstruction", f"links__{name}.csv"), index=False)
        pd.DataFrame({"rank_adjacency_pooled": r_pool, "rank_adjacency_within_file": r_within}).to_csv(
            os.path.join(OUT, "per_query", f"recon__{name}.csv"), index=False)
        dump(res, "reconstruction", f"{name}.json")
        print("reconstruction", name, round(cm["adjacency_recall"], 3), round(cm["join_precision"], 3),
              round(rep["oracle_stop"]["chain_purity_weighted"], 3), flush=True)


# ------------------------------------------------------------------ HEIC leave-one-type-out
def run_loto(models):
    if "loto_heic_s42" not in models:
        return
    pool = provenance_pool(EVAL_SPLIT)
    mk = masks(pool)
    heic = pool["ext"] == "heic"
    inv = heic[:, None] | heic[None, :]
    both = heic[:, None] & heic[None, :]
    names = [m for m in ("loto_heic_s42", "heuristic_s42", "heuristic_s43", "heuristic_s44") if m in models] + ["hist_cosine"]
    scores = score_matrices(pool, [m for m in names if m != "hist_cosine"], EVAL_SPLIT)
    out = {"pool_heic_fragments": int(heic.sum()), "heic_files": int(len(np.unique(pool["file"][heic])))}
    for name in names:
        S = scores[name]
        r = {"same_type_auc_pairs_with_heic": E.PairAUC(S, pool["file_idx"], mk["same_type"], inv).auc(),
             "same_file_auc_heic_pairs": E.PairAUC(S, pool["file_idx"], mk["same_file"], both).auc()}
        reg = os.path.join(OUT, "regression", f"{name}.json")
        if os.path.exists(reg):
            j = json.load(open(reg))
            r["regression_pairs_with_heic"] = j["pairs_with_heic"]
            r["regression_pairs_without_heic"] = j["pairs_without_heic"]
        out[name] = r
    dump(out, "loto_heic.json")


# ------------------------------------------------------------------ float32 scoring sensitivity (declared)
def run_fp32(models):
    """Secondary analysis: pair head and encoder in float32 (no autocast) for the three main schemes, to show whether
    float16 output quantisation (ties) affects the provenance conclusions."""
    pool = provenance_pool(EVAL_SPLIT)
    mk = masks(pool)
    for name in ("heuristic_s42", "uniform_s42", "learnability_s42"):
        dst = os.path.join(OUT, "provenance", f"fp32__{name}.json")
        if name not in models or os.path.exists(dst):
            continue
        S, a = E.model_scores(ckpt(name), pool["frags"], DEV, fp32=True)
        np.save(os.path.join(OUT, "scores", f"fp32__{name}.npy"), S)
        pq = per_query_table(S, pool, mk)
        pq.to_csv(os.path.join(OUT, "per_query", f"fp32__{name}.csv"), index=False)
        res = provenance_metrics(S, pool, mk, pq)
        res["distinct_score_values"] = int(len(np.unique(S)))
        res["order_asymmetry"] = a
        dump({"method": name, "scoring": "float32", **res}, "provenance", f"fp32__{name}.json")
        print("fp32", name, round(res["auc_same_file_within_type"], 3), round(res["retrieval_all"]["hit@1"], 3), flush=True)


# ------------------------------------------------------------------ cost
def run_cost(models):
    pool = provenance_pool(EVAL_SPLIT)
    n = len(pool["frags"])
    out = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "pool_fragments": n,
           "caching": "embeddings computed once per fragment and cached; pair head evaluated for every ordered pair; "
                      "byte features computed once per fragment; cosine similarities by matrix multiplication",
           "note": "GPU shared with another user process during measurement; timings are wall-clock upper bounds"}
    t = time.time()
    feat = C.fragment_features(pool["frags"])
    out["baseline_feature_seconds"] = time.time() - t
    t = time.time()
    C.baseline_matrices(feat)
    out["baseline_all_pairs_seconds"] = time.time() - t
    if "heuristic_s42" in models:
        model, _ = M.load_model(ckpt("heuristic_s42"), DEV)
        torch.cuda.synchronize()
        t = time.time()
        Z = E.embed(model, pool["frags"], DEV)
        torch.cuda.synchronize()
        out["model_embedding_seconds"] = time.time() - t
        t = time.time()
        P = E.head_matrix(model, Z, DEV)
        torch.cuda.synchronize()
        out["model_pair_head_all_ordered_pairs_seconds"] = time.time() - t
        out["model_fragments_per_second"] = n / out["model_embedding_seconds"]
        out["model_pairs_per_second"] = n * n / out["model_pair_head_all_ordered_pairs_seconds"]
    mk = masks(pool)
    S = np.load(os.path.join(OUT, "scores", f"{EVAL_SPLIT}__hist_cosine.npy"))
    t = time.time()
    E.first_relevant_rank(S, mk["same_file"] & mk["not_self"], mk["not_self"], DEV)
    out["ranking_all_queries_seconds"] = time.time() - t
    dump(out, "cost.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()
    if EVAL_SPLIT == "test":
        fz = freeze.check_frozen()
        models = all_models(fz)
    else:   # pre-freeze dry run of the code on validation data; never touches test fragments
        models = list(ALIAS)
    steps = {"regression": run_regression, "provenance": run_provenance, "shortlist": run_shortlist,
             "reconstruction": run_reconstruction, "loto": run_loto, "fp32": run_fp32, "cost": run_cost}
    for k, fn in steps.items():
        if args.only and k not in args.only:
            continue
        fn(models)
    print("LOCAL EVALUATION DONE", flush=True)


if __name__ == "__main__":
    main()
