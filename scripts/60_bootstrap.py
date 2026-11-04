"""1000-replicate source-file cluster bootstrap for the local test pool, the reconstruction pool and Govdocs1.

usage: python scripts/60_bootstrap.py local|recon|govdocs [--ari] [--workers 12]
  * one set of file draws per pool (seed from the frozen protocol; local = seed, recon = seed+1, govdocs = seed+2),
    shared by every method, so method differences are paired
  * AUCs: bootlib.GroupedAUC (pairs between two copies of one file are never formed)
  * query metrics: per-query ranks from the evaluation, averaged over the drawn files' queries with multiplicity
  * chaining (recon, govdocs): the full-pool chaining is kept fixed and files are reweighted through per-file
    contributions (bootlib.chaining_file_stats), like the per-query metrics
  * --ari: average-linkage clustering re-run on the distinct drawn files (duplicates collapsed); ARI depends on pool
    size, so these intervals describe subsets of about 63% of the files and are reported with that caveat
Outputs: results/bootstrap/<pool>__<method>.npz (replicates) and <pool>_summary.json (estimates, 95% CIs, paired
differences against hist_cosine).
"""
import argparse
import importlib
import json
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd

import bootlib as B
import evallib as E
from paths import FRAGMENTS, GOVDOCS, PAIRS, RESULTS

freeze = importlib.import_module("40_freeze_protocol")
EVAL_SPLIT = os.environ.get("FSR_EVAL_SPLIT", "test")   # "val" only for the pre-freeze dry run
OUT = os.environ.get("FSR_BOOT_OUT", os.path.join(RESULTS, "bootstrap"))
os.makedirs(OUT, exist_ok=True)
LOCAL = os.environ.get("FSR_LOCAL_OUT", os.path.join(RESULTS, "local"))
GOV = os.path.join(RESULTS, "govdocs1")
MAIN_MODELS = ["heuristic_s42", "heuristic_s43", "heuristic_s44", "uniform_s42", "learnability_s42"]
BASE = ["hist_cosine", "entropy_sim", "ngram_cosine", "compress_sim", "uniform_fusion_4", "random"]
_G = {}


def _init(S, file_idx, offset, block, kind):
    _G.update(S=S, f=file_idx, o=offset, block=block, kind=kind)


def _one(counts):
    keep_files = np.flatnonzero(counts > 0)
    idx = np.flatnonzero(np.isin(_G["f"], keep_files))
    S = _G["S"][np.ix_(idx, idx)]
    f = _G["f"][idx]
    if _G["kind"] == "ari":
        return E.cluster_ari(S, f)["ari"]
    _, rep = E.chaining_report(S, f.tolist(), _G["o"][idx].tolist(), _G["block"])
    return [rep["full"]["adjacency_recall"], rep["full"]["join_precision"], rep["oracle_stop"]["chain_purity_weighted"]]


def resample_nondecomposable(S, f, o, block, C, kind, workers):
    with Pool(workers, initializer=_init, initargs=(S, f, o, block, kind)) as p:
        return np.array(p.map(_one, list(C), chunksize=8), dtype=np.float64)


def summarise(reps, point):
    out = {}
    for k, v in reps.items():
        out[k] = {"estimate": point.get(k, float(np.nanmean(v))), "ci95": B.ci(v)}
    return out


def local_pool():
    meta = pd.read_csv(os.path.join(FRAGMENTS, f"{EVAL_SPLIT}_fragments.csv"))
    z = np.load(os.path.join(PAIRS, f"{EVAL_SPLIT}_pairs.npz"))
    used = np.union1d(z["a"], z["b"])
    m = meta.iloc[used].reset_index(drop=True)
    files = {fid: i for i, fid in enumerate(sorted(m.collection_id.unique()))}
    t = {e: i for i, e in enumerate(sorted(m.ext.unique()))}
    mt = m.ext.replace({"jpeg": "jpg"})
    tm = {e: i for i, e in enumerate(sorted(mt.unique()))}
    return {"file_idx": m.collection_id.map(files).to_numpy(), "type_idx": m.ext.map(t).to_numpy(),
            "mtype_idx": mt.map(tm).to_numpy(), "offset": m.offset.to_numpy(), "n_files": len(files)}


def run_local(fz, ari, workers):
    pool = local_pool()
    f = pool["file_idx"]
    C = B.draws(pool["n_files"], fz["bootstrap"]["replicates"], fz["bootstrap"]["seed"])
    same_file = f[:, None] == f[None, :]
    st = pool["type_idx"][:, None] == pool["type_idx"][None, :]
    smt = pool["mtype_idx"][:, None] == pool["mtype_idx"][None, :]
    methods = [m[len(EVAL_SPLIT) + 2:-5] for m in os.listdir(os.path.join(LOCAL, "provenance"))
               if m.startswith(f"{EVAL_SPLIT}__") and m.endswith(".json") and not m.endswith("order_asymmetry.json")]
    summary = {"replicates": int(len(C)), "files": int(pool["n_files"]), "methods": {}}
    reps_all = {}
    for m in sorted(methods):
        S = np.load(os.path.join(LOCAL, "scores", f"{EVAL_SPLIT}__{m}.npy"))
        point = json.load(open(os.path.join(LOCAL, "provenance", f"{EVAL_SPLIT}__{m}.json")))
        reps, pt, dev = {}, {}, {}
        for key, pos, sub in (("auc_same_file_all", same_file, None), ("auc_same_file_within_type", same_file, st),
                              ("auc_same_file_within_type_jpg_jpeg_merged", same_file, smt),
                              ("auc_same_type", st, None), ("auc_same_type_jpg_jpeg_merged", smt, None)):
            G = B.GroupedAUC(S, f, pos, sub)
            reps[key] = G.auc_many(C)
            pt[key] = point[key]
            dev[key] = abs(G.auc() - point[key])
        pq = pd.read_csv(os.path.join(LOCAL, "per_query", f"{EVAL_SPLIT}__{m}.csv"))
        for task in ("retrieval_all", "retrieval_type", "retrieval_type_merged", "adjacency_within_file", "adjacency_pooled"):
            for k, v in B.rank_metrics_per_query(pq[f"rank_{task}"].to_numpy()).items():
                reps[f"{task}_{k}"] = B.query_means(v, f, C, pool["n_files"])
                pt[f"{task}_{k}"] = point[task][k]
        if ari and m in BASE + MAIN_MODELS:
            reps["ari"] = resample_nondecomposable(S, f, pool["offset"], 512, C, "ari", workers)
            pt["ari"] = point["ari"]
        np.savez_compressed(os.path.join(OUT, f"local__{m}.npz"), **reps)
        reps_all[m] = reps
        summary["methods"][m] = {"metrics": summarise(reps, pt), "binned_vs_exact_auc_abs_diff": dev}
        print("local", m, {k: [round(x, 3) for x in v["ci95"]] for k, v in list(summary["methods"][m]["metrics"].items())[:2]},
              flush=True)
    summary["paired_differences_vs_hist_cosine"] = paired(reps_all)
    json.dump(summary, open(os.path.join(OUT, "local_summary.json"), "w"), indent=1, default=float)


def paired(reps_all, ref="hist_cosine"):
    out = {}
    if ref not in reps_all:
        return out
    for m, reps in reps_all.items():
        if m == ref:
            continue
        out[m] = {}
        for k, v in reps.items():
            if k in reps_all[ref]:
                d = v - reps_all[ref][k]
                out[m][k] = {"mean_difference": float(np.nanmean(d)), "ci95": B.ci(d),
                             "share_replicates_method_better": float(np.nanmean(d > 0))}
    return out


def run_recon(fz, workers):
    meta = pd.read_csv(os.path.join(LOCAL, "reconstruction", "pool_fragments.csv"))
    files = {fid: i for i, fid in enumerate(sorted(meta.collection_id.unique()))}
    f = meta.collection_id.map(files).to_numpy()
    o = meta.offset.to_numpy()
    C = B.draws(len(files), fz["bootstrap"]["replicates"], fz["bootstrap"]["seed"] + 1)
    summary = {"replicates": int(len(C)), "files": len(files), "methods": {}}
    reps_all = {}
    for fn in sorted(os.listdir(os.path.join(LOCAL, "reconstruction"))):
        if not fn.endswith(".json") or fn == "pool.json":
            continue
        m = fn[:-5]
        point = json.load(open(os.path.join(LOCAL, "reconstruction", fn)))
        links = [tuple(x) for x in pd.read_csv(os.path.join(LOCAL, "reconstruction", f"links__{m}.csv"))[["a", "b"]].to_numpy()]
        st = B.chaining_file_stats(links, links[:point["oracle_stop"]["links_allowed"]], f, o, 512, len(files))
        reps = B.chaining_replicates(st, C)
        pq = pd.read_csv(os.path.join(LOCAL, "per_query", f"recon__{m}.csv"))
        for task in ("adjacency_pooled", "adjacency_within_file"):
            for k, v in B.rank_metrics_per_query(pq[f"rank_{task}"].to_numpy()).items():
                reps[f"{task}_{k}"] = B.query_means(v, f, C, len(files))
        pt = {"adjacency_recall": point["adjacency_recall"], "join_precision": point["join_precision"],
              "chain_purity_weighted": point["oracle_stop"]["chain_purity_weighted"]}
        np.savez_compressed(os.path.join(OUT, f"recon__{m}.npz"), **reps)
        reps_all[m] = reps
        summary["methods"][m] = summarise(reps, pt)
        print("recon", m, summary["methods"][m]["adjacency_recall"], flush=True)
    summary["paired_differences_vs_hist_cosine"] = paired(reps_all)
    json.dump(summary, open(os.path.join(OUT, "recon_summary.json"), "w"), indent=1, default=float)


def run_govdocs(fz, ari, workers):
    summary = {"replicates": fz["bootstrap"]["replicates"], "conditions": {}}
    for cond in fz["govdocs1"]["conditions"]:
        meta = pd.read_csv(os.path.join(GOVDOCS, f"{cond}_meta.csv"))
        files = {fid: i for i, fid in enumerate(sorted(meta.file_id.unique()))}
        f = meta.file_id.map(files).to_numpy()
        t = meta.ext.map({e: i for i, e in enumerate(sorted(meta.ext.unique()))}).to_numpy()
        o = meta.offset.to_numpy()
        block = int(np.load(os.path.join(GOVDOCS, f"{cond}.npz"))["frags"].shape[1])
        C = B.draws(len(files), fz["bootstrap"]["replicates"], fz["bootstrap"]["seed"] + 2)
        sf, st = f[:, None] == f[None, :], t[:, None] == t[None, :]
        res, reps_all = {}, {}
        for fn in sorted(os.listdir(GOV)):
            if not (fn.startswith(cond + "__") and fn.endswith(".json")):
                continue
            m = fn[len(cond) + 2:-5]
            S = np.load(os.path.join(GOV, "scores", f"{cond}__{m}.npy"))
            point = json.load(open(os.path.join(GOV, fn)))
            reps, pt = {}, {}
            for key, pos, sub in (("auc_same_file_all", sf, None), ("auc_same_file_within_type", sf, st),
                                  ("auc_same_type", st, None)):
                reps[key] = B.GroupedAUC(S, f, pos, sub).auc_many(C)
                pt[key] = point[key]
            pq = pd.read_csv(os.path.join(GOV, "per_query", f"{cond}__{m}.csv"))
            for task in ("retrieval_all", "retrieval_type", "adjacency_within_file", "adjacency_pooled"):
                for k, v in B.rank_metrics_per_query(pq[task].to_numpy()).items():
                    reps[f"{task}_{k}"] = B.query_means(v, f, C, len(files))
                    pt[f"{task}_{k}"] = point[task][k]
            links = [tuple(x) for x in pd.read_csv(os.path.join(GOV, "links", f"{cond}__{m}.csv"))[["a", "b"]].to_numpy()]
            cr = B.chaining_replicates(B.chaining_file_stats(
                links, links[:point["greedy_oracle_stop"]["links_allowed"]], f, o, block, len(files)), C)
            reps.update({"greedy_adjacency_recall": cr["adjacency_recall"], "greedy_join_precision": cr["join_precision"],
                         "greedy_chain_purity": cr["chain_purity_weighted"]})
            pt.update({"greedy_adjacency_recall": point["greedy"]["adjacency_recall"],
                       "greedy_join_precision": point["greedy"]["join_precision"],
                       "greedy_chain_purity": point["greedy_oracle_stop"]["chain_purity_weighted"]})
            if ari:
                reps["ari"] = resample_nondecomposable(S, f, o, block, C, "ari", workers)
                pt["ari"] = point["ari"]
            np.savez_compressed(os.path.join(OUT, f"govdocs_{cond}__{m}.npz"), **reps)
            reps_all[m] = reps
            res[m] = summarise(reps, pt)
            print("govdocs", cond, m, res[m]["auc_same_file_within_type"], flush=True)
        summary["conditions"][cond] = {"methods": res, "paired_differences_vs_hist_cosine": paired(reps_all)}
    json.dump(summary, open(os.path.join(OUT, "govdocs_summary.json"), "w"), indent=1, default=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pool", choices=["local", "recon", "govdocs"])
    ap.add_argument("--ari", action="store_true")
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    fz = freeze.check_frozen() if EVAL_SPLIT == "test" else {**freeze.PROTOCOL, "bootstrap": {"replicates": 50, "seed": 1}}
    {"local": lambda: run_local(fz, a.ari, a.workers), "recon": lambda: run_recon(fz, a.workers),
     "govdocs": lambda: run_govdocs(fz, a.ari, a.workers)}[a.pool]()


if __name__ == "__main__":
    main()
