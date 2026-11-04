"""External Govdocs1 evaluation of the replacement-collection models (no retraining). Requires the frozen protocol.

Per condition (aligned, unaligned, encrypted, compressed, b4096) and method (byte baselines, random, frozen model list):
same-file AUC (all / within type), same-type AUC, retrieval (all candidates; type-restricted = oracle), adjacency
(within-file = oracle, archived; pooled = practical), ARI, greedy chaining (adjacency recall, join precision, chain
purity, exact ordered sequences) and pool-wide neighbour Hit@1 (archived metric).
4096-byte fragments: model scores from the re-normalised mean of eight 512-byte window embeddings; baselines on the
full 4096 bytes. Outputs: results/govdocs1/{condition}__{method}.json, scores/, per_query/, links/
"""
import importlib
import json
import os

import numpy as np
import pandas as pd
import torch

import evallib as E
from paths import GOVDOCS, RESULTS, RUNS

freeze = importlib.import_module("40_freeze_protocol")
OUT = os.path.join(RESULTS, "govdocs1")
for sub in ("scores", "per_query", "links"):
    os.makedirs(os.path.join(OUT, sub), exist_ok=True)
DEV = torch.device("cuda")
CONDITIONS = ["aligned", "unaligned", "encrypted", "compressed", "b4096"]
BASELINES = ["hist_cosine", "entropy_sim", "ngram_cosine", "compress_sim", "uniform_fusion_4", "random"]


def load_pool(cond):
    frags = np.load(os.path.join(GOVDOCS, f"{cond}.npz"))["frags"]
    meta = pd.read_csv(os.path.join(GOVDOCS, f"{cond}_meta.csv"))
    files = {f: i for i, f in enumerate(sorted(meta.file_id.unique()))}
    types = {t: i for i, t in enumerate(sorted(meta.ext.unique()))}
    return {"frags": frags, "file": meta.file_id.to_numpy(), "ext": meta.ext.to_numpy(),
            "file_idx": meta.file_id.map(files).to_numpy(), "type_idx": meta.ext.map(types).to_numpy(),
            "offset": meta.offset.to_numpy().astype(np.int64)}


def main():
    fz = freeze.check_frozen()
    # A run counts as finished only once summary_val.json exists; best_model.pt is rewritten while a run is training.
    models = {m: os.path.join(RUNS, m, "best_model.pt") for m in fz["models"]["govdocs1_models"]
              if os.path.exists(os.path.join(RUNS, m, "best_model.pt"))
              and os.path.exists(os.path.join(RUNS, m, "summary_val.json"))}
    evaluate(models, OUT)
    print("GOVDOCS EVALUATION DONE", flush=True)


def evaluate(models, out_dir, conditions=CONDITIONS, baselines=BASELINES):
    """models: name -> checkpoint path. Writes {cond}__{name}.json, scores/, per_query/, links/ under out_dir."""
    for sub in ("scores", "per_query", "links"):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)
    for cond in conditions:
        pool = load_pool(cond)
        n, L = pool["frags"].shape
        f, t, o = pool["file_idx"], pool["type_idx"], pool["offset"]
        same_file = f[:, None] == f[None, :]
        same_type = t[:, None] == t[None, :]
        not_self = ~np.eye(n, dtype=bool)
        adj = same_file & (np.abs(o[:, None] - o[None, :]) == L)
        todo = [m for m in list(baselines) + list(models) if not os.path.exists(os.path.join(out_dir, f"{cond}__{m}.json"))]
        if not todo:
            continue
        scores = {}
        if any(m in baselines for m in todo):
            scores.update(E.baseline_scores(pool["frags"]))
        asym = {}
        for m, path in models.items():
            if m in todo:
                scores[m], asym[m] = E.model_scores(path, pool["frags"], DEV)
        for m in todo:
            S = scores[m]
            OUTD = out_dir
            np.save(os.path.join(OUTD, "scores", f"{cond}__{m}.npy"), S)
            ranks = {}
            for key, rel, cand in (("retrieval_all", same_file & not_self, not_self),
                                   ("retrieval_type", same_file & not_self, not_self & same_type),
                                   ("adjacency_within_file", adj, same_file & not_self),
                                   ("adjacency_pooled", adj, not_self)):
                ranks[key], nc = E.first_relevant_rank(S, rel, cand, DEV)
                ranks[f"chance_{key}"] = (rel & cand).sum(1) / np.maximum(nc, 1)
            pd.DataFrame({"file": pool["file"], "ext": pool["ext"], "offset": o,
                          **{k: v for k, v in ranks.items() if not k.startswith("chance")}}).to_csv(
                os.path.join(OUTD, "per_query", f"{cond}__{m}.csv"), index=False)
            links, chain_rep = E.chaining_report(S, f.tolist(), o.tolist(), L)
            pd.DataFrame(links, columns=["a", "b"]).to_csv(os.path.join(OUTD, "links", f"{cond}__{m}.csv"), index=False)
            Sd = S.copy()
            np.fill_diagonal(Sd, -np.inf)
            has = adj.any(1)
            res = {"condition": cond, "method": m, "block": int(L), "fragments": int(n), "files": int(len(np.unique(f))),
                   "auc_same_file_all": E.PairAUC(S, f, same_file).auc(),
                   "auc_same_file_within_type": E.PairAUC(S, f, same_file, same_type).auc(),
                   "auc_same_type": E.PairAUC(S, f, same_type).auc(),
                   **{k: E.summarize_ranks(v) for k, v in ranks.items() if not k.startswith("chance")},
                   **{f"{k}_chance_hit@1": float(np.nanmean(np.where(np.isnan(ranks[k]), np.nan, ranks[f"chance_{k}"])))
                      for k in ("retrieval_all", "retrieval_type", "adjacency_within_file", "adjacency_pooled")},
                   **E.cluster_ari(S, f), "greedy": chain_rep["full"], "greedy_oracle_stop": chain_rep["oracle_stop"],
                   "neighbor_hit1_poolwide_archived": float(adj[np.arange(n), Sd.argmax(1)][has].mean()),
                   "order_asymmetry": asym.get(m)}
            json.dump(res, open(os.path.join(OUTD, f"{cond}__{m}.json"), "w"), indent=1, default=float)
            print(cond, m, round(res["auc_same_file_within_type"], 3), round(res["retrieval_all"]["hit@1"], 3),
                  round(res["adjacency_within_file"]["hit@1"], 3), round(res["greedy"]["adjacency_recall"], 3), flush=True)


if __name__ == "__main__":
    main()
