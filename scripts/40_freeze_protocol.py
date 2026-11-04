"""Freeze the evaluation protocol before any test-set evaluation. Writes configs/frozen_protocol.json.

Test-evaluation scripts call check_frozen(): they stop if the protocol file is missing or if a frozen script has
changed, unless the change is logged as an amendment (configs/protocol_amendments.log) with its reason.
"""
import datetime as dt
import hashlib
import json
import os
import sys

from paths import CONFIGS, ROOT

FROZEN_SCRIPTS = ["scripts/evallib.py", "scripts/bootlib.py", "scripts/components.py", "scripts/model.py",
                  "scripts/30_train.py", "scripts/31_queue.py", "scripts/41_evaluate_local.py",
                  "scripts/51_govdocs_eval.py", "scripts/60_bootstrap.py"]
PATH = os.path.join(CONFIGS, "frozen_protocol.json")
AMEND = os.path.join(CONFIGS, "protocol_amendments.log")

PROTOCOL = {
    "statement": "Frozen before any model was evaluated on test fragments, test pairs, the reconstruction pool or "
                 "Govdocs1. Weights, seeds, hyperparameters, candidate rules, shortlist rule and analysis definitions "
                 "are not changed after test results are seen.",
    "models": {
        "reported_seed_settings": ["heuristic_s42", "heuristic_s43", "heuristic_s44", "uniform_s42", "learnability_s42"],
        "robustness_seeds_not_in_manuscript": ["uniform_s43", "uniform_s44", "learnability_s43", "learnability_s44"],
        "component_models": ["comp_hist", "comp_ent", "comp_ngram", "comp_comp", "comp_pos", "comp_type"],
        "ablations": ["no_hist", "no_ent", "no_ngram", "no_comp", "no_pos", "no_type", "hist_ent", "hist_ngram",
                      "hist_ent_comp", "pos_type"],
        "leave_one_type_out": ["loto_heic_s42"],
        "govdocs1_models": ["heuristic_s42", "uniform_s42", "learnability_s42", "heuristic_s43", "heuristic_s44"]},
    "baselines": {"hist_cosine": "cosine of 256-bin byte counts", "entropy_sim": "1-|H_i-H_j|/8",
                  "ngram_cosine": "cosine of 512-bin hashed bigram counts (archived hash table)",
                  "compress_sim": "clip(1-|c_i-c_j|,0,1), classic zlib level 6",
                  "uniform_fusion_4": "unweighted mean of the four",
                  "random": "0.5*(R+R^T), R ~ U(0,1), numpy default_rng(0)"},
    "regression": {"pairs": "80,000 sampled test pairs, ordered as sampled (archived convention)",
                   "metrics": ["rmse", "mae", "r2", "pearson", "spearman"],
                   "breakdowns": ["pair_mode", "actual_relation"],
                   "order_reversal": "predictions for (b,a) also computed; |p(a,b)-p(b,a)| reported; metrics of "
                                     "0.5*(p(a,b)+p(b,a)) reported as a secondary analysis",
                   "target": "each model against its own scheme's target"},
    "provenance": {
        "pool": "distinct test fragments appearing in at least one sampled test pair (archived convention)",
        "model_scores": "S = 0.5*(P+P^T) with P[i,j]=head(z_i,z_j) (archived convention, declared before testing)",
        "ties": "fixed pseudo-random tie-breaking key (numpy default_rng(12345) permutation of pool indices)",
        "auc": {"same_file_all": "pos = same source file, all unordered pairs",
                "same_file_within_type": "pairs with equal recorded extension only",
                "same_file_within_type_jpg_jpeg_merged": "sensitivity analysis: JPG and JPEG treated as one group",
                "same_type": "pos = equal recorded extension", "same_type_jpg_jpeg_merged": "sensitivity analysis"},
        "retrieval": {"relevant": "other fragments of the query's source file",
                      "all_candidates": "every other pool fragment (practical)",
                      "type_restricted": "ORACLE: candidates with the query's recorded extension",
                      "type_restricted_jpg_jpeg_merged": "ORACLE sensitivity analysis", "metrics": ["hit@1", "hit@5", "hit@10", "mrr"]},
        "adjacency": {"relevant": "same source file and |offset difference| == 512",
                      "within_file": "ORACLE (archived): candidates restricted to the query's source file",
                      "pooled": "practical: candidates = every other pool fragment; no source identity used",
                      "queries": "fragments with at least one adjacent retained block in the pool"},
        "clustering": "average-linkage agglomerative clustering on D = max(S) - S; ORACLE number of clusters = "
                      "number of source files; ARI and NMI",
        "jpg_jpeg": "primary analyses keep JPG and JPEG as separate recorded-extension groups; merged versions are "
                    "labelled sensitivity analyses",
        "fp32_sensitivity": "secondary: heuristic_s42, uniform_s42 and learnability_s42 rescored with encoder and "
                            "pair head in float32 (no autocast); primary results use the archived autocast scoring, "
                            "whose float16 sigmoid output creates ties"},
    "shortlist_workflow": {"shortlist_model": "heuristic_s42 (archived workflow); uniform_s42 and learnability_s42 secondary",
                           "rerank": "hist_cosine", "grid": [25, 50, 100, 200, 400, 800, 1600, 3200],
                           "selection": "M maximising validation MRR on the validation pool (archived rule); ties -> smaller M",
                           "tasks": ["same-file retrieval (all candidates)", "pooled adjacency"],
                           "effort": "rank of the first relevant candidate = candidates inspected; candidates needed "
                                     "for recall 0.5/0.8/0.9/0.95; shortlist recall at the frozen M"},
    "reconstruction": {"pool": "test files with >= 24 complete aligned blocks; one run of 24 consecutive blocks per "
                               "file; start block = random.Random('recon-v1|'+collection_id).randrange(0, n_blocks-24+1)",
                       "chaining": "unordered pairs in decreasing score (stable); accept if both degrees < 2 and no "
                                   "cycle; stop at n-1 links; scores only (no source identity or offsets)",
                       "metrics": ["adjacency_recall and join_precision over all accepted links (archived procedure, "
                                   "links accepted until n-1, i.e. one path when complete)",
                                   "oracle_stop: the same link order stopped after n - F links (F = known number of "
                                   "source files) -> chain_purity_weighted, share of fragments in pure chains, "
                                   "files_recovered_as_exact_ordered_sequence (ordering check only; no decoder validation)"],
                       "design_note": "the oracle stop was added after a validation-only dry run showed that the "
                                      "archived n-1 rule always yields a single path, so purity is undefined there",
                       "also": ["pooled adjacency ranking", "within-file adjacency ranking (oracle)"]},
    "loto_heic": {"regression": "test pairs with / without a HEIC fragment", "same_type_auc": "pairs with >= 1 HEIC fragment",
                  "same_file_auc": "pairs of two HEIC fragments", "comparison": ["heuristic_s42", "heuristic_s43",
                                                                              "heuristic_s44", "hist_cosine"]},
    "govdocs1": {"conditions": ["aligned", "unaligned", "encrypted", "compressed", "b4096"],
                 "b4096": "model: mean of eight 512-byte window embeddings, re-normalised, then the pair head; "
                          "baselines on the full 4096-byte fragments",
                 "tasks": "same as local provenance + greedy chaining; no retraining"},
    "bootstrap": {"replicates": 1000, "seed": 20261027, "unit": "source file (cluster bootstrap, with replacement)",
                  "duplicates": "pair weights c_a*c_b for different files and c_a within a file; pairs between two "
                                "copies of one file are never formed; query metrics averaged over queries of drawn "
                                "files with multiplicity, using ranks computed on the full pool",
                  "paired_differences": "identical file draws for every method; difference method - hist_cosine",
                  "chaining": "full-pool chaining fixed; files reweighted through per-file contributions (correct and "
                              "true links per file; accepted links split half per endpoint file; purity attributed to "
                              "each chain's majority file)",
                  "ari": "per replicate the distinct drawn files are re-clustered (duplicates collapsed, so no "
                         "artificial same-file matches); pool-size dependent, reported with that caveat",
                  "interval": "percentile 2.5-97.5"},
}


def sha(p):
    return hashlib.sha256(open(os.path.join(ROOT, p), "rb").read()).hexdigest()


def check_frozen():
    if not os.path.exists(PATH):
        sys.exit("frozen_protocol.json missing: run scripts/40_freeze_protocol.py before test evaluation")
    fz = json.load(open(PATH))
    changed = [p for p, h in fz["script_sha256"].items() if sha(p) != h]
    amended = open(AMEND, encoding="utf-8").read() if os.path.exists(AMEND) else ""
    unlogged = [p for p in changed if f"{p} {sha(p)}" not in amended]
    if unlogged:
        sys.exit(f"frozen scripts changed without a logged amendment: {unlogged}")
    return fz


def amend(path, reason):
    with open(AMEND, "a", encoding="utf-8") as f:
        f.write(f"{dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')} {path} {sha(path)} :: {reason}\n")


def main():
    if os.path.exists(PATH):
        sys.exit("protocol already frozen; use amend() for documented corrections")
    out = {"frozen_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), **PROTOCOL,
           "script_sha256": {p: sha(p) for p in FROZEN_SCRIPTS if os.path.exists(os.path.join(ROOT, p))}}
    json.dump(out, open(PATH, "w"), indent=1)
    print("frozen:", PATH)


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "amend":
        amend(sys.argv[2], sys.argv[3])
    else:
        main()
