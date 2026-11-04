"""Result tables (CSV + Markdown) from saved outputs only. Missing inputs are reported, never filled in.
Output: results/tables/*.csv, *.md and results/tables/README.md (index with source files)."""
import glob
import json
import os

import numpy as np
import pandas as pd

from paths import FRAGMENTS, MANIFESTS, PAIRS, RESULTS, RUNS

SPLIT = os.environ.get("FSR_EVAL_SPLIT", "test")
T = os.environ.get("FSR_TABLE_OUT", os.path.join(RESULTS, "tables"))
os.makedirs(T, exist_ok=True)
L = os.environ.get("FSR_LOCAL_OUT", os.path.join(RESULTS, "local"))
B = os.environ.get("FSR_BOOT_OUT", os.path.join(RESULTS, "bootstrap"))
G = os.path.join(RESULTS, "govdocs1")
INDEX = []
LABEL = {"hist_cosine": "Histogram cosine", "entropy_sim": "Entropy similarity", "ngram_cosine": "Bigram cosine",
         "compress_sim": "Compression-ratio similarity", "uniform_fusion_4": "Mean of four byte measures",
         "random": "Random scores", "heuristic_s42": "Model, heuristic (seed 42)", "heuristic_s43": "Model, heuristic (seed 43)",
         "heuristic_s44": "Model, heuristic (seed 44)", "uniform_s42": "Model, uniform (seed 42)",
         "learnability_s42": "Model, learnability-aware (seed 42)", "uniform_s43": "Model, uniform (seed 43)",
         "uniform_s44": "Model, uniform (seed 44)", "learnability_s43": "Model, learnability-aware (seed 43)",
         "learnability_s44": "Model, learnability-aware (seed 44)", "comp_hist": "Model, s_hist target",
         "comp_ent": "Model, s_ent target", "comp_ngram": "Model, s_ngram target", "comp_comp": "Model, s_comp target",
         "comp_pos": "Model, s_pos target", "comp_type": "Model, s_type target", "pos_type": "Model, s_pos + s_type target",
         "hist_ent": "Model, s_hist + s_ent target", "hist_ngram": "Model, s_hist + s_ngram target",
         "hist_ent_comp": "Model, s_hist + s_ent + s_comp target", "loto_heic_s42": "Model, heuristic without HEIC (seed 42)"}
for c in ("hist", "ent", "ngram", "comp", "pos", "type"):
    LABEL[f"no_{c}"] = f"Model, heuristic without s_{c}"
ORDER = ["random", "entropy_sim", "compress_sim", "ngram_cosine", "hist_cosine", "uniform_fusion_4", "heuristic_s42",
         "heuristic_s43", "heuristic_s44", "uniform_s42", "learnability_s42", "uniform_s43", "uniform_s44",
         "learnability_s43", "learnability_s44", "pos_type", "comp_hist"]


def jl(p):
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def save(df, name, title, sources, note=""):
    df.to_csv(os.path.join(T, f"{name}.csv"), index=False)
    with open(os.path.join(T, f"{name}.md"), "w", encoding="utf-8") as f:
        f.write(f"### {title}\n\n{df.to_markdown(index=False)}\n\n")
        if note:
            f.write(note + "\n\n")
        f.write("Sources: " + "; ".join(sources) + "\n")
    INDEX.append((name, title, sources))


def fmt(x, d=3):
    return "" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def ci_str(est, ci, d=3):
    if est is None:
        return ""
    return f"{est:.{d}f} ({ci[0]:.{d}f}-{ci[1]:.{d}f})" if ci and not np.isnan(ci[0]) else f"{est:.{d}f}"


def dataset_tables():
    man = pd.read_csv(os.path.join(MANIFESTS, "source_manifest.csv"))
    order = ["PNG", "JPG", "HEIC", "PDF", "MP4", "DOCX", "PPTX", "JPEG", "WEBM", "CSV"]
    rows = []
    frs = {s: pd.read_csv(os.path.join(FRAGMENTS, f"{s}_fragments.csv")) for s in ("train", "val", "test")}
    for g in order:
        sub = man[man.extension_group == g]
        r = {"Extension group": g, "Source": "; ".join(sorted(sub.source_name.unique()))}
        for s in ("train", "val", "test"):
            r[f"Files {s}"] = int((sub.partition == s).sum())
        r["Files total"] = int(len(sub))
        for s in ("train", "val", "test"):
            r[f"Fragments {s}"] = int((frs[s].ext == g.lower()).sum())
        r["Median size (bytes)"] = int(sub.size_bytes.median())
        rows.append(r)
    tot = {"Extension group": "Total", "Source": ""}
    for k in rows[0]:
        if k.startswith(("Files", "Fragments")):
            tot[k] = int(sum(r[k] for r in rows))
    tot["Median size (bytes)"] = int(man.size_bytes.median())
    rows.append(tot)
    save(pd.DataFrame(rows), "T01_dataset", "Replacement collection: files and retained fragments by extension group and partition",
         ["data/manifests/source_manifest.csv", "data/fragments/*_fragments.csv"])
    ps = jl(os.path.join(PAIRS, "pair_summary.json"))
    rows = []
    for s in ("train", "val", "test"):
        p = ps[s]
        rows.append({"Split": s, "Pairs": p["pairs"], "Fragments in split": p["fragments_in_split"],
                     "Distinct fragments in pairs": p["distinct_fragments_in_pairs"],
                     "Random-mode pairs that are same-file": p["relation_by_mode"]["random"].get("same_file", 0),
                     "Random-mode same-extension pairs": p["relation_by_mode"]["random"].get("same_ext_diff_file", 0),
                     "Repeated unordered pairs": p["repeated_unordered_pairs"], "Self-pairs": p["self_pairs"],
                     "Adjacent pairs": p["adjacent_pairs"],
                     **{f"mean {k}": round(v, 3) for k, v in p["component_means"].items()}})
    save(pd.DataFrame(rows), "T02_pairs", "Sampled pairs (four modes, 25% each)", ["data/pairs/pair_summary.json"])


def learnability_table():
    lw = jl(os.path.join(RUNS, "learnability_weights.json"))
    rows = []
    for run, comp in (("comp_hist", "s_hist"), ("comp_ent", "s_ent"), ("comp_ngram", "s_ngram"), ("comp_comp", "s_comp"),
                      ("comp_pos", "s_pos"), ("comp_type", "s_type")):
        s = jl(os.path.join(RUNS, run, "summary_val.json"))
        r = jl(os.path.join(L, "regression", f"{run}.json"))
        rows.append({"Component": comp, "Validation R2": fmt(s["best_val_r2"]) if s else "not run",
                     "Test R2": fmt(r["test"]["r2"]) if r else "", "Test r": fmt(r["test"]["pearson"]) if r else "",
                     "Derived weight": fmt((lw["weights"] or {}).get(comp, 0.0)) if lw and lw.get("weights") is not None else "",
                     "Best epoch / epochs run": f"{s['best_epoch']}/{s['epochs_run']}" if s else ""})
    note = ("Weights: w_k = max(R2_val,k, 0) / sum of positive values (validation only). "
            + ("All validation R2 were non-positive: learnability-aware target undefined." if lw and lw["undefined"] else ""))
    save(pd.DataFrame(rows), "T03_learnability", "Component learnability (seed 42) and derived weights",
         ["runs/comp_*/summary_val.json", "results/local/regression/comp_*.json", "runs/learnability_weights.json"], note)


def regression_tables():
    def summ(names):
        rs = [jl(os.path.join(L, "regression", f"{n}.json")) for n in names]
        rs = [r for r in rs if r]
        if not rs:
            return None
        out = {"Runs": len(rs)}
        for k in ("rmse", "mae", "r2", "pearson", "spearman"):
            v = [r["test"][k] for r in rs]
            out[k] = f"{np.mean(v):.3f} ± {np.std(v, ddof=1):.3f}" if len(v) > 1 else f"{v[0]:.3f}"
        return out
    rows = []
    for lab, names in (("Heuristic weights (seeds 42, 43, 44)", ["heuristic_s42", "heuristic_s43", "heuristic_s44"]),
                       ("Uniform weights (seed 42)", ["uniform_s42"]), ("Learnability-aware weights (seed 42)", ["learnability_s42"])):
        s = summ(names)
        if s:
            rows.append({"Weighting scheme": lab, **s})
    save(pd.DataFrame(rows), "T04_regression", "Agreement with each scheme's own composite target (test pairs)",
         ["results/local/regression/<run>.json"], "Each scheme defines a different target; values are not comparable across schemes as evidence of forensic value.")
    rows = []
    for lab, names in (("Heuristic", ["heuristic_s42", "heuristic_s43", "heuristic_s44"]),
                       ("Uniform", ["uniform_s42", "uniform_s43", "uniform_s44"]),
                       ("Learnability-aware", ["learnability_s42", "learnability_s43", "learnability_s44"])):
        s = summ(names)
        if s:
            rows.append({"Weighting scheme (robustness analysis)": lab, **s})
    if rows:
        save(pd.DataFrame(rows), "T04b_regression_three_seeds", "Additional robustness analysis: all schemes with seeds 42, 43, 44",
             ["results/local/regression/<run>.json"], "Not part of the manuscript's original comparison (uniform and learnability-aware were single runs there).")
    rows = []
    for n in ("heuristic_s42", "uniform_s42", "learnability_s42"):
        r = jl(os.path.join(L, "regression", f"{n}.json"))
        if not r:
            continue
        for grp, key in (("pair mode", "by_pair_mode"), ("actual relation", "by_relation")):
            for m, v in r[key].items():
                rows.append({"Model": LABEL[n], "Breakdown": grp, "Group": m, "Pairs": v["n"], "RMSE": fmt(v["rmse"]),
                             "R2": fmt(v["r2"]), "r": fmt(v["pearson"]), "rho": fmt(v["spearman"])})
        rows.append({"Model": LABEL[n], "Breakdown": "order reversal", "Group": "mean |p(a,b)-p(b,a)|",
                     "Pairs": "", "RMSE": fmt(r["order_reversal"]["mean_abs_diff"], 4), "R2": "", "r": "", "rho": ""})
    save(pd.DataFrame(rows), "T05_regression_by_mode", "Regression by pair-sampling mode and by actual provenance relation",
         ["results/local/regression/<run>.json"])


def provenance_tables():
    bs = jl(os.path.join(B, "local_summary.json"))
    methods = [m for m in ORDER if os.path.exists(os.path.join(L, "provenance", f"{SPLIT}__{m}.json"))]
    rows, rows2 = [], []
    for m in methods:
        p = jl(os.path.join(L, "provenance", f"{SPLIT}__{m}.json"))
        b = (bs or {}).get("methods", {}).get(m, {}).get("metrics", {})

        def c(key, est):
            return ci_str(est, b.get(key, {}).get("ci95"))
        rows.append({"Scoring method": LABEL[m],
                     "Same-file AUC": c("auc_same_file_all", p["auc_same_file_all"]),
                     "Same-file AUC within type": c("auc_same_file_within_type", p["auc_same_file_within_type"]),
                     "Same-type AUC": c("auc_same_type", p["auc_same_type"]),
                     "Retrieval Hit@1": c("retrieval_all_hit@1", p["retrieval_all"]["hit@1"]),
                     "Retrieval Hit@1, type known (oracle)": c("retrieval_type_hit@1", p["retrieval_type"]["hit@1"]),
                     "Adjacent Hit@1, source known (oracle)": c("adjacency_within_file_hit@1", p["adjacency_within_file"]["hit@1"]),
                     "Adjacent Hit@1, pooled": c("adjacency_pooled_hit@1", p["adjacency_pooled"]["hit@1"]),
                     "ARI (oracle cluster count)": c("ari", p["ari"])})
        rows2.append({"Scoring method": LABEL[m], **{f"{t} {k}": fmt(p[t][k]) for t in ("retrieval_all", "retrieval_type",
                      "adjacency_within_file", "adjacency_pooled") for k in ("hit@1", "hit@5", "hit@10", "mrr")},
                      "retrieval queries": p["retrieval_all"]["queries"], "adjacency queries": p["adjacency_within_file"]["queries"],
                      "chance adjacent Hit@1 (within file)": fmt(p["adjacency_within_file"]["chance_hit@1"]),
                      "chance adjacent Hit@1 (pooled)": fmt(p["adjacency_pooled"]["chance_hit@1"], 4)})
    note = "Parentheses: 95% source-file cluster-bootstrap CI (1000 replicates). 'Oracle' columns use the known source file or recorded extension to restrict candidates and do not describe a deployed pipeline." if bs else "Bootstrap CIs not yet computed."
    save(pd.DataFrame(rows), "T06_provenance", "Provenance tasks on the test pool", ["results/local/provenance/test__*.json",
         "results/bootstrap/local_summary.json"], note)
    save(pd.DataFrame(rows2), "T06b_provenance_rankings", "Ranking metrics (Hit@1/5/10, MRR) and chance levels",
         ["results/local/provenance/test__*.json"])
    if bs and bs.get("paired_differences_vs_hist_cosine"):
        rows = []
        for m, d in bs["paired_differences_vs_hist_cosine"].items():
            if m not in ORDER:
                continue
            r = {"Method minus histogram cosine": LABEL.get(m, m)}
            for k in ("auc_same_file_within_type", "retrieval_all_hit@1", "adjacency_within_file_hit@1", "adjacency_pooled_hit@1"):
                if k in d:
                    r[k] = f"{d[k]['mean_difference']:+.3f} ({d[k]['ci95'][0]:+.3f} to {d[k]['ci95'][1]:+.3f})"
            rows.append(r)
        save(pd.DataFrame(rows), "T07_paired_differences", "Paired bootstrap differences against histogram cosine (same file draws)",
             ["results/bootstrap/local_summary.json"])
    rows = []
    for m in methods:
        p = jl(os.path.join(L, "provenance", f"{SPLIT}__{m}.json"))
        rows.append({"Scoring method": LABEL[m], "Same-file AUC within type (JPG, JPEG separate)": fmt(p["auc_same_file_within_type"]),
                     "... JPG+JPEG merged": fmt(p["auc_same_file_within_type_jpg_jpeg_merged"]),
                     "Same-type AUC (separate)": fmt(p["auc_same_type"]), "Same-type AUC (merged)": fmt(p["auc_same_type_jpg_jpeg_merged"]),
                     "Hit@1 type-restricted (separate)": fmt(p["retrieval_type"]["hit@1"]),
                     "Hit@1 type-restricted (merged)": fmt(p["retrieval_type_merged"]["hit@1"])})
    save(pd.DataFrame(rows), "T08_jpg_jpeg_sensitivity", "Sensitivity analysis: JPG and JPEG merged into one extension group",
         ["results/local/provenance/test__*.json"], "JPG and JPEG are extension groups of one format (JPEG/JFIF); the primary analysis keeps the recorded extensions separate.")
    rows = []
    for m in ("heuristic_s42", "uniform_s42", "learnability_s42"):
        a, f = jl(os.path.join(L, "provenance", f"{SPLIT}__{m}.json")), jl(os.path.join(L, "provenance", f"fp32__{m}.json"))
        if a and f:
            rows.append({"Model": LABEL[m], "Within-type AUC (fp16 autocast, primary)": fmt(a["auc_same_file_within_type"]),
                         "Within-type AUC (fp32)": fmt(f["auc_same_file_within_type"]),
                         "Retrieval Hit@1 (primary)": fmt(a["retrieval_all"]["hit@1"]), "Retrieval Hit@1 (fp32)": fmt(f["retrieval_all"]["hit@1"]),
                         "Distinct score values (fp32)": f["distinct_score_values"]})
    if rows:
        save(pd.DataFrame(rows), "T09_fp32_sensitivity", "Sensitivity of provenance results to float16 score quantisation",
             ["results/local/provenance/test__*.json", "results/local/provenance/fp32__*.json"])


def by_group_table():
    rows = []
    for m in ("hist_cosine", "uniform_fusion_4", "ngram_cosine", "heuristic_s42", "uniform_s42", "learnability_s42", "random"):
        p = os.path.join(L, "per_query", f"{SPLIT}__{m}.csv")
        if not os.path.exists(p):
            continue
        d = pd.read_csv(p)
        for g, x in d.groupby("ext"):
            r1 = x.rank_retrieval_all.dropna()
            ra = x.rank_adjacency_within_file.dropna()
            rp = x.rank_adjacency_pooled.dropna()
            rows.append({"Method": LABEL[m], "Extension group": g.upper(), "Retrieval queries": len(r1),
                         "Retrieval Hit@1": fmt(float((r1 < 1).mean())) if len(r1) else "",
                         "Retrieval Hit@10": fmt(float((r1 < 10).mean())) if len(r1) else "",
                         "Adjacency queries": len(ra), "Adjacent Hit@1 within file (oracle)": fmt(float((ra < 1).mean())) if len(ra) else "",
                         "Adjacent Hit@1 pooled": fmt(float((rp < 1).mean())) if len(rp) else ""})
    if rows:
        save(pd.DataFrame(rows), "T06c_by_extension_group", "Retrieval and adjacency by recorded extension group (test pool)",
             [f"results/local/per_query/{SPLIT}__*.csv"])


def shortlist_table():
    s = jl(os.path.join(L, "shortlist.json"))
    if not s:
        return
    rows = []
    for m in ("heuristic_s42", "uniform_s42", "learnability_s42"):
        if m not in s:
            continue
        for task, v in s[m].items():
            t = v["test"]
            rows.append({"Shortlist model": LABEL[m], "Task": task, "Selected M (validation)": v["selected_M"],
                         "M / candidates": fmt(v["shortlist_fraction_of_candidates"]),
                         "Direct histogram Hit@1": fmt(t["direct_hist_cosine"]["summary"]["hit@1"]),
                         "Shortlist+histogram Hit@1": fmt(t["two_stage"]["summary"]["hit@1"]),
                         "Direct learned Hit@1": fmt(t["direct_learned"]["summary"]["hit@1"]),
                         "Direct histogram MRR": fmt(t["direct_hist_cosine"]["summary"]["mrr"]),
                         "Shortlist+histogram MRR": fmt(t["two_stage"]["summary"]["mrr"]),
                         "Shortlist recall": fmt(t["two_stage"]["summary"]["shortlist_recall"]),
                         "Candidates for 80% recall: histogram / learned / shortlist": "{} / {} / {}".format(
                             t["direct_hist_cosine"]["candidates_for_recall"]["0.8"], t["direct_learned"]["candidates_for_recall"]["0.8"],
                             t["two_stage"]["candidates_for_recall"]["0.8"]),
                         "Mean rank of first match: histogram / learned": "{:.1f} / {:.1f}".format(
                             t["direct_hist_cosine"]["summary"]["mean_rank"], t["direct_learned"]["summary"]["mean_rank"])})
    save(pd.DataFrame(rows), "T10_shortlist_effort", "Learned shortlist + histogram re-ranking versus direct ranking (examiner effort)",
         ["results/local/shortlist.json", "results/local/per_query/shortlist__*.csv"],
         "M chosen by maximum validation MRR over {25,...,3200} (archived rule) and frozen before testing. 'Candidates for 80% recall' = smallest k such that 80% of queries have a true match within the top k; None = not reachable within the shortlist.")


def reconstruction_table():
    pool = jl(os.path.join(L, "reconstruction", "pool.json"))
    if not pool:
        return
    bs = jl(os.path.join(B, "recon_summary.json"))
    rows = []
    for m in ORDER:
        r = jl(os.path.join(L, "reconstruction", f"{m}.json"))
        if not r:
            continue
        b = (bs or {}).get("methods", {}).get(m, {})
        rows.append({"Scoring method": LABEL[m],
                     "Adjacency recall": ci_str(r["adjacency_recall"], b.get("adjacency_recall", {}).get("ci95")),
                     "Join precision": ci_str(r["join_precision"], b.get("join_precision", {}).get("ci95")),
                     "Correct / accepted links": f"{r['correct_links']} / {r['links']}",
                     "Chain purity (oracle stop)": ci_str(r["oracle_stop"]["chain_purity_weighted"],
                                                          b.get("chain_purity_weighted", {}).get("ci95")),
                     "Adjacency recall / precision (oracle stop)": f"{r['oracle_stop']['adjacency_recall']:.3f} / "
                                                                   f"{r['oracle_stop']['join_precision']:.3f}",
                     "Files as exact ordered run (oracle stop)": f"{r['oracle_stop']['files_recovered_as_exact_ordered_sequence']} / {r['files']}",
                     "Pooled adjacent Hit@1": fmt(r["adjacency_pooled"]["hit@1"]),
                     "Within-file adjacent Hit@1 (oracle)": fmt(r["adjacency_within_file_oracle"]["hit@1"])})
    note = (f"Test-only pool: {pool['files']} files with >= 24 complete blocks, one run of 24 consecutive 512-byte blocks each "
            f"({pool['fragments']} fragments, {pool['true_adjacent_pairs']} true adjacent pairs); "
            f"{len(pool['excluded_files_fewer_than_24_blocks'])} test files excluded (fewer than 24 blocks). "
            "Greedy chaining uses scores only. Recall and precision: archived procedure (links accepted until n-1, which "
            "ends in a single path). Oracle stop: the same link order stopped after n - F links (F = known number of "
            "source files) so that chains can be scored for purity. 'Exact ordered run' checks order only; no decoder "
            "validation was possible for mid-file runs, so no file reconstruction is claimed.")
    save(pd.DataFrame(rows), "T11_reconstruction", "Greedy chaining on consecutive held-out blocks (adjacency recovery)",
         ["results/local/reconstruction/*.json", "results/bootstrap/recon_summary.json"], note)


def ablation_table():
    rows = []
    for name, lab in (("heuristic_s42", "All six (heuristic)"), ("no_hist", "Without s_hist"), ("no_ent", "Without s_ent"),
                      ("no_ngram", "Without s_ngram"), ("no_comp", "Without s_comp"), ("no_pos", "Without s_pos"),
                      ("no_type", "Without s_type"), ("hist_ent", "s_hist + s_ent (0.75, 0.25)"),
                      ("hist_ngram", "s_hist + s_ngram (0.40, 0.60)"), ("hist_ent_comp", "s_hist + s_ent + s_comp (0.45, 0.20, 0.35)"),
                      ("pos_type", "s_pos + s_type (0.60, 0.40)"), ("comp_hist", "s_hist only"), ("comp_ngram", "s_ngram only")):
        r = jl(os.path.join(L, "regression", f"{name}.json"))
        p = jl(os.path.join(L, "provenance", f"{SPLIT}__{name}.json"))
        if not r:
            rows.append({"Target components": lab, "R2": "not run"})
            continue
        rows.append({"Target components": lab, "R2": fmt(r["test"]["r2"]), "r": fmt(r["test"]["pearson"]),
                     "rho": fmt(r["test"]["spearman"]),
                     "Same-file AUC within type": fmt(p["auc_same_file_within_type"]) if p else "",
                     "Same-type AUC": fmt(p["auc_same_type"]) if p else "",
                     "Retrieval Hit@1": fmt(p["retrieval_all"]["hit@1"]) if p else ""})
    save(pd.DataFrame(rows), "T12_ablation", "Component removal and subset targets (seed 42)",
         ["results/local/regression/*.json", "results/local/provenance/test__*.json"],
         "Removing a component changes the target only; the network input is always the raw bytes.")


def loto_table():
    lo = jl(os.path.join(L, "loto_heic.json"))
    if not lo:
        return
    rows = []
    for m in ("loto_heic_s42", "heuristic_s42", "heuristic_s43", "heuristic_s44", "hist_cosine"):
        if m not in lo:
            continue
        v = lo[m]
        rows.append({"Method": LABEL[m], "R2 on test pairs with HEIC": fmt(v.get("regression_pairs_with_heic", {}).get("r2")),
                     "R2 on test pairs without HEIC": fmt(v.get("regression_pairs_without_heic", {}).get("r2")),
                     "Same-type AUC, pairs with HEIC": fmt(v["same_type_auc_pairs_with_heic"]),
                     "Same-file AUC among HEIC fragments": fmt(v["same_file_auc_heic_pairs"])})
    save(pd.DataFrame(rows), "T13_loto_heic", "Leave-one-type-out: HEIC removed from training and validation pairs",
         ["results/local/loto_heic.json", "results/local/regression/*.json"],
         f"Test pool: {lo['heic_files']} HEIC files, {lo['pool_heic_fragments']} HEIC fragments.")


def govdocs_tables():
    rep = jl(os.path.join(RESULTS, "..", "data", "govdocs1", "build_report.json"))
    bs = jl(os.path.join(B, "govdocs_summary.json"))
    methods = ["random", "entropy_sim", "compress_sim", "ngram_cosine", "hist_cosine", "uniform_fusion_4", "heuristic_s42",
               "heuristic_s43", "heuristic_s44", "uniform_s42", "learnability_s42"]
    rows = []
    for m in methods:
        r = jl(os.path.join(G, f"aligned__{m}.json"))
        if not r:
            continue
        b = (bs or {}).get("conditions", {}).get("aligned", {}).get("methods", {}).get(m, {})

        def c(k, est):
            return ci_str(est, b.get(k, {}).get("ci95"))
        rows.append({"Scoring method": LABEL[m], "Same-file AUC within type": c("auc_same_file_within_type", r["auc_same_file_within_type"]),
                     "Same-type AUC": c("auc_same_type", r["auc_same_type"]), "Retrieval Hit@1": c("retrieval_all_hit@1", r["retrieval_all"]["hit@1"]),
                     "Adjacent Hit@1 (within file, oracle)": c("adjacency_within_file_hit@1", r["adjacency_within_file"]["hit@1"]),
                     "Adjacent Hit@1 (pooled)": c("adjacency_pooled_hit@1", r["adjacency_pooled"]["hit@1"]),
                     "Greedy adjacency recall": c("greedy_adjacency_recall", r["greedy"]["adjacency_recall"]),
                     "Join precision": c("greedy_join_precision", r["greedy"]["join_precision"]), "ARI": fmt(r["ari"])})
    if rows:
        arch = (rep or {}).get("archives", {})
        same = bool(arch) and all(v.get("identical") for v in arch.values())
        check = (f"Archives {', '.join(sorted(arch))} are byte-identical to the archived copies (SHA-256 values in "
                 "data/govdocs1/build_report.json)." if same else "Archive comparison: see data/govdocs1/build_report.json.")
        save(pd.DataFrame(rows), "T14_govdocs_aligned", "Govdocs1 external test set, aligned 512-byte runs (no retraining)",
             ["results/govdocs1/aligned__*.json", "results/bootstrap/govdocs_summary.json"],
             f"150 files, 10 extension groups, 24 consecutive blocks per file. {check}")
    rows = []
    for m in methods:
        r = {"Scoring method": LABEL[m]}
        for cond in ("aligned", "unaligned", "encrypted", "compressed", "b4096"):
            x = jl(os.path.join(G, f"{cond}__{m}.json"))
            r[cond] = (f"{x['auc_same_file_within_type']:.2f} / {x['adjacency_within_file']['hit@1']:.2f} / "
                       f"{x['greedy']['adjacency_recall']:.2f}") if x else ""
        rows.append(r)
    note = "Cells: same-file AUC within type / within-file adjacent Hit@1 (oracle) / greedy adjacency recall."
    if rep:
        note += (f" Compressed condition: {rep['files_long_enough_after_compression']} of {rep['n_files']} files long enough. "
                 f"Duplicates with the replacement collection: {rep['duplicates_with_replacement_collection']}.")
    save(pd.DataFrame(rows), "T15_govdocs_conditions", "Govdocs1 under alignment, encryption, compression and 4096-byte blocks",
         ["results/govdocs1/*.json", "data/govdocs1/build_report.json"], note)


def cost_table():
    c = jl(os.path.join(L, "cost.json"))
    if c:
        save(pd.DataFrame([{k: (round(v, 3) if isinstance(v, float) else v) for k, v in c.items()}]), "T16_cost",
             "Measured inference and ranking cost", ["results/local/cost.json"])


def main():
    for fn in (dataset_tables, learnability_table, regression_tables, provenance_tables, by_group_table, shortlist_table,
               reconstruction_table, ablation_table, loto_table, govdocs_tables, cost_table):
        try:
            fn()
        except Exception as e:  # noqa: BLE001 - a missing input must not hide other tables
            print("table skipped:", fn.__name__, type(e).__name__, e)
    # T12b is written by scripts/61_ablation_paired.py (an analysis added after the freeze); listed here if present
    if os.path.exists(os.path.join(T, "T12b_ablation_paired_differences.md")):
        INDEX.append(("T12b_ablation_paired_differences",
                      "Ablation and subset models minus the all-six heuristic model, paired bootstrap (added after the "
                      "freeze; written by scripts/61_ablation_paired.py)",
                      ["results/bootstrap/local__*.npz", "results/bootstrap/local_paired_vs_heuristic_s42.json"]))
    with open(os.path.join(T, "README.md"), "w", encoding="utf-8") as f:
        f.write("# Result tables\n\nGenerated by `python scripts/70_tables.py` from saved outputs.\n\n")
        for name, title, src in INDEX:
            f.write(f"- `{name}`: {title} (from {', '.join(src)})\n")
    print("tables:", [i[0] for i in INDEX])


if __name__ == "__main__":
    main()
