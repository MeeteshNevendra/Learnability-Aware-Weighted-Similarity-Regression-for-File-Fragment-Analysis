"""Paired bootstrap differences of the ablation and subset models against the six-component heuristic model (seed 42).

Added on 2026-11-03, after the protocol freeze and after all results were computed. The frozen protocol defines paired
differences against hist_cosine only (results/bootstrap/local_summary.json); this script adds a comparison against the
all-six heuristic model so that ablation effects can be judged against file-sampling uncertainty. It computes nothing
new from the fragments: it reuses the stored replicates of the frozen local bootstrap (results/bootstrap/local__<m>.npz),
which share one set of file draws (seed 20261027), so the differences are paired. Seeds 43 and 44 of the heuristic
scheme are included to show the size of a difference caused by the training seed alone.

usage: python scripts/61_ablation_paired.py
Outputs: results/bootstrap/local_paired_vs_heuristic_s42.json; results/tables/T12b_ablation_paired_differences.{csv,md}
"""
import json
import os

import numpy as np
import pandas as pd

import bootlib as B
from paths import RESULTS

BOOT = os.path.join(RESULTS, "bootstrap")
TABLES = os.path.join(RESULTS, "tables")
REF = "heuristic_s42"
METHODS = [("heuristic_s43", "Heuristic, seed 43 (training seed only)"),
           ("heuristic_s44", "Heuristic, seed 44 (training seed only)"),
           ("no_hist", "Without s_hist"), ("no_ent", "Without s_ent"), ("no_ngram", "Without s_ngram"),
           ("no_comp", "Without s_comp"), ("no_pos", "Without s_pos"), ("no_type", "Without s_type"),
           ("hist_ent", "s_hist + s_ent"), ("hist_ngram", "s_hist + s_ngram"), ("hist_ent_comp", "s_hist + s_ent + s_comp"),
           ("pos_type", "s_pos + s_type"), ("comp_hist", "s_hist only"), ("comp_ngram", "s_ngram only")]
METRICS = [("auc_same_file_within_type", "Same-file AUC within type"), ("auc_same_type", "Same-type AUC"),
           ("retrieval_all_hit@1", "Retrieval Hit@1")]


def main():
    summary = json.load(open(os.path.join(BOOT, "local_summary.json"), encoding="utf-8"))
    est = {m: {k: v["estimate"] for k, v in d["metrics"].items()} for m, d in summary["methods"].items()}
    ref = np.load(os.path.join(BOOT, f"local__{REF}.npz"))
    out = {"reference": REF, "replicates": int(summary["replicates"]), "files": int(summary["files"]),
           "added": "2026-11-03, after the protocol freeze; not part of the frozen protocol, which defines paired "
                    "differences against hist_cosine only",
           "estimate": "difference of the full-sample values (method minus reference)",
           "interval": "2.5-97.5 percentiles of the paired replicate differences (same file draws)",
           "methods": {}}
    rows = []
    for m, label in METHODS:
        z = np.load(os.path.join(BOOT, f"local__{m}.npz"))
        out["methods"][m] = {}
        row = {"Model minus all-six heuristic (seed 42)": label}
        for k, klab in METRICS:
            d = z[k] - ref[k]
            if len(d) != summary["replicates"]:
                raise SystemExit(f"{m} {k}: {len(d)} replicates, expected {summary['replicates']}")
            e = est[m][k] - est[REF][k]
            ci = B.ci(d)
            out["methods"][m][k] = {"difference": float(e), "ci95": ci,
                                    "share_replicates_method_higher": float(np.mean(d > 0))}
            row[klab] = f"{e:+.3f} ({ci[0]:+.3f} to {ci[1]:+.3f})"
        rows.append(row)
    json.dump(out, open(os.path.join(BOOT, "local_paired_vs_heuristic_s42.json"), "w"), indent=1)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(TABLES, "T12b_ablation_paired_differences.csv"), index=False)
    with open(os.path.join(TABLES, "T12b_ablation_paired_differences.md"), "w", encoding="utf-8") as f:
        f.write("### Ablation and subset models minus the all-six heuristic model (seed 42), paired over test files\n\n")
        f.write(df.to_markdown(index=False) + "\n\n")
        f.write("Added after the protocol freeze (the frozen protocol compares methods with histogram cosine only). "
                "Estimate: difference of the full-sample values; parentheses: 2.5-97.5 percentiles of 1000 paired "
                "source-file bootstrap differences (the draws of the frozen local bootstrap). The intervals describe "
                "test-file sampling only; every ablation is a single training run, and the two seed rows show the "
                "size of a difference caused by the training seed alone.\n\n")
        f.write("Sources: results/bootstrap/local__*.npz; results/bootstrap/local_summary.json; "
                "results/bootstrap/local_paired_vs_heuristic_s42.json\n")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
