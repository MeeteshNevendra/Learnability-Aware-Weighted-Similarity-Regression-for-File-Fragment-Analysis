### Ablation and subset models minus the all-six heuristic model (seed 42), paired over test files

| Model minus all-six heuristic (seed 42)   | Same-file AUC within type   | Same-type AUC             | Retrieval Hit@1           |
|:------------------------------------------|:----------------------------|:--------------------------|:--------------------------|
| Heuristic, seed 43 (training seed only)   | +0.001 (-0.008 to +0.014)   | -0.000 (-0.012 to +0.009) | +0.003 (-0.008 to +0.015) |
| Heuristic, seed 44 (training seed only)   | +0.000 (-0.014 to +0.017)   | -0.000 (-0.014 to +0.010) | -0.005 (-0.014 to +0.004) |
| Without s_hist                            | -0.015 (-0.040 to +0.009)   | -0.017 (-0.042 to +0.005) | -0.004 (-0.016 to +0.009) |
| Without s_ent                             | -0.008 (-0.028 to +0.012)   | -0.007 (-0.026 to +0.009) | -0.006 (-0.017 to +0.008) |
| Without s_ngram                           | -0.007 (-0.022 to +0.009)   | +0.014 (-0.002 to +0.029) | -0.004 (-0.014 to +0.004) |
| Without s_comp                            | -0.002 (-0.024 to +0.022)   | -0.008 (-0.029 to +0.010) | +0.002 (-0.009 to +0.014) |
| Without s_pos                             | -0.017 (-0.038 to +0.002)   | -0.013 (-0.030 to +0.001) | -0.007 (-0.021 to +0.007) |
| Without s_type                            | +0.003 (-0.015 to +0.028)   | -0.031 (-0.054 to -0.014) | -0.001 (-0.016 to +0.014) |
| s_hist + s_ent                            | -0.005 (-0.022 to +0.018)   | -0.023 (-0.042 to -0.008) | -0.013 (-0.025 to -0.003) |
| s_hist + s_ngram                          | -0.004 (-0.020 to +0.016)   | -0.026 (-0.046 to -0.012) | -0.017 (-0.028 to -0.007) |
| s_hist + s_ent + s_comp                   | -0.010 (-0.022 to +0.001)   | -0.018 (-0.027 to -0.010) | -0.018 (-0.031 to -0.007) |
| s_pos + s_type                            | +0.003 (-0.023 to +0.026)   | +0.008 (-0.013 to +0.032) | -0.014 (-0.025 to -0.003) |
| s_hist only                               | -0.009 (-0.022 to +0.005)   | -0.018 (-0.033 to -0.007) | -0.003 (-0.016 to +0.009) |
| s_ngram only                              | -0.013 (-0.042 to +0.022)   | -0.052 (-0.081 to -0.029) | -0.020 (-0.038 to -0.002) |

Added after the protocol freeze (the frozen protocol compares methods with histogram cosine only). Estimate: difference of the full-sample values; parentheses: 2.5-97.5 percentiles of 1000 paired source-file bootstrap differences (the draws of the frozen local bootstrap). The intervals describe test-file sampling only; every ablation is a single training run, and the two seed rows show the size of a difference caused by the training seed alone.

Sources: results/bootstrap/local__*.npz; results/bootstrap/local_summary.json; results/bootstrap/local_paired_vs_heuristic_s42.json
