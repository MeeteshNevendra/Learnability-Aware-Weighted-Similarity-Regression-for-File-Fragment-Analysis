### Component learnability (seed 42) and derived weights

| Component   |   Validation R2 |   Test R2 |   Test r |   Derived weight | Best epoch / epochs run   |
|:------------|----------------:|----------:|---------:|-----------------:|:--------------------------|
| s_hist      |           0.915 |     0.913 |    0.96  |            0.251 | 16/20                     |
| s_ent       |           0.969 |     0.984 |    0.993 |            0.266 | 15/20                     |
| s_ngram     |           0.849 |     0.832 |    0.928 |            0.233 | 18/20                     |
| s_comp      |           0.912 |     0.926 |    0.964 |            0.25  | 8/13                      |
| s_pos       |          -0.121 |    -0.126 |    0.243 |            0     | 19/20                     |
| s_type      |          -0.46  |    -0.418 |    0.296 |            0     | 7/12                      |

Weights: w_k = max(R2_val,k, 0) / sum of positive values (validation only). 

Sources: runs/comp_*/summary_val.json; results/local/regression/comp_*.json; runs/learnability_weights.json
