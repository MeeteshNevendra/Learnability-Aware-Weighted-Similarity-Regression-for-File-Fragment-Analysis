### Govdocs1 under alignment, encryption, compression and 4096-byte blocks

| Scoring method                      | aligned            | unaligned          | encrypted          | compressed         | b4096              |
|:------------------------------------|:-------------------|:-------------------|:-------------------|:-------------------|:-------------------|
| Random scores                       | 0.50 / 0.08 / 0.00 | 0.50 / 0.08 / 0.00 | 0.50 / 0.08 / 0.00 | 0.50 / 0.08 / 0.00 | 0.49 / 0.33 / 0.00 |
| Entropy similarity                  | 0.71 / 0.13 / 0.01 | 0.72 / 0.13 / 0.01 | 0.50 / 0.09 / 0.00 | 0.54 / 0.10 / 0.00 | 0.72 / 0.44 / 0.05 |
| Compression-ratio similarity        | 0.64 / 0.11 / 0.00 | 0.64 / 0.12 / 0.00 | 0.50 / 0.09 / 0.00 | 0.50 / 0.08 / 0.00 | 0.69 / 0.41 / 0.03 |
| Bigram cosine                       | 0.77 / 0.25 / 0.15 | 0.78 / 0.26 / 0.15 | 0.50 / 0.09 / 0.00 | 0.52 / 0.10 / 0.01 | 0.80 / 0.56 / 0.29 |
| Histogram cosine                    | 0.76 / 0.23 / 0.14 | 0.78 / 0.24 / 0.14 | 0.50 / 0.08 / 0.00 | 0.57 / 0.15 / 0.03 | 0.81 / 0.57 / 0.35 |
| Mean of four byte measures          | 0.81 / 0.24 / 0.14 | 0.81 / 0.24 / 0.14 | 0.50 / 0.09 / 0.00 | 0.56 / 0.14 / 0.03 | 0.82 / 0.56 / 0.34 |
| Model, heuristic (seed 42)          | 0.71 / 0.16 / 0.06 | 0.74 / 0.17 / 0.05 | 0.50 / 0.07 / 0.00 | 0.55 / 0.11 / 0.00 | 0.75 / 0.48 / 0.22 |
| Model, heuristic (seed 43)          | 0.70 / 0.15 / 0.06 | 0.74 / 0.17 / 0.06 | 0.50 / 0.08 / 0.00 | 0.55 / 0.13 / 0.01 | 0.74 / 0.48 / 0.20 |
| Model, heuristic (seed 44)          | 0.70 / 0.16 / 0.05 | 0.73 / 0.16 / 0.06 | 0.50 / 0.08 / 0.00 | 0.55 / 0.10 / 0.01 | 0.74 / 0.48 / 0.21 |
| Model, uniform (seed 42)            | 0.68 / 0.17 / 0.06 | 0.71 / 0.16 / 0.05 | 0.50 / 0.08 / 0.00 | 0.55 / 0.12 / 0.01 | 0.71 / 0.48 / 0.17 |
| Model, learnability-aware (seed 42) | 0.74 / 0.15 / 0.06 | 0.76 / 0.15 / 0.06 | 0.50 / 0.09 / 0.00 | 0.53 / 0.11 / 0.00 | 0.75 / 0.47 / 0.20 |

Cells: same-file AUC within type / within-file adjacent Hit@1 (oracle) / greedy adjacency recall. Compressed condition: 133 of 150 files long enough. Duplicates with the replacement collection: 0.

Sources: results/govdocs1/*.json; data/govdocs1/build_report.json
