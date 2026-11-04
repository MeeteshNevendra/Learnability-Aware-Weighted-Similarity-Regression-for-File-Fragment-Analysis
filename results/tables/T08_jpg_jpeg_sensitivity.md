### Sensitivity analysis: JPG and JPEG merged into one extension group

| Scoring method                      |   Same-file AUC within type (JPG, JPEG separate) |   ... JPG+JPEG merged |   Same-type AUC (separate) |   Same-type AUC (merged) |   Hit@1 type-restricted (separate) |   Hit@1 type-restricted (merged) |
|:------------------------------------|-------------------------------------------------:|----------------------:|---------------------------:|-------------------------:|-----------------------------------:|---------------------------------:|
| Random scores                       |                                            0.501 |                 0.5   |                      0.5   |                    0.5   |                              0.148 |                            0.129 |
| Entropy similarity                  |                                            0.535 |                 0.528 |                      0.585 |                    0.595 |                              0.187 |                            0.171 |
| Compression-ratio similarity        |                                            0.495 |                 0.482 |                      0.555 |                    0.569 |                              0.166 |                            0.152 |
| Bigram cosine                       |                                            0.553 |                 0.539 |                      0.574 |                    0.589 |                              0.259 |                            0.24  |
| Histogram cosine                    |                                            0.58  |                 0.585 |                      0.619 |                    0.619 |                              0.377 |                            0.356 |
| Mean of four byte measures          |                                            0.583 |                 0.58  |                      0.606 |                    0.613 |                              0.366 |                            0.345 |
| Model, heuristic (seed 42)          |                                            0.558 |                 0.568 |                      0.63  |                    0.627 |                              0.273 |                            0.251 |
| Model, heuristic (seed 43)          |                                            0.559 |                 0.565 |                      0.63  |                    0.63  |                              0.279 |                            0.261 |
| Model, heuristic (seed 44)          |                                            0.559 |                 0.562 |                      0.63  |                    0.631 |                              0.27  |                            0.246 |
| Model, uniform (seed 42)            |                                            0.562 |                 0.568 |                      0.63  |                    0.628 |                              0.278 |                            0.257 |
| Model, learnability-aware (seed 42) |                                            0.557 |                 0.558 |                      0.6   |                    0.604 |                              0.28  |                            0.252 |
| Model, uniform (seed 43)            |                                            0.547 |                 0.542 |                      0.62  |                    0.627 |                              0.267 |                            0.247 |
| Model, uniform (seed 44)            |                                            0.546 |                 0.551 |                      0.628 |                    0.627 |                              0.256 |                            0.236 |
| Model, learnability-aware (seed 43) |                                            0.556 |                 0.558 |                      0.602 |                    0.604 |                              0.287 |                            0.261 |
| Model, learnability-aware (seed 44) |                                            0.559 |                 0.565 |                      0.607 |                    0.606 |                              0.268 |                            0.245 |
| Model, s_pos + s_type target        |                                            0.561 |                 0.565 |                      0.638 |                    0.637 |                              0.28  |                            0.258 |
| Model, s_hist target                |                                            0.55  |                 0.555 |                      0.612 |                    0.611 |                              0.279 |                            0.256 |

JPG and JPEG are extension groups of one format (JPEG/JFIF); the primary analysis keeps the recorded extensions separate.

Sources: results/local/provenance/test__*.json
