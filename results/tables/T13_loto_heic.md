### Leave-one-type-out: HEIC removed from training and validation pairs

| Method                                  |   R2 on test pairs with HEIC |   R2 on test pairs without HEIC |   Same-type AUC, pairs with HEIC |   Same-file AUC among HEIC fragments |
|:----------------------------------------|-----------------------------:|--------------------------------:|---------------------------------:|-------------------------------------:|
| Model, heuristic without HEIC (seed 42) |                        0.575 |                           0.663 |                            0.657 |                                0.499 |
| Model, heuristic (seed 42)              |                        0.594 |                           0.671 |                            0.674 |                                0.497 |
| Model, heuristic (seed 43)              |                        0.569 |                           0.647 |                            0.691 |                                0.5   |
| Model, heuristic (seed 44)              |                        0.606 |                           0.672 |                            0.689 |                                0.499 |
| Histogram cosine                        |                              |                                 |                            0.69  |                                0.5   |

Test pool: 8 HEIC files, 640 HEIC fragments.

Sources: results/local/loto_heic.json; results/local/regression/*.json
