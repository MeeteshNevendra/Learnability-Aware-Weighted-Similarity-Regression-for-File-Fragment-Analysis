### Regression by pair-sampling mode and by actual provenance relation

| Model                               | Breakdown       | Group                |   Pairs |   RMSE |     R2 |     r |   rho |
|:------------------------------------|:----------------|:---------------------|--------:|-------:|-------:|------:|------:|
| Model, heuristic (seed 42)          | pair mode       | diff_type            |   20000 | 0.075  |  0.644 | 0.949 | 0.772 |
| Model, heuristic (seed 42)          | pair mode       | random               |   20000 | 0.076  |  0.619 | 0.909 | 0.68  |
| Model, heuristic (seed 42)          | pair mode       | same_file            |   20000 | 0.129  | -0.347 | 0.855 | 0.569 |
| Model, heuristic (seed 42)          | pair mode       | same_type_diff_file  |   20000 | 0.07   |  0.439 | 0.825 | 0.554 |
| Model, heuristic (seed 42)          | actual relation | diff_ext             |   37935 | 0.076  |  0.614 | 0.946 | 0.743 |
| Model, heuristic (seed 42)          | actual relation | same_ext_diff_file   |   21812 | 0.069  |  0.439 | 0.825 | 0.551 |
| Model, heuristic (seed 42)          | actual relation | same_file            |   20253 | 0.129  | -0.349 | 0.854 | 0.568 |
| Model, heuristic (seed 42)          | order reversal  | mean |p(a,b)-p(b,a)| |         | 0.0063 |        |       |       |
| Model, uniform (seed 42)            | pair mode       | diff_type            |   20000 | 0.115  |  0.105 | 0.925 | 0.756 |
| Model, uniform (seed 42)            | pair mode       | random               |   20000 | 0.116  |  0.176 | 0.84  | 0.655 |
| Model, uniform (seed 42)            | pair mode       | same_file            |   20000 | 0.158  | -1.608 | 0.792 | 0.515 |
| Model, uniform (seed 42)            | pair mode       | same_type_diff_file  |   20000 | 0.093  | -0.22  | 0.761 | 0.459 |
| Model, uniform (seed 42)            | actual relation | diff_ext             |   37935 | 0.117  |  0.015 | 0.921 | 0.727 |
| Model, uniform (seed 42)            | actual relation | same_ext_diff_file   |   21812 | 0.092  | -0.217 | 0.763 | 0.46  |
| Model, uniform (seed 42)            | actual relation | same_file            |   20253 | 0.158  | -1.609 | 0.791 | 0.514 |
| Model, uniform (seed 42)            | order reversal  | mean |p(a,b)-p(b,a)| |         | 0.0048 |        |       |       |
| Model, learnability-aware (seed 42) | pair mode       | diff_type            |   20000 | 0.033  |  0.968 | 0.987 | 0.819 |
| Model, learnability-aware (seed 42) | pair mode       | random               |   20000 | 0.033  |  0.96  | 0.984 | 0.764 |
| Model, learnability-aware (seed 42) | pair mode       | same_file            |   20000 | 0.034  |  0.929 | 0.971 | 0.699 |
| Model, learnability-aware (seed 42) | pair mode       | same_type_diff_file  |   20000 | 0.04   |  0.899 | 0.961 | 0.676 |
| Model, learnability-aware (seed 42) | actual relation | diff_ext             |   37935 | 0.033  |  0.966 | 0.986 | 0.799 |
| Model, learnability-aware (seed 42) | actual relation | same_ext_diff_file   |   21812 | 0.04   |  0.899 | 0.961 | 0.673 |
| Model, learnability-aware (seed 42) | actual relation | same_file            |   20253 | 0.034  |  0.929 | 0.971 | 0.699 |
| Model, learnability-aware (seed 42) | order reversal  | mean |p(a,b)-p(b,a)| |         | 0.001  |        |       |       |

Sources: results/local/regression/<run>.json
