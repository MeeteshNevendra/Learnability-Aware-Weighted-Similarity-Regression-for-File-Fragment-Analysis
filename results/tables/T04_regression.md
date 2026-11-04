### Agreement with each scheme's own composite target (test pairs)

| Weighting scheme                     |   Runs | rmse          | mae           | r2            | pearson       | spearman      |
|:-------------------------------------|-------:|:--------------|:--------------|:--------------|:--------------|:--------------|
| Heuristic weights (seeds 42, 43, 44) |      3 | 0.092 ± 0.002 | 0.074 ± 0.002 | 0.655 ± 0.015 | 0.828 ± 0.002 | 0.623 ± 0.002 |
| Uniform weights (seed 42)            |      1 | 0.123         | 0.102         | 0.505         | 0.739         | 0.573         |
| Learnability-aware weights (seed 42) |      1 | 0.035         | 0.023         | 0.950         | 0.980         | 0.750         |

Each scheme defines a different target; values are not comparable across schemes as evidence of forensic value.

Sources: results/local/regression/<run>.json
