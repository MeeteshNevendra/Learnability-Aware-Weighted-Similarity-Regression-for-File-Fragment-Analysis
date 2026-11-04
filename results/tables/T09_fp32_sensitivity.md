### Sensitivity of provenance results to float16 score quantisation

| Model                               |   Within-type AUC (fp16 autocast, primary) |   Within-type AUC (fp32) |   Retrieval Hit@1 (primary) |   Retrieval Hit@1 (fp32) |   Distinct score values (fp32) |
|:------------------------------------|-------------------------------------------:|-------------------------:|----------------------------:|-------------------------:|-------------------------------:|
| Model, heuristic (seed 42)          |                                      0.558 |                    0.558 |                       0.133 |                    0.133 |                        5468658 |
| Model, uniform (seed 42)            |                                      0.562 |                    0.562 |                       0.115 |                    0.116 |                        6116207 |
| Model, learnability-aware (seed 42) |                                      0.557 |                    0.557 |                       0.123 |                    0.122 |                        4348906 |

Sources: results/local/provenance/test__*.json; results/local/provenance/fp32__*.json
