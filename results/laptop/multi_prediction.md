# Prediction for the multi-source runs

From single-source data in `an/bench`. Model and its assumptions: see the docstring of scripts/predict_multi.py.

| Graph | Scenario (searches @ cores) | Predicted cpu_hooked wall ms | Predicted gpu wall ms | Predicted speedup gpu vs cpu_hooked | Mean batch size (gpu) |
|---|---|---|---|---|---|
| USA-road-d.NY | 1@6 | 136 | 484 | 0.282x | 1.00 |
| USA-road-d.NY | 5@6 | 139 | 830 | 0.167x | 2.48 |
| USA-road-d.NY | 8@2 | 531 | 957 | 0.555x | 3.96 |
| grid500 | 1@6 | 130 | 484 | 0.268x | 1.00 |
| grid500 | 5@6 | 130 | 844 | 0.154x | 2.49 |
| grid500 | 8@2 | 488 | 954 | 0.512x | 3.98 |
| rmat16 | 1@6 | 63 | 75 | 0.841x | 1.00 |
| rmat16 | 5@6 | 64 | 86 | 0.745x | 1.08 |
| rmat16 | 8@2 | 248 | 246 | 1.006x | 1.89 |
| rmat18 | 1@6 | 308 | 317 | 0.971x | 1.00 |
| rmat18 | 5@6 | 318 | 354 | 0.899x | 1.13 |
| rmat18 | 8@2 | 1239 | 1156 | 1.072x | 1.80 |
| rmat20 | 1@6 | 1486 | 1456 | 1.020x | 1.00 |
| rmat20 | 5@6 | 1486 | 1605 | 0.926x | 1.11 |
| rmat20 | 8@2 | 5829 | 5364 | 1.087x | 1.75 |
