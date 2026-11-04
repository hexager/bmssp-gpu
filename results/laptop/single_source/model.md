# Bottleneck model

Per (graph, source): rep-1 runs of sync_cpu and gpu_all, calls paired one to one. Values are medians over sources (ms).

| Graph | sync_cpu run | FindPivots share | Bound: FindPivots free | Bound: best per-call choice | GPU wins (calls) | Max speedup vs sync_cpu (free / oracle) | builtin run |
|---|---|---|---|---|---|---|---|
| USA-road-d.NY | 162.3 | 17.9% | 134.2 | 162.3 | 0 of 4036 | 1.219x / 1.000x | 138.0 |
| grid500 | 148.3 | 22.3% | 114.8 | 148.3 | 1 of 3784 | 1.287x / 1.000x | 121.4 |
| rmat16 | 64.4 | 11.7% | 56.9 | 63.9 | 9 of 149 | 1.132x / 1.009x | 58.5 |
| rmat18 | 313.5 | 13.1% | 272.4 | 302.3 | 61 of 183 | 1.151x / 1.037x | 297.0 |
| rmat20 | 1458.3 | 13.3% | 1264.0 | 1381.6 | 220 of 305 | 1.153x / 1.056x | 1395.4 |

Per-call GPU cost by phase, fitted over all calls of a graph except the first (ms = a + b x):

| Graph | Upload: us + ns/vertex | Rounds: us + us/round + ns/edge | Download: us + ns/vertex | Fixed floor (2 rounds), us | CPU sync: us + ns/edge | Break-even edges | Calls above | R^2 up/rounds/down/cpu |
|---|---|---|---|---|---|---|---|---|
| USA-road-d.NY | 11 + 8.8 | 26 + 13 + 30.42 | 14 + 11.0 | 77 | -2 + 47.47 | 4725 | 0 of 20106 | 0.80 / 0.17 / 0.04 / 0.85 |
| grid500 | 11 + 8.5 | 19 + 20 + 17.91 | 14 + 9.8 | 85 | -1 + 31.12 | 6585 | 40 of 18960 | 0.80 / 0.52 / 0.12 / 0.93 |
| rmat16 | 11 + 13.9 | 35 + 5 + 5.27 | 15 + 11.6 | 69 | -11 + 18.22 | 6429 | 83 of 740 | 0.97 / 0.56 / 0.80 / 0.91 |
| rmat18 | 17 + 16.0 | 4 + 33 + 2.82 | 16 + 16.1 | 103 | 45 + 16.19 | 5470 | 491 of 910 | 0.98 / 0.50 / 0.97 / 0.90 |
| rmat20 | 23 + 19.8 | 13 + 36 + 4.23 | 15 + 22.0 | 122 | 85 + 20.60 | 4724 | 1341 of 1522 | 1.00 / 0.63 / 0.99 / 0.89 |

Model check, leave one graph out: phases fitted on the other graphs, then the total GPU FindPivots time of each gpu_all run of this graph predicted from its logged call sizes:

| Graph | Measured GPU FindPivots ms (median over sources) | Predicted | Error |
|---|---|---|---|
| USA-road-d.NY | 345.0 | 374.1 | +8.5% |
| grid500 | 353.1 | 333.2 | -5.6% |
| rmat16 | 15.1 | 17.5 | +16.3% |
| rmat18 | 34.7 | 35.3 | +1.6% |
| rmat20 | 119.6 | 81.7 | -31.7% |
