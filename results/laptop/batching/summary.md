# Multi-source runs

GPU: NVIDIA GeForce GTX 1660 Ti (sm_75). Median wall time (ms) from the start of the first search to the end of the last. Speedup = cpu_hooked / configuration, paired on rep. Batch = mean GPU calls per batch.

## 1 searches, all cores

| Graph | dijkstra | builtin | cpu_hooked | gpu_g0 | gpu_g1024 | gpu_g4096 | gpu_a0 | Best GPU vs cpu_hooked | Predicted gpu_g0 | Measured gpu_g0 | Batch (gpu_g0) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| USA-road-d.NY | 29 | 148 | 156 | 753 | 159 | 140 | 800 | 1.119x (gpu_g4096) | 0.282x | 0.205x | 1.00 |
| grid500 | 40 | 120 | 128 | 891 | 163 | 133 | 817 | 0.969x (gpu_g4096) | 0.268x | 0.142x | 1.00 |
| rmat16 | 13 | 65 | 66 | 102 | 94 | 80 | 98 | 0.812x (gpu_g4096) | 0.841x | 0.641x | 1.00 |
| rmat18 | 69 | 306 | 312 | 372 | 362 | 358 | 374 | 0.871x (gpu_g4096) | 0.971x | 0.837x | 1.00 |
| rmat20 | 337 | 1482 | 1498 | 1619 | 1570 | 1599 | 1572 | 0.954x (gpu_g1024) | 1.020x | 0.928x | 1.00 |

## 5 searches, all cores

| Graph | dijkstra | builtin | cpu_hooked | gpu_g0 | gpu_g1024 | gpu_g4096 | gpu_a0 | Best GPU vs cpu_hooked | Predicted gpu_g0 | Measured gpu_g0 | Batch (gpu_g0) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| USA-road-d.NY | 29 | 195 | 195 | 1614 | 216 | 197 | 1507 | 1.043x (gpu_g4096) | 0.167x | 0.124x | 2.43 |
| grid500 | 39 | 202 | 186 | 1738 | 200 | 174 | 1387 | 1.146x (gpu_g4096) | 0.154x | 0.107x | 2.45 |
| rmat16 | 15 | 100 | 103 | 153 | 137 | 118 | 172 | 0.873x (gpu_g4096) | 0.745x | 0.674x | 1.77 |
| rmat18 | 94 | 589 | 517 | 608 | 575 | 565 | 725 | 0.924x (gpu_g4096) | 0.899x | 0.861x | 1.26 |
| rmat20 | 648 | 2499 | 2487 | 2580 | 2666 | 2555 | 3472 | 0.973x (gpu_g4096) | 0.926x | 0.953x | 1.16 |

## 8 searches, 2 cores

| Graph | dijkstra | builtin | cpu_hooked | gpu_g0 | gpu_g1024 | gpu_g4096 | gpu_a0 | Best GPU vs cpu_hooked | Predicted gpu_g0 | Measured gpu_g0 | Batch (gpu_g0) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| USA-road-d.NY | 35 | 683 | 722 | 3593 | 758 | 697 | 2828 | 1.034x (gpu_g4096) | 0.555x | 0.201x | 4.44 |
| grid500 | 49 | 575 | 604 | 3270 | 685 | 632 | 2512 | 0.954x (gpu_g4096) | 0.512x | 0.182x | 4.63 |
| rmat16 | 21 | 315 | 321 | 451 | 434 | 371 | 439 | 0.863x (gpu_g4096) | 1.006x | 0.715x | 4.09 |
| rmat18 | 138 | 1568 | 1605 | 1809 | 1782 | 1776 | 1866 | 0.904x (gpu_g4096) | 1.072x | 0.884x | 3.94 |
| rmat20 | 720 | 7136 | 7277 | 7433 | 7734 | 7361 | 7867 | 0.986x (gpu_g4096) | 1.087x | 0.979x | 2.96 |

