# Multi-source runs

GPU: NVIDIA GeForce GTX 1660 Ti (sm_75). Median wall time (ms) from the start of the first search to the end of the last. Speedup = cpu_hooked / configuration, paired on rep. Batch = mean GPU calls per batch.

## 1 searches, 1 cores

| Graph | cpu_hooked | gpu_g0 | gpu_g0_spin | gpu_g0_w | gpu_g0_spin_w | Best GPU vs cpu_hooked | Predicted gpu_g0 | Measured gpu_g0 | Batch (gpu_g0) |
|---|---|---|---|---|---|---|---|---|---|
| USA-road-d.NY | 151 | 943 | 511 | 743 | 510 | 0.296x (gpu_g0_spin_w) | - | 0.159x | 1.00 |
| grid500 | 129 | 1011 | 535 | 818 | 528 | 0.244x (gpu_g0_spin_w) | - | 0.128x | 1.00 |
| rmat16 | 64 | 95 | 80 | 92 | 85 | 0.802x (gpu_g0_spin) | - | 0.677x | 1.00 |
| rmat18 | 319 | 365 | 338 | 365 | 350 | 0.946x (gpu_g0_spin) | - | 0.874x | 1.00 |
| rmat20 | 1533 | 1572 | 1524 | 1609 | 1571 | 1.006x (gpu_g0_spin) | - | 0.975x | 1.00 |

## 8 searches, 2 cores

| Graph | cpu_hooked | gpu_g0 | gpu_g0_spin | gpu_g0_w | gpu_g0_spin_w | Best GPU vs cpu_hooked | Predicted gpu_g0 | Measured gpu_g0 | Batch (gpu_g0) |
|---|---|---|---|---|---|---|---|---|---|
| USA-road-d.NY | 707 | 3659 | 1883 | 2438 | 1352 | 0.524x (gpu_g0_spin_w) | - | 0.197x | 4.38 |
| grid500 | 597 | 3272 | 1698 | 2349 | 1255 | 0.474x (gpu_g0_spin_w) | - | 0.184x | 4.53 |
| rmat16 | 320 | 452 | 388 | 422 | 353 | 0.905x (gpu_g0_spin_w) | - | 0.707x | 4.29 |
| rmat18 | 1585 | 1814 | 1653 | 1754 | 1569 | 1.010x (gpu_g0_spin_w) | - | 0.877x | 3.99 |
| rmat20 | 7350 | 7378 | 7189 | 7271 | 6905 | 1.066x (gpu_g0_spin_w) | - | 0.997x | 2.79 |

