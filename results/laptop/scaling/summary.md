# Scaling study: BMSSP against Dijkstra

Medians over 3 sources per graph (ms); ratios paired per source.

| Graph | n | m | Dijkstra | BMSSP | BMSSP / Dijkstra | D ops / Dijkstra | Max BMSSP speedup if all edge work were free |
|---|---|---|---|---|---|---|---|
| rmat14 | 16384 | 131072 | 2.1 | 13.7 | 6.41x | 5.01x | 1.19x |
| rmat16 | 65536 | 524288 | 10.5 | 66.3 | 6.34x | 4.59x | 1.30x |
| rmat18 | 262144 | 2097152 | 60.9 | 329.3 | 5.47x | 3.85x | 1.29x |
| rmat20 | 1048576 | 8388608 | 319.9 | 1595.3 | 4.97x | 3.58x | 1.28x |
| rmat21 | 2097152 | 16777216 | 673.5 | 3396.1 | 5.04x | 3.52x | 1.33x |
| rmat22 | 4194304 | 33554432 | 1439.2 | 7880.4 | 5.51x | 3.48x | 1.47x |
| grid128 | 16384 | 65024 | 1.7 | 7.5 | 4.44x | 2.63x | 1.43x |
| grid256 | 65536 | 261120 | 7.5 | 30.4 | 4.04x | 2.55x | 1.35x |
| grid512 | 262144 | 1046528 | 34.2 | 135.1 | 3.95x | 2.14x | 1.50x |
| grid1024 | 1048576 | 4190208 | 157.2 | 652.2 | 4.07x | 2.08x | 1.70x |
| grid2048 | 4194304 | 16769024 | 807.2 | 4226.4 | 5.22x | 2.38x | 1.93x |

Fit of BMSSP / Dijkstra = a * (log2 n)^b over each family (least squares in log space). Theory predicts b = -1/3.

| Family | a | b | Fitted ratio at n = 2^30 | log2 n where the fitted ratio reaches 1 |
|---|---|---|---|---|
| rmat | 24.85 | -0.513 | 4.33x | 522 |
| grid | 2.04 | 0.258 | 4.92x | never (ratio does not fall) |
