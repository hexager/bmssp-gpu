#!/usr/bin/env bash
# H3: sweep k and t. Needs the BMSSP binary (./run_bmssp graph source k t). PENDING: BMSSP code.
set -euo pipefail; OUT=results/week2/h3; mkdir -p $OUT
for g in "RMAT16 data/rmat16.gr 1" "GRID500 data/grid500.gr 1" "NY data/USA-road-d.NY.gr 1"; do set -- $g
 for k in 2 4 8; do for t in 1 2 4; do ./run_bmssp $2 $3 $k $t > $OUT/${1}_k${k}_t${t}.txt; done; done; done
