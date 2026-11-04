#!/usr/bin/env bash
set -euo pipefail
OUT="results/week2/baselines"; REPS=3; mkdir -p "$OUT"
run_graph() {
  NAME=$1; GRAPH=$2; DELTA=$3; shift 3
  for SRC in "$@"; do for REP in $(seq 1 $REPS); do
    ./run_dijkstra "$GRAPH" "$SRC" > "$OUT/${NAME}_source${SRC}_rep${REP}_dijkstra.txt"
    ./run_delta "$GRAPH" "$SRC" "$DELTA" > "$OUT/${NAME}_source${SRC}_rep${REP}_cpu_delta.txt"
  done; done
}
run_graph RMAT16  data/rmat16.gr          16   1 132 263 394 525
run_graph GRID500 data/grid500.gr         16   1 501 1001 1501 2001
run_graph NY      data/USA-road-d.NY.gr   5000 1 529 1058 1587 2115
