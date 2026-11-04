#!/usr/bin/env bash

set -euo pipefail

OUT="results/week2/delta_sweep"
mkdir -p "$OUT"

REPS=3


run_sweep()
{
    NAME="$1"
    GRAPH="$2"
    SOURCE="$3"
    shift 3

    DELTAS=("$@")


    echo
    echo "========================================"
    echo "GPU DELTA SWEEP: $NAME"
    echo "========================================"


    for DELTA in "${DELTAS[@]}"; do

        echo "Delta $DELTA"


        for REP in $(seq 1 "$REPS"); do

            echo "  repetition $REP/$REPS"


            FILE="$OUT/${NAME}_delta${DELTA}_rep${REP}.txt"


            ./run_gpu_delta \
                "$GRAPH" \
                "$SOURCE" \
                "$DELTA" \
                > "$FILE"


            if ! grep -q \
                "Correctness : PASS" \
                "$FILE"
            then

                echo "FAILED"
                echo "Dataset: $NAME"
                echo "Delta: $DELTA"
                echo "Rep: $REP"

                exit 1
            fi

        done

    done
}


run_sweep \
    "RMAT16" \
    "data/rmat16.gr" \
    1 \
    1 2 4 8 16 32 64


run_sweep \
    "GRID500" \
    "data/grid500.gr" \
    1 \
    2 4 8 16 32 64 128


run_sweep \
    "NY" \
    "data/USA-road-d.NY.gr" \
    1 \
    250 500 1000 2000 5000


echo
echo "========================================"
echo "GPU DELTA SWEEP COMPLETE"
echo "========================================"
