#!/usr/bin/env bash

set -euo pipefail

OUT="results/week2/baselines"

mkdir -p "$OUT"

REPS=5


run_graph()
{
    NAME="$1"
    GRAPH="$2"
    DELTA="$3"
    shift 3

    SOURCES=("$@")


    echo
    echo "========================================"
    echo "DATASET: $NAME"
    echo "GRAPH:   $GRAPH"
    echo "DELTA:   $DELTA"
    echo "========================================"


    for SOURCE in "${SOURCES[@]}"; do

        echo
        echo "Source $SOURCE"


        for REP in $(seq 1 "$REPS"); do

            echo "  repetition $REP/$REPS"


            ./run_dijkstra \
                "$GRAPH" \
                "$SOURCE" \
                > "$OUT/${NAME}_source${SOURCE}_rep${REP}_dijkstra.txt"


            ./run_delta \
                "$GRAPH" \
                "$SOURCE" \
                "$DELTA" \
                > "$OUT/${NAME}_source${SOURCE}_rep${REP}_cpu_delta.txt"


            ./run_gpu_delta \
                "$GRAPH" \
                "$SOURCE" \
                "$DELTA" \
                > "$OUT/${NAME}_source${SOURCE}_rep${REP}_gpu_delta.txt"


            if ! grep -q \
                "Correctness : PASS" \
                "$OUT/${NAME}_source${SOURCE}_rep${REP}_cpu_delta.txt"
            then

                echo "CPU DELTA FAILED"
                echo "Dataset: $NAME"
                echo "Source: $SOURCE"
                echo "Rep: $REP"

                exit 1
            fi


            if ! grep -q \
                "Correctness : PASS" \
                "$OUT/${NAME}_source${SOURCE}_rep${REP}_gpu_delta.txt"
            then

                echo "GPU DELTA FAILED"
                echo "Dataset: $NAME"
                echo "Source: $SOURCE"
                echo "Rep: $REP"

                exit 1
            fi

        done

    done
}


run_graph \
    "RMAT16" \
    "data/rmat16.gr" \
    16 \
    1 132 263 394 525


run_graph \
    "GRID500" \
    "data/grid500.gr" \
    16 \
    1 501 1001 1501 2001


run_graph \
    "NY" \
    "data/USA-road-d.NY.gr" \
    5000 \
    1 529 1058 1587 2115


echo
echo "========================================"
echo "ALL WEEK 2 BASELINE RUNS COMPLETED"
echo "========================================"
