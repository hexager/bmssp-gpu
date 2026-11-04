#!/usr/bin/env bash

set -e

mkdir -p data/fuzz
mkdir -p results

LOG="results/gpu_fuzz.log"

: > "$LOG"

echo "GPU DELTA-STEPPING FUZZ TEST" | tee -a "$LOG"
echo "============================" | tee -a "$LOG"

TOTAL=0

for seed in $(seq 1 20); do

    FILE="data/fuzz/random_${seed}.gr"

    python3 scripts/generate_random_graph.py \
        --vertices 100 \
        --edges 600 \
        --seed "$seed" \
        --output "$FILE" \
        >> "$LOG"

    for delta in 1 2 4 8 16 32; do

        TOTAL=$((TOTAL + 1))

        echo \
            "Test $TOTAL: seed=$seed delta=$delta"

        if ! ./run_gpu_delta \
            "$FILE" \
            1 \
            "$delta" \
            >> "$LOG" 2>&1
        then

            echo
            echo "FAILED"
            echo "seed=$seed"
            echo "delta=$delta"
            echo "graph=$FILE"
            echo
            echo "See $LOG"

            exit 1
        fi

        if ! tail -30 "$LOG" |
            grep -q "Correctness : PASS"
        then

            echo
            echo "FAILED: PASS marker missing"
            echo "seed=$seed delta=$delta"
            echo "See $LOG"

            exit 1
        fi

    done

done

echo
echo "================================"
echo "ALL GPU FUZZ TESTS PASSED"
echo "Total tests: $TOTAL"
echo "================================"
