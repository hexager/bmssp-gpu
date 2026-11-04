#!/usr/bin/env bash
# With dirty tracking on, only the first GPU call may upload all n vertices.
# (Regression: an empty dirty list was passed as nullptr and caused a full upload.)
#   usage: tests/test_full_upload_once.sh [BIN] [GRAPH] [SOURCE]
set -euo pipefail
cd "$(dirname "$0")/.."
BIN="${1:-bin/fp_hybrid_emu}"; G="${2:-data/rmat16.gr}"; S="${3:-59526}"
n=$(grep -m1 '^p' "$G" | awk '{print $3}')
log=$(mktemp -u); trap 'rm -f "$log"' EXIT
FP_MODE=gpu FP_GPU_MIN_EDGES=0 FP_GPU_DIRTY=1 FP_CALL_LOG="$log" "$BIN" "$G" "$S" > /dev/null 2>&1
col=$(head -1 "$log" | tr ',' '\n' | grep -n '^upload_vertices$' | cut -d: -f1)
full=$(awk -F, -v c="$col" -v n="$n" 'NR > 1 && $c == n' "$log" | wc -l)
if [ "$full" -eq 1 ]; then echo "PASS full uploads=1"; else echo "FAIL full uploads=$full (expected 1)"; exit 1; fi
