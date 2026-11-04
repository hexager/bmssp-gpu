#!/usr/bin/env bash
# Correctness campaign for the batched GPU FindPivots (fp_multi, MULTI_MODE=check). In every batch, each call is
# compared with the serial synchronous version on the same input state (P, W order, all of d/pred/path_sz), and the
# distances of all sources are compared with independent Dijkstra runs. Any failure stops the script.
#   usage: scripts/check_multi.sh [BIN]     (default bin/fp_multi; bin/fp_multi_emu works without a GPU)
#   FUZZ_DIR=dir (default fuzz/ if it exists): also every .gr file in dir, 3 sources each
set -uo pipefail
cd "$(dirname "$0")/.."
BIN="${1:-bin/fp_multi}"
FUZZ_DIR="${FUZZ_DIR:-$([ -d fuzz ] && echo fuzz)}"
OUT=results/check_multi; mkdir -p "$OUT"
SUM="$OUT/summary.txt"; : > "$SUM"
fail=0; runs=0; calls=0

one() {  # graph "s1 s2 ..." policy threshold
  local g=$1 srcs=$2 pol=$3 thr=$4 tag
  tag="$(basename "$g" .gr)_b${pol}_t${thr}_n$(echo $srcs | wc -w)"
  for s in $srcs; do bin/dijkstra_ref "$g" "$s" 2>/dev/null; done > "$OUT/ref.txt"
  MULTI_MODE=check MULTI_BATCH=$pol FP_GPU_MIN_EDGES=$thr "$BIN" "$g" "$(echo $srcs | tr ' ' ',')" > "$OUT/out.txt" 2> "$OUT/$tag.err"
  local rc=$? ok=1
  [ $rc -eq 0 ] || ok=0
  cmp -s "$OUT/out.txt" "$OUT/ref.txt" || ok=0
  runs=$((runs + 1))
  local c; c=$(grep -o 'checked=[0-9]*' "$OUT/$tag.err" | cut -d= -f2); calls=$((calls + ${c:-0}))
  local info; info=$(grep -oE 'checked=[0-9]+|batches=[0-9]+|max_batch=[0-9]+' "$OUT/$tag.err" | tr '\n' ' ')
  if [ $ok -eq 1 ]; then echo "PASS $tag $info" | tee -a "$SUM"; rm -f "$OUT/$tag.err"
  else echo "FAIL $tag rc=$rc (log: $OUT/$tag.err)" | tee -a "$SUM"; fail=1; fi
}

while read -r name srcs; do
  case "$name" in ''|\#*) continue;; esac
  g="data/$name.gr"; [ -f "$g" ] || { echo "missing $g"; exit 2; }
  srcs=${srcs%%#*}
  one "$g" "$srcs" greedy 0;    [ $fail -eq 0 ] || break
  one "$g" "$srcs" all 0;       [ $fail -eq 0 ] || break
  one "$g" "$srcs" greedy 4096; [ $fail -eq 0 ] || break
done < sources.txt

if [ $fail -eq 0 ] && [ -n "${FUZZ_DIR:-}" ]; then
  for g in "$FUZZ_DIR"/*.gr; do
    n=$(grep -m1 '^p' "$g" | awk '{print $3}')
    srcs="1 $((n / 3 > 0 ? n / 3 : 1)) $((2 * n / 3 > 0 ? 2 * n / 3 : 1))"
    one "$g" "$srcs" greedy 0; [ $fail -eq 0 ] || break
    one "$g" "$srcs" all 0;    [ $fail -eq 0 ] || break
  done
fi
rm -f "$OUT/out.txt" "$OUT/ref.txt"
echo "checked runs: $runs, batched calls compared: $calls, result: $([ $fail -eq 0 ] && echo ALL PASS || echo FAILURE)" | tee -a "$SUM"
exit $fail
