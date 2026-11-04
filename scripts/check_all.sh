#!/usr/bin/env bash
# Correctness campaign. Every GPU FindPivots call is compared with the serial synchronous version on the
# same state (P, W order, all of d/pred/path_sz), BMSSP_CHECK=1 checks the FindPivots invariants, and the
# final distances are compared with an independent Dijkstra. The GPU Near-Far baseline is run with every delta
# of the benchmark sweep (it stops if two deltas disagree) and compared with Dijkstra. Any failure stops the script.
#   usage: scripts/check_all.sh [BIN [NF_BIN]]   (defaults bin/fp_hybrid bin/nearfar; the *_emu builds need no GPU)
#   FUZZ_DIR=dir   also check every .gr file in dir (3 sources each); default fuzz/ if it exists
set -uo pipefail
cd "$(dirname "$0")/.."
BIN="${1:-bin/fp_hybrid}"
NF_BIN="${2:-bin/nearfar}"
NF_DELTAS="1,7,x0.0625,x0.125,x0.25,x0.5,x1,x2,x4,x8,x32"
FUZZ_DIR="${FUZZ_DIR:-$([ -d fuzz ] && echo fuzz)}"
OUT=results/check; mkdir -p "$OUT"
export CUDA_MODULE_LOADING=EAGER
{ date -u; uname -a; lscpu 2>&1 | head -20; free -g 2>&1 | head -2; nvidia-smi 2>&1 | head -15; nvcc --version 2>&1 | tail -2; g++ --version | head -1; } > results/env.txt
SUM="$OUT/summary.txt"; : > "$SUM"
fail=0; runs=0

one() {  # graph source threshold dirty
  local g=$1 s=$2 thr=$3 dirty=$4 tag
  tag="$(basename "$g" .gr)_s${s}_t${thr}_d${dirty}"
  bin/dijkstra_ref "$g" "$s" > "$OUT/ref.txt"
  FP_MODE=check FP_GPU_MIN_EDGES=$thr FP_GPU_DIRTY=$dirty BMSSP_CHECK=1 "$BIN" "$g" "$s" > "$OUT/out.txt" 2> "$OUT/$tag.err"
  local rc=$? ok=1
  [ $rc -eq 0 ] || ok=0
  cmp -s "$OUT/out.txt" "$OUT/ref.txt" || ok=0
  grep -q "check_fail=0" "$OUT/$tag.err" || ok=0
  runs=$((runs + 1))
  local calls; calls=$(grep -o 'calls_gpu=[0-9]*' "$OUT/$tag.err" || echo calls_gpu=?)
  if [ $ok -eq 1 ]; then echo "PASS $tag $calls" | tee -a "$SUM"; rm -f "$OUT/$tag.err"
  else echo "FAIL $tag rc=$rc (log: $OUT/$tag.err)" | tee -a "$SUM"; fail=1; fi
}

nf() {  # graph source
  local g=$1 s=$2 tag
  tag="nearfar_$(basename "$g" .gr)_s${s}"
  bin/dijkstra_ref "$g" "$s" > "$OUT/ref.txt" 2> /dev/null
  "$NF_BIN" "$g" "$s" "$NF_DELTAS" > "$OUT/out.txt" 2> "$OUT/$tag.err"
  local rc=$? ok=1
  [ $rc -eq 0 ] || ok=0
  cmp -s "$OUT/out.txt" "$OUT/ref.txt" || ok=0
  runs=$((runs + 1))
  if [ $ok -eq 1 ]; then echo "PASS $tag" | tee -a "$SUM"; rm -f "$OUT/$tag.err"
  else echo "FAIL $tag rc=$rc (log: $OUT/$tag.err)" | tee -a "$SUM"; fail=1; fi
}

while read -r name srcs; do
  case "$name" in ''|\#*) continue;; esac
  g="data/$name.gr"; [ -f "$g" ] || { echo "missing $g (run scripts/make_graphs.sh)"; exit 2; }
  for s in ${srcs%%#*}; do
    one "$g" "$s" 0 1                       # every call on the GPU, dirty upload
    nf "$g" "$s"
    [ $fail -eq 0 ] || break 2
  done
  first=$(echo "${srcs%%#*}" | awk '{print $1}')
  one "$g" "$first" 4096 1                  # mixed CPU/GPU calls, dirty upload
  one "$g" "$first" 4096 0                  # mixed CPU/GPU calls, full upload
  [ $fail -eq 0 ] || break
done < sources.txt

if [ $fail -eq 0 ] && [ -n "${FUZZ_DIR:-}" ]; then
  for g in "$FUZZ_DIR"/*.gr; do
    n=$(grep -m1 '^p' "$g" | awk '{print $3}')
    for s in 1 $((n / 3 > 0 ? n / 3 : 1)) $((2 * n / 3 > 0 ? 2 * n / 3 : 1)); do
      one "$g" "$s" 0 1; nf "$g" "$s"; [ $fail -eq 0 ] || break 2
    done
  done
fi
rm -f "$OUT/out.txt" "$OUT/ref.txt"
echo "checked runs: $runs, result: $([ $fail -eq 0 ] && echo ALL PASS || echo FAILURE)" | tee -a "$SUM"
exit $fail
