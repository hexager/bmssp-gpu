#!/usr/bin/env bash
# Puts the five benchmark graphs in data/ and verifies them byte-for-byte.
#   NY, Grid500, RMAT16: copy from Siddharth's bmssp-baselines/data (frozen set).
#   RMAT18, RMAT20: generated here with the same generator and seed as RMAT16.
# usage: scripts/make_graphs.sh PATH/TO/bmssp-baselines/data
set -euo pipefail
cd "$(dirname "$0")/.."
SRC="${1:?usage: make_graphs.sh PATH_TO_bmssp-baselines_data}"
mkdir -p data
for g in USA-road-d.NY grid500 rmat16; do [ -f "data/$g.gr" ] || cp "$SRC/$g.gr" data/; done
for sc in 18 20; do
  [ -f "data/rmat$sc.gr" ] || python3 scripts/generate_rmat.py --scale "$sc" --edge-factor 8 --seed 42 --output "data/rmat$sc.gr"
done
(cd data && sha256sum -c "$OLDPWD/graphs.sha256")
