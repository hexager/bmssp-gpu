#!/usr/bin/env bash
# H2: Nsight Systems profile of the hybrid GPU build. Needs NVIDIA GPU + Siddhanth hybrid binary. PENDING.
set -euo pipefail; mkdir -p results/week2/h2
nsys profile -o results/week2/h2/rmat16_prof --stats=true ./run_hybrid data/rmat16.gr 1 > results/week2/h2/rmat16_nsys.txt
