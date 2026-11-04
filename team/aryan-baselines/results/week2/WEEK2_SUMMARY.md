# Aryan - Week 2 Measurement (H1 baselines, H2/H3 harness)

## 1. Status

| Item | Status |
|---|---|
| Relaxation counters, Dijkstra and CPU delta-stepping | DONE (reused Siddharth counters) |
| H1 baseline table and plot | DONE |
| Counter header for BMSSP (include/instrument.hpp) | DONE, hooks pending BMSSP code |
| H1 BMSSP column | PENDING (needs Aniketh BMSSP) |
| H3 sweep of k and t | Script ready, PENDING BMSSP |
| H2 Nsight profiling | Script ready, PENDING GPU hybrid + NVIDIA GPU |

## 2. Datasets (identical to Siddharth Week 2)

| Dataset | Vertices | Edges |
|---|---:|---:|
| RMAT16 | 65,536 | 524,288 |
| Grid500 | 250,000 | 998,000 |
| USA-road-d.NY | 264,346 | 733,846 |

## 3. Method

3 datasets, 5 frozen sources each (see sources.txt), 3 repetitions per source,
2 algorithms (CPU Dijkstra, CPU delta-stepping), 90 runs. Delta: RMAT16=16, Grid500=16, NY=5000.
Delta-stepping distances checked against Dijkstra: 0 mismatches.

Relaxation and edge counts are deterministic and independent of hardware.
Runtimes in this folder were measured on a shared single-core cloud sandbox,
not on the team machine, so use them only for relative comparison.

## 4. H1 baseline results (median over 5 sources)

| Dataset | Dijkstra relax | Delta relax | Delta / Dijkstra | BMSSP relax |
|---|---:|---:|---:|---|
| RMAT16 | 65,831 | 58,300 | 0.886 | PENDING |
| Grid500 | 329,459 | 325,366 | 0.988 | PENDING |
| NY | 299,009 | 323,665 | 1.082 | PENDING |

Delta-stepping does fewer successful relaxations than Dijkstra on RMAT16 and Grid500,
and more on NY (delta=5000 re-relaxes vertices). Edges examined are higher for
delta-stepping on all three graphs. BMSSP must beat these numbers for H1.

Plot: plots/h1_baseline_relaxations.png

## 5. Remaining work

1. Receive BMSSP source, add hooks from include/instrument.hpp, run H1 column.
2. scripts/run_h3_sweep.sh for H3.
3. scripts/run_h2_nsys.sh for H2 (NVIDIA GPU).
