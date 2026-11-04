# Siddharth - Week 2 Baseline Experiments

## 1. Experimental Hardware

- CPU: Intel Core i7-12700
- Logical CPUs: 20
- RAM: 15 GiB
- GPU: NVIDIA GeForce GT 1030
- GPU memory: approximately 2 GiB
- Compute capability: 6.1
- CUDA Toolkit: 11.8
- OS: Debian GNU/Linux 12

## 2. Dataset Table

| Dataset | Type | Vertices | Edges |
|---|---|---:|---:|
| RMAT16 | Synthetic RMAT | 65,536 | 524,288 |
| Grid500 | Synthetic 2D Grid | 250,000 | 998,000 |
| USA-road-d.NY | DIMACS road network | 264,346 | 733,846 |

All three datasets passed CSR validation.

## 3. Experimental Method

The fixed baseline experiment used:

- 3 graph datasets
- 5 frozen source vertices per graph
- 5 repetitions per source
- 3 available baseline algorithms
- 225 total measured executions

Algorithms tested:

1. CPU Dijkstra
2. CPU Delta-Stepping
3. CUDA GPU Delta-Stepping

CPU and GPU Delta-Stepping distances were checked against Dijkstra.

## 4. Frozen Sources

# Week 2 frozen source vertices
# DIMACS numbering (1-based)

RMAT16 1 132 263 394 525
GRID500 1 501 1001 1501 2001
NY 1 529 1058 1587 2115

## 5. Fixed Baseline Results

The following results contain 25 measurements per algorithm/dataset
(5 sources x 5 repetitions).

For GPU Delta-Stepping, GPU section time is used for the algorithm
comparison rather than CUDA initialization/setup time.

Dataset     Algorithm          N    Median(ms)      Mean(ms)     Min(ms)     Max(ms)
GRID500     cpu_delta         25       30.6968       30.7296     30.5587     31.4829
GRID500     dijkstra          25       31.6547       31.6890     31.4204     32.1378
GRID500     gpu_delta         25      442.2550      444.4063    438.1280    477.5770
NY          cpu_delta         25       23.7082       23.7280     23.4596     24.4964
NY          dijkstra          25       23.6248       23.6101     23.3493     23.9290
NY          gpu_delta         25      278.2580      275.1086    264.8140    281.8640
RMAT16      cpu_delta         25       10.5164       10.6397     10.1929     11.2254
RMAT16      dijkstra          25       10.6738       10.6824     10.5171     11.1573
RMAT16      gpu_delta         25       42.1256       41.9974     34.1483     53.9701

## 6. GPU Delta Parameter Sweep

Three repetitions were performed for every tested Delta value using
source vertex 1.

Dataset      Delta   GPU med(ms)     Total med    Iterations         Edges
GRID500          2      2000.820      2049.150         11369       1664753
GRID500          4      1083.970      1131.570          5773       2058463
GRID500          8       662.222       708.991          2911       2656231
GRID500         16       442.476       489.641          1461       3525439
GRID500         32       326.767       373.987           732       4821309
GRID500         64       245.634       292.926           366       6753125
GRID500        128       188.773       235.731           183       9371762
NY             250      1065.150      1111.720          5651       1460407
NY             500       730.956       776.700          2844       2105319
NY            1000       523.537       569.660          1431       2896517
NY            2000       386.597       433.316           719       4074634
NY            5000       285.001       331.426           289       6858664
RMAT16           1        42.985        87.591           177       1003917
RMAT16           2        35.525        79.406            94       1357274
RMAT16           4        30.662        74.091            49       1968326
RMAT16           8        35.652        78.469            25       2675542
RMAT16          16        45.467        88.711            13       3991582
RMAT16          32        43.382        86.811             7       4421781
RMAT16          64        49.088        93.221             4       4723599

## 7. Delta Sweep Findings

The best tested Delta depended strongly on graph structure.

- RMAT16 achieved its lowest tested GPU section time at Delta = 4.
- Grid500 continued improving through the tested range up to Delta = 128.
- USA-road-d.NY continued improving through the tested range up to Delta = 5000.

Increasing Delta generally reduced the number of bucket iterations
while increasing the number of examined edges.

This shows a trade-off between bucket-management overhead and
redundant relaxation work.

## 8. GPU Correctness Validation

GPU Delta-Stepping was validated against CPU Dijkstra using:

- Tiny known-answer graphs
- Multiple Delta values
- A light/heavy-edge stress graph
- A disconnected graph
- Grid10
- RMAT8
- 120 randomized graph/Delta tests
- RMAT16
- Grid500
- USA-road-d.NY

All tested configurations produced zero distance mismatches.

## 9. Current GPU Baseline Limitation

The CUDA implementation is a correctness-first baseline rather than
a fully optimized GPU implementation.

It performs redundant frontier and relaxation work. Consequently,
its edge-examination counts can be substantially larger than the
CPU Delta-Stepping reference.

The current GPU timings should therefore be interpreted as a
functional GPU Delta-Stepping baseline, not as the final optimized
GPU performance of the project.

## 10. Optional Gunrock Baseline Attempt

OPTIONAL GUNROCK BASELINE ATTEMPT
=================================

Gunrock version:
v1.2 checkout / release commit 5ee3df50c

Target hardware:
NVIDIA GeForce GT 1030
Compute capability: 6.1

CUDA Toolkit:
11.8

Initial host compiler:
GCC/G++ 12.2

Compatible host compiler tested:
GCC/G++ 11.3

Configuration:
Gunrock was successfully configured specifically for SM 6.1:
-gencode=arch=compute_61,code=sm_61
-gencode=arch=compute_61,code=compute_61

Build target:
sssp

Result:
BUILD FAILED due to compatibility problems between the legacy
Gunrock v1.2 dependency stack and the CUDA 11.8 Thrust/CUB stack.

Initial issues:
1. CUDA 11.8 rejected GCC 12 as an unsupported host compiler.
2. This was addressed by switching to the already-installed GCC/G++ 11.3.

Remaining issue:
Gunrock v1.2 uses an older bundled CUB dependency, while CUDA 11.8
ships a newer integrated Thrust/CUB stack.

Compilation produced extensive CUB/Thrust incompatibility errors,
including template/redefinition errors and CUB namespace/configuration
errors.

Decision:
The optional Gunrock baseline was not pursued further because resolving
the legacy dependency incompatibility would require invasive dependency
or toolchain changes and could destabilize the validated CUDA experiment
environment.

The project's own GPU Delta-Stepping baseline remains available and
correctness-validated.

## 11. Pending Team Integration

The following items require implementations/results from other team members:

- Sequential Duan/BMSSP reference
- Hybrid BMSSP implementation
- FindPivots GPU measurements
- Final all-algorithm comparison

These algorithms should use the same frozen datasets and source
vertices documented in this experiment.
