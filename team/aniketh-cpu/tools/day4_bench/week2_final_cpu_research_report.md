# BMSSP Week 2 — Final CPU/OpenMP Research Report

Date: 21 October 2026

## 1. Scope

CPU/OpenMP FindPivots experimental research.

Frozen CPU release: cpu-v1.4
Commit: 503232b02c74948c61bdbe084179397bb1dc67f7

The underlying BMSSP reference implementation is attributed
to Castro et al. The experimental contribution concerns
instrumentation, validation, CSR integration and OpenMP
FindPivots evaluation.

This report does not claim GPU acceleration.

## 2. Correctness

OpenMP v3 completed:

- 700 randomized correctness executions across 100 generated graphs.
- 54 multi-source correctness executions across six benchmark graphs.
- Final distances matched Dijkstra in these tests.
- Reported FindPivots invariants passed.

These executions reuse graph instances and are not independent
graph samples. Formal equivalence and race freedom are not proven.

## 3. FindPivots crossover

The per-call analysis found no sustained workload-size range
where four-thread synchronous OpenMP consistently beat the
sequential fallback under the analyzer's criteria.

Different execution modes can have different intermediate
call workloads. Crossover comparisons are descriptive.

## 4. Amdahl analysis

FindPivots-only idealized acceleration ceilings were
approximately 1.21-1.24x across the six evaluated graphs.

This is a theoretical zero-overhead upper bound derived
from CPU component shares, not a GPU measurement.

Accelerating FindPivots alone is insufficient to overcome
the observed BMSSP/Dijkstra execute-only gap.

Broader acceleration involving recursion-loop relaxation
and D.pull has greater theoretical potential, but requires
actual implementation and measurement.

## 5. OpenMP performance methodology

AMD Ryzen 5 3500U, four physical CPU cores.
OpenMP: four threads for parallel configurations.
CPU affinity: logical CPUs 0,2,4,6.

Three graphs, three selected sources per graph,
six configurations and seven planned repetitions.

The 378 executions were collected in two sessions:

- Original session: 310 records.
- Recovery session: 68 records.

All 63 graph-source-repeat groups had matching
distance hashes across their recorded configurations.

The two sessions are not established as statistically
equivalent. Within-session complete repeats are used
for the following descriptive summary.

## 6. Performance results

| Session | Graph | Source | Repeats | Built-in ms | Best experimental mode | Best ms | Speedup |
|---|---|---:|---:|---:|---|---:|---:|
| original | USA-road-t.NY.gr | 111478 | 7 | 195.902 | fallback | 213.234 | 0.9187x |
| original | USA-road-t.NY.gr | 164482 | 7 | 182.934 | fallback | 201.369 | 0.9085x |
| original | USA-road-t.NY.gr | 241025 | 7 | 183.693 | fallback | 200.619 | 0.9156x |
| original | r18.gr | 147059 | 7 | 373.456 | fallback | 372.689 | 1.0021x |
| original | r18.gr | 15904 | 7 | 374.090 | fallback | 373.593 | 1.0013x |
| original | r18.gr | 229561 | 7 | 374.381 | fallback | 373.857 | 1.0014x |
| original | r20.gr | 421302 | 7 | 2115.050 | fallback | 2232.700 | 0.9473x |
| original | r20.gr | 690940 | 2 | 2149.970 | fallback | 2248.200 | 0.9563x |
| recovery | r20.gr | 133736 | 7 | 2041.460 | fallback | 2138.180 | 0.9548x |
| recovery | r20.gr | 690940 | 4 | 2239.325 | omp4 | 3038.790 | 0.7369x |

The table selects the fastest experimental configuration
within each graph-source-session group. This is exploratory
post-hoc selection and is not a confirmatory speedup claim.

A speedup greater than 1 in one group does not establish
a general or statistically significant improvement.

## 7. Session stability

Recovery sequential-baseline controls:

- r20 source 690940: -20.80% before-to-after drift.
- r20 source 133736: -31.32% before-to-after drift.

The original session had no sequential baseline for
source 133736, preventing a direct session comparison.

Original and recovery results must remain separately labeled.
Do not claim a single stable uninterrupted 378-run experiment.

## 8. Limitations

- One CPU system.
- Only three graphs in the multi-source timing matrix.
- Seven planned repetitions per configuration.
- Interrupted experiment and separate recovery session.
- Substantial baseline drift in recovery controls.
- No formal race-freedom proof.
- No demonstrated general OpenMP speedup.
- No measured GPU acceleration.
- Potential thermal and frequency confounding.

## 9. Paper interpretation

The evaluated OpenMP FindPivots variants passed the tested
correctness campaigns, but the current synchronous parallel
design has not established a reproducible end-to-end advantage
over the sequential BMSSP implementation.

Amdahl analysis suggests that accelerating FindPivots alone
cannot close the observed CPU performance gap to Dijkstra.

These are useful negative and analytical findings.
They do not establish a GPU acceleration contribution.

## 10. Evidence

- openmp_day3_v1/raw_timings.csv
- openmp_day3_v1/metadata.json
- openmp_day3_recovery_v1/recovery_timings.csv
- openmp_day3_recovery_v1/baseline_controls.csv
- openmp_day3_recovery_v1/session_comparability_audit_v1.txt
- openmp_day3_session_summary_v1.csv
- amdahl_bounds_v1.csv
- fp_crossover_analysis_v1.txt
