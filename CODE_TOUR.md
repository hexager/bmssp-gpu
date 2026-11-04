# Code tour

A guided path through the code, in the order of the talk. Line numbers refer to this commit.

## Map

| Folder / file | What it is |
|---|---|
| `src/bmssp_hybrid.hpp` | The CPU reference: Castro et al.'s BMSSP (MIT licence), with instrumentation and a FindPivots hook. |
| `src/sync_fp.hpp` | Serial synchronous FindPivots: the correctness reference for the GPU versions. |
| `src/fp_gpu.cu`, `src/fp_gpu.h` | GPU FindPivots (one search). |
| `src/fp_hybrid.cpp` | Driver: runs BMSSP and sends FindPivots calls to the CPU or the GPU. |
| `src/fp_batch.cu`, `src/fp_multi.cpp` | Batched GPU FindPivots for several searches at once. |
| `src/nearfar.cu` | GPU Near-Far SSSP baseline. |
| `src/dijkstra_ref.cpp` | Binary-heap Dijkstra: CPU baseline and correctness oracle. |
| `src/make_breakdown_header.py`, `src/breakdown.cpp` | Time breakdown: records and replays every operation on the data structure D. |
| `scripts/` | Correctness campaigns, benchmarks, model, prediction, figures. |
| `team/` | Earlier work of each team member (baselines, OpenMP FindPivots, profiling). |
| `results/laptop/` | All measurements on the GTX 1660 Ti laptop. |
| `paper/` | The paper (`main.pdf`) and its LaTeX source. |

## 1. The algorithm (Siddharth)

- `src/bmssp_hybrid.hpp:1419` `bmsspRec`: one BMSSP call (Algorithm 3 of the paper).
  - `:1430` FindPivots is called, through the hook if one is installed.
  - `:1471` `D.pull()`: take the next batch of closest vertices.
  - `:1490` the relax loop over the edges of completed vertices; improved vertices go back into D.
  - `:1511` `D.batchPrepend`.
- `src/bmssp_hybrid.hpp:1333` `findPivots`: the original in-place FindPivots (Algorithm 1).
- `src/bmssp_hybrid.hpp:703` `batchPQ`: the data structure D (a list of blocks, a set of block bounds, hash maps).
  `:748` insert and `:787` pull show why each operation is expensive: several pointer-chasing steps.
- Graphs: `scripts/generate_rmat.py` (R-MAT generator), `sources.txt` (5 sources per graph),
  `graphs.sha256` (exact graph files), `team/siddharth-baselines/` (graph tools, Dijkstra and delta-stepping).

## 2. Baselines and work counts (Aryan)

- `team/aryan-baselines/src/dijkstra.cpp`, `delta_stepping.cpp`: CPU baselines with relaxation counters.
- `team/aryan-baselines/include/instrument.hpp`: the counter header.
- `team/aryan-baselines/results/week2/WEEK2_SUMMARY.md`: Dijkstra against delta-stepping, distances checked.
- BMSSP's edge counts come from the `STATS` line of the reference (`fp_edges`, `rec_edges`, `base_edges`):
  BMSSP examines 2.5 to 4.2 times as many edges as Dijkstra.

## 3. CPU reference and where the time goes (Aniketh)

- `src/bmssp_hybrid.hpp:1232` `struct Stats`: per-phase timers and counters (the instrumentation).
- `src/bmssp_hybrid.hpp:1437-1446`: invariant checks of every FindPivots call (`BMSSP_CHECK=1`).
- `src/bmssp_hybrid.hpp:1262` `findPivotsImpl`: the hook that lets the driver replace FindPivots.
- `team/aniketh-cpu/week2_research_tools/fp_openmp_v3.cpp`: the OpenMP synchronous FindPivots.
- `src/sync_fp.hpp`: its serial port, used as the reference.
  - `:30` each proposal is built from the state at the start of the round.
  - `:39` winners are committed in vertex order, so the result does not depend on thread order.
  - `:47` early exit when W grows beyond k|S|.
- Time breakdown: `src/make_breakdown_header.py:13` adds a record of every D operation;
  `:48` `replay_D` replays it on fresh D objects and times each kind; `:67` each replayed pull is checked by hash.

## 4. GPU FindPivots (Siddhanth)

`src/fp_gpu.cu`:

- `:107` `proposal`: thread e finds its edge with a binary search over the edge offsets of the active list, then builds the
  candidate key (distance, hops, vertex, predecessor) and compares it with the current one.
- `:128` `k_pass1`: every improving proposal does `atomicMin` on the candidate distance.
- `:134` `k_pass2`: proposals with the winning distance do one 64-bit `atomicMin` on `(hops << 32) | pred`.
- `:140` `k_commit_a` and `:144` `k_commit_b`: write the winners. Two kernels, so that no thread reads `root[]` while
  another writes it (merging them is a real race, and the check catches it).
- `:270` `fpgpu_findpivots`: the round loop. `:287` up to k rounds; `:307-312` the launches; `:326` new vertices sorted
  like the CPU version; `:334` early exit; `:340-346` pivots P from tree sizes.
- `:247` `fpgpu_upload_state`: uploads only the vertices the CPU changed since the last call.
  The change log is one added line in the reference, `src/bmssp_hybrid.hpp:1221`.

`src/fp_hybrid.cpp`:

- `:109` the hook. `:118` calls below the edge threshold stay on the CPU.
- `:122` check mode: copy the state, run the serial reference, restore, run the GPU, compare everything;
  `:133` any difference stops the run.

## 5. Batching, Near-Far, model (Siddhanth)

- `src/fp_batch.cu:115-118`: search i's vertex v is `i*n + v`, so one launch serves calls from many searches.
  `:157` each search keeps its own bound. `:318` results are split back per search.
- `src/fp_multi.cpp:162` the GPU worker thread; `:173` greedy or wait-for-all batching; `:244` a search hands over its call.
- `src/nearfar.cu:87` `k_relax`: one thread per frontier edge; `:172` CUB scan of frontier degrees; `:195` the far-phase jump.
- `scripts/model.py:62`: the two bounds (FindPivots free, best per-call choice).
- `scripts/predict_multi.py:41` `simulate`: the event simulation behind the batching prediction; `:55` cost of one batch.

## How correctness is checked

| Script | Checks |
|---|---|
| `scripts/check_all.sh` | Every GPU FindPivots call equals the reference; Near-Far equals Dijkstra for 11 values of delta. |
| `scripts/check_multi.sh` | Every batched call equals the reference; all distances equal Dijkstra. |
| `tests/test_full_upload_once.sh` | Regression test: only the first GPU call uploads the whole state. |
| `results/laptop/check_gpu/summary.txt`, `check_batching_gpu/summary.txt` | The passing runs on the GPU. |
