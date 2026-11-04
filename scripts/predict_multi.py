#!/usr/bin/env python3
"""Predicts the multi-source runs (fp_multi) from single-source measurements, before they are run.

usage: scripts/predict_multi.py BENCH_DIR [--scenarios 5@6,8@2] [--calibrate MULTI_DIR] > prediction.md
--calibrate: the batched code path has an extra cost per batch (thread hand-off and blocking synchronization). It is
measured from the one-search runs (scenario 1@0, one call per batch) in MULTI_DIR as (gpu_g0 wall - single-source
gpu_all run) / number of GPU calls, per graph, and added to the fixed cost F of every batch.
BENCH_DIR is a results/bench folder of the single-source benchmark (runs.csv and calls/*.csv).

Model (event simulation, one per graph and scenario "b@c" = b searches on c cores):
  - Search j replays the logs of source j mod 5 of that graph.
  - gpu: from the gpu_all run of that source: G_i = time of FindPivots call i (logged), C = run time - sum G_i is
    the CPU work outside FindPivots, split evenly into the gaps before each call.
  - cpu_hooked: the whole cpu_hooked run time is CPU work (no GPU).
  - CPU work of all runnable searches shares the c cores fairly (each runs at rate min(1, c / runnable)).
  - One GPU worker. When it is free it takes every waiting call as one batch. A batch costs F + sum_i (G_i - F),
    where F is the fixed per-call cost from the phase fit (upload, rounds and download intercepts, two rounds).
    So a batch of one costs exactly the logged time, and each extra call in a batch adds only its own marginal part.
  - Not modelled: CPU time used by the GPU worker, memory-bandwidth contention between threads, and SMT.
Output: predicted wall time of cpu_hooked and gpu (threshold 0, greedy), and their ratio.
"""
import csv, glob, heapq, os, statistics as st, sys
import numpy as np

def load_runs(path):
    last = {}
    for r in csv.DictReader(open(path)): last[(r["graph"], r["source"], r["rep"], r["config"])] = r
    return list(last.values())


def fixed_cost(rows):
    """Fixed part of one GPU call (ms): intercepts of the upload, rounds (2 rounds) and download fits."""
    f = lambda k: np.array([float(r[k]) for r in rows])
    def fit(y, *xs):
        X = np.column_stack([np.ones(len(y)), *xs]); c, *_ = np.linalg.lstsq(X, y, rcond=None); return c
    up = fit(f("upload_ms"), f("upload_vertices")); rd = fit(f("rounds_ms"), f("rounds"), f("edges"))
    dn = fit(f("download_ms"), f("download_vertices"))
    return max(0.0, up[0] + rd[0] + 2 * rd[1] + dn[0])


def simulate(searches, cores, F):
    """searches: list of (gaps, calls); gaps[i] = CPU ms before call i, plus a final gap; calls[i] = GPU ms.
    Returns (wall time in ms, number of GPU batches)."""
    S = len(searches)
    phase = [0] * S                    # index of the next gap/call
    rem = [s[0][0] for s in searches]  # remaining CPU work in the current gap
    state = ["cpu"] * S                # cpu | wait | gpu | done
    t = 0.0; queue = []; gpu_busy_until = None; batch = []; nbatches = 0
    while True:
        runnable = [j for j in range(S) if state[j] == "cpu"]
        if not runnable and gpu_busy_until is None and not queue: break
        # start a batch if the GPU is free
        if gpu_busy_until is None and queue:
            batch = queue[:]; queue.clear()
            cost = F + sum(max(0.0, searches[j][1][phase[j]] - F) for j in batch)
            gpu_busy_until = t + cost; nbatches += 1
            for j in batch: state[j] = "gpu"
            continue
        rate = min(1.0, cores / len(runnable)) if runnable else 0.0
        # next event: a CPU gap ends or the batch ends
        dt_cpu = min((rem[j] / rate for j in runnable), default=float("inf"))
        dt_gpu = (gpu_busy_until - t) if gpu_busy_until is not None else float("inf")
        dt = min(dt_cpu, dt_gpu)
        for j in runnable: rem[j] -= dt * rate
        t += dt
        if gpu_busy_until is not None and dt_gpu <= dt_cpu + 1e-12:
            gpu_busy_until = None
            for j in batch:
                phase[j] += 1; state[j] = "cpu"; rem[j] = searches[j][0][phase[j]]
            batch = []
        for j in runnable:
            if state[j] == "cpu" and rem[j] <= 1e-9:
                if phase[j] < len(searches[j][1]): state[j] = "wait"; queue.append(j)
                else: state[j] = "done"
    return t, nbatches


def main():
    D = sys.argv[1]
    scen = sys.argv[sys.argv.index("--scenarios") + 1] if "--scenarios" in sys.argv else "5@6,8@2"
    scenarios = [tuple(int(x) for x in s.split("@")) for s in scen.split(",")]
    calib = sys.argv[sys.argv.index("--calibrate") + 1] if "--calibrate" in sys.argv else None


    runs = load_runs(os.path.join(D, "runs.csv"))
    graphs = list(dict.fromkeys(r["graph"] for r in runs))
    med_exec = {}
    for r in runs: med_exec.setdefault((r["graph"], r["source"], r["config"]), []).append(float(r["execute_ms"]))
    med_exec = {k: st.median(v) for k, v in med_exec.items()}
    extra = {}
    if calib:
        mr = list(csv.DictReader(open(os.path.join(calib, "runs.csv"))))
        for g in graphs:
            rr = [r for r in mr if r["graph"] == g and r["size"] == "1" and r["config"] == "gpu_g0"]
            if not rr: continue
            s0 = next(r["source"] for r in runs if r["graph"] == g)   # first source = the one-search source
            wall = st.median(float(r["wall_ms"]) for r in rr); nc = int(rr[0]["calls_gpu"])
            extra[g] = (wall - med_exec[(g, s0, "gpu_all")]) / nc
            print(f"{g}: calibrated extra per GPU call {extra[g] * 1000:.0f} us", file=sys.stderr)


    print("# Prediction for the multi-source runs\n")
    print(f"From single-source data in `{D}`" + (f", per-call cost calibrated on the one-search runs in `{calib}`" if calib else "") + ". Model and its assumptions: see the docstring of scripts/predict_multi.py.\n")
    print("| Graph | Scenario (searches @ cores) | Predicted cpu_hooked wall ms | Predicted gpu wall ms | Predicted speedup gpu vs cpu_hooked | Mean batch size (gpu) |")
    print("|---|---|---|---|---|---|")
    for g in graphs:
        srcs = sorted({r["source"] for r in runs if r["graph"] == g}, key=lambda s: [x["source"] for x in runs if x["graph"] == g].index(s))
        logs = {}
        for s in srcs:
            p = os.path.join(D, "calls", f"{g}_s{s}_gpu_all.csv")
            if os.path.exists(p): logs[s] = list(csv.DictReader(open(p)))
        if not logs: continue
        F = fixed_cost([r for rows in logs.values() for r in rows[1:]]) + extra.get(g, 0.0)
        gpu_s, cpu_s = [], []
        for s in srcs:
            logged = [float(r["ms"]) for r in logs[s]]
            T = med_exec[(g, s, "gpu_all")]; C = max(0.0, T - sum(logged))   # CPU work outside FindPivots
            calls = [x + extra.get(g, 0.0) for x in logged]
            gpu_s.append(([C / (len(calls) + 1)] * (len(calls) + 1), calls))
            cpu_s.append(([med_exec[(g, s, "cpu_hooked")]], []))
        for b, c in scenarios:
            G = [gpu_s[j % len(gpu_s)] for j in range(b)]; Cc = [cpu_s[j % len(cpu_s)] for j in range(b)]
            tc, _ = simulate(Cc, c, F); tg, nb = simulate(G, c, F)
            ncalls = sum(len(x[1]) for x in G)
            print(f"| {g} | {b}@{c} | {tc:.0f} | {tg:.0f} | {tc / tg:.3f}x | {ncalls / max(1, nb):.2f} |")
        print(f"\n{g}: fixed per-call GPU cost F = {F * 1000:.0f} us", file=sys.stderr)


if __name__ == "__main__":
    main()
