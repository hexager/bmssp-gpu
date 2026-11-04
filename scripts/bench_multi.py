#!/usr/bin/env python3
"""Benchmark of several sources at once (fp_multi): graphs x number of sources x repetitions x configurations.

usage: scripts/bench_multi.py [--bin bin/fp_multi] [--reps 3] [--scenarios 1@0,5@0,8@2] [--graphs ...] [--out results/multi]

A scenario b@c runs the first b sources of the graph's line in sources8.txt at the same time, one CPU thread each,
on c physical cores (0 = no pinning). With b > c, a search waiting for the GPU frees its core for another search.
Configurations:
  dijkstra        binary-heap Dijkstra per thread
  builtin         BMSSP per thread, no hook
  cpu_hooked      BMSSP per thread, hook and dirty tracking, all FindPivots on the CPU (control for the GPU runs)
  gpu_g0 / gpu_g1024 / gpu_g4096   GPU worker takes every waiting call (greedy); threshold 0 / 1024 / 4096 edges
  gpu_a0          GPU worker waits until every unfinished search is waiting (all); threshold 0
Diagnosis configurations (not in the default list): gpu_g0_spin (worker spin-waits on the GPU instead of sleeping),
  gpu_g0_w (worker gets one more physical core of its own; needs a pinned scenario), gpu_g0_spin_w (both).
Order is shuffled per (graph, size, rep) with a fixed seed; one fresh process per run. All configurations of a
(graph, size) must print the same distances (hash checked). A restarted run overwrites, not appends.
"""
import argparse, csv, hashlib, os, random, re, subprocess, sys, time

CONFIGS = {
    "dijkstra":   {"MULTI_MODE": "dijkstra"},
    "builtin":    {"MULTI_MODE": "builtin"},
    "cpu_hooked": {"MULTI_MODE": "cpu_hooked"},
    "gpu_g0":     {"MULTI_MODE": "gpu", "MULTI_BATCH": "greedy", "FP_GPU_MIN_EDGES": "0"},
    "gpu_g1024":  {"MULTI_MODE": "gpu", "MULTI_BATCH": "greedy", "FP_GPU_MIN_EDGES": "1024"},
    "gpu_g4096":  {"MULTI_MODE": "gpu", "MULTI_BATCH": "greedy", "FP_GPU_MIN_EDGES": "4096"},
    "gpu_a0":     {"MULTI_MODE": "gpu", "MULTI_BATCH": "all", "FP_GPU_MIN_EDGES": "0"},
    # diagnosis of the batched path's extra cost (use with pinned scenarios, e.g. --scenarios 1@1,8@2)
    "gpu_g0_spin":   {"MULTI_MODE": "gpu", "MULTI_BATCH": "greedy", "FP_GPU_MIN_EDGES": "0", "FP_BATCH_SYNC": "spin"},
    "gpu_g0_w":      {"MULTI_MODE": "gpu", "MULTI_BATCH": "greedy", "FP_GPU_MIN_EDGES": "0", "MULTI_WORKER_CORE": "1"},
    "gpu_g0_spin_w": {"MULTI_MODE": "gpu", "MULTI_BATCH": "greedy", "FP_GPU_MIN_EDGES": "0", "FP_BATCH_SYNC": "spin",
                      "MULTI_WORKER_CORE": "1"},
}
DEFAULT_CONFIGS = ["dijkstra", "builtin", "cpu_hooked", "gpu_g0", "gpu_g1024", "gpu_g4096", "gpu_a0"]
FIELDS = ["graph", "size", "cores", "rep", "order", "config", "wall_ms", "sum_ms", "calls_cpu", "calls_gpu", "batches",
          "max_batch", "upload_ms", "rounds_ms", "download_ms", "wait_ms", "edges", "launches", "hist", "cpus", "device",
          "dist_sha256", "start_utc"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default="bin/fp_multi")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--scenarios", default="1@0,5@0,8@2")
    ap.add_argument("--sources", default="sources8.txt")
    ap.add_argument("--graphs", default="")
    ap.add_argument("--configs", default=",".join(DEFAULT_CONFIGS))
    ap.add_argument("--out", default="results/multi")
    ap.add_argument("--seed", type=int, default=20261026)
    a = ap.parse_args()
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    os.makedirs(a.out, exist_ok=True)
    sources = {}
    for line in open(a.sources):
        line = line.split("#")[0].split()
        if line: sources[line[0]] = [int(x) for x in line[1:]]
    graphs = a.graphs.split(",") if a.graphs else list(sources)
    scenarios = [tuple(int(x) for x in sc.split("@")) for sc in a.scenarios.split(",")]
    configs = a.configs.split(",")
    f = open(os.path.join(a.out, "runs.csv"), "w", newline=""); w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader()
    rng = random.Random(a.seed)
    env0 = dict(os.environ, CUDA_MODULE_LOADING="EAGER")
    bad = 0
    for g in graphs:
        for b, cores in scenarios:
            if b > len(sources[g]): sys.exit(f"{g}: only {len(sources[g])} sources in {a.sources}")
            src = ",".join(str(s) for s in sources[g][:b])
            env_s = dict(env0, MULTI_CORES=str(cores)) if cores > 0 else env0
            hashes = {}
            for rep in range(1, a.reps + 1):
                order = configs[:]; rng.shuffle(order)
                for i, c in enumerate(order):
                    t = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                    p = subprocess.run([a.bin, f"data/{g}.gr", src], env=dict(env_s, **CONFIGS[c]), capture_output=True)
                    err = p.stderr.decode()
                    if p.returncode != 0: sys.exit(f"run failed: {g} b={b} {c} rc={p.returncode}\n{err[-2000:]}")
                    h = hashlib.sha256(p.stdout).hexdigest(); hashes.setdefault(h, []).append(c)
                    kv = {k: v.strip('"') for k, v in re.findall(r'(\w+)=("[^"]*"|\S+)',
                          " ".join(l for l in err.splitlines() if l.startswith("MULTI")))}
                    row = {k: kv.get(k, "") for k in FIELDS}
                    row.update(graph=g, size=b, cores=cores, rep=rep, order=i, config=c, dist_sha256=h, start_utc=t)
                    w.writerow(row); f.flush()
                    print(f"{g} {b}@{cores} rep={rep} {c:11s} wall_ms={kv.get('wall_ms')} batches={kv.get('batches', '')}", flush=True)
            if len(hashes) != 1: bad += 1; print(f"DISTANCE HASH DISAGREEMENT {g} {b}@{cores}: {hashes}", flush=True)
    f.close()
    print("done;", "all distance hashes agree" if bad == 0 else f"{bad} (graph, size) scenarios DISAGREE")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
