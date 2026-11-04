#!/usr/bin/env python3
"""Benchmark matrix: graphs x sources x repetitions x configurations, one fresh process per run.

usage: scripts/bench.py [--bin bin/fp_hybrid] [--reps 5] [--graphs rmat20,rmat18] [--out results/bench]

Configurations (same binary, environment only):
  builtin      no hook, built-in in-place FindPivots (the CPU reference)
  cpu_hooked   hook installed, every call still built-in (measures the hook cost)
  sync_cpu     every call uses the serial synchronous algorithm on the CPU (separates algorithm from hardware)
  gpu_all      every call on the GPU (threshold 0)
  gpu_t1024 / gpu_t4096 / gpu_t16384   calls with first-round edge work >= threshold on the GPU
  gpu_t32768 / gpu_t65536              higher thresholds (fewer, larger GPU calls)
  gpu_t4096_full                       as gpu_t4096 but full state upload on every GPU call (no dirty tracking)
Baselines (other binaries):
  dijkstra     bin/dijkstra_ref, binary-heap Dijkstra on the CPU
  nearfar      bin/nearfar, GPU Near-Far SSSP; one process runs a warm-up and then every delta in NEARFAR_DELTAS,
               one row per delta (config name nearfar_<delta>)
Configuration order is shuffled per (graph, source, rep) with a fixed seed. Every run's distances are hashed;
all configurations of one (graph, source) must agree. Per-call logs are written for rep 1.
"""
import argparse, csv, hashlib, os, random, re, subprocess, sys, time

CONFIGS = {
    "builtin":        {"FP_MODE": "builtin"},
    "cpu_hooked":     {"FP_MODE": "gpu", "FP_GPU_MIN_EDGES": "1000000000000"},
    "sync_cpu":       {"FP_MODE": "sync"},
    "gpu_all":        {"FP_MODE": "gpu", "FP_GPU_MIN_EDGES": "0"},
    "gpu_t1024":      {"FP_MODE": "gpu", "FP_GPU_MIN_EDGES": "1024"},
    "gpu_t4096":      {"FP_MODE": "gpu", "FP_GPU_MIN_EDGES": "4096"},
    "gpu_t16384":     {"FP_MODE": "gpu", "FP_GPU_MIN_EDGES": "16384"},
    "gpu_t32768":     {"FP_MODE": "gpu", "FP_GPU_MIN_EDGES": "32768"},
    "gpu_t65536":     {"FP_MODE": "gpu", "FP_GPU_MIN_EDGES": "65536"},
    "gpu_t4096_full": {"FP_MODE": "gpu", "FP_GPU_MIN_EDGES": "4096", "FP_GPU_DIRTY": "0"},
    "dijkstra":       None,
    "nearfar":        None,
}
NEARFAR_DELTAS = "x0.0625,x0.125,x0.25,x0.5,x1,x2,x4,x8,x32,x128,x512"
LOG_CONFIGS = {"cpu_hooked", "sync_cpu", "gpu_all", "gpu_t4096"}
FIELDS = ["graph", "source", "rep", "order", "config", "execute_ms", "prepare_ms", "gpu_setup_ms", "calls_builtin",
          "calls_gpu", "calls_sync", "gpu_upload_ms", "gpu_rounds_ms", "gpu_download_ms", "gpu_kernel_ms", "gpu_xfer_ms",
          "gpu_host_ms", "upload_vertices", "download_vertices", "gpu_edges", "launches", "rounds", "delta", "iterations",
          "far_phases", "edges_examined", "device", "dist_sha256", "start_utc"]


def kv_of(line):
    return {k: v.strip('"') for k, v in re.findall(r'(\w+)=("[^"]*"|\S+)', line)}


def parse(stderr, prefix="HYBRID"):
    return kv_of(" ".join(l for l in stderr.splitlines() if l.startswith(prefix)))


def run_config(c, a, path, s, env):
    """Returns (stdout, [(config_name, fields), ...])."""
    if c == "dijkstra": cmd = [a.dijkstra, path, str(s)]
    elif c == "nearfar": cmd = [a.nearfar, path, str(s), NEARFAR_DELTAS]
    else: cmd = [a.bin, path, str(s)]
    p = subprocess.run(cmd, env=env, capture_output=True)
    err = p.stderr.decode()
    if p.returncode != 0:
        sys.exit(f"run failed: {path} {s} {c} rc={p.returncode}\n{err[-2000:]}")
    if c == "dijkstra": return p.stdout, [(c, parse(err, "DIJKSTRA"))]
    if c == "nearfar":
        out = []
        for l in err.splitlines():
            if l.startswith("NEARFAR"):
                kv = kv_of(l); kv["gpu_setup_ms"] = kv.pop("setup_ms", "")
                out.append((f"nearfar_{kv['delta_arg']}", kv))
        return p.stdout, out
    return p.stdout, [(c, parse(err))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default="bin/fp_hybrid")
    ap.add_argument("--dijkstra", default="bin/dijkstra_ref")
    ap.add_argument("--nearfar", default="bin/nearfar")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--graphs", default="")
    ap.add_argument("--configs", default=",".join(CONFIGS))
    ap.add_argument("--out", default="results/bench")
    ap.add_argument("--seed", type=int, default=20261024)
    a = ap.parse_args()
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    os.makedirs(os.path.join(a.out, "calls"), exist_ok=True)
    configs = a.configs.split(",")
    sources = {}
    for line in open("sources.txt"):
        line = line.split("#")[0].split()
        if line: sources[line[0]] = [int(x) for x in line[1:]]
    graphs = a.graphs.split(",") if a.graphs else list(sources)
    csv_path = os.path.join(a.out, "runs.csv")
    new = not os.path.exists(csv_path)
    f = open(csv_path, "a", newline=""); w = csv.DictWriter(f, fieldnames=FIELDS)
    if new: w.writeheader()
    rng = random.Random(a.seed)
    env0 = dict(os.environ, CUDA_MODULE_LOADING="EAGER")
    bad = 0
    for g in graphs:
        path = f"data/{g}.gr"
        for s in sources[g]:
            hashes = {}
            for rep in range(1, a.reps + 1):
                order = configs[:]; rng.shuffle(order)
                for i, c in enumerate(order):
                    env = dict(env0, **(CONFIGS[c] or {}))
                    log = os.path.join(a.out, "calls", f"{g}_s{s}_{c}.csv")
                    if rep == 1 and c in LOG_CONFIGS and not os.path.exists(log): env["FP_CALL_LOG"] = log
                    t = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                    out, results = run_config(c, a, path, s, env)
                    h = hashlib.sha256(out).hexdigest()
                    hashes.setdefault(h, []).append(c)
                    for name, kv in results:
                        row = {k: kv.get(k, "") for k in FIELDS}
                        row.update(graph=g, source=s, rep=rep, order=i, config=name, dist_sha256=h, start_utc=t)
                        w.writerow(row); f.flush()
                        print(f"{g} s={s} rep={rep} {name:15s} execute_ms={kv.get('execute_ms')} gpu_calls={kv.get('calls_gpu', '')}",
                              flush=True)
            if len(hashes) != 1:
                bad += 1; print(f"DISTANCE HASH DISAGREEMENT {g} {s}: {hashes}", flush=True)
    f.close()
    print("done;", "all distance hashes agree" if bad == 0 else f"{bad} (graph, source) pairs DISAGREE")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
