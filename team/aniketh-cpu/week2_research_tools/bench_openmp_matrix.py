#!/usr/bin/env python3
"""One-file benchmark runner for BMSSP configurations: fresh process per run, rotated order, per-run machine state.

  python3 tools/bench_openmp_matrix.py --config-file tools/configs_day3.json \
      --sources-csv tools/day4_bench/source_sample_v1/selected_sources.csv --per-graph 3 \
      --graph-dir tools/day4_bench --graph-dir ~/bmssp/tests/graphs \
      --graphs USA-road-t.NY.gr r18.gr r20.gr --repeats 7 --require-ac --cooldown-sec 3 --out tools/day4_bench/openmp_day3_v1

Config file (JSON list), {graph} and {source} are substituted; "env" is added to the environment of that config only:
  [{"name":"builtin","cmd":["tools/sp_run_instr","bmssp","{graph}","{source}","1"],"env":{"OMP_NUM_THREADS":"4"}},
   {"name":"fallback","cmd":["tools/fp_omp3","{graph}","{source}"],"env":{"FP_MODE":"fallback","OMP_NUM_THREADS":"4"}}]
For every run it records: wall time, execute time parsed from the program ("OPENMP_EXECUTE_MS x" or "run 1 execute_ms x"),
FindPivots time, serial-commit time, CPU user/system seconds (resource usage of the child), AC state, 1-min load average,
CPU frequency (mean/max over cores) and temperature before and after, and the SHA-256 of the distances. All configs must
return identical distances in every (graph, source, repeat). The output directory must not exist. Analyse the result with
openmp_session_report.py. This script never modifies anything outside the output directory.
"""
import argparse, csv, glob, hashlib, json, os, platform, re, resource, statistics, subprocess, sys, time
from datetime import datetime, timezone

ap = argparse.ArgumentParser()
ap.add_argument("--config-file", required=True); ap.add_argument("--sources-csv", required=True)
ap.add_argument("--per-graph", type=int, default=3); ap.add_argument("--graph-dir", action="append", default=[])
ap.add_argument("--graphs", nargs="+", required=True); ap.add_argument("--repeats", type=int, default=5)
ap.add_argument("--require-ac", action="store_true"); ap.add_argument("--cooldown-sec", type=float, default=0.0)
ap.add_argument("--out", required=True); ap.add_argument("--timeout", type=float, default=600.0)
a = ap.parse_args()
if os.path.exists(a.out): sys.exit(f"STOP: output directory exists: {a.out}")

def find_graph(name):
    for d in a.graph_dir + ["."]:
        p = os.path.join(os.path.expanduser(d), name)
        if os.path.isfile(p): return p
    sys.exit(f"graph {name} not found in {a.graph_dir}")

def sha256(path, text=False):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

def read(p):
    try: return open(p).read().strip()
    except OSError: return ""

def ac_online():
    vals = []
    for ps in glob.glob("/sys/class/power_supply/*"):
        if read(ps + "/type") == "Mains" and read(ps + "/online") != "": vals.append(read(ps + "/online"))
    return "" if not vals else ("1" if "1" in vals else "0")

def freqs():
    v = [int(x) for x in (read(p) for p in glob.glob("/sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_cur_freq")) if x.isdigit()]
    return (round(statistics.mean(v) / 1000, 1), round(max(v) / 1000, 1)) if v else ("", "")

def temp():
    for hw in glob.glob("/sys/class/hwmon/hwmon*"):
        if read(hw + "/name") in ("k10temp", "coretemp", "zenpower") and read(hw + "/temp1_input").isdigit():
            return round(int(read(hw + "/temp1_input")) / 1000, 1)
    return ""

configs = json.load(open(a.config_file))
names = [c["name"] for c in configs]
bins = {}
for c in configs:
    exe = os.path.expanduser(c["cmd"][0])
    if not os.path.isfile(exe): sys.exit(f"binary missing: {exe}")
    bins[c["name"]] = sha256(exe)
src = {}
for r in csv.DictReader(open(a.sources_csv, newline="")):
    src.setdefault(r["graph"], []).append(int(r["source"]))
plan = [(g, s) for g in a.graphs for s in src[g][:a.per_graph]]
if a.require_ac and ac_online() != "1": sys.exit("STOP: AC power not connected")
os.makedirs(a.out)
FIELDS = ["graph", "source", "repeat", "config", "order_in_repeat", "start_utc", "end_utc", "execute_ms", "findpivots_ms", "serial_commit_ms", "prepare_ms",
          "wall_ms", "cpu_user_s", "cpu_sys_s", "ac_before", "ac_after", "load1", "freq_mean_before_mhz", "freq_max_before_mhz",
          "freq_mean_after_mhz", "freq_max_after_mhz", "temp_before_c", "temp_after_c", "distance_sha256"]
meta = dict(argv=sys.argv, started_utc=datetime.now(timezone.utc).isoformat(), host=platform.node(), python=platform.python_version(),
            governor=read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"), config_file_sha256=sha256(a.config_file),
            configs=configs, binary_sha256=bins, plan=[list(p) for p in plan], repeats=a.repeats)
try: meta["git_head"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
except Exception: meta["git_head"] = ""
f = open(os.path.join(a.out, "raw_timings.csv"), "w", newline=""); w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader()

def num(pat, text):
    m = re.search(pat, text); return float(m.group(1)) if m else ""
rows = []; completed = 0; total = len(plan) * a.repeats * len(configs)
try:
    for si, (g, s) in enumerate(plan):
        path = find_graph(g)
        for rep in range(1, a.repeats + 1):
            shift = (rep - 1 + si) % len(configs)
            order = names[shift:] + names[:shift]
            for oi, name in enumerate(order):
                c = next(x for x in configs if x["name"] == name)
                cmd = [os.path.expanduser(x).replace("{graph}", path).replace("{source}", str(s)) if isinstance(x, str) else str(x) for x in c["cmd"]]
                env = dict(os.environ); env.update(c.get("env", {}))
                ac0, (fm0, fx0), t0c, load = ac_online(), freqs(), temp(), os.getloadavg()[0]
                if a.require_ac and ac0 != "1": raise RuntimeError("AC lost before a run")
                ru0 = resource.getrusage(resource.RUSAGE_CHILDREN); st = datetime.now(timezone.utc); t0 = time.perf_counter()
                p = subprocess.run(cmd, capture_output=True, env=env, timeout=a.timeout)
                wall = (time.perf_counter() - t0) * 1000; en = datetime.now(timezone.utc); ru1 = resource.getrusage(resource.RUSAGE_CHILDREN)
                ac1, (fm1, fx1), t1c = ac_online(), freqs(), temp()
                if p.returncode != 0: raise RuntimeError(f"{name} failed (exit {p.returncode}) on {g} {s}: {p.stderr.decode(errors='replace')[:300]}")
                err = p.stderr.decode(errors="replace")
                ex = num(r"OPENMP_EXECUTE_MS ([0-9.eE+-]+)", err)
                if ex == "": ex = num(r"run 1 execute_ms ([0-9.eE+-]+)", err)
                if ex == "": raise RuntimeError(f"{name}: no execute time found in stderr")
                fp = num(r"hook_ms=([0-9.eE+-]+)", err)
                if fp == "": fp = num(r"\bfp_ms=([0-9.eE+-]+)", err)
                row = dict(graph=g, source=s, repeat=rep, config=name, order_in_repeat=oi, start_utc=st.isoformat(), end_utc=en.isoformat(), execute_ms=ex,
                           findpivots_ms=fp, serial_commit_ms=num(r"serial_commit_ms=([0-9.eE+-]+)", err), prepare_ms=num(r"prepare_ms ([0-9.eE+-]+)", err),
                           wall_ms=round(wall, 3), cpu_user_s=round(ru1.ru_utime - ru0.ru_utime, 4), cpu_sys_s=round(ru1.ru_stime - ru0.ru_stime, 4),
                           ac_before=ac0, ac_after=ac1, load1=load, freq_mean_before_mhz=fm0, freq_max_before_mhz=fx0, freq_mean_after_mhz=fm1,
                           freq_max_after_mhz=fx1, temp_before_c=t0c, temp_after_c=t1c, distance_sha256=hashlib.sha256(p.stdout).hexdigest())
                w.writerow(row); f.flush(); rows.append(row); completed += 1
                print(f"[{completed}/{total}] {g} src={s} rep={rep} {name:12s} execute_ms={ex:9.2f} temp={t1c} freq_max={fx1}", flush=True)
                if a.require_ac and ac1 != "1": raise RuntimeError("AC lost during a run")
                if a.cooldown_sec: time.sleep(a.cooldown_sec)
finally:
    f.close(); meta["finished_utc"] = datetime.now(timezone.utc).isoformat(); meta["completed_runs"] = completed; meta["expected_runs"] = total
    json.dump(meta, open(os.path.join(a.out, "metadata.json"), "w"), indent=1)
by = {}
for r in rows: by.setdefault((r["graph"], r["source"], r["repeat"]), set()).add(r["distance_sha256"])
bad = sum(1 for v in by.values() if len(v) != 1)
print(f"\ncompleted {completed}/{total} runs; (graph,source,repeat) groups with differing distance hashes: {bad}")
print(f"analyse with: python3 tools/openmp_session_report.py {a.out}/raw_timings.csv --ref USA-road-t.NY.gr=198.1 --ref r18.gr=381.3 --ref r20.gr=1941.9 --baseline builtin")
sys.exit(1 if bad or completed != total else 0)
