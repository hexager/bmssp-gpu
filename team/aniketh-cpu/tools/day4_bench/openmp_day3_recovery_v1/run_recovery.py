#!/usr/bin/env python3
import csv
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path.home() / "BMSSP_Project"
BENCH = ROOT / "tools/day4_bench"
OUT = BENCH / "openmp_day3_recovery_v1"
PLAN = OUT / "missing_plan.csv"
CONFIG = ROOT / "week2_research_tools/configs_day3_verified.json"
RAW = OUT / "recovery_timings.csv"
CONTROL = OUT / "baseline_controls.csv"

COMMIT = "503232b02c74948c61bdbe084179397bb1dc67f7"

EXPECTED = {
    PLAN: "38b8a939cb41d4733c04cb7647f9d1aaf215d87cf31ecf31a2333c4409e1379b",
    CONFIG: "7bb43d444ad152e3ee916d2e6fa861477eb233e2f0a62a4f315461645e101e4e",
    ROOT / "tools/sp_run_instr":
        "34e52c7c4009b8d257e0e5547d8d4444a7864aeabc437aeb9f07e2a400f5b4d4",
    ROOT / "week2_research_tools/fp_openmp_v3":
        "0fb4f3bb62262a3ce797dcf0c9a68a61e52b22cebe20a028b040adab1b0d33e2",
}

FIELDS = [
    "graph", "source", "repeat", "config", "order_in_repeat",
    "start_utc", "end_utc", "execute_ms", "wall_ms",
    "distance_sha256", "ac_before", "ac_after",
    "temp_before_c", "temp_after_c", "session"
]

def stop(msg):
    raise RuntimeError(msg)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def git(*args):
    return subprocess.check_output(
        ["git", "-C", str(ROOT), *args], text=True
    ).strip()

def ac():
    p = Path("/sys/class/power_supply/AC/online")
    return p.read_text().strip() if p.exists() else ""

def temperature():
    for p in Path("/sys/class/hwmon").glob("hwmon*"):
        name = p / "name"
        sensor = p / "temp1_input"
        if name.exists() and sensor.exists():
            if name.read_text().strip() in ("k10temp", "zenpower", "coretemp"):
                return float(sensor.read_text().strip()) / 1000
    return None

def graph_path(name):
    if name == "USA-road-t.NY.gr":
        return Path.home() / "bmssp/tests/graphs/USA-road-t.NY.gr"
    return BENCH / name

def run_one(item, configs, session):
    name = item["config"]
    config = configs[name]
    graph = graph_path(item["graph"])

    if not graph.is_file():
        stop("graph missing: " + str(graph))

    env = os.environ.copy()

    for key in (
        "FP_MODE", "FP_OMP_MIN_EDGES", "FP_CALL_LOG",
        "FP_MIX_SEED", "BMSSP_FP", "BMSSP_CHECK",
        "BMSSP_DUMP", "BMSSP_K", "BMSSP_T",
        "OMP_NUM_THREADS", "OMP_DYNAMIC", "OMP_PLACES",
        "OMP_PROC_BIND", "OMP_WAIT_POLICY"
    ):
        env.pop(key, None)

    env.update(config.get("env", {}))

    cmd = [
        str(x).replace("{graph}", str(graph))
              .replace("{source}", str(item["source"]))
        for x in config["cmd"]
    ]

    if ac() != "1":
        stop("AC disconnected")

    t_before = temperature()

    if t_before is not None and t_before >= 93:
        stop("temperature safety threshold reached before run")

    start = datetime.now(timezone.utc).isoformat()
    tick = time.perf_counter()

    result = subprocess.run(
        cmd,
        env=env,
        capture_output=True,
        timeout=600
    )

    wall = (time.perf_counter() - tick) * 1000
    end = datetime.now(timezone.utc).isoformat()
    t_after = temperature()

    if result.returncode:
        stop(
            f"{name} failed on {item['graph']} "
            f"source={item['source']} exit={result.returncode}: "
            + result.stderr.decode(errors="replace")[-500:]
        )

    if ac() != "1":
        stop("AC disconnected during run")

    if t_after is not None and t_after >= 93:
        stop("temperature safety threshold reached after run")

    stderr = result.stderr.decode(errors="replace")

    m = re.search(r"OPENMP_EXECUTE_MS\s+([0-9.eE+-]+)", stderr)

    if not m:
        m = re.search(r"run 1 execute_ms\s+([0-9.eE+-]+)", stderr)

    if not m:
        stop("execution timing missing")

    return {
        "graph": item["graph"],
        "source": item["source"],
        "repeat": item["repeat"],
        "config": name,
        "order_in_repeat": item["order_in_repeat"],
        "start_utc": start,
        "end_utc": end,
        "execute_ms": float(m.group(1)),
        "wall_ms": wall,
        "distance_sha256": hashlib.sha256(result.stdout).hexdigest(),
        "ac_before": "1",
        "ac_after": "1",
        "temp_before_c": t_before,
        "temp_after_c": t_after,
        "session": session,
    }

def main():
    print("========== BMSSP RECOVERY ==========", flush=True)

    if git("rev-parse", "HEAD") != COMMIT:
        stop("unexpected Git HEAD")

    if git("rev-parse", "cpu-v1.4^{commit}") != COMMIT:
        stop("release tag changed")

    if git("diff", "--cached", "--name-only"):
        stop("Git staging not empty")

    for path, digest in EXPECTED.items():
        if not path.is_file() or sha(path) != digest:
            stop("input hash mismatch: " + str(path))

    if RAW.exists() or CONTROL.exists():
        stop("recovery output already exists")

    if ac() != "1":
        stop("AC power required")

    plan = list(csv.DictReader(PLAN.open(newline="")))
    config_list = json.loads(CONFIG.read_text())
    configs = {c["name"]: c for c in config_list}

    if len(plan) != 68 or len(configs) != 6:
        stop("unexpected recovery plan")

    session = datetime.now(timezone.utc).strftime(
        "recovery_%Y%m%dT%H%M%SZ"
    )

    # Collect sequential controls before and after the recovery.
    controls = [
        {"graph": "r20.gr", "source": "690940",
         "repeat": 0, "config": "builtin", "order_in_repeat": 0},
        {"graph": "r20.gr", "source": "133736",
         "repeat": 0, "config": "builtin", "order_in_repeat": 0},
    ]

    with CONTROL.open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()

        for item in controls:
            row = run_one(item, configs, session + "_before")
            writer.writerow(row)
            f.flush()

    print("PASS: baseline controls before recovery", flush=True)

    completed = 0

    with RAW.open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()

        for item in plan:
            row = run_one(item, configs, session)
            writer.writerow(row)
            f.flush()

            completed += 1
            print(
                f"[{completed}/68] {item['graph']} "
                f"source={item['source']} repeat={item['repeat']} "
                f"{item['config']} execute_ms={row['execute_ms']:.3f}",
                flush=True
            )

            time.sleep(3)

    with CONTROL.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)

        for item in controls:
            row = run_one(item, configs, session + "_after")
            writer.writerow(row)
            f.flush()

    print("PASS: baseline controls after recovery", flush=True)

    rows = list(csv.DictReader(RAW.open(newline="")))

    if len(rows) != 68:
        stop("incomplete recovery")

    keys = {
        (r["graph"], r["source"], r["repeat"], r["config"])
        for r in rows
    }

    if len(keys) != 68:
        stop("duplicate recovery records")

    print("COMPLETED_RUNS:", len(rows), flush=True)
    print("RECOVERY_SHA256:", sha(RAW), flush=True)
    print("CONTROLS_SHA256:", sha(CONTROL), flush=True)
    print("PASS: recovery experiment completed", flush=True)

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("FAIL:", exc, flush=True)
        sys.exit(1)
