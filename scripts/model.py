#!/usr/bin/env python3
"""Bottleneck model from the per-call logs of results/bench (rep 1 of sync_cpu and gpu_all).

usage: scripts/model.py [results/bench]
writes: model.md, model.png in the same folder

The CPU synchronous FindPivots and the GPU FindPivots have the same semantics, so their call sequences are
identical (checked here) and the calls can be paired one to one. For each graph and source:
  T_sync        execute time of the sync_cpu run (log run)
  F             sum of FindPivots call times in that run
  free_fp       T_sync - F                 lower bound on the run time if FindPivots cost nothing
  oracle        T_sync - F + sum_i min(cpu_i, gpu_i)   best per-call CPU/GPU choice with the measured costs
GPU per-call cost is fitted as alpha + rho * rounds + beta * edges (least squares on all calls of a graph),
CPU as c0 + gamma * edges. Break-even edges are given at the median number of rounds per call.
"""
import csv, glob, os, statistics as st, sys
from collections import defaultdict

import matplotlib
import numpy as np
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def load_runs(path):
    """runs.csv rows; when a (graph, source, rep, config) was measured more than once (a restarted benchmark
    appends), only the last measurement is kept."""
    last = {}
    for r in csv.DictReader(open(path)): last[(r["graph"], r["source"], r["rep"], r["config"])] = r
    return list(last.values())


D = sys.argv[1] if len(sys.argv) > 1 else "results/bench"
INK, GRID = "#1f1f1e", "#e4e3dc"
runs = load_runs(os.path.join(D, "runs.csv"))
exec_ms = {(r["graph"], r["source"], r["rep"], r["config"]): float(r["execute_ms"]) for r in runs}
graphs = list(dict.fromkeys(r["graph"] for r in runs))


def load(p): return list(csv.DictReader(open(p))) if os.path.exists(p) else None


def fit(xs, ys):  # least squares y = a + b x
    n = len(xs); mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx if sxx else 0.0
    return my - b * mx, b


res = defaultdict(list); calls_all = defaultdict(list)
for g in graphs:
    for p in sorted(glob.glob(os.path.join(D, "calls", f"{g}_s*_sync_cpu.csv"))):
        s = os.path.basename(p)[len(g) + 2:].split("_", 1)[0]
        cpu, gpu = load(p), load(p.replace("_sync_cpu.csv", "_gpu_all.csv"))
        if not gpu: continue
        if [(r["work0"], r["S"]) for r in cpu] != [(r["work0"], r["S"]) for r in gpu]:
            sys.exit(f"call sequences differ for {g} s={s}: pairing is not valid")
        T = exec_ms.get((g, s, "1", "sync_cpu"))
        if T is None: continue
        c = [float(r["ms"]) for r in cpu]; q = [float(r["ms"]) for r in gpu]
        F = sum(c)
        gpu_wins = [i for i in range(len(c)) if q[i] < c[i]]
        res[g].append(dict(source=s, T=T, F=F, free=T - F, oracle=T - F + sum(min(a, b) for a, b in zip(c, q)),
                           calls=len(c), wins=len(gpu_wins), saved=sum(c[i] - q[i] for i in gpu_wins),
                           gpu_all=exec_ms.get((g, s, "1", "gpu_all")), builtin=exec_ms.get((g, s, "1", "builtin"))))
        n_full = max(int(r["upload_vertices"]) for r in gpu)
        for rc, rg in zip(cpu, gpu):
            if int(rg["upload_vertices"]) == n_full: continue   # the first call's full state upload is one-time cost
            calls_all[g].append((rc, rg))

med = lambda xs: st.median(xs) if xs else float("nan")


def lsq(y, *xs):
    X = np.column_stack([np.ones(len(y)), *xs]); c, *_ = np.linalg.lstsq(X, y, rcond=None)
    return c, 1 - np.sum((X @ c - y) ** 2) / np.sum((y - y.mean()) ** 2)


def phase_fit(pairs):
    f = lambda k: np.array([float(rg[k]) for _, rg in pairs])
    up, r_up = lsq(f("upload_ms"), f("upload_vertices"))
    rd, r_rd = lsq(f("rounds_ms"), f("rounds"), f("edges"))
    dn, r_dn = lsq(f("download_ms"), f("download_vertices"))
    cp, r_cp = lsq(np.array([float(rc["ms"]) for rc, _ in pairs]), f("edges"))
    return dict(up=up, rd=rd, dn=dn, cp=cp, r2=(r_up, r_rd, r_dn, r_cp))


def predict_gpu(m, rg):
    return (m["up"][0] + m["up"][1] * float(rg["upload_vertices"]) + m["rd"][0] + m["rd"][1] * float(rg["rounds"])
            + m["rd"][2] * float(rg["edges"]) + m["dn"][0] + m["dn"][1] * float(rg["download_vertices"]))


L = ["# Bottleneck model", "", "Per (graph, source): rep-1 runs of sync_cpu and gpu_all, calls paired one to one. "
     "Values are medians over sources (ms).", "",
     "| Graph | sync_cpu run | FindPivots share | Bound: FindPivots free | Bound: best per-call choice | GPU wins (calls) | "
     "Max speedup vs sync_cpu (free / oracle) | builtin run |", "|---|---|---|---|---|---|---|---|"]
for g in graphs:
    R = res.get(g)
    if not R: continue
    L.append(f"| {g} | {med([r['T'] for r in R]):.1f} | {med([r['F'] / r['T'] for r in R]) * 100:.1f}% | "
             f"{med([r['free'] for r in R]):.1f} | {med([r['oracle'] for r in R]):.1f} | "
             f"{med([r['wins'] for r in R]):.0f} of {med([r['calls'] for r in R]):.0f} | "
             f"{med([r['T'] / r['free'] for r in R]):.3f}x / {med([r['T'] / r['oracle'] for r in R]):.3f}x | "
             f"{med([r['builtin'] for r in R if r['builtin']]):.1f} |")
L += ["", "Per-call GPU cost by phase, fitted over all calls of a graph except the first (ms = a + b x):", "",
      "| Graph | Upload: us + ns/vertex | Rounds: us + us/round + ns/edge | Download: us + ns/vertex | Fixed floor (2 rounds), us | "
      "CPU sync: us + ns/edge | Break-even edges | Calls above | R^2 up/rounds/down/cpu |", "|---|---|---|---|---|---|---|---|---|"]
for g in graphs:
    P = calls_all.get(g)
    if not P: continue
    m = phase_fit(P)
    floor = (m["up"][0] + m["rd"][0] + 2 * m["rd"][1] + m["dn"][0]) * 1000
    um, dm = med([float(rg["upload_vertices"]) for _, rg in P]), med([float(rg["download_vertices"]) for _, rg in P])
    per_edge_gap = m["cp"][1] - m["rd"][2]
    be = ((floor / 1000 + m["up"][1] * um + m["dn"][1] * dm - m["cp"][0]) / per_edge_gap) if per_edge_gap > 0 else float("inf")
    above = sum(1 for _, rg in P if float(rg["edges"]) >= be)
    L.append(f"| {g} | {m['up'][0]*1e3:.0f} + {m['up'][1]*1e6:.1f} | {m['rd'][0]*1e3:.0f} + {m['rd'][1]*1e3:.0f} + {m['rd'][2]*1e6:.2f} | "
             f"{m['dn'][0]*1e3:.0f} + {m['dn'][1]*1e6:.1f} | {floor:.0f} | {m['cp'][0]*1e3:.0f} + {m['cp'][1]*1e6:.2f} | "
             f"{be:.0f} | {above} of {len(P)} | " + " / ".join(f"{r:.2f}" for r in m["r2"]) + " |")
L += ["", "Model check, leave one graph out: phases fitted on the other graphs, then the total GPU FindPivots time of each "
      "gpu_all run of this graph predicted from its logged call sizes:", "",
      "| Graph | Measured GPU FindPivots ms (median over sources) | Predicted | Error |", "|---|---|---|---|"]
for g in graphs:
    if not calls_all.get(g): continue
    others = [pr for h in graphs if h != g for pr in calls_all.get(h, [])]
    if not others: continue
    m = phase_fit(others)
    meas, pred = [], []
    for p in sorted(glob.glob(os.path.join(D, "calls", f"{g}_s*_gpu_all.csv"))):
        rows = load(p); nf = max(int(r["upload_vertices"]) for r in rows)
        rows = [r for r in rows if int(r["upload_vertices"]) != nf]
        meas.append(sum(float(r["ms"]) for r in rows)); pred.append(sum(predict_gpu(m, r) for r in rows))
    mm, pp = med(meas), med(pred)
    L.append(f"| {g} | {mm:.1f} | {pp:.1f} | {(pp - mm) / mm * 100:+.1f}% |")
open(os.path.join(D, "model.md"), "w").write("\n".join(L) + "\n")
print("\n".join(L))

# Figure: per graph, sync_cpu run split into non-FindPivots and FindPivots time, with the two bounds marked.
gs = [g for g in graphs if res.get(g)]
fig, ax = plt.subplots(figsize=(max(6.5, 1.4 * len(gs) + 2), 4.0), dpi=150)
for j, g in enumerate(gs):
    R = res[g]; T = med([r["T"] for r in R]); fr = med([r["F"] / r["T"] for r in R])
    orc = med([r["oracle"] / r["T"] for r in R])
    ax.bar(j, 1 - fr, 0.6, color="#2a78d6", zorder=3, label="Other BMSSP work (CPU)" if j == 0 else None)
    ax.bar(j, fr, 0.6, bottom=1 - fr, color="#eb6834", zorder=3, label="FindPivots (CPU synchronous)" if j == 0 else None)
    ax.plot([j - 0.38, j + 0.38], [orc, orc], color=INK, lw=2, zorder=4, label="Best per-call CPU/GPU choice" if j == 0 else None)
    ax.text(j, 1.02, f"{T:.0f} ms", ha="center", fontsize=7, color=INK)
ax.set_xticks(range(len(gs))); ax.set_xticklabels(gs); ax.set_ylim(0, 1.12)
ax.set_ylabel("Fraction of the sync_cpu run time")
ax.set_title("Where BMSSP spends its time, and the most a GPU FindPivots can remove", loc="left", fontsize=10)
ax.grid(axis="y", color=GRID, zorder=0)
for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
ax.legend(frameon=False, fontsize=7, loc="lower right")
fig.tight_layout(); fig.savefig(os.path.join(D, "model.png"))
print(f"wrote {D}/model.md, model.png")
