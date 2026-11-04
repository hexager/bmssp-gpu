#!/usr/bin/env python3
"""Summarizes results/bench: an end-to-end table, a speedup figure and a per-call figure.

usage: scripts/summarize.py [results/bench]
writes: summary.md, speedup.png, baselines.png, nearfar_delta.png, per_call.png in the same folder
Speedup = builtin execute time / configuration execute time, paired on (source, rep), median over pairs.
"""
import csv, glob, math, os, statistics as st, sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = sys.argv[1] if len(sys.argv) > 1 else "results/bench"
ORDER = ["builtin", "cpu_hooked", "sync_cpu", "gpu_all", "gpu_t1024", "gpu_t4096", "gpu_t16384", "gpu_t32768",
         "gpu_t65536", "gpu_t4096_full"]
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dc"

def load_runs(path):
    """runs.csv rows; when a (graph, source, rep, config) was measured more than once (a restarted benchmark
    appends), only the last measurement is kept."""
    last = {}
    for r in csv.DictReader(open(path)): last[(r["graph"], r["source"], r["rep"], r["config"])] = r
    return list(last.values())


rows = load_runs(os.path.join(D, "runs.csv"))
graphs = list(dict.fromkeys(r["graph"] for r in rows))
by = defaultdict(list)
for r in rows: by[(r["graph"], r["config"])].append(r)
base = {(r["graph"], r["source"], r["rep"]): float(r["execute_ms"]) for r in rows if r["config"] == "builtin"}
device = next((r["device"] for r in rows if r["device"] not in ("", "none") and "emulation" not in r["device"]),
              next((r["device"] for r in rows if r["device"] not in ("", "none")), "unknown"))


def med(xs): return st.median(xs) if xs else float("nan")


lines = [f"# Hybrid BMSSP benchmark\n", f"GPU: {device}. Runs: {len(rows)}. Times are execute-only (ms), medians.\n",
         "Speedup is builtin / configuration, paired on (source, rep). GPU columns are summed over the run.\n",
         "| Graph | Config | Execute ms | Speedup | GPU calls | Upload ms | Rounds ms | (kernel / xfer / host) | Download ms | Upload vertices |",
         "|---|---|---|---|---|---|---|---|---|---|"]
speed = {}
for g in graphs:
    for c in ORDER:
        rs = by.get((g, c), [])
        if not rs: continue
        sp = [base[(g, r["source"], r["rep"])] / float(r["execute_ms"]) for r in rs if (g, r["source"], r["rep"]) in base]
        speed[(g, c)] = (med(sp), min(sp) if sp else float("nan"), max(sp) if sp else float("nan"))
        f = lambda k: med([float(r[k]) for r in rs if r[k] != ""])
        lines.append(f"| {g} | {c} | {f('execute_ms'):.1f} | {speed[(g, c)][0]:.3f}x | {f('calls_gpu'):.0f} | "
                     f"{f('gpu_upload_ms'):.1f} | {f('gpu_rounds_ms'):.1f} | {f('gpu_kernel_ms'):.1f} / {f('gpu_xfer_ms'):.1f} / "
                     f"{f('gpu_host_ms'):.1f} | {f('gpu_download_ms'):.1f} | {f('upload_vertices'):.0f} |")

# Baselines: CPU Dijkstra, CPU BMSSP, best hybrid configuration, GPU Near-Far at its best delta (per graph, by median).
mexec = {k: med([float(r["execute_ms"]) for r in v]) for k, v in by.items()}
NF = sorted({c for (_, c) in by if c.startswith("nearfar_")}, key=lambda c: float(c.split("_x")[1]) if "_x" in c else 0)
HYB = [c for c in ORDER if c.startswith("gpu_")]
best = {}
lines += ["", "## Baselines", "",
          "Median execute ms over all (source, rep). Best hybrid and best Near-Far delta are chosen per graph by median, "
          "which favours those columns. Ratio = Dijkstra / column (above 1 = faster than Dijkstra).", "",
          "| Graph | Dijkstra (CPU) | BMSSP (CPU) | Best hybrid | Near-Far (GPU), best delta | BMSSP / Dijkstra | Near-Far / Dijkstra |",
          "|---|---|---|---|---|---|---|"]
for g in graphs:
    dj, bm = mexec.get((g, "dijkstra"), float("nan")), mexec.get((g, "builtin"), float("nan"))
    hy = min(((mexec[(g, c)], c) for c in HYB if (g, c) in mexec), default=(float("nan"), "-"))
    nf = min(((mexec[(g, c)], c) for c in NF if (g, c) in mexec), default=(float("nan"), "-"))
    best[g] = dict(dijkstra=dj, bmssp=bm, hybrid=hy, nearfar=nf)
    lines.append(f"| {g} | {dj:.1f} | {bm:.1f} | {hy[0]:.1f} ({hy[1]}) | {nf[0]:.1f} ({nf[1].replace('nearfar_', '')}) | "
                 f"{dj / bm:.3f} | {dj / nf[0]:.3f} |")
if NF:
    lines += ["", "Near-Far delta sweep (median execute ms; delta given as a multiple of the average edge weight):", "",
              "| Graph | " + " | ".join(c.replace("nearfar_", "") for c in NF) + " |", "|---|" + "---|" * len(NF)]
    for g in graphs:
        lines.append(f"| {g} | " + " | ".join(f"{mexec.get((g, c), float('nan')):.1f}" for c in NF) + " |")
open(os.path.join(D, "summary.md"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))

# Figure 1: end-to-end speedup per graph and configuration (median, whisker = min..max over pairs).
cfgs = [c for c in ORDER if c != "builtin" and any((g, c) in speed for g in graphs)]
colors = ["#6b6a64", "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
fig, ax = plt.subplots(figsize=(max(7, 1.6 * len(graphs) + 2), 4.2), dpi=150)
wdt = 0.8 / max(1, len(cfgs))
for i, c in enumerate(cfgs):
    xs, ys, lo, hi = [], [], [], []
    for j, g in enumerate(graphs):
        if (g, c) not in speed: continue
        m, a, b = speed[(g, c)]
        xs.append(j - 0.4 + (i + 0.5) * wdt); ys.append(m); lo.append(m - a); hi.append(b - m)
    ax.bar(xs, ys, wdt * 0.9, color=colors[i % len(colors)], label=c, zorder=3)
    ax.errorbar(xs, ys, yerr=[lo, hi], fmt="none", ecolor=INK, elinewidth=0.8, capsize=2, zorder=4)
ax.axhline(1, color=INK, lw=1, ls="--", zorder=5, label="CPU built-in (1x)")
ax.set_xticks(range(len(graphs))); ax.set_xticklabels(graphs)
ax.set_ylabel("Speedup over CPU built-in (execute time)")
ax.set_title(f"End-to-end BMSSP speedup by FindPivots configuration ({device})", loc="left", fontsize=10)
ax.grid(axis="y", color=GRID, zorder=0)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.legend(frameon=False, fontsize=7, ncol=2)
fig.tight_layout(); fig.savefig(os.path.join(D, "speedup.png"))

# Figure 2: absolute execute time of the four approaches per graph (log scale).
if any(not math.isnan(best[g]["dijkstra"]) for g in graphs):
    fig, ax = plt.subplots(figsize=(max(7, 1.6 * len(graphs) + 2), 4.2), dpi=150)
    ser = [("dijkstra", "Dijkstra (CPU)", "#6b6a64"), ("bmssp", "BMSSP (CPU)", "#2a78d6"),
           ("hybrid", "BMSSP + GPU FindPivots (best threshold)", "#1baf7a"), ("nearfar", "Near-Far (GPU, best delta)", "#eb6834")]
    wd = 0.8 / len(ser)
    for i, (k, label, col) in enumerate(ser):
        vals = [best[g][k][0] if isinstance(best[g][k], tuple) else best[g][k] for g in graphs]
        ax.bar([j - 0.4 + (i + 0.5) * wd for j in range(len(graphs))], vals, wd * 0.9, color=col, label=label, zorder=3)
    ax.set_yscale("log"); ax.set_xticks(range(len(graphs))); ax.set_xticklabels(graphs)
    ax.set_ylabel("Execute time (ms, log scale, median)")
    ax.set_title(f"SSSP execute time by approach ({device})", loc="left", fontsize=10)
    ax.grid(axis="y", color=GRID, zorder=0)
    for sp_ in ("top", "right"): ax.spines[sp_].set_visible(False)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(D, "baselines.png"))

# Figure 3: Near-Far execute time against delta, normalised to the best delta of each graph.
if NF:
    fig, ax = plt.subplots(figsize=(6.4, 3.8), dpi=150)
    xs = [float(c.split("_x")[1]) for c in NF]
    for j, g in enumerate(graphs):
        ys = [mexec.get((g, c), float("nan")) for c in NF]
        b = min(y for y in ys if not math.isnan(y))
        ax.plot(xs, [y / b for y in ys], marker="o", ms=4, lw=2, color=colors[j % len(colors)], label=g)
    ax.set_xscale("log", base=2); ax.set_xlabel("Delta (multiple of the average edge weight)")
    ax.set_ylabel("Execute time / best delta")
    ax.set_title("Near-Far sensitivity to delta", loc="left", fontsize=10)
    ax.grid(color=GRID, lw=0.6)
    for sp_ in ("top", "right"): ax.spines[sp_].set_visible(False)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(D, "nearfar_delta.png"))

# Figure 4: time per FindPivots call against its first-round edge work (log2 bins), CPU vs GPU, for each graph.
calls = defaultdict(lambda: defaultdict(list))
for p in glob.glob(os.path.join(D, "calls", "*.csv")):
    name = os.path.basename(p)[:-4]
    g, rest = name.split("_s", 1)
    c = rest.split("_", 1)[1]
    for r in csv.DictReader(open(p)): calls[g][c].append(r)
gs = [g for g in graphs if calls.get(g)]
if gs:
    fig, axes = plt.subplots(1, len(gs), figsize=(3.6 * len(gs), 3.6), dpi=150, squeeze=False)
    series = [("cpu_hooked", "CPU built-in", "#6b6a64", "ms"), ("sync_cpu", "CPU synchronous", "#2a78d6", "ms"),
              ("gpu_all", "GPU total", "#1baf7a", "ms"), ("gpu_all", "GPU rounds only", "#eb6834", "rounds_ms")]
    for ax, g in zip(axes[0], gs):
        for c, label, col, key in series:
            b = defaultdict(list)
            for r in calls[g].get(c, []):
                wk = int(r["work0"])
                if wk > 0: b[int(math.log2(wk))].append(float(r[key]) * 1000)
            ks = sorted(k for k in b if len(b[k]) >= 3)
            if ks: ax.plot([2 ** k for k in ks], [med(b[k]) for k in ks], marker="o", ms=4, lw=2, color=col, label=label)
        ax.set_xscale("log", base=2); ax.set_yscale("log")
        ax.set_title(g, loc="left", fontsize=10); ax.set_xlabel("First-round edge work (edges, log2 bin)")
        ax.grid(color=GRID, lw=0.6)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
    axes[0][0].set_ylabel("Median time per call (µs)")
    axes[0][-1].legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(D, "per_call.png"))
print(f"wrote {D}/summary.md and figures")
