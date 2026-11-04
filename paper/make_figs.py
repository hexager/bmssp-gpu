#!/usr/bin/env python3
"""Column-width paper figures from a results/bench folder.  usage: make_figs.py BENCH_DIR OUT_DIR"""
import csv, glob, math, os, statistics as st, sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def load_runs(path):
    """runs.csv rows; when a (graph, source, rep, config) was measured more than once (a restarted benchmark
    appends), only the last measurement is kept."""
    last = {}
    for r in csv.DictReader(open(path)): last[(r["graph"], r["source"], r["rep"], r["config"])] = r
    return list(last.values())


D, O = sys.argv[1], sys.argv[2]
os.makedirs(O, exist_ok=True)
plt.rcParams.update({"font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "legend.fontsize": 6,
                     "xtick.labelsize": 6, "ytick.labelsize": 6, "font.family": "serif"})
INK, GRID = "#1f1f1e", "#e4e3dc"
NAME = {"USA-road-d.NY": "NY", "grid500": "Grid500", "rmat16": "R-MAT16", "rmat18": "R-MAT18", "rmat20": "R-MAT20"}
runs = load_runs(os.path.join(D, "runs.csv"))
graphs = list(dict.fromkeys(r["graph"] for r in runs))
T = {(r["graph"], r["source"], r["config"]): float(r["execute_ms"]) for r in runs if r["rep"] == "1"}
med = lambda xs: st.median(xs) if xs else float("nan")
load = lambda p: list(csv.DictReader(open(p)))


def style(ax):
    ax.grid(color=GRID, lw=0.5, zorder=0)
    for s in ("top", "right"): ax.spines[s].set_visible(False)


# Figure: FindPivots share and best per-call bound, per graph (fraction of the sync_cpu run).
share, orc = {}, {}
for g in graphs:
    fs, os_ = [], []
    for p in glob.glob(os.path.join(D, "calls", f"{g}_s*_sync_cpu.csv")):
        s = os.path.basename(p)[len(g) + 2:].split("_", 1)[0]
        c = [float(r["ms"]) for r in load(p)]
        q = [float(r["ms"]) for r in load(p.replace("_sync_cpu", "_gpu_all"))]
        t = T[(g, s, "sync_cpu")]
        fs.append(sum(c) / t); os_.append((t - sum(c) + sum(map(min, c, q))) / t)
    share[g], orc[g] = med(fs), med(os_)
fig, ax = plt.subplots(figsize=(3.4, 1.9), dpi=300)
for j, g in enumerate(graphs):
    ax.bar(j, 1 - share[g], 0.62, color="#2a78d6", zorder=3, label="Other BMSSP work" if j == 0 else None)
    ax.bar(j, share[g], 0.62, bottom=1 - share[g], color="#eb6834", zorder=3, label="FindPivots" if j == 0 else None)
    ax.plot([j - 0.36, j + 0.36], [orc[g]] * 2, color=INK, lw=1.4, zorder=4, label="Best per-call CPU/GPU" if j == 0 else None)
ax.set_xticks(range(len(graphs))); ax.set_xticklabels([NAME.get(g, g) for g in graphs])
ax.set_ylim(0, 1.0); ax.set_ylabel("Fraction of run time")
style(ax); ax.legend(frameon=False, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 1.0), borderaxespad=0.2,
                    columnspacing=0.9, handlelength=1.2)
fig.tight_layout(pad=0.3); fig.savefig(os.path.join(O, "model.pdf"))

# Figure: median time per call against first-round work, CPU synchronous vs GPU total, three graphs.
sel = [g for g in ("USA-road-d.NY", "rmat18", "rmat20") if g in graphs]
fig, axes = plt.subplots(1, len(sel), figsize=(3.4, 1.7), dpi=300, sharey=True)
for ax, g in zip(axes, sel):
    for cfg, label, col in (("sync_cpu", "CPU", "#2a78d6"), ("gpu_all", "GPU", "#1baf7a")):
        b = defaultdict(list)
        for p in glob.glob(os.path.join(D, "calls", f"{g}_s*_{cfg}.csv")):
            rows = load(p); nf = max(int(r["upload_vertices"]) for r in rows)
            for r in rows:
                if cfg == "gpu_all" and int(r["upload_vertices"]) == nf: continue
                w = int(r["work0"])
                if w > 0: b[int(math.log2(w))].append(float(r["ms"]) * 1000)
        ks = sorted(k for k in b if len(b[k]) >= 3)
        ax.plot([2 ** k for k in ks], [med(b[k]) for k in ks], marker="o", ms=2, lw=1.2, color=col, label=label)
    ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_title(NAME[g], loc="left"); style(ax)
    ax.set_xticks([2 ** e for e in range(0, 21, 5) if 2 ** e <= 2 ** max(ks)])
axes[0].set_ylabel("Time per call (µs)"); axes[0].legend(frameon=False, loc="upper left")
fig.supxlabel("First-round edge work $w_0$ (edges)", fontsize=7)
fig.tight_layout(pad=0.3); fig.savefig(os.path.join(O, "per_call.pdf"))
print("wrote", O)
