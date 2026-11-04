#!/usr/bin/env python3
"""Column-width paper figures: time breakdown against Dijkstra, and batching (predicted vs measured).
usage: make_figs2.py BREAKDOWN_DIR MULTI_DIR PREDICTION.md OUT_DIR"""
import csv, os, re, statistics as st, sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

B, M, PRED, O = sys.argv[1:5]
os.makedirs(O, exist_ok=True)
plt.rcParams.update({"font.size": 7, "axes.labelsize": 7, "legend.fontsize": 6, "xtick.labelsize": 6,
                     "ytick.labelsize": 6, "font.family": "serif"})
NAME = {"USA-road-d.NY": "NY", "grid500": "Grid500", "rmat16": "R-MAT16", "rmat18": "R-MAT18", "rmat20": "R-MAT20"}
GRID = "#e4e3dc"


def style(ax, axis="both"):
    ax.grid(axis=axis, color=GRID, lw=0.5, zorder=0)
    for s in ("top", "right"): ax.spines[s].set_visible(False)


# Breakdown: BMSSP time in units of one Dijkstra run (paired per source, median).
rows = list(csv.DictReader(open(os.path.join(B, "runs.csv"))))
graphs = list(dict.fromkeys(r["graph"] for r in rows))
def ratio(g, key):
    return st.median(float(r[key]) / float(r["dijkstra_ms"]) for r in rows if r["graph"] == g)
def rest(r): return float(r["total_ms"]) - float(r["d_ops_ms"]) - float(r["edge_ms"])
fig, ax = plt.subplots(figsize=(3.4, 1.95), dpi=300)
parts = [("d_ops_ms", "D operations", "#eb6834"), ("edge_ms", "Edge work (parallel)", "#2a78d6")]
for j, g in enumerate(graphs):
    left = 0
    for key, label, col in parts:
        v = ratio(g, key); ax.barh(j, v, 0.62, left=left, color=col, zorder=3, label=label if j == 0 else None); left += v
    v = st.median(rest(r) / float(r["dijkstra_ms"]) for r in rows if r["graph"] == g)
    ax.barh(j, v, 0.62, left=left, color="#b8b7b0", zorder=3, label="Other" if j == 0 else None)
ax.axvline(1, color="#1f1f1e", lw=1, ls="--", zorder=4, label="One Dijkstra run")
ax.set_yticks(range(len(graphs))); ax.set_yticklabels([NAME.get(g, g) for g in graphs]); ax.invert_yaxis()
ax.set_xlabel("BMSSP time / Dijkstra time")
style(ax, "x")
ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=2, columnspacing=1.0, handlelength=1.2,
          borderaxespad=0.2)
fig.tight_layout(pad=0.3); fig.savefig(os.path.join(O, "breakdown.pdf"))

# Batching: predicted vs measured speedup of gpu_g0 over cpu_hooked, per graph, for the two multi-search settings.
pred = {}
for line in open(PRED):
    m = re.match(r"\| (\S+) \| (\d+)@(\d+) \| [\d.]+ \| [\d.]+ \| ([\d.]+)x", line)
    if m: pred[(m.group(1), m.group(2))] = float(m.group(4))
W = defaultdict(dict)
for r in csv.DictReader(open(os.path.join(M, "runs.csv"))): W[(r["graph"], r["size"], r["rep"])][r["config"]] = float(r["wall_ms"])
mg = list(dict.fromkeys(k[0] for k in W))
fig, axes = plt.subplots(1, 2, figsize=(3.4, 1.8), dpi=300, sharey=True)
for ax, (b, title) in zip(axes, [("5", "5 searches, 1 core each"), ("8", "8 searches on 2 cores")]):
    xs = range(len(mg))
    meas = [st.median(v["cpu_hooked"] / v["gpu_g0"] for k, v in W.items() if k[0] == g and k[1] == b) for g in mg]
    pr = [pred.get((g, b), float("nan")) for g in mg]
    ax.bar([x - 0.19 for x in xs], pr, 0.36, color="#b8b7b0", zorder=3, label="Predicted")
    ax.bar([x + 0.19 for x in xs], meas, 0.36, color="#1baf7a", zorder=3, label="Measured")
    ax.axhline(1, color="#1f1f1e", lw=0.8, ls="--", zorder=4)
    ax.set_xticks(list(xs)); ax.set_xticklabels([{"USA-road-d.NY": "NY", "grid500": "Grid", "rmat16": "R16", "rmat18": "R18", "rmat20": "R20"}.get(g, g) for g in mg])
    ax.set_title(title, fontsize=7, loc="left"); style(ax, "y")
axes[0].set_ylabel("Speedup over CPU FindPivots"); axes[0].legend(frameon=False, loc="upper left")
fig.tight_layout(pad=0.3); fig.savefig(os.path.join(O, "batching.pdf"))
print("wrote", O)
