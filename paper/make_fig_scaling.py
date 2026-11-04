#!/usr/bin/env python3
"""Column-width scaling figure. usage: make_fig_scaling.py SCALING_DIR OUT_DIR"""
import csv, math, os, statistics as st, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
D, O = sys.argv[1], sys.argv[2]
plt.rcParams.update({"font.size": 7, "axes.labelsize": 7, "legend.fontsize": 6, "xtick.labelsize": 6, "ytick.labelsize": 6, "font.family": "serif"})
rows = list(csv.DictReader(open(os.path.join(D, "runs.csv"))))
graphs = list(dict.fromkeys(r["graph"] for r in rows))
fig, ax = plt.subplots(figsize=(3.4, 1.9), dpi=300)
for fam, col, mk, lab in (("rmat", "#eb6834", "o", "R-MAT"), ("grid", "#2a78d6", "s", "Grid")):
    gs = [g for g in graphs if next(r for r in rows if r["graph"] == g)["family"] == fam]
    xs = [math.log2(float(next(r for r in rows if r["graph"] == g)["n"])) for g in gs]
    tot = [st.median(float(r["total_ms"]) / float(r["dijkstra_ms"]) for r in rows if r["graph"] == g) for g in gs]
    dop = [st.median(float(r["d_ops_ms"]) / float(r["dijkstra_ms"]) for r in rows if r["graph"] == g) for g in gs]
    ax.plot(xs, tot, marker=mk, ms=3, lw=1.4, color=col, label=f"{lab}: BMSSP")
    ax.plot(xs, dop, marker=mk, ms=3, lw=1.0, ls="--", color=col, label=f"{lab}: $D$ operations")
ax.axhline(1, color="#1f1f1e", lw=0.8, label="Dijkstra")
ax.set_xlabel("$\\log_2 n$"); ax.set_ylabel("Time / Dijkstra time"); ax.set_ylim(0, 7)
ax.grid(color="#e4e3dc", lw=0.5)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), columnspacing=0.8, handlelength=1.6, borderaxespad=0.2)
fig.tight_layout(pad=0.3); fig.savefig(os.path.join(O, "scaling.pdf")); print("wrote", O)
