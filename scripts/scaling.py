#!/usr/bin/env python3
"""Scaling study: does BMSSP get closer to Dijkstra as the graph grows? CPU only.

usage: scripts/scaling.py [--sources 3] [--reps 3] [--rmat 14,16,18,20,21,22] [--grid 128,256,512,1024,2048]
                          [--out results/scaling] [--keep-graphs]
Graphs are generated with bin/gen_graph (seed 42) into data/scaling/ and deleted after use unless --keep-graphs.
For each graph, bin/breakdown picks K sources (seed 7) that reach at least 10% of the vertices and reports the BMSSP
time breakdown and Dijkstra's time for each. writes: runs.csv, summary.md, scaling.png

Theory: BMSSP is O(m log^(2/3) n) and binary-heap Dijkstra is O(m log n), so their time ratio should fall like
(log n)^(-1/3). We fit ratio = a * (log2 n)^b per family and report where the fitted trend would reach 1.
"""
import argparse, csv, math, os, re, statistics as st, subprocess, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--sources", type=int, default=3)
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--rmat", default="14,16,18,20,21,22")
ap.add_argument("--grid", default="128,256,512,1024,2048")
ap.add_argument("--out", default="results/scaling")
ap.add_argument("--keep-graphs", action="store_true")
a = ap.parse_args()
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.makedirs(a.out, exist_ok=True); os.makedirs("data/scaling", exist_ok=True)

jobs = [("rmat", int(s)) for s in a.rmat.split(",") if s] + [("grid", int(s)) for s in a.grid.split(",") if s]
rows = []
for fam, size in jobs:
    name = f"rmat{size}" if fam == "rmat" else f"grid{size}"
    path = f"data/scaling/{name}.gr"
    if not os.path.exists(path):
        cmd = ["bin/gen_graph", "rmat", str(size), "8", "42", path] if fam == "rmat" else ["bin/gen_graph", "grid", str(size), "42", path]
        subprocess.run(cmd, check=True)
    p = subprocess.run(["bin/breakdown", path, f"pick:{a.sources}:7", str(a.reps)], capture_output=True, text=True)
    if p.returncode != 0: sys.exit(f"breakdown failed on {name} rc={p.returncode}\n{p.stderr[-2000:]}")
    for line in p.stdout.splitlines():
        if not line.startswith("BREAKDOWN"): continue
        k = {x: float(y) for x, y in re.findall(r"(\w+)=([-\d.e+]+)", line)}
        k["d_ops_ms"] = sum(k[x] for x in ("replay_init_ms", "replay_insert_pivots_ms", "replay_pull_ms", "replay_relax_ms", "replay_prepend_ms"))
        k["edge_ms"] = k["fp_ms"] + k["base_ms"] + k["relax_scan_ms"]
        k.update(family=fam, graph=name); rows.append(k)
        print(f"{name} s={int(k['source'])} bmssp={k['total_ms']:.0f} dijkstra={k['dijkstra_ms']:.0f} "
              f"ratio={k['total_ms'] / k['dijkstra_ms']:.2f} d_ops/dijkstra={k['d_ops_ms'] / k['dijkstra_ms']:.2f}", flush=True)
    if not a.keep_graphs: os.remove(path)

keys = ["family", "graph"] + [x for x in rows[0] if x not in ("family", "graph")]
with open(os.path.join(a.out, "runs.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)

graphs = list(dict.fromkeys(r["graph"] for r in rows))
G = {g: [r for r in rows if r["graph"] == g] for g in graphs}
med = lambda g, f: st.median([f(r) for r in G[g]])
L = ["# Scaling study: BMSSP against Dijkstra", "", f"Medians over {a.sources} sources per graph (ms); ratios paired per source.", "",
     "| Graph | n | m | Dijkstra | BMSSP | BMSSP / Dijkstra | D ops / Dijkstra | Max BMSSP speedup if all edge work were free |",
     "|---|---|---|---|---|---|---|---|"]
for g in graphs:
    r0 = G[g][0]
    L.append(f"| {g} | {int(r0['n'])} | {int(r0['m'])} | {med(g, lambda r: r['dijkstra_ms']):.1f} | {med(g, lambda r: r['total_ms']):.1f} | "
             f"{med(g, lambda r: r['total_ms'] / r['dijkstra_ms']):.2f}x | {med(g, lambda r: r['d_ops_ms'] / r['dijkstra_ms']):.2f}x | "
             f"{med(g, lambda r: r['total_ms'] / (r['total_ms'] - r['edge_ms'])):.2f}x |")
fits = {}
L += ["", "Fit of BMSSP / Dijkstra = a * (log2 n)^b over each family (least squares in log space). Theory predicts b = -1/3.", "",
      "| Family | a | b | Fitted ratio at n = 2^30 | log2 n where the fitted ratio reaches 1 |", "|---|---|---|---|---|"]
for fam in ("rmat", "grid"):
    pts = [(math.log2(r["n"]), r["total_ms"] / r["dijkstra_ms"]) for r in rows if r["family"] == fam]
    if len({x for x, _ in pts}) < 3: continue
    X = np.log([x for x, _ in pts]); Y = np.log([y for _, y in pts])
    b, la = np.polyfit(X, Y, 1); A = math.exp(la); fits[fam] = (A, b)
    cross = (1 / A) ** (1 / b) if b < 0 else float("inf")
    L.append(f"| {fam} | {A:.2f} | {b:.3f} | {A * 30 ** b:.2f}x | " + (f"{cross:.3g}" if math.isfinite(cross) else "never (ratio does not fall)") + " |")
open(os.path.join(a.out, "summary.md"), "w").write("\n".join(L) + "\n")
print("\n".join(L))

fig, ax = plt.subplots(figsize=(6.0, 3.4), dpi=150)
style = {"rmat": ("#eb6834", "o", "R-MAT"), "grid": ("#2a78d6", "s", "Grid")}
for fam, (col, mk, lab) in style.items():
    gs = [g for g in graphs if G[g][0]["family"] == fam]
    if not gs: continue
    xs = [math.log2(G[g][0]["n"]) for g in gs]
    ax.plot(xs, [med(g, lambda r: r["total_ms"] / r["dijkstra_ms"]) for g in gs], marker=mk, color=col, lw=2, label=f"{lab}: BMSSP / Dijkstra")
    ax.plot(xs, [med(g, lambda r: r["d_ops_ms"] / r["dijkstra_ms"]) for g in gs], marker=mk, color=col, lw=1.2, ls="--",
            label=f"{lab}: D operations / Dijkstra")
    if fam in fits:
        A, b = fits[fam]; xx = np.linspace(min(xs), max(xs), 50)
        ax.plot(xx, A * xx ** b, color=col, lw=0.8, ls=":")
ax.axhline(1, color="#1f1f1e", lw=1, label="Dijkstra")
ax.set_xlabel("log2 n"); ax.set_ylabel("Time / Dijkstra time"); ax.set_ylim(bottom=0)
ax.grid(color="#e4e3dc", lw=0.6)
for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
ax.legend(frameon=False, fontsize=7, ncol=2)
fig.tight_layout(); fig.savefig(os.path.join(a.out, "scaling.png"))
print(f"wrote {a.out}/runs.csv, summary.md, scaling.png")
