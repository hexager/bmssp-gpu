#!/usr/bin/env python3
"""Time breakdown of BMSSP (CPU reference) against Dijkstra, all graphs and sources in sources.txt.

usage: scripts/breakdown.py [--bin bin/breakdown] [--reps 3] [--graphs rmat20,...] [--out results/breakdown]
writes: runs.csv, summary.md, breakdown.png in --out

Parts of one BMSSP run:
  D operations  initialize, insert, pull, erase, batch prepend of the data structure D, timed by replaying the
                recorded operations (on the laptop, replayed/in-run pull and prepend time: median 0.88 to 1.06 per graph)
  edge work     FindPivots + base case + edge scanning of the relax loop (relax loop minus its D operations):
                the parts a GPU could run in parallel
  other         the rest (recursion bookkeeping)
"""
import argparse, csv, os, re, statistics as st, subprocess, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--bin", default="bin/breakdown")
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--graphs", default="")
ap.add_argument("--out", default="results/breakdown")
a = ap.parse_args()
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.makedirs(a.out, exist_ok=True)
sources = {}
for line in open("sources.txt"):
    line = line.split("#")[0].split()
    if line: sources[line[0]] = [int(x) for x in line[1:]]
graphs = a.graphs.split(",") if a.graphs else list(sources)

rows = []
for g in graphs:
    for s in sources[g]:
        p = subprocess.run([a.bin, f"data/{g}.gr", str(s), str(a.reps)], capture_output=True, text=True)
        if p.returncode != 0: sys.exit(f"breakdown failed: {g} {s} rc={p.returncode}\n{p.stderr[-2000:]}")
        line = next(l for l in p.stdout.splitlines() if l.startswith("BREAKDOWN"))
        k = {x: float(y) for x, y in re.findall(r"(\w+)=([-\d.e+]+)", line)}
        k["d_ops_ms"] = sum(k[x] for x in ("replay_init_ms", "replay_insert_pivots_ms", "replay_pull_ms",
                                           "replay_relax_ms", "replay_prepend_ms"))
        k["edge_ms"] = k["fp_ms"] + k["base_ms"] + k["relax_scan_ms"]
        k["rest_ms"] = k["total_ms"] - k["d_ops_ms"] - k["edge_ms"]
        k.update(graph=g, source=s); rows.append(k)
        print(f"{g} s={s} bmssp={k['total_ms']:.1f} dijkstra={k['dijkstra_ms']:.1f} d_ops={k['d_ops_ms']:.1f} "
              f"edge={k['edge_ms']:.1f}", flush=True)

keys = ["graph", "source"] + [x for x in rows[0] if x not in ("graph", "source")]
with open(os.path.join(a.out, "runs.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)

med = lambda g, key: st.median([r[key] for r in rows if r["graph"] == g])
ratio = lambda g, num: st.median([r[num] / r["dijkstra_ms"] for r in rows if r["graph"] == g])
L = ["# BMSSP time breakdown against Dijkstra", "",
     f"Medians over {len(sources[graphs[0]])} sources (ms). Ratios are paired per source, then the median is taken.", "",
     "| Graph | Dijkstra | BMSSP | BMSSP / Dijkstra | D operations | D ops / Dijkstra | Edge work (FindPivots + base + relax scan) | Other | "
     "Max BMSSP speedup if all edge work were free |", "|---|---|---|---|---|---|---|---|---|"]
for g in graphs:
    bound = st.median([r["total_ms"] / (r["total_ms"] - r["edge_ms"]) for r in rows if r["graph"] == g])
    L.append(f"| {g} | {med(g, 'dijkstra_ms'):.1f} | {med(g, 'total_ms'):.1f} | {ratio(g, 'total_ms'):.2f}x | "
             f"{med(g, 'd_ops_ms'):.1f} | {ratio(g, 'd_ops_ms'):.2f}x | {med(g, 'edge_ms'):.1f} | {med(g, 'rest_ms'):.1f} | {bound:.2f}x |")
L += ["", "D operations by kind (replay, ms): initialize / insert pivots / pull / relax-loop erase+insert / batch prepend", ""]
for g in graphs:
    L.append(f"- {g}: " + " / ".join(f"{med(g, x):.1f}" for x in ("replay_init_ms", "replay_insert_pivots_ms",
                                                                   "replay_pull_ms", "replay_relax_ms", "replay_prepend_ms"))
             + f"; in-run timers: pull {med(g, 'pull_ms'):.1f}, prepend {med(g, 'prepend_ms'):.1f}")
open(os.path.join(a.out, "summary.md"), "w").write("\n".join(L) + "\n")
print("\n".join(L))

# Figure: BMSSP time in units of one Dijkstra run, split into D operations, edge work and other.
fig, ax = plt.subplots(figsize=(6.4, 0.55 * len(graphs) + 1.2), dpi=150)
parts = [("d_ops_ms", "D operations (sequential)", "#eb6834"), ("edge_ms", "Edge work (GPU-parallel)", "#2a78d6"),
         ("rest_ms", "Other", "#b8b7b0")]
for j, g in enumerate(graphs):
    left = 0
    for key, label, col in parts:
        v = ratio(g, key)
        ax.barh(j, v, 0.6, left=left, color=col, zorder=3, label=label if j == 0 else None)
        left += v
ax.axvline(1, color="#1f1f1e", lw=1.2, ls="--", zorder=4, label="One Dijkstra run")
ax.set_yticks(range(len(graphs))); ax.set_yticklabels(graphs); ax.invert_yaxis()
ax.set_xlabel("BMSSP time / Dijkstra time (same graph and source)")
ax.grid(axis="x", color="#e4e3dc", zorder=0)
for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
ax.legend(frameon=False, fontsize=7, loc="lower left", bbox_to_anchor=(0, 1.01), ncol=4, borderaxespad=0)
fig.tight_layout(); fig.savefig(os.path.join(a.out, "breakdown.png"))
print(f"wrote {a.out}/runs.csv, summary.md, breakdown.png")
