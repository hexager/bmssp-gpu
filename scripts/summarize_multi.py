#!/usr/bin/env python3
"""Summarizes results/multi (bench_multi.py) and compares with the prediction made beforehand.

usage: scripts/summarize_multi.py [results/multi] [PREDICTION.md]
writes: summary.md and predicted_vs_measured.png in the results folder
Speedups are paired on rep (same graph, scenario and repetition), then the median is taken.
"""
import csv, os, re, statistics as st, sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = sys.argv[1] if len(sys.argv) > 1 else "results/multi"
PRED = sys.argv[2] if len(sys.argv) > 2 else "results/multi_prediction.md"
rows = list(csv.DictReader(open(os.path.join(D, "runs.csv"))))
graphs = list(dict.fromkeys(r["graph"] for r in rows))
scen = list(dict.fromkeys((r["size"], r["cores"]) for r in rows))
W = defaultdict(dict)
for r in rows: W[(r["graph"], r["size"], r["cores"], r["rep"])][r["config"]] = r
ORDER = [c for c in ["dijkstra", "builtin", "cpu_hooked", "gpu_g0", "gpu_g1024", "gpu_g4096", "gpu_a0", "gpu_g0_spin",
                     "gpu_g0_w", "gpu_g0_spin_w"] if any(r["config"] == c for r in rows)]
pred = {}
if os.path.exists(PRED):
    for line in open(PRED):
        m = re.match(r"\| (\S+) \| (\d+)@(\d+) \| [\d.]+ \| [\d.]+ \| ([\d.]+)x \|", line)
        if m: pred[(m.group(1), m.group(2), m.group(3))] = float(m.group(4))
device = next((r["device"] for r in rows if r["device"] not in ("", "none")), "unknown")

def label(b, c): return f"{b} searches, {'all' if c == '0' else c} cores"
def pred_key(g, b, c): return (g, b, "6" if c == "0" else c)   # the prediction uses 6 cores for an unpinned run

L = ["# Multi-source runs", "", f"GPU: {device}. Median wall time (ms) from the start of the first search to the end of the "
     "last. Speedup = cpu_hooked / configuration, paired on rep. Batch = mean GPU calls per batch.", ""]
meas = {}
for b, c in scen:
    L += [f"## {label(b, c)}", "", "| Graph | " + " | ".join(ORDER) + " | Best GPU vs cpu_hooked | Predicted gpu_g0 | Measured gpu_g0 | Batch (gpu_g0) |",
          "|---|" + "---|" * (len(ORDER) + 4)]
    for g in graphs:
        reps = [v for (gg, bb, cc, _), v in W.items() if (gg, bb, cc) == (g, b, c)]
        if not reps: continue
        mw = {k: st.median([float(v[k]["wall_ms"]) for v in reps if k in v]) for k in ORDER if any(k in v for v in reps)}
        sp = {k: st.median([float(v["cpu_hooked"]["wall_ms"]) / float(v[k]["wall_ms"]) for v in reps if k in v])
              for k in ORDER if k.startswith("gpu") and any(k in v for v in reps)}
        best = max(sp, key=sp.get) if sp else None
        bs = st.median([int(v["gpu_g0"]["calls_gpu"]) / max(1, int(v["gpu_g0"]["batches"])) for v in reps if "gpu_g0" in v]) if "gpu_g0" in sp else float("nan")
        meas[(g, b, c)] = sp.get("gpu_g0", float("nan"))
        p = pred.get(pred_key(g, b, c))
        L.append(f"| {g} | " + " | ".join(f"{mw[k]:.0f}" if k in mw else "-" for k in ORDER) +
                 f" | {sp[best]:.3f}x ({best}) | " + (f"{p:.3f}x" if p else "-") + f" | {sp.get('gpu_g0', float('nan')):.3f}x | {bs:.2f} |")
    L.append("")
open(os.path.join(D, "summary.md"), "w").write("\n".join(L) + "\n")
print("\n".join(L))

pts = [(pred[pred_key(g, b, c)], meas[(g, b, c)], g, f"{b}@{c}") for (g, b, c) in meas if pred_key(g, b, c) in pred]
if pts:
    fig, ax = plt.subplots(figsize=(4.4, 4.0), dpi=150)
    cols = {"USA-road-d.NY": "#6b6a64", "grid500": "#b8b7b0", "rmat16": "#2a78d6", "rmat18": "#1baf7a", "rmat20": "#eb6834"}
    marks = {}
    for p, m, g, s in pts:
        mk = marks.setdefault(s, "osD^v"[len(marks) % 5])
        ax.scatter(p, m, color=cols.get(g, "#1f1f1e"), marker=mk, s=28, zorder=3)
    lo = min(min(p, m) for p, m, *_ in pts) * 0.9; hi = max(max(p, m) for p, m, *_ in pts) * 1.1
    ax.plot([lo, hi], [lo, hi], color="#1f1f1e", lw=0.8, ls="--", zorder=2)
    ax.axhline(1, color="#e4e3dc", lw=0.8, zorder=1); ax.axvline(1, color="#e4e3dc", lw=0.8, zorder=1)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("Predicted speedup of gpu_g0 over cpu_hooked"); ax.set_ylabel("Measured speedup")
    for g, col in cols.items():
        if any(x[2] == g for x in pts): ax.scatter([], [], color=col, label=g, s=20)
    for s, mk in marks.items(): ax.scatter([], [], color="#1f1f1e", marker=mk, label=s, s=20)
    ax.legend(frameon=False, fontsize=6, loc="upper left")
    for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(D, "predicted_vs_measured.png"))
print(f"wrote {D}/summary.md" + (", predicted_vs_measured.png" if pts else ""))
