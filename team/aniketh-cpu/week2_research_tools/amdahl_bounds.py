#!/usr/bin/env python3
"""Upper bounds on what offloading/parallelising parts of BMSSP could achieve, from MEASURED component shares.

  python3 amdahl_bounds.py --components tools/day4_bench/component_profiles_v1/component_breakdown.csv \
                           --stats tools/day4_bench/cpu_stats_recomputed_v1.csv --out tools/day4_bench/amdahl_bounds_v1.csv

--components : component_breakdown.csv (columns graph, fp_pct, pull_pct, rec_relax_pct, ...; one row per profiled source)
--stats      : output of recompute_cpu_stats.py (columns graph, median_ratio_execute = BMSSP/Dijkstra, execute only)
For every graph the script takes the MEDIAN share of each component and reports, for three scenarios
(A: FindPivots only, B: FindPivots + relaxation loop, C: B + D.pull), the Amdahl ceiling 1/(1-f) for an accelerator of
INFINITE speed and ZERO transfer cost, the BMSSP/Dijkstra ratio that would remain, and the fraction f* of runtime that
would have to disappear for BMSSP to merely tie Dijkstra (f* = 1 - 1/ratio).
These are ceilings under stated assumptions (shares from instrumented runs of 3 sources; other components unchanged;
execute-only boundary), not predictions. Real accelerators have finite speed and transfer/launch costs.
"""
import argparse, csv, statistics
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("--components", required=True); ap.add_argument("--stats", required=True); ap.add_argument("--out", default="amdahl_bounds.csv")
a = ap.parse_args()
sh = defaultdict(lambda: defaultdict(list))
for r in csv.DictReader(open(a.components, newline="")):
    for k in ("fp_pct", "pull_pct", "rec_relax_pct"): sh[r["graph"]][k].append(float(r[k]))
ratio = {r["graph"]: float(r["median_ratio_execute"]) for r in csv.DictReader(open(a.stats, newline=""))}
rows = []
print(f"{'graph':18s}{'BMSSP/Dij':>10s}{'f* to tie':>10s} | {'A: FP only':^22s} | {'B: FP + relax loop':^22s} | {'C: B + pull':^22s}")
print(f"{'':18s}{'':>10s}{'':>10s} | {'f':>6s}{'ceiling':>8s}{'left':>8s} | {'f':>6s}{'ceiling':>8s}{'left':>8s} | {'f':>6s}{'ceiling':>8s}{'left':>8s}")
for g in sh:
    m = {k: statistics.median(v) / 100 for k, v in sh[g].items()}
    R = ratio[g]; fstar = 1 - 1 / R
    out = dict(graph=g, bmssp_over_dijkstra=round(R, 3), f_needed_to_tie=round(fstar, 3))
    cells = []
    for name, f in (("A", m["fp_pct"]), ("B", m["fp_pct"] + m["rec_relax_pct"]), ("C", m["fp_pct"] + m["rec_relax_pct"] + m["pull_pct"])):
        ceil = 1 / (1 - f); left = R / ceil
        out[f"{name}_f"] = round(f, 4); out[f"{name}_ceiling"] = round(ceil, 3); out[f"{name}_ratio_left"] = round(left, 3)
        cells.append(f"{f:6.3f}{ceil:7.2f}x{left:7.2f}x")
    print(f"{g:18s}{R:9.2f}x{fstar:10.3f} | " + " | ".join(cells))
    rows.append(out)
with open(a.out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(f"\nwrote {a.out}. 'left' = BMSSP/Dijkstra ratio that remains after an infinitely fast accelerator removes the offloaded part; "
      "values above 1 mean BMSSP would still be slower than Dijkstra.")
