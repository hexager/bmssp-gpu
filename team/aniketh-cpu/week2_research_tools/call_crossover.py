#!/usr/bin/env python3
"""Per-call crossover analysis from FP_CALL_LOG files written by fp_openmp_v3.

  python3 call_crossover.py --log builtin=logs/r20_fallback.csv --log serial_sync=logs/r20_omp1.csv --log par4=logs/r20_omp4.csv
         [--ref builtin] [--min-calls 5]

Calls are binned by floor(log2(work0)), where work0 = edges out of the first-round frontier S. For each bin the script
prints the number of calls, their share of total FindPivots time in the reference log, and the MEDIAN time per call
for every log. For every non-reference log it reports the smallest bin from which it is faster than the reference in
EVERY larger bin that has at least --min-calls calls in both logs (a conservative crossover), the time share above that
bin in the reference log, and the time that would be saved if all those calls ran at the other log's median.
Different configurations follow slightly different call sequences, so this is a comparison of per-call cost by size
class, not of identical calls.
"""
import argparse, csv, math, statistics
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("--log", action="append", required=True, help="LABEL=path")
ap.add_argument("--ref"); ap.add_argument("--min-calls", type=int, default=5)
a = ap.parse_args()
logs = {}
for item in a.log:
    lab, path = item.split("=", 1)
    d = defaultdict(list)
    for r in csv.DictReader(open(path, newline="")):
        w = int(r["work0"]); b = int(math.floor(math.log2(w))) if w > 0 else -1
        d[b].append(float(r["ms"]))
    logs[lab] = d
labels = list(logs); ref = a.ref or labels[0]
bins = sorted({b for d in logs.values() for b in d})
tot_ref = sum(sum(v) for v in logs[ref].values())
print(f"reference: {ref}   (total FindPivots time {tot_ref:.1f} ms in {sum(len(v) for v in logs[ref].values())} calls)")
head = f"{'work0 bin':>14s}{'calls':>7s}{'ref share':>10s}" + "".join(f"{l+' us':>14s}" for l in labels)
print(head)
for b in bins:
    n = len(logs[ref].get(b, []))
    sh = 100 * sum(logs[ref].get(b, [0])) / tot_ref if tot_ref else 0
    name = "0" if b < 0 else f"[{2**b},{2**(b+1)})"
    cells = ""
    for l in labels:
        v = logs[l].get(b, [])
        cells += f"{1000*statistics.median(v):14.2f}" if v else f"{'-':>14s}"
    print(f"{name:>14s}{n:7d}{sh:9.1f}%{cells}")
for l in labels:
    if l == ref: continue
    good = [b for b in bins if len(logs[ref].get(b, [])) >= a.min_calls and len(logs[l].get(b, [])) >= a.min_calls]
    star = None
    for i in range(len(good)):
        if all(statistics.median(logs[l][b]) < statistics.median(logs[ref][b]) for b in good[i:]):
            star = good[i]; break
    if star is None:
        print(f"\n{l}: never faster than {ref} in a sustained range of size classes (bins with >= {a.min_calls} calls in both logs).")
    else:
        above = [b for b in good if b >= star]
        share = 100 * sum(sum(logs[ref][b]) for b in above) / tot_ref
        saved = sum(len(logs[ref][b]) * (statistics.median(logs[ref][b]) - statistics.median(logs[l][b])) for b in above)
        print(f"\n{l}: faster than {ref} from work0 >= {2**star} edges; those calls are {share:.1f}% of {ref}'s FindPivots time; "
              f"estimated saving {saved:.2f} ms of {tot_ref:.1f} ms.")
