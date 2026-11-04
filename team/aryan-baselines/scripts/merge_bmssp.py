#!/usr/bin/env python3
# Usage: python3 scripts/merge_bmssp.py aniketh_rows.csv
# Input columns (as in Aniketh h1_strict_counts): graph,source,bmssp_edges_examined,bmssp_strict_improvements[,bmssp_leq_successes]
import csv, sys, statistics as st, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
NAME={"rmat16.gr":"RMAT16","grid500.gr":"GRID500","USA-road-d.NY.gr":"NY"}
rows=list(csv.DictReader(open(sys.argv[1])))
b={}
for r in rows:
    g=NAME.get(r["graph"].split("/")[-1])
    if g: b.setdefault(g,[]).append((int(r["bmssp_strict_improvements"]),int(r["bmssp_edges_examined"])))
t=list(csv.DictReader(open("results/week2/h1/h1_table.csv")))
for r in t:
    x=b.get(r["dataset"])
    if x:
        r["bmssp_relax"]=int(st.median(a for a,_ in x)); r["bmssp_edges"]=int(st.median(c for _,c in x))
        r["bmssp_over_dijkstra"]=round(r["bmssp_relax"]/int(r["dijkstra_relax"]),3)
        r["bmssp_over_delta"]=round(r["bmssp_relax"]/int(r["delta_relax"]),3)
    else:
        print("no BMSSP rows for",r["dataset"])
keys=[]; [keys.append(k) for r in t for k in r if k not in keys]
w=csv.DictWriter(open("results/week2/h1/h1_table_full.csv","w",newline=""),fieldnames=keys,restval="PENDING"); w.writeheader(); w.writerows(t)
n=range(len(t)); wd=0.27
for i,(k,l) in enumerate([("dijkstra_relax","Dijkstra"),("delta_relax","CPU delta-stepping"),("bmssp_relax","BMSSP")]):
    plt.bar([j+(i-1)*wd for j in n],[float(r[k]) if str(r.get(k,"")).replace(".","").isdigit() else 0 for r in t],wd,label=l)
plt.xticks(list(n),[r["dataset"] for r in t]); plt.ylabel("strict relaxations (median over sources)"); plt.legend(); plt.tight_layout()
plt.savefig("results/week2/plots/h1_all.png",dpi=200)
print(open("results/week2/h1/h1_table_full.csv").read())
