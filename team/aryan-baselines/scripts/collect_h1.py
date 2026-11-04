#!/usr/bin/env python3
import csv, glob, os, re, statistics as st
rows=[]
def v(p,t):
    m=re.search(p,t); return m.group(1) if m else ""
for path in sorted(glob.glob("results/week2/baselines/*.txt")):
    m=re.match(r"(.+)_source(\d+)_rep(\d+)_(dijkstra|cpu_delta)\.txt",os.path.basename(path))
    if not m: continue
    t=open(path).read()
    rt=v(r"Runtime\s*:\s*([\d.]+)",t) or v(r"Delta CPU\s*:\s*([\d.]+)",t)
    rows.append(dict(dataset=m[1],source=m[2],rep=m[3],algorithm=m[4],
      edges_examined=v(r"Edges examined\s*:\s*(\d+)",t),
      relaxations=v(r"Successful relaxations\s*:\s*(\d+)",t),
      runtime_ms=rt,correctness=v(r"Correctness\s*:\s*(\w+)",t)))
with open("results/week2/h1/h1_runs.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
# median per dataset/algorithm (counts are deterministic per source; median over sources)
out=[]
for ds in ["RMAT16","GRID500","NY"]:
    r={}
    for a in ["dijkstra","cpu_delta"]:
        x=[q for q in rows if q["dataset"]==ds and q["algorithm"]==a]
        r[a]=(st.median(int(q["relaxations"]) for q in x),st.median(int(q["edges_examined"]) for q in x),st.median(float(q["runtime_ms"]) for q in x),len(x))
    out.append((ds,r))
with open("results/week2/h1/h1_table.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["dataset","dijkstra_relax","delta_relax","delta_over_dijkstra_relax","dijkstra_edges","delta_edges","dijkstra_ms","delta_ms","bmssp_relax"])
    for ds,r in out:
        d,c=r["dijkstra"],r["cpu_delta"]
        w.writerow([ds,int(d[0]),int(c[0]),round(c[0]/d[0],3),int(d[1]),int(c[1]),round(d[2],2),round(c[2],2),"PENDING"])
print(open("results/week2/h1/h1_table.csv").read())
print("mismatch/FAIL count:",sum(1 for q in rows if q["correctness"] not in ("","PASS")))
