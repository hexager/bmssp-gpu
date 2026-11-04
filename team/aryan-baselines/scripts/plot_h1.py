import csv, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
r=list(csv.DictReader(open("results/week2/h1/h1_table.csv")))
x=range(len(r)); w=0.38
plt.bar([i-w/2 for i in x],[int(a["dijkstra_relax"]) for a in r],w,label="Dijkstra")
plt.bar([i+w/2 for i in x],[int(a["delta_relax"]) for a in r],w,label="CPU delta-stepping")
plt.xticks(list(x),[a["dataset"] for a in r]); plt.ylabel("successful relaxations (median over 5 sources)")
plt.title("H1 baselines (BMSSP bars pending)"); plt.legend(); plt.tight_layout()
plt.savefig("results/week2/plots/h1_baseline_relaxations.png",dpi=200)
