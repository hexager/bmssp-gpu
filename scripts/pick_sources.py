#!/usr/bin/env python3
# Picks sources uniformly at random (fixed seed) among vertices that reach at least 10% of the graph.
# usage: pick_sources.py DIJKSTRA_BIN SEED PER_GRAPH GRAPH.gr [GRAPH.gr ...] > sources.txt
import os, random, subprocess, sys

dij, seed, per = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
print(f"# {per} random sources per graph, seed {seed}, each reaching >= 10% of vertices (DIMACS 1-based ids)")
for path in sys.argv[4:]:
    n = next(int(l.split()[2]) for l in open(path) if l.startswith("p"))
    rng = random.Random(f"{seed}:{os.path.basename(path)}")
    picked, tried = [], 0
    while len(picked) < per:
        s = rng.randint(1, n); tried += 1
        if s in picked: continue
        out = subprocess.run([dij, path, str(s)], capture_output=True, text=True, check=True).stdout.split()
        reach = sum(1 for x in out if x != "-1")
        if reach >= 0.10 * n: picked.append(s)
    name = os.path.basename(path)[:-3]
    print(name, *picked, f"# tried {tried}")
