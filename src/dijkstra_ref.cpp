// Independent reference and CPU baseline: binary-heap Dijkstra on a DIMACS .gr file (CSR adjacency).
// usage: dijkstra_ref GRAPH.gr SOURCE > distances.txt   (one distance per line, -1 if unreachable)
// stderr: "DIJKSTRA execute_ms=..." (search only; graph loading excluded)
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <queue>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 3) { std::cerr << "usage: dijkstra_ref GRAPH.gr SOURCE\n"; return 2; }
    std::ifstream in(argv[1]);
    if (!in) { std::cerr << "cannot open " << argv[1] << "\n"; return 2; }
    int n = 0; std::vector<std::tuple<int, int, long long>> edges; std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::istringstream ss(line); char c; ss >> c;
        if (c == 'p') { std::string sp; long long m; ss >> sp >> n >> m; }
        else if (c == 'a') { int u, v; long long w; ss >> u >> v >> w; edges.emplace_back(u - 1, v - 1, w); }
    }
    std::vector<int> off(n + 1, 0), to(edges.size()); std::vector<long long> wt(edges.size());
    for (auto& [u, v, w] : edges) off[u + 1]++;
    for (int i = 0; i < n; i++) off[i + 1] += off[i];
    { std::vector<int> pos(off.begin(), off.end() - 1); for (auto& [u, v, w] : edges) { int p = pos[u]++; to[p] = v; wt[p] = w; } }
    const long long INF = -1;
    auto t0 = std::chrono::steady_clock::now();
    std::vector<long long> d(n, INF);
    using QE = std::pair<long long, int>;
    std::priority_queue<QE, std::vector<QE>, std::greater<QE>> pq;
    int s = std::atoi(argv[2]) - 1; d[s] = 0; pq.push({0, s});
    while (!pq.empty()) {
        auto [du, u] = pq.top(); pq.pop();
        if (du != d[u]) continue;
        for (int i = off[u]; i < off[u + 1]; i++) {
            int v = to[i]; long long nd = du + wt[i];
            if (d[v] == INF || nd < d[v]) { d[v] = nd; pq.push({nd, v}); }
        }
    }
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::string out; char buf[32];
    for (long long x : d) { int len = std::snprintf(buf, sizeof buf, "%lld\n", x); out.append(buf, len); }
    std::fwrite(out.data(), 1, out.size(), stdout);
    std::cerr << "DIJKSTRA execute_ms=" << ms << "\n";
    return 0;
}
