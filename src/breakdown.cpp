// Time breakdown of one BMSSP run (CPU reference, built-in FindPivots), next to Dijkstra on the same graph.
//   run 1: normal run, phase timers of the reference implementation (FindPivots, base case, pull, prepend, relax loop)
//   run 2: same run with the D-operation trace on (its time is not used)
//   replay: the trace replayed on fresh D objects, REPEATS times; the median time per operation kind is reported
//   Dijkstra: binary heap on CSR, REPEATS times, median; its distances must equal BMSSP's (exit 3 if not)
// usage: breakdown GRAPH.gr SOURCE [REPEATS=3]    stdout: one BREAKDOWN line per source (key=value, times in ms)
//   SOURCE is a 1-based vertex id, or pick:K:SEED = K random sources (fixed seed) that reach at least 10% of the
//   vertices, chosen with Dijkstra; the graph is loaded once for all of them
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <queue>
#include <random>
#include <cstdio>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include "bmssp_breakdown.hpp"

using T = long long;
using Clock = std::chrono::steady_clock;
static double ms_since(Clock::time_point a) { return std::chrono::duration<double, std::milli>(Clock::now() - a).count(); }
static double median(std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }

static std::vector<T> dijkstra(int n, const std::vector<int>& off, const std::vector<int>& to, const std::vector<T>& wt, int s) {
    std::vector<T> d(n, -1);
    using QE = std::pair<T, int>;
    std::priority_queue<QE, std::vector<QE>, std::greater<QE>> pq;
    d[s] = 0; pq.push({0, s});
    while (!pq.empty()) {
        auto [du, u] = pq.top(); pq.pop();
        if (du != d[u]) continue;
        for (int i = off[u]; i < off[u + 1]; i++) {
            T nd = du + wt[i]; int v = to[i];
            if (d[v] == -1 || nd < d[v]) { d[v] = nd; pq.push({nd, v}); }
        }
    }
    return d;
}

int main(int argc, char** argv) {
    if (argc < 3) { std::cerr << "usage: breakdown GRAPH.gr SOURCE|pick:K:SEED [REPEATS]\n"; return 2; }
    const int reps = argc > 3 ? std::max(1, std::atoi(argv[3])) : 3;
    int n = 0; std::vector<std::tuple<int, int, T>> edges;
    {   // fast DIMACS reader (the scaling graphs have up to 34M arcs)
        FILE* f = std::fopen(argv[1], "rb"); if (!f) { std::cerr << "cannot open " << argv[1] << "\n"; return 2; }
        std::vector<char> buf(1 << 24); std::string carry;
        size_t got;
        auto handle = [&](const char* p, const char* e) {
            if (p == e) return;
            if (*p == 'p') { char sp[8]; long long m; std::sscanf(std::string(p, e).c_str(), "p %7s %d %lld", sp, &n, &m); edges.reserve(m); }
            else if (*p == 'a') {
                char* q; long long u = std::strtoll(p + 1, &q, 10), v = std::strtoll(q, &q, 10), w = std::strtoll(q, &q, 10);
                edges.emplace_back((int)u - 1, (int)v - 1, (T)w);
            }
        };
        while ((got = std::fread(buf.data(), 1, buf.size(), f)) > 0) {
            const char* p = buf.data(); const char* end = p + got;
            while (p < end) {
                const char* nl = (const char*)std::memchr(p, '\n', end - p);
                if (!nl) { carry.append(p, end); break; }
                if (!carry.empty()) { carry.append(p, nl); handle(carry.data(), carry.data() + carry.size()); carry.clear(); }
                else handle(p, nl);
                p = nl + 1;
            }
        }
        if (!carry.empty()) handle(carry.data(), carry.data() + carry.size());
        std::fclose(f);
    }
    std::vector<int> off(n + 1, 0), to(edges.size()); std::vector<T> wt(edges.size());
    for (auto& [u, v, w] : edges) off[u + 1]++;
    for (int i = 0; i < n; i++) off[i + 1] += off[i];
    { std::vector<int> pos(off.begin(), off.end() - 1); for (auto& [u, v, w] : edges) { int p = pos[u]++; to[p] = v; wt[p] = w; } }

    std::vector<int> sources;
    const std::string sarg = argv[2];
    if (sarg.rfind("pick:", 0) == 0) {
        int K = 0; unsigned long long seed = 0;
        if (std::sscanf(sarg.c_str(), "pick:%d:%llu", &K, &seed) != 2 || K < 1) { std::cerr << "bad pick spec\n"; return 2; }
        std::mt19937_64 rng(seed); std::uniform_int_distribution<int> pick(0, n - 1);
        for (int tries = 0; (int)sources.size() < K && tries < 1000; tries++) {
            int s = pick(rng);
            if (std::find(sources.begin(), sources.end(), s) != sources.end()) continue;
            auto d = dijkstra(n, off, to, wt, s);
            long long reach = 0; for (T x : d) reach += x != -1;
            if (reach * 10 >= n) sources.push_back(s);
        }
        if ((int)sources.size() < K) { std::cerr << "could not find " << K << " sources\n"; return 2; }
    } else sources.push_back(std::atoi(argv[2]) - 1);

    auto make = [&]() {
        auto* b = new spp::bmssp<T>(n);
        for (auto& [u, v, w] : edges) b->addEdge(u, v, w);
        b->prepare_graph(false);
        return b;
    };
    for (int s : sources) {
        auto* run1 = make();
        auto res = run1->execute(s);
        const auto st = run1->stats;
        std::vector<T> bd(n);
        for (int i = 0; i < n; i++) bd[i] = res.first[i] == run1->oo ? -1 : (T)res.first[i];
        delete run1;                                   // only one solver in memory at a time

        auto* run2 = make();
        run2->dtrace_on = true;
        run2->execute(s);
        std::vector<double> ri, rp, rl, rr, rb;
        long long ops = 0, mism = 0;
        for (int r = 0; r < reps; r++) {
            auto x = run2->replay_D();
            ri.push_back(x.init_ms); rp.push_back(x.insert_pivots_ms); rl.push_back(x.pull_ms);
            rr.push_back(x.relax_ms); rb.push_back(x.prepend_ms); ops = x.ops; mism += x.pull_mismatch;
        }
        delete run2;

        std::vector<double> dj; std::vector<T> d;
        for (int r = 0; r < reps; r++) { auto t0 = Clock::now(); d = dijkstra(n, off, to, wt, s); dj.push_back(ms_since(t0)); }
        long long bad = 0;
        for (int i = 0; i < n; i++) bad += bd[i] != d[i];
        if (bad) { std::cerr << "breakdown: BMSSP and Dijkstra distances differ at " << bad << " vertices\n"; return 3; }
        if (mism) { std::cerr << "breakdown: replay of D differs from the run (" << mism << " pulls)\n"; return 3; }

        const double relax_d = median(rr);
        std::cout << "BREAKDOWN source=" << s + 1 << " n=" << n << " m=" << edges.size()
                  << " total_ms=" << st.total_ms << " fp_ms=" << st.fp_ms << " base_ms=" << st.base_ms
                  << " pull_ms=" << st.pull_ms << " prepend_ms=" << st.prepend_ms << " relax_loop_ms=" << st.rec_relax_ms
                  << " other_ms=" << st.total_ms - st.fp_ms - st.base_ms - st.pull_ms - st.prepend_ms - st.rec_relax_ms
                  << " replay_init_ms=" << median(ri) << " replay_insert_pivots_ms=" << median(rp)
                  << " replay_pull_ms=" << median(rl) << " replay_relax_ms=" << relax_d << " replay_prepend_ms=" << median(rb)
                  << " relax_scan_ms=" << st.rec_relax_ms - relax_d
                  << " d_ops=" << ops << " relax_edges=" << st.rec_edges << " fp_calls=" << st.fp_calls
                  << " pull_calls=" << st.pull_calls << " dijkstra_ms=" << median(dj) << " reps=" << reps << std::endl;
    }
    return 0;
}
