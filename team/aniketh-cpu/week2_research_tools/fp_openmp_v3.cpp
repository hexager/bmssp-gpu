// OpenMP FindPivots v3 = v2 + per-call log (FP_CALL_LOG=path.csv) for crossover analysis (external hook; uses fp_state() + findPivotsImpl; does not change the released header).
//
//   g++ -O3 -std=c++20 -fopenmp -I ../cpu fp_openmp_v3.cpp -o fp_omp3
//   FP_MODE=adaptive FP_OMP_MIN_EDGES=2048 OMP_NUM_THREADS=4 BMSSP_CHECK=1 ./fp_omp2 GRAPH.gr SOURCE > distances.txt
//
// FP_MODE (strictly validated):
//   fallback  every call uses the built-in in-place FindPivots   (CONTROL: measures the cost of the hook alone)
//   omp       every call uses synchronous rounds; a round runs in parallel when its edge work >= FP_OMP_MIN_EDGES
//   adaptive  a call whose first-round edge work (sum of out-degrees of S) < FP_OMP_MIN_EDGES uses the built-in
//             in-place version, larger calls use synchronous rounds (parallel per round when work >= threshold)
//   mix       TEST ONLY: each call randomly (FP_MIX_PROB, FP_MIX_SEED) uses the built-in or synchronous rounds
//             (parallel threshold 0). Used to fuzz that interleaving both implementations is safe.
// FP_CALL_LOG=out.csv writes one row per FindPivots call: call,path(0=built-in,1=synchronous),work0(first-round edges),S,W,ms,parallel_rounds
// (refuses to overwrite an existing file). Use call_crossover.py to compare logs of different configurations.
// Changes against fp_openmp.cpp: per-thread proposal buffers are cache-line aligned (no false sharing between threads),
// all work buffers are reused across calls, strict environment parsing, call/time counters per path.
// Synchronous rounds read only the state at the start of a round and commit once, deterministically (minimum key per
// vertex, increasing vertex order), so results do not depend on thread count or schedule. Per-call P and W may differ
// from the built-in in-place version; correctness is validated by final distances and BMSSP_CHECK=1 invariants.
#include <algorithm>
#include <chrono>
#include <cerrno>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "bmssp.hpp"

using T = long long;
using Key = std::tuple<T, int, int, int>;   // (dist, hops, v, pred)
struct Prop { int v; T d; int hops; int u; int root_u; };
struct alignas(64) PerThread { std::vector<Prop> v; };   // one cache line per thread buffer header

static long long env_ll(const char* name, long long dflt, long long lo, long long hi) {
    const char* s = std::getenv(name);
    if (!s) return dflt;
    char* end = nullptr; errno = 0; long long x = std::strtoll(s, &end, 10);
    if (end == s || *end != '\0' || std::isspace((unsigned char)*s) || errno == ERANGE || x < lo || x > hi) {
        std::cerr << "ERROR: " << name << "='" << s << "' invalid (integer in [" << lo << "," << hi << "] required)\n"; std::exit(2);
    }
    return x;
}
static double env_prob(const char* name, double dflt) {
    const char* s = std::getenv(name);
    if (!s) return dflt;
    char* end = nullptr; double x = std::strtod(s, &end);
    if (end == s || *end != '\0' || !(x >= 0.0 && x <= 1.0)) { std::cerr << "ERROR: " << name << " must be in [0,1]\n"; std::exit(2); }
    return x;
}
using Clock = std::chrono::steady_clock;
static double ms_since(Clock::time_point a) { return std::chrono::duration<double, std::milli>(Clock::now() - a).count(); }

int main(int argc, char** argv) {
    if (argc < 3) { std::cerr << "usage: fp_omp2 GRAPH.gr SOURCE\n"; return 2; }
    int n = 0; std::vector<std::tuple<int,int,T>> edges;
    std::ifstream in(argv[1]); std::string line;
    while (std::getline(in, line)) {
        std::istringstream ss(line); char c; ss >> c;
        if (c == 'p') { std::string sp; long long m; ss >> sp >> n >> m; }
        else if (c == 'a') { int u, v; long long w; ss >> u >> v >> w; edges.emplace_back(u - 1, v - 1, w); }
    }
    int s = std::atoi(argv[2]) - 1;
    std::string mode = std::getenv("FP_MODE") ? std::getenv("FP_MODE") : "adaptive";
    if (mode != "fallback" && mode != "omp" && mode != "adaptive" && mode != "mix") { std::cerr << "ERROR: FP_MODE must be fallback|omp|adaptive|mix\n"; return 2; }
    const long long min_edges = env_ll("FP_OMP_MIN_EDGES", 2048, 0, 1000000000LL);
    const double mix_prob = env_prob("FP_MIX_PROB", 0.5);
    std::mt19937_64 rng((unsigned long long)env_ll("FP_MIX_SEED", 1, 0, 1000000000LL));
#ifdef _OPENMP
    const int maxthreads = omp_get_max_threads();
#else
    const int maxthreads = 1;
#endif
    spp::bmssp<T> solver(n);
    for (auto& [u, v, w] : edges) solver.addEdge(u, v, w);
    solver.prepare_graph(false);

    std::vector<int> root(n, 0), seen(n, 0), bseen(n, 0), treesz(n, 0);
    std::vector<Prop> best(n);
    int stamp = 0, bstamp = 0;
    std::vector<PerThread> local(maxthreads);
    std::vector<int> vis, active, nw, touched;           // reused buffers
    struct CallRec { int path; long long work0, s, w; double ms; int par_rounds; };
    std::vector<CallRec> calllog;
    const char* log_path = std::getenv("FP_CALL_LOG");
    const bool want_log = (log_path != nullptr);
    if (want_log) { std::ifstream t(log_path); if (t) { std::cerr << "ERROR: FP_CALL_LOG file exists; refusing to overwrite\n"; return 2; } calllog.reserve(1 << 16); }
    long long calls_builtin = 0, calls_sync = 0, par_rounds = 0, ser_rounds = 0, props_total = 0;
    double hook_ms = 0, commit_ms = 0, builtin_ms = 0;

    solver.findPivotsImpl = [&](auto B, const std::vector<int>& S) -> std::pair<std::vector<int>, std::vector<int>> {
        auto t_call = Clock::now();
        auto st = solver.fp_state();
        const int k = st.k;
        long long w0 = 0; for (int u : S) w0 += st.csr_off[u + 1] - st.csr_off[u];
        bool use_builtin = (mode == "fallback");
        if (mode == "adaptive") {
            use_builtin = (w0 < min_edges);
        } else if (mode == "mix") {
            use_builtin = (std::uniform_real_distribution<double>(0.0, 1.0)(rng) < mix_prob);
        }
        if (use_builtin) {
            auto r = solver.findPivots(B, S);
            calls_builtin++; double dt = ms_since(t_call); builtin_ms += dt; hook_ms += dt;
            if (want_log) calllog.push_back({0, w0, (long long)S.size(), (long long)r.second.size(), dt, 0});
            return r;
        }
        calls_sync++;
        const long long round_threshold = (mode == "mix") ? 0 : min_edges;
        const Key Bk = static_cast<const Key&>(B);
        ++stamp;
        vis.clear(); active.assign(S.begin(), S.end());
        for (int x : S) { vis.push_back(x); seen[x] = stamp; root[x] = x; treesz[x] = 0; }
        bool early = false; int pr_call = 0;
        for (int round = 1; round <= k && !early; round++) {
            long long work = 0;
            for (int u : active) work += st.csr_off[u + 1] - st.csr_off[u];
            const bool parallel = (maxthreads > 1) && work >= round_threshold;
            if (parallel) { par_rounds++; pr_call++; } else ser_rounds++;
            for (auto& l : local) l.v.clear();
            auto body = [&](long i, std::vector<Prop>& out) {
                int u = active[i];
                for (int e = st.csr_off[u]; e < st.csr_off[u + 1]; e++) {
                    int v = st.csr_to[e];
                    Key cand(st.d[u] + st.csr_w[e], st.path_sz[u] + 1, v, u);
                    Key cur(st.d[v], st.path_sz[v], v, st.pred[v]);
                    if (cand <= cur) out.push_back({v, std::get<0>(cand), std::get<1>(cand), u, root[u]});
                }
            };
            const long cnt = (long)active.size();
            if (parallel) {
#ifdef _OPENMP
                #pragma omp parallel
                {
                    const int tid = omp_get_thread_num();
                    if (tid >= (int)local.size()) { std::cerr << "ERROR: thread id out of range\n"; std::abort(); }
                    auto& out = local[tid].v;
                    #pragma omp for schedule(dynamic, 64) nowait
                    for (long i = 0; i < cnt; i++) body(i, out);
                }
#endif
            } else {
                for (long i = 0; i < cnt; i++) body(i, local[0].v);
            }
            auto t_commit = Clock::now();                       // ---- serial deterministic commit ----
            ++bstamp; touched.clear();
            for (auto& l : local)
                for (const Prop& p : l.v) {
                    props_total++;
                    if (bseen[p.v] != bstamp) { bseen[p.v] = bstamp; best[p.v] = p; touched.push_back(p.v); }
                    else if (Key(p.d, p.hops, p.v, p.u) < Key(best[p.v].d, best[p.v].hops, p.v, best[p.v].u)) best[p.v] = p;
                }
            std::sort(touched.begin(), touched.end());
            nw.clear();
            for (int v : touched) {
                const Prop& p = best[v];
                st.pred[v] = p.u; st.d[v] = p.d; st.path_sz[v] = p.hops;
                if (Key(st.d[v], st.path_sz[v], v, st.pred[v]) < Bk) { root[v] = p.root_u; nw.push_back(v); }
            }
            for (int x : nw) if (seen[x] != stamp) { seen[x] = stamp; vis.push_back(x); }
            commit_ms += ms_since(t_commit);
            if (vis.size() > (size_t)k * S.size()) early = true; else active.swap(nw);
        }
        std::vector<int> P;
        if (early) P = S;
        else {
            for (int u : vis) treesz[root[u]]++;
            for (int u : S) if (treesz[u] >= k) P.push_back(u);
        }
        { double dt = ms_since(t_call); hook_ms += dt;
          if (want_log) calllog.push_back({1, w0, (long long)S.size(), (long long)vis.size(), dt, pr_call}); }
        return {P, vis};
    };

    auto t_exec = Clock::now();
    auto res = solver.execute(s);
    const double exec_ms = ms_since(t_exec);
    for (int i = 0; i < n; i++) std::printf("%lld\n", res.first[i] == solver.oo ? -1LL : (long long)res.first[i]);
    std::cerr << "OPENMP_EXECUTE_MS " << exec_ms << "\n";
    if (want_log) { std::ofstream lf(log_path); lf << "call,path,work0,S,W,ms,parallel_rounds\n";
        for (size_t i = 0; i < calllog.size(); i++) lf << i << ',' << calllog[i].path << ',' << calllog[i].work0 << ',' << calllog[i].s << ',' << calllog[i].w << ',' << calllog[i].ms << ',' << calllog[i].par_rounds << '\n'; }
    std::cerr << "FPOMP mode=" << mode << " threads=" << maxthreads << " min_edges=" << min_edges << " calls_builtin=" << calls_builtin
              << " calls_sync=" << calls_sync << " parallel_rounds=" << par_rounds << " serial_rounds=" << ser_rounds
              << " proposals=" << props_total << " hook_ms=" << hook_ms << " builtin_ms=" << builtin_ms << " serial_commit_ms=" << commit_ms << "\n";
    return 0;
}
