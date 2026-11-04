// GPU Near-Far SSSP baseline (delta-stepping with two buckets, after Davidson et al., IPDPS 2014).
// Fully device-resident; edge-parallel relaxation (exclusive scan of frontier degrees + binary search).
//
//   near phase  relax every out-edge of the near frontier with atomicMin; an improved vertex goes to the next
//               near frontier if its new distance is below the split, otherwise to the far pile
//   far phase   when near is empty, move far entries with distance in [old split, split) to near; entries below
//               the old split are already settled and dropped; if nothing moves, the split jumps to the smallest
//               remaining far distance
//
// usage: nearfar GRAPH.gr SOURCE [DELTAS]  (comma list; each an integer or xF = F x average edge weight; default x8)
// stdout: one distance per line (-1 if unreachable); stderr: one NEARFAR line per delta
// Build: nvcc -O3 -std=c++17 nearfar.cu   |  emulation: g++ -O2 -std=c++17 -DFP_EMULATE -x c++ nearfar.cu
#include <algorithm>
#include <chrono>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

typedef long long ll;
static const ll DINF = LLONG_MAX;

#ifdef FP_EMULATE
#include <random>
#define __global__
#define __device__
static int emu_tid_;
static std::mt19937_64 emu_rng_(std::getenv("FP_EMU_SEED") ? std::strtoull(std::getenv("FP_EMU_SEED"), nullptr, 10) : 1);
#define TID() (emu_tid_)
template <class K, class... A> static void emu_launch(int n, K kern, A... args) {
    std::vector<int> ord(n); std::iota(ord.begin(), ord.end(), 0); std::shuffle(ord.begin(), ord.end(), emu_rng_);
    for (int i : ord) { emu_tid_ = i; kern(args...); }
}
#define LAUNCH(kern, n, ...) do { if ((n) > 0) { emu_launch((int)(n), kern, __VA_ARGS__); launches++; } } while (0)
static ll atomicMin(ll* p, ll v) { ll o = *p; if (v < o) *p = v; return o; }
static int atomicExch(int* p, int v) { int o = *p; *p = v; return o; }
static int atomicAdd(int* p, int v) { int o = *p; *p += v; return o; }
static void* dmalloc(size_t b) { return std::calloc(1, b ? b : 1); }
static void dfree(void* p) { std::free(p); }
static void h2d(void* d, const void* s, size_t b) { if (b) std::memcpy(d, s, b); }
static void d2h(void* d, const void* s, size_t b) { if (b) std::memcpy(d, s, b); }
static void dzero(void* p, size_t b) { std::memset(p, 0, b); }
static void dsync() {}
static void exclusive_scan(const int* in, int* out, int n) { std::exclusive_scan(in, in + n, out, 0); }
static std::string device_name() { return "CPU emulation (shuffled thread order)"; }
#else
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    std::fprintf(stderr, "CUDA error: %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); std::exit(4); } } while (0)
#define TID() ((int)(blockIdx.x * blockDim.x + threadIdx.x))
#define LAUNCH(kern, n, ...) do { if ((n) > 0) { \
    kern<<<(int)(((n) + 255) / 256), 256>>>(__VA_ARGS__); CK(cudaGetLastError()); launches++; } } while (0)
static void* dmalloc(size_t b) { void* p = nullptr; CK(cudaMalloc(&p, b ? b : 1)); CK(cudaMemset(p, 0, b ? b : 1)); return p; }
static void dfree(void* p) { if (p) cudaFree(p); }
static void h2d(void* d, const void* s, size_t b) { if (b) CK(cudaMemcpy(d, s, b, cudaMemcpyHostToDevice)); }
static void d2h(void* d, const void* s, size_t b) { if (b) CK(cudaMemcpy(d, s, b, cudaMemcpyDeviceToHost)); }
static void dzero(void* p, size_t b) { CK(cudaMemsetAsync(p, 0, b)); }
static void dsync() { CK(cudaDeviceSynchronize()); }
static void* scan_tmp = nullptr; static size_t scan_tmp_bytes = 0;
static void exclusive_scan(const int* in, int* out, int n) {
    size_t need = 0; CK(cub::DeviceScan::ExclusiveSum(nullptr, need, in, out, n));
    if (need > scan_tmp_bytes) { dfree(scan_tmp); scan_tmp = dmalloc(need); scan_tmp_bytes = need; }
    CK(cub::DeviceScan::ExclusiveSum(scan_tmp, need, in, out, n));
}
static std::string device_name() {
    int dev = 0; cudaDeviceProp pr; CK(cudaGetDevice(&dev)); CK(cudaGetDeviceProperties(&pr, dev));
    return std::string(pr.name) + " (sm_" + std::to_string(pr.major) + std::to_string(pr.minor) + ")";
}
#endif
static int launches = 0;

struct G { const int* off; const int* to; const ll* w; ll* d; };

__global__ void k_init(ll* d, int n, int s) { int v = TID(); if (v >= n) return; d[v] = (v == s) ? 0 : DINF; }
__global__ void k_deg(G g, const int* fr, int nf, int* deg) {
    int i = TID(); if (i >= nf) return; int u = fr[i]; deg[i] = g.off[u + 1] - g.off[u];
}
// cnt[0] next-near size, cnt[1] far size.
__global__ void k_relax(G g, const int* fr, int nf, const int* start, int E, ll split, int stamp, int* nst,
                        int* nxt, int* far, int far_cap, int* cnt) {
    int e = TID(); if (e >= E) return;
    int lo = 0, hi = nf;                                  // start[lo] <= e < start[hi] (start[nf] == E)
    while (hi - lo > 1) { int mid = (lo + hi) >> 1; if (start[mid] <= e) lo = mid; else hi = mid; }
    int u = fr[lo];
    int idx = g.off[u] + (e - start[lo]);
    int v = g.to[idx];
    ll nd = g.d[u] + g.w[idx];
    ll old = atomicMin(&g.d[v], nd);
    if (nd >= old) return;
    if (nd < split) { if (atomicExch(&nst[v], stamp) != stamp) nxt[atomicAdd(&cnt[0], 1)] = v; }
    else { int p = atomicAdd(&cnt[1], 1); if (p < far_cap) far[p] = v; }
}
// Splits the far pile: [lo_split, split) -> near; >= split -> new far pile (deduplicated); < lo_split -> dropped.
// cnt[0] near size, cnt[1] new far size, fmin = smallest distance left in far.
__global__ void k_split(G g, const int* far, int nfar, ll lo_split, ll split, int stamp, int* nst, int* fst,
                        int* nxt, int* far2, int* cnt, ll* fmin) {
    int i = TID(); if (i >= nfar) return;
    int v = far[i]; ll dv = g.d[v];
    if (dv < lo_split) return;
    if (dv < split) { if (atomicExch(&nst[v], stamp) != stamp) nxt[atomicAdd(&cnt[0], 1)] = v; }
    else if (atomicExch(&fst[v], stamp) != stamp) { far2[atomicAdd(&cnt[1], 1)] = v; atomicMin(fmin, dv); }
}

using Clock = std::chrono::steady_clock;
static double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

int main(int argc, char** argv) {
    if (argc < 3) { std::cerr << "usage: nearfar GRAPH.gr SOURCE [DELTA]\n"; return 2; }
    int n = 0; std::vector<std::tuple<int, int, ll>> edges;
    {
        std::ifstream in(argv[1]); if (!in) { std::cerr << "cannot open " << argv[1] << "\n"; return 2; }
        std::string line;
        while (std::getline(in, line)) {
            if (line.empty()) continue;
            std::istringstream ss(line); char c; ss >> c;
            if (c == 'p') { std::string sp; long long m; ss >> sp >> n >> m; }
            else if (c == 'a') { int u, v; ll w; ss >> u >> v >> w; edges.emplace_back(u - 1, v - 1, w); }
        }
    }
    const int m = (int)edges.size();
    std::vector<int> off(n + 1, 0), to(m); std::vector<ll> w(m);
    for (auto& [u, v, x] : edges) off[u + 1]++;
    for (int i = 0; i < n; i++) off[i + 1] += off[i];
    { std::vector<int> pos(off.begin(), off.end() - 1); for (auto& [u, v, x] : edges) { int p = pos[u]++; to[p] = v; w[p] = x; } }
    double wsum = 0; for (ll x : w) wsum += (double)x;
    // DELTA list: comma-separated; each an integer or xF = F times the average edge weight (default x8).
    // With several values, a warm-up run of the first value is done and not reported; then each value runs once.
    std::vector<std::string> dargs; { std::stringstream ss(argc > 3 ? argv[3] : "x8"); std::string t; while (std::getline(ss, t, ',')) dargs.push_back(t); }
    const double avgw = wsum / std::max(1, m);
    auto to_delta = [&](const std::string& a) {
        return a[0] == 'x' ? std::max<ll>(1, (ll)(avgw * std::atof(a.c_str() + 1))) : std::max<ll>(1, std::atoll(a.c_str())); };
    const int s = std::atoi(argv[2]) - 1;
    auto ts = Clock::now();
    const std::string name = device_name();
    G g;
    int* doff = (int*)dmalloc((size_t)(n + 1) * 4); h2d(doff, off.data(), (size_t)(n + 1) * 4); g.off = doff;
    int* dto = (int*)dmalloc((size_t)m * 4); h2d(dto, to.data(), (size_t)m * 4); g.to = dto;
    ll* dw = (ll*)dmalloc((size_t)m * 8); h2d(dw, w.data(), (size_t)m * 8); g.w = dw;
    g.d = (ll*)dmalloc((size_t)n * 8);
    const int far_cap = 2 * (m + n);
    int *fr = (int*)dmalloc((size_t)n * 4), *nxt = (int*)dmalloc((size_t)n * 4), *deg = (int*)dmalloc((size_t)n * 4);
    int *start = (int*)dmalloc((size_t)(n + 1) * 4), *nst = (int*)dmalloc((size_t)n * 4), *fst = (int*)dmalloc((size_t)n * 4);
    int *far = (int*)dmalloc((size_t)far_cap * 4), *far2 = (int*)dmalloc((size_t)far_cap * 4);
    int* cnt = (int*)dmalloc(2 * sizeof(int)); ll* fmin = (ll*)dmalloc(sizeof(ll));
    dsync();
    const double setup_ms = ms_since(ts);


    int stamp = 0;                                             // never reset: nst/fst keep stamps across runs
    std::vector<ll> d(n), d0;
    if (dargs.size() > 1) dargs.insert(dargs.begin(), dargs[0]);
    for (size_t run = 0; run < dargs.size(); run++) {
    const bool warm = dargs.size() > 1 && run == 0;
    const ll delta = to_delta(dargs[run]);
    launches = 0;
    auto te = Clock::now();
    LAUNCH(k_init, n, g.d, n, s);
    int nf = 1; h2d(fr, &s, 4);
    int nfar = 0, iters = 0, phases = 0; long long edges_examined = 0;
    ll split = delta;
    while (true) {
        while (nf > 0) {                                       // near phase
            LAUNCH(k_deg, nf, g, fr, nf, deg);
            exclusive_scan(deg, start, nf);
            int last[2]; d2h(&last[0], start + nf - 1, 4); d2h(&last[1], deg + nf - 1, 4);
            const long long E = (long long)last[0] + last[1];
            if (E > INT_MAX) { std::fprintf(stderr, "nearfar: frontier edge count exceeds int range\n"); return 4; }
            int Ei = (int)E; h2d(start + nf, &Ei, 4);
            ++stamp; dzero(cnt, 4);                            // reset next-near count; far count carries over
            h2d(cnt + 1, &nfar, 4);
            LAUNCH(k_relax, Ei, g, fr, nf, start, Ei, split, stamp, nst, nxt, far, far_cap, cnt);
            int c[2]; d2h(c, cnt, 8);
            if (c[1] > far_cap) { std::fprintf(stderr, "nearfar: far pile overflow\n"); return 4; }
            nf = c[0]; nfar = c[1]; std::swap(fr, nxt);
            edges_examined += E; iters++;
        }
        if (nfar == 0) break;
        ll lo_split = split;                                   // far phase: advance the split
        split += delta;
        while (true) {
            ++stamp; dzero(cnt, 8); ll big = DINF; h2d(fmin, &big, 8);
            LAUNCH(k_split, nfar, g, far, nfar, lo_split, split, stamp, nst, fst, fr, far2, cnt, fmin);
            int c[2]; d2h(c, cnt, 8); ll fm; d2h(&fm, fmin, 8);
            nf = c[0]; nfar = c[1]; std::swap(far, far2); phases++;
            if (nf > 0 || nfar == 0) break;
            lo_split = split;                                  // nothing in the window: jump to the next distance
            split = (fm / delta + 1) * delta;
        }
        if (nf == 0 && nfar == 0) break;
    }
    dsync();
    const double exec_ms = ms_since(te);
    d2h(d.data(), g.d, (size_t)n * 8);
    if (d0.empty()) d0 = d;
    else if (d != d0) { std::fprintf(stderr, "nearfar: distances differ between delta values\n"); return 5; }
    if (warm) continue;
    std::cerr << "NEARFAR device=\"" << name << "\" delta_arg=" << dargs[run] << " delta=" << delta << " avg_weight=" << avgw
              << " execute_ms=" << exec_ms << " setup_ms=" << setup_ms << " iterations=" << iters << " far_phases=" << phases
              << " edges_examined=" << edges_examined << " launches=" << launches << "\n";
    }
    std::string out; out.reserve((size_t)n * 8); char buf[32];
    for (ll x : d0) { int len = std::snprintf(buf, sizeof buf, "%lld\n", x == DINF ? -1LL : x); out.append(buf, len); }
    std::fwrite(out.data(), 1, out.size(), stdout);
    void* ptrs[] = {doff, dto, dw, g.d, fr, nxt, deg, start, nst, fst, far, far2, cnt, fmin};
    for (void* p : ptrs) dfree(p);
    return 0;
}
