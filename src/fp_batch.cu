// Batched GPU FindPivots: one set of kernel launches serves FindPivots calls from several independent BMSSP
// searches (one per source). The searches are treated as disjoint copies of the graph: vertex v of search i is
// gid = i*n + v. The CSR graph is stored once; a gid's neighbours are i*n + to[e]. Each call keeps its own bound
// B, its own early exit and its own W and P. Per call, the result equals fp_gpu.cu / the serial synchronous
// version: proposals read the state at the start of the round and the winner is the lexicographic minimum
// (dist, hops, pred), with vertex and predecessor ids local to their search.
//
// Build for a GPU:  nvcc -O3 -std=c++17 -c fp_batch.cu     Emulation: g++ -O2 -std=c++17 -DFP_EMULATE -x c++ -c fp_batch.cu
#include "fp_batch.h"

#include <algorithm>
#include <chrono>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

typedef long long ll;
typedef unsigned long long ull;
static const ll DINF = LLONG_MAX;
static const ull UINF = ULLONG_MAX;

#ifdef FP_EMULATE
#include <random>
#define __global__
#define __device__
static int emu_tid_;
static std::mt19937_64 emu_rng_(std::getenv("FP_EMU_SEED") ? std::strtoull(std::getenv("FP_EMU_SEED"), nullptr, 10) : 1);
#define TID() (emu_tid_)
template <class K, class... A> static void emu_launch(int n, K kern, A... args) {
    std::vector<int> ord(n);
    std::iota(ord.begin(), ord.end(), 0);
    std::shuffle(ord.begin(), ord.end(), emu_rng_);
    for (int i : ord) { emu_tid_ = i; kern(args...); }
}
#define LAUNCH(kern, n, ...) do { if ((n) > 0) { emu_launch((int)(n), kern, __VA_ARGS__); g->launches++; } } while (0)
static ll atomicMin(ll* p, ll v) { ll o = *p; if (v < o) *p = v; return o; }
static ull atomicMin(ull* p, ull v) { ull o = *p; if (v < o) *p = v; return o; }
static int atomicExch(int* p, int v) { int o = *p; *p = v; return o; }
static int atomicAdd(int* p, int v) { int o = *p; *p += v; return o; }
static void* dmalloc(size_t b) { return std::calloc(1, b ? b : 1); }
static void dfree(void* p) { std::free(p); }
static void* hmalloc(size_t b) { return std::malloc(b ? b : 1); }
static void hfree(void* p) { std::free(p); }
static void h2d(void* dst, const void* src, size_t b) { if (b) std::memcpy(dst, src, b); }
static void d2h(void* dst, const void* src, size_t b) { if (b) std::memcpy(dst, src, b); }
static void dzero(void* p, size_t b) { std::memset(p, 0, b); }
static void dsync() {}
static void set_blocking_sync() {}
static std::string device_name() { return "CPU emulation (shuffled thread order)"; }
#else
#include <cuda_runtime.h>
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    std::fprintf(stderr, "CUDA error: %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); std::exit(4); } } while (0)
#define TID() ((int)(blockIdx.x * blockDim.x + threadIdx.x))
#define LAUNCH(kern, n, ...) do { if ((n) > 0) { \
    kern<<<(int)(((n) + 255) / 256), 256>>>(__VA_ARGS__); CK(cudaGetLastError()); g->launches++; } } while (0)
static void* dmalloc(size_t b) { void* p = nullptr; CK(cudaMalloc(&p, b ? b : 1)); CK(cudaMemset(p, 0, b ? b : 1)); return p; }
static void dfree(void* p) { if (p) cudaFree(p); }
static void* hmalloc(size_t b) { void* p = nullptr; CK(cudaMallocHost(&p, b ? b : 1)); return p; }
static void hfree(void* p) { if (p) cudaFreeHost(p); }
static void h2d(void* dst, const void* src, size_t b) { if (b) CK(cudaMemcpy(dst, src, b, cudaMemcpyHostToDevice)); }
static void d2h(void* dst, const void* src, size_t b) { if (b) CK(cudaMemcpy(dst, src, b, cudaMemcpyDeviceToHost)); }
static void dzero(void* p, size_t b) { CK(cudaMemsetAsync(p, 0, b)); }
static void dsync() { CK(cudaDeviceSynchronize()); }
// Blocking synchronization: a waiting host thread sleeps instead of spinning, so the CPU threads of other
// searches can use the core. Must run before the first CUDA call that creates the context.
// FP_BATCH_SYNC=spin selects spin-waiting instead (for measuring what blocking synchronization costs).
static void set_blocking_sync() {
    const char* m = std::getenv("FP_BATCH_SYNC");
    if (m && std::string(m) == "spin") CK(cudaSetDeviceFlags(cudaDeviceScheduleSpin));
    else CK(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync));
}
static std::string device_name() {
    int dev = 0; cudaDeviceProp pr; CK(cudaGetDevice(&dev)); CK(cudaGetDeviceProperties(&pr, dev));
    return std::string(pr.name) + " (sm_" + std::to_string(pr.major) + std::to_string(pr.minor) + ")";
}
#endif

struct Bound { ll d; int h, v, p; };

struct KArgs {
    const int* off; const int* to; const ll* w;       // CSR graph (one copy)
    ll* d; int* pred; int* hops;                       // state of all searches, indexed by gid; pred is local
    int* root; int* nroot;                             // FindPivots tree roots (gid)
    ll* bd; ull* bhp;                                  // best proposal of the current round, per gid
    int* tst; int* dst;                                // round stamp (touched) and batch stamp (changed)
    int* touched; int* nw; int* dirty; int* cnt;       // lists (gids); cnt[0] touched, cnt[1] nw, cnt[2] dirty
    const int* active; const int* start;               // active gids and their edge prefix sums
    const Bound* bnd;                                  // bound of each search in this batch, by search index
    int na; int E; int rs; int cs; int n; int N;
};

__device__ inline bool le3(ll a, int b, int c, ll x, int y, int z) {
    return a < x || (a == x && (b < y || (b == y && c <= z)));
}
__device__ inline bool lt4(ll a, int b, int c, int e, ll x, int y, int z, int f) {
    if (a != x) return a < x;
    if (b != y) return b < y;
    if (c != z) return c < z;
    return e < f;
}

struct Prop { int ug, vg, u, base; ll cd; int ch; bool ok; };

__device__ inline Prop proposal(const KArgs& a, int e) {
    int lo = 0, hi = a.na;                       // invariant: start[lo] <= e < start[hi]
    while (hi - lo > 1) { int mid = (lo + hi) >> 1; if (a.start[mid] <= e) lo = mid; else hi = mid; }
    Prop r;
    r.ug = a.active[lo];
    r.base = (r.ug / a.n) * a.n;
    r.u = r.ug - r.base;
    int idx = a.off[r.u] + (e - a.start[lo]);
    r.vg = r.base + a.to[idx];
    r.cd = a.d[r.ug] + a.w[idx];
    r.ch = a.hops[r.ug] + 1;
    r.ok = le3(r.cd, r.ch, r.u, a.d[r.vg], a.hops[r.vg], a.pred[r.vg]);
    return r;
}

__global__ void k_init(KArgs a) {
    int v = TID(); if (v >= a.N) return;
    a.bd[v] = DINF; a.bhp[v] = UINF;
}
__global__ void k_roots(KArgs a) {
    int i = TID(); if (i >= a.na) return;
    int x = a.active[i]; a.root[x] = x;
}
__global__ void k_pass1(KArgs a) {
    int e = TID(); if (e >= a.E) return;
    Prop r = proposal(a, e); if (!r.ok) return;
    atomicMin(&a.bd[r.vg], r.cd);
    if (atomicExch(&a.tst[r.vg], a.rs) != a.rs) a.touched[atomicAdd(&a.cnt[0], 1)] = r.vg;
}
__global__ void k_pass2(KArgs a) {
    int e = TID(); if (e >= a.E) return;
    Prop r = proposal(a, e);
    if (r.ok && r.cd == a.bd[r.vg]) atomicMin(&a.bhp[r.vg], ((ull)(unsigned)r.ch << 32) | (ull)(unsigned)r.u);
}
// Two kernels so that no thread reads root[] while another writes it.
__global__ void k_commit_a(KArgs a) {
    int i = TID(); if (i >= a.cnt[0]) return;
    int vg = a.touched[i]; int base = (vg / a.n) * a.n;
    a.nroot[vg] = a.root[base + (int)(a.bhp[vg] & 0xffffffffULL)];
}
__global__ void k_commit_b(KArgs a) {
    int i = TID(); if (i >= a.cnt[0]) return;
    int vg = a.touched[i]; int s = vg / a.n; int v = vg - s * a.n;
    ll nd = a.bd[vg]; ull hp = a.bhp[vg];
    int nh = (int)(hp >> 32), np = (int)(hp & 0xffffffffULL);
    a.d[vg] = nd; a.hops[vg] = nh; a.pred[vg] = np;
    a.bd[vg] = DINF; a.bhp[vg] = UINF;
    const Bound b = a.bnd[s];
    if (lt4(nd, nh, v, np, b.d, b.h, b.v, b.p)) {
        a.root[vg] = a.nroot[vg];
        int p = atomicAdd(&a.cnt[1], 1); a.nw[2 * p] = vg; a.nw[2 * p + 1] = a.nroot[vg];
    }
    if (atomicExch(&a.dst[vg], a.cs) != a.cs) a.dirty[atomicAdd(&a.cnt[2], 1)] = vg;
}
__global__ void k_pack(KArgs a, ll* out, int nd) {
    int i = TID(); if (i >= nd) return;
    int vg = a.dirty[i];
    out[3 * i] = vg; out[3 * i + 1] = a.d[vg];
    out[3 * i + 2] = (ll)(((ull)(unsigned)a.hops[vg] << 32) | (ull)(unsigned)a.pred[vg]);
}
__global__ void k_scatter(KArgs a, const ll* in, int nl) {
    int i = TID(); if (i >= nl) return;
    int vg = (int)in[3 * i]; ull hp = (ull)in[3 * i + 2];
    a.d[vg] = in[3 * i + 1]; a.hops[vg] = (int)(hp >> 32); a.pred[vg] = (int)(hp & 0xffffffffULL);
}

struct FpBatch {
    int n = 0, S = 0, N = 0; int launches = 0;
    int roundstamp = 0, callstamp = 0;
    KArgs a{};
    std::vector<int> off;
    std::vector<char> uploaded;                        // search slice already on the device
    int* d_in = nullptr; ll* d_pack = nullptr; Bound* d_bnd = nullptr;
    int* h_in = nullptr; int* h_nw = nullptr; int* h_cnt = nullptr; ll* h_pack = nullptr; Bound* h_bnd = nullptr;
    std::string name;
    // per request host bookkeeping
    struct Work { std::vector<int> act, hseen, hroot; int stamp = 0; bool early = false, done = false; };
    std::vector<Work> work;                            // by search index
    std::vector<int> htst, hcnt; int tstamp = 0;
};

using Clock = std::chrono::steady_clock;
static double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

FpBatch* fpb_create(int n, int searches, const int* off, const int* to, const ll* w) {
    FpBatch* g = new FpBatch();
    g->n = n; g->S = searches;
    const long long Nll = (long long)n * searches;
    if (Nll > INT_MAX / 2) { std::fprintf(stderr, "fpb: n * searches too large\n"); std::exit(4); }
    const int N = (int)Nll; g->N = N;
    set_blocking_sync();
    g->name = device_name();
    const int m = off[n];
    g->off.assign(off, off + n + 1);
    g->uploaded.assign(searches, 0);
    g->work.resize(searches);
    for (auto& wk : g->work) { wk.hseen.assign(n, 0); wk.hroot.assign(n, 0); }
    g->htst.assign(n, 0); g->hcnt.assign(n, 0);
    KArgs& a = g->a;
    a.n = n; a.N = N;
    int* doff = (int*)dmalloc((size_t)(n + 1) * 4); h2d(doff, off, (size_t)(n + 1) * 4); a.off = doff;
    int* dto = (int*)dmalloc((size_t)m * 4);        h2d(dto, to, (size_t)m * 4);        a.to = dto;
    ll* dw = (ll*)dmalloc((size_t)m * 8);           h2d(dw, w, (size_t)m * 8);          a.w = dw;
    a.d = (ll*)dmalloc((size_t)N * 8); a.pred = (int*)dmalloc((size_t)N * 4); a.hops = (int*)dmalloc((size_t)N * 4);
    a.root = (int*)dmalloc((size_t)N * 4); a.nroot = (int*)dmalloc((size_t)N * 4);
    a.bd = (ll*)dmalloc((size_t)N * 8); a.bhp = (ull*)dmalloc((size_t)N * 8);
    a.tst = (int*)dmalloc((size_t)N * 4); a.dst = (int*)dmalloc((size_t)N * 4);
    a.touched = (int*)dmalloc((size_t)N * 4); a.nw = (int*)dmalloc((size_t)N * 8); a.dirty = (int*)dmalloc((size_t)N * 4);
    a.cnt = (int*)dmalloc(4 * sizeof(int));
    g->d_in = (int*)dmalloc((size_t)(2 * N + 1) * 4);
    g->d_pack = (ll*)dmalloc((size_t)N * 24);
    g->d_bnd = (Bound*)dmalloc((size_t)searches * sizeof(Bound)); a.bnd = g->d_bnd;
    g->h_in = (int*)hmalloc((size_t)(2 * N + 1) * 4); g->h_nw = (int*)hmalloc((size_t)N * 8);
    g->h_cnt = (int*)hmalloc(4 * sizeof(int)); g->h_pack = (ll*)hmalloc((size_t)N * 24);
    g->h_bnd = (Bound*)hmalloc((size_t)searches * sizeof(Bound));
    LAUNCH(k_init, N, a);
    dsync();
    return g;
}

void fpb_destroy(FpBatch* g) {
    if (!g) return;
    KArgs& a = g->a;
    void* dev[] = {(void*)a.off, (void*)a.to, (void*)a.w, a.d, a.pred, a.hops, a.root, a.nroot, a.bd, a.bhp,
                   a.tst, a.dst, a.touched, a.nw, a.dirty, a.cnt, g->d_in, g->d_pack, g->d_bnd};
    for (void* p : dev) dfree(p);
    void* host[] = {g->h_in, g->h_nw, g->h_cnt, g->h_pack, g->h_bnd};
    for (void* p : host) hfree(p);
    delete g;
}

const char* fpb_device_name(FpBatch* g) { return g->name.c_str(); }

void fpb_run(FpBatch* g, std::vector<FpbReq>& reqs, int k, FpbStats* st) {
    const int before = g->launches;
    const int n = g->n;
    KArgs a = g->a;
    auto t0 = Clock::now();
    // 1. Upload: a search's first call uploads its whole slice; later calls upload only the vertices its CPU changed.
    int np = 0;
    for (auto& r : reqs) {
        const int base = r.search * n;
        if (!g->uploaded[r.search]) {
            h2d(a.d + base, r.d, (size_t)n * 8); h2d(a.pred + base, r.pred, (size_t)n * 4); h2d(a.hops + base, r.hops, (size_t)n * 4);
            g->uploaded[r.search] = 1; st->upload_vertices += n;
        } else {
            for (int i = 0; i < r.ndirty; i++) {
                int v = r.dirty[i];
                g->h_pack[3 * np] = base + v; g->h_pack[3 * np + 1] = r.d[v];
                g->h_pack[3 * np + 2] = (ll)(((ull)(unsigned)r.hops[v] << 32) | (ull)(unsigned)r.pred[v]);
                np++;
            }
            st->upload_vertices += r.ndirty;
        }
        g->h_bnd[r.search] = {r.Bd, r.Bh, r.Bv, r.Bp};
    }
    if (np > 0) { h2d(g->d_pack, g->h_pack, (size_t)np * 24); LAUNCH(k_scatter, np, a, g->d_pack, np); }
    h2d(g->d_bnd, g->h_bnd, (size_t)g->S * sizeof(Bound));
    dsync();
    st->upload_ms += ms_since(t0);

    // 2. Rounds over the union of the active lists.
    auto t1 = Clock::now();
    a.cs = ++g->callstamp;
    for (auto& r : reqs) {
        auto& wk = g->work[r.search];
        wk.stamp++; wk.early = false; wk.done = false;
        r.W->assign(r.S->begin(), r.S->end());
        for (int x : *r.S) { wk.hseen[x] = wk.stamp; wk.hroot[x] = x; }
        wk.act.assign(r.S->begin(), r.S->end());
    }
    dzero(a.cnt, 3 * sizeof(int));
    int nd = 0;
    std::vector<std::vector<std::pair<int, int>>> split(g->S);
    for (int round = 1; round <= k; round++) {
        int* hs = g->h_in; int na = 0; long long E = 0;
        for (auto& r : reqs) { auto& wk = g->work[r.search]; if (!wk.done) na += (int)wk.act.size(); }
        if (na == 0) break;
        hs[0] = 0; int i = 0;
        for (auto& r : reqs) {
            auto& wk = g->work[r.search]; if (wk.done) continue;
            const int base = r.search * n;
            for (int u : wk.act) {
                hs[na + 1 + i] = base + u;
                E += g->off[u + 1] - g->off[u];
                if (E > INT_MAX) { std::fprintf(stderr, "fpb: round edge count exceeds int range\n"); std::exit(4); }
                hs[++i] = (int)E;
            }
        }
        h2d(g->d_in, hs, (size_t)(2 * na + 1) * 4);
        if (round > 1) dzero(a.cnt, 2 * sizeof(int));
        a.start = g->d_in; a.active = g->d_in + na + 1;
        a.na = na; a.E = (int)E; a.rs = ++g->roundstamp;
        if (round == 1) LAUNCH(k_roots, na, a);
        LAUNCH(k_pass1, a.E, a);
        LAUNCH(k_pass2, a.E, a);
        const int nl = (int)std::min<long long>(E, g->N);
        LAUNCH(k_commit_a, nl, a);
        LAUNCH(k_commit_b, nl, a);
        dsync();
        d2h(g->h_cnt, a.cnt, 3 * sizeof(int));
        const int nnw = g->h_cnt[1];
        nd = g->h_cnt[2];
        d2h(g->h_nw, a.nw, (size_t)nnw * 8);
        st->edges += E; st->rounds++;
        for (auto& v : split) v.clear();
        for (int j = 0; j < nnw; j++) {
            int vg = g->h_nw[2 * j], rg = g->h_nw[2 * j + 1], s = vg / n;
            split[s].push_back({vg - s * n, rg - s * n});
        }
        // Per search, the same bookkeeping as the single-search version.
        for (auto& r : reqs) {
            auto& wk = g->work[r.search]; if (wk.done) continue;
            auto& pr = split[r.search];
            std::sort(pr.begin(), pr.end());
            wk.act.clear();
            for (auto& [v, rt] : pr) {
                wk.hroot[v] = rt; wk.act.push_back(v);
                if (wk.hseen[v] != wk.stamp) { wk.hseen[v] = wk.stamp; r.W->push_back(v); }
            }
            if (r.W->size() > (size_t)k * r.S->size()) { wk.early = true; wk.done = true; }
            if (round == k) wk.done = true;
        }
    }
    for (auto& r : reqs) {
        auto& wk = g->work[r.search];
        r.P->clear();
        if (wk.early) { *r.P = *r.S; continue; }
        const int ts = ++g->tstamp;
        for (int u : *r.W) {
            int rt = wk.hroot[u];
            if (g->htst[rt] != ts) { g->htst[rt] = ts; g->hcnt[rt] = 0; }
            g->hcnt[rt]++;
        }
        for (int u : *r.S) if (g->htst[u] == ts && g->hcnt[u] >= k) r.P->push_back(u);
    }
    st->rounds_ms += ms_since(t1);

    // 3. Download every vertex changed in this batch to its search's host arrays.
    auto t2 = Clock::now();
    if (nd > 0) {
        LAUNCH(k_pack, nd, a, g->d_pack, nd);
        d2h(g->h_pack, g->d_pack, (size_t)nd * 24);
        std::vector<int> owner(g->S, -1);
        for (int j = 0; j < (int)reqs.size(); j++) owner[reqs[j].search] = j;
        for (int i = 0; i < nd; i++) {
            int vg = (int)g->h_pack[3 * i]; int s = vg / n, v = vg - s * n; ull hp = (ull)g->h_pack[3 * i + 2];
            const FpbReq& r = reqs[owner[s]];
            r.d[v] = g->h_pack[3 * i + 1]; r.hops[v] = (int)(hp >> 32); r.pred[v] = (int)(hp & 0xffffffffULL);
        }
    }
    st->download_vertices += nd;
    st->download_ms += ms_since(t2);
    st->launches += g->launches - before;
    st->batches++; st->calls += (long long)reqs.size();
}
