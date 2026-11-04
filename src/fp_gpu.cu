// GPU FindPivots: synchronous Bellman-Ford rounds with a deterministic per-vertex commit.
//
// One round, for the active list A (edge-parallel, one thread per out-edge of A):
//   pass 1  each improving proposal (cand <= current key) does atomicMin on the distance
//   pass 2  proposals that hold the winning distance do atomicMin on (hops << 32 | pred)
//   commit  write the winner; a vertex whose new key is below B joins the next active list
// The passes only read state from the start of the round, so the winner is the lexicographic
// minimum (dist, hops, pred) whatever the thread order. This equals the serial synchronous version.
// hops and pred are non-negative 31-bit values, so the packed 64-bit order equals the (hops, pred) order.
//
// Build for a GPU:          nvcc -O3 -std=c++17 -c fp_gpu.cu
// Build for emulation:      g++ -O2 -std=c++17 -DFP_EMULATE -x c++ -c fp_gpu.cu
// Emulation runs every kernel as a loop over its threads in a shuffled order (FP_EMU_SEED).
#include "fp_gpu.h"

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
static std::string device_name() {
    int dev = 0; cudaDeviceProp pr; CK(cudaGetDevice(&dev)); CK(cudaGetDeviceProperties(&pr, dev));
    return std::string(pr.name) + " (sm_" + std::to_string(pr.major) + std::to_string(pr.minor) + ")";
}
#endif

// Everything a kernel needs, passed by value (only device pointers and scalars).
struct KArgs {
    const int* off; const int* to; const ll* w;       // CSR graph
    ll* d; int* pred; int* hops;                       // solver state (device copy)
    int* root; int* nroot;                             // FindPivots tree roots
    ll* bd; ull* bhp;                                  // per-vertex best proposal of the current round
    int* tst; int* dst;                                // round stamp (touched) and call stamp (changed)
    int* touched; int* nw; int* dirty; int* cnt;       // lists; cnt[0] touched, cnt[1] nw, cnt[2] dirty
    const int* active; const int* start;               // active list and its edge prefix sums
    int na; int E; int rs; int cs; int n;
    ll Bd; int Bh, Bv, Bp;
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

struct Prop { int u, v; ll cd; int ch; bool ok; };

// Proposal of edge number e of the active list: candidate key (d[u]+w, hops[u]+1, v, u)
// against the current key (d[v], hops[v], v, pred[v]). The vertex v is the same on both sides.
__device__ inline Prop proposal(const KArgs& a, int e) {
    int lo = 0, hi = a.na;                       // invariant: start[lo] <= e < start[hi]
    while (hi - lo > 1) { int mid = (lo + hi) >> 1; if (a.start[mid] <= e) lo = mid; else hi = mid; }
    Prop r;
    r.u = a.active[lo];
    int idx = a.off[r.u] + (e - a.start[lo]);
    r.v = a.to[idx];
    r.cd = a.d[r.u] + a.w[idx];
    r.ch = a.hops[r.u] + 1;
    r.ok = le3(r.cd, r.ch, r.u, a.d[r.v], a.hops[r.v], a.pred[r.v]);
    return r;
}

__global__ void k_init(KArgs a) {
    int v = TID(); if (v >= a.n) return;
    a.bd[v] = DINF; a.bhp[v] = UINF;
}
__global__ void k_roots(KArgs a) {
    int i = TID(); if (i >= a.na) return;
    int x = a.active[i]; a.root[x] = x;
}
__global__ void k_pass1(KArgs a) {
    int e = TID(); if (e >= a.E) return;
    Prop r = proposal(a, e); if (!r.ok) return;
    atomicMin(&a.bd[r.v], r.cd);
    if (atomicExch(&a.tst[r.v], a.rs) != a.rs) a.touched[atomicAdd(&a.cnt[0], 1)] = r.v;
}
__global__ void k_pass2(KArgs a) {
    int e = TID(); if (e >= a.E) return;
    Prop r = proposal(a, e);
    if (r.ok && r.cd == a.bd[r.v]) atomicMin(&a.bhp[r.v], ((ull)(unsigned)r.ch << 32) | (ull)(unsigned)r.u);
}
// Two kernels so that no thread reads root[] while another writes it.
__global__ void k_commit_a(KArgs a) {
    int i = TID(); if (i >= a.cnt[0]) return;
    int v = a.touched[i]; a.nroot[v] = a.root[(int)(a.bhp[v] & 0xffffffffULL)];
}
__global__ void k_commit_b(KArgs a) {
    int i = TID(); if (i >= a.cnt[0]) return;
    int v = a.touched[i];
    ll nd = a.bd[v]; ull hp = a.bhp[v];
    int nh = (int)(hp >> 32), np = (int)(hp & 0xffffffffULL);
    a.d[v] = nd; a.hops[v] = nh; a.pred[v] = np;
    a.bd[v] = DINF; a.bhp[v] = UINF;
    if (lt4(nd, nh, v, np, a.Bd, a.Bh, a.Bv, a.Bp)) {
        a.root[v] = a.nroot[v];
        int p = atomicAdd(&a.cnt[1], 1); a.nw[2 * p] = v; a.nw[2 * p + 1] = a.nroot[v];
    }
    if (atomicExch(&a.dst[v], a.cs) != a.cs) a.dirty[atomicAdd(&a.cnt[2], 1)] = v;
}
// Packs changed vertices as (v, d, hops << 32 | pred) for a single copy back.
__global__ void k_pack(KArgs a, ll* out, int nd) {
    int i = TID(); if (i >= nd) return;
    int v = a.dirty[i];
    out[3 * i] = v; out[3 * i + 1] = a.d[v];
    out[3 * i + 2] = (ll)(((ull)(unsigned)a.hops[v] << 32) | (ull)(unsigned)a.pred[v]);
}
// Applies host-side changes packed the same way.
__global__ void k_scatter(KArgs a, const ll* in, int nl) {
    int i = TID(); if (i >= nl) return;
    int v = (int)in[3 * i]; ull hp = (ull)in[3 * i + 2];
    a.d[v] = in[3 * i + 1]; a.hops[v] = (int)(hp >> 32); a.pred[v] = (int)(hp & 0xffffffffULL);
}

struct FpGpu {
    int n = 0; int launches = 0;
    int roundstamp = 0, callstamp = 0, hstamp = 0, tstamp = 0;
    KArgs a{};
    std::vector<int> off;                              // host copy of offsets (degrees of active lists)
    std::vector<int> hseen, hroot, htst, hcnt;         // host bookkeeping for W order and pivots
    std::vector<int> act;
    std::vector<std::pair<int, int>> pairs;
    int* d_in = nullptr;                               // per-round input: start[na+1] then active[na]
    ll* d_pack = nullptr;                              // packed vertex records (upload and download)
    int* h_in = nullptr; int* h_nw = nullptr; int* h_cnt = nullptr; ll* h_pack = nullptr;   // pinned staging
    std::vector<void*> pinned;
    std::string name;
};

using Clock = std::chrono::steady_clock;
static double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

FpGpu* fpgpu_create(int n, const int* off, const int* to, const ll* w) {
    FpGpu* g = new FpGpu();
    g->n = n;
    g->name = device_name();
    const int m = off[n];
    g->off.assign(off, off + n + 1);
    g->hseen.assign(n, 0); g->hroot.assign(n, 0); g->htst.assign(n, 0); g->hcnt.assign(n, 0);
    KArgs& a = g->a;
    a.n = n;
    int* doff = (int*)dmalloc((size_t)(n + 1) * 4); h2d(doff, off, (size_t)(n + 1) * 4); a.off = doff;
    int* dto = (int*)dmalloc((size_t)m * 4);        h2d(dto, to, (size_t)m * 4);        a.to = dto;
    ll* dw = (ll*)dmalloc((size_t)m * 8);           h2d(dw, w, (size_t)m * 8);          a.w = dw;
    a.d = (ll*)dmalloc((size_t)n * 8); a.pred = (int*)dmalloc((size_t)n * 4); a.hops = (int*)dmalloc((size_t)n * 4);
    a.root = (int*)dmalloc((size_t)n * 4); a.nroot = (int*)dmalloc((size_t)n * 4);
    a.bd = (ll*)dmalloc((size_t)n * 8); a.bhp = (ull*)dmalloc((size_t)n * 8);
    a.tst = (int*)dmalloc((size_t)n * 4); a.dst = (int*)dmalloc((size_t)n * 4);
    a.touched = (int*)dmalloc((size_t)n * 4); a.nw = (int*)dmalloc((size_t)n * 8); a.dirty = (int*)dmalloc((size_t)n * 4);
    a.cnt = (int*)dmalloc(4 * sizeof(int));
    g->d_in = (int*)dmalloc((size_t)(2 * n + 1) * 4);
    g->d_pack = (ll*)dmalloc((size_t)n * 24);
    g->h_in = (int*)hmalloc((size_t)(2 * n + 1) * 4); g->h_nw = (int*)hmalloc((size_t)n * 8);
    g->h_cnt = (int*)hmalloc(4 * sizeof(int)); g->h_pack = (ll*)hmalloc((size_t)n * 24);
    LAUNCH(k_init, n, a);
    dsync();
    return g;
}

void fpgpu_destroy(FpGpu* g) {
    if (!g) return;
#ifndef FP_EMULATE
    for (void* p : g->pinned) cudaHostUnregister(p);
#endif
    KArgs& a = g->a;
    void* dev[] = {(void*)a.off, (void*)a.to, (void*)a.w, a.d, a.pred, a.hops, a.root, a.nroot, a.bd, a.bhp,
                   a.tst, a.dst, a.touched, a.nw, a.dirty, a.cnt, g->d_in, g->d_pack};
    for (void* p : dev) dfree(p);
    void* host[] = {g->h_in, g->h_nw, g->h_cnt, g->h_pack};
    for (void* p : host) hfree(p);
    delete g;
}

const char* fpgpu_device_name(FpGpu* g) { return g->name.c_str(); }

bool fpgpu_pin_host(FpGpu* g, ll* d, int* pred, int* hops) {
#ifdef FP_EMULATE
    (void)g; (void)d; (void)pred; (void)hops;
    return false;
#else
    void* ptr[] = {d, pred, hops};
    size_t sz[] = {(size_t)g->n * 8, (size_t)g->n * 4, (size_t)g->n * 4};
    for (int i = 0; i < 3; i++) {
        if (cudaHostRegister(ptr[i], sz[i], cudaHostRegisterDefault) != cudaSuccess) { cudaGetLastError(); return false; }
        g->pinned.push_back(ptr[i]);
    }
    return true;
#endif
}

void fpgpu_upload_state(FpGpu* g, const ll* d, const int* pred, const int* hops, const int* list, int nlist,
                        FpGpuStats* st) {
    auto t0 = Clock::now();
    KArgs& a = g->a;
    const int before = g->launches;
    if (list == nullptr) {
        h2d(a.d, d, (size_t)g->n * 8); h2d(a.pred, pred, (size_t)g->n * 4); h2d(a.hops, hops, (size_t)g->n * 4);
        st->upload_vertices += g->n;
    } else if (nlist > 0) {
        for (int i = 0; i < nlist; i++) {
            int v = list[i];
            g->h_pack[3 * i] = v; g->h_pack[3 * i + 1] = d[v];
            g->h_pack[3 * i + 2] = (ll)(((ull)(unsigned)hops[v] << 32) | (ull)(unsigned)pred[v]);
        }
        h2d(g->d_pack, g->h_pack, (size_t)nlist * 24);
        LAUNCH(k_scatter, nlist, a, g->d_pack, nlist);
        dsync();
        st->upload_vertices += nlist;
    }
    st->launches += g->launches - before;
    st->upload_ms += ms_since(t0);
}

void fpgpu_findpivots(FpGpu* g, ll* d, int* pred, int* hops, int k, ll Bd, int Bh, int Bv, int Bp,
                      const std::vector<int>& S, std::vector<int>& P, std::vector<int>& W, FpGpuStats* st) {
    const int before = g->launches;
    auto t0 = Clock::now();
    KArgs a = g->a;
    a.Bd = Bd; a.Bh = Bh; a.Bv = Bv; a.Bp = Bp;
    a.cs = ++g->callstamp;
    const int stamp = ++g->hstamp;
    W.assign(S.begin(), S.end());
    for (int x : S) { g->hseen[x] = stamp; g->hroot[x] = x; }
    g->act.assign(S.begin(), S.end());
    double host = 0, xfer = 0, kern = 0;
    auto th = Clock::now();
    dzero(a.cnt, 3 * sizeof(int));
    xfer += ms_since(th);
    bool early = false;
    int nd = 0;
    for (int round = 1; round <= k && !early; round++) {
        th = Clock::now();
        const int na = (int)g->act.size();
        long long E = 0;
        int* hs = g->h_in;                 // start[0..na], then active[0..na-1]
        hs[0] = 0;
        for (int i = 0; i < na; i++) {
            int u = g->act[i]; hs[na + 1 + i] = u;
            E += g->off[u + 1] - g->off[u];
            if (E > INT_MAX) { std::fprintf(stderr, "fpgpu: round edge count exceeds int range\n"); std::exit(4); }
            hs[i + 1] = (int)E;
        }
        host += ms_since(th);
        th = Clock::now();
        h2d(g->d_in, hs, (size_t)(2 * na + 1) * 4);
        if (round > 1) dzero(a.cnt, 2 * sizeof(int));
        xfer += ms_since(th);
        a.start = g->d_in; a.active = g->d_in + na + 1;
        a.na = na; a.E = (int)E; a.rs = ++g->roundstamp;
        th = Clock::now();
        if (round == 1) LAUNCH(k_roots, na, a);
        LAUNCH(k_pass1, a.E, a);
        LAUNCH(k_pass2, a.E, a);
        const int nl = (int)std::min<long long>(E, g->n);
        LAUNCH(k_commit_a, nl, a);
        LAUNCH(k_commit_b, nl, a);
        dsync();
        kern += ms_since(th);
        th = Clock::now();
        d2h(g->h_cnt, a.cnt, 3 * sizeof(int));          // nw count and the running dirty count
        const int nnw = g->h_cnt[1];
        nd = g->h_cnt[2];
        d2h(g->h_nw, a.nw, (size_t)nnw * 8);
        xfer += ms_since(th);
        st->edges += E; st->rounds++;
        th = Clock::now();
        // Same bookkeeping as the serial version: new vertices in increasing id order.
        g->pairs.resize(nnw);
        for (int i = 0; i < nnw; i++) g->pairs[i] = {g->h_nw[2 * i], g->h_nw[2 * i + 1]};
        std::sort(g->pairs.begin(), g->pairs.end());
        g->act.clear();
        for (auto& pr : g->pairs) {
            int v = pr.first;
            g->hroot[v] = pr.second;
            g->act.push_back(v);
            if (g->hseen[v] != stamp) { g->hseen[v] = stamp; W.push_back(v); }
        }
        if (W.size() > (size_t)k * S.size()) early = true;
        host += ms_since(th);
    }
    th = Clock::now();
    P.clear();
    if (early) {
        P = S;
    } else {
        const int ts = ++g->tstamp;
        for (int u : W) {
            int r = g->hroot[u];
            if (g->htst[r] != ts) { g->htst[r] = ts; g->hcnt[r] = 0; }
            g->hcnt[r]++;
        }
        for (int u : S) if (g->htst[u] == ts && g->hcnt[u] >= k) P.push_back(u);
    }
    host += ms_since(th);
    st->rounds_ms += ms_since(t0);
    st->host_ms += host; st->xfer_ms += xfer; st->kernel_ms += kern;

    auto t1 = Clock::now();
    if (nd > 0) {
        LAUNCH(k_pack, nd, a, g->d_pack, nd);
        d2h(g->h_pack, g->d_pack, (size_t)nd * 24);
        for (int i = 0; i < nd; i++) {
            int v = (int)g->h_pack[3 * i]; ull hp = (ull)g->h_pack[3 * i + 2];
            d[v] = g->h_pack[3 * i + 1]; hops[v] = (int)(hp >> 32); pred[v] = (int)(hp & 0xffffffffULL);
        }
    }
    st->download_vertices += nd;
    st->download_ms += ms_since(t1);
    st->launches += g->launches - before;
}
