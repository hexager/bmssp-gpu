// BMSSP hybrid driver: CPU recursion (Castro et al. reference, cpu-v1.4 + dirty tracking) with
// FindPivots calls above an edge-work threshold sent to the GPU (fp_gpu.cu).
//
//   FP_MODE=builtin   no hook: every call uses the built-in in-place FindPivots (CPU reference)
//   FP_MODE=sync      every call uses the serial synchronous-round FindPivots (port of OpenMP v3, 1 thread)
//   FP_MODE=gpu       calls with first-round edge work >= FP_GPU_MIN_EDGES go to the GPU, others built-in
//   FP_MODE=check     as gpu, and every GPU call is compared with the serial synchronous version on the same
//                     state: P, W (order included) and all of d/pred/path_sz must be identical. Exit 3 if not.
//   FP_GPU_MIN_EDGES  threshold (default 4096; 0 sends every call to the GPU)
//   FP_GPU_DIRTY      1 (default): upload only vertices the CPU changed since the last GPU call; 0: full upload
//   FP_GPU_PIN        1 (default): page-lock the solver state arrays
//   FP_CALL_LOG       per-call CSV (refuses to overwrite)
//
// usage: fp_hybrid GRAPH.gr SOURCE  > distances.txt   (stdout: one distance per line, -1 if unreachable)
#include <algorithm>
#include <chrono>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include "bmssp_hybrid.hpp"
#include "fp_gpu.h"
#include "sync_fp.hpp"

using Clock = std::chrono::steady_clock;
static double ms_since(Clock::time_point a) { return std::chrono::duration<double, std::milli>(Clock::now() - a).count(); }

static long long env_ll(const char* name, long long dflt, long long lo, long long hi) {
    const char* s = std::getenv(name);
    if (!s) return dflt;
    char* end = nullptr; errno = 0; long long x = std::strtoll(s, &end, 10);
    if (end == s || *end != '\0' || errno == ERANGE || x < lo || x > hi) {
        std::cerr << "ERROR: " << name << "='" << s << "' invalid\n"; std::exit(2);
    }
    return x;
}

int main(int argc, char** argv) {
    if (argc < 3) { std::cerr << "usage: fp_hybrid GRAPH.gr SOURCE\n"; return 2; }
    int n = 0; std::vector<std::tuple<int, int, T>> edges;
    {
        std::ifstream in(argv[1]); if (!in) { std::cerr << "cannot open " << argv[1] << "\n"; return 2; }
        std::string line;
        while (std::getline(in, line)) {
            if (line.empty()) continue;
            std::istringstream ss(line); char c; ss >> c;
            if (c == 'p') { std::string sp; long long m; ss >> sp >> n >> m; }
            else if (c == 'a') { int u, v; long long w; ss >> u >> v >> w; edges.emplace_back(u - 1, v - 1, w); }
        }
    }
    const int s = std::atoi(argv[2]) - 1;
    const std::string mode = std::getenv("FP_MODE") ? std::getenv("FP_MODE") : "gpu";
    if (mode != "builtin" && mode != "sync" && mode != "gpu" && mode != "check") {
        std::cerr << "ERROR: FP_MODE must be builtin|sync|gpu|check\n"; return 2;
    }
    const long long thr = env_ll("FP_GPU_MIN_EDGES", 4096, 0, 1000000000000LL);
    const bool dirty = env_ll("FP_GPU_DIRTY", 1, 0, 1) == 1;
    const bool pin = env_ll("FP_GPU_PIN", 1, 0, 1) == 1;
    const bool use_gpu = (mode == "gpu" || mode == "check");

    spp::bmssp<T> solver(n);
    for (auto& [u, v, w] : edges) solver.addEdge(u, v, w);
    edges.clear(); edges.shrink_to_fit();
    auto tp = Clock::now();
    solver.prepare_graph(false);
    const double prepare_ms = ms_since(tp);
    auto st0 = solver.fp_state();

    FpGpu* gpu = nullptr; bool pinned = false; double gpu_setup_ms = 0;
    if (use_gpu) {
        auto tg = Clock::now();
        gpu = fpgpu_create(st0.n, st0.csr_off, st0.csr_to, st0.csr_w);
        if (pin) pinned = fpgpu_pin_host(gpu, st0.d, st0.pred, st0.path_sz);
        gpu_setup_ms = ms_since(tg);
        if (dirty) solver.enable_dirty_tracking();
    }

    SyncFP sync(n);
    struct CallRec { int path; long long work0, S, W; double ms; FpGpuStats g; };
    std::vector<CallRec> log;
    const char* log_path = std::getenv("FP_CALL_LOG");
    if (log_path) { std::ifstream t(log_path); if (t) { std::cerr << "ERROR: FP_CALL_LOG exists\n"; return 2; } }
    long long calls_builtin = 0, calls_gpu = 0, calls_sync = 0, checked = 0;
    FpGpuStats tot;
    bool need_full = true;
    std::vector<T> snap_d, ref_d; std::vector<int> snap_p, snap_h, ref_p, ref_h;

    auto gpu_call = [&](auto& st, const Key& Bk, const std::vector<int>& S, FpGpuStats& gs) {
        if (need_full || !dirty) {
            fpgpu_upload_state(gpu, st.d, st.pred, st.path_sz, nullptr, 0, &gs);
            need_full = false;
        } else if (!solver.dirty_list.empty()) {   // empty list: nothing to upload (data() may be nullptr = "all")
            fpgpu_upload_state(gpu, st.d, st.pred, st.path_sz, solver.dirty_list.data(), (int)solver.dirty_list.size(), &gs);
        }
        if (dirty) solver.clear_dirty();
        std::vector<int> P, W;
        fpgpu_findpivots(gpu, st.d, st.pred, st.path_sz, st.k, std::get<0>(Bk), std::get<1>(Bk), std::get<2>(Bk),
                         std::get<3>(Bk), S, P, W, &gs);
        return std::make_pair(P, W);
    };

    if (mode != "builtin") {
        solver.findPivotsImpl = [&](auto B, const std::vector<int>& S) -> std::pair<std::vector<int>, std::vector<int>> {
            auto t = Clock::now();
            auto st = solver.fp_state();
            const Key Bk = static_cast<const Key&>(B);
            long long w0 = 0; for (int u : S) w0 += st.csr_off[u + 1] - st.csr_off[u];
            std::pair<std::vector<int>, std::vector<int>> r;
            FpGpuStats gs; int path = 0;
            if (mode == "sync") {
                r = sync.run(st, Bk, S); path = 2; calls_sync++;
            } else if (w0 < thr) {
                r = solver.findPivots(B, S); path = 0; calls_builtin++;
            } else if (mode == "gpu") {
                r = gpu_call(st, Bk, S, gs); path = 1; calls_gpu++;
            } else {   // check
                snap_d.assign(st.d, st.d + st.n); snap_p.assign(st.pred, st.pred + st.n); snap_h.assign(st.path_sz, st.path_sz + st.n);
                auto ref = sync.run(st, Bk, S);
                ref_d.assign(st.d, st.d + st.n); ref_p.assign(st.pred, st.pred + st.n); ref_h.assign(st.path_sz, st.path_sz + st.n);
                std::copy(snap_d.begin(), snap_d.end(), st.d); std::copy(snap_p.begin(), snap_p.end(), st.pred);
                std::copy(snap_h.begin(), snap_h.end(), st.path_sz);
                r = gpu_call(st, Bk, S, gs); path = 1; calls_gpu++;
                long long bad_state = 0;
                for (int v = 0; v < st.n; v++)
                    bad_state += (st.d[v] != ref_d[v]) || (st.pred[v] != ref_p[v]) || (st.path_sz[v] != ref_h[v]);
                if (r.first != ref.first || r.second != ref.second || bad_state) {
                    std::cerr << "CHECK FAIL call=" << (calls_builtin + calls_gpu) << " |S|=" << S.size()
                              << " P " << r.first.size() << " vs " << ref.first.size()
                              << " W " << r.second.size() << " vs " << ref.second.size()
                              << " W_equal=" << (r.second == ref.second) << " P_equal=" << (r.first == ref.first)
                              << " state_mismatch_vertices=" << bad_state << "\n";
                    std::exit(3);
                }
                checked++;
            }
            const double ms = ms_since(t);
            tot.upload_ms += gs.upload_ms; tot.rounds_ms += gs.rounds_ms; tot.download_ms += gs.download_ms;
            tot.kernel_ms += gs.kernel_ms; tot.xfer_ms += gs.xfer_ms; tot.host_ms += gs.host_ms;
            tot.upload_vertices += gs.upload_vertices; tot.download_vertices += gs.download_vertices;
            tot.edges += gs.edges; tot.launches += gs.launches; tot.rounds += gs.rounds;
            if (log_path) log.push_back({path, w0, (long long)S.size(), (long long)r.second.size(), ms, gs});
            return r;
        };
    }

    auto te = Clock::now();
    auto res = solver.execute(s);
    const double exec_ms = ms_since(te);

    std::string out; out.reserve((size_t)n * 8);
    char buf[32];
    for (int i = 0; i < n; i++) {
        long long x = res.first[i] == solver.oo ? -1LL : (long long)res.first[i];
        int len = std::snprintf(buf, sizeof buf, "%lld\n", x); out.append(buf, len);
    }
    std::fwrite(out.data(), 1, out.size(), stdout);

    if (log_path) {
        std::ofstream lf(log_path);
        lf << "call,path,work0,S,W,ms,upload_ms,rounds_ms,download_ms,kernel_ms,xfer_ms,host_ms,upload_vertices,download_vertices,edges,launches,rounds\n";
        for (size_t i = 0; i < log.size(); i++) {
            const auto& c = log[i];
            lf << i << ',' << c.path << ',' << c.work0 << ',' << c.S << ',' << c.W << ',' << c.ms << ',' << c.g.upload_ms << ','
               << c.g.rounds_ms << ',' << c.g.download_ms << ',' << c.g.kernel_ms << ',' << c.g.xfer_ms << ',' << c.g.host_ms << ','
               << c.g.upload_vertices << ',' << c.g.download_vertices << ','
               << c.g.edges << ',' << c.g.launches << ',' << c.g.rounds << '\n';
        }
    }
    std::cerr << "HYBRID mode=" << mode << " min_edges=" << thr << " dirty=" << dirty << " pinned=" << pinned
              << " device=\"" << (gpu ? fpgpu_device_name(gpu) : "none") << "\"\n"
              << "HYBRID execute_ms=" << exec_ms << " prepare_ms=" << prepare_ms << " gpu_setup_ms=" << gpu_setup_ms << "\n"
              << "HYBRID calls_builtin=" << calls_builtin << " calls_gpu=" << calls_gpu << " calls_sync=" << calls_sync
              << " checked=" << checked << "\n"
              << "HYBRID gpu_upload_ms=" << tot.upload_ms << " gpu_rounds_ms=" << tot.rounds_ms
              << " gpu_download_ms=" << tot.download_ms << " gpu_kernel_ms=" << tot.kernel_ms << " gpu_xfer_ms=" << tot.xfer_ms
              << " gpu_host_ms=" << tot.host_ms << " upload_vertices=" << tot.upload_vertices
              << " download_vertices=" << tot.download_vertices << " gpu_edges=" << tot.edges
              << " launches=" << tot.launches << " rounds=" << tot.rounds << "\n";
    fpgpu_destroy(gpu);
    return 0;
}
