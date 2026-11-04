// Several single-source searches on one graph at the same time, one CPU thread per source.
//
//   MULTI_MODE=builtin     BMSSP per thread, built-in FindPivots, no hook (CPU reference)
//   MULTI_MODE=cpu_hooked  BMSSP per thread, hook and dirty tracking on, every FindPivots call on the CPU
//                          (same process setup as gpu, so it is the control for the GPU effect)
//   MULTI_MODE=gpu         BMSSP per thread; FindPivots calls with first-round edge work >= FP_GPU_MIN_EDGES go to
//                          one GPU worker thread, which runs all waiting calls as one batch (fp_batch.cu)
//   MULTI_MODE=check       as gpu, and in every batch each call is first run with the serial synchronous version on a
//                          copy of the same state; P, W (order included) and all of d/pred/path_sz must be equal (exit 3)
//   MULTI_MODE=dijkstra    binary-heap Dijkstra per thread (CPU baseline)
//   FP_GPU_MIN_EDGES       threshold (default 0: every call to the GPU)
//   MULTI_BATCH=greedy     (default) the worker takes every call that is waiting when the GPU becomes free
//   MULTI_BATCH=all        the worker waits until every unfinished search is waiting, then runs them together
//   MULTI_CORES=c          after the solvers are built, run all threads on c physical cores (one hardware thread
//                          per core); with more sources than cores, a search waiting for the GPU frees its core
//   MULTI_WORKER_CORE=1    with MULTI_CORES=c: the GPU worker thread gets one more physical core of its own
//   FP_BATCH_SYNC=spin     the GPU worker spin-waits on the GPU instead of sleeping (default: block)
//
// usage: fp_multi GRAPH.gr SRC1,SRC2,...    stdout: the distances of each source in order (-1 unreachable)
// stderr: MULTI line with wall_ms (start of the first search to the end of the last), batches, calls and timings
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <mutex>
#include <set>
#include <sched.h>
#include <unistd.h>
#include <queue>
#include <sstream>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#include "bmssp_hybrid.hpp"
#include "fp_batch.h"
#include "sync_fp.hpp"

using Clock = std::chrono::steady_clock;
static double ms_between(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); }

struct Pending { FpbReq req; void* st; bool done = false; };

// Restricts the calling thread (and the threads it creates later) to the first hardware thread of `cores` distinct
// physical cores. With `extra`, one more core is chosen but not used here (it is returned last, for the worker).
// Returns the chosen CPU ids.
static void pin_thread(const std::vector<int>& cpus) {
    cpu_set_t set; CPU_ZERO(&set); for (int c : cpus) CPU_SET(c, &set);
    if (sched_setaffinity(0, sizeof set, &set) != 0) { std::cerr << "ERROR: sched_setaffinity failed\n"; std::exit(2); }
}
static std::vector<int> pin_to_cores(int cores, bool extra = false) {
    const int want = cores + (extra ? 1 : 0);
    std::vector<int> chosen; std::set<std::pair<int, int>> seen;
    const long ncpu = sysconf(_SC_NPROCESSORS_ONLN);
    for (int c = 0; c < ncpu && (int)chosen.size() < want; c++) {
        auto rd = [&](const char* f) { std::ifstream in("/sys/devices/system/cpu/cpu" + std::to_string(c) + "/topology/" + f);
                                       int x = -1; in >> x; return x; };
        std::pair<int, int> key(rd("physical_package_id"), rd("core_id"));
        if (key.second < 0) key = {0, c};
        if (seen.insert(key).second) chosen.push_back(c);
    }
    if ((int)chosen.size() < want) { std::cerr << "ERROR: cannot pin to " << want << " physical cores\n"; std::exit(2); }
    pin_thread(std::vector<int>(chosen.begin(), chosen.begin() + cores));
    return chosen;
}

int main(int argc, char** argv) {
    if (argc < 3) { std::cerr << "usage: fp_multi GRAPH.gr SRC1,SRC2,...\n"; return 2; }
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
    std::vector<int> src;
    { std::stringstream ss(argv[2]); std::string t; while (std::getline(ss, t, ',')) src.push_back(std::atoi(t.c_str()) - 1); }
    const int B = (int)src.size();
    const std::string mode = std::getenv("MULTI_MODE") ? std::getenv("MULTI_MODE") : "gpu";
    const std::string policy = std::getenv("MULTI_BATCH") ? std::getenv("MULTI_BATCH") : "greedy";
    const long long thr = std::getenv("FP_GPU_MIN_EDGES") ? std::atoll(std::getenv("FP_GPU_MIN_EDGES")) : 0;
    if (mode != "builtin" && mode != "cpu_hooked" && mode != "gpu" && mode != "check" && mode != "dijkstra") {
        std::cerr << "ERROR: MULTI_MODE must be builtin|cpu_hooked|gpu|check|dijkstra\n"; return 2;
    }
    if (policy != "greedy" && policy != "all") { std::cerr << "ERROR: MULTI_BATCH must be greedy|all\n"; return 2; }
    const bool use_gpu = (mode == "gpu" || mode == "check");

    std::vector<std::vector<T>> dist(B);
    std::vector<double> run_ms(B, 0);
    Clock::time_point t_start, t_end;

    // CSR for Dijkstra and the GPU (the solvers build their own).
    std::vector<int> off(n + 1, 0), to(edges.size()); std::vector<T> wt(edges.size());
    for (auto& [u, v, w] : edges) off[u + 1]++;
    for (int i = 0; i < n; i++) off[i + 1] += off[i];
    { std::vector<int> pos(off.begin(), off.end() - 1); for (auto& [u, v, w] : edges) { int p = pos[u]++; to[p] = v; wt[p] = w; } }

    if (mode == "dijkstra") {
        std::atomic<int> ready{0};
        std::vector<std::thread> th;
        t_start = Clock::now();
        for (int i = 0; i < B; i++) th.emplace_back([&, i]() {
            auto t0 = Clock::now();
            auto& d = dist[i]; d.assign(n, -1);
            using QE = std::pair<T, int>;
            std::priority_queue<QE, std::vector<QE>, std::greater<QE>> pq;
            d[src[i]] = 0; pq.push({0, src[i]});
            while (!pq.empty()) {
                auto [du, u] = pq.top(); pq.pop();
                if (du != d[u]) continue;
                for (int e = off[u]; e < off[u + 1]; e++) {
                    T nd = du + wt[e]; int v = to[e];
                    if (d[v] == -1 || nd < d[v]) { d[v] = nd; pq.push({nd, v}); }
                }
            }
            run_ms[i] = ms_between(t0, Clock::now());
        });
        for (auto& t : th) t.join();
        t_end = Clock::now();
    } else {
        // Build one solver per source in parallel (not timed).
        std::vector<spp::bmssp<T>*> solver(B, nullptr);
        { std::vector<std::thread> th;
          for (int i = 0; i < B; i++) th.emplace_back([&, i]() {
              solver[i] = new spp::bmssp<T>(n);
              for (auto& [u, v, w] : edges) solver[i]->addEdge(u, v, w);
              solver[i]->prepare_graph(false);
              solver[i]->fp_state();                 // builds the CSR copy used by the hook
          });
          for (auto& t : th) t.join(); }

        std::string pinned = "none"; int worker_cpu = -1;
        if (const char* mc = std::getenv("MULTI_CORES")) {
            const bool own = std::getenv("MULTI_WORKER_CORE") && std::string(std::getenv("MULTI_WORKER_CORE")) == "1";
            auto ids = pin_to_cores(std::atoi(mc), own); pinned.clear();
            if (own) { worker_cpu = ids.back(); ids.pop_back(); }
            for (size_t j = 0; j < ids.size(); j++) pinned += (j ? "+" : "") + std::to_string(ids[j]);
            if (own) pinned += "_worker" + std::to_string(worker_cpu);
        }
        FpBatch* gpu = nullptr; double gpu_setup_ms = 0;
        if (use_gpu) { auto tg = Clock::now(); gpu = fpb_create(n, B, off.data(), to.data(), wt.data()); gpu_setup_ms = ms_between(tg, Clock::now()); }
        (void)gpu_setup_ms;

        std::mutex mu; std::condition_variable cv_work, cv_done;
        std::vector<Pending*> queue; int active = B; bool stop = false;
        FpbStats gst; long long checked = 0, calls_cpu = 0, calls_gpu = 0, max_batch = 0;
        double wait_ms_total = 0;
        std::vector<long long> batch_hist(B + 1, 0);

        using FPState = decltype(solver[0]->fp_state());
        std::thread worker;
        if (use_gpu) worker = std::thread([&]() {
            if (worker_cpu >= 0) pin_thread({worker_cpu});
            std::vector<SyncFP> sync; if (mode == "check") for (int i = 0; i < B; i++) sync.emplace_back(n);
            const int k = solver[0]->fp_state().k;
            while (true) {
                std::vector<Pending*> batch;
                {
                    std::unique_lock<std::mutex> lk(mu);
                    cv_work.wait(lk, [&]() {
                        if (stop) return true;
                        if (queue.empty()) return false;
                        return policy == "greedy" || (int)queue.size() >= active;
                    });
                    if (stop && queue.empty()) return;
                    batch.swap(queue);
                }
                std::vector<FpbReq> reqs; for (auto* p : batch) reqs.push_back(p->req);
                // check mode: reference result of every call, computed on a copy of the same input state
                const size_t nb = batch.size();
                std::vector<std::pair<std::vector<int>, std::vector<int>>> ref(nb);
                std::vector<std::vector<T>> rd(nb); std::vector<std::vector<int>> rp(nb), rh(nb);
                if (mode == "check") {
                    for (size_t j = 0; j < nb; j++) {
                        auto& r = reqs[j]; FPState& st = *(FPState*)batch[j]->st;
                        std::vector<T> sd(st.d, st.d + n); std::vector<int> sp(st.pred, st.pred + n), sh(st.path_sz, st.path_sz + n);
                        ref[j] = sync[r.search].run(st, Key(r.Bd, r.Bh, r.Bv, r.Bp), *r.S);
                        rd[j].assign(st.d, st.d + n); rp[j].assign(st.pred, st.pred + n); rh[j].assign(st.path_sz, st.path_sz + n);
                        std::copy(sd.begin(), sd.end(), st.d); std::copy(sp.begin(), sp.end(), st.pred);
                        std::copy(sh.begin(), sh.end(), st.path_sz);
                    }
                }
                fpb_run(gpu, reqs, k, &gst);
                if (mode == "check") {
                    for (size_t j = 0; j < nb; j++) {
                        auto& r = reqs[j]; FPState& st = *(FPState*)batch[j]->st;
                        long long bad = 0;
                        for (int v = 0; v < n; v++) bad += (st.d[v] != rd[j][v]) || (st.pred[v] != rp[j][v]) || (st.path_sz[v] != rh[j][v]);
                        if (*r.P != ref[j].first || *r.W != ref[j].second || bad) {
                            std::cerr << "CHECK FAIL search=" << r.search << " batch_size=" << nb << " |S|=" << r.S->size()
                                      << " P_equal=" << (*r.P == ref[j].first) << " W_equal=" << (*r.W == ref[j].second)
                                      << " state_mismatch_vertices=" << bad << "\n";
                            std::exit(3);
                        }
                        checked++;
                    }
                }
                {
                    std::lock_guard<std::mutex> lk(mu);
                    max_batch = std::max<long long>(max_batch, (long long)nb);
                    batch_hist[std::min<size_t>(nb, B)]++;
                    for (auto* p : batch) p->done = true;
                }
                cv_done.notify_all();
            }
        });
        (void)checked;

        std::atomic<int> started{0};
        std::vector<std::thread> th;
        std::vector<Clock::time_point> t0(B), t1(B);
        std::mutex start_mu; std::condition_variable start_cv; bool go = false;
        for (int i = 0; i < B; i++) th.emplace_back([&, i]() {
            auto& sv = *solver[i];
            if (mode != "builtin") {
                sv.enable_dirty_tracking();
                sv.findPivotsImpl = [&, i](auto Bk_, const std::vector<int>& S) -> std::pair<std::vector<int>, std::vector<int>> {
                    auto st = sv.fp_state();
                    long long w0 = 0; for (int u : S) w0 += st.csr_off[u + 1] - st.csr_off[u];
                    if (!use_gpu || w0 < thr) {
                        { std::lock_guard<std::mutex> lk(mu); calls_cpu++; }
                        return sv.findPivots(Bk_, S);
                    }
                    const Key Bk = static_cast<const Key&>(Bk_);
                    std::vector<int> P, W;
                    Pending pd;
                    pd.req = FpbReq{i, st.d, st.pred, st.path_sz, sv.dirty_list.data(), (int)sv.dirty_list.size(),
                                    std::get<0>(Bk), std::get<1>(Bk), std::get<2>(Bk), std::get<3>(Bk), &S, &P, &W};
                    pd.st = &st;
                    auto tw = Clock::now();
                    {
                        std::unique_lock<std::mutex> lk(mu);
                        calls_gpu++;
                        queue.push_back(&pd);
                        cv_work.notify_one();
                        cv_done.wait(lk, [&]() { return pd.done; });
                        wait_ms_total += ms_between(tw, Clock::now());
                    }
                    sv.clear_dirty();
                    return {P, W};
                };
            }
            { std::unique_lock<std::mutex> lk(start_mu); started++; start_cv.notify_all(); start_cv.wait(lk, [&]() { return go; }); }
            t0[i] = Clock::now();
            auto res = sv.execute(src[i]);
            t1[i] = Clock::now();
            dist[i].resize(n);
            for (int v = 0; v < n; v++) dist[i][v] = res.first[v] == sv.oo ? -1 : (T)res.first[v];
            { std::lock_guard<std::mutex> lk(mu); active--; }
            cv_work.notify_one();
        });
        { std::unique_lock<std::mutex> lk(start_mu); start_cv.wait(lk, [&]() { return started == B; }); go = true; }
        start_cv.notify_all();
        for (auto& t : th) t.join();
        t_start = *std::min_element(t0.begin(), t0.end()); t_end = *std::max_element(t1.begin(), t1.end());
        for (int i = 0; i < B; i++) run_ms[i] = ms_between(t0[i], t1[i]);
        if (use_gpu) { { std::lock_guard<std::mutex> lk(mu); stop = true; } cv_work.notify_one(); worker.join(); }
        std::cerr << "MULTI_GPU device=\"" << (gpu ? fpb_device_name(gpu) : "none") << "\" calls_cpu=" << calls_cpu
                  << " calls_gpu=" << calls_gpu << " checked=" << checked << " batches=" << gst.batches << " max_batch=" << max_batch
                  << " upload_ms=" << gst.upload_ms << " rounds_ms=" << gst.rounds_ms << " download_ms=" << gst.download_ms
                  << " wait_ms=" << wait_ms_total << " cpus=" << pinned << " edges=" << gst.edges << " launches=" << gst.launches << " hist=";
        for (int b = 1; b <= B; b++) std::cerr << batch_hist[b] << (b < B ? "," : "");
        std::cerr << "\n";
        fpb_destroy(gpu);
    }

    std::string out; char buf[32];
    for (int i = 0; i < B; i++) for (T x : dist[i]) { int len = std::snprintf(buf, sizeof buf, "%lld\n", (long long)x); out.append(buf, len); }
    std::fwrite(out.data(), 1, out.size(), stdout);
    double sum = 0; for (double x : run_ms) sum += x;
    std::cerr << "MULTI mode=" << mode << " batch=" << policy << " min_edges=" << thr << " sources=" << B
              << " wall_ms=" << ms_between(t_start, t_end) << " sum_ms=" << sum << "\n";
    return 0;
}
