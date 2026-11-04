// Serial synchronous-round FindPivots: the correctness reference for the GPU versions.
// Same operations as the serial path of fp_openmp_v3.cpp (Aniketh).
#pragma once
#include <algorithm>
#include <tuple>
#include <vector>

using T = long long;
using Key = std::tuple<T, int, int, int>;   // (dist, hops, vertex, pred)

struct SyncFP {
    std::vector<int> root, seen, bseen, treesz, vis, active, nw, touched;
    struct Prop { int v; T d; int hops; int u; int root_u; };
    std::vector<Prop> best, props;
    int stamp = 0, bstamp = 0;
    explicit SyncFP(int n) : root(n, 0), seen(n, 0), bseen(n, 0), treesz(n, 0), best(n) {}

    template <class St>
    std::pair<std::vector<int>, std::vector<int>> run(St& st, const Key& Bk, const std::vector<int>& S) {
        const int k = st.k;
        ++stamp;
        vis.clear(); active.assign(S.begin(), S.end());
        for (int x : S) { vis.push_back(x); seen[x] = stamp; root[x] = x; treesz[x] = 0; }
        bool early = false;
        for (int round = 1; round <= k && !early; round++) {
            props.clear();
            for (int u : active)
                for (int e = st.csr_off[u]; e < st.csr_off[u + 1]; e++) {
                    int v = st.csr_to[e];
                    Key cand(st.d[u] + st.csr_w[e], st.path_sz[u] + 1, v, u);
                    Key cur(st.d[v], st.path_sz[v], v, st.pred[v]);
                    if (cand <= cur) props.push_back({v, std::get<0>(cand), std::get<1>(cand), u, root[u]});
                }
            ++bstamp; touched.clear();
            for (const Prop& p : props) {
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
            if (vis.size() > (size_t)k * S.size()) early = true; else active.swap(nw);
        }
        std::vector<int> P;
        if (early) P = S;
        else {
            for (int u : vis) treesz[root[u]]++;
            for (int u : S) if (treesz[u] >= k) P.push_back(u);
        }
        return {P, vis};
    }
};

