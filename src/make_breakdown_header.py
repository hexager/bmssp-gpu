# Builds bmssp_breakdown.hpp from bmssp_hybrid.hpp by recording every operation on the data structure D
# (initialize, insert, pull, erase, batchPrepend) in a trace, and adding replay_D(), which replays the trace
# on fresh D objects and times each kind of operation. Recording is off unless dtrace_on is set.
# Only the first solver class is patched. Each anchor must occur exactly once in that class.
import sys

src, dst = sys.argv[1], sys.argv[2]
s = open(src).read()
cut = s.find("class bmssp {", s.find("class bmssp {") + 1)
head, tail = (s[:cut], s[cut:]) if cut != -1 else (s, "")

R = "if(dtrace_on) dtrace.push_back"
edits = [
    ("        D.initialize(batch_size, B);\n",
     "        D.initialize(batch_size, B); " + R + "({'I', (short)(l - 1), B, (long long)batch_size});\n"),
    ("        for(int p: P) D.insert(getDist(p));\n",
     "        for(int p: P) { D.insert(getDist(p)); " + R + "({'i', (short)(l - 1), getDist(p), 0}); }\n"),
    ("            auto [trying_B, miniS] = D.pull();\n",
     "            auto [trying_B, miniS] = D.pull(); " + R + "({'L', (short)(l - 1), trying_B, pull_hash(miniS)});\n"),
    ("                D.erase(u); // priority queue fix\n",
     "                D.erase(u); " + R + "({'E', (short)(l - 1), {}, u}); // priority queue fix\n"),
    ("                            D.insert(new_dist); // d[v]",
     "                            D.insert(new_dist); " + R + "({'R', (short)(l - 1), new_dist, 0}); // d[v]"),
    ("            D.batchPrepend(can_prepend);\n",
     "            D.batchPrepend(can_prepend);\n"
     "            if(dtrace_on) { dtrace.push_back({'P', (short)(l - 1), {}, (long long)can_prepend.size()});\n"
     "                            for(auto &x: can_prepend) dtrace.push_back({'p', (short)(l - 1), x, 0}); }\n"),
]
for old, new in edits:
    assert head.count(old) == 1, "anchor not unique in first class: " + old[:60]
    head = head.replace(old, new)

anchor = "    std::vector<batchPQ<uniqueDistT>> Ds;\n"
assert head.count(anchor) == 1
head = head.replace(anchor, anchor + r'''
    // breakdown: trace of D operations and its replay
    struct DOp { char op; short lvl; uniqueDistT x; long long a; };
    bool dtrace_on = false; std::vector<DOp> dtrace;
    static long long pull_hash(const std::vector<int> &v) {   // order-sensitive hash of a pulled batch
        unsigned long long h = 1469598103934665603ull ^ v.size();
        for(int x: v) { h ^= (unsigned)x; h *= 1099511628211ull; }
        return (long long)(h >> 1);
    }
    struct DReplay { double init_ms = 0, insert_pivots_ms = 0, pull_ms = 0, relax_ms = 0, prepend_ms = 0;
                     long long ops = 0, pull_mismatch = 0; };
    // Replays the trace on fresh D objects. Times runs of consecutive operations of one kind, not single
    // operations, so the timer cost is per run. Relax = the erase and insert calls of the relax loop.
    DReplay replay_D() {
        DReplay r; std::vector<batchPQ<uniqueDistT>> R(Ds.size(), (int)adj.size());
        std::vector<std::vector<uniqueDistT>> prep;              // batchPrepend arguments, built before timing
        for(size_t i = 0; i < dtrace.size(); i++) if(dtrace[i].op == 'P') {
            std::vector<uniqueDistT> v; v.reserve(dtrace[i].a);
            for(long long j = 1; j <= dtrace[i].a; j++) v.push_back(dtrace[i + j].x);
            prep.push_back(std::move(v));
        }
        auto cls = [](char c) { return c == 'E' || c == 'R' ? 'R' : c; };
        size_t i = 0, pi = 0;
        while(i < dtrace.size()) {
            const char c = cls(dtrace[i].op);
            auto t0 = clk_t::now();
            while(i < dtrace.size() && cls(dtrace[i].op) == c) {
                const DOp &o = dtrace[i]; auto &D = R[o.lvl];
                switch(o.op) {
                    case 'I': D.initialize((int)o.a, o.x); break;
                    case 'i': case 'R': D.insert(o.x); break;
                    case 'E': D.erase((int)o.a); break;
                    case 'L': { auto pr = D.pull(); if(pr.first != o.x || pull_hash(pr.second) != o.a) r.pull_mismatch++; break; }
                    case 'P': D.batchPrepend(prep[pi++]); i += o.a; break;
                }
                i++; r.ops++;
            }
            const double ms = ms_since(t0);
            if(c == 'I') r.init_ms += ms; else if(c == 'i') r.insert_pivots_ms += ms; else if(c == 'L') r.pull_ms += ms;
            else if(c == 'R') r.relax_ms += ms; else r.prepend_ms += ms;
        }
        return r;
    }
''')
open(dst, "w").write(head + tail)
