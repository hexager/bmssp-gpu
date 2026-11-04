# Builds bmssp_hybrid.hpp from the frozen CPU header (cpu-v1.4) by adding dirty-vertex tracking.
# Two edits only: updateDist() records the vertex it changes, and public members hold that list.
# The header contains two solver classes; only the first (the one with findPivotsImpl) is patched.
import sys

src, dst = sys.argv[1], sys.argv[2]
s = open(src).read()

second_class = s.find("class bmssp {", s.find("class bmssp {") + 1)
old_upd = "        path_sz[v] = path_sz[u] + 1;\n    }\n\n    // ==="
new_upd = ("        path_sz[v] = path_sz[u] + 1;\n"
           "        if(track_dirty && !dirty_flag[v]) { dirty_flag[v] = 1; dirty_list.push_back(v); }  // hybrid: CPU-side change log\n"
           "    }\n\n    // ===")
first = s.find(old_upd)
assert first != -1 and (second_class == -1 or first < second_class), "updateDist anchor not found in first class"
s = s[:first] + new_upd + s[first + len(old_upd):]

anchor = ("    std::function<std::pair<std::vector<int>, std::vector<int>>(uniqueDistT, const std::vector<int>&)>"
          " findPivotsImpl;\n")
assert s.count(anchor) == 1, "findPivotsImpl anchor not unique"
s = s.replace(anchor, anchor +
    "    // hybrid: vertices whose (d, pred, path_sz) the CPU changed since the last clear. Off unless track_dirty.\n"
    "    bool track_dirty = false; std::vector<int> dirty_list; std::vector<char> dirty_flag;\n"
    "    void enable_dirty_tracking() { track_dirty = true; dirty_flag.assign(adj.size(), 0); dirty_list.clear(); }\n"
    "    void clear_dirty() { for(int v: dirty_list) dirty_flag[v] = 0; dirty_list.clear(); }\n")
open(dst, "w").write(s)
