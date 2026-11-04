// Graph generator for the scaling study (DIMACS .gr, 1-based ids, integer weights uniform in [1, 100]).
//   gen_graph rmat SCALE EDGE_FACTOR SEED OUT.gr   R-MAT (a, b, c, d) = (0.57, 0.19, 0.19, 0.05), n = 2^SCALE,
//                                                  m = EDGE_FACTOR * n; self-loops are redrawn, duplicates kept
//   gen_graph grid SIDE SEED OUT.gr                SIDE x SIDE four-neighbour grid, both directions of every edge
// Same distributions as scripts/generate_rmat.py and the grid500 benchmark graph; not the same bytes.
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: gen_graph rmat SCALE EF SEED OUT | grid SIDE SEED OUT\n"); return 2; }
    const std::string kind = argv[1];
    std::mt19937_64 rng;
    std::uniform_int_distribution<int> wdist(1, 100);
    std::vector<char> buf; buf.reserve(1 << 24);
    FILE* out = nullptr;
    auto flush = [&]() { std::fwrite(buf.data(), 1, buf.size(), out); buf.clear(); };
    char line[64];
    auto arc = [&](long long u, long long v) {
        int len = std::snprintf(line, sizeof line, "a %lld %lld %d\n", u + 1, v + 1, wdist(rng));
        buf.insert(buf.end(), line, line + len);
        if (buf.size() > (1 << 24) - 64) flush();
    };
    if (kind == "rmat" && argc == 6) {
        const int scale = std::atoi(argv[2]); const long long ef = std::atoll(argv[3]);
        rng.seed(std::strtoull(argv[4], nullptr, 10));
        out = std::fopen(argv[5], "wb"); if (!out) { std::perror(argv[5]); return 2; }
        const long long n = 1LL << scale, m = ef * n;
        std::fprintf(out, "c R-MAT scale %d edge factor %lld seed %s (gen_graph.cpp)\np sp %lld %lld\n", scale, ef, argv[4], n, m);
        std::uniform_real_distribution<double> U(0.0, 1.0);
        for (long long e = 0; e < m; e++) {
            long long u, v;
            do {
                u = 0; v = 0;
                for (int b = 0; b < scale; b++) {
                    double r = U(rng);
                    int q = r < 0.57 ? 0 : r < 0.76 ? 1 : r < 0.95 ? 2 : 3;
                    u = (u << 1) | (q >> 1); v = (v << 1) | (q & 1);
                }
            } while (u == v);
            arc(u, v);
        }
    } else if (kind == "grid" && argc == 5) {
        const long long side = std::atoll(argv[2]);
        rng.seed(std::strtoull(argv[3], nullptr, 10));
        out = std::fopen(argv[4], "wb"); if (!out) { std::perror(argv[4]); return 2; }
        const long long n = side * side, m = 4 * side * (side - 1);
        std::fprintf(out, "c %lld x %lld grid seed %s (gen_graph.cpp)\np sp %lld %lld\n", side, side, argv[3], n, m);
        for (long long r = 0; r < side; r++)
            for (long long c = 0; c < side; c++) {
                long long id = r * side + c;
                if (c + 1 < side) { arc(id, id + 1); arc(id + 1, id); }
                if (r + 1 < side) { arc(id, id + side); arc(id + side, id); }
            }
    } else { std::fprintf(stderr, "bad arguments\n"); return 2; }
    flush(); std::fclose(out);
    return 0;
}
