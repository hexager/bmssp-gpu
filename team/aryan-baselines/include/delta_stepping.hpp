#ifndef DELTA_STEPPING_HPP
#define DELTA_STEPPING_HPP

#include "graph.hpp"

#include <cstdint>
#include <vector>


struct DeltaSteppingResult {

    std::vector<Distance> distances;

    std::uint64_t edges_examined = 0;

    std::uint64_t successful_relaxations = 0;

    std::uint64_t buckets_processed = 0;

    double runtime_ms = 0.0;
};


DeltaSteppingResult delta_stepping_cpu(
    const CSRGraph& graph,
    int source,
    Distance delta
);


#endif
