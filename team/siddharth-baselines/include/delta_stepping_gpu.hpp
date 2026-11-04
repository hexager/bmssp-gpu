#ifndef DELTA_STEPPING_GPU_HPP
#define DELTA_STEPPING_GPU_HPP

#include "graph.hpp"

#include <cstdint>
#include <vector>


struct GPUDeltaResult {

    std::vector<Distance> distances;

    double total_runtime_ms = 0.0;

    double transfer_to_gpu_ms = 0.0;

    double kernel_runtime_ms = 0.0;

    double transfer_from_gpu_ms = 0.0;

    std::uint64_t edges_examined = 0;

    std::uint64_t successful_relaxations = 0;

    std::uint64_t iterations = 0;
};


GPUDeltaResult delta_stepping_gpu(
    const CSRGraph& graph,
    int source,
    Distance delta
);


#endif
