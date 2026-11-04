#ifndef DIJKSTRA_HPP
#define DIJKSTRA_HPP

#include "graph.hpp"

#include <cstdint>
#include <vector>


struct DijkstraResult {

    // Final shortest distance to every vertex.
    std::vector<Distance> distances;

    // Number of edges examined.
    std::uint64_t edges_examined = 0;

    // Number of successful distance updates.
    std::uint64_t successful_relaxations = 0;

    // Runtime of the algorithm itself in milliseconds.
    double runtime_ms = 0.0;
};


/*
 * Run Dijkstra from one source vertex.
 *
 * source uses INTERNAL 0-based numbering.
 */
DijkstraResult dijkstra(
    const CSRGraph& graph,
    int source
);

#endif

