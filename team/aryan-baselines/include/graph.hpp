#ifndef GRAPH_HPP
#define GRAPH_HPP

#include <cstdint>
#include <string>
#include <vector>

using Weight = std::int64_t;
using Distance = std::int64_t;

/*
 * Shared CSR representation of a directed weighted graph.
 *
 * For vertex u, its outgoing edges occupy:
 *
 *   row_offsets[u] ... row_offsets[u + 1] - 1
 *
 * For an edge at position e:
 *
 *   col_indices[e] = destination vertex
 *   weights[e]     = edge weight
 *
 * Vertices are stored internally using 0-based indexing.
 */
struct CSRGraph {
    int num_vertices = 0;
    std::int64_t num_edges = 0;

    std::vector<std::int64_t> row_offsets;
    std::vector<int> col_indices;
    std::vector<Weight> weights;
};


/*
 * Load a DIMACS shortest-path .gr file and convert
 * it directly into CSR format.
 */
CSRGraph load_dimacs_graph(const std::string& filename);


/*
 * Perform basic structural validation of a CSR graph.
 *
 * Throws std::runtime_error if the graph is invalid.
 */
void validate_csr(const CSRGraph& graph);


/*
 * Print useful information about a graph.
 */
void print_graph_info(const CSRGraph& graph);

#endif
