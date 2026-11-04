#include "graph.hpp"

#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>


struct Edge {
    int from;
    int to;
    Weight weight;
};


CSRGraph load_dimacs_graph(const std::string& filename)
{
    std::ifstream file(filename);

    if (!file.is_open()) {
        throw std::runtime_error(
            "Could not open graph file: " + filename
        );
    }


    int num_vertices = -1;
    std::int64_t declared_edges = -1;

    bool problem_line_seen = false;

    std::vector<Edge> edges;

    std::string line;

    std::int64_t line_number = 0;


    while (std::getline(file, line)) {

        ++line_number;


        // Ignore blank lines.
        if (line.empty()) {
            continue;
        }


        std::istringstream stream(line);

        char type;

        stream >> type;


        /*
         * DIMACS comment:
         *
         * c this is a comment
         */
        if (type == 'c') {
            continue;
        }


        /*
         * Problem definition:
         *
         * p sp <vertices> <edges>
         */
        if (type == 'p') {

            if (problem_line_seen) {
                throw std::runtime_error(
                    "Multiple DIMACS problem lines found."
                );
            }


            std::string format;

            if (!(stream >> format
                         >> num_vertices
                         >> declared_edges)) {

                throw std::runtime_error(
                    "Malformed problem line at line " +
                    std::to_string(line_number)
                );
            }


            if (format != "sp") {
                throw std::runtime_error(
                    "Expected DIMACS shortest-path format 'sp'."
                );
            }


            if (num_vertices <= 0) {
                throw std::runtime_error(
                    "Number of vertices must be positive."
                );
            }


            if (declared_edges < 0) {
                throw std::runtime_error(
                    "Number of edges cannot be negative."
                );
            }


            problem_line_seen = true;

            edges.reserve(
                static_cast<std::size_t>(declared_edges)
            );

            continue;
        }


        /*
         * Directed weighted arc:
         *
         * a <from> <to> <weight>
         */
        if (type == 'a') {

            if (!problem_line_seen) {
                throw std::runtime_error(
                    "Arc encountered before problem line."
                );
            }


            int from;
            int to;
            Weight weight;


            if (!(stream >> from >> to >> weight)) {

                throw std::runtime_error(
                    "Malformed arc at line " +
                    std::to_string(line_number)
                );
            }


            if (from < 1 || from > num_vertices ||
                to   < 1 || to   > num_vertices) {

                throw std::runtime_error(
                    "Vertex ID outside valid range at line " +
                    std::to_string(line_number)
                );
            }


            /*
             * Dijkstra and delta-stepping require
             * non-negative edge weights.
             */
            if (weight < 0) {

                throw std::runtime_error(
                    "Negative edge weight at line " +
                    std::to_string(line_number)
                );
            }


            /*
             * DIMACS uses 1-based vertex numbering.
             *
             * Our algorithms use 0-based numbering.
             */
            edges.push_back({
                from - 1,
                to - 1,
                weight
            });


            continue;
        }


        throw std::runtime_error(
            "Unknown DIMACS line type at line " +
            std::to_string(line_number)
        );
    }


    if (!problem_line_seen) {
        throw std::runtime_error(
            "DIMACS problem line was not found."
        );
    }


    if (static_cast<std::int64_t>(edges.size())
        != declared_edges) {

        throw std::runtime_error(
            "DIMACS edge count mismatch: header says " +
            std::to_string(declared_edges) +
            ", but file contains " +
            std::to_string(edges.size())
        );
    }


    /*
     * --------------------------------------------------
     * Convert edge list -> CSR
     * --------------------------------------------------
     */

    CSRGraph graph;

    graph.num_vertices = num_vertices;
    graph.num_edges = declared_edges;


    graph.row_offsets.assign(
        static_cast<std::size_t>(num_vertices) + 1,
        0
    );


    /*
     * Count outgoing edges.
     */
    for (const Edge& edge : edges) {

        ++graph.row_offsets[
            static_cast<std::size_t>(edge.from) + 1
        ];
    }


    /*
     * Prefix sum.
     *
     * After this:
     *
     * row_offsets[u]
     *
     * gives the starting position of vertex u's
     * outgoing edges.
     */
    for (int vertex = 1;
         vertex <= num_vertices;
         ++vertex) {

        graph.row_offsets[vertex] +=
            graph.row_offsets[vertex - 1];
    }


    graph.col_indices.resize(
        static_cast<std::size_t>(declared_edges)
    );

    graph.weights.resize(
        static_cast<std::size_t>(declared_edges)
    );


    /*
     * Temporary write pointers.
     */
    std::vector<std::int64_t> next =
        graph.row_offsets;


    /*
     * Place every edge into its CSR position.
     */
    for (const Edge& edge : edges) {

        std::int64_t position =
            next[edge.from]++;


        graph.col_indices[position] =
            edge.to;


        graph.weights[position] =
            edge.weight;
    }


    validate_csr(graph);

    return graph;
}


void validate_csr(const CSRGraph& graph)
{
    if (graph.num_vertices <= 0) {
        throw std::runtime_error(
            "CSR validation failed: invalid vertex count."
        );
    }


    if (graph.num_edges < 0) {
        throw std::runtime_error(
            "CSR validation failed: invalid edge count."
        );
    }


    if (graph.row_offsets.size() !=
        static_cast<std::size_t>(
            graph.num_vertices + 1
        )) {

        throw std::runtime_error(
            "CSR validation failed: incorrect "
            "row_offsets size."
        );
    }


    if (graph.col_indices.size() !=
        static_cast<std::size_t>(
            graph.num_edges
        )) {

        throw std::runtime_error(
            "CSR validation failed: incorrect "
            "col_indices size."
        );
    }


    if (graph.weights.size() !=
        static_cast<std::size_t>(
            graph.num_edges
        )) {

        throw std::runtime_error(
            "CSR validation failed: incorrect "
            "weights size."
        );
    }


    if (graph.row_offsets.front() != 0) {

        throw std::runtime_error(
            "CSR validation failed: first offset "
            "must be zero."
        );
    }


    if (graph.row_offsets.back() !=
        graph.num_edges) {

        throw std::runtime_error(
            "CSR validation failed: final offset "
            "must equal edge count."
        );
    }


    /*
     * Offsets must never decrease.
     */
    for (int vertex = 0;
         vertex < graph.num_vertices;
         ++vertex) {

        if (graph.row_offsets[vertex] >
            graph.row_offsets[vertex + 1]) {

            throw std::runtime_error(
                "CSR validation failed: "
                "row offsets are not monotonic."
            );
        }
    }


    /*
     * Every destination must be a valid vertex.
     */
    for (int destination :
         graph.col_indices) {

        if (destination < 0 ||
            destination >= graph.num_vertices) {

            throw std::runtime_error(
                "CSR validation failed: "
                "invalid destination vertex."
            );
        }
    }


    /*
     * Verify weights once again.
     */
    for (Weight weight :
         graph.weights) {

        if (weight < 0) {

            throw std::runtime_error(
                "CSR validation failed: "
                "negative weight."
            );
        }
    }
}


void print_graph_info(const CSRGraph& graph)
{
    std::cout
        << "Vertices : "
        << graph.num_vertices
        << '\n';


    std::cout
        << "Edges    : "
        << graph.num_edges
        << '\n';


    if (graph.num_vertices > 0) {

        double average_degree =
            static_cast<double>(
                graph.num_edges
            )
            /
            static_cast<double>(
                graph.num_vertices
            );


        std::cout
            << "Avg out-degree : "
            << average_degree
            << '\n';
    }
}
