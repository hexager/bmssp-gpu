#include "dijkstra.hpp"

#include <chrono>
#include <functional>
#include <limits>
#include <queue>
#include <stdexcept>
#include <utility>
#include <vector>


DijkstraResult dijkstra(
    const CSRGraph& graph,
    int source
)
{
    if (source < 0 ||
        source >= graph.num_vertices) {

        throw std::runtime_error(
            "Dijkstra source vertex is outside valid range."
        );
    }


    const Distance INF =
        std::numeric_limits<Distance>::max();


    DijkstraResult result;


    result.distances.assign(
        graph.num_vertices,
        INF
    );


    /*
     * Priority queue entry:
     *
     *     (distance, vertex)
     *
     * std::greater makes this a MIN heap.
     */
    using QueueEntry =
        std::pair<Distance, int>;


    std::priority_queue<
        QueueEntry,
        std::vector<QueueEntry>,
        std::greater<QueueEntry>
    > queue;


    result.distances[source] = 0;

    queue.push({
        0,
        source
    });


    auto start =
        std::chrono::steady_clock::now();


    while (!queue.empty()) {

        auto [current_distance, u] =
            queue.top();

        queue.pop();


        /*
         * Ignore stale priority-queue entries.
         *
         * Example:
         *
         * vertex 5 may first be discovered with
         * distance 20 and later improved to 12.
         *
         * The old (20, 5) entry remains in the
         * priority queue and must be ignored.
         */
        if (current_distance !=
            result.distances[u]) {

            continue;
        }


        /*
         * CSR range containing all outgoing
         * edges of vertex u.
         */
        std::int64_t begin =
            graph.row_offsets[u];

        std::int64_t end =
            graph.row_offsets[u + 1];


        for (std::int64_t edge = begin;
             edge < end;
             ++edge) {

            ++result.edges_examined;


            int v =
                graph.col_indices[edge];

            Weight weight =
                graph.weights[edge];


            /*
             * Prevent signed integer overflow.
             */
            if (current_distance >
                INF - weight) {

                continue;
            }


            Distance candidate =
                current_distance + weight;


            /*
             * Relax edge u -> v.
             */
            if (candidate <
                result.distances[v]) {

                result.distances[v] =
                    candidate;


                ++result.successful_relaxations;


                queue.push({
                    candidate,
                    v
                });
            }
        }
    }


    auto finish =
        std::chrono::steady_clock::now();


    result.runtime_ms =
        std::chrono::duration<
            double,
            std::milli
        >(finish - start).count();


    return result;
}
