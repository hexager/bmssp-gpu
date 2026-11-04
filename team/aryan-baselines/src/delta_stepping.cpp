#include "delta_stepping.hpp"

#include <chrono>
#include <limits>
#include <stdexcept>
#include <unordered_set>
#include <vector>


DeltaSteppingResult delta_stepping_cpu(
    const CSRGraph& graph,
    int source,
    Distance delta
)
{
    if (source < 0 ||
        source >= graph.num_vertices) {

        throw std::runtime_error(
            "Delta-stepping source is outside valid range."
        );
    }


    if (delta <= 0) {

        throw std::runtime_error(
            "Delta must be positive."
        );
    }


    const Distance INF =
        std::numeric_limits<Distance>::max();


    DeltaSteppingResult result;

    result.distances.assign(
        graph.num_vertices,
        INF
    );


    /*
     * Buckets are created dynamically.
     *
     * bucket[i] contains vertices whose
     * tentative distance lies in:
     *
     * [i * delta, (i + 1) * delta)
     */
    std::vector<
        std::unordered_set<int>
    > buckets(1);


    auto ensure_bucket =
        [&](std::size_t index) {

            if (index >= buckets.size()) {

                buckets.resize(
                    index + 1
                );
            }
        };


    auto insert_vertex =
        [&](int vertex,
            Distance distance) {

            std::size_t bucket_index =
                static_cast<std::size_t>(
                    distance / delta
                );

            ensure_bucket(
                bucket_index
            );

            buckets[bucket_index].insert(
                vertex
            );
        };


    result.distances[source] = 0;

    insert_vertex(
        source,
        0
    );


    auto start =
        std::chrono::steady_clock::now();


    std::size_t current_bucket = 0;


    while (true) {

        /*
         * Find next non-empty bucket.
         */
        while (
            current_bucket <
                buckets.size()
            &&
            buckets[
                current_bucket
            ].empty()
        ) {

            ++current_bucket;
        }


        if (current_bucket >=
            buckets.size()) {

            break;
        }


        ++result.buckets_processed;


        /*
         * R contains vertices removed from
         * the current bucket.
         *
         * Heavy edges are processed after
         * all light-edge closure work.
         */
        std::vector<int> removed;


        /*
         * ------------------------------------------------
         * LIGHT EDGES
         *
         * weight <= delta
         * ------------------------------------------------
         */
        while (
            !buckets[
                current_bucket
            ].empty()
        ) {

            std::vector<int> frontier;

            frontier.reserve(
                buckets[
                    current_bucket
                ].size()
            );


            for (int vertex :
                 buckets[
                     current_bucket
                 ]) {

                frontier.push_back(
                    vertex
                );
            }


            buckets[
                current_bucket
            ].clear();


            for (int u : frontier) {

                /*
                 * Ignore stale entries.
                 */
                std::size_t actual_bucket =
                    static_cast<std::size_t>(
                        result.distances[u]
                        /
                        delta
                    );


                if (actual_bucket !=
                    current_bucket) {

                    continue;
                }


                removed.push_back(u);


                std::int64_t begin =
                    graph.row_offsets[u];

                std::int64_t end =
                    graph.row_offsets[u + 1];


                for (std::int64_t edge = begin;
                     edge < end;
                     ++edge) {

                    Weight weight =
                        graph.weights[edge];


                    if (weight > delta) {
                        continue;
                    }


                    ++result.edges_examined;


                    int v =
                        graph.col_indices[edge];


                    Distance du =
                        result.distances[u];


                    if (du >
                        INF - weight) {

                        continue;
                    }


                    Distance candidate =
                        du + weight;


                    if (candidate <
                        result.distances[v]) {

                        Distance old_distance =
                            result.distances[v];


                        /*
                         * Remove v from its old
                         * bucket when possible.
                         */
                        if (old_distance != INF) {

                            std::size_t old_bucket =
                                static_cast<
                                    std::size_t
                                >(
                                    old_distance
                                    /
                                    delta
                                );


                            if (old_bucket <
                                buckets.size()) {

                                buckets[
                                    old_bucket
                                ].erase(v);
                            }
                        }


                        result.distances[v] =
                            candidate;


                        ++result.
                            successful_relaxations;


                        insert_vertex(
                            v,
                            candidate
                        );
                    }
                }
            }
        }


        /*
         * ------------------------------------------------
         * HEAVY EDGES
         *
         * weight > delta
         * ------------------------------------------------
         */
        for (int u : removed) {

            Distance du =
                result.distances[u];


            std::int64_t begin =
                graph.row_offsets[u];

            std::int64_t end =
                graph.row_offsets[u + 1];


            for (std::int64_t edge = begin;
                 edge < end;
                 ++edge) {

                Weight weight =
                    graph.weights[edge];


                if (weight <= delta) {
                    continue;
                }


                ++result.edges_examined;


                int v =
                    graph.col_indices[edge];


                if (du >
                    INF - weight) {

                    continue;
                }


                Distance candidate =
                    du + weight;


                if (candidate <
                    result.distances[v]) {

                    Distance old_distance =
                        result.distances[v];


                    if (old_distance != INF) {

                        std::size_t old_bucket =
                            static_cast<
                                std::size_t
                            >(
                                old_distance
                                /
                                delta
                            );


                        if (old_bucket <
                            buckets.size()) {

                            buckets[
                                old_bucket
                            ].erase(v);
                        }
                    }


                    result.distances[v] =
                        candidate;


                    ++result.
                        successful_relaxations;


                    insert_vertex(
                        v,
                        candidate
                    );
                }
            }
        }


        ++current_bucket;
    }


    auto finish =
        std::chrono::steady_clock::now();


    result.runtime_ms =
        std::chrono::duration<
            double,
            std::milli
        >(
            finish - start
        ).count();


    return result;
}
