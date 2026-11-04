#include "delta_stepping_gpu.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>


#define CUDA_CHECK(call)                                      \
do {                                                          \
    cudaError_t error = (call);                               \
    if (error != cudaSuccess) {                               \
        throw std::runtime_error(                             \
            std::string("CUDA error: ") +                     \
            cudaGetErrorString(error)                         \
        );                                                    \
    }                                                         \
} while (0)


using DeviceDistance = unsigned long long;

static constexpr DeviceDistance DEVICE_INF =
    0x7fffffffffffffffULL;


struct DeviceGraph {
    std::int64_t* row_offsets = nullptr;
    int* col_indices = nullptr;
    Weight* weights = nullptr;

    int num_vertices = 0;
    std::int64_t num_edges = 0;
};


/*
 * ----------------------------------------------------------
 * Initialize distances.
 * ----------------------------------------------------------
 */
__global__
void initialize_distances(
    DeviceDistance* distances,
    int num_vertices,
    int source
)
{
    int vertex =
        blockIdx.x * blockDim.x +
        threadIdx.x;

    if (vertex >= num_vertices) {
        return;
    }

    distances[vertex] =
        (vertex == source)
        ? 0ULL
        : DEVICE_INF;
}


/*
 * ----------------------------------------------------------
 * Build frontier for one Delta bucket.
 *
 * A vertex belongs to bucket i when:
 *
 * lower <= distance < upper
 * ----------------------------------------------------------
 */
__global__
void build_bucket_frontier(
    const DeviceDistance* distances,
    int num_vertices,
    DeviceDistance lower,
    DeviceDistance upper,
    int* frontier,
    int* frontier_size
)
{
    int vertex =
        blockIdx.x * blockDim.x +
        threadIdx.x;

    if (vertex >= num_vertices) {
        return;
    }

    DeviceDistance distance =
        distances[vertex];

    if (distance >= lower &&
        distance < upper) {

        int position =
            atomicAdd(
                frontier_size,
                1
            );

        frontier[position] =
            vertex;
    }
}


/*
 * ----------------------------------------------------------
 * Append current frontier S to removed set R.
 *
 * R contains all vertices processed from the current bucket
 * during light-edge closure.
 * ----------------------------------------------------------
 */
__global__
void append_frontier_to_removed(
    const int* frontier,
    int frontier_size,
    int* removed,
    int* removed_size
)
{
    int index =
        blockIdx.x * blockDim.x +
        threadIdx.x;

    if (index >= frontier_size) {
        return;
    }

    int position =
        atomicAdd(
            removed_size,
            1
        );

    removed[position] =
        frontier[index];
}


/*
 * ----------------------------------------------------------
 * Relax LIGHT edges:
 *
 * weight <= Delta
 * ----------------------------------------------------------
 */
__global__
void relax_light_edges(
    const std::int64_t* row_offsets,
    const int* col_indices,
    const Weight* weights,
    DeviceDistance* distances,
    const int* frontier,
    int frontier_size,
    DeviceDistance delta,
    unsigned long long* edges_examined,
    unsigned long long* successful_relaxations,
    int* changed
)
{
    int index =
        blockIdx.x * blockDim.x +
        threadIdx.x;

    if (index >= frontier_size) {
        return;
    }

    int u =
        frontier[index];

    DeviceDistance du =
        distances[u];

    std::int64_t begin =
        row_offsets[u];

    std::int64_t end =
        row_offsets[u + 1];


    for (std::int64_t edge = begin;
         edge < end;
         ++edge) {

        DeviceDistance weight =
            static_cast<DeviceDistance>(
                weights[edge]
            );

        if (weight > delta) {
            continue;
        }

        atomicAdd(
            edges_examined,
            1ULL
        );

        int v =
            col_indices[edge];

        if (du >
            DEVICE_INF - weight) {

            continue;
        }

        DeviceDistance candidate =
            du + weight;

        DeviceDistance old =
            atomicMin(
                &distances[v],
                candidate
            );

        if (candidate < old) {

            atomicAdd(
                successful_relaxations,
                1ULL
            );

            atomicExch(
                changed,
                1
            );
        }
    }
}


/*
 * ----------------------------------------------------------
 * Relax HEAVY edges:
 *
 * weight > Delta
 *
 * These are processed from all vertices in R.
 * ----------------------------------------------------------
 */
__global__
void relax_heavy_edges(
    const std::int64_t* row_offsets,
    const int* col_indices,
    const Weight* weights,
    DeviceDistance* distances,
    const int* removed,
    int removed_size,
    DeviceDistance delta,
    unsigned long long* edges_examined,
    unsigned long long* successful_relaxations
)
{
    int index =
        blockIdx.x * blockDim.x +
        threadIdx.x;

    if (index >= removed_size) {
        return;
    }

    int u =
        removed[index];

    DeviceDistance du =
        distances[u];

    std::int64_t begin =
        row_offsets[u];

    std::int64_t end =
        row_offsets[u + 1];


    for (std::int64_t edge = begin;
         edge < end;
         ++edge) {

        DeviceDistance weight =
            static_cast<DeviceDistance>(
                weights[edge]
            );

        if (weight <= delta) {
            continue;
        }

        atomicAdd(
            edges_examined,
            1ULL
        );

        int v =
            col_indices[edge];

        if (du >
            DEVICE_INF - weight) {

            continue;
        }

        DeviceDistance candidate =
            du + weight;

        DeviceDistance old =
            atomicMin(
                &distances[v],
                candidate
            );

        if (candidate < old) {

            atomicAdd(
                successful_relaxations,
                1ULL
            );
        }
    }
}


/*
 * ----------------------------------------------------------
 * Copy CSR graph to GPU.
 * ----------------------------------------------------------
 */
static DeviceGraph copy_graph_to_gpu(
    const CSRGraph& graph
)
{
    DeviceGraph device_graph;

    device_graph.num_vertices =
        graph.num_vertices;

    device_graph.num_edges =
        graph.num_edges;


    std::size_t offsets_bytes =
        graph.row_offsets.size() *
        sizeof(std::int64_t);

    std::size_t columns_bytes =
        graph.col_indices.size() *
        sizeof(int);

    std::size_t weights_bytes =
        graph.weights.size() *
        sizeof(Weight);


    CUDA_CHECK(
        cudaMalloc(
            reinterpret_cast<void**>(
                &device_graph.row_offsets
            ),
            offsets_bytes
        )
    );

    CUDA_CHECK(
        cudaMalloc(
            reinterpret_cast<void**>(
                &device_graph.col_indices
            ),
            columns_bytes
        )
    );

    CUDA_CHECK(
        cudaMalloc(
            reinterpret_cast<void**>(
                &device_graph.weights
            ),
            weights_bytes
        )
    );


    CUDA_CHECK(
        cudaMemcpy(
            device_graph.row_offsets,
            graph.row_offsets.data(),
            offsets_bytes,
            cudaMemcpyHostToDevice
        )
    );

    CUDA_CHECK(
        cudaMemcpy(
            device_graph.col_indices,
            graph.col_indices.data(),
            columns_bytes,
            cudaMemcpyHostToDevice
        )
    );

    CUDA_CHECK(
        cudaMemcpy(
            device_graph.weights,
            graph.weights.data(),
            weights_bytes,
            cudaMemcpyHostToDevice
        )
    );


    return device_graph;
}


/*
 * ----------------------------------------------------------
 * Free CSR graph.
 * ----------------------------------------------------------
 */
static void free_device_graph(
    DeviceGraph& graph
)
{
    if (graph.row_offsets) {
        cudaFree(graph.row_offsets);
    }

    if (graph.col_indices) {
        cudaFree(graph.col_indices);
    }

    if (graph.weights) {
        cudaFree(graph.weights);
    }

    graph.row_offsets = nullptr;
    graph.col_indices = nullptr;
    graph.weights = nullptr;
}


/*
 * ----------------------------------------------------------
 * GPU Delta-Stepping.
 *
 * Correctness-first implementation:
 *
 * - CSR remains on GPU.
 * - Distance array remains on GPU.
 * - Relaxation is performed by CUDA kernels.
 * - CPU controls bucket progression.
 *
 * This is not yet the final optimized implementation.
 * ----------------------------------------------------------
 */
GPUDeltaResult delta_stepping_gpu(
    const CSRGraph& graph,
    int source,
    Distance delta
)
{
    if (source < 0 ||
        source >= graph.num_vertices) {

        throw std::runtime_error(
            "GPU Delta-Stepping source outside valid range."
        );
    }

    if (delta <= 0) {

        throw std::runtime_error(
            "Delta must be positive."
        );
    }


    GPUDeltaResult result;


    auto total_start =
        std::chrono::steady_clock::now();


    /*
     * --------------------------------------------------
     * GPU allocation and graph transfer.
     * --------------------------------------------------
     */

    auto transfer_start =
        std::chrono::steady_clock::now();


    DeviceGraph device_graph =
        copy_graph_to_gpu(
            graph
        );


    DeviceDistance* d_distances =
        nullptr;

    int* d_frontier =
        nullptr;

    int* d_frontier_size =
        nullptr;

    int* d_removed =
        nullptr;

    int* d_removed_size =
        nullptr;

    int* d_changed =
        nullptr;

    unsigned long long*
        d_edges_examined =
            nullptr;

    unsigned long long*
        d_successful_relaxations =
            nullptr;


    CUDA_CHECK(
        cudaMalloc(
            reinterpret_cast<void**>(
                &d_distances
            ),
            graph.num_vertices *
            sizeof(DeviceDistance)
        )
    );


    CUDA_CHECK(
        cudaMalloc(
            reinterpret_cast<void**>(
                &d_frontier
            ),
            graph.num_vertices *
            sizeof(int)
        )
    );


    CUDA_CHECK(
        cudaMalloc(
            reinterpret_cast<void**>(
                &d_frontier_size
            ),
            sizeof(int)
        )
    );


    /*
     * R can theoretically contain repeated entries
     * in this correctness-first implementation.
     *
     * Allocate more than V entries to provide room
     * during validation.
     */
    std::size_t removed_capacity =
        static_cast<std::size_t>(
            graph.num_vertices
        ) * 8ULL;


    CUDA_CHECK(
        cudaMalloc(
            reinterpret_cast<void**>(
                &d_removed
            ),
            removed_capacity *
            sizeof(int)
        )
    );


    CUDA_CHECK(
        cudaMalloc(
            reinterpret_cast<void**>(
                &d_removed_size
            ),
            sizeof(int)
        )
    );


    CUDA_CHECK(
        cudaMalloc(
            reinterpret_cast<void**>(
                &d_changed
            ),
            sizeof(int)
        )
    );


    CUDA_CHECK(
        cudaMalloc(
            reinterpret_cast<void**>(
                &d_edges_examined
            ),
            sizeof(unsigned long long)
        )
    );


    CUDA_CHECK(
        cudaMalloc(
            reinterpret_cast<void**>(
                &d_successful_relaxations
            ),
            sizeof(unsigned long long)
        )
    );


    CUDA_CHECK(
        cudaMemset(
            d_edges_examined,
            0,
            sizeof(unsigned long long)
        )
    );


    CUDA_CHECK(
        cudaMemset(
            d_successful_relaxations,
            0,
            sizeof(unsigned long long)
        )
    );


    const int threads =
        256;


    int vertex_blocks =
        (
            graph.num_vertices +
            threads - 1
        )
        /
        threads;


    initialize_distances<<<
        vertex_blocks,
        threads
    >>>(
        d_distances,
        graph.num_vertices,
        source
    );


    CUDA_CHECK(
        cudaGetLastError()
    );


    CUDA_CHECK(
        cudaDeviceSynchronize()
    );


    auto transfer_end =
        std::chrono::steady_clock::now();


    result.transfer_to_gpu_ms =
        std::chrono::duration<
            double,
            std::milli
        >(
            transfer_end -
            transfer_start
        ).count();


    /*
     * --------------------------------------------------
     * Delta-Stepping bucket loop.
     * --------------------------------------------------
     */

    auto kernel_start =
        std::chrono::steady_clock::now();


    DeviceDistance bucket_index =
        0;


    bool work_remaining =
        true;


    const std::uint64_t max_iterations =
        10000000ULL;


    while (work_remaining) {

        if (result.iterations >=
            max_iterations) {

            throw std::runtime_error(
                "GPU Delta-Stepping exceeded "
                "iteration safety limit."
            );
        }


        DeviceDistance lower =
            bucket_index *
            static_cast<DeviceDistance>(
                delta
            );


        DeviceDistance upper =
            lower +
            static_cast<DeviceDistance>(
                delta
            );


        int frontier_size =
            0;


        CUDA_CHECK(
            cudaMemset(
                d_frontier_size,
                0,
                sizeof(int)
            )
        );


        build_bucket_frontier<<<
            vertex_blocks,
            threads
        >>>(
            d_distances,
            graph.num_vertices,
            lower,
            upper,
            d_frontier,
            d_frontier_size
        );


        CUDA_CHECK(
            cudaGetLastError()
        );


        CUDA_CHECK(
            cudaMemcpy(
                &frontier_size,
                d_frontier_size,
                sizeof(int),
                cudaMemcpyDeviceToHost
            )
        );


        /*
         * ------------------------------------------------
         * Empty bucket.
         * ------------------------------------------------
         */
        if (frontier_size == 0) {

            ++bucket_index;


            /*
             * Correctness-first approach:
             *
             * Copy distances to CPU and determine whether
             * any later bucket still contains work.
             */
            std::vector<DeviceDistance>
                snapshot(
                    graph.num_vertices
                );


            CUDA_CHECK(
                cudaMemcpy(
                    snapshot.data(),
                    d_distances,
                    graph.num_vertices *
                    sizeof(DeviceDistance),
                    cudaMemcpyDeviceToHost
                )
            );


            work_remaining =
                false;


            for (DeviceDistance distance :
                 snapshot) {

                if (distance ==
                    DEVICE_INF) {

                    continue;
                }


                DeviceDistance index =
                    distance /
                    static_cast<DeviceDistance>(
                        delta
                    );


                if (index >=
                    bucket_index) {

                    work_remaining =
                        true;

                    break;
                }
            }


            continue;
        }


        /*
         * ------------------------------------------------
         * Start R = empty for this bucket.
         * ------------------------------------------------
         */

        CUDA_CHECK(
            cudaMemset(
                d_removed_size,
                0,
                sizeof(int)
            )
        );


        /*
         * ------------------------------------------------
         * LIGHT-EDGE CLOSURE
         * ------------------------------------------------
         */

        bool continue_light =
            true;


        while (continue_light) {

            /*
             * Append current frontier S to R.
             */

            int blocks =
                (
                    frontier_size +
                    threads - 1
                )
                /
                threads;


            append_frontier_to_removed<<<
                blocks,
                threads
            >>>(
                d_frontier,
                frontier_size,
                d_removed,
                d_removed_size
            );


            CUDA_CHECK(
                cudaGetLastError()
            );


            /*
             * Reset change flag.
             */

            CUDA_CHECK(
                cudaMemset(
                    d_changed,
                    0,
                    sizeof(int)
                )
            );


            /*
             * Relax light edges.
             */

            relax_light_edges<<<
                blocks,
                threads
            >>>(
                device_graph.row_offsets,
                device_graph.col_indices,
                device_graph.weights,
                d_distances,
                d_frontier,
                frontier_size,
                static_cast<DeviceDistance>(
                    delta
                ),
                d_edges_examined,
                d_successful_relaxations,
                d_changed
            );


            CUDA_CHECK(
                cudaGetLastError()
            );


            int host_changed =
                0;


            CUDA_CHECK(
                cudaMemcpy(
                    &host_changed,
                    d_changed,
                    sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );


            if (host_changed == 0) {

                continue_light =
                    false;

                break;
            }


            /*
             * Rebuild the current bucket.
             *
             * Any vertex whose updated distance still
             * lies in [lower, upper) participates in the
             * next light-edge round.
             */

            CUDA_CHECK(
                cudaMemset(
                    d_frontier_size,
                    0,
                    sizeof(int)
                )
            );


            build_bucket_frontier<<<
                vertex_blocks,
                threads
            >>>(
                d_distances,
                graph.num_vertices,
                lower,
                upper,
                d_frontier,
                d_frontier_size
            );


            CUDA_CHECK(
                cudaGetLastError()
            );


            CUDA_CHECK(
                cudaMemcpy(
                    &frontier_size,
                    d_frontier_size,
                    sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );


            if (frontier_size == 0) {

                continue_light =
                    false;

                break;
            }


            int removed_size_check =
                0;


            CUDA_CHECK(
                cudaMemcpy(
                    &removed_size_check,
                    d_removed_size,
                    sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );


            if (
                static_cast<std::size_t>(
                    removed_size_check
                )
                +
                static_cast<std::size_t>(
                    frontier_size
                )
                >
                removed_capacity
            ) {

                throw std::runtime_error(
                    "GPU removed-set capacity exceeded."
                );
            }
        }


        /*
         * ------------------------------------------------
         * HEAVY-EDGE PHASE
         *
         * Process heavy edges from every vertex in R.
         * ------------------------------------------------
         */

        int removed_size =
            0;


        CUDA_CHECK(
            cudaMemcpy(
                &removed_size,
                d_removed_size,
                sizeof(int),
                cudaMemcpyDeviceToHost
            )
        );


        if (removed_size > 0) {

            int blocks =
                (
                    removed_size +
                    threads - 1
                )
                /
                threads;


            relax_heavy_edges<<<
                blocks,
                threads
            >>>(
                device_graph.row_offsets,
                device_graph.col_indices,
                device_graph.weights,
                d_distances,
                d_removed,
                removed_size,
                static_cast<DeviceDistance>(
                    delta
                ),
                d_edges_examined,
                d_successful_relaxations
            );


            CUDA_CHECK(
                cudaGetLastError()
            );
        }


        CUDA_CHECK(
            cudaDeviceSynchronize()
        );


        ++result.iterations;

        ++bucket_index;
    }


    CUDA_CHECK(
        cudaDeviceSynchronize()
    );


    auto kernel_end =
        std::chrono::steady_clock::now();


    result.kernel_runtime_ms =
        std::chrono::duration<
            double,
            std::milli
        >(
            kernel_end -
            kernel_start
        ).count();


    /*
     * --------------------------------------------------
     * Copy results back.
     * --------------------------------------------------
     */

    auto back_start =
        std::chrono::steady_clock::now();


    std::vector<DeviceDistance>
        host_distances(
            graph.num_vertices
        );


    CUDA_CHECK(
        cudaMemcpy(
            host_distances.data(),
            d_distances,
            graph.num_vertices *
            sizeof(DeviceDistance),
            cudaMemcpyDeviceToHost
        )
    );


    result.distances.resize(
        graph.num_vertices
    );


    for (int vertex = 0;
         vertex < graph.num_vertices;
         ++vertex) {

        if (host_distances[vertex] ==
            DEVICE_INF) {

            result.distances[vertex] =
                std::numeric_limits<
                    Distance
                >::max();
        }

        else {

            result.distances[vertex] =
                static_cast<Distance>(
                    host_distances[vertex]
                );
        }
    }


    CUDA_CHECK(
        cudaMemcpy(
            &result.edges_examined,
            d_edges_examined,
            sizeof(unsigned long long),
            cudaMemcpyDeviceToHost
        )
    );


    CUDA_CHECK(
        cudaMemcpy(
            &result.successful_relaxations,
            d_successful_relaxations,
            sizeof(unsigned long long),
            cudaMemcpyDeviceToHost
        )
    );


    CUDA_CHECK(
        cudaDeviceSynchronize()
    );


    auto back_end =
        std::chrono::steady_clock::now();


    result.transfer_from_gpu_ms =
        std::chrono::duration<
            double,
            std::milli
        >(
            back_end -
            back_start
        ).count();


    /*
     * --------------------------------------------------
     * Cleanup.
     * --------------------------------------------------
     */

    cudaFree(
        d_distances
    );

    cudaFree(
        d_frontier
    );

    cudaFree(
        d_frontier_size
    );

    cudaFree(
        d_removed
    );

    cudaFree(
        d_removed_size
    );

    cudaFree(
        d_changed
    );

    cudaFree(
        d_edges_examined
    );

    cudaFree(
        d_successful_relaxations
    );


    free_device_graph(
        device_graph
    );


    auto total_end =
        std::chrono::steady_clock::now();


    result.total_runtime_ms =
        std::chrono::duration<
            double,
            std::milli
        >(
            total_end -
            total_start
        ).count();


    return result;
}
