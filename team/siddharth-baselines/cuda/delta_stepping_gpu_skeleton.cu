#include "delta_stepping_gpu.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>


#define CUDA_CHECK(call)                                      \
do {                                                          \
    cudaError_t error = (call);                               \
                                                            \
    if (error != cudaSuccess) {                               \
        throw std::runtime_error(                             \
            std::string("CUDA error: ") +                     \
            cudaGetErrorString(error)                         \
        );                                                    \
    }                                                         \
} while (0)


struct DeviceGraph {

    std::int64_t* row_offsets = nullptr;

    int* col_indices = nullptr;

    Weight* weights = nullptr;

    int num_vertices = 0;

    std::int64_t num_edges = 0;
};


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
        graph.row_offsets.size()
        *
        sizeof(std::int64_t);


    std::size_t columns_bytes =
        graph.col_indices.size()
        *
        sizeof(int);


    std::size_t weights_bytes =
        graph.weights.size()
        *
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


static void free_device_graph(
    DeviceGraph& graph
)
{
    if (graph.row_offsets) {

        CUDA_CHECK(
            cudaFree(
                graph.row_offsets
            )
        );
    }


    if (graph.col_indices) {

        CUDA_CHECK(
            cudaFree(
                graph.col_indices
            )
        );
    }


    if (graph.weights) {

        CUDA_CHECK(
            cudaFree(
                graph.weights
            )
        );
    }


    graph.row_offsets = nullptr;
    graph.col_indices = nullptr;
    graph.weights = nullptr;
}


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


    auto transfer_start =
        std::chrono::steady_clock::now();


    DeviceGraph device_graph =
        copy_graph_to_gpu(
            graph
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
     * Actual GPU Delta-Stepping kernels
     * will be added in the next stage.
     */


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
