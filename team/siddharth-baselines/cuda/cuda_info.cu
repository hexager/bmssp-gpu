#include <cuda_runtime.h>

#include <iostream>


int main()
{
    int device_count = 0;


    cudaError_t error =
        cudaGetDeviceCount(
            &device_count
        );


    if (error != cudaSuccess) {

        std::cerr
            << "CUDA ERROR: "
            << cudaGetErrorString(error)
            << '\n';

        return 1;
    }


    std::cout
        << "CUDA devices: "
        << device_count
        << "\n\n";


    for (int device = 0;
         device < device_count;
         ++device) {

        cudaDeviceProp properties;


        cudaGetDeviceProperties(
            &properties,
            device
        );


        std::cout
            << "Device "
            << device
            << '\n';


        std::cout
            << "Name: "
            << properties.name
            << '\n';


        std::cout
            << "Compute capability: "
            << properties.major
            << '.'
            << properties.minor
            << '\n';


        std::cout
            << "Global memory: "
            << (
                properties.totalGlobalMem
                /
                (1024 * 1024)
            )
            << " MB\n";


        std::cout
            << "Multiprocessors: "
            << properties.multiProcessorCount
            << '\n';


        std::cout
            << "Max threads/block: "
            << properties.maxThreadsPerBlock
            << "\n\n";
    }


    return 0;
}
