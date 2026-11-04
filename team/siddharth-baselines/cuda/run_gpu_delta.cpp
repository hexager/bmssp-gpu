#include "delta_stepping_gpu.hpp"
#include "dijkstra.hpp"
#include "graph.hpp"

#include <iostream>
#include <stdexcept>
#include <string>


int main(int argc, char** argv)
{
    if (argc != 4) {

        std::cerr
            << "Usage: "
            << argv[0]
            << " <graph.gr> <source> <delta>\n";

        return 1;
    }


    try {

        std::string filename =
            argv[1];

        int dimacs_source =
            std::stoi(argv[2]);

        Distance delta =
            std::stoll(argv[3]);


        CSRGraph graph =
            load_dimacs_graph(
                filename
            );


        if (dimacs_source < 1 ||
            dimacs_source >
                graph.num_vertices) {

            throw std::runtime_error(
                "Invalid source."
            );
        }


        int source =
            dimacs_source - 1;


        std::cout
            << "Running CPU Dijkstra...\n";


        DijkstraResult reference =
            dijkstra(
                graph,
                source
            );


        std::cout
            << "Running GPU Delta-Stepping...\n";


        GPUDeltaResult gpu =
            delta_stepping_gpu(
                graph,
                source,
                delta
            );


        std::size_t mismatches = 0;


        if (gpu.distances.size() !=
            reference.distances.size()) {

            throw std::runtime_error(
                "GPU returned incorrect distance-array size."
            );
        }


        for (std::size_t vertex = 0;
             vertex <
                 reference.distances.size();
             ++vertex) {

            if (
                reference.distances[vertex]
                !=
                gpu.distances[vertex]
            ) {

                if (mismatches < 10) {

                    std::cerr
                        << "Mismatch vertex "
                        << (vertex + 1)
                        << ": CPU="
                        << reference.distances[
                               vertex
                           ]
                        << " GPU="
                        << gpu.distances[
                               vertex
                           ]
                        << '\n';
                }


                ++mismatches;
            }
        }


        std::cout
            << "\n================================\n";

        std::cout
            << "GPU DELTA-STEPPING\n";

        std::cout
            << "================================\n";

        std::cout
            << "Graph       : "
            << filename
            << '\n';

        std::cout
            << "Vertices    : "
            << graph.num_vertices
            << '\n';

        std::cout
            << "Edges       : "
            << graph.num_edges
            << '\n';

        std::cout
            << "Source      : "
            << dimacs_source
            << '\n';

        std::cout
            << "Delta       : "
            << delta
            << '\n';

        std::cout
            << "CPU Dijkstra: "
            << reference.runtime_ms
            << " ms\n";

        std::cout
            << "GPU total   : "
            << gpu.total_runtime_ms
            << " ms\n";

        std::cout
            << "H2D/setup   : "
            << gpu.transfer_to_gpu_ms
            << " ms\n";

        std::cout
            << "GPU section : "
            << gpu.kernel_runtime_ms
            << " ms\n";

        std::cout
            << "D2H         : "
            << gpu.transfer_from_gpu_ms
            << " ms\n";

        std::cout
            << "Iterations  : "
            << gpu.iterations
            << '\n';

        std::cout
            << "Edges examined : "
            << gpu.edges_examined
            << '\n';

        std::cout
            << "Successful relaxations : "
            << gpu.successful_relaxations
            << '\n';

        std::cout
            << "Mismatches  : "
            << mismatches
            << '\n';


        if (mismatches == 0) {

            std::cout
                << "Correctness : PASS\n";
        }

        else {

            std::cout
                << "Correctness : FAIL\n";

            return 1;
        }


        std::cout
            << "================================\n";
    }


    catch (const std::exception& exception) {

        std::cerr
            << "ERROR: "
            << exception.what()
            << '\n';

        return 1;
    }


    return 0;
}
