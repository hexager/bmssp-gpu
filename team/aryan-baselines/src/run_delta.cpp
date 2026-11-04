#include "delta_stepping.hpp"
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
                "Source vertex outside valid range."
            );
        }


        if (delta <= 0) {

            throw std::runtime_error(
                "Delta must be positive."
            );
        }


        int source =
            dimacs_source - 1;


        std::cout
            << "Running Dijkstra reference...\n";


        DijkstraResult reference =
            dijkstra(
                graph,
                source
            );


        std::cout
            << "Running CPU Delta-Stepping...\n";


        DeltaSteppingResult result =
            delta_stepping_cpu(
                graph,
                source,
                delta
            );


        std::size_t mismatches = 0;


        for (std::size_t vertex = 0;
             vertex <
                 reference.distances.size();
             ++vertex) {

            if (
                reference.distances[vertex]
                !=
                result.distances[vertex]
            ) {

                if (mismatches < 10) {

                    std::cerr
                        << "Mismatch at vertex "
                        << (vertex + 1)
                        << ": Dijkstra="
                        << reference.distances[
                               vertex
                           ]
                        << " Delta="
                        << result.distances[
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
            << "DELTA-STEPPING CHECK\n";

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
            << "Dijkstra    : "
            << reference.runtime_ms
            << " ms\n";

        std::cout
            << "Delta CPU   : "
            << result.runtime_ms
            << " ms\n";

        std::cout
            << "Buckets     : "
            << result.buckets_processed
            << '\n';

        std::cout
            << "Edges examined : "
            << result.edges_examined
            << '\n';

        std::cout
            << "Successful relaxations : "
            << result.successful_relaxations
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
