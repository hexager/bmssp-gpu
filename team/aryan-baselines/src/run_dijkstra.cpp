#include "dijkstra.hpp"
#include "graph.hpp"

#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>


int main(int argc, char** argv)
{
    if (argc != 3) {

        std::cerr
            << "Usage: "
            << argv[0]
            << " <graph.gr> <source>\n\n"

            << "Source uses DIMACS numbering "
            << "(starts from 1).\n";

        return 1;
    }


    try {

        const char* filename =
            argv[1];


        int dimacs_source =
            std::stoi(argv[2]);


        CSRGraph graph =
            load_dimacs_graph(filename);


        if (dimacs_source < 1 ||
            dimacs_source >
                graph.num_vertices) {

            throw std::runtime_error(
                "Source must be between 1 and " +
                std::to_string(
                    graph.num_vertices
                )
            );
        }


        /*
         * User enters DIMACS-style 1-based ID.
         *
         * Algorithm uses 0-based ID.
         */
        int source =
            dimacs_source - 1;


        DijkstraResult result =
            dijkstra(
                graph,
                source
            );


        const Distance INF =
            std::numeric_limits<
                Distance
            >::max();


        std::cout
            << "================================\n";

        std::cout
            << "CPU DIJKSTRA\n";

        std::cout
            << "================================\n";


        std::cout
            << "Graph      : "
            << filename
            << '\n';


        std::cout
            << "Vertices   : "
            << graph.num_vertices
            << '\n';


        std::cout
            << "Edges      : "
            << graph.num_edges
            << '\n';


        std::cout
            << "Source     : "
            << dimacs_source
            << '\n';


        std::cout
            << "Runtime    : "
            << result.runtime_ms
            << " ms\n";


        std::cout
            << "Edges examined : "
            << result.edges_examined
            << '\n';


        std::cout
            << "Successful relaxations : "
            << result.successful_relaxations
            << '\n';


        /*
         * Printing millions of distances would
         * destroy benchmark performance.
         *
         * Only print them for tiny graphs.
         */
        if (graph.num_vertices <= 20) {

            std::cout
                << "\nShortest distances:\n";


            for (int vertex = 0;
                 vertex < graph.num_vertices;
                 ++vertex) {

                std::cout
                    << (vertex + 1)
                    << " : ";


                if (result.distances[vertex]
                    == INF) {

                    std::cout
                        << "INF";
                }

                else {

                    std::cout
                        << result.distances[
                               vertex
                           ];
                }


                std::cout << '\n';
            }
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
