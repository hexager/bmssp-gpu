#include "graph.hpp"

#include <iostream>


int main(int argc, char** argv)
{
    if (argc != 2) {

        std::cerr
            << "Usage: "
            << argv[0]
            << " <graph.gr>\n";

        return 1;
    }


    try {

        CSRGraph graph =
            load_dimacs_graph(argv[1]);


        std::cout
            << "=============================\n";

        std::cout
            << "CSR GRAPH LOADER TEST\n";

        std::cout
            << "=============================\n\n";


        print_graph_info(graph);


        /*
         * For small test graphs only,
         * print the actual CSR arrays.
         */
        if (graph.num_vertices <= 20 &&
            graph.num_edges <= 100) {


            std::cout
                << "\nrow_offsets:\n";

            for (auto value :
                 graph.row_offsets) {

                std::cout
                    << value
                    << ' ';
            }


            std::cout
                << "\n\ncol_indices:\n";

            for (auto value :
                 graph.col_indices) {

                std::cout
                    << value
                    << ' ';
            }


            std::cout
                << "\n\nweights:\n";

            for (auto value :
                 graph.weights) {

                std::cout
                    << value
                    << ' ';
            }


            std::cout << '\n';
        }


        std::cout
            << "\nCSR validation: PASS\n";

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
