#include "delta_stepping.hpp"
#include "dijkstra.hpp"
#include "graph.hpp"

#include <iostream>
#include <string>
#include <vector>


bool test_graph(
    const std::string& filename,
    int source,
    Distance delta
)
{
    CSRGraph graph =
        load_dimacs_graph(filename);


    DijkstraResult reference =
        dijkstra(
            graph,
            source
        );


    DeltaSteppingResult delta_result =
        delta_stepping_cpu(
            graph,
            source,
            delta
        );


    if (reference.distances.size() !=
        delta_result.distances.size()) {

        std::cerr
            << "[FAIL] "
            << filename
            << " size mismatch\n";

        return false;
    }


    std::size_t mismatches = 0;


    for (std::size_t vertex = 0;
         vertex <
             reference.distances.size();
         ++vertex) {

        if (
            reference.distances[vertex]
            !=
            delta_result.distances[vertex]
        ) {

            if (mismatches < 10) {

                std::cerr
                    << "Mismatch vertex "
                    << (vertex + 1)
                    << ": Dijkstra="
                    << reference.distances[
                           vertex
                       ]
                    << " Delta="
                    << delta_result.distances[
                           vertex
                       ]
                    << '\n';
            }


            ++mismatches;
        }
    }


    if (mismatches != 0) {

        std::cerr
            << "[FAIL] "
            << filename
            << " source="
            << (source + 1)
            << " delta="
            << delta
            << " mismatches="
            << mismatches
            << '\n';

        return false;
    }


    std::cout
        << "[PASS] "
        << filename
        << " source="
        << (source + 1)
        << " delta="
        << delta
        << " | buckets="
        << delta_result.buckets_processed
        << " | edges="
        << delta_result.edges_examined
        << '\n';


    return true;
}


int main()
{
    bool passed = true;


    /*
     * Tiny graph.
     */
    passed &=
        test_graph(
            "data/tiny.gr",
            0,
            1
        );


    passed &=
        test_graph(
            "data/tiny.gr",
            0,
            2
        );


    passed &=
        test_graph(
            "data/tiny.gr",
            0,
            4
        );


    /*
     * Second known-answer graph.
     */
    passed &=
        test_graph(
            "data/test2.gr",
            0,
            3
        );


    /*
     * Generated grid.
     */
    passed &=
        test_graph(
            "data/grid10.gr",
            0,
            10
        );


    /*
     * RMAT.
     */
    passed &=
        test_graph(
            "data/rmat8.gr",
            0,
            10
        );


    if (!passed) {

        std::cerr
            << "\nDELTA-STEPPING TESTS FAILED\n";

        return 1;
    }


    std::cout
        << "\nALL DELTA-STEPPING TESTS PASSED\n";

    return 0;
}
