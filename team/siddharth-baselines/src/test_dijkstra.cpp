#include "dijkstra.hpp"
#include "graph.hpp"

#include <iostream>
#include <limits>
#include <string>
#include <vector>


bool compare_distances(
    const std::vector<Distance>& actual,
    const std::vector<Distance>& expected,
    const std::string& test_name
)
{
    if (actual.size() != expected.size()) {

        std::cerr
            << "[FAIL] "
            << test_name
            << ": distance vector size mismatch\n";

        return false;
    }


    for (std::size_t i = 0;
         i < actual.size();
         ++i) {

        if (actual[i] != expected[i]) {

            std::cerr
                << "[FAIL] "
                << test_name
                << ": vertex "
                << (i + 1)
                << " expected "
                << expected[i]
                << " but got "
                << actual[i]
                << '\n';

            return false;
        }
    }


    std::cout
        << "[PASS] "
        << test_name
        << '\n';

    return true;
}


int main()
{
    bool all_passed = true;


    /*
     * -----------------------------------------
     * TEST 1
     *
     * tiny.gr, source = DIMACS vertex 1
     * -----------------------------------------
     */

    {
        CSRGraph graph =
            load_dimacs_graph(
                "data/tiny.gr"
            );


        DijkstraResult result =
            dijkstra(
                graph,
                0
            );


        std::vector<Distance> expected = {
            0,
            3,
            1,
            4
        };


        all_passed &=
            compare_distances(
                result.distances,
                expected,
                "tiny.gr source 1"
            );
    }


    /*
     * -----------------------------------------
     * TEST 2
     *
     * tiny.gr, source = DIMACS vertex 3
     * -----------------------------------------
     */

    {
        CSRGraph graph =
            load_dimacs_graph(
                "data/tiny.gr"
            );


        DijkstraResult result =
            dijkstra(
                graph,
                2
            );


        const Distance INF =
            std::numeric_limits<
                Distance
            >::max();


        std::vector<Distance> expected = {
            INF,
            2,
            0,
            3
        };


        all_passed &=
            compare_distances(
                result.distances,
                expected,
                "tiny.gr source 3"
            );
    }


    /*
     * -----------------------------------------
     * TEST 3
     *
     * test2.gr, source = DIMACS vertex 1
     * -----------------------------------------
     */

    {
        CSRGraph graph =
            load_dimacs_graph(
                "data/test2.gr"
            );


        DijkstraResult result =
            dijkstra(
                graph,
                0
            );


        std::vector<Distance> expected = {
            0,
            7,
            9,
            20,
            20,
            11
        };


        all_passed &=
            compare_distances(
                result.distances,
                expected,
                "test2.gr source 1"
            );
    }


    std::cout
        << '\n';


    if (!all_passed) {

        std::cerr
            << "ONE OR MORE TESTS FAILED\n";

        return 1;
    }


    std::cout
        << "ALL DIJKSTRA TESTS PASSED\n";

    return 0;
}
