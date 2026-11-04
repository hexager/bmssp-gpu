#include "dijkstra.hpp"
#include "graph.hpp"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>


int main(int argc, char** argv)
{
    if (argc != 5) {
        std::cerr
            << "Usage: "
            << argv[0]
            << " <graph.gr> <sources> <repetitions> <output.csv>\n";

        return 1;
    }

    try {

        std::string filename = argv[1];
        int requested_sources = std::stoi(argv[2]);
        int repetitions = std::stoi(argv[3]);
        std::string output_file = argv[4];

        if (requested_sources <= 0 || repetitions <= 0) {
            throw std::runtime_error(
                "Sources and repetitions must be positive."
            );
        }

        std::cout << "Loading graph...\n";

        CSRGraph graph =
            load_dimacs_graph(filename);

        std::cout
            << "Vertices: "
            << graph.num_vertices
            << '\n';

        std::cout
            << "Edges: "
            << graph.num_edges
            << '\n';


        /*
         * ------------------------------------------------
         * Select useful deterministic sources.
         *
         * We only accept a source if Dijkstra examines
         * at least 10% of the graph's edges.
         *
         * This avoids isolated / nearly-isolated RMAT
         * sources producing meaningless timings.
         * ------------------------------------------------
         */

        std::vector<int> sources;

        const std::uint64_t minimum_edges =
            static_cast<std::uint64_t>(
                graph.num_edges / 10
            );


        /*
         * Generate deterministic candidate vertices
         * spread across the graph.
         */
        const int candidate_count =
            std::min(
                graph.num_vertices,
                requested_sources * 100
            );


        std::cout
            << "Selecting useful sources...\n";


        for (int i = 0;
             i < candidate_count &&
             static_cast<int>(sources.size()) <
                 requested_sources;
             ++i) {

            long long position =
                (
                    static_cast<long long>(i) *
                    graph.num_vertices
                )
                /
                candidate_count;

            int source =
                static_cast<int>(position);


            DijkstraResult probe =
                dijkstra(
                    graph,
                    source
                );


            if (probe.edges_examined >=
                minimum_edges) {

                sources.push_back(source);

                std::cout
                    << "Accepted source "
                    << (source + 1)
                    << " ("
                    << probe.edges_examined
                    << " edges reached)\n";
            }
        }


        if (sources.empty()) {

            throw std::runtime_error(
                "Could not find a useful source vertex."
            );
        }


        if (static_cast<int>(sources.size()) <
            requested_sources) {

            std::cout
                << "Warning: requested "
                << requested_sources
                << " sources but found "
                << sources.size()
                << ".\n";
        }


        std::ofstream csv(output_file);

        if (!csv.is_open()) {

            throw std::runtime_error(
                "Could not create output CSV."
            );
        }


        csv
            << "algorithm,"
            << "graph,"
            << "vertices,"
            << "edges,"
            << "source,"
            << "repetition,"
            << "runtime_ms,"
            << "edges_examined,"
            << "successful_relaxations\n";


        std::vector<double> all_times;


        std::cout
            << "\nRunning benchmark...\n";


        for (int source : sources) {

            /*
             * Warm-up for this source.
             */
            dijkstra(
                graph,
                source
            );


            std::vector<double> source_times;


            for (int repetition = 0;
                 repetition < repetitions;
                 ++repetition) {

                DijkstraResult result =
                    dijkstra(
                        graph,
                        source
                    );


                source_times.push_back(
                    result.runtime_ms
                );

                all_times.push_back(
                    result.runtime_ms
                );


                csv
                    << "dijkstra,"
                    << filename << ','
                    << graph.num_vertices << ','
                    << graph.num_edges << ','
                    << (source + 1) << ','
                    << (repetition + 1) << ','
                    << std::fixed
                    << std::setprecision(6)
                    << result.runtime_ms << ','
                    << result.edges_examined << ','
                    << result.successful_relaxations
                    << '\n';
            }


            std::sort(
                source_times.begin(),
                source_times.end()
            );


            double source_median;

            std::size_t n =
                source_times.size();


            if (n % 2 == 1) {

                source_median =
                    source_times[n / 2];
            }

            else {

                source_median =
                    (
                        source_times[n / 2 - 1] +
                        source_times[n / 2]
                    )
                    /
                    2.0;
            }


            std::cout
                << "Source "
                << (source + 1)
                << " median: "
                << source_median
                << " ms\n";
        }


        csv.close();


        std::sort(
            all_times.begin(),
            all_times.end()
        );


        double mean =
            std::accumulate(
                all_times.begin(),
                all_times.end(),
                0.0
            )
            /
            all_times.size();


        double median;

        std::size_t n =
            all_times.size();


        if (n % 2 == 1) {

            median =
                all_times[n / 2];
        }

        else {

            median =
                (
                    all_times[n / 2 - 1] +
                    all_times[n / 2]
                )
                /
                2.0;
        }


        std::cout
            << "\n============================\n";

        std::cout
            << "BENCHMARK SUMMARY\n";

        std::cout
            << "============================\n";

        std::cout
            << "Sources     : "
            << sources.size()
            << '\n';

        std::cout
            << "Repetitions : "
            << repetitions
            << '\n';

        std::cout
            << "Total runs  : "
            << all_times.size()
            << '\n';

        std::cout
            << "Mean        : "
            << mean
            << " ms\n";

        std::cout
            << "Median      : "
            << median
            << " ms\n";

        std::cout
            << "Min         : "
            << all_times.front()
            << " ms\n";

        std::cout
            << "Max         : "
            << all_times.back()
            << " ms\n";

        std::cout
            << "\nCSV written to: "
            << output_file
            << '\n';
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
