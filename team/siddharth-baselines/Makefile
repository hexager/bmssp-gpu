CXX = g++
CXXFLAGS = -O3 -std=c++20 -Wall -Wextra -Iinclude

all: test_loader run_dijkstra test_dijkstra benchmark_cpu test_delta run_delta

test_loader: src/graph.cpp src/test_loader.cpp include/graph.hpp
	$(CXX) $(CXXFLAGS) src/graph.cpp src/test_loader.cpp -o test_loader

run_dijkstra: src/graph.cpp src/dijkstra.cpp src/run_dijkstra.cpp include/graph.hpp include/dijkstra.hpp
	$(CXX) $(CXXFLAGS) src/graph.cpp src/dijkstra.cpp src/run_dijkstra.cpp -o run_dijkstra

test_dijkstra: src/graph.cpp src/dijkstra.cpp src/test_dijkstra.cpp include/graph.hpp include/dijkstra.hpp
	$(CXX) $(CXXFLAGS) src/graph.cpp src/dijkstra.cpp src/test_dijkstra.cpp -o test_dijkstra

benchmark_cpu: src/graph.cpp src/dijkstra.cpp src/benchmark_cpu.cpp include/graph.hpp include/dijkstra.hpp
	$(CXX) $(CXXFLAGS) src/graph.cpp src/dijkstra.cpp src/benchmark_cpu.cpp -o benchmark_cpu

test_delta: src/graph.cpp src/dijkstra.cpp src/delta_stepping.cpp src/test_delta.cpp include/graph.hpp include/dijkstra.hpp include/delta_stepping.hpp
	$(CXX) $(CXXFLAGS) src/graph.cpp src/dijkstra.cpp src/delta_stepping.cpp src/test_delta.cpp -o test_delta
run_delta: src/graph.cpp src/dijkstra.cpp src/delta_stepping.cpp src/run_delta.cpp include/graph.hpp include/dijkstra.hpp include/delta_stepping.hpp
	$(CXX) $(CXXFLAGS) src/graph.cpp src/dijkstra.cpp src/delta_stepping.cpp src/run_delta.cpp -o run_delta

test: test_loader test_dijkstra test_delta
	./test_loader data/tiny.gr
	./test_dijkstra
	./test_delta

clean:
	rm -f test_loader run_dijkstra test_dijkstra benchmark_cpu test_delta run_delta
