CXX = g++
CXXFLAGS = -O3 -std=c++20 -Wall -Wextra -Iinclude

all: run_dijkstra run_delta

run_dijkstra: src/graph.cpp src/dijkstra.cpp src/run_dijkstra.cpp include/graph.hpp include/dijkstra.hpp
	$(CXX) $(CXXFLAGS) src/graph.cpp src/dijkstra.cpp src/run_dijkstra.cpp -o run_dijkstra

run_delta: src/graph.cpp src/dijkstra.cpp src/delta_stepping.cpp src/run_delta.cpp include/graph.hpp include/dijkstra.hpp include/delta_stepping.hpp
	$(CXX) $(CXXFLAGS) src/graph.cpp src/dijkstra.cpp src/delta_stepping.cpp src/run_delta.cpp -o run_delta

clean:
	rm -f run_dijkstra run_delta
