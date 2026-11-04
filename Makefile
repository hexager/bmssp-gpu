# BMSSP hybrid (CPU recursion + GPU FindPivots)
#   make            GPU builds (need nvcc) + emulation builds + reference Dijkstra
#   make emu        emulation build only (no GPU needed)
# ARCH defaults to the GPU in this machine (-arch=native, CUDA 11.6+). Build elsewhere with e.g.
#   make ARCH="-gencode arch=compute_75,code=sm_75"     (GTX 1660 Ti; CUDA 13 no longer builds sm_61)
# If nvcc rejects your gcc version, point it at another host compiler: make NVCC_CCBIN=/usr/bin/g++-13
NVCC       ?= nvcc
CXX        ?= g++
NVCC_CCBIN ?=
ARCH       ?= -arch=native
CCBIN      := $(if $(NVCC_CCBIN),-ccbin $(NVCC_CCBIN),)

all: bin/fp_hybrid bin/fp_hybrid_emu bin/nearfar bin/nearfar_emu bin/dijkstra_ref bin/breakdown bin/fp_multi bin/fp_multi_emu bin/gen_graph
emu: bin/fp_hybrid_emu bin/nearfar_emu bin/dijkstra_ref bin/breakdown bin/fp_multi_emu bin/gen_graph

bin:
	mkdir -p bin

bin/fp_hybrid.o: src/fp_hybrid.cpp src/bmssp_hybrid.hpp src/fp_gpu.h src/sync_fp.hpp | bin
	$(CXX) -O3 -std=c++20 -c src/fp_hybrid.cpp -o $@

bin/fp_gpu.o: src/fp_gpu.cu src/fp_gpu.h | bin
	$(NVCC) $(CCBIN) -O3 -std=c++17 $(ARCH) -c src/fp_gpu.cu -o $@

bin/fp_gpu_emu.o: src/fp_gpu.cu src/fp_gpu.h | bin
	$(CXX) -O2 -std=c++17 -DFP_EMULATE -x c++ -c src/fp_gpu.cu -o $@

bin/fp_hybrid: bin/fp_hybrid.o bin/fp_gpu.o
	$(NVCC) $(CCBIN) $(ARCH) bin/fp_hybrid.o bin/fp_gpu.o -o $@

bin/fp_hybrid_emu: bin/fp_hybrid.o bin/fp_gpu_emu.o
	$(CXX) bin/fp_hybrid.o bin/fp_gpu_emu.o -o $@

bin/dijkstra_ref: src/dijkstra_ref.cpp | bin
	$(CXX) -O3 -std=c++17 src/dijkstra_ref.cpp -o $@

bin/nearfar: src/nearfar.cu | bin
	$(NVCC) $(CCBIN) -O3 -std=c++17 $(ARCH) src/nearfar.cu -o $@

bin/nearfar_emu: src/nearfar.cu | bin
	$(CXX) -O2 -std=c++17 -DFP_EMULATE -x c++ src/nearfar.cu -o $@

src/bmssp_breakdown.hpp: src/bmssp_hybrid.hpp src/make_breakdown_header.py
	python3 -I src/make_breakdown_header.py src/bmssp_hybrid.hpp $@

bin/breakdown: src/breakdown.cpp src/bmssp_breakdown.hpp | bin
	$(CXX) -O3 -std=c++20 src/breakdown.cpp -o $@

bin/fp_multi.o: src/fp_multi.cpp src/bmssp_hybrid.hpp src/fp_batch.h src/sync_fp.hpp | bin
	$(CXX) -O3 -std=c++20 -c src/fp_multi.cpp -o $@

bin/fp_batch.o: src/fp_batch.cu src/fp_batch.h | bin
	$(NVCC) $(CCBIN) -O3 -std=c++17 $(ARCH) -c src/fp_batch.cu -o $@

bin/fp_batch_emu.o: src/fp_batch.cu src/fp_batch.h | bin
	$(CXX) -O2 -std=c++17 -DFP_EMULATE -x c++ -c src/fp_batch.cu -o $@

bin/fp_multi: bin/fp_multi.o bin/fp_batch.o
	$(NVCC) $(CCBIN) $(ARCH) bin/fp_multi.o bin/fp_batch.o -Xcompiler -pthread -o $@

bin/fp_multi_emu: bin/fp_multi.o bin/fp_batch_emu.o
	$(CXX) bin/fp_multi.o bin/fp_batch_emu.o -pthread -o $@

bin/gen_graph: src/gen_graph.cpp | bin
	$(CXX) -O2 -std=c++17 src/gen_graph.cpp -o $@

clean:
	rm -rf bin

.PHONY: all emu clean
