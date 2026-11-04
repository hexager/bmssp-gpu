// GPU FindPivots for the BMSSP hybrid.
// Semantics: synchronous Bellman-Ford rounds, identical to the serial path of Aniketh's OpenMP v3
// (week2_research_tools/fp_openmp_v3.cpp). Every round reads the state at the start of the round and
// commits the minimum key (dist, hops, pred) per vertex, so the result does not depend on thread order.
//
// This header is plain C++17 so the CUDA file never includes the C++20 solver header.
#pragma once
#include <vector>

struct FpGpu;

struct FpGpuStats {             // filled per call, accumulated by the caller
    double upload_ms = 0;       // host-to-device state upload before the call
    double rounds_ms = 0;       // all relaxation rounds, including per-round small transfers
    double download_ms = 0;     // device-to-host copy and host scatter of changed vertices
    double kernel_ms = 0;       // inside rounds: kernel launches until the device is idle
    double xfer_ms = 0;         // inside rounds: host<->device copies and memsets
    double host_ms = 0;         // inside rounds: host bookkeeping (prefix sums, sorting, W and P)
    long long upload_vertices = 0;
    long long download_vertices = 0;
    long long edges = 0;        // edges examined over all rounds
    int launches = 0;
    int rounds = 0;
};

// Copies the CSR graph to the device once. off has n+1 entries.
FpGpu* fpgpu_create(int n, const int* off, const int* to, const long long* w);
void fpgpu_destroy(FpGpu* g);
const char* fpgpu_device_name(FpGpu* g);

// Optional: page-lock the solver's state arrays for faster copies. Returns false if not possible.
bool fpgpu_pin_host(FpGpu* g, long long* d, int* pred, int* hops);

// Makes the device state equal to the host state for the listed vertices (list == nullptr: all n).
void fpgpu_upload_state(FpGpu* g, const long long* d, const int* pred, const int* hops,
                        const int* list, int nlist, FpGpuStats* st);

// One FindPivots(B, S) call. B = (Bd, Bh, Bv, Bp). Writes every changed vertex back into d/pred/hops.
// P and W are cleared and filled with the same order as the serial synchronous version.
void fpgpu_findpivots(FpGpu* g, long long* d, int* pred, int* hops, int k,
                      long long Bd, int Bh, int Bv, int Bp, const std::vector<int>& S,
                      std::vector<int>& P, std::vector<int>& W, FpGpuStats* st);
