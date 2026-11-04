// Batched GPU FindPivots for several independent BMSSP searches on the same graph (see fp_batch.cu).
#pragma once
#include <vector>

struct FpBatch;

struct FpbStats {
    double upload_ms = 0, rounds_ms = 0, download_ms = 0;
    long long upload_vertices = 0, download_vertices = 0, edges = 0, calls = 0;
    int launches = 0, rounds = 0, batches = 0;
};

// One FindPivots call of search `search`. d/pred/hops are that search's host arrays (n entries each).
// dirty lists the vertices its CPU changed since its previous call (ignored on the search's first call).
struct FpbReq {
    int search;
    long long* d; int* pred; int* hops;
    const int* dirty; int ndirty;
    long long Bd; int Bh, Bv, Bp;
    const std::vector<int>* S; std::vector<int>* P; std::vector<int>* W;
};

FpBatch* fpb_create(int n, int searches, const int* off, const int* to, const long long* w);
void fpb_destroy(FpBatch* g);
const char* fpb_device_name(FpBatch* g);
// Runs all requests together (at most one per search). Writes P, W and every changed vertex back.
void fpb_run(FpBatch* g, std::vector<FpbReq>& reqs, int k, FpbStats* st);
