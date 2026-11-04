#pragma once
// Counters for BMSSP (findPivots / baseCase / bmsspRec). Add hooks once Aniketh code is available.
#include <chrono>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <cstdio>
struct SizeStat { uint64_t n=0,total=0,mx=0; void add(size_t s){n++;total+=s;mx=std::max<uint64_t>(mx,s);} };
struct Metrics {
  uint64_t edge_scans=0,update_calls=0,relaxations=0,fp_calls=0,bc_calls=0,rec_calls=0;
  std::vector<uint64_t> rec_by_level; SizeStat S,W,P,U; double fp_ns=0,bc_ns=0,rec_ns=0;
  void level(int l){ if((int)rec_by_level.size()<=l) rec_by_level.resize(l+1,0); rec_by_level[l]++; }
  void reset(){ *this=Metrics(); }
};
inline Metrics g_m;
struct ScopedTimer { double& acc; std::chrono::steady_clock::time_point t0;
  explicit ScopedTimer(double& a):acc(a),t0(std::chrono::steady_clock::now()){}
  ~ScopedTimer(){ acc+=std::chrono::duration<double,std::nano>(std::chrono::steady_clock::now()-t0).count(); } };
