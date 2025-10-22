#include <metal_stdlib>
#include <metal_atomic>

#define USE_CELL_INDICES

#include "../libs/geometry.h"
#include "../libs/pcg/pcg_basic.c"
#include "./MetalIsingMetropolis.hpp"

using namespace metal;

kernel void initialize(
  device       int8_t         *lattice [[buffer(0)]],
  device const uint32_t       &L       [[buffer(1)]],
  device       pcg32_random_t *rng     [[buffer(2)]],
  device const uint64_t       *random  [[buffer(3)]],
  uint2    index   [[thread_position_in_grid]]
) {
  int plainIdx = index.y*L + index.x;
#if LATTICE==LATTICE_HEX
  pcg32_random_t local_rng1 = rng[2*plainIdx  ];
  pcg32_random_t local_rng2 = rng[2*plainIdx+1];
  pcg32_srandom_r( &local_rng1, random[4*plainIdx  ], random[4*plainIdx+1] );
  pcg32_srandom_r( &local_rng2, random[4*plainIdx+2], random[4*plainIdx+3] );
  rng[2*plainIdx  ] = local_rng1;
  rng[2*plainIdx+1] = local_rng2;
  lattice[2*plainIdx  ]=1;
  lattice[2*plainIdx+1]=1;
#else
  pcg32_random_t local_rng = rng[plainIdx];
  pcg32_srandom_r( &local_rng, random[2*plainIdx  ], random[2*plainIdx+1] );
  rng[plainIdx] = local_rng;
  lattice[plainIdx]=1;
#endif
}

inline uint plainIndex(uint2 index, uint L) {
  return L*index.x + index.y;
}

#if LATTICE==LATTICE_SQR

inline int8_t neigh1(uint2 actualIdx, uint h, uint w, device const int8_t* lattice) 
  { return lattice[ plainIndex( uint2( (actualIdx.x  +1)%h,  actualIdx.y       ), w ) ]; }
inline int8_t neigh2(uint2 actualIdx, uint h, uint w, device const int8_t* lattice) 
  { return lattice[ plainIndex( uint2( (actualIdx.x+h-1)%h,  actualIdx.y       ), w ) ]; }
inline int8_t neigh3(uint2 actualIdx, uint h, uint w, device const int8_t* lattice) 
  { return lattice[ plainIndex( uint2(  actualIdx.x       , (actualIdx.y  +1)%w), w ) ]; }
inline int8_t neigh4(uint2 actualIdx, uint h, uint w, device const int8_t* lattice) 
  { return lattice[ plainIndex( uint2(  actualIdx.x       , (actualIdx.y+w-1)%w), w ) ]; }

#elif LATTICE==LATTICE_TRI

inline int8_t neigh1(uint2 actualIdx, uint h, uint w, device const int8_t* lattice) 
  { return lattice[ plainIndex( uint2( (actualIdx.x  +1)%h,  actualIdx.y       ), w ) ]; }
inline int8_t neigh2(uint2 actualIdx, uint h, uint w, device const int8_t* lattice) 
  { return lattice[ plainIndex( uint2( (actualIdx.x+h-1)%h,  actualIdx.y       ), w ) ]; }
inline int8_t neigh3(uint2 actualIdx, uint h, uint w, device const int8_t* lattice) 
  { return lattice[ plainIndex( uint2(  actualIdx.x       , (actualIdx.y  +1)%w), w ) ]; }
inline int8_t neigh4(uint2 actualIdx, uint h, uint w, device const int8_t* lattice) 
  { return lattice[ plainIndex( uint2(  actualIdx.x       , (actualIdx.y+w-1)%w), w ) ]; }
inline int8_t neigh5(uint2 actualIdx, uint h, uint w, device const int8_t* lattice) 
  { return lattice[ plainIndex( uint2( (actualIdx.x  +1)%h, (actualIdx.y  +1)%w), w ) ]; }
inline int8_t neigh6(uint2 actualIdx, uint h, uint w, device const int8_t* lattice) 
  { return lattice[ plainIndex( uint2( (actualIdx.x+h-1)%h, (actualIdx.y+w-1)%w), w ) ]; }

#elif LATTICE==LATTICE_HEX

inline int8_t neigh01(uint2 actualIdx, uint h, uint w, device const int8_t* lattice) 
  { return lattice[ plainIndex( uint2( (actualIdx.x  +1)%h, (actualIdx.y+w-1)%w), w ) ]; }
inline int8_t neigh02(uint2 actualIdx, uint h, uint w, device const int8_t* lattice) 
  { return lattice[ plainIndex( uint2( (actualIdx.x+h-1)%h,  actualIdx.y  +1   ), w ) ]; }
inline int8_t neigh03(uint2 actualIdx, uint h, uint w, device const int8_t* lattice) 
  { return lattice[ plainIndex( uint2(  actualIdx.x       ,  actualIdx.y  +1   ), w ) ]; }
  // %w not necessary due to geometry

inline int8_t neigh11(uint2 actualIdx, uint h, uint w, device const int8_t* lattice) 
  { return lattice[ plainIndex( uint2( (actualIdx.x+h-1)%h, (actualIdx.y  +1)%w), w ) ]; }
inline int8_t neigh12(uint2 actualIdx, uint h, uint w, device const int8_t* lattice) 
  { return lattice[ plainIndex( uint2( (actualIdx.x  +1)%h,  actualIdx.y  -1   ), w ) ]; }
inline int8_t neigh13(uint2 actualIdx, uint h, uint w, device const int8_t* lattice) 
  { return lattice[ plainIndex( uint2(  actualIdx.x       ,  actualIdx.y  -1   ), w ) ]; }
  // %w not necessary due to geometry

#endif

kernel void update(
  device       int8_t         *lattice [[buffer(0)]],
  device       pcg32_random_t *rng     [[buffer(1)]],
  device const uint32_t       *probs   [[buffer(2)]],
  device const uint32_t       &rOffset [[buffer(3)]],
  device const uint32_t       &cOffset [[buffer(4)]],
  uint2 index    [[thread_position_in_grid]],
  uint2 locIdx   [[thread_position_in_threadgroup]],
  uint2 thrgSize [[threads_per_threadgroup]],
  uint2 gridSize [[threads_per_grid]]
) {
  uint2   actualIdx      = 2*index+uint2(cOffset,rOffset);
  uint2   actualGridSize = 2*gridSize;

  uint h = actualGridSize.x;
  uint w = actualGridSize.y;

  uint    linearIdx = plainIndex(actualIdx,w);

  // load lattice (optimized for lattice geometry)
  int8_t current = lattice[ linearIdx ];

  pcg32_random_t local_rng = rng[ linearIdx ];
  uint alignedNeighs = 0;
  bool is_equal;

#if LATTICE==LATTICE_SQR
  is_equal = current == neigh1( actualIdx, h, w, lattice );
  alignedNeighs += select((int)0, (int)1, is_equal);
  is_equal = current == neigh2( actualIdx, h, w, lattice );
  alignedNeighs += select((int)0, (int)1, is_equal);
  is_equal = current == neigh3( actualIdx, h, w, lattice );
  alignedNeighs += select((int)0, (int)1, is_equal);
  is_equal = current == neigh4( actualIdx, h, w, lattice );
  alignedNeighs += select((int)0, (int)1, is_equal);
#elif LATTICE==LATTICE_TRI
  is_equal = current == neigh1( actualIdx, h, w, lattice );
  alignedNeighs += select((int)0, (int)1, is_equal);
  is_equal = current == neigh2( actualIdx, h, w, lattice );
  alignedNeighs += select((int)0, (int)1, is_equal);
  is_equal = current == neigh3( actualIdx, h, w, lattice );
  alignedNeighs += select((int)0, (int)1, is_equal);
  is_equal = current == neigh4( actualIdx, h, w, lattice );
  alignedNeighs += select((int)0, (int)1, is_equal);
  is_equal = current == neigh5( actualIdx, h, w, lattice );
  alignedNeighs += select((int)0, (int)1, is_equal);
  is_equal = current == neigh6( actualIdx, h, w, lattice );
  alignedNeighs += select((int)0, (int)1, is_equal);
#elif LATTICE==LATTICE_HEX
  if(rOffset == 0) {
    // We're on even lattice
    is_equal = current == neigh01( actualIdx, h, w, lattice );
    alignedNeighs += select((int)0, (int)1, is_equal);
    is_equal = current == neigh02( actualIdx, h, w, lattice );
    alignedNeighs += select((int)0, (int)1, is_equal);
    is_equal = current == neigh03( actualIdx, h, w, lattice );
    alignedNeighs += select((int)0, (int)1, is_equal);
  } else {
    // Using odds's neighbors
    is_equal = current == neigh11( actualIdx, h, w, lattice );
    alignedNeighs += select((int)0, (int)1, is_equal);
    is_equal = current == neigh12( actualIdx, h, w, lattice );
    alignedNeighs += select((int)0, (int)1, is_equal);
    is_equal = current == neigh13( actualIdx, h, w, lattice );
    alignedNeighs += select((int)0, (int)1, is_equal);
  }
#endif

  bool flip_condition = pcg32_random_r(&local_rng) < probs[alignedNeighs];
  lattice[linearIdx] = (int8_t) select(current,-current,flip_condition);
  rng[linearIdx] = local_rng;
}

kernel void redsum(
  device const int64_t* input1          [[buffer(0)]],
  device const int64_t* input2          [[buffer(1)]],
  device       int64_t* partial_sums1   [[buffer(2)]],
  device       int64_t* partial_sums2   [[buffer(3)]],
  device       size_t  &total_elements  [[buffer(4)]],
  uint tid            [[thread_index_in_threadgroup]],
  uint bid           [[threadgroup_position_in_grid]]
) {
  threadgroup int64_t shared_data1[MSL_THSPTHG];
  threadgroup int64_t shared_data2[MSL_THSPTHG];
  
  uint index = bid * MSL_THSPTHG + tid;

  if(index < total_elements) {
    shared_data1[tid] = input1[index];
    shared_data2[tid] = input2[index];
  } else {
    shared_data1[tid] = 0;
    shared_data2[tid] = 0;
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);
  
  // Reduction
  for (uint s = MSL_THSPTHG/2; s > 2; s >>= 1) {
      if (tid < s) {
          shared_data1[tid] += shared_data1[tid + s];
          shared_data2[tid] += shared_data2[tid + s];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  
  if (tid == 0) {
      partial_sums1[bid] = shared_data1[0] + shared_data1[1] + shared_data1[2] + shared_data1[3];
      partial_sums2[bid] = shared_data2[0] + shared_data2[1] + shared_data2[2] + shared_data2[3];
  }
}

kernel void energy_magnetization(
  device const int8_t   *lattice        [[buffer(0)]],
  device       int64_t  *enerRed        [[buffer(1)]],
  device       int64_t  *magnRed        [[buffer(2)]],
  device       uint32_t &L              [[buffer(3)]],
  uint2 index    [[thread_position_in_grid]],
  uint2 gridSize [[threads_per_grid]],
  uint2 idxInThg [[thread_position_in_threadgroup]],
  uint2 tIdxInGr [[threadgroup_position_in_grid]],
  uint2 thgInGr  [[threadgroups_per_grid]]
) {
  threadgroup int64_t shared_magn[MSL_THSPTHG], shared_ener[MSL_THSPTHG];

  uint h = L;
#if LATTICE==LATTICE_HEX
  uint w = 2*L;
#else
  uint w = L;
#endif

  uint linearIdx      = plainIndex(index,w);
  uint linearIdxInThg = plainIndex(idxInThg, MSL_THSPTHG/MSL_THSPTHGSQR);

  int64_t energy;
  int64_t magnet;

#if LATTICE==LATTICE_HEX
  if( index.x < L && index.y < 2*L ) {
#else
  if( index.x < L && index.y < L ) {
#endif
    magnet = (int64_t) lattice[ linearIdx ];
#if LATTICE==LATTICE_SQR 
    energy = - magnet * (
      neigh1( index, h, w, lattice ) +
      neigh2( index, h, w, lattice ) +
      neigh3( index, h, w, lattice ) +
      neigh4( index, h, w, lattice )
    );
#elif LATTICE==LATTICE_TRI
    energy = - magnet * (
      neigh1( index, h, w, lattice ) +
      neigh2( index, h, w, lattice ) +
      neigh3( index, h, w, lattice ) +
      neigh4( index, h, w, lattice )
    );
#elif LATTICE==LATTICE_HEX
    bool isOdd = (index.y) & 0x01;
    energy = - magnet * (
      select( neigh01( index, h, w, lattice ), neigh11( index, h, w, lattice ), isOdd ) +
      select( neigh02( index, h, w, lattice ), neigh12( index, h, w, lattice ), isOdd ) +
      select( neigh03( index, h, w, lattice ), neigh13( index, h, w, lattice ), isOdd )
    );
#endif
  } else {
    magnet = 0;
    energy = 0;
  }

  shared_magn[linearIdxInThg] = magnet;
  shared_ener[linearIdxInThg] = energy;

  threadgroup_barrier(mem_flags::mem_threadgroup);
  
  // Reduction
  for (uint s = MSL_THSPTHG/2; s > 2; s >>= 1) {
      if (linearIdxInThg < s) {
          shared_magn[linearIdxInThg] += shared_magn[linearIdxInThg + s];
          shared_ener[linearIdxInThg] += shared_ener[linearIdxInThg + s];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  
  if (linearIdxInThg == 0) {
      uint bid = plainIndex(tIdxInGr, thgInGr.y);
      enerRed[bid] = shared_ener[0] + shared_ener[1] + shared_ener[2] + shared_ener[3];
      magnRed[bid] = shared_magn[0] + shared_magn[1] + shared_magn[2] + shared_magn[3];
  }
}