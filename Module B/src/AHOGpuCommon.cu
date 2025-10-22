#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#pragma GCC diagnostic ignored "-Wconversion"

#define __extension__

#include <cstdint>
#include <errors.h>
#include <progressBar.h>
#include <iomanip>
#include <pcg/pcg_basic.c>
#include <iostream>
#include <fstream>
#include <math_constants.h>

#pragma GCC diagnostic pop

using namespace std;

constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint32_t SAMPLES_PER_COMMIT = 100;
constexpr uint32_t MAX_X_POW   = 4;
constexpr uint32_t MAX_LAGSC   = 6;
constexpr float   TWOTOM32L   = 1.0 / 4294967296.0;
constexpr float   EPSILON     = 1.0 / 16777216.0;
constexpr uint32_t pair_counts = MAX_X_POW * MAX_X_POW / 2;

/* - Functions -------------------------------------------------------------- */
__device__ inline float randomFloat( pcg32_random_t* rng ) {
  return (float)pcg32_random_r(rng) * TWOTOM32L;
}

// It samples two variables from normal distribution
__device__ inline void boxMuller(
  pcg32_random_t* rng,
  float &out1,
  float &out2
 ) {
  // seed1 ^ 2 in order to skip 2 * log
  // seed2 * 2 in order to have it in the range 0, 2pi (after scaling)
  float seed1 = randomFloat( rng );
  seed1 = seed1 * seed1 + EPSILON;
  float seed2 = 2.0  * randomFloat( rng );
  float r = sqrt( -log( seed1 ) );
  out1 = r * cospi( seed2 );
  out2 = r * sinpi( seed2 );
}

// Find the next position
__device__ inline float neigh_p(const float* path, uint32_t idx, uint32_t n) {
  return path[(idx  +1)%n];
}

// Find the previous position
__device__ inline float neigh_m(const float* path, uint32_t idx, uint32_t n) {
  return path[(idx+n-1)%n];
}

__host__ inline uint32_t nextNumBlock( uint32_t num_blocks ) {
  return ( num_blocks + BLOCK_SIZE - 1 ) / BLOCK_SIZE;
}

/* - GPU kernels ------------------------------------------------------------ */
__global__ void d_computePows(
  const float*   path,
  const uint32_t  n,
  const uint32_t  N,
  const uint32_t  ups,
  uint32_t*       accepted,
  float*         measures
) {
  // measure[o+N*i] is the measure about the i-th point,
  // where o is the observables number and runs from 0 to N-1,
  // o = 0               : accept/reject rate
  // o = 1..MAX_X_POW    : x^k from 1 to MAX_X_POW
  // o = MAX_X_POW+1...N : correlator of obs_i and obs_(i+tau)
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  if(idx<n) {
    measures[idx*N + 0] = (float)accepted[idx] / ups;
    float x = path[idx];
    float xn = 1.0;
    for(uint32_t i = 1; i <= MAX_X_POW; i++) {
      xn *= x;
      measures[idx*N + i] = xn;
    }
  }
}

__global__ void d_measures(
  const float*   path,
  const uint32_t  n,
  const uint32_t  N,
  const uint32_t  lagsc,
  const uint32_t* lagsv,
  float*         measures
) {
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  if(idx<n) {
    uint32_t base    = idx * N;
    uint32_t address = base + 1 + MAX_X_POW;
    // correlations
    for( uint32_t lag_idx = 0; lag_idx < lagsc; lag_idx++ ) {
      uint32_t base_2 = ( ( idx + lagsv[lag_idx] ) % n ) * N;
      for( uint32_t i = 1; i <= MAX_X_POW; i++ )
        for( uint32_t j = i; j <= MAX_X_POW; j++ )
          measures[ address++ ] = measures[base+i] * measures[base_2+j];
    }
  }
}

__global__ void d_redsum(
  const float*  measures,
  const uint32_t n,
  const uint32_t N,
  float*        output
) {
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ float sums[BLOCK_SIZE * (MAX_LAGSC+1) * MAX_X_POW * (MAX_X_POW + 1)/2];

  // Init sums
  for(uint32_t i=0; i<N; i++)
    sums[N*threadIdx.x + i] = 0.0;
  // Load values
  if( idx < n )
    for(uint32_t i=0; i<N; i++)
      sums[N*threadIdx.x + i] = measures[N*idx + i];
  __syncthreads();

  // Reduction
  for(uint32_t s = BLOCK_SIZE/2; s>0; s>>=1) {
    if(threadIdx.x < s)
      for(uint32_t i=0; i<N; i++)
        sums[N*threadIdx.x+i] += sums[N*(threadIdx.x+s)+i];
    __syncthreads();
  }

  // Write the output
  if(threadIdx.x == 0) {
    for(uint32_t i=0; i<N; i++) {
      output[N*blockIdx.x+i] = sums[i];
    }
  }
}