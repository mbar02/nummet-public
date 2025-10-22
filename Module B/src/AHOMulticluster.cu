#include "./AHOGpuCommon.cu"

/* - Functions -------------------------------------------------------------- */
// Compute alpha and beta coefficients
__device__ inline float alpha(float a, float g, float x0, float x2) {
  return sqrt(
    a/2. * ( x0 + 3. * g * x2 )
  );
}
__device__ inline float beta (float a, float g, float x1,   float x3, float alpha) {
  return a/2. * (
    x1 + g * x3
  ) / alpha;
}

// Compute potential energy density
__device__ inline float potential(float x2, float x4, float g, float a) {
  return a * ( x2 / 2.0 + g * x4 / 4.0 );
}

__device__ inline float annexProb(float delta_x, float a, float gamma) {
  return 1. - gamma * exp( delta_x * delta_x / (2.*a) );
}

/* - GPU kernels ------------------------------------------------------------ */
__global__ void d_init(
  float*         path,
  pcg32_random_t* rng,
  uint64_t*       rseed,
  uint64_t*       rstat,
  const uint32_t  n
) {
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < n) {
    // Set up RNG
    pcg32_random_t local_rng = rng[idx];
    pcg32_srandom_r(&local_rng, rseed[idx], rstat[idx]);  
    rng[idx] = local_rng;

    // Set up the path
    path[idx] = 0.0;
  }
}

__global__ void d_reset_accepted(
  uint32_t*       accepted,
  const uint32_t  n
) {
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < n) {
    accepted[idx] = 0;
  }
}

__global__ void d_initBuildClusters1(
  pcg32_random_t* rng,
  const float*   path,
  const float    a,
  const float    gamma,
  const uint32_t  n,
  uint8_t*        annexBits
) {
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < n) {
    pcg32_random_t local_rng = rng[idx];

    // Is the next site in the same cluster of this one?
    annexBits[idx] = randomFloat(&local_rng) < annexProb(
      path[idx]-neigh_p(path,idx,n), a, gamma
    );

    rng[idx] = local_rng;
  }
}

__global__ void d_initBuildClusters2(
  const uint32_t  n,
  uint8_t*        annexBits,
  uint32_t*       head,
  uint32_t*       queue
) {
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < n) {
    // Init head and queue (of each cluster)
    head[idx]  = ( idx - annexBits[(idx + n - 1) % n] + n) % n;
    queue[idx] = ( idx + annexBits[ idx ] ) % n;
  }
}

__global__ void d_buildClusters(
  const uint8_t*  annexBits,
  const uint32_t* old_head,
  const uint32_t* old_queue,
  const uint32_t  n,
  uint32_t*       new_head,
  uint32_t*       new_queue
) {
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < n) {
    new_head  [idx] = old_head  [ old_head  [idx] ];
    new_queue [idx] = old_queue [ old_queue [idx] ];
  }
}

__global__ void d_notOnlyOneCluster(
  float*         path,
  uint32_t*       head,
  uint32_t*       queue,
  const uint8_t*  annexBits,
  const uint32_t  n,
  float*         x0,
  float*         x1,
  float*         x2,
  float*         x3,
  float*         x4
) {
  // Check if not only one cluster
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < n) {
    if( annexBits[queue[idx]] ) {
      head  [idx] = 0;
      queue [idx] = n-1;
    }
    // Init for the sum of x^n
    x0[idx] = 1.0;
    x1[idx] =           path[idx];
    x2[idx] = x1[idx] * path[idx];
    x3[idx] = x2[idx] * path[idx];
    x4[idx] = x3[idx] * path[idx];
  }
}

__global__ void d_colorClusters(
  const uint32_t* head,
  const uint32_t* queue,
  const uint32_t* old_color,
  const uint32_t* old_wrt,
  const uint32_t  n,
  const uint32_t  delta,
  uint32_t*       new_color,
  uint32_t*       new_wrt,
  float*         x0,
  float*         x1,
  float*         x2,
  float*         x3,
  float*         x4
) {
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < n) {
    // color the cluster
    if(head[idx] == idx) {
      // Cluster zero contains idx = 0 or
      // the previous position is not in a cluster
      if( idx > queue[idx] || idx == 0 ) {
        new_wrt[idx]   = idx;
        new_color[idx] = 0;
      }
    } else {
      new_color[idx] = old_color[idx] + old_color[ old_wrt[idx] ];
      new_wrt[idx]   = old_wrt[ old_wrt[idx] ];
    }

    // meantime, compute x1, x2, x3, x4
    uint32_t idxPlusDelta = (idx+delta) % n;
    if(
      head[idxPlusDelta] == head[idx] &&              // in the same cluster
      (n+idx-head[idx])%n < (n+delta+idx-head[idx])%n // is the lower position
    ) {
      x0[idx] += x0[idx+delta];
      x1[idx] += x1[idx+delta];
      x2[idx] += x2[idx+delta];
      x3[idx] += x3[idx+delta];
      x4[idx] += x4[idx+delta];
    }
  }
}

__global__ void d_endColorCluster(
  const uint32_t* head,
  const uint32_t* queue,
  uint32_t* color,
  const uint32_t  n
) {
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < n) {
    // Check that there are no near cluster with the same color.
    // It can appen only between the 0th and the -1th cluster
    if( color[(queue[idx]+1)%n] == 0 && color[idx] % 3 == 0 )
      color[idx]++;
  }
}

__global__ void d_extract(
  const float*   path,
  const uint32_t* head,
  const uint32_t* queue,
  const uint32_t* color,
  const uint32_t  n,
  const float*   x0,
  const float*   x1,
  const float*   x2,
  const float*   x3,
  const float*   x4,
  const float    a,
  const float    gamma,
  const float    g,
  pcg32_random_t* rng,
  float*         delta
) {
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < n) {
    if( idx == head[idx] ) {
      pcg32_random_t local_rng = rng[idx];

      // Extract delta
      float old_alpha = alpha(a,g,x0[idx],x2[idx]);
      float old_beta  = beta (a,g,x1[idx],x3[idx],old_alpha);
      float delta_tmp;
      boxMuller(&local_rng, delta_tmp, delta_tmp);
      delta_tmp = delta_tmp / (old_alpha * M_SQRT2) - old_beta/old_alpha;

      // Compute accept/reject rate
      float x1b, x2b, x3b, delta2, delta3, delta4;
      delta2 = delta_tmp*delta_tmp;
      delta3 = delta2*delta_tmp;
      delta4 = delta3*delta_tmp;
      x1b = x0[idx] * delta_tmp + x1[idx];
      x2b = x0[idx] * delta2 + 2. * delta_tmp  * x1[idx] + x2[idx];
      x3b = x0[idx] * delta3 + 3. * delta2 * x1[idx] + 3. * delta_tmp * x2[idx] + x3[idx];

      float new_alpha = alpha(a,g,x0[idx],x2b);
      float new_beta  = beta (a,g,x1b,x3b,new_alpha);

      float newSqrtFreeAction = new_alpha * delta_tmp - new_beta;

      float old_p_max, old_p_min, raw_old_p_max, raw_old_p_min;

      raw_old_p_max = annexProb(
        path[queue[idx]] - neigh_p(path,queue[idx],n), a, gamma
      );
      raw_old_p_min = annexProb(
        path[      idx ] - neigh_m(path,      idx ,n), a, gamma
      );
      old_p_max = raw_old_p_max > 0.0 ? raw_old_p_max : 0.0;
      old_p_min = raw_old_p_min > 0.0 ? raw_old_p_min : 0.0;

      float new_p_max, new_p_min, raw_new_p_max, raw_new_p_min;

      raw_new_p_max = annexProb(
        path[queue[idx]] - neigh_p(path,queue[idx],n) + delta_tmp, a, gamma
      );
      raw_new_p_min = annexProb(
        path[      idx ] - neigh_m(path,      idx ,n) + delta_tmp, a, gamma
      );
      new_p_max = raw_new_p_max > 0.0 ? raw_new_p_max : 0.0;
      new_p_min = raw_new_p_min > 0.0 ? raw_new_p_min : 0.0;

      float ar = exp(
        log( new_alpha / old_alpha ) +
        old_beta*old_beta - newSqrtFreeAction*newSqrtFreeAction
        - a * g * ( x0[idx] * delta4/4. + x1[idx] * delta3 )
        + log( 1. - new_p_max ) - log( 1. - raw_new_p_max )
        + log( 1. - new_p_min ) - log( 1. - raw_new_p_min )
        - log( 1. - old_p_max ) + log( 1. - raw_old_p_max )
        - log( 1. - old_p_min ) + log( 1. - raw_old_p_min )
      );

      if(randomFloat(&local_rng)<ar) {
        delta[idx] = delta_tmp;
      }
      else {
        delta[idx] = 0.0;
      }

      rng[idx] = local_rng;
    }
  }
}

__global__ void d_update(
  float*         path,
  const uint32_t  n,
  const uint32_t* head,
  const float*   delta,
  uint32_t*       accepted,
  uint32_t*       color,
  uint32_t        j
) {
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if( idx<n ) {
    if ( j == color[head[idx]] % 3 ) {
      float dv = delta[head[idx]];
      if(dv!=0.0) {
        path[idx] += dv;
        accepted[idx]++;
      }
    }
  }
}

/* - CPU functions ---------------------------------------------------------- */
int main(int argc, char** argv) {
  // Parse arguments from command line
  /*
    Arguments: n ups samples beta g outputfile [lags for correlations]...
  */
  if(argc < 8) {
    cout << "Syntax: " << argv[0]
         << " [n] [updates per samples] [samples] [beta] [g] [outputfile]"
         << " [lags for correlations]... " << endl;
    return TEINVAL;
  }
  uint32_t n       = (uint32_t)std::stoul(argv[1]);
  uint32_t ups     = (uint32_t)std::stoul(argv[2]);
  uint32_t samples = (uint32_t)std::stoul(argv[3]);
  float   beta    = (float)  std::stod (argv[4]);
  float   g       = (float)  std::stod (argv[5]);

  float a = beta / (float)n;

  char* outfile = argv[6];

  uint32_t* h_lags = new uint32_t[argc-7];
  for(int i=0; i<argc-7; i++)
    h_lags[i] = (uint32_t)std::stoul(argv[7+i]);
  
  uint32_t lagsc = argc-7;

  if( n%2 ) {
    cout << "n must be even. Fallback to n+1." << endl;
    n++;
  }

  if(lagsc > MAX_LAGSC) {
    cerr << "There are too much lags. The maximum is "
         << MAX_LAGSC << "." << endl;
    return TEINVAL;
  }

  /* - Init the simulation (RNG, buffers, allocations, ...) ----------------- */
  // Output file stream
  ofstream outputfile(outfile);
  outputfile  << std::scientific        // Scientific notation
              << std::showpos           // Always show the sign (+/-)
              << std::setprecision(14); // 6 decimal digits

  // RNG
  pcg32_random_t* h_rng = new pcg32_random_t;
  pcg32_random_t* d_rng;
  cudaMalloc( &d_rng, n * sizeof(pcg32_random_t) );

  // Path
  float* d_path;
  cudaMalloc( &d_path, n * sizeof(float) );

  // Annex parameter
  float gamma = (float)2.0 / sqrt( (float)n );

  // Minimum 2^exponent > n
  uint32_t exponent = 0;
  {
    uint32_t m = n;
    while(m) {
      m <<= 2;
      exponent++;
    }
    exponent--;
  }

  // GPU stuff
  uint32_t num_blocks = ( n + BLOCK_SIZE - 1 ) / BLOCK_SIZE;

  const uint32_t pair_count = MAX_X_POW * (MAX_X_POW + 1)/2;
  uint32_t N = 1 + MAX_X_POW + lagsc * pair_count;

  cudaStream_t streams[2];              // double buffering
  cudaStreamCreate(&streams[0]);
  cudaStreamCreate(&streams[1]);

  uint32_t* d_accepted;
  cudaMalloc( &d_accepted, n * sizeof(uint32_t) );

  uint32_t *d_lags;                       // lags

  uint8_t*  d_annexBits;                  // for clusters
  uint32_t* d_heads [2];
  uint32_t* d_queues[2];
  float*   d_deltas;
  float *x0, *x1, *x2, *x3, *x4;
  uint32_t* d_colors[2];
  uint32_t* d_wrt[2];

  cudaMalloc( &d_annexBits,  sizeof(uint8_t)  * n );
  cudaMalloc( &d_heads [0],  sizeof(uint32_t) * n );
  cudaMalloc( &d_heads [1],  sizeof(uint32_t) * n );
  cudaMalloc( &d_queues[0],  sizeof(uint32_t) * n );
  cudaMalloc( &d_queues[1],  sizeof(uint32_t) * n );
  cudaMalloc( &x0,           sizeof(float)   * n );
  cudaMalloc( &x1,           sizeof(float)   * n );
  cudaMalloc( &x2,           sizeof(float)   * n );
  cudaMalloc( &x3,           sizeof(float)   * n );
  cudaMalloc( &x4,           sizeof(float)   * n );
  cudaMalloc( &d_colors[0],  sizeof(uint32_t) * n );
  cudaMalloc( &d_colors[1],  sizeof(uint32_t) * n );
  cudaMalloc( &d_wrt   [0],  sizeof(uint32_t) * n );
  cudaMalloc( &d_wrt   [1],  sizeof(uint32_t) * n );
  cudaMalloc( &d_deltas,     sizeof(float)   * n );
  
  float *d_gaussian, *d_uniform;        // random numbers
  float* d_sum[2];                      // for recursive sums

  float *d_results[2], *h_results[2];   // for the results

  cudaMalloc( &d_lags, lagsc * sizeof(uint32_t) );
  cudaMemcpy( d_lags, h_lags, lagsc*sizeof(uint32_t), cudaMemcpyHostToDevice );

  cudaMalloc( &d_gaussian, n * sizeof(float) );
  cudaMalloc( &d_uniform,  n * sizeof(float) );

  cudaMalloc( &d_sum[0], n * sizeof(float) * N );
  cudaMalloc( &d_sum[1], n * sizeof(float) * N );
  
  cudaMalloc(&d_results[0], N * sizeof(float) * SAMPLES_PER_COMMIT);
  cudaMalloc(&d_results[1], N * sizeof(float) * SAMPLES_PER_COMMIT);
  
  cudaMallocHost(
    (void**)&h_results[0], N * sizeof(float) * SAMPLES_PER_COMMIT
  );
  cudaMallocHost(
    (void**)&h_results[1], N * sizeof(float) * SAMPLES_PER_COMMIT
  );

  {
    // CPU (Ensure 0x10 < init_rng[i] < -0x10)
    std::ifstream randomFile("/dev/random",std::ios::binary);
    uint64_t init_rng[2] = {0x0, 0x0};
    while(
      init_rng[0] < 0x10 || init_rng[0] > 0xfffffffffffffff0 ||
      init_rng[1] < 0x10 || init_rng[1] > 0xfffffffffffffff0
    ) {
      randomFile.read((char*) init_rng, 2*sizeof(uint64_t));
    }
    randomFile.close();
    pcg32_srandom_r(h_rng, init_rng[0], init_rng[1]);

    // GPU
    uint64_t* h_rseed = new uint64_t[n];
    uint64_t* h_rstat = new uint64_t[n];
    uint64_t *d_rseed, *d_rstat;
    for(uint32_t i=0; i<n; i++) {
      init_rng[0] = 0;
      init_rng[1] = 0;
      while(
        init_rng[0] < 0x10 || init_rng[0] > 0xfffffffffffffff0 ||
        init_rng[1] < 0x10 || init_rng[1] > 0xfffffffffffffff0
      ) {
        init_rng[0] = ( (uint64_t)pcg32_random_r(h_rng) << 32 ) +
                                                        pcg32_random_r(h_rng);
        init_rng[1] = ( (uint64_t)pcg32_random_r(h_rng) << 32 ) +
                                                        pcg32_random_r(h_rng);
      }
      h_rseed[i] = init_rng[0];
      h_rstat[i] = init_rng[1];
    }
    cudaMalloc( (void**)&d_rseed, n * sizeof(uint64_t) );
    cudaMalloc( (void**)&d_rstat, n * sizeof(uint64_t) );
    cudaMemcpy(
      (void*)d_rseed, (void*)h_rseed,
      n * sizeof(uint64_t), cudaMemcpyHostToDevice
    );
    cudaMemcpy(
      (void*)d_rstat, (void*)h_rstat,
      n * sizeof(uint64_t), cudaMemcpyHostToDevice
    );

    d_init<<<num_blocks, BLOCK_SIZE>>>(d_path, d_rng, d_rseed, d_rstat, n);
    cudaDeviceSynchronize();

    delete[] h_rseed;     delete[] h_rstat;
    cudaFree(d_rseed);  cudaFree(d_rstat);
  }

  /* - Run the simulation --------------------------------------------------- */
  // Setup the progress bar
  asyncProgressBar* pb = new asyncProgressBar(samples, 80);
  uint32_t stream_idx = 0;
  bool saveResults = false;
  uint32_t these_samples = SAMPLES_PER_COMMIT;
  while(samples > 0) {
    these_samples = samples > SAMPLES_PER_COMMIT ? SAMPLES_PER_COMMIT : samples;
    samples -= these_samples;
    for(uint32_t i = 0; i<these_samples; i++) {
      // reset for the cycle
      d_reset_accepted<<<num_blocks, BLOCK_SIZE, 0, streams[stream_idx]>>>(
        d_accepted,
        n
      );
      // do ups updates
      for(uint32_t j=0; j<ups; j++) {
        // Init build cluster
        d_initBuildClusters1<<<num_blocks, BLOCK_SIZE, 0, streams[stream_idx]>>>(
          d_rng, d_path, a, gamma, n, d_annexBits
        );
        d_initBuildClusters2<<<num_blocks, BLOCK_SIZE, 0, streams[stream_idx]>>>(
          n, d_annexBits,
          d_heads[0], d_queues[0]
        );
        uint32_t parityBC = 0;
        // Build cluster
        for(uint32_t i=0; i<exponent; i++) {
          d_buildClusters<<<num_blocks, BLOCK_SIZE, 0, streams[stream_idx]>>>(
            d_annexBits, d_heads[parityBC], d_queues[parityBC],
            n, d_heads[1^parityBC], d_queues[1^parityBC]
          );
          parityBC ^= 1;
        }
        // Fix the case with one big cluster
        d_notOnlyOneCluster<<<num_blocks, BLOCK_SIZE, 0, streams[stream_idx]>>>(
          d_path, d_heads[parityBC], d_queues[parityBC], d_annexBits, n,
          x0, x1, x2, x3, x4
        );
        uint32_t colParity = 0;
        // Build cluster
        for(uint32_t i=0; i<exponent; i++) {
          d_colorClusters<<<num_blocks, BLOCK_SIZE, 0, streams[stream_idx]>>>(
            d_heads[parityBC], d_queues[parityBC],
            d_colors[colParity], d_wrt[colParity],
            n, 1 << ( exponent - i - 1),
            d_colors[colParity^1], d_wrt[colParity^1],
            x0, x1, x2, x3, x4
          );
          colParity ^= 1;
        }
        d_endColorCluster<<<num_blocks, BLOCK_SIZE, 0, streams[stream_idx]>>>(
          d_heads[parityBC], d_queues[parityBC],
          d_colors[colParity], n
        );
        d_extract<<<num_blocks, BLOCK_SIZE, 0, streams[stream_idx]>>>(
          d_path,
          d_heads[parityBC], d_queues[parityBC],
          d_colors[colParity], n,
          x0, x1, x2, x3, x4, a, gamma, g, d_rng, d_deltas
        );
        uint32_t start = pcg32_boundedrand_r(h_rng, 3);
        for(uint32_t j = 0; j<3; j++) {
          d_update<<<num_blocks, BLOCK_SIZE, 0, streams[stream_idx]>>>(
            d_path, n, d_heads[parityBC],
            d_deltas, d_accepted, d_colors[colParity], (j + start) % 3
          );
        }
      }
      // take measurments
      uint32_t full_numblock = n;
      uint32_t next_numblock = nextNumBlock( full_numblock );
      uint32_t stepParity = 0;
      d_computePows<<<next_numblock, BLOCK_SIZE, 0, streams[stream_idx]>>>(
        d_path, n, N, ups, d_accepted, d_sum[0]
      );
      d_measures<<<next_numblock, BLOCK_SIZE, 0, streams[stream_idx]>>>(
        d_path, n, N, lagsc, d_lags, d_sum[0]
      );
      // recursive sum
      while( full_numblock > BLOCK_SIZE ) {
        d_redsum<<<next_numblock, BLOCK_SIZE>>>(
          d_sum[stepParity], full_numblock,
          N, d_sum[stepParity^1]
        );
        stepParity = stepParity^1;
        full_numblock = next_numblock;
        next_numblock = nextNumBlock(full_numblock);
      }
      d_redsum<<<next_numblock, BLOCK_SIZE, 0, streams[stream_idx]>>>(
          d_sum[stepParity], full_numblock,
          N, d_results[stream_idx] + N * i
      );
    }
    cudaMemcpyAsync(
      h_results[stream_idx], d_results[stream_idx],
      sizeof(float) * these_samples * N, cudaMemcpyDeviceToHost,
      streams[stream_idx]
    );
    // ever SAMPLES_PER_COMMIT save on CPU (not if first run)
    if( saveResults ) {
      cudaStreamSynchronize(streams[stream_idx^1]);
      for( uint32_t i = 0; i<these_samples; i++ ) {
        outputfile << h_results[stream_idx^1][N*i] / (float)n;
        for( uint32_t j = 1; j<N; j++ )
          outputfile << ", " << h_results[stream_idx^1][j+N*i] / (float)n;
        outputfile << endl;
      }
    }
    stream_idx = stream_idx ^ 1;
    saveResults = true;
    pb->update(these_samples);
  }
  cudaStreamSynchronize(streams[stream_idx^1]);
  for( uint32_t i = 0; i<these_samples; i++ ) {
    outputfile << h_results[stream_idx^1][N*i] / (float)n;
    for( uint32_t j = 1; j<N; j++ )
      outputfile << ", " << h_results[stream_idx^1][j+N*i] / (float)n;
    outputfile << endl;
  }
  
  /* - Free memory, deallocate, ... ----------------------------------------- */
  cudaFree( d_rng );
  cudaFree( d_path );

  cudaFree( d_lags );

  cudaFree( d_accepted );

  cudaFree( d_gaussian );
  cudaFree( d_uniform );

  cudaFree( d_sum[0] );
  cudaFree( d_sum[1] );
  
  cudaFree( d_results[0] );
  cudaFree( d_results[1] );

  cudaFree( d_annexBits );
  cudaFree( d_heads [0] );
  cudaFree( d_heads [1] );
  cudaFree( d_queues[0] );
  cudaFree( d_queues[1] );
  cudaFree( x0          );
  cudaFree( x1          );
  cudaFree( x2          );
  cudaFree( x3          );
  cudaFree( x4          );
  cudaFree( d_colors[0] );
  cudaFree( d_colors[1] );
  cudaFree( d_wrt   [0] );
  cudaFree( d_wrt   [1] );
  cudaFree( d_deltas    );
  
  cudaFreeHost( h_results[0] );
  cudaFreeHost( h_results[1] );

  cudaStreamDestroy(streams[0]);
  cudaStreamDestroy(streams[1]);

  delete h_rng;
  delete pb;

  return 0;
}
