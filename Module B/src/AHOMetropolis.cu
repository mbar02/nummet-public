#include "./AHOGpuCommon.cu"

/* - Functions -------------------------------------------------------------- */
// Compute discrete laplacian
__device__ inline float lapl(float x, float xm, float xp, float a) {
  return ( xp + xm - x - x ) / ( a*a );
}

// Compute alpha coefficient
__device__ inline float alpha(float a, float g, float x) {
  return sqrt( 1.0/a + a/2.0*( 1.0 + 3.0 * g * x * x ) );
}

// Compute beta coefficient
__device__ inline float beta(float a, float g, float x, float laplx, float alpha) {
  return a/2.0 * ( x + g*(x*x*x) - laplx ) / alpha;
}

// Compute kinetic energy density
__device__ inline float kinetic(float x, float xm, float xp, float a) {
  return x * ( x - xm - xp ) / a;
}

// Compute potential energy density
__device__ inline float potential(float x, float g, float a) {
  float x2 = x*x;
  return a * ( x2 / 2.0 + g*x2*x2/4.0 );
}

// Compute the variation of the (interactive) action
__device__ inline float deltaS(
  float x,
  float xm, float xp,
  float a,
  float g,
  float delta
) {
  return (
    kinetic( x+delta, xm, xp, a ) - kinetic( x, xm, xp, a ) +
    potential( x+delta, g, a )    - potential( x, g, a )
  );
}

// (The opposite of) the difference between forward and backward free action
__device__ inline float freeS(
  float new_alpha,  float new_beta,
  float old_alpha,  float old_beta,
  float delta
) {
  float old_s = old_alpha*(+delta) + old_beta;
  old_s = old_s * old_s;
  float new_s = new_alpha*(-delta) + new_beta;
  new_s = new_s * new_s;
  return old_s - new_s;
}

__device__ inline float accept_prob(
  float x, float xm, float xp,
  float a, float g,  float delta,
  float old_alpha, float old_beta
) {
  float new_alpha = alpha(a, g, x+delta);
  float new_beta  = beta(a, g, x+delta, lapl(x+delta,xm,xp,a), new_alpha);

  return new_alpha / old_alpha * exp(
    + freeS( new_alpha, new_beta, old_alpha, old_beta, delta )
    - deltaS( x, xm, xp, a, g, delta )
  );
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
  if(idx < n/2) {
    // Set up RNG
    pcg32_random_t local_rng = rng[idx];
    pcg32_srandom_r(&local_rng, rseed[idx], rstat[idx]);  
    rng[idx] = local_rng;

    // Set up the path
    path[2*idx  ] = 0.0;
    path[2*idx+1] = 0.0;
  }
}

__global__ void d_reset_accepted(
  uint32_t* accepted,
  uint32_t  n
) {
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < n/2) {
    accepted[2*idx  ] = 0;
    accepted[2*idx+1] = 0;
  }
}

__global__ void d_generate_random(
  pcg32_random_t* rng,      // rng states
  float*         gaussian, // extracted gaussian numbers
  float*         uniform,  // extracted numbers in uniform distribution
  const uint32_t  n
) {
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < n/2) {
    pcg32_random_t local_rng = rng[idx];

    boxMuller(&local_rng, gaussian[idx*2  ], gaussian[idx*2+1]);
    uniform[2*idx  ] = randomFloat(&local_rng);
    uniform[2*idx+1] = randomFloat(&local_rng);

    rng[idx] = local_rng;
  }
}

__global__ void d_update(
  float*         path,     // x(tau)
  const float*   normal,   // numbers from normal distribution
  const float*   uniform,  // numbers from uniform distribution
  const float    a,        // time step
  const float    g,        // coupling constant
  const uint32_t  parity,   // parity of the first site to upgrade
  const uint32_t  n,        // path size
  uint32_t*       accepted  // accept rate counter
) {
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx<n/2) {
    uint32_t loc_idx = 2*idx + parity;
    
    float x = path[loc_idx];
    float xm = neigh_m(path, loc_idx, n);
    float xp = neigh_p(path, loc_idx, n);

    // compute coefficients for the distribution
    float old_alpha = alpha(a, g, x);
    float old_beta  = beta(a, g, x, lapl(x,xm,xp,a), old_alpha);

    // random normal-extracted number
    float delta = ( normal[loc_idx] * CUDART_SQRT_HALF - old_beta ) / old_alpha;

    // probability of accepting the update
    float pacc = accept_prob(x,xm,xp,a,g,delta,old_alpha,old_beta);

    // Accept/reject the update
    if( uniform[loc_idx] < pacc ) {
      accepted[loc_idx]++;
      path[loc_idx] += delta;
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

  // Number of measures for each site

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
  cudaMalloc( &d_rng, n/2 * sizeof(pcg32_random_t) );

  // Path
  float* d_path;
  cudaMalloc( &d_path, n * sizeof(float) );

  // GPU stuff
  uint32_t num_blocks = ( n/2 + BLOCK_SIZE - 1 ) / BLOCK_SIZE;

  const uint32_t pair_count = MAX_X_POW * (MAX_X_POW + 1)/2;
  uint32_t N = 1 + MAX_X_POW + lagsc * pair_count;

  cudaStream_t streams[2];              // double buffering
  cudaStreamCreate(&streams[0]);
  cudaStreamCreate(&streams[1]);

  uint32_t* d_accepted;
  cudaMalloc( &d_accepted, n * sizeof(uint32_t) );

  uint32_t *d_lags;                     // lags
  
  float *d_gaussian, *d_uniform;       // random numbers
  float* d_sum[2];                     // for recursive sums

  float *d_results[2], *h_results[2];  
  // for the results

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
    for(uint32_t i=0; i<n/2; i++) {
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
    cudaMalloc( (void**)&d_rseed, n/2 * sizeof(uint64_t) );
    cudaMalloc( (void**)&d_rstat, n/2 * sizeof(uint64_t) );
    cudaMemcpy(
      (void*)d_rseed, (void*)h_rseed,
      n/2 * sizeof(uint64_t), cudaMemcpyHostToDevice
    );
    cudaMemcpy(
      (void*)d_rstat, (void*)h_rstat,
      n/2 * sizeof(uint64_t), cudaMemcpyHostToDevice
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
        uint32_t parity = pcg32_boundedrand_r(h_rng, 2);
        d_generate_random<<<num_blocks, BLOCK_SIZE, 0, streams[stream_idx]>>>(
          d_rng, d_gaussian, d_uniform, n
        );
        d_update<<<num_blocks, BLOCK_SIZE, 0, streams[stream_idx]>>>
        (
          d_path, d_gaussian, d_uniform, a, g, parity, n, d_accepted
        );
        d_update<<<num_blocks, BLOCK_SIZE, 0, streams[stream_idx]>>>
        (
          d_path, d_gaussian, d_uniform, a, g, parity ^ 1, n, d_accepted
        );
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
      d_redsum<<<next_numblock, BLOCK_SIZE>>>(
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
  
  cudaFreeHost( h_results[0] );
  cudaFreeHost( h_results[1] );

  cudaStreamDestroy(streams[0]);
  cudaStreamDestroy(streams[1]);

  delete h_rng;
  delete pb;

  return 0;
}
