#include <cstdio>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <thread>

const int NNUM = 1000;
const int BLOCK_SIZE = 64;

__global__ void initRNG(curandState *states, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NNUM)
        curand_init(seed, idx, 0, &states[idx]);
}

__global__ void sumRandom(curandState *states, double *blockSums) {
    __shared__ double shared[BLOCK_SIZE];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    double val = 0.0;
    if (idx < NNUM)
        val = curand_uniform_double(&states[idx]);
    shared[threadIdx.x] = val;
    __syncthreads();

    // riduzione intra-block
    for (int s = BLOCK_SIZE/2; s > 0; s >>= 1) {
        if (threadIdx.x < s) shared[threadIdx.x] += shared[threadIdx.x + s];
        __syncthreads();
    }

    if (threadIdx.x == 0)
        blockSums[blockIdx.x] = shared[0];
}

int main() {
    int numBlocks = (NNUM + BLOCK_SIZE - 1) / BLOCK_SIZE;

    curandState *d_states;
    double *d_blockSums;
    cudaMalloc(&d_states, NNUM * sizeof(curandState));
    cudaMalloc(&d_blockSums, numBlocks * sizeof(double));

    // due buffer pinned per doppio buffering
    double *h_results[2];
    cudaMallocHost(&h_results[0], sizeof(double) * numBlocks);
    cudaMallocHost(&h_results[1], sizeof(double) * numBlocks);

    initRNG<<<numBlocks, BLOCK_SIZE>>>(d_states, 1234);
    cudaDeviceSynchronize();

    cudaStream_t streams[2];
    cudaStreamCreate(&streams[0]);
    cudaStreamCreate(&streams[1]);

    int N = 10;
    for (int i = 0; i < N; ++i) {
        int stream_idx = i % 2;       // stream alternato
        int cpu_idx = (i+1) % 2;      // buffer già pronto per CPU

        // kernel di somma sulla GPU in uno stream
        sumRandom<<<numBlocks, BLOCK_SIZE, 0, streams[stream_idx]>>>(d_states, d_blockSums);

        // copia asincrona del risultato nella pinned memory
        cudaMemcpyAsync(h_results[stream_idx], d_blockSums, sizeof(double)*numBlocks, cudaMemcpyDeviceToHost, streams[stream_idx]);

        // CPU legge il risultato del passo precedente (se i>0)
        if (i > 0) {
            double total = 0.0;
            cudaStreamSynchronize(streams[cpu_idx]);
            for(int j = 0; j < numBlocks; j++)
              total += h_results[cpu_idx][j];
            // assicurarsi che la copia asincrona sia completata
            printf("Step %d: sum = %f\n", i-1, total);
        }
    }

    // ultimo passo
    double total = 0.0;
    cudaStreamSynchronize(streams[1-N%2]);
    for(int j = 0; j < numBlocks; j++)
      total += h_results[1-N%2][j];
    printf("Step %d: sum = %f\n", N-1, total);

    cudaFree(d_states);
    cudaFree(d_blockSums);
    cudaFreeHost(h_results[0]);
    cudaFreeHost(h_results[1]);
    cudaStreamDestroy(streams[0]);
    cudaStreamDestroy(streams[1]);

    return 0;
}