// vecAdd_simple.cu
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <cuda_runtime.h>

#define CUDA_CHECK(err) \
  do { \
    cudaError_t e = (err); \
    if (e != cudaSuccess) { \
      std::cerr << "CUDA error " << e << " at " << __FILE__ << ":" << __LINE__ \
                << " -> " << cudaGetErrorString(e) << std::endl; \
      std::exit(EXIT_FAILURE); \
    } \
  } while (0)

__global__ void vecAdd(const double* a, const double* b, double* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}

int main() {
  int N = 1 << 20; // 1M elementi
  size_t bytes = N * sizeof(double);

  // Alloca host
  double *h_a = (double*)malloc(bytes);
  double *h_b = (double*)malloc(bytes);
  double *h_c = (double*)malloc(bytes);

  for (int i = 0; i < N; ++i) {
    h_a[i] = double(i) * 0.001;
    h_b[i] = double(i) * 0.002;
  }

  // Alloca device
  double *d_a, *d_b, *d_c;
  CUDA_CHECK(cudaMalloc(&d_a, bytes));
  CUDA_CHECK(cudaMalloc(&d_b, bytes));
  CUDA_CHECK(cudaMalloc(&d_c, bytes));

  // Copia input -> device
  CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

  // Lancia kernel
  int block = 256;
  int grid = (N + block - 1) / block;
  vecAdd<<<grid, block>>>(d_a, d_b, d_c, N);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  // Copia risultato indietro
  CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));

  // Verifica
  bool ok = true;
  for (int i = 0; i < N; ++i) {
    double expected = h_a[i] + h_b[i];
    if (fabs(h_c[i] - expected) > 1e-12) { ok = false; break; }
  }
  std::cout << "Result: " << (ok ? "OK" : "ERROR") << std::endl;

  // Pulizia
  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_c));
  free(h_a); free(h_b); free(h_c);

  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
