#include <cstdio>
#include <cuda_runtime.h>
int main(){
  int rv = 0, dv = 0;
  cudaRuntimeGetVersion(&rv);
  cudaDriverGetVersion(&dv);
  printf("cudaRuntimeGetVersion: %d\ncudaDriverGetVersion: %d\n", rv, dv);
  return 0;
}
