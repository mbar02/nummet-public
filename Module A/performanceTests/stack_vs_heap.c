#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <alloca.h>

#define N 1000000

void sum(int* vector, volatile int* s, int samples, int reps) {
  *s = 0;
  for(volatile int i=0; i<reps; i++) {
    *s -= *s/i;
    for(volatile int j=0; j<samples; j++)
      *s += vector[j];
  }
  return;
}

int main(int argc, char** argv) {
  int* heapArray =  (int*)malloc( sizeof(int) * N );
  int* stackArray = (int*)malloc( sizeof(int) * N );

  volatile int samples = ( argc > 1 ) ? atoi(argv[1]) : N;
  volatile int s = 0;
  volatile int r = ( argc > 2 ) ? atoi(argv[2]) : 10;

  clock_t start, end;
  double used_time;

  // Init Data
  srand( time(NULL) );
  for(int i = 0; i < samples; i++) {
    heapArray[i] = stackArray[i] = rand();
  }

  printf("Test with %d samples:\n", samples);

  // Heap Allocation
  start = clock();
    sum(heapArray, &s, samples, r);
  end = clock();
  used_time = ((double)(end-start)) / CLOCKS_PER_SEC;
  printf("Heap allocation:\n\tresult: %d\n\tCPU time: %lf\n",s,used_time);

  // Stack Allocation
  start = clock();
    sum(stackArray, &s, samples, r);
  end = clock();
  used_time = ((double)(end-start)) / CLOCKS_PER_SEC;
  printf("Stack allocation:\n\tresult: %d\n\tCPU time: %lf\n",s,used_time);

  free(heapArray);
  return 0;
}
