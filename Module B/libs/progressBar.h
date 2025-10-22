#pragma once

#if __cplusplus
extern "C" {
#endif

#include<time.h>

#define progressBar(code, length, columns)                                \
  clock_t progressBar_start;                                              \
  progressBar_start = clock();                                            \
                                                                          \
  uint32_t progressBar_vcol;                                              \
  if(columns < 10)                                                        \
    progressBar_vcol = 78;                                                \
  else if(columns > 200)                                                  \
    progressBar_vcol = 198;                                               \
  else                                                                    \
    progressBar_vcol = columns - 2;                                       \
  for(                                                                    \
    uint32_t progressBar_i=0;                                             \
    progressBar_i<(uint32_t)(length % progressBar_vcol);                  \
    progressBar_i++                                                       \
    ) {                                                                   \
    code;                                                                 \
  }                                                                       \
  for(                                                                    \
    uint32_t progressBar_i=0;                                             \
    progressBar_i<progressBar_vcol;                                       \
    progressBar_i++                                                       \
  ){                                                                      \
    printf("<");                                                          \
    for(                                                                  \
      uint32_t progressBar_j=0;                                           \
      progressBar_j<=progressBar_i;                                       \
      progressBar_j++                                                     \
    )                                                                     \
      printf("#");                                                        \
    for(                                                                  \
      uint32_t progressBar_j=progressBar_i+1;                             \
      progressBar_j<progressBar_vcol;                                     \
      progressBar_j++                                                     \
    )                                                                     \
      printf(".");                                                        \
    printf(">\n");                                                        \
                                                                          \
    printf(                                                               \
        "Simulazione completata al %0u%%. Tempo impiegato: %lfs\033[A\r", \
        100u*(progressBar_i+1u)/progressBar_vcol,                         \
        (double)(clock()-progressBar_start) / CLOCKS_PER_SEC              \
    );                                                                    \
    fflush(stdout);                                                       \
    for(                                                                  \
        uint32_t progressBar_j=0;                                         \
        progressBar_j<(uint32_t)(length/progressBar_vcol);                \
        progressBar_j++                                                   \
    ) {                                                                   \
      code;                                                               \
    }                                                                     \
  }                                                                       \
  printf("\n\n");

#if __cplusplus
}
#endif

#if __cplusplus

#include <iostream>
class asyncProgressBar {
  public:
    asyncProgressBar(uint64_t samples, uint32_t columns, std::ostream *ostr = &std::cout);
    
    void update(uint64_t step = 1);

    private:
      uint32_t lastLim;
      uint32_t columns;
      uint64_t steps;
      uint64_t status;
      std::ostream* ostr;
};

#endif
