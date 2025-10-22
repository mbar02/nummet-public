#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <math.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include "../libs/pcg/pcg_basic.h"

#include "../libs/progressBar.h"
#include "../libs/errors.h"

#include "../libs/geometry.h"

int64_t buildCluster(
  cell_t*         lattice,
  cell_t**        cluster,
  uint32_t        p,
  uint32_t        volume,
  uint32_t        UPS,
  pcg32_random_t* rng
) {
  int64_t delta_m = 0;
  // Fai UPS updates
  for(uint32_t u=0; u<UPS; u++) {
    uint32_t n_old = 0, n_new = 1;
    cluster[0] = lattice + pcg32_boundedrand_r(rng, volume);
    int8_t spinSign = cluster[0]->value;
    cluster[0]->value = -spinSign;

    while(n_old < n_new) {
      for(int i=0; i<N_NEIGH; i++)
        if(
          cluster[n_old]->neighbors[i]->value == spinSign &&
          pcg32_random_r(rng) <= p
        ) {
          cluster[n_new] = cluster[n_old]->neighbors[i];
          cluster[n_new]->value = -spinSign;
          n_new++;
        }
      // end reps
      n_old++;
    }
    delta_m -= 2 * (int64_t)n_new * spinSign;
  }
  return delta_m;
}

int main(int argc, char** argv) {
  if(argc < 6) {
    printf("Sintassi: %s [L] [updates per samples] [samples] [beta] [outputfile]\n", argv[0]);
    return TEINVAL;
  }

  const uint32_t L       = (uint32_t)strtoul(argv[1],NULL,0);
  const uint32_t UPS     = (uint32_t)strtoul(argv[2],NULL,0);
  const uint32_t SAMPLES = (uint32_t)strtoul(argv[3],NULL,0);
  const float    beta    = (float)   strtof (argv[4],NULL);

  const uint32_t volume  = VOL_FROM_L(L);

  // Colonne nel terminale
  struct winsize w;
  int retval_winsize = ioctl(STDOUT_FILENO, TIOCGWINSZ, &w) == -1;
  const uint32_t COLUMNS = (
      retval_winsize == -1 ? 80 : ( (uint32_t)w.ws_col > 80 ? 80 : w.ws_col )
  );

  // Inizializza generatore di numeri casuali
  FILE* file_p = fopen("/dev/random","rb");
  if(file_p == NULL) {
    fprintf(stderr, "Impossibile aprire /dev/random. ABORT.\n");
    return TEIO;
  }
  uint64_t init_rng[2];
  fread(init_rng, sizeof(uint64_t), 2, file_p);
  fclose(file_p);

  pcg32_random_t rng;
  pcg32_srandom_r(&rng, init_rng[0], init_rng[1]);

  // Inizializza output file
  file_p = fopen(argv[5], "w");
  if(file_p == NULL) {
    fprintf(stderr, "Impossibile aprire %s. ABORT.\n", argv[5]);
    return TEIO;
  }

  // Allocazioni necessarie
  cell_t*  lattice = (cell_t*)  malloc( volume*sizeof(cell_t)  );
  if ( lattice == NULL ) {
    fclose(file_p);
    fprintf(stderr, "Impossibile allocare lattice. ABORT.\n");
    return TENOMEM;
  }

  cell_t** cluster = (cell_t**) malloc( volume*sizeof(cell_t*) );
  if ( cluster == NULL ) {
    free(lattice);
    fclose(file_p);
    fprintf(stderr, "Impossibile allocare cluster. ABORT.\n");
    return TENOMEM;
  }

  // Inizializzazione reticolo e vicini
  for(uint32_t i=0; i<L; i++)
    for(uint32_t j=0; j<L; j++)
      initCell(lattice, i, j, L);

  // Calcolo probabilità di flippare
  float    p_f = 1.f - expf(-2.f*beta);
  uint32_t p_u = (uint32_t)(((uint64_t)1 << 32)*p_f);

  int64_t m = (int64_t)volume;

  // Esperimento
  fprintf(file_p, "[M], [E]\n");
  progressBar(
    {
      m += buildCluster(lattice, cluster, p_u, volume, UPS, &rng);
      fprintf(file_p,"%ld, %ld\n", (long int)m, (long int)energy(lattice, volume));
    },
    SAMPLES,
    COLUMNS
  );

  printf("Simulazione terminata.\n");

  free(lattice);
  free(cluster);

  fflush(file_p);
  fclose(file_p);

  return 0;
}
