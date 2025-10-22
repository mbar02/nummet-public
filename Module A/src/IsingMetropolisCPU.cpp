#include <cstdint>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <sys/ioctl.h>
#include <unistd.h>
#include <cmath>

#include "../libs/geometry.h"
#include "../libs/pcg/pcg_basic.h"
#include "../libs/errors.h"
#include "../libs/progressBar.h"

uint32_t L,UPS,SAMPLES;
float    beta;

void updateLattice(
  cell_t* lattice,
  uint32_t* p_u,
  uint32_t L,
  uint32_t volume,
  pcg32_random_t &rng,
  uint32_t UPS,
  int64_t &mag,
  int64_t &ene) {
  uint32_t w =
#if LATTICE==LATTICE_HEX
               L;
#else
               L/2;
#endif
  for(uint32_t s=0; s<UPS; s++) {
    mag = 0;
    for(uint8_t pi=0; pi<2; pi++) for(uint8_t pj=0; pj<2; pj++)
      for(uint32_t i=0; i<L/2; i++) for(uint32_t j=0; j<w; j++) {
        int      alignedNeigh = 0;
        uint32_t linearIdx    = 2*w*(2*i+pi) + (2*j+pj);
        for(uint32_t n=0; n<N_NEIGH; n++)
          if( lattice[ linearIdx ].value == lattice[ linearIdx ].neighbors[n]->value ) alignedNeigh++;
        if( pcg32_random_r(&rng) < p_u[ alignedNeigh ] )
          lattice[ linearIdx ].value = -lattice[ linearIdx ].value;
        mag += lattice[ linearIdx ].value;
      }
  }
  ene = energy(lattice, volume);
  return;
}

int main(int argc, char** argv) {
  if(argc < 6) {
    std::cout << "Sintassi: " << argv[0] << " [L] [updates per samples] [samples] [beta] [outputfile]\n";
    return TEINVAL;
  }

  L       = (uint32_t)std::stoul(argv[1]);
  UPS     = (uint32_t)std::stoul(argv[2]);
  SAMPLES = (uint32_t)std::stoul(argv[3]);
  beta    = (float)   std::stof (argv[4]);

  // Ottengo larghezza terminale
  struct winsize w;
  int retval_winsize = ioctl(STDOUT_FILENO, TIOCGWINSZ, &w) == -1;
  const uint32_t COLUMNS = (
      retval_winsize == -1 ? 80 : ( (uint32_t)w.ws_col > 80 ? 80 : w.ws_col )
  );

  // Per ottimizzazione, L pari
  if(L % 2 != 0) {
    L = L+1;
    std::cerr << "Warning: due to memory concurrency, L must be even. Fallback to L+1." << std::endl;
  }

  // Inizializza generatore di numeri casuali
  std::ifstream randomFile("/dev/random",std::ios::binary);
  uint64_t init_rng[2];
  randomFile.read((char*) init_rng, 2*sizeof(uint64_t));
  randomFile.close();

  pcg32_random_t rng;
  pcg32_srandom_r(&rng, init_rng[0], init_rng[1]);

  // Inizializza output file
  std::ofstream outputFile(argv[5]);

  uint32_t volume = VOL_FROM_L(L);

  // allocazioni necessarie
  cell_t *lattice  = new cell_t[volume];

  // Calcolo probabilità di flippare dati $i vicini con stesso segno (pre-flip)
  uint32_t *p_u = new uint32_t[N_NEIGH+1];

  for(int32_t i=0; i<N_NEIGH+1; i++){
    double p_f = 1. / ( 1. + exp( -2. * beta * ( N_NEIGH - 2*i ) ) );
    p_u[i] = (uint64_t)(pow(2.,32)*p_f) & 0xffffffff;
  }

  // Inizializzazione reticolo e vicini
  for(uint32_t i=0; i<L; i++)
    for(uint32_t j=0; j<L; j++)
      initCell(lattice, i, j, L);

  // Esperimento
  outputFile << "[M], [E]" << std::endl;

  int64_t mag = volume;                  // magnetizzazione
  int64_t ene = energy(lattice, volume); // energia

  progressBar(
    ({
      outputFile << mag << ", " << ene << std::endl;
      updateLattice(lattice, p_u, L, volume, rng, UPS, mag, ene);
    }),
    SAMPLES,
    COLUMNS
  );

  std::cout << "Simulazione terminata. " << std::endl;

  delete[] lattice;
  delete[] p_u;

  outputFile.close();

  return 0;
}
