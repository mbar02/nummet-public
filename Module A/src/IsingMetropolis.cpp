#include <cstdint>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <sys/ioctl.h>
#include <cmath>

#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include "./MetalIsingMetropolis.hpp"

#include "../libs/pcg/pcg_basic.h"
#include "../libs/errors.h"

int main(int argc, char** argv) {
  if(argc < 6) {
    std::cout << "Sintassi: " << argv[0] << " [L] [updates per samples] [samples] [beta] [outputfile]\n";
    return TEINVAL;
  }
  uint32_t L,UPS,SAMPLES;
  float    beta;

  L       = (uint32_t)std::stoul(argv[1]);
  UPS     = (uint32_t)std::stoul(argv[2]);
  SAMPLES = (uint32_t)std::stoul(argv[3]);
  beta    = (float)   std::stof (argv[4]);

  /*
  // Ottengo larghezza terminale
  struct winsize w;
  int retval_winsize = ioctl(STDOUT_FILENO, TIOCGWINSZ, &w) == -1;
  const uint32_t COLUMNS = (
      retval_winsize == -1 ? 80 : ( (uint32_t)w.ws_col > 80 ? 80 : w.ws_col )
  );
  */

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

  // Inizializza GPU e oggetto
  MetalIsingMetropolis *mim = new MetalIsingMetropolis(
    argv[0],
    L,
    beta,
    &rng
  );

  // Esperimento
  outputFile << "[M], [E]" << std::endl;

  std::cout << "Inizio simulazione." << std::endl << std::flush;

  mim->simulation(UPS, SAMPLES, outputFile);

  std::cout << "Simulazione terminata. " << std::endl;

  outputFile.flush();
  outputFile.close();

  delete mim;

  return 0;
}
