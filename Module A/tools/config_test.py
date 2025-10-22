import numpy as np
import multiprocessing

TH_CR_BETAS    = { 'sqr': 0.4407, 'tri': 0.2747, 'hex': 0.6585 }

# --- CONFIGURATION PARAMETERS         ---
#   Change only here!
def config():
  setting = {
    'SIM_ALGORITHMS' : [
      'wolff',
      'metropolis',
    ],
    'UPS_SIM'        : lambda alg, geo, L, beta: 10,
    'SAMPLES_SIM'    : lambda alg, geo, L, beta: 10000,
    'TEMPERAT_SIM'   : lambda alg, geo, L: np.linspace(TH_CR_BETAS[geo]*0.95, TH_CR_BETAS[geo]*1.025, 3),
    'LATTICE_SIZES'  : lambda alg, geo: [ 5, 10, 15, 20 ],
    'OUTPUT_DIR'     : './python_launcher_tests',

    'MAX_PARAL_JOBS' : max(1, multiprocessing.cpu_count()),
  }
  return setting