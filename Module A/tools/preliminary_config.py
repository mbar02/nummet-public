import numpy as np
import math
import multiprocessing

TH_CR_BETAS    = { 'sqr': 0.4407, 'tri': 0.2747, 'hex': 0.6585 }

# --- CONFIGURATION PARAMETERS         ---
#   Change only here!
def config(outdir='analysis/preliminar'):
  setting = {
    'SIM_ALGORITHMS' : [
      'wolff',
      #'metropolis',
    ],
    'UPS_SIM'        : lambda alg, geo, L, beta: 10,
    'SAMPLES_SIM'    : lambda alg, geo, L, beta: int(math.sqrt(L)*1e4),
    'TEMPERAT_SIM'   : lambda alg, geo, L: np.linspace(TH_CR_BETAS[geo]*0.95, TH_CR_BETAS[geo]*1.025, 20),
    'LATTICE_SIZES'  : lambda alg, geo: [ 30, 34, 38, 44, 48, 54, 62, 70, 80, 90, 100, 114, 128, 144, 164, 184, 208, 236, 266, 300 ],
    'OUTPUT_DIR'     : outdir,

    'MAX_PARAL_JOBS' : max(1, multiprocessing.cpu_count()),
  }
  return setting