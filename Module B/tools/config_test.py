import numpy as np
import multiprocessing

# --- CONFIGURATION PARAMETERS         ---
#   Change only here!
def config():
  setting = {
    'SIM_ALGORITHMS' : [
      'wolff',
      'metropolis',
      'multicluster',
    ],
    'UPS_SIM'        : lambda alg, n, beta, g: 10,
    'SAMPLES_SIM'    : lambda alg, n, beta, g: 10000,
    'TEMPERAT_SIM'   : lambda n, g:            [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0],
    'LAGS_SIM'       : lambda n, beta, g:      [ int(i*n) for i in [0.5, 0.2, 0.1] ],
    'PATH_SIZE'      : lambda g:               [200, 500, 1000],
    'COUPLING_CONST' : [0, 1, 10, 100],
    'OUTPUT_DIR'     : './python_launcher_tests',

    'MAX_PARAL_JOBS' : max(1, multiprocessing.cpu_count()),
  }
  return setting