import numpy as np
import multiprocessing

upss = {
  #'wolff':        30,
  'wolff':        300,
  'metropolis':   30,
  'multicluster': 10,
}

betas = {
  0:    10,
  10:   4,
  100:  2,
}

n = 100

# --- CONFIGURATION PARAMETERS         ---
#   Change only here!
def config():
  setting = {
    'SIM_ALGORITHMS'     : [
      'wolff',
      #'metropolis',
      #'multicluster',
    ],
    'UPS_SIM'            : lambda alg, n, beta, g: upss[ alg ],
    'SAMPLES_SIM'        : lambda alg, n, beta, g: int( 3e6 ),
    'TEMPERAT_SIM'       : lambda n, g:            [ betas[g] ],
    'LAGS_SIM'           : lambda n, beta, g:      [ n // 10 ],
    'PATH_SIZE'          : lambda g:               [ n ],
    'COUPLING_CONST'     : [0, 10, 100],
    'OUTPUT_DIR'         : './check_cluster',

    'MAX_PARAL_JOBS'     : 8,
    'MAX_PARAL_JOBS_GPU' : 4,
  }
  return setting
