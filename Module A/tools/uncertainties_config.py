import json
import multiprocessing

# --- CONFIGURATION PARAMETERS         ---
#   Change only here!
def config(datafile, outdir='analysis/uncertainties'):
  data = json.load(open(datafile, 'r'))

  setting = {
    'SIM_ALGORITHMS' : [
      'wolff',
      'metropolis',
    ],
    'UPS_SIM'        : lambda alg, geo, L, beta: 20 if alg=='wolff' else 200,
    'SAMPLES_SIM'    : lambda alg, geo, L, beta: 500000,
    'TEMPERAT_SIM'   : lambda alg, geo, L: ( x['beta_max'] for x in data if x['geometry']==geo and x['L']==L ),
    'LATTICE_SIZES'  : lambda alg, geo:    ( x['L']        for x in data if x['geometry']==geo ),
    'OUTPUT_DIR'     : outdir,

    'MAX_PARAL_JOBS' : max(1, multiprocessing.cpu_count()),
  }
  return setting