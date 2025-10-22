import json
import multiprocessing

TH_CR_BETAS    = { 'sqr': 0.4407, 'tri': 0.2747, 'hex': 0.6585 }

# --- CONFIGURATION PARAMETERS         ---
#   Change only here!
def config(datafile, outdir='analysis/uncertainties'):
  data = json.load(open(datafile, 'r'))

  setting = {
    'SIM_ALGORITHMS' : [
      'wolff',
      'metropolis',
    ],
    'UPS_SIM'        : lambda alg, geo, L, beta: 1 if alg=='wolff' else 10,
    'SAMPLES_SIM'    : lambda alg, geo, L, beta: 2000000,
    'TEMPERAT_SIM'   : lambda alg, geo, L: { TH_CR_BETAS[geo] for x in data if x['geometry']==geo },
    'LATTICE_SIZES'  : lambda alg, geo:    { x['L']           for x in data if x['geometry']==geo },
    'OUTPUT_DIR'     : outdir,

    'MAX_PARAL_JOBS' : max(1, multiprocessing.cpu_count()),
  }
  return setting