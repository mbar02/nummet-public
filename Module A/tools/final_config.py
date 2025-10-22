import json
import multiprocessing

# --- CONFIGURATION PARAMETERS         ---
#   Change only here!
def config(datafile, outdir='analysis/final'):
  data = json.load(open(datafile, 'r'))

  setting = {
    'SIM_ALGORITHMS' : [
      'wolff',
      'metropolis',
    ],
    'UPS_SIM'        : lambda alg, geo, L, beta: 20 if alg=='wolff' else 200,
    'SAMPLES_SIM'    : lambda alg, geo, L, beta: samples(data, alg, geo, L, beta),
    'TEMPERAT_SIM'   : lambda alg, geo, L: temperatures(data, alg, geo, L),
    'LATTICE_SIZES'  : lambda alg, geo: set([ d['L'] for d in data if alg==d['algorithm'] and geo==d['geometry'] ]),
    'OUTPUT_DIR'     : outdir,

    'MAX_PARAL_JOBS' : max(1, multiprocessing.cpu_count()),
  }
  return setting

def temperatures(data, alg, geo, L):
  if alg == 'metropolis' and L>100:
    return []
  else:
    return  [ d for d in data if alg==d['algorithm'] and geo==d['geometry'] and L==d['L'] ][0]['betas']

def samples(data, alg, geo, L, beta):
  suggested_sam = [ d for d in data if alg==d['algorithm'] and geo==d['geometry'] and L==d['L'] ][0]['samples']
  if alg == 'metropolis':
    return suggested_sam / (L**0.5)
  else:
    return suggested_sam