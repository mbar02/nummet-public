import copy
import sys

import numpy  as np

import json
from pathlib import Path
import argparse

from tqdm import tqdm

data = []

def process_file(descriptor):
  global data

  # read json descriptor
  with open(descriptor, 'r') as f:
    meta = json.load(f)

  # find maximum blocking factor
  k = max( (int(v) for k, v in meta.items() if k.endswith("_k")), default=None)

  if k is None:
    print(f"Error: no blocking factor found in {descriptor}")
    sys.exit(136)

  # read data
  datafile = Path( meta["folder"] ) / "data.csv"
  volume = meta['L']**2 * ( 2 if meta['geometry'] == 'hex' else 1 )

  measures = np.loadtxt(datafile, delimiter=',', skiprows=5*k, dtype=int)

  noBlockedMeasures = measures.shape[0] // k
  blockedMeasures = np.zeros( ( noBlockedMeasures, 2 ), dtype=float )

  for i in range(noBlockedMeasures):
    blockedMeasures[i,0] = np.mean( np.abs(measures[i*k:(i+1)*k, 0])   ) / volume
    blockedMeasures[i,1] = np.mean(        measures[i*k:(i+1)*k, 0]**2 ) / volume**2

  # compute reduced magnetization
  am = blockedMeasures[:,0]
  m2 = blockedMeasures[:,1]

  m2_mean = np.mean(m2)
  am_mean = np.mean(am)

  amsigma = ( np.var(am) )**0.5 / am_mean
  m2sigma = ( np.var(m2) )**0.5 / m2_mean

  sigma_tot = amsigma + m2sigma # upper bound: sum of the two sqrt(variances)
  sigma2_tot = sigma_tot**2     # note: this is the variance of a single measurement!
                                # => it's NOT the variance of the mean!

  # add this to data
  compatible_data = [ i for i, datum in enumerate(data)
                          if datum['algorithm'] == meta['algorithm']
                          and datum['geometry'] == meta['geometry']
                          and datum['L'] == meta['L'] ]
  if len(compatible_data) == 0:
    print(f"Error: no entry in the JSON file match with {descriptor})\n")
    parser.print_help()
    sys.exit(136)
  elif len(compatible_data) > 1:
    print(f"Error: more than one entry ({len(compatible_data)}, in particular) in the JSON file match with {descriptor})\n")
    parser.print_help()
    sys.exit(136)
  else:
    i = compatible_data[0]
    N = int( sigma2_tot * data[i]['Xmax'] * 2e4 / 9 ) * k + 5 * k
    data[i]['samples'] = N
    deltaBeta = 0.65*data[i]['beta0']*(data[i]['Xmax']**0.25)
    data[i]['betas']   = np.linspace( data[i]['beta_max']-deltaBeta,data[i]['beta_max']+deltaBeta,10 ).tolist()
    data[i]['k']       = k

if __name__ == "__main__":
  # parse arguments from command line
  parser = argparse.ArgumentParser( description="It computes the parameters of last bunch of measurements." )
  parser.add_argument(
    "--maxBetasFile", type=str, required=True, help="""
    JSON input file with maxima preliminar parameters.
    Each entry must have a different geometry or L""",
  )
  parser.add_argument(
    "--paramsFile", type=str, required=True, help="""
    JSON output file with final simulations parameters.""",
  )
  parser.add_argument(
    "sim_files", type=str, nargs='+', help="""
    List of the simulation data files.
    Each simulation must match to one entry
    in the maxBetasFile."""
  )

  args = parser.parse_args()

  in_file  = args.maxBetasFile
  out_file = args.paramsFile

  # read json descriptor
  with open(in_file, 'r') as f:
    data = json.load(f)

  if len(args.sim_files) < 2 * len(data):
    print("There are too few simulations for this JSON file.")
    parser.print_help()
    sys.exit(136)

  def append_metropolis(x):
    y = copy.deepcopy(x)
    y['algorithm'] = "metropolis"
    return y

  def append_wolff(x):
    y = copy.deepcopy(x)
    y['algorithm'] = "wolff"
    return y

  data = [ append_foo(datum) for datum in data for append_foo in [append_metropolis, append_wolff] ]

  for f in tqdm(args.sim_files):
    process_file(f)
  
  # save data
  outdata = sorted(data, key=lambda x: (x['algorithm'], x['geometry'], x['L']) )
  with open(out_file, 'w') as f:
    json.dump(outdata, f, indent=2)