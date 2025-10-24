import os
import sys

import numpy  as np
import cmath
from matplotlib        import pyplot    as plt
from matplotlib        import colormaps as cmap
from scipy.optimize    import curve_fit
from scipy.linalg      import eigh
import pandas as pd

import json
from pathlib import Path
import argparse
import multiprocessing

from tqdm import tqdm

import asyncio

plt.rcParams.update({
    "font.size": 10,
    "font.family": "Times New Roman",
    "axes.titlesize": 10,
    "axes.labelsize": 8,
    "figure.dpi": 500,
    "figure.figsize": (4.3,3.5),
    "legend.labelspacing": 0.1,
    "legend.handletextpad": 0.2,
    "legend.borderpad": 0.2,
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{babel}[italian]",
    "lines.linewidth": 0.25,
    "lines.markersize": 1
})

eta = 0.9e-1

data     = []
fit_data = {}

def autocorrelation(x, cutoff):
  dx = x - np.mean(x)
  ac = []
  ks = []
  s  = np.var(x)
  l = len(x)
  for i in range( 0, cutoff, max( int(cutoff/200), 1 ) ):
    ks.append(i)
    ac.append( np.mean( [ dx[j] * dx[i+j] for j in range(l) if i+j < l ] ) / s )
  return (np.array(ks), np.array(ac))


procfile_semaphore_counter = asyncio.Semaphore( 4 )
procfile_semaphore_update  = asyncio.Semaphore( 1 )

async def process_file(descriptor,labels,obs_idx, noLags, MAX_X_POW, plot_dir):
  sim_data  = json.load(open(descriptor,'r'))
  alg       = sim_data['algorithm']           # algorithm use
  g         = sim_data['coupling']            # perturbation .25*g*q^4 parameter
  path_size = sim_data["path_size"]
  beta      = sim_data['beta']                # max beta of the simulation
  samples   = sim_data['samples']             # how many samples
  lags      = np.array(sim_data['lags'])      # array of lags used for correlators
  lags      = lags * beta/path_size

  # find k_blocking
  vals = [ sim_data[l] for l in labels if not np.isnan(sim_data[l]) ]
  vals = np.sort(vals).tolist()
  if len(vals) == 0:
    vals.append( samples // 15 )
    print("Warning: k_blocking not found!")
  while(len(vals) < 3):
    vals.append(vals[-1])
  k_blocking = int(vals[-3])

  ncols = 1 + MAX_X_POW + MAX_X_POW*(MAX_X_POW+1)//2 * len(lags)

  # load simulated data
  filepath          = Path(sim_data['folder']) / 'data.csv'
  simulated_data    = pd.read_csv(filepath, sep=',', skiprows=samples//5, engine='c').to_numpy().reshape(-1,ncols)

  cutoff = int(75)
  therm  = samples//5
  skip   = 1

  for i in range(ncols):
    ks, ac = autocorrelation(simulated_data[range(0,therm,skip),i], cutoff)
    plt.plot(skip*ks,abs(ac), label=f"col {i}")
  plt.xlabel("lag")
  plt.ylabel("Autocorrelation")
  plt.yscale('log')
  plt.ylim([0.02,1.02])
  plt.xlim([0,skip*cutoff*1.05])
  plt.savefig(plot_dir / f"autocorrelation_{alg}_{int(1e3 *g ):06d}_{int(100*beta/path_size):02d}.svg")
  plt.clf()
  
  procfile_semaphore_update.release()

async def main():
  # parse arguments from command line
  parser = argparse.ArgumentParser( description="It does the final fit." )
  parser.add_argument( "--plotDir"   ,   type=str, help="Output plot directory.", required=True )
  parser.add_argument( "input"       ,   type=str, help="JSON descriptors of the simulation.", nargs='+' )

  args = parser.parse_args()

  plot_dir    = Path( args.plotDir    )

  # Create output folder
  os.makedirs(plot_dir, exist_ok=True)

  # read input files
  files = args.input

  MAX_X_POW     = 4                            # first powers of x taken into account
  noCouples     = MAX_X_POW*(MAX_X_POW+1)//2   # number of correlators computed for each lag
  noLags        = 2                            # number of lags taken into account

  obs_idx = [[] for _ in range(noLags)]
  counter  = MAX_X_POW + 1
  labels   = []

  for k in range(noLags):
    for i in range(1,MAX_X_POW+1):
      for j in range(i, MAX_X_POW+1):
        labels.append( f"C{{{i},{j}}}({k+1})_k")
        obs_idx[k].append(counter)
        counter +=1
  for f in tqdm(files):
    await procfile_semaphore_counter.acquire()
    await process_file(f,labels,obs_idx,noLags,MAX_X_POW,plot_dir)
    procfile_semaphore_counter.release()

  # for algorithm in set( d['algorithm'] for d in data ):
  #   data_alg      = [d for d in data if d['algorithm'] == algorithm]
  #   gg            = []
  #   scaled_gaps   = []
  #   scaled_sigma  = []
  #   for g in set(( d['coupling'] for d in data_alg)):
  #     x=0

if __name__ == "__main__":
  asyncio.run(main())