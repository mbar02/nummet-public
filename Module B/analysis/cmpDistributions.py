import numpy  as np
from matplotlib        import pyplot    as plt
from scipy.stats       import ks_2samp
import pandas as pd

import json
from pathlib import Path

import tqdm

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

files = [
  "check_cluster/Metro_0100_10000_0000000_11d622.json",
  "check_cluster/Metro_0100_2000_0010000_f996c6.json",
  "check_cluster/Metro_0100_4000_0001000_757941.json",
  "check_cluster/Multi_0100_10000_0000000_b1bf11.json",
  "check_cluster/Multi_0100_2000_0010000_c2672e.json",
  "check_cluster/Multi_0100_4000_0001000_7979d0.json",
  "check_cluster/Wolff_0100_10000_0000000_d2651e.json",
  "check_cluster/Wolff_0100_2000_0010000_5f711f.json",
  "check_cluster/Wolff_0100_4000_0001000_bc554a.json",
  "check_cluster/Wolff_0100_10000_0000000_9df170.json",
  "check_cluster/Wolff_0100_4000_0001000_dc670f.json",
  "check_cluster/Wolff_0100_2000_0010000_fd0b00.json",
]

data = []
maxxpow = 4

noCouples     = maxxpow*(maxxpow+1)//2   # number of correlators computed for each lag
noLags        = 1                            # number of lags taken into account

klabels   = []
for k in range(noLags):
  for i in range(1,maxxpow+1):
    for j in range(i, maxxpow+1):
      klabels.append( f"C{{{i},{j}}}({k+1})_k")

plot_dir = Path( "./check_cluster_plots" )

labels = []
for i in range(1,maxxpow+1):
  labels.append(f"x{i}")
for i in range(1,maxxpow+1):
  for j in range(i,maxxpow+1):
    labels.append(f"C_{i}_{j}")

slices   = 10
interval = { 0: 1, 100: 0.5, 10: 0.7 }

for file in tqdm.tqdm(files):
  sim_data  = json.load(open(file,'r'))
  alg       = sim_data['algorithm']           # algorithm use
  g         = sim_data['coupling']            # perturbation .25*g*q^4 parameter
  samples   = sim_data['samples']             # samples

  ncols = 1 + maxxpow + maxxpow * (maxxpow+1) // 2

  therm_samples = samples // 7

  filepath          = Path(sim_data['folder']) / 'data.csv'
  #simulated_data    = pd.read_csv(filepath, sep=',', skiprows=samples//2, engine='c').to_numpy().reshape(-1,ncols)
  simulated_data    = np.abs( pd.read_csv(filepath, sep=',', skiprows=therm_samples, engine='c').to_numpy().reshape(-1,ncols) )

  s_blocks = int( np.nanmax( [ sim_data[ klabel ] for klabel in klabels ] ) )
  n_blocks = ( samples - therm_samples ) // s_blocks

  blocked_data = np.zeros( (n_blocks, ncols) )

  for i in range(n_blocks):
    for j in range(ncols):
#      for j in range(s_blocks):
        blocked_data[i,j] += np.mean(simulated_data[i*s_blocks:(i+1)*s_blocks, j])

  sim_dict = {
    'alg' : alg,
    'g'   : g,
    'ups' : sim_data['ups'],
    'raw' : blocked_data
  }

  for i in range(1,ncols):
    sim_dict[labels[i-1]] = np.histogram( np.abs(simulated_data[:,i]), slices, range=(0,interval[g]) )

  data.append(sim_dict)

gg = set( sim['g'] for sim in data )

for lidx, label in enumerate(labels):
  print(label)
  for g in gg:
    sims = [ simm for simm in data if simm['g'] == g ]
    for idx, sim in enumerate(sims):
      # plt.errorbar(
      #   ( sim[label][1][1:] + sim[label][1][:-1] )/2 + ( sim[label][1][-1] - sim[label][1][0] ) / (3 * slices) * (idx+1/2) / (len(sims)),
      #   sim[label][0],
      #   yerr      = np.sqrt( sim[label][0] ),
      #   label     = f"{sim['alg']} {sim['g']}",
      #   fmt       = 'x',
      #   linestyle = '--'
      # )
      if sim['alg'] == 'metropolis':
        for idx2, sim2 in enumerate(sims):
          if idx < idx2:# and sim2['alg'] == 'wolff':
            print(
              f"alg {sim['alg']}, g {g:03d}, ups {sim['ups']:03d} vs alg {sim2['alg']}, g {g}, ups {sim2['ups']:03d}: {ks_2samp( sim['raw'][1+lidx], sim2['raw'][1+lidx], 'two-sided' ).pvalue}"
            )
    # plt.xlabel(r"$O$")
    # plt.ylabel(r"$n$")
    # plt.legend()
    # plt.savefig( plot_dir / f"not_blocked_{g}_{label}.svg" )
    # plt.clf()
  print("")