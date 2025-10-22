import os
import sys

import numpy  as np
from matplotlib        import pyplot    as plt
from matplotlib        import colormaps as cmap
from scipy.optimize    import curve_fit

import json
from pathlib import Path
import argparse

from tqdm import tqdm

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
})

# each element will be a dictionary with the usual info + autocorrelation times 
data = []
# dictionary with scaling results z and z'
fit_data = {}

# Achtung: in ac() x must have vanishing mean!
def _ac(dx, i, s):
  l = len(dx)
  return np.mean( [ dx[j] * dx[i+j] for j in range(l) if i+j < l ] ) / s

def autocorrelation(x, low_cutoff=0.005, high_cutoff=0.5):
  dx = x - np.mean(x)
  ac = []
  ks = []
  s  = np.var(x)
  l = len(x)
  lastk  = 0
  lastac = _ac(dx,lastk,s)
  nextk  = lastk + 1
  nextac = _ac(dx,nextk,s)
  # while over the cutoff, ignore
  while nextac > high_cutoff:
    lastk  = nextk
    lastac = nextac
    nextk  = lastk + 1
    nextac = _ac(dx,nextk,s)
  # while in the range of interest, save
  while nextac > low_cutoff:
    ac.append(lastac)
    ks.append(lastk)
    lastk  = nextk
    lastac = nextac
    nextk  = lastk + 1
    nextac = _ac(dx,nextk,s)
  # write last couple
  ac.append(nextac)
  ks.append(nextk)
  # ignore the rest
  return (ks, ac)

def exponential_fit(x, a, tau):
  return a * np.exp(-x/tau)

def tau_scaling(x,a,z):
  return a*(x**z)

def process_file(descriptor):
  global last_tau_exp
  # load metadata
  sim_data = json.load( open(descriptor, 'r') )

  k      = 300 # 10*max(int(sim_data['am_k']),int(sim_data['m2_k']))
  volume = sim_data['L']**2 * (2 if sim_data['geometry'] == "hex" else 1)
  geometry = sim_data['geometry']
  algorithm= sim_data['algorithm']

  # load simulated data
  filepath       = Path( sim_data['folder']  ) / 'data.csv'
  simulated_data = np.loadtxt(filepath, delimiter=',', skiprows=1)
  am = abs(simulated_data[:,0])/ volume

  # tau_exp fit
  lags, corr = autocorrelation(am)
  print("\n",lags,corr,"\n")
  plt.plot(lags,corr)
  plt.savefig(plot_dir / f"autocorr_{algorithm}_{geometry}_{sim_data['L']}.svg")
  plt.clf()
  popt, _ = curve_fit(exponential_fit, lags, corr)
  _, tau_exp = popt
  sim_data['tau_exp'] = tau_exp

  # blocking and tau_int^(am)
  am = am[int(5*tau_exp):] #exclude thermalization
  n_blocked  = int(len(am) // k)
  blocked_am = np.zeros(n_blocked, float)
  for i in range(n_blocked):
    blocked_am[i] = np.mean(am[i*k : (i+1)*k])
  sigma_naive = np.var(am)
  sigma_real  = np.var(blocked_am) / (n_blocked)
  tau_int = 0.5 * ( (n_blocked*k) * sigma_real / sigma_naive - 1 )
  sim_data['tau_int'] = tau_int

  data.append(sim_data)


if __name__ == "__main__":
  # parse arguments from command line
  parser = argparse.ArgumentParser( description="It does the final fit." )
  parser.add_argument( "--plotDir",   type=str, help="Output plot directory.", required=True )
  parser.add_argument( "input"    ,   type=str, help="JSON descriptors of the simulation.", nargs='+' )
  args = parser.parse_args()
  plot_dir    = Path( args.plotDir )

  # Create output folder
  os.makedirs(plot_dir, exist_ok=True)
  results_file = plot_dir / "z_exponents.txt"
  # clear old file if exists
  with open(results_file, "w") as f:
    f.write("# Algorithm  Geometry   z' (tau_int)   z (tau_exp)\n")

  # read input files
  files = args.input

  for f in tqdm(files):
    process_file(f)
  
  colmap = cmap['winter']
  colors_geo = {
  'hex': colmap(0.25),
  'sqr': colmap(0.5),
  'tri': colmap(0.75)
  }

  # loop over algorithms and geometries for plotting tau(L)
  for algorithm in set(d['algorithm'] for d in data):
    data_alg = [d for d in data if d['algorithm'] == algorithm]
    fit_data[algorithm] = {}
    for geometry in set(d['geometry'] for d in data if d['algorithm'] == algorithm):
      fit_data[algorithm][geometry] = {}
      data_alg_geo = [d for d in data_alg if d['geometry'] == geometry]
      color = colmap(0.5)

      # Build data arrays from dicts
      L = np.asarray([d['L'] for d in data_alg_geo])
      LL = np.linspace(0.95*L.min(),1.05*L.max(),100)
      tau_int = np.asarray([d['tau_int'] for d in data_alg_geo])
      tau_exp = np.asarray([d['tau_exp'] for d in data_alg_geo])

      # fit and plot tau_int
      popt, _ = curve_fit(tau_scaling, L, tau_int)
      _ , zprime = popt
      plt.figure(1)
      plt.plot(L, tau_int,'o', label=geometry, color=colors_geo[geometry])
      plt.plot(LL, tau_scaling(LL,*popt), '--', color = [*colors_geo[geometry][:-1],0.5])

      # fit and plot tau_exp
      popt,_ = curve_fit(tau_scaling, L, tau_exp)
      _, z = popt
      plt.figure(2)
      plt.plot(L, tau_exp,'o', label=geometry, color=colors_geo[geometry])
      plt.plot(LL, tau_scaling(LL,*popt), '--', color = [*colors_geo[geometry][:-1],0.5])
      
      # save results
      fit_data[algorithm][geometry]['tau_int'] = zprime
      fit_data[algorithm][geometry]['tau_exp'] = z
      
      # append to text file
      with open(results_file, "a") as f:
        f.write(f"{algorithm:10s} {geometry:6s} {zprime:10.3f} {z:10.3f}\n")

      plt.xlabel(r"$L$")
      plt.ylabel(r"$\tau_{exp}$")
      plt.xscale('log')
      plt.yscale('log')
      plt.legend()
      plt.tight_layout()
      # plt.title(f"$\zeta = {z:.2f}$")
      plt.savefig(plot_dir / f"tau_exp_{algorithm}_{geometry}.svg")
      plt.clf()

      plt.figure(1)
      plt.xlabel(r"$L$")
      plt.ylabel(r"$\tau_{int} ^{|m|}$")
      plt.xscale('log')
      plt.yscale('log')
      plt.legend()
      plt.tight_layout()
      # plt.title(f"$\zeta ' = {zprime:.2f} $")
      plt.savefig(plot_dir / f"tau_int_{algorithm}_{geometry}.svg")
      plt.clf()





 