import sys

import numpy  as np
import pandas as pd
import glob
import os
import re
from tqdm   import tqdm
from matplotlib import pyplot as plt
from matplotlib        import colormaps as cmap
from scipy.optimize import curve_fit

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
    "text.latex.preamble": r"\usepackage\{babel}[italian]",
})
colmap = cmap['winter']

def autocorrelation(x, cutoff):
  dx = ( x - np.mean(x) ) / max(x)
  ac = []
  ks = []
  s  = np.var(x)
  l = len(x)
  for i in tqdm.tqdm( range( 0, cutoff, max( int(cutoff/200), 1 ) ) ):
    ks.append(i)
    ac.append( np.mean( [ dx[j] * dx[i+j] for j in range(l) if i+j < l ] ) / s )
  return (ks, ac)

def exponential_fit(x, a, tau):
  return a * np.exp(-x/tau)

def linear_fit(x, a, b):
  return a * x + b

tau_opts=[]
L_opts=[]
def process_file(filename):
  data = np.readcsv(filename, delimiter=",", skiprows=0)
  E = data[:, 1] # energy
  ks, ac = autocorrelation(E[10000:], 30)
  L = int(re.match(r"^.*\_(\d{2,3})\_critical\.csv$", filename).group(1))
  L_opts.append(L)
  plt.plot(ks, ac, label=f"$L={L}$", color=colmap(0.0098 * L))
  
  popt, pcov = curve_fit(exponential_fit, ks, ac, p0=(1, 1))
  _, tau = popt
  tau_opts.append(tau)
  plt.plot(ks, exponential_fit(np.array(ks), *popt), linestyle='--', color=colmap(0.0098 * L), linewidth=0.5)
  return


if __name__ == "__main__":
  if len(sys.argv) < 3:
    print("Syntax: python3 visualize.py <input_dir> <simulation_type>")
    sys.exit(1)

  input_dir = sys.argv[1]
  simulation_type = sys.argv[2]
  pattern = f"{simulation_type}_*.csv"
  files = glob.glob(os.path.join(input_dir, pattern))
  print(files)

  
  if not files:
    print(f"No files matching {pattern} in {input_dir}")
    sys.exit(1)

  for p in tqdm.tqdm(files):
    process_file(p)

  plt.yscale('log')
  plt.xlabel(r"lag $t$")
  plt.ylabel(r"$\hat{\rho}(t)$")
  plt.title(f"Autocorrelation — {simulation_type}")
  plt.legend(fontsize=6, loc="best")
  plt.tight_layout()
  plt.savefig(f"analysis/figures/autocorrelation_{simulation_type}.svg", bbox_inches='tight')
  plt.clf()

  plt.plot(L_opts, tau_opts, '+',label="$\tau\ped{exp} (L)$ extracted")
  popt, pcov = curve_fit(linear_fit, np.log(L_opts), np.log(tau_opts), p0=(1, 1))
  plt.plot(L_opts, np.exp(popt[1]) * np.array(L_opts) ** popt[0], linestyle='--', color='black', label="Fitted $\tau\ped{exp} (L) = a L^b$")
  print(f"Fitted parameters: a={np.exp(popt[1]):.2f}, b={popt[0]:.2f}")
  plt.xscale('log')
  plt.yscale('log')
  plt.xlabel(r"$L$")
  plt.ylabel(r"$\tau\ped{exp}$")
  plt.savefig(f"analysis/figures/autocorrelation_scaling_{simulation_type}.svg", bbox_inches='tight')
