import sys

import numpy  as np
from matplotlib import pyplot as plt

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

def autocorrelation(x, cutoff):
  dx = x - np.mean(x)
  ac = []
  ks = []
  s  = np.var(x)
  l = len(x)
  for i in tqdm.tqdm( range( 0, cutoff, max( int(cutoff/200), 1 ) ) ):
    ks.append(i)
    ac.append( np.mean( [ dx[j] * dx[i+j] for j in range(l) if i+j < l ] ) / s )
  return (ks, ac)

if __name__ == "__main__":
  if len(sys.argv) < 3:
    print("Syntax: python3 visualize.py <namefile.csv> <thermalization> <max_t>")
    exit(-1)

  print("Script started")

  filename = sys.argv[1]
  thermal  = int(sys.argv[2])
  max_t    = int(sys.argv[3])

  # High temperature binning, varying L
  data = []
  
  if "blocked" in filename:
    data = np.loadtxt(filename, delimiter=r",", skiprows=3)
  else:
    data = np.loadtxt(filename, delimiter=r",", skiprows=1)
  
  dataT = np.asarray(data[thermal:])

  print("data imported. Make histogram")

  counts, edges  = np.histogram(dataT[:,0], bins = 100, density = True)

  centers = (edges[:-1] + edges[1:])/2 

  plt.plot(centers, counts)

  plt.xlabel(r"$\bar{m}$")
  plt.ylabel("Density")
  plt.savefig(r"analysis/figures/binning.svg")
  plt.clf()
  #plt.show()

  print("Plot thermalization")

  # Thermalization
  plt.plot(data[0:thermal,0])
  plt.xlabel("Number of iterations")
  plt.ylabel(r"$m$")
  plt.savefig(r"analysis/figures/thermalization.svg")
  #plt.show()
  plt.clf()

  # Thermalization
  plt.plot(np.abs(data[0:thermal,0]))
  plt.xlabel("Number of iterations")
  plt.ylabel(r"$\bar{m}$")
  plt.savefig(r"analysis/figures/abs_thermalization.svg")
  #plt.show()
  plt.clf()

  print("Plot autocorrelation")

  # Exponetial autocorrelation time
  plt.plot(*autocorrelation(np.abs(dataT[:,0]),max_t))

  plt.xlabel("$k$ (Lag)")
  plt.ylabel(r"$C(k)$")
  plt.savefig(r"analysis/figures/autocorrelation.svg")
  #plt.show()


