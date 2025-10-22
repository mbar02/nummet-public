import os
import sys

import numpy  as np
from scipy.interpolate import splrep, splev
from matplotlib        import pyplot    as plt
from matplotlib        import colormaps as cmap

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
    "figure.figsize": (7,5),
    "legend.labelspacing": 0.1,
    "legend.handletextpad": 0.2,
    "legend.borderpad": 0.2,
    "text.usetex": True,
})

data = []

def parabola(x, a, b, c):
  return a * x**2 + b * x + c

def process_file(descriptor):
  global data

  # read json descriptor
  with open(descriptor, 'r') as f:
    meta = json.load(f)

  # read data
  datafile = Path( meta["folder"] ) / "data.csv"

  measures = np.loadtxt(datafile, delimiter=',', skiprows=1000, dtype=int)

  volume = meta["L"]**2 * ( 2 if meta["geometry"] == 'hex' else 1 )

  # compute reduced magnetization
  m = measures[:,0] / volume
  m2 = m*m
  am = np.abs(m)

  m2_mean = np.mean(m2)
  am_mean = np.mean(am)

  amsigma2 = np.var(am) / am_mean
  m2sigma2 = np.var(m2) / m2_mean

  # store data
  data.append( {
    'algorithm': meta["algorithm"],
    'geometry' : meta["geometry"],
    'L'        : meta["L"],
    'beta'     : meta["beta"],
    'am'       : am_mean,
    'm2'       : m2_mean,
    'amsigma'  : amsigma2,
    'm2sigma'  : m2sigma2,
  } )

if __name__ == "__main__":
  # parse arguments from command line
  parser = argparse.ArgumentParser( description="Plots the reduced magnetization." )
  parser.add_argument( "--plotDir",   type=str, help="Output plot directory.", required=True )
  parser.add_argument( "--outData",   type=str, help="JSON output data file.", required=True )
  parser.add_argument( "input",     type=str, help="JSON descriptors of the simulation.", nargs='+' )

  args = parser.parse_args()

  plot_dir    = Path( args.plotDir )
  out_file    = args.outData

  if not os.path.isdir(plot_dir):
    print(f"Error: {plot_dir} is not a valid directory")
    sys.exit(1)

  for f in tqdm(args.input):
    process_file(f)

  colmap = cmap['winter']

  outdata = []

  # loop over algorithms and geometries for plotting
  for algorithm in set( d['algorithm'] for d in data ):
    data_alg = [ d for d in data if d['algorithm'] == algorithm ]
    for geometry in set( d['geometry'] for d in data if d['algorithm'] == algorithm ):
      data_alg_geo = [ d for d in data_alg if d['geometry'] == geometry ]
      maxL = max( d['L'] for d in data_alg_geo )
      # loop over L
      for L in sorted( set( d['L'] for d in data_alg_geo ), reverse=True ):
        data_alg_geo_L = [ d for d in data_alg_geo if d['L'] == L ]
        color = colmap( 0.05 + 0.90*(L/maxL) )

        # extract measurements
        sorted_data = sorted(data_alg_geo_L, key=lambda x: x['beta'])
        betas = np.array( [ d['beta'] for d in sorted_data ] )
        m2s   = np.array( [ d['m2']   for d in sorted_data ] )
        ams   = np.array( [ d['am']   for d in sorted_data ] )
        chiPrime = ( m2s - ams**2 ) * betas * L**2

        colorAlphed = [*color[:-1], 0.5]

        min_beta = min(betas)
        max_beta = max(betas)

        # plots data
        plt.plot(
          betas,
          chiPrime,
          '+',
          label = f"L = {int(L)}",
          color=color
        )

        # plots spline
        tck = splrep(betas,np.log10(chiPrime),s=0.0003,k=3)
        betas_sp    = np.linspace(min_beta,max_beta, 100)
        chiPrime_sp = 10**splev(betas_sp, tck)
        plt.plot(
          betas_sp,
          chiPrime_sp,
          color=colorAlphed
        )

        maxIdx = np.argmax(chiPrime)
        length = len(betas)

        # fit with a parabola around the maximum
        data_fit = [
          ( betas[i], chiPrime[i] )
          for i in range(len(betas))
          if abs(i-maxIdx) <= 1 or chiPrime[i] >= 0.8*chiPrime[maxIdx]
        ]

        betasF, chiPrimeF = zip(*data_fit)

        popt = np.polyfit(betasF, chiPrimeF, 2)
        a,b,c = popt

        delta = -np.sqrt(1/4 * b**2 - a*c)/(1.5*a)
        beta_fit = np.linspace( max(-b/(2*a) - delta, min_beta), min(-b/(2*a) + delta, max_beta), 100)

        plt.plot(
          beta_fit,
          parabola(beta_fit, *popt),
          '--',
          color=colorAlphed,
        )
        beta0 = np.sqrt(abs(0.5/a))
        Xmax  = -b**2/(4*a)+c

        # store data about the maximum
        outdata.append( {
          'algorithm': algorithm,
          'geometry' : geometry,
          'L'        : L,
          'beta_max' : -b/(2*a),
          'Xmax'     : Xmax,
          'beta0'    : beta0,
        } )

      plt.xlabel(r"$\beta$")
      plt.ylabel(r"$\chi '$")
      plt.xlim(1.01*min_beta-0.01*max_beta, 1.01*max_beta-0.01*min_beta)
      plt.yscale("log")
      plt.legend().get_frame().set_boxstyle("square")
      plt.tight_layout()
      plt.savefig(plot_dir / f"chi_{algorithm}_{geometry}.svg")
      plt.clf()

  # save data
  outdata = sorted(outdata, key=lambda x: (x['algorithm'], x['geometry'], x['L']) )
  with open(out_file, 'w') as f:
    json.dump(outdata, f, indent=2)