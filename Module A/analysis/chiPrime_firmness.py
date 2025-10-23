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

# contains preliminary peak data
peak_data     = []

# contains final peak data
new_peak_data = []

# contains fit results
fit_data      = {}

def parabola(x, a, b, c):
  return a * x**2 + b * x + c

def jackknife_var_chi(abs_m, m2):
  """
  Jackknife variance of chiP = <m^2> - <|m|>^2
  given time series arrays abs_m = |m| and m2 = m^2 (per unit volume).
  """
  # ensure data are numpy arrays
  abs_m = np.asarray(abs_m, dtype=float)
  m2    = np.asarray(m2,    dtype=float)

  # limit case
  n = abs_m.size
  if n <= 1:
    return 0.0

  sum_am  = abs_m.sum()
  sum_m2  = m2.sum()

  # leave-one-out means
  mean_am_lo = (sum_am - abs_m) / (n - 1)
  mean_m2_lo = (sum_m2 - m2)    / (n - 1)

  # jackknife replicates of F
  chiP_lo = mean_m2_lo - mean_am_lo**2

  # jackknife variance
  chiP_bar = chiP_lo.mean()
  var_jk = (n - 1) / n * np.sum((chiP_lo - chiP_bar)**2)
  return (chiP_bar, var_jk)

def process_file(descriptor):
  # load metadata
  sim_data = json.load( open(descriptor, 'r') )

  L      = sim_data['L']
  geo    = sim_data['geometry']
  volume = L*L*(2 if geo == 'hex' else 1)
  alg    = sim_data['algorithm']

  # # for debugging
  # if alg=='metropolis' and L<40:
  #  return

  # find index of the peak
  indices = [ i for i, peak in enumerate(peak_data) if peak['L']==L and peak['geometry']==geo and peak['algorithm']==alg ]
  if len(indices) != 1:
    print(f"Error: for the descriptor {descriptor} there are {len(indices)} matching peaks! Indices: ", *indices)
    sys.exit(136)
  
  index = indices[0]
  peak  = peak_data[index]
  k     = peak['k']

  # load simulated data
  filepath       = Path( sim_data['folder']  ) / 'data.csv'
  simulated_data = np.loadtxt(filepath, delimiter=',', skiprows=1+5*k)

  # blocks data
  noBlockedMeasures = simulated_data.shape[0] // k
  blockedMeasures = np.zeros( ( noBlockedMeasures, 2 ), dtype=float )

  for i in range(noBlockedMeasures):
    blockedMeasures[i,0] = np.mean( np.abs(simulated_data[i*k:(i+1)*k, 0])   ) / volume
    blockedMeasures[i,1] = np.mean(        simulated_data[i*k:(i+1)*k, 0]**2 ) / volume**2

  m  = blockedMeasures[:,0]
  m2 = blockedMeasures[:,1]
  am = np.abs(m)

  # Jackknife variance for F = <m^2> - <|m|>^2
  # Note: this variance is not reduced! (It is not a "relative error")
  chiP, var_chiP_jk = jackknife_var_chi(am, m2)

  # convert to chi' error at THIS beta & L:
  beta = sim_data['beta']

  chi_prime_err = (beta * volume) * np.sqrt(var_chiP_jk)
  chi_prime     = (beta * volume) * chiP

  peak_data[index]['sim_results'].append( {'beta': beta, 'chi_prime': chi_prime, 'chi_prime_uncert': chi_prime_err } )

if __name__ == "__main__":
  # parse arguments from command line
  parser = argparse.ArgumentParser( description="It does the final fit." )
  parser.add_argument( "--peaksInfo", type=str, help="JSON data file with peaks of chi (it contains blocking k).", required=True )
  parser.add_argument( "input",       type=str, help="JSON descriptors of the simulation.", nargs='+' )

  args = parser.parse_args()

  # load peaks info
  with open(args.peaksInfo) as p:
    peak_data = json.load(p)
  
  for datum in peak_data:
    datum['sim_results'] = []

  # read input files
  files = args.input

  for f in tqdm(files):
    process_file(f)

  colmap = cmap['winter']

  for peak_datum in peak_data:
    algorithm = peak_datum['algorithm']
    geometry  = peak_datum['geometry']
    L         = peak_datum['L']

    # sort data by beta
    peak_datum['sim_results'] = sorted( peak_datum['sim_results'], key=lambda x: x['beta'] )

    betas         = np.asarray([ s['beta']             for s in peak_datum['sim_results'] ])
    chiPrime      = np.asarray([ s['chi_prime']        for s in peak_datum['sim_results'] ])
    sigmaChiPrime = np.asarray([ s['chi_prime_uncert'] for s in peak_datum['sim_results'] ])

    if len(betas) == 0:
      continue

    # find maxima
    popt, pcov = curve_fit(
      parabola,
      betas,
      chiPrime,
      sigma = sigmaChiPrime,
      p0=[
        1/(2*peak_datum['beta0']**2),
        -peak_datum['beta_max']/peak_datum['beta0']**2,
        peak_datum['Xmax']+peak_datum['beta_max']**2/(2*peak_datum['beta0']**2) ],
      absolute_sigma=True,
      )
    a,b,c = popt

    beta_max     = -b/(2*a)
    chiPrime_max = parabola(beta_max, *popt)
    # Jacobians at (a,b,c)
    # x* = -b/(2a)  ->  [∂/∂a, ∂/∂b, ∂/∂c] = [ 0.5*b/a^2,  -1/(2a),  0 ]
    Jx = np.array([0.5*b/(a**2), -1.0/(2.0*a), 0.0])

    # y* = c - b^2/(4a)  ->  [ b^2/(4a^2),  -b/(2a),  1 ]
    Jy = np.array([(b*b)/(4.0*a*a), -b/(2.0*a), 1.0])

    # Variances via error propagation
    sigma_beta_max    = np.sqrt( Jx @ pcov @ Jx )
    sigma_chiPrime_pc = np.sqrt( Jy @ pcov @ Jy )
    
    print(f"ALG: {algorithm},\tGEO: {geometry},\tL: {L}")

    fit_beta  = []
    fit_sigma = []

    # parity check
    for p in [0,1]:
      peak_datum_p = [ j for i,j in enumerate(peak_datum['sim_results']) if i % 2 == p ]

      betas         = np.asarray([ s['beta']             for s in peak_datum_p ])
      chiPrime      = np.asarray([ s['chi_prime']        for s in peak_datum_p ])
      sigmaChiPrime = np.asarray([ s['chi_prime_uncert'] for s in peak_datum_p ])

      if len(betas) == 0:
        continue

      # find maxima
      popt, pcov = curve_fit(
        parabola,
        betas,
        chiPrime,
        sigma = sigmaChiPrime,
        p0=[
          1/(2*peak_datum['beta0']**2),
          -peak_datum['beta_max']/peak_datum['beta0']**2,
          peak_datum['Xmax']+peak_datum['beta_max']**2/(2*peak_datum['beta0']**2) ],
        absolute_sigma=True,
        )
      a,b,c = popt
      beta_max     = -b/(2*a)
      # Jacobians at (a,b,c)
      # x* = -b/(2a)  ->  [∂/∂a, ∂/∂b, ∂/∂c] = [ 0.5*b/a^2,  -1/(2a),  0 ]
      Jx = np.array([0.5*b/(a**2), -1.0/(2.0*a), 0.0])

      # Variances via error propagation
      sigma_beta    = np.sqrt( Jx @ pcov @ Jx )

      fit_beta.append(beta_max)
      fit_sigma.append(sigma_beta)

    print(f"Parity firmness check    :      delta beta/sigma beta = {np.max(np.abs(fit_beta-beta_max)/fit_sigma)}")

    # ignoring one sim check
    for p in range(len(peak_datum['sim_results'])):
      peak_datum_p = [ j for i,j in enumerate(peak_datum['sim_results']) if i != p ]

      betas         = np.asarray([ s['beta']             for s in peak_datum_p ])
      chiPrime      = np.asarray([ s['chi_prime']        for s in peak_datum_p ])
      sigmaChiPrime = np.asarray([ s['chi_prime_uncert'] for s in peak_datum_p ])

      if len(betas) == 0:
        continue

      # find maxima
      popt, pcov = curve_fit(
        parabola,
        betas,
        chiPrime,
        sigma = sigmaChiPrime,
        p0=[
          1/(2*peak_datum['beta0']**2),
          -peak_datum['beta_max']/peak_datum['beta0']**2,
          peak_datum['Xmax']+peak_datum['beta_max']**2/(2*peak_datum['beta0']**2) ],
        absolute_sigma=True,
        )
      a,b,c = popt
      beta_max     = -b/(2*a)
      # Jacobians at (a,b,c)
      # x* = -b/(2a)  ->  [∂/∂a, ∂/∂b, ∂/∂c] = [ 0.5*b/a^2,  -1/(2a),  0 ]
      Jx = np.array([0.5*b/(a**2), -1.0/(2.0*a), 0.0])

      # Variances via error propagation
      sigma_beta    = np.sqrt( Jx @ pcov @ Jx )

      fit_beta.append(beta_max)
      fit_sigma.append(sigma_beta)
      
    print(f"Ignoring one point check : max  delta beta/sigma beta = {np.max(np.abs(fit_beta-beta_max)/fit_sigma)}")
    print(f"Ignoring one point check : mean delta beta/sigma beta = {np.mean(np.abs(fit_beta-beta_max)/fit_sigma)}")
    print("")