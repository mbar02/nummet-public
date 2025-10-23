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

  if ( alg == 'wolff' and L < 1e2 ) or ( alg == 'metropolis' and L < 4e1 ):
    return

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
  parser.add_argument( "--plotDir",   type=str, help="Output plot directory.", required=True )
  parser.add_argument( "--outData",   type=str, help="JSON output data file.", required=True )
  parser.add_argument( "input",       type=str, help="JSON descriptors of the simulation.", nargs='+' )

  args = parser.parse_args()

  plot_dir    = Path( args.plotDir )
  out_file    = args.outData

  # Create output folder
  os.makedirs(plot_dir, exist_ok=True)

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

    color = colmap( 0.5 )
    colorAlphed = [*color[:-1], 0.5]

    # sort data by beta
    peak_datum['sim_results'] = sorted( peak_datum['sim_results'], key=lambda x: x['beta'] )

    betas         = np.asarray([ s['beta']             for s in peak_datum['sim_results'] ])
    chiPrime      = np.asarray([ s['chi_prime']        for s in peak_datum['sim_results'] ])
    sigmaChiPrime = np.asarray([ s['chi_prime_uncert'] for s in peak_datum['sim_results'] ])

    if len(betas) == 0:
      continue

    min_beta = min(betas)
    max_beta = max(betas)

    plt.subplot(5,1,(1,3))
    plt.plot(
      betas,
      chiPrime,
      '+',
      color=color
    )

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
    new_peak_data.append({
      'algorithm'   : algorithm,
      'geometry'    : geometry,
      'L'           : L,
      'beta_max'    : beta_max,
      's_beta_max'  : sigma_beta_max,
      'Xmax'        : chiPrime_max,
      's_Xmax'      : sigma_chiPrime_pc,
    })

    residuals = (chiPrime - parabola(betas, *popt)) / sigmaChiPrime

    beta_plot = np.linspace( min_beta, max_beta, 100 )
    plt.plot(
      beta_plot,
      parabola(beta_plot, *popt),
      '--',
      color=color
    )

    plt.ylabel(r"$\chi '$")
    plt.xlim( min_beta-(max_beta-min_beta)*0.05, max_beta+(max_beta-min_beta)*0.05 )

    plt.subplot(4,1,4)
    plt.errorbar(
      betas,
      residuals,
      yerr=1.0,
      fmt='o',
      color=colorAlphed,
      markersize=2,
    )
    plt.axhline(0, color='black', linestyle='--', linewidth=0.5)
    plt.xlabel(r"$\beta$")
    plt.ylabel(r"Normalized residuals")
    plt.savefig( plot_dir / f"chiMax_{algorithm}_{geometry}_{L}.svg" )
    plt.clf()

  def beta_scaling(x, beta_c, nu, p):
    y = beta_c + p / x ** ( 1/nu )
    return y
  
  def chi_scaling(x, c0, c1, gamma):
    return c0 + c1 * x**gamma
  
  expected_beta = {
    'sqr': 0.4406,
    'tri': 0.2747,
    'hex': 0.6585
  }

  colors_geo = {
    'hex': colmap(0.25),
    'sqr': colmap(0.5),
    'tri': colmap(0.75)
  }

  # loop over algorithms and geometries for plotting beta(L)
  for algorithm in set( d['algorithm'] for d in new_peak_data ):
    data_alg = [ d for d in new_peak_data if d['algorithm'] == algorithm ]
    fit_data[algorithm] = {}
    for geometry in set( d['geometry'] for d in new_peak_data if d['algorithm'] == algorithm ):
      fit_data[algorithm][geometry] = {}
      data_alg_geo = [ d for d in data_alg if d['geometry'] == geometry ]
      color = colmap( 0.5 )

      Ls         = np.asarray([ d['L']          for d in data_alg_geo ])
      betas      = np.asarray([ d['beta_max']   for d in data_alg_geo ])
      chiPrimes  = np.asarray([ d['Xmax']       for d in data_alg_geo ])
      sBetas     = np.asarray([ d['s_beta_max'] for d in data_alg_geo ])
      sChiPrimes = np.asarray([ d['s_Xmax']     for d in data_alg_geo ])

      try:
        popt, pcov = curve_fit(
          beta_scaling,
          Ls,
          betas,
          sigma=sBetas,
          absolute_sigma=True,
          p0=[ expected_beta[geometry], 1, -0.6 ]
        )
        beta_c, nu, p = popt

      except Exception as e:
        print(f"ALG: {algorithm},\tGEO: {geometry}")
        print(e)

        plt.errorbar(
          Ls, beta_c - betas, 
          capsize=2,
          yerr=sBetas,
          label=geometry,
          color=colors_geo[geometry],
          fmt='o', linestyle='',
          mfc='none', mec='black'
        )
      else:
        residuals = (betas - beta_scaling(Ls, *popt)) / sBetas
        chisq = np.sum(residuals**2) / (len(residuals)-len(popt))

        LL = np.linspace(
          0.95*Ls.min(),
          1.05*Ls.max(),
          100
        )
        plt.errorbar(
          Ls, beta_c - betas, 
          capsize=2,
          yerr=sBetas,
          label=geometry,
          color=colors_geo[geometry],
          fmt='o', linestyle='',
          mfc='none', mec='black'
        )
        plt.plot(
          LL,
          beta_c - beta_scaling(LL, *popt),
          color=[ *colors_geo[geometry][:-1], 0.5 ]
        )

        # save fit data
        fit_data[algorithm][geometry]['betaL-popt']  = popt
        fit_data[algorithm][geometry]['betaL-sigma'] = np.sqrt( np.diag(pcov) )
        fit_data[algorithm][geometry]['betaL-pcov']  = pcov
        fit_data[algorithm][geometry]['betaL-chisq'] = chisq
     
    plt.xlabel(r"$L$")
    plt.ylabel(r"$\beta_{cr}-\beta_{max}$")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig( plot_dir /  f"beta_scaling_{algorithm}.svg")
    plt.clf()

  # loop over algorithms and geometries for plotting chi'(L)
  for algorithm in set( d['algorithm'] for d in new_peak_data ):
    data_alg = [ d for d in new_peak_data if d['algorithm'] == algorithm ]
    for geometry in set( d['geometry'] for d in new_peak_data if d['algorithm'] == algorithm ):
      data_alg_geo = [ d for d in data_alg if d['geometry'] == geometry ]
      color = colmap( 0.5 )

      Ls         = [ d['L']          for d in data_alg_geo ]
      betas      = [ d['beta_max']   for d in data_alg_geo ]
      chiPrimes  = [ d['Xmax']       for d in data_alg_geo ]
      sBetas     = [ d['s_beta_max'] for d in data_alg_geo ]
      sChiPrimes = [ d['s_Xmax']     for d in data_alg_geo ]

      try:
        popt, pcov = curve_fit(
          chi_scaling,
          Ls,
          chiPrimes,
          sigma=sChiPrimes,
          absolute_sigma=True
        )
        c0, c1, gamma = popt
      except Exception as e:
        print(f"ALG: {algorithm},\tGEO: {geometry}")
        print(e)

        plt.errorbar(
          Ls, chiPrimes, 
          capsize=2,
          yerr=sChiPrimes,
          label=f"{geometry}",
          color=colors_geo[geometry],
          fmt='o', linestyle='',
          mfc='none', mec='black'
        )
      else:
        residuals = (chiPrimes - chi_scaling(Ls, *popt)) / sChiPrimes
        chisq = np.sum(residuals**2) / (len(residuals)-len(popt))

        LL = np.linspace(
          0.95*min(Ls),
          1.05*max(Ls),
          100
        )
        plt.errorbar(
          Ls, chiPrimes, 
          capsize=2,
          yerr=sChiPrimes,
          label=f"{geometry}",
          color=colors_geo[geometry],
          fmt='o', linestyle='',
          mfc='none', mec='black'
        )
        plt.plot(
          LL,
          chi_scaling(LL, *popt),
          color=[ *colors_geo[geometry][:-1], 0.5 ]
        )

        # save fit data
        fit_data[algorithm][geometry]['chiL-popt']  = popt
        fit_data[algorithm][geometry]['chiL-sigma'] = np.sqrt( np.diag(pcov) )
        fit_data[algorithm][geometry]['chiL-pcov']  = pcov
        fit_data[algorithm][geometry]['chiL-chisq'] = chisq
        
    plt.xlabel(r"$L$")
    plt.ylabel(r"$\chi'_{max}$")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig( plot_dir /  f"chi_scaling_{algorithm}.svg")
    plt.clf()
  
  def default_numpy(o):
    # ndarray -> list, numpy scalar -> python scalar
    if hasattr(o, 'tolist'):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    raise TypeError(f'Object of type {type(o)} is not JSON serializable')

  with open(out_file, 'w') as f:
    json.dump(fit_data, f, indent=2, default=default_numpy)