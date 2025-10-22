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

def covariance(x,y):
  """Compute covariance between two 1D arrays"""
  dx = x-np.mean(x)
  dy = y-np.mean(y)
  return np.sum(dx*dy)/( (len(x)-1) * len(x) )

def parabola(x,a,c):
  return a*x*x + c

def Power(a,b):
  return a ** b

def Sqrt(a):
  return cmath.sqrt(a)

def Re(a):
  return a.real

# Mathematica's CForm says that:
def energy(gg,n):
  return np.array([(
    (g*(3+6*n+6*Power(n,2)))/(16.*Power(Re((2*Power(2,0.3333333333333333)*(1+2*n))/Power(324*g+1944*g*n+4536*g*Power(n,2)+5184*g*Power(n,3)+2592*g*Power(n,4)+Sqrt(-6912*Power(1+2*n,6)+Power(324*g+1944*g*n+4536*g*Power(n,2)+5184*g*Power(n,3)+2592*g*Power(n,4),2)),0.3333333333333333)+Power(324*g+1944*g*n+4536*g*Power(n,2)+5184*g*Power(n,3)+2592*g*Power(n,4)+Sqrt(-6912*Power(1+2*n,6)+Power(324*g+1944*g*n+4536*g*Power(n,2)+5184*g*Power(n,3)+2592*g*Power(n,4),2)),0.3333333333333333)/(6.*Power(2,0.3333333333333333)*(1+2*n))),2))+((0.5+n)*(1+Power(Re((2*Power(2,0.3333333333333333)*(1+2*n))/Power(324*g+1944*g*n+4536*g*Power(n,2)+5184*g*Power(n,3)+2592*g*Power(n,4)+Sqrt(-6912*Power(1+2*n,6)+Power(324*g+1944*g*n+4536*g*Power(n,2)+5184*g*Power(n,3)+2592*g*Power(n,4),2)),0.3333333333333333)+Power(324*g+1944*g*n+4536*g*Power(n,2)+5184*g*Power(n,3)+2592*g*Power(n,4)+Sqrt(-6912*Power(1+2*n,6)+Power(324*g+1944*g*n+4536*g*Power(n,2)+5184*g*Power(n,3)+2592*g*Power(n,4),2)),0.3333333333333333)/(6.*Power(2,0.3333333333333333)*(1+2*n))),2)))/(2.*Re((2*Power(2,0.3333333333333333)*(1+2*n))/Power(324*g+1944*g*n+4536*g*Power(n,2)+5184*g*Power(n,3)+2592*g*Power(n,4)+Sqrt(-6912*Power(1+2*n,6)+Power(324*g+1944*g*n+4536*g*Power(n,2)+5184*g*Power(n,3)+2592*g*Power(n,4),2)),0.3333333333333333)+Power(324*g+1944*g*n+4536*g*Power(n,2)+5184*g*Power(n,3)+2592*g*Power(n,4)+Sqrt(-6912*Power(1+2*n,6)+Power(324*g+1944*g*n+4536*g*Power(n,2)+5184*g*Power(n,3)+2592*g*Power(n,4),2)),0.3333333333333333)/(6.*Power(2,0.3333333333333333)*(1+2*n))))
  ) for g in gg])

maxn = 4

def gevpsigma(eigval, noObs, k, eigvec, cov_mat):
  sigma2 = 0
  for t1 in range(2):
    tmp1 = (-eigval if t1==0 else 1)
    for t2 in range(2):
      tmp2 = (-eigval if t2==0 else 1)
      for i1 in range(noObs):
        for j1 in range(noObs):
          for i2 in range(noObs):
            for j2 in range(noObs):
              sigma2 += cov_mat[t1*(1+k),t2*(1+k),i1,j1,i2,j2]*tmp1*tmp2*eigvec[i1]*eigvec[j1]*eigvec[i2]*eigvec[j2]
  return sigma2

def gevp(noObs, noLags, reduced_lags, correlators, cov_mat, th_energies):
  data_shape  = (noLags-1, noObs)
  lambda_mat  = np.zeros(data_shape)
  energy_mat  = np.zeros(data_shape)
  sigma_lam   = np.zeros(data_shape)
  sigma_E     = np.zeros(data_shape)
  sigma_E_sys = np.zeros(data_shape)

  # for k in range(noLags):
  #   negval = False
  #   for i in range(noObs):
  #     for j in range(noObs):
  #       if correlators[k,i,j] < 0:
  #         correlators[k,i,j] = 0
  #         negval = True
  #   if negval:
  #     print(correlators[k])

  raw_corr = []

  for k in range(noLags):
    raw_corr.append( correlators[k,:,:].tolist() )
    for i in range(noObs):
      for j in range(noObs):
        raw_corr[k][i][j] = float(raw_corr[k][i][j])

  for k in range(noLags-1):
    tau = reduced_lags[k]
    lam, eig    = eigh(
      raw_corr[1+k], raw_corr[0],
      driver="gv", check_finite=False
    )

    order       = np.argsort(-lam)
    lam         = lam[order]
    #DEBUG
    print("\n")
    print(lam)
    deb_sigma = []
    for idx in range(len(lam)):
      eigval = lam[idx]
      eigvec = eig[:,idx]

      sigma2 = gevpsigma(eigval, noObs, k, eigvec, cov_mat)

      deb_sigma.append(float(np.sqrt(sigma2)))
    print(deb_sigma)
    #END DEBUG
    eig         = eig[:,order]
    energy      = - np.log(np.abs(lam))/tau
    #energy      = - np.log(lam)/tau

    lambda_mat[k] = lam
    energy_mat[k] = energy

    for idx in range(len(lam)):
      eigval = lam[idx]
      eigvec = eig[:,idx]

      sigma2 = gevpsigma(eigval, noObs, k, eigvec, cov_mat)
      
      sigma_lam[k,idx]    = np.sqrt(sigma2)
      sigma_E[k,idx]      = abs(np.sqrt(sigma2)/(tau*eigval))
      sigma_E_sys[k, idx] = np.exp( - tau * (th_energies[-1] - th_energies[idx]) ) / tau
  
  return energy_mat, sigma_E, sigma_E_sys

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
  simulated_data    = pd.read_csv(filepath, sep=',', skiprows=samples//7, engine='c').to_numpy().reshape(-1,ncols)

  # blocks relevant data -> exclude even-odd powers correlators
  noBlockedMeasures = simulated_data.shape[0]//k_blocking
  blockedMeasures   = np.zeros((noBlockedMeasures, ncols - 1))
  
  for i in range(noBlockedMeasures):
    for j in range(ncols-1):
      blockedMeasures[i,j] = np.mean(simulated_data[i*k_blocking:(i+1)*k_blocking,1+j])

  # build correlation matrix (avg of blocked measures) and covariance matrix for correlators 
  noObs    = MAX_X_POW
  powMeans = np.array([np.mean(blockedMeasures[:, i]) for i in range(noObs)])
  corr     = np.zeros((noLags, noObs, noObs))
  cov      = np.zeros((noLags,noLags,noObs,noObs,noObs,noObs))

  for k in range(noLags):
    counter  = 0
    for i in range(noObs):
      for j in range(i,noObs):
        corr[k,j,i] = corr[k,i,j] = np.mean( blockedMeasures[:, obs_idx[k][counter]-1] ) - powMeans[i] * powMeans[j]
        #print(k,i,j,obs_idx[k][counter])
        counter += 1

  for k1 in range(noLags):
    for k2 in range(k1,noLags):
      counter1 = 0
      for i1 in range(noObs):
        for j1 in range(i1,noObs):
          counter2 = 0
          for i2 in range(noObs):
            for j2 in range(i2,noObs):
              tmp = (
                covariance( blockedMeasures[:, obs_idx[k1][counter1]-1],blockedMeasures[:, obs_idx[k2][counter2]-1] ) + 
              - covariance( blockedMeasures[:, obs_idx[k1][counter1]-1],blockedMeasures[:, i2] ) * powMeans[j2]   + 
              - covariance( blockedMeasures[:, obs_idx[k1][counter1]-1],blockedMeasures[:, j2] ) * powMeans[i2]   + 
              - covariance( blockedMeasures[:, obs_idx[k2][counter2]-1],blockedMeasures[:, i1] ) * powMeans[j1]   + 
              - covariance( blockedMeasures[:, obs_idx[k2][counter2]-1],blockedMeasures[:, j1] ) * powMeans[i1]   + 
                covariance( blockedMeasures[:, i1],blockedMeasures[:, i2] ) * powMeans[j1] * powMeans[j2]     + 
                covariance( blockedMeasures[:, i1],blockedMeasures[:, j2] ) * powMeans[j1] * powMeans[i2]     + 
                covariance( blockedMeasures[:, j1],blockedMeasures[:, i2] ) * powMeans[i1] * powMeans[j2]     + 
                covariance( blockedMeasures[:, j1],blockedMeasures[:, j2] ) * powMeans[i1] * powMeans[i2]
              )
              kk = [k1,k2]
              ii = [i1,i2]
              jj = [j1,j2]
              for l in range(2):
                cov[ kk[l], kk[1-l], ii[l], jj[l], ii[1-l], jj[1-l] ] = \
                cov[ kk[l], kk[1-l], jj[l], ii[l], ii[1-l], jj[1-l] ] = \
                cov[ kk[l], kk[1-l], ii[l], jj[l], jj[1-l], ii[1-l] ] = \
                cov[ kk[l], kk[1-l], jj[l], ii[l], jj[1-l], ii[1-l] ] = tmp
              counter2 += 1
          counter1 += 1

  await procfile_semaphore_update.acquire()

  try:
    # Solve GEVP and find gaps
    reduced_lags    = lags[1:] - lags[0]
    gaps,sigma_gaps,syssigma_gaps = gevp(
      noObs, noLags, reduced_lags, corr, cov,
      [ energy( [g], n )[0] for n in range(noObs+2) ]
    )

    # print("alg, g, beta, n")
    # print( alg, g, beta, path_size )
    # print(gaps)
    # print("\n")

    bestGaps     = np.array([gaps[np.nanargmin([(sigma_gaps/gaps)[:,col]]),col] for col in range(MAX_X_POW)])
    bestSigma    = np.array([sigma_gaps[np.nanargmin([(sigma_gaps/gaps)[:,col]]),col] for col in range(MAX_X_POW)])
    bestSysSigma = np.array([syssigma_gaps[np.nanargmin([(sigma_gaps/gaps)[:,col]]),col] for col in range(MAX_X_POW)])
    order        = np.argsort(bestGaps)
    bestGaps     = bestGaps[order]
    bestSigma    = bestSigma[order]
    bestSysSigma = bestSysSigma[order]

  except Exception as e:
    print("alg, g, beta, n")
    print( alg, g, beta, path_size)
    print("Something wrong happened")
    print("\n")
    print(e)
    
  else:
    print(bestGaps)
    print(bestSigma)

    data.append({
      'algorithm'    : alg,
      'coupling'     : g,
      'step'         : beta/path_size,
      'gaps'         : bestGaps,
      'sigma_gaps'   : bestSigma,
      'syssigma_gaps': bestSysSigma,
    })
    
    
    # for i in range(MAX_X_POW):
    #   plt.errorbar(reduced_lags, gaps[:,i], sigma_gaps[:,i], label=f"$\\Delta E_{i}$")
    # plt.xlabel(r"$\tau$")
    # plt.ylabel(r"$\Delta E$")
    # plt.legend()
    # plt.savefig(plot_dir / f"gaps_{alg}_{int(1000*g):06d}.svg")
    # plt.clf()
  
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

  for algorithm in set( d['algorithm'] for d in data ):
    data_alg      = [d for d in data if d['algorithm'] == algorithm]
    gg            = []
    scaled_gaps   = []
    scaled_sigma  = []
    for g in set(( d['coupling'] for d in data_alg)):
      data_alg_g = [d for d in data_alg if d['coupling'] == g ]
      gg.append(g)
      aa     = []
      EE     = []
      sigEE  = []
      ssigEE = []
      for d in data_alg_g:
        aa.append(d['step'])
        EE.append(d['gaps'])
        sigEE.append(d['sigma_gaps'])
        ssigEE.append(d['syssigma_gaps'])

      aa        = np.array(aa)
      EE        = np.array(EE)
      sigEE     = np.array(sigEE)
      ssigEE    = np.array(ssigEE)
      tmp       = []
      sigma_tmp = []
      xx = np.logspace(np.log10(min(aa)*0.9), np.log10(max(aa)*1.1),100)
      if len(aa) == 1:
        tmp = EE[0]
        sigma_tmp = sigEE[0] + ssigEE[0]
        
      elif len(aa) > 1:
        for i in range(MAX_X_POW):
          popt, pcov = curve_fit(parabola,aa,EE[:,i],sigma=sigEE[:,i],p0=[1,EE[0,i]], absolute_sigma=True)
          tmp.append(popt[1])
          sigma_tmp.append(np.abs(pcov[1,1]+np.mean(ssigEE)))
          plt.errorbar(aa, EE[:,i],sigEE[:,i],fmt='x', label=f"$\\Delta E_{i+1}$")
          plt.plot(xx, parabola(xx,*popt),'--')
        plt.xlabel("$a$")
        plt.ylabel(r"$\Delta E$")
        plt.legend()
        plt.savefig(plot_dir / f"scaling_{algorithm}_{int(g*1000):06d}.svg")
        plt.clf()

      scaled_gaps.append(tmp)
      scaled_sigma.append(sigma_tmp)
    order        = np.argsort(gg)
    gg           = np.array(gg) [order]
    scaled_gaps  = np.array(scaled_gaps) [order]
    scaled_sigma = np.array(scaled_sigma) [order]

    xx = np.logspace(np.log10(gg[0]*0.9),np.log10(gg[-1]*1.1),100)
    E0_pv = energy(xx,0)

    colmap = cmap['winter']

    rmFree = 0

    for i in range(MAX_X_POW):
      plt.errorbar(
        gg * (1 + 0.0 * (i-MAX_X_POW/2)/MAX_X_POW), scaled_gaps[:,i]-rmFree*(i+1),
        scaled_sigma[:,i],
        label = f"$\\Delta E_{i+1}$", fmt=".", color=colmap( i/MAX_X_POW )
      )
      plt.plot(xx,energy(xx,i+1)-E0_pv-rmFree*(i+1),'--', color=colmap( i/MAX_X_POW ))
    plt.xlabel(r"$g$")
    plt.ylabel(r"$\Delta E_n$" if rmFree == 0 else r"$\Delta E_n - \Delta E_n^{\text{(free)}}$")
    plt.yscale('log')
    plt.xscale('log')
    plt.ylim([-0.89*rmFree+0.9,-1*rmFree+33])
    plt.legend()
    plt.savefig(plot_dir / f"gaps_coupling_{algorithm}.svg")
    plt.clf()

if __name__ == "__main__":
  asyncio.run(main())