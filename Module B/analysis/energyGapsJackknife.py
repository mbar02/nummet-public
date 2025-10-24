import numpy  as np
import cmath
from scipy.linalg      import eigh
import pandas as pd

import json
from pathlib import Path
import argparse

from tqdm import tqdm

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

def jackknife(noObs, noLags, obs_idx, reduced_lags, correlators, data, powMeans, th_energies):
  mockPow      = np.zeros(noObs)
  mockCorr     = np.zeros(correlators.shape)
  nrows        = data.shape[0]
  crossProd    = np.zeros(correlators.shape)
  for k in range(2):
    crossProd[k,:,:]  = correlators[k,:,:] + np.outer(powMeans,powMeans)
  mockEnergies = []
  mockLamdas   = []

  for jackRow in range(nrows):
    mockPow = [(nrows*powMeans[i] - data[jackRow, i])/(nrows-1) for i in range(noObs)]
    for k in range(noLags):
      counter = 0
      for i in range(noObs):
        for j in range(i, noObs):
          mockCorr[k,j,i] = mockCorr[k,i,j] = (nrows*crossProd[k,i,j] - data[jackRow,obs_idx[k][counter]-1])/(nrows-1) - mockPow[i]*mockPow[j]
          counter += 1

    for k in range(noLags-1):
      tau      = reduced_lags[k]
      lam, _   = eigh(mockCorr[1+k], mockCorr[0],driver="gvx",check_finite=False)
      order    = np.argsort(-lam)
      lam      = lam[order]
      gaps     = - np.log(abs(lam))/tau
    mockEnergies.append(gaps)
    mockLamdas.append(lam)
  
  mockEnergies = np.vstack(mockEnergies)
  mockLamdas   = np.vstack(mockLamdas)
  gaps         = [np.nanmean(mockEnergies[:,i]) for i in range(noObs)]
  sigma_gaps   = [np.sqrt(len(mockLamdas[:,i]) * np.nanvar(mockEnergies[:,i], ddof=1)) for i in range(noObs)]
  lambdas      = [np.nanmean(mockLamdas[:,i]) for i in range(noObs)]
  sigma_lambdas= [np.sqrt(len(mockLamdas[:,i]) * np.nanvar(mockLamdas[:,i])) for i in range(noObs)]
  sigma_E_sys  = [np.exp( - tau * (th_energies[-1] - th_energies[i]) ) / tau for i in range(noObs)]

  return gaps, sigma_gaps, sigma_E_sys, sigma_lambdas

def gevpsigma(eigval, noObs, k, eigvec, cov_mat, norm):
  sigma2 = 0
  for t1 in range(2):
    tmp1 = (-eigval if t1==0 else 1)
    for t2 in range(2):
      tmp2 = (-eigval if t2==0 else 1)
      for i1 in range(noObs):
        for j1 in range(noObs):
          for i2 in range(noObs):
            for j2 in range(noObs):
              sigma2 += cov_mat[t1*(1+k),t2*(1+k),i1,j1,i2,j2]*tmp1*tmp2*eigvec[i1]*eigvec[j1]*eigvec[i2]*eigvec[j2] #/ \
              #  ( eigvec @ norm @ eigvec ) ** 2
  return sigma2

def gevp(noObs, noLags, reduced_lags, correlators, cov_mat, th_energies):
  data_shape  = (noLags-1, noObs)
  lambda_mat  = np.zeros(data_shape)
  energy_mat  = np.zeros(data_shape)
  sigma_lam   = np.zeros(data_shape)
  sigma_E     = np.zeros(data_shape)
  sigma_E_sys = np.zeros(data_shape)

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

    eig         = eig[:,order]
    energy      = - np.log(np.abs(lam))/tau

    lambda_mat[k] = lam
    energy_mat[k] = energy

    for idx in range(len(lam)):
      eigval = lam[idx]
      eigvec = eig[:,idx]

      sigma2 = gevpsigma(eigval, noObs, k, eigvec, cov_mat, raw_corr[0])
      
      sigma_lam[k,idx]    = np.sqrt(sigma2)
      sigma_E[k,idx]      = abs(np.sqrt(sigma2)/(tau*eigval))
      sigma_E_sys[k, idx] = np.exp( - tau * (th_energies[-1] - th_energies[idx]) ) / tau
  
  return energy_mat, sigma_E, sigma_E_sys, sigma_lam

def process_file(descriptor,labels,obs_idx, noLags, MAX_X_POW):
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

  # Solve GEVP and find gaps
  try:
    reduced_lags    = lags[1:] - lags[0]
    jgaps, jsigma_gaps,jsysSigma,jsigma_lam = jackknife(
      noObs,noLags,obs_idx,reduced_lags,corr,blockedMeasures,powMeans,
      [energy([g],n)[0] for n in range(noObs + 2)])
    gaps,sigma_gaps,syssigma_gaps, sigma_lam = gevp(
      noObs, noLags, reduced_lags, corr, cov,
      [ energy( [g], n )[0] for n in range(noObs+2) ]
    )
    bestGaps     = np.array([gaps[np.nanargmin([(sigma_gaps/gaps)[:,col]]),col] for col in range(MAX_X_POW)])
    bestSigma    = np.array([sigma_gaps[np.nanargmin([(sigma_gaps/gaps)[:,col]]),col] for col in range(MAX_X_POW)])
    bestSysSigma = np.array([syssigma_gaps[np.nanargmin([(sigma_gaps/gaps)[:,col]]),col] for col in range(MAX_X_POW)])
    bestSigmaLam = np.array([sigma_lam[np.nanargmin([(sigma_gaps/gaps)[:,col]]),col] for col in range(MAX_X_POW)])
    
    order        = np.argsort(bestGaps)
    bestGaps     = bestGaps[order]
    bestSigma    = bestSigma[order]
    bestSysSigma = bestSysSigma[order]
    bestSigmaLam = bestSigmaLam[order]

  except Exception as e: 
    print("alg, g, beta, n")
    print( alg, g, beta, path_size)
    print("Something wrong happened")
    print(e)
    print("\n")
  else:
    print("alg, g, beta, n")
    print( alg, g, beta, path_size)
    print("jackknife sigmas:        ", *(np.array(jsigma_gaps).flatten()) )
    print("hellmann-feynman sigmas: ", *(bestSigma.flatten()            ) )
    print("ratio: ", *((jsigma_gaps/bestSigma).flatten()                ) )
    print("\n")

def main():
  # parse arguments from command line
  parser = argparse.ArgumentParser( description="It does the final fit." )
  parser.add_argument( "input"       ,   type=str, help="JSON descriptors of the simulation.", nargs='+' )

  args = parser.parse_args()

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
    process_file(f,labels,obs_idx,noLags,MAX_X_POW)

if __name__ == "__main__":
  main()