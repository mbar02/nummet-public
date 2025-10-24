import numpy as np
import cmath
import multiprocessing

eta_sys = 0.9e-1
eta_vac = 1e-2
maxn = 4

def Power(a,b):
  return a ** b

def Sqrt(a):
  return cmath.sqrt(a)

def Re(a):
  return a.real

# Mathematica's CForm says that:
def energy(g,n):
  return (
    (g*(3+6*n+6*Power(n,2)))/(16.*Power(Re((2*Power(2,0.3333333333333333)*(1+2*n))/Power(324*g+1944*g*n+4536*g*Power(n,2)+5184*g*Power(n,3)+2592*g*Power(n,4)+Sqrt(-6912*Power(1+2*n,6)+Power(324*g+1944*g*n+4536*g*Power(n,2)+5184*g*Power(n,3)+2592*g*Power(n,4),2)),0.3333333333333333)+Power(324*g+1944*g*n+4536*g*Power(n,2)+5184*g*Power(n,3)+2592*g*Power(n,4)+Sqrt(-6912*Power(1+2*n,6)+Power(324*g+1944*g*n+4536*g*Power(n,2)+5184*g*Power(n,3)+2592*g*Power(n,4),2)),0.3333333333333333)/(6.*Power(2,0.3333333333333333)*(1+2*n))),2))+((0.5+n)*(1+Power(Re((2*Power(2,0.3333333333333333)*(1+2*n))/Power(324*g+1944*g*n+4536*g*Power(n,2)+5184*g*Power(n,3)+2592*g*Power(n,4)+Sqrt(-6912*Power(1+2*n,6)+Power(324*g+1944*g*n+4536*g*Power(n,2)+5184*g*Power(n,3)+2592*g*Power(n,4),2)),0.3333333333333333)+Power(324*g+1944*g*n+4536*g*Power(n,2)+5184*g*Power(n,3)+2592*g*Power(n,4)+Sqrt(-6912*Power(1+2*n,6)+Power(324*g+1944*g*n+4536*g*Power(n,2)+5184*g*Power(n,3)+2592*g*Power(n,4),2)),0.3333333333333333)/(6.*Power(2,0.3333333333333333)*(1+2*n))),2)))/(2.*Re((2*Power(2,0.3333333333333333)*(1+2*n))/Power(324*g+1944*g*n+4536*g*Power(n,2)+5184*g*Power(n,3)+2592*g*Power(n,4)+Sqrt(-6912*Power(1+2*n,6)+Power(324*g+1944*g*n+4536*g*Power(n,2)+5184*g*Power(n,3)+2592*g*Power(n,4),2)),0.3333333333333333)+Power(324*g+1944*g*n+4536*g*Power(n,2)+5184*g*Power(n,3)+2592*g*Power(n,4)+Sqrt(-6912*Power(1+2*n,6)+Power(324*g+1944*g*n+4536*g*Power(n,2)+5184*g*Power(n,3)+2592*g*Power(n,4),2)),0.3333333333333333)/(6.*Power(2,0.3333333333333333)*(1+2*n))))
  )

steps = np.array([10, 20, 40])

def tau(g, maxn):
  return np.abs( np.log(eta_sys) ) / ( energy(g, maxn + 2) - energy(g, maxn) )

def betaontau(g, maxn):
  tau_l = tau(g, maxn)
  beta = ( ( energy(g, maxn) - energy(g, 0) ) * tau_l - np.log(eta_vac) ) / ( energy(g, 1) - energy(g, 0) ) * 100
  bot = np.ceil( beta / tau_l )
  return int( bot )

max_gpu_processes = 2

# --- CONFIGURATION PARAMETERS         ---
#   Change only here!
def config():
  setting = {
    'SIM_ALGORITHMS' : [
      'wolff',
      #'metropolis',
      #'multicluster',
    ],
    'UPS_SIM'        : lambda alg, n, beta, g: int( 100 ),
    'SAMPLES_SIM'    : lambda alg, n, beta, g:
      int(np.ceil(
        ( np.exp(tau(g, maxn)*(energy(g, maxn)-energy(g, (maxn+1)%2+1))) / eta_sys ) ** 2
      * n / 1000 ) ) * 10,
    'TEMPERAT_SIM'   : lambda n, g:            [ betaontau(g, maxn) * tau(g, maxn) ],
    'LAGS_SIM'       : lambda n, beta, g:      [ int( i * n/betaontau(g, maxn) ) for i in [1,2] ],
    'PATH_SIZE'      : lambda g:               np.array( np.array(steps) * betaontau(g, maxn), dtype=int ).tolist(),
    'COUPLING_CONST' : (np.logspace(-2,2,20).tolist()),
    'OUTPUT_DIR'     : '/home/contazzi/local/final-aho-low-g-2',

    'MAX_PARAL_JOBS'     : 12-max_gpu_processes,
    'MAX_PARAL_JOBS_GPU' : max_gpu_processes,
  }
  return setting