import numpy as np
import cmath
import multiprocessing

eta = 1e-3

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

maxn = 4

def tau_i(g):
  return np.log(10/eta)/(energy(g,maxn+2)-energy(g,maxn))

def tau_f(g):
  return np.log(eta/1e-14)/(energy(g,maxn)-energy(g,0))

steps = np.array([3, 10, 30, 100])

# --- CONFIGURATION PARAMETERS         ---
#   Change only here!
def config():
  setting = {
    'SIM_ALGORITHMS' : [
      #'wolff',
      #'metropolis',
      'multicluster',
    ],
    'UPS_SIM'        : lambda alg, n, beta, g: int( 10000 if alg == 'wolff' else 3000 if alg == 'metro' else 10 ),
    'SAMPLES_SIM'    : lambda alg, n, beta, g: int( 1e5 ),
    'TEMPERAT_SIM'   : lambda n, g:            [ 12 * (tau_f(g)-tau_i(g)) ],
    'LAGS_SIM'       : lambda n, beta, g:      [ int( i * n/72 ) for i in range(1,7) ],
    'PATH_SIZE'      : lambda g:               np.array( np.array(steps) * 72, dtype=int ).tolist(),
    'COUPLING_CONST' : np.logspace(-2,2,20).tolist(),
    'OUTPUT_DIR'     : '/home/contazzi/local/preliminary-aho',

    'MAX_PARAL_JOBS' : max(1, multiprocessing.cpu_count()-1),
  }
  return setting