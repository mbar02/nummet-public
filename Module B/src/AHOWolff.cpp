#include <cstdint>
#include <iostream>
#include <fstream>
#include <cmath>
#include <iomanip>

#include "../libs/pcg/pcg_basic.h"

#include "../libs/progressBar.h"
#include "../libs/errors.h"

using namespace std;

constexpr uint32_t MAX_X_POW   = 4;

inline double randomDouble(pcg32_random_t* rng) {
  return (double)( ( (uint64_t)pcg32_random_r(rng) << 32 ) + pcg32_random_r(rng) )/0x1.0p64; // Hex float literal is magic
}

double boxMullerState = NAN;
// It extract 2 numbers and return one, the other is in boxMullerState
inline double boxMuller(pcg32_random_t* rng) {
  double retval;
  if( isnan(boxMullerState) ) {
    double x, y, S;
    do {
      x = 1. - 2.*randomDouble(rng);
      y = 1. - 2.*randomDouble(rng);
      S = x*x + y*y;
    } while ( S > 1. || S == 0. );
    double r = sqrt( -2. * log(S) / S );
    retval         = x * r;
    boxMullerState = y * r;
  } else {
    retval = boxMullerState;
    boxMullerState = NAN;
  }
  return retval;
}

inline double annexProb(double delta_x, double a, double gamma) {
  return 1. - gamma * exp( delta_x * delta_x / (2.*a) );
}

inline void annexIdx(double* path, uint32_t idx, double& x1, double& x2, double& x3, double& x4) {
  x1 += path[idx];
  x2 += path[idx]*path[idx];
  x3 += path[idx]*path[idx]*path[idx];
  x4 += path[idx]*path[idx]*path[idx]*path[idx];
}

inline double alphaf(double a, double g, uint32_t Nc, double x2) {
  return sqrt(
    a/2. * ( Nc + 3. * g * x2 )
  );
}
inline double betaf (double a, double g, double x1,   double x3, double alpha) {
  return a/2. * (
    x1 + g * x3
  ) / alpha;
}

inline uint32_t prevIdx(uint32_t idx, uint32_t n) {
  if(idx == 0)
    return n-1;
  else
    return idx-1;
}

inline uint32_t nextIdx(uint32_t idx, uint32_t n) {
  uint32_t retval = idx+1;
  if(retval >= n)
    return retval - n;
  else
    return retval;
}

inline void update(
  double*         path,
  uint32_t        n,
  double          a,
  double          gamma,
  double          g,
  uint64_t        &accepted,
  pcg32_random_t* rng
) {
  uint32_t start = pcg32_boundedrand_r(rng, n);
  uint32_t end   = start;

  uint32_t size  = 1;

  // sum of x^n in the cluster
  double x1=0.0, x2=0.0, x3=0.0, x4=0.0;
  
  // probabilities at borders (raw ones are which can go over 1 or under 0)
  // un-raw values are canonical probabilities (0<=p<=1)
  double p_min = 0.0, p_max = 0.0, raw_p_min = 0.0, raw_p_max = 0.0;

  // not included neighbors
  uint32_t nidx_next=prevIdx(start,n), nidx_prev=nextIdx(end,n);

  annexIdx(path, start, x1, x2, x3, x4);

  // Annex the next
  {
  uint32_t max_last_site = prevIdx( start, n );
  do {
    uint32_t tmp = nextIdx( end, n );
    raw_p_max = annexProb( path[end] - path[tmp], a, gamma );
    if( raw_p_max <= randomDouble(rng) ) {
      nidx_next = tmp;
      break;
    }
    annexIdx(path, tmp, x1, x2, x3, x4);
    size++;
    end = tmp;
  } while( end != max_last_site );
  }

  // Annex the prec. Note that it starts with a while (not a do-while!)
  {
  uint32_t min_first_site = nextIdx( end, n );
  while( start != min_first_site ) {
    uint32_t tmp = prevIdx( start, n );
    raw_p_min = annexProb( path[start] - path[tmp], a, gamma );
    if( raw_p_min <= randomDouble(rng) ) {
      nidx_prev   = tmp;
      break;
    }
    annexIdx(path, tmp, x1, x2, x3, x4);
    size++;
    start = tmp;
  }
  }

  // Extract with gaussian weight
  double alpha = alphaf(a,g,size,x2);
  double beta  = betaf (a,g,x1,x3,alpha);

  double delta  = boxMuller(rng)/(alpha * M_SQRT2) - beta/alpha ;

  // Compute accept/reject rate
  // b values are the analogues of un-b-ed but computed in the trial state
  double x1b, x2b, x3b, delta2, delta3, delta4;
  delta2 = delta*delta;
  delta3 = delta2*delta;
  delta4 = delta3*delta;
  x1b = size * delta + x1;
  x2b = size * delta2 + 2. * delta  * x1 + x2;
  x3b = size * delta3 + 3. * delta2 * x1 + 3. * delta * x2 + x3;

  double alphab = alphaf(a,g,size,x2b);
  double betab  = betaf(a,g,x1b,x3b,alphab);

  double sqrtfreeactionb = (alphab * delta - betab);

  double p_max_b, p_min_b, raw_p_max_b, raw_p_min_b;

  if( size != n ) {
    raw_p_max_b = annexProb( path[end]   - path[nidx_next] + delta, a, gamma );
    raw_p_min_b = annexProb( path[start] - path[nidx_prev] + delta, a, gamma );
    p_max   = raw_p_max   > 0. ? raw_p_max : 0.;
    p_min   = raw_p_min   > 0. ? raw_p_min : 0.;
    p_max_b = raw_p_max_b > 0. ? raw_p_max_b : 0.;
    p_min_b = raw_p_min_b > 0. ? raw_p_min_b : 0.;
  } else {
    p_max_b = 0.;
    p_min_b = 0.;
    p_max   = 0.;
    p_min   = 0.;
    raw_p_max_b = 0.;
    raw_p_min_b = 0.;
    raw_p_max   = 0.;
    raw_p_min   = 0.;
  }

  double ar = alphab / alpha * exp(
    beta*beta - sqrtfreeactionb*sqrtfreeactionb - a * g * ( size * delta4/4. + x1 * delta3 )
    + log( 1. - p_max_b ) - log( 1. - raw_p_max_b )
    + log( 1. - p_min_b ) - log( 1. - raw_p_min_b )
    - log( 1. - p_max )   + log( 1. - raw_p_max )
    - log( 1. - p_min )   + log( 1. - raw_p_min )
  );

  // accept/reject
  if(randomDouble(rng)<ar) {
  uint32_t idx = start;
    for(uint32_t i = 0; i<size; i++) {
      path[idx] += delta;
      idx = nextIdx(idx,n);
    }
    accepted++;
    return;
  }
}

int main(int argc, char** argv) {
  // Parsing arguments from terminal
  if(argc < 8) {
    cout << "Syntax: " << argv[0]
         << " [n] [updates per samples] [samples] [beta] [g] [outputfile]"
         << " [lags for correlations]... " << endl;
    return TEINVAL;
  }
  uint32_t n       = (uint32_t)std::stoul(argv[1]);
  uint32_t UPS     = (uint32_t)std::stoul(argv[2]);
  uint32_t samples = (uint32_t)std::stoul(argv[3]);
  double   beta    =  (double) std::stod (argv[4]);
  double   g       =  (double) std::stod (argv[5]);

  double a = beta/n;

  uint32_t* h_lags = new uint32_t[argc-7];
  for(int i=0; i<argc-7; i++)
    h_lags[i] = (uint32_t)std::stoul(argv[7+i]);
  
  uint32_t lagsc = argc-7;

  if( n%2 ) {
    cout << "n must be even. Fallback to n+1." << endl;
    n++;
  }

  // Terminal columns
  const uint32_t columns = 80;

  // Init random number 
  uint64_t init_rng[2] = {0x0,0x0};
  ifstream randomFile("/dev/random",ios::binary);
  while(
    init_rng[0] < 0x10 || init_rng[0] > 0xfffffffffffffff0 ||
    init_rng[1] < 0x10 || init_rng[1] > 0xfffffffffffffff0
  ) {
    randomFile.read((char*) init_rng, 2*sizeof(uint64_t));
  }
  randomFile.close();
  pcg32_random_t rng;
  pcg32_srandom_r(&rng, init_rng[0], init_rng[1]);

  // Init output file
  ofstream file_p(argv[6]);

  // Fix output format
  file_p << std::scientific       // Scientific notation
         << std::showpos          // Always show the sign (+/-)
         << std::setprecision(8); // 8 decimal digits

  // Needed allocations
  double* path = new double[n];

  // Init path
  for(uint32_t i=0; i<n; i++)
    path[i] = 0.0;

  // Annex parameter
  double gamma = 1.0 / sqrt((double)n);

  // Progress bar
  asyncProgressBar pb(samples, columns);

  // Number of measures for each site (without ar rate)
  uint32_t N = MAX_X_POW + MAX_X_POW * (MAX_X_POW + 1) / 2 * lagsc;

  for(uint32_t i=0; i<samples; i++) {
    uint64_t ar=0;                        // acceptance rate
    // Update the path (UPS times)
    for(uint32_t j=0; j< UPS; j++)
      update(path, n, a, gamma, g, ar, &rng);
    // Save measures: declare array and initialize to zero

    double measures[N];
    double pows[n*MAX_X_POW];
    for(uint32_t site = 0; site < n; site++) {
      double xn = 1.0;
      for(uint32_t powidx = 0; powidx<MAX_X_POW; powidx++) {
        xn *= path[site];
        pows[site*MAX_X_POW+powidx] = xn;
      }
    }

    for(uint32_t idx=0;idx<N;idx++)
      measures[idx] = 0;
    uint32_t k = 0;
    for(uint32_t site = 0; site<n; site++){
      k = 0;
      for(uint32_t powidx = 0; powidx < MAX_X_POW; powidx++) {
        measures[k++] += pows[site*MAX_X_POW + powidx];
      }

      for(uint32_t lag_idx = 0; lag_idx < lagsc; lag_idx++){
        for(uint32_t pow1 = 0; pow1<MAX_X_POW; pow1++){
          for(uint32_t pow2 = pow1; pow2<MAX_X_POW; pow2++){
            measures[k++] += pows[ site*MAX_X_POW + pow1 ] *
                        pows[ ( (site+h_lags[lag_idx]) % n ) * MAX_X_POW + pow2 ];
          }
        }
      }
    }

    file_p << (double)ar / UPS;
    for(uint32_t i = 0; i<N; i++) {
      file_p << ", " << measures[i] / n;
    }
    file_p << endl;
    // Progress bar
    pb.update();
  }

  delete[] path;

  file_p << flush;
  file_p.close();

  return 0;
}
