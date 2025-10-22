import os
import sys
import json
from   pathlib import Path
import argparse
import importlib
import re

# --- GENERAL PARAMETERS              ---
EXE_PATH = "build/"
BINARIES       = {
  'wolff'        : "AHOWolff",
  'metropolis'   : "AHOMetropolis",
  'multicluster' : "AHOMulticluster",
}
PREFIXES       = {
  'wolff'        : "Wolff",
  'metropolis'   : "Metro",
  'multicluster' : "Multi",
}
PROCESSOR_TYPE = {
  'wolff'        : "cpu",
  'metropolis'   : "gpu",
  'multicluster' : "gpu",
}
PROCESSOR_TYPES = set(PROCESSOR_TYPE.values())
OUTPUT_DIR     = "./simulations"
MAX_PARAL_JOBS     = 8
MAX_PARAL_JOBS_GPU = 1

SIM_ALGORITHMS = [ ]
UPS_SIM        = lambda alg, n, beta, g:  0
SAMPLES_SIM    = lambda alg, n, beta, g:  0
LAGS_SIM       = lambda n, beta, g:      [ ]
TEMPERAT_SIM   = lambda n, g:            [ ]
PATH_SIZE      = lambda g:               [ ]
COUPLING_CONST = [ ]

# --- RUNNING                          ---
def main():
  # print banner
  print("""
### Welcome to QUSUMaNO.py! ###############################
# Quick Utility for Simulations:                          #
#                 Unified Managenent for Numerical Output #
###########################################################
  """)

  # parse arguments from command line
  parser = argparse.ArgumentParser( description="It starts the simulations." )
  parser.add_argument( "config",       type=str, help="Python config file." )
  parser.add_argument( "--configargs", type=str, help="Extra arguments for the config() function.", nargs='+' )

  args = parser.parse_args()
  configArgs = args.configargs if args.configargs is not None else []

  config_path = Path( args.config )
  if not os.path.isfile(config_path):
    print(f"Error: {config_path} is not a valid file")
    sys.exit(1)

  config_dir = str(config_path.parent)
  
  # add the config directory to the path
  if config_dir not in sys.path:
    sys.path.append( config_dir )
  
  # import the config module and run the config function
  try:
    config_module = importlib.import_module( config_path.stem )
    if hasattr( config_module, 'config' ):
      setting = config_module.config(*configArgs)
      for key, value in setting.items():
        globals()[key] = value
    else:
      print(f"Error: {config_path} does not have a config() function")
      sys.exit(1)
  except Exception as e:
    print(f"Error: could not import {config_path}: {e}")
    sys.exit(1)

  # Create output folder
  os.makedirs(OUTPUT_DIR, exist_ok=True)

  # Create queue
  process_queue = {
    key: [] for key in PROCESSOR_TYPES
  }

  for alg in SIM_ALGORITHMS:
    for g in COUPLING_CONST:
      for n in PATH_SIZE(g):
        for beta in TEMPERAT_SIM(n, g):
          dir    = OUTPUT_DIR
          prefix = f"{PREFIXES[alg]}_{n:04d}_{int(1e3*beta):04d}_{int(100*g):07d}_"
          print("Prefix: ", prefix)

          pattern = re.compile(prefix + r"[0-9a-f]{6}$")
          for outLabel in Path(dir).glob("*"):
            if pattern.search(outLabel.name):
              print(outLabel)

              ups     = UPS_SIM(alg, n, beta, g)
              samples = SAMPLES_SIM(alg, n, beta, g)
              data = {
                'algorithm' : alg,
                'coupling'  : g,
                'path_size' : n,
                'beta'      : beta,
                'ups'       : ups,
                'samples'   : samples,
                'lags'      : LAGS_SIM(n,beta,g),
                'folder'    : str(outLabel)
              }
              with open(outLabel.with_suffix(".json"), 'w') as f:
                json.dump(data, f, indent=2)

if __name__ == '__main__':
  main()