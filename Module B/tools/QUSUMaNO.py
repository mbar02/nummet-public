import os
import sys
import tqdm
import asyncio
import json
import random
from   pathlib import Path
import argparse
import importlib

# --- FUNCTIONS AND METHODS            ---
def make_unique_file(prefix="sim_", length=6, dir="."):
  """
  Create a filename with 6 ending hexadecimal random digits.
  """
  while True:
    suffix = f"{random.randrange(16**length):0{length}x}"
    name = f"{prefix}{suffix}"
    path = Path(dir) / name
    if not os.path.exists(path) and not os.path.exists(Path(path).with_suffix(".json")):
      os.makedirs(path)
      return path

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
async def main():
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
          outLabel = Path(
            make_unique_file(
              dir    = OUTPUT_DIR,
              prefix = f"{PREFIXES[alg]}_{n:04d}_{int(1e3*beta):04d}_{int(100*g):07d}_"
            )
          )
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
          raw_cmd = [
            n, ups, samples, f"{float( beta ):.8f}", g, outLabel / "data.csv", *LAGS_SIM(n, beta, g)
          ]
          process_queue[PROCESSOR_TYPE[alg]].append(
            {
              'cmd'  : [ str(i) for i in raw_cmd ],
              'data' : data
            }
          )

  #  for i in PROCESSOR_TYPES:
  #    for j in process_queue[i]:
  #      print( ' '.join(j['cmd']) )
  
  # set up the semaphore
  SEMAPHORES = {
    'cpu' : asyncio.Semaphore(MAX_PARAL_JOBS),
    'gpu' : asyncio.Semaphore(MAX_PARAL_JOBS_GPU)
  }

  PBARS = {
    i : tqdm.tqdm(total=len(process_queue[i])) for i in PROCESSOR_TYPES
  }

  # command runner
  async def run_cmd( cmd, data, semaphore, pbar ):
    proc = await asyncio.create_subprocess_exec(
      *cmd,
      stdout=open( Path(data['folder']) / 'sim_stdout.log', 'w' ),
      stderr=open( Path(data['folder']) / 'sim_stderr.log', 'w' ),
    )
    await proc.wait()
    if( proc.returncode == 0 ):
      tqdm.tqdm.write(
        f"[ END    ] {PREFIXES[data['algorithm']]} n={data['path_size']:05d} beta={data['beta']:06.3f} g={data['coupling']}"
      )
    else:
      tqdm.tqdm.write(
        f"[ FAILED ] {PREFIXES[data['algorithm']]} n={data['path_size']:05d} beta={data['beta']:06.3f} g={data['coupling']}"
      )
    pbar.update()
    semaphore.release()

  # manager
  async def run_queue(proc_type):
    queue     = process_queue[proc_type]
    tasks     = []
    pbar      = PBARS[ proc_type ]
    for i in queue:
      data = i['data']
      cmd  = [ str( Path(EXE_PATH) / BINARIES[ data['algorithm'] ] ), *(i['cmd'])]
      await SEMAPHORES[ proc_type ].acquire()
      tqdm.tqdm.write(
        f"[ START  ] {PREFIXES[data['algorithm']]} n={data['path_size']:05d} beta={data['beta']:06.3f} g={data['coupling']}"
      )
      tasks.append( asyncio.create_task( run_cmd(
        cmd,
        data,
        SEMAPHORES[ proc_type ],
        pbar
      ) ) )
    await asyncio.gather(*tasks)
  
  # run tasks
  tasks = [ asyncio.create_task( run_queue(proc_type) ) for proc_type in PROCESSOR_TYPES ]
  await asyncio.gather( *tasks )

if __name__ == '__main__':
  asyncio.run(main())