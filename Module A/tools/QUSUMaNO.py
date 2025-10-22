import os
import sys
import tqdm
import asyncio
import json
import random
from pathlib import Path
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

# --- GENERAL PARAMETERS               ---
GEOMETRIES     = {'sqr': 1, 'tri': 2, 'hex': 3}

EXE_PATH       = "build/"
BINARIES       =  {
    'metrocpu'  : { 'sqr': "IsingMetropolisCPU", 'tri': "IsingMetropolisCPUTri", 'hex': "IsingMetropolisCPUHex" },
    'metrogpu'  : { 'sqr': "IsingMetropolis",    'tri': "IsingMetropolisTri",    'hex': "IsingMetropolisHex"    },
    'wolff'     : { 'sqr': "IsingWolff",         'tri': "IsingWolffTri",         'hex': "IsingWolffHex"         },
}

PREFIXES       = {
  'wolff':      "Wolff",
  'metropolis': "Metro",
}

SIM_ALGORITHMS = []
UPS_SIM        = lambda alg, geo, L, beta: []
SAMPLES_SIM    = lambda alg, geo, L, beta: []
TEMPERAT_SIM   = lambda alg, geo, L: []
LATTICE_SIZES  = lambda alg, geo: []
OUTPUT_DIR     = "./simulations"
MAX_PARAL_JOBS = 1

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
    'wolff':      [],
    'metropolis': [],
  }

  for alg in SIM_ALGORITHMS:
    for geo, _ in GEOMETRIES.items():
      for L in LATTICE_SIZES(alg, geo):
        for beta in TEMPERAT_SIM(alg, geo, L):
          outLabel = Path( make_unique_file( 
            dir    = OUTPUT_DIR,
            prefix = f"{PREFIXES[alg]}_{geo}_{L:03d}_{1e3*beta:3.0f}_",
          ) )
          ups     = UPS_SIM(alg, geo, L, beta)
          samples = SAMPLES_SIM(alg, geo, L, beta)
          data = {
            'algorithm' : alg,
            'geometry'  : geo,
            'L'         : L,
            'beta'      : beta,
            'UPS'       : ups,
            'samples'   : samples,
            'folder'    : str(outLabel),
          }
          with open(outLabel.with_suffix(".json"), 'w') as f:
            json.dump(data, f, indent=2)
          raw_cmd = [
            L,
            ups,
            samples,
            f"{float( beta ):.8f}",
            outLabel / "data.csv",
          ]
          process_queue[alg].append(
            {
              'cmd'   : [ str(i) for i in raw_cmd ],
              'data'  : data,
            }
          )
  
  # prepare the semaphore:
  SEMAPHORE = {
    'parallel'  : asyncio.Semaphore(MAX_PARAL_JOBS),
    'serial'    : asyncio.Semaphore(1),
    'metrolist' : asyncio.Semaphore(1),
  }

  PBARS = {
    'wolff'     : tqdm.tqdm(total=len(process_queue['wolff'])),
    'metropolis': tqdm.tqdm(total=len(process_queue['metropolis'])),
  }

  # order metropolis queue by L (smallest first for CPU)
  process_queue['metropolis'].sort(key=lambda x: x['data']['L'])

  # order wolff queue by L (largest first)
  process_queue['wolff'].sort(key=lambda x: x['data']['L'], reverse=True)

  # command runner
  async def run_cmd( cmd, data, semaphore, pbar ):
    proc = await asyncio.create_subprocess_exec(
      *cmd,
      stdout=open( Path(data['folder']) / 'sim_stdout.log', 'w' ),
      stderr=open( Path(data['folder']) / 'sim_stderr.log', 'w' ),
    )
    await proc.wait()
    tqdm.tqdm.write(f"[ END   ] {data['algorithm']} {data['geometry']} L={data['L']:3d} beta={data['beta']:.5f}")
    pbar.update()
    semaphore.release()

  # wolff manager
  async def run_wolff_queue():
    queue     = process_queue['wolff']
    tasks     = []
    pbar      = PBARS[ 'wolff' ]
    for i in queue:
      data = i['data']
      cmd  = [ str( Path(EXE_PATH) / BINARIES[ 'wolff' ][ data['geometry'] ] ), *(i['cmd'])]
      await SEMAPHORE[ 'parallel' ].acquire()
      tqdm.tqdm.write(f"[ START ] {data['algorithm']}      {data['geometry']} L={data['L']:3d} beta={data['beta']:.5f}")
      tasks.append( asyncio.create_task( run_cmd(
        cmd,
        data,
        SEMAPHORE[ 'parallel' ],
        pbar
      ) ) )
    await asyncio.gather(*tasks)

  # metropolis (GPU) manager
  async def run_metro_gpu_queue():
    queue     = process_queue['metropolis']
    tasks     = []
    pbar      = PBARS[ 'metropolis' ]
    while len(queue) > 0:
      await SEMAPHORE[ 'metrolist' ].acquire()
      if( len(queue) > 0 ):
        i = queue.pop(-1) # big L for GPU
      else:
        SEMAPHORE[ 'metrolist' ].release()
        break
      SEMAPHORE[ 'metrolist' ].release()
      data = i['data']
      cmd  = [ str( Path(EXE_PATH) / BINARIES[ 'metrogpu' ][ data['geometry'] ] ), *(i['cmd'])]
      await SEMAPHORE[ 'serial' ].acquire()
      tqdm.tqdm.write(f"[ START ] {data['algorithm']} (GPU) {data['geometry']} L={data['L']:3d} beta={data['beta']:.5f}")
      tasks.append( asyncio.create_task( run_cmd(
        cmd,
        data,
        SEMAPHORE[ 'serial' ],
        pbar
      ) ) )
    await asyncio.gather(*tasks)

  gpu_cpu_factor = 3 # how much faster is GPU vs CPU, emprirically determined for L >~ 100

  #metropolis (CPU) manager
  async def run_metro_cpu_queue():
    queue     = process_queue['metropolis']
    tasks     = []
    pbar      = PBARS[ 'metropolis' ]
    while len(queue) > gpu_cpu_factor:
      await SEMAPHORE[ 'metrolist' ].acquire()
      if( len(queue) > gpu_cpu_factor ):
        i = queue.pop(0) # little L for CPU
      else:
        SEMAPHORE[ 'metrolist' ].release()
        break
      SEMAPHORE[ 'metrolist' ].release()
      data = i['data']
      cmd  = [ str( Path(EXE_PATH) / BINARIES[ 'metrocpu' ][ data['geometry'] ] ), *(i['cmd'])]
      await SEMAPHORE[ 'parallel' ].acquire()
      tqdm.tqdm.write(f"[ START ] {data['algorithm']} (CPU) {data['geometry']} L={data['L']:3d} beta={data['beta']:.5f}")
      tasks.append( asyncio.create_task( run_cmd(
        cmd,
        data,
        SEMAPHORE[ 'parallel' ],
        pbar
      ) ) )
    await asyncio.gather(*tasks) 

  # start running wolff and metropolis GPU
  tasks = [ asyncio.create_task( f() ) for f in [ run_wolff_queue, run_metro_gpu_queue ] ]
  await asyncio.gather(tasks[0])
  # when wolff is done, start metropolis CPU
  tasks.append( asyncio.create_task( run_metro_cpu_queue() ) )
  await asyncio.gather(*tasks)

if __name__ == "__main__":
  asyncio.run(main())