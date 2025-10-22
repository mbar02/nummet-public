import sys
import tqdm
import asyncio
import json
from pathlib import Path
import argparse
import multiprocessing

MAX_PARAL_JOBS = max(1, multiprocessing.cpu_count())
EXE_PATH       = "build/"
BINARIES       =  {
    'metrocpu'  : { 'sqr': "IsingMetropolisCPU", 'tri': "IsingMetropolisCPUTri", 'hex': "IsingMetropolisCPUHex" },
    'metrogpu'  : { 'sqr': "IsingMetropolis",    'tri': "IsingMetropolisTri",    'hex': "IsingMetropolisHex"    },
    'wolff'     : { 'sqr': "IsingWolff",         'tri': "IsingWolffTri",         'hex': "IsingWolffHex"         },
}

# --- RUNNING                          ---
async def main():
  # print banner
  print("""
### Welcome to QUSUMaNO_recover.py! #######################
# Quick Utility for Simulations:                          #
#  Unified Managenent for Numerical Output - recover util #
###########################################################
  """)

  # parse arguments from command line
  parser = argparse.ArgumentParser( description="It restarts the simulations." )
  parser.add_argument( "descriptors", type=str, help="JSON descriptors to check.", nargs='+' )

  args = parser.parse_args()
  descriptors_path = [ Path( i ) for i in args.descriptors ]

  # check if there is something to do
  if len(descriptors_path) == 0:
    print("No descriptor passed.")
    parser.print_help()
    sys.exit(136)
  
  # Create queue
  process_queue = {
    'wolff':      [],
    'metropolis': [],
  }
  
  # check if simulation completed
  for i in descriptors_path:
    datum  = json.load(open(i, 'r'))
    folder = Path(datum['folder'])
    log    = folder / "sim_stdout.log"
    if not log.exists() or not "Simulazione terminata" in log.read_text():
      raw_cmd = [
        datum['L'],
        datum['UPS'],
        datum['samples'],
        datum['beta'],
        folder / "data.csv",
      ]
      process_queue[ datum['algorithm'] ].append(
        {
          'cmd'   : [ str(i) for i in raw_cmd ],
          'data'  : datum,
        }
      )

  # ask if all ok
  print("Do you confirm that they are the simulations to recover?")
  for q in sum(process_queue.values(),[]):
    print(q['data']['folder'])
  
  resp = input("Are you sure? [y/N] ").strip().lower()
  if resp != "y":
      print("Abort")
      sys.exit(1)

  print("Start recovering simulations")
  
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