"""Same direct NVML receiver, explicitly pinned to the newly authorized A10080."""
from pathlib import Path
import sys
import nvml_direct_observer_20261029 as original
GPU='GPU-9f0b4728-5ee6-2c98-315f-f77a6b7c3ac5'
def observer_command(gpu,path):
    if gpu!=GPU:raise ValueError('Exact new rental GPU required')
    return [sys.executable,'-u',str(Path(__file__).resolve()),'--observer','--gpu',gpu,'--raw',str(path)]
if __name__=='__main__':
    original.GPU=GPU
    original.main()
