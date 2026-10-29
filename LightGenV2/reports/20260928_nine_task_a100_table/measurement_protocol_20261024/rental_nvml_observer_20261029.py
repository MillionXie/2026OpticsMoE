"""Same direct NVML receiver, explicitly pinned to the newly authorized A10080."""
from pathlib import Path
import sys
import nvml_direct_observer_20261029 as original
# Before any formal worker, the same SSH endpoint exposed a different sole
# A10080. Preserve both hardware snapshots; never mix initial/measurement UUID.
GPU='GPU-30fecb94-0222-a3d5-d952-a91f2cab2f43'
def observer_command(gpu,path):
    if gpu!=GPU:raise ValueError('Exact new rental GPU required')
    return [sys.executable,'-u',str(Path(__file__).resolve()),'--observer','--gpu',gpu,'--raw',str(path)]

def configure_receiver():
    original.GPU=GPU
    original.POWER_LIMIT_MW=400000
    original.observer_command=observer_command

if __name__=='__main__':
    configure_receiver()
    original.main()
