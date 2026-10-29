"""Isolated board-power acquisition. No blocking utilization/memory queries.

Power is the same real NVML mW reading with actual API-return timestamps,
50ms requested cadence and unchanged 250ms call/support limits. Utilization
and memory are explicitly missing, not fabricated or used for energy. The
original separate MPS/process resource guard remains unchanged.
"""
import math
import os
from pathlib import Path
import time
import nvml_direct_observer_20261029 as direct
import rental_nvml_observer_20261029 as rental

GPU=rental.GPU
VERSION='rental400_actual_NVML_board_power_only_250ms_ownAffinity_v2'

def acquire(nvml,handle,phase,clock=time.monotonic,unix=time.time):
    start=clock();mw=nvml.nvmlDeviceGetPowerUsage(handle);receipt=clock();stamp=unix()
    watts=float(mw)/1000.
    if not 0<watts<=1000 or not math.isfinite(watts):raise ValueError('Invalid actual NVML power')
    if not start<=receipt or receipt-start>.25:raise ValueError(f'Power API exceeds original250ms bound: start={start!r}, return={receipt!r}, elapsed={receipt-start!r}, power_mW={mw!r}')
    sample=direct.Sample(stamp,receipt,phase,watts,None,None)
    return sample,dict(query_start_monotonic_s=start,power_API_return_monotonic_s=receipt,
        query_end_monotonic_s=receipt,power_milliwatts=mw,phase=phase,valid_numeric_row=True,
        utilization_percent=None,memory_used_mib=None,not_queried=['utilization','memory'],
        timestamp_source='actual power API return; no backfill or interpolation')

ORIGINAL_SAMPLER=direct.Sampler
def cpu_ticks():
    result={}
    for line in Path('/proc/stat').read_text().splitlines():
        label,*values=line.split()
        if label.startswith('cpu') and label[3:].isdigit():
            v=list(map(int,values));result[int(label[3:])]=(sum(v[:8]),v[3]+v[4])
    return result

def select_cpu(allowed,before,after):
    def idle(cpu):
        a,b=before[cpu],after[cpu];total=b[0]-a[0]
        return ((b[1]-a[1])/total if total>0 else 0.,cpu)
    return max(allowed,key=idle)

class PowerSampler(ORIGINAL_SAMPLER):
    def start(self):
        # This container denies priority elevation. Pin only our observer to one
        # currently least-busy allowed CPU to reduce migrations. No reservation,
        # realtime scheduling, changes to model threads or other users' processes.
        allowed=sorted(os.sched_getaffinity(0));before=cpu_ticks();time.sleep(.2)
        selected=select_cpu(allowed,before,cpu_ticks());os.sched_setaffinity(0,{selected})
        if os.sched_getaffinity(0)!={selected}:raise RuntimeError('Own observer affinity request failed')
        super().start()
        self.identity.update(version=VERSION,query_functions=['nvmlDeviceGetPowerUsage'],
            own_observer_CPU_affinity=[selected],exclusive_CPU_reservation=False,realtime_scheduling=False,
            missing_diagnostic_fields=['utilization_percent','memory_used_mib'],
            resource_guard='unchanged independent original MPS/process guard',
            power_observer_source_sha256=direct.hashlib.sha256(Path(__file__).read_bytes()).hexdigest())

def configure():
    rental.configure_receiver();direct.acquire=acquire;direct.Sampler=PowerSampler
    direct.VERSION=VERSION;direct.observer_command=observer_command

def observer_command(gpu,path):
    if gpu!=GPU:raise ValueError('Exact approved rental GPU required')
    return [direct.sys.executable,'-u',str(Path(__file__).resolve()),'--observer','--gpu',gpu,'--raw',str(path)]

if __name__=='__main__':configure();direct.main()
