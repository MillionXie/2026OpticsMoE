"""Isolated board-power acquisition. No blocking utilization/memory queries.

Power is the same real NVML mW reading with actual API-return timestamps,
50ms requested cadence and unchanged 250ms call/support limits. Utilization
and memory are explicitly missing, not fabricated or used for energy. The
original separate MPS/process resource guard remains unchanged.
"""
import math
from pathlib import Path
import time
import nvml_direct_observer_20261029 as direct
import rental_nvml_observer_20261029 as rental

GPU=rental.GPU
VERSION='rental400_actual_NVML_board_power_only_250ms_v1'

def acquire(nvml,handle,phase,clock=time.monotonic,unix=time.time):
    start=clock();mw=nvml.nvmlDeviceGetPowerUsage(handle);receipt=clock();stamp=unix()
    watts=float(mw)/1000.
    if not 0<watts<=1000 or not math.isfinite(watts):raise ValueError('Invalid actual NVML power')
    if not start<=receipt or receipt-start>.25:raise ValueError('Power API exceeds original250ms bound')
    sample=direct.Sample(stamp,receipt,phase,watts,None,None)
    return sample,dict(query_start_monotonic_s=start,power_API_return_monotonic_s=receipt,
        query_end_monotonic_s=receipt,power_milliwatts=mw,phase=phase,valid_numeric_row=True,
        utilization_percent=None,memory_used_mib=None,not_queried=['utilization','memory'],
        timestamp_source='actual power API return; no backfill or interpolation')

ORIGINAL_SAMPLER=direct.Sampler
class PowerSampler(ORIGINAL_SAMPLER):
    def start(self):
        super().start()
        self.identity.update(version=VERSION,query_functions=['nvmlDeviceGetPowerUsage'],
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
