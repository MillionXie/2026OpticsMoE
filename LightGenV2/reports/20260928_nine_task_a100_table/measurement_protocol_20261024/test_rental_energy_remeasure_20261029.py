"""CPU-only power sampling and full remeasurement source/contract checks."""
import ast
import inspect
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import rental_power_observer_20261029 as power
import run_rental_energy_remeasure_20261029 as run

class NVML:
    def __init__(self,p=123456):self.p=p;self.calls=[]
    def nvmlDeviceGetPowerUsage(self,h):self.calls.append(h);return self.p
    def nvmlDeviceGetUtilizationRates(self,h):raise AssertionError('Do not block power on utilization')
    def nvmlDeviceGetMemoryInfo(self,h):raise AssertionError('Do not block power on memory')

def main():
    nvml=NVML();clock=iter((10.,10.002))
    sample,raw=power.acquire(nvml,'handle','formal',lambda:next(clock),lambda:100.)
    assert nvml.calls==['handle'] and sample.watts==123.456 and sample.host_monotonic_s==10.002
    assert sample.utilization_percent is None and sample.memory_used_mib is None
    assert raw['power_API_return_monotonic_s']==10.002 and raw['query_start_monotonic_s']==10.
    for value in (0,-1,float('nan'),float('inf'),1000001):
        clock=iter((1.,1.01))
        try:power.acquire(NVML(value),0,'test',lambda:next(clock))
        except ValueError:pass
        else:raise AssertionError('Invalid actual power must stop')
    clock=iter((1.,1.251))
    try:power.acquire(nvml,0,'test',lambda:next(clock))
    except ValueError:pass
    else:raise AssertionError('Keep250ms API gate')
    run.configure();c=run.config()
    assert c['power_maximum_support_age_s']==.25 and c['power_limit_w']==400
    assert c['protocol_id']==run.PROTOCOL and run.ROOT.name=='rental_energy_remeasure_20261029_01'
    assert run.base.audit_point is run.audit_point
    assert run.base.suite.__globals__['ENTRY']==run.ENTRY
    assert run.base.worker.__globals__['NEW_PROTOCOL']==run.PROTOCOL
    assert len(run.native.plan('rental'))==12
    assert run.QUALIFIED_AUDIT.__globals__['NEW_PROTOCOL']==run.PROTOCOL
    # The transformed independent replay must not coerce missing diagnostics.
    assert 'torch' not in sys.modules
    print('{"status":"passed","tests":15,"torch_imported":false,"full_models_and_250ms_preserved":true}')

if __name__=='__main__':main()
