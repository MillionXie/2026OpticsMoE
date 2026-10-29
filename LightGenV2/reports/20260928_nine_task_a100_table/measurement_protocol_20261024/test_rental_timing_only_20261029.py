"""Pure CPU timing-scope and explicit energy-N/A regression; no CUDA/model."""
import copy
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_rental_timing_only_20261029 as entry

def calls():
    return [dict(sample_id=str(i),preprocess_start=float(i),model_call_start=i+.1,
                 first_block_start=i+.2,gpu_output=i+.3,call_return=i+.4) for i in range(512)]

def main():
    c=calls();r=entry.timing_cost(c)
    assert r['videos']==512 and r['groups16']==32 and r['frames']==2048
    assert abs(r['continuous_wall_s']-511.4)<1e-9
    assert abs(r['time16_s']-511.4/32)<1e-9
    assert r['energy_qualified'] is False and r['gpu_energy_j'] is None and r['estimated_system_energy_j'] is None
    bad=copy.deepcopy(c);bad[1]['preprocess_start']=.2
    try:entry.timing_cost(bad)
    except ValueError:pass
    else:raise AssertionError('Inter-request overlap must fail')
    entry.configure();cfg=entry.config()
    assert cfg['energy_qualified'] is False and cfg['power_maximum_support_age_s']==.25
    assert entry.base.suite.__globals__['ENTRY']==entry.ENTRY
    assert entry.base.audit_point is entry.audit_point
    assert len(entry.native.plan('rental'))==12
    assert entry.base.initialize_profile is entry.native.initialize
    assert 'torch' not in sys.modules
    print('{"status":"passed","tests":5,"torch_imported":false,"energy_qualified":false}')

if __name__=='__main__':main()
