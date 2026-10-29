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
    assert entry.ROOT.name=='rental_timing_only_20261029_02'
    assert entry.PIPE.endswith('_02')
    rows=[dict(sample_id='one')]
    stored=dict(records=[dict(sample_id='one',sha256='abc',bytes=3,original_path='/original')])
    actual={'one':dict(sha256='abc',size_bytes=3,path='/original')}
    entry.validate_storage_input_hashes(stored,rows,actual)
    for field,value in [('sha256','wrong'),('size_bytes',4),('path','/other')]:
        bad=copy.deepcopy(actual);bad['one'][field]=value
        try:entry.validate_storage_input_hashes(stored,rows,bad)
        except ValueError:pass
        else:raise AssertionError('Byte/path/size gate must reject')
    try:entry.storage_receipt('{}')
    except ValueError:pass
    else:raise AssertionError('Unbound storage receipt must reject')
    assert 'torch' not in sys.modules
    print('{"status":"passed","tests":10,"torch_imported":false,"energy_qualified":false}')

if __name__=='__main__':main()
