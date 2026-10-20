"""Read-only TRAIN probes of existing DVP captures, not an optical correction."""
import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys
import time
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image

HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(HERE.parents[2])]
import lab_caltech20_dvp_layerwise_20261019 as core
from lab_caltech20_residual_terminal_cpu_20261019 import phase_response_field,field_agreement


def selftest():
    torch.set_num_threads(2);torch.manual_seed(73)
    amplitude=torch.rand(2,12,12);phase=torch.randint(0,256,(12,12),dtype=torch.uint8)
    original=amplitude.to(torch.complex64)*torch.exp(2j*np.pi*(phase.float()/256))[None]
    assert torch.equal(original,phase_response_field(amplitude,phase,1.,1.))
    result=field_agreement(amplitude,amplitude)
    assert max(abs(v-1.) for v in result['pixel_Pearson'])<1e-6
    assert max(result['unit_RMS_error'])==0
    for scale,gamma in ((1.,1.),(.8,1.),(1.2,1.),(1.,.8),(1.,1.2)):
        assert torch.isfinite(phase_response_field(amplitude,phase,scale,gamma)).all()
    print(core.json.dumps(dict(passed=True,nominal_field_byte_identical=True,read_only_TRAIN_probe=True,SDK_used=False,GPU_used=False)))


def diagnose(a):
    torch.set_num_threads(2)
    import psutil
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    cfg=core.read(a.profile);assert core.sha(core.__file__)==cfg['core_source_SHA256']
    assert core.sha(HERE/'lab_caltech20_residual_terminal_cpu_20261019.py')==cfg['physics_source_SHA256']
    assert cfg['TRAIN_records']==40 and cfg['TRAIN_each_class']==2 and cfg['batch']==8
    curves=cfg['counterfactuals']
    assert [(v['scale'],v['gamma']) for v in curves]==[(1.,1.),(.8,1.),(1.2,1.),(1.,.8),(1.,1.2)]
    assert len({v['name'] for v in curves})==5 and not core.LOCK.exists()
    session=a.sequence/f'L{a.depth}';parent=a.checkpoint.parent
    assert parent.resolve().is_relative_to((session/'fits').resolve()) and a.checkpoint.name=='best.pt'
    report=core.read(parent/'report.json');protocol=core.read(parent/'protocol.json');audit=core.read(parent/'independent_audit.json')
    assert report['state']=='complete' and audit['passed'] and protocol['prefix']==a.depth+1
    assert audit['report_SHA256']==core.sha(parent/'report.json') and audit['protocol_SHA256']==core.sha(parent/'protocol.json')
    assert core.sha(a.checkpoint)==a.digest==report['results']['best']['PT_SHA256']
    assert a.out.resolve().is_relative_to((session/'reviews').resolve()) and not a.out.exists()
    data,dm=core.data_for(a.sequence)
    assert np.bincount(data['train'][1][:40],minlength=20).tolist()==[2]*20
    fields,identities=core.fields_for(session,a.depth+1,data,dm,core.session_hardware_digest(a.sequence))
    assert identities==protocol['prefix_SHA256'] and dm['sha256']==protocol['data_SHA256']
    model,_=core.model_for(a.checkpoint,a.digest,a.depth,session/'source_reference.ts');original=core.state_cpu(model)
    a.out.mkdir();start=time.monotonic();rows=[];summaries=[];stage_order=['router']+[str(i) for i in range(a.depth)]
    for position,stage in enumerate(stage_order):
        folder=session/'capture'/stage;capture=core.read(folder/'report.json');assert core.read(folder/'independent_capture_audit.json')['passed']
        phase_checked=False;current=[]
        for begin in range(0,40,8):
            x,n,end=core.padded(data,'train',begin)
            def inspect(received,amplitude,phase,ids):
                nonlocal phase_checked
                assert received==stage and np.array_equal(ids,data['train'][2][begin:end])
                if not phase_checked:
                    assert core.sha(folder/'phase.bmp')==capture['phase_SHA256']
                    with Image.open(folder/'phase.bmp') as im:assert np.array_equal(np.asarray(im),core.phase_native(phase))
                    phase_checked=True
                for offset in range(n):
                    index=begin+offset;base=folder/'train'/f'{index:06d}';receipt=core.read(base.with_suffix('.json'))
                    assert receipt['sample_id']==str(ids[offset]) and receipt['canonical_SHA256']==core.sha(base.with_name(base.name+'_input.png'))
                    assert receipt['phase_SHA256']==capture['phase_SHA256'] and receipt['data_SHA256']==dm['sha256']
                    with Image.open(base.with_name(base.name+'_input.png')) as im:
                        assert np.array_equal(np.asarray(im),np.rint(amplitude[offset].clip(0,1)*255).astype(np.uint8))
                observed=torch.from_numpy(np.asarray(fields[stage]['train'][begin:end]).copy()).float()
                output=[]
                with torch.inference_mode():
                    for curve in curves:
                        field=phase_response_field(torch.from_numpy(amplitude.copy()),torch.from_numpy(phase.copy()),curve['scale'],curve['gamma'])
                        predicted=model.propagator(F.pad(field,(20,20,20,20))).abs().square()[:,20:498,20:498]
                        output.append((curve['name'],field_agreement(predicted[:n],observed)))
                for offset in range(n):
                    current.append(dict(index=begin+offset,canonical_input_exact=True,phase_BMP_exact=True,
                        metrics={name:{key:values[offset] for key,values in result.items()} for name,result in output}))
                return np.zeros((n,478,478),np.uint8)
            prefix={key:np.asarray(fields[key]['train'][begin:end]).copy() for key in stage_order[:position]}
            core.layer_forward(model,x,data['train'][2][begin:end],stage,prefix,inspect)
        assert phase_checked and len(current)==40
        summary={curve['name']:{key:float(np.median([r['metrics'][curve['name']][key] for r in current]))
            for key in ('pixel_Pearson','coarse16_Pearson','unit_RMS_error')} for curve in curves}
        item=dict(stage=stage,TRAIN=40,TRAIN_each_class=2,phase_SHA256=capture['phase_SHA256'],
                  capture_report_SHA256=core.sha(folder/'report.json'),summary=summary)
        summaries.append(item);rows.append(dict(stage=stage,records=current))
        core.save(a.out/'progress.json',dict(completed_physical_stages=position+1,total_physical_stages=len(stage_order),**item))
        print(core.json.dumps(item),flush=True)
    assert all(torch.equal(value,core.state_cpu(model)[key]) for key,value in original.items()) and core.sha(a.checkpoint)==a.digest
    q=dict(passed=True,depth=a.depth,TRAIN_records=40,TRAIN_each_class=2,TEST_inference=False,VALID=0,
        original_PT_unchanged=True,PT_SHA256=a.digest,data_SHA256=dm['sha256'],prefix_SHA256=identities,
        source_git=core.git_head(),source_SHA256=core.sha(__file__),profile_SHA256=core.sha(a.profile),
        all_canonical_inputs_and_phase_BMPs_exact=True,counterfactuals=curves,stages=summaries,records=rows,
        elapsed_s=time.monotonic()-start,SDK_used=False,GPU_used=False,new_CCD=0,new_PT=0,
        scope=cfg['scope'])
    core.save(a.out/'report.json',q)
    print(core.json.dumps(dict(passed=True,depth=a.depth,stages=len(summaries),TEST_inference=False,new_CCD=0,report_SHA256=core.sha(a.out/'report.json'))))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--selftest',action='store_true')
    for name in ('sequence','checkpoint','profile','out'):parser.add_argument('--'+name,type=Path)
    parser.add_argument('--depth',type=int,choices=(4,6));parser.add_argument('--digest');a=parser.parse_args()
    try:selftest() if a.selftest else diagnose(a)
    except BaseException as error:
        if a.out and a.out.exists():core.save(a.out/'failure.json',dict(error=repr(error),automatic_retry=False))
        raise
