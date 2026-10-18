"""TRAIN-only finite numerical-boundary diagnostic, never a deployed graph.

Keep actual quantized incident amplitude/phase, physical17um/532nm/10cm and
observed prefixes fixed. Compare canonical518 against zero padding1024/1536.
No camera, TEST score, parameter fitting or automatic numerical-graph adoption.
"""
import argparse,json,math,sys,traceback
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
import lab_caltech20_residual_sequence_20261018 as seq
from lab_caltech20_residual_terminal_cpu_20261019 import field_agreement
from lab_caltech10_20261015 import save
from LightGenV2.tasks.t07_abo_image_retrieval.standalone.optics import Propagation


def transfer(n):
    f=torch.fft.fftfreq(n,d=17e-6,dtype=torch.float64);fy,fx=torch.meshgrid(f,f,indexing='ij');argument=(2*math.pi)**2*((1/532e-9)**2-fx.square()-fy.square());h=torch.exp(1j*(.1*argument.clamp_min(0).sqrt())).to(torch.complex64);return torch.where(argument>=0,h,0)


def padded_intensity(field,h):
    n=h.shape[-1];border=(n-478)//2;assert n>=478 and (n-478)%2==0;out=torch.fft.ifft2(torch.fft.fft2(F.pad(field,(border,)*4))*h);return out.abs().square()[:,border:border+478,border:border+478]


def selftest():
    torch.set_num_threads(2);assert torch.equal(transfer(518),Propagation().transfer);torch.manual_seed(73);field=torch.rand(2,478,478).to(torch.complex64)*torch.exp(1j*torch.rand(2,478,478));p=padded_intensity(field,transfer(518));q=Propagation()(F.pad(field,(20,)*4)).abs().square()[:,20:498,20:498];assert torch.equal(p,q);h=transfer(1024);u=F.pad(field,(273,)*4);v=torch.fft.ifft2(torch.fft.fft2(u)*h);assert torch.allclose(u.abs().square().sum((-2,-1)),v.abs().square().sum((-2,-1)),rtol=1e-5);print(json.dumps(dict(passed=True,canonical518_same_host_exact=True,larger_zero_padding_preserves_full_plane_energy=True,SDK_opened=False,GPU_used=False)))


def run(a):
    torch.set_num_threads(2);assert a.prefix==a.depth+1 and not a.out.exists();_,data,dm,model,original,canonical,fields,identities=seq.dataset_model(a);indices=np.array([np.flatnonzero(data['train'][1]==k)[0] for k in range(20)]);assert len(indices)==20;hs={518:canonical,1024:transfer(1024),1536:transfer(1536)};records={k:[] for k in fields};a.out.mkdir()
    with torch.inference_mode():
        for begin in range(0,20,2):
            ix=indices[begin:begin+2]
            def replay(stage,amp,phase,ids):
                assert stage in fields and np.array_equal(ids,data['train'][2][ix]);field=torch.from_numpy(amp.copy()).to(torch.complex64)*torch.exp(2j*np.pi*(torch.from_numpy(phase).float()/256))[None];observed=torch.from_numpy(np.asarray(fields[stage]['train'][ix]).copy());values={str(n):field_agreement(padded_intensity(field,h),observed) for n,h in hs.items()}
                for j,index in enumerate(ix):records[stage].append(dict(TRAIN_index=int(index),sample_id=str(ids[j]),canvas={n:{k:v[j] for k,v in row.items()} for n,row in values.items()}))
                return observed
            model(seq.encode_images(data['train'][0][ix]),callback=replay,ids=data['train'][2][ix]);save(a.out/'progress.json',dict(state='TRAIN20_numerical_padding_diagnostic',completed=min(begin+2,20),total=20))
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in original.items());summary={stage:{n:{key:float(np.median([v['canvas'][n][key] for v in rows])) for key in ('pixel_Pearson','coarse16_Pearson','unit_RMS_error')} for n in ('518','1024','1536')} for stage,rows in records.items()};save(a.out/'records.json',records);save(a.out/'report.json',dict(state='diagnostic_complete',depth=a.depth,TRAIN=20,TRAIN_each_class=1,TEST_used=False,VALID_used=False,source_SHA256=seq.sha(__file__),parent_PT_SHA256=a.checkpoint_sha,data_SHA256=dm['sha256'],prefix_SHA256=identities,all_parameters_frozen=True,physical_pitch_wavelength_distance_unchanged=True,summary=summary,SDK_opened=False,GPU_used=False,new_CCD=0,NOT_a_performance_result=True,adopted_in_model_or_hardware=False,limitation='Finite TRAIN-only numerical-boundary hypotheses; correlations alone do not establish physical cause, calibration or accuracy improvement. No parameter or propagator is modified in the existing model.'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--selftest',action='store_true');p.add_argument('--depth',type=int);p.add_argument('--prefix',type=int);p.add_argument('--checkpoint-sha')
    for n in ('session','profile','checkpoint','out'):p.add_argument('--'+n,type=Path)
    a=p.parse_args()
    try:selftest() if a.selftest else run(a)
    except BaseException as e:
        if a.out:save(a.out.with_suffix('.failure.json'),dict(error=repr(e),traceback=traceback.format_exc(),automatic_retry=False))
        raise
