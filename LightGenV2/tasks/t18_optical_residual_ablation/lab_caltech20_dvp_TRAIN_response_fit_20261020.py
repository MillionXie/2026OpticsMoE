"""TRAIN-only observation surrogate. No deployed model or hardware mutation.

Four shared nuisance parameters (phase scale/gamma, blur sigma, relative DC)
are fitted without labels. Odd-index TRAIN probes are held out from derivatives.
This is an empirical observation model, NOT a calibrated phase LUT or unique
physical identification. Its later use is restricted to simulation training.
"""
import argparse
import os
from pathlib import Path
import time


def response_parameters(raw):
    import torch
    lo=raw.new_tensor([.25,.4,0.,0.]);hi=raw.new_tensor([2.,2.5,12.,3.])
    return lo+(hi-lo)*torch.sigmoid(raw)


def response_intensity(amplitude,turns,propagator,parameters):
    import torch
    from torch.nn import functional as F
    scale,gamma,sigma,background=parameters
    phase=turns.clamp_min(1e-8).pow(gamma)*scale
    field=amplitude.to(torch.complex64)*torch.exp(2j*torch.pi*phase)
    propagated=propagator(F.pad(field,(20,20,20,20))).abs().square()
    n=propagated.shape[-1]
    freq=torch.fft.fftfreq(n,device=amplitude.device,dtype=amplitude.dtype)
    kernel=torch.exp(-2*torch.pi**2*sigma.square()*(freq[:,None].square()+freq[None,:].square()))
    blurred=torch.fft.ifft2(torch.fft.fft2(propagated)*kernel).real.clamp_min(0)
    cropped=blurred[:,20:498,20:498]
    return cropped+background*cropped.mean((-2,-1),keepdim=True)


def normalized_loss(predicted,observed):
    from torch.nn import functional as F
    def norm(x):return x/x.square().mean((-2,-1),keepdim=True).sqrt().clamp_min(1e-8)
    p,o=norm(predicted),norm(observed)
    return .5*(p-o).square().mean()+.5*(F.adaptive_avg_pool2d(p[:,None],(16,16))-F.adaptive_avg_pool2d(o[:,None],(16,16))).square().mean()


def selftest():
    import torch
    from torch.nn import functional as F
    torch.set_num_threads(2);torch.manual_seed(73)
    amplitude=torch.rand(2,478,478);turns=torch.rand(478,478)
    # A fixed nontrivial Fourier propagator; no model or SDK dependency.
    freq=torch.fft.fftfreq(518)
    kernel=torch.exp(3j*(freq[:,None].square()+freq[None,:].square()))
    propagator=lambda x:torch.fft.ifft2(torch.fft.fft2(x)*kernel)
    expected=propagator(F.pad(amplitude.to(torch.complex64)*torch.exp(2j*torch.pi*turns),(20,20,20,20))).abs().square()[:,20:498,20:498]
    got=response_intensity(amplitude,turns,propagator,torch.tensor([1.,1.,0.,0.]))
    assert torch.allclose(got,expected,atol=2e-6,rtol=2e-6)
    raw=torch.tensor([0.,0.,-1.,-2.],requires_grad=True)
    value=normalized_loss(response_intensity(amplitude,turns,propagator,response_parameters(raw)),expected)
    value.backward();assert torch.isfinite(raw.grad).all() and raw.grad.abs().min()>0
    direction=torch.tensor([.3,-.2,.1,-.4]);h=.003
    def obj(x):return normalized_loss(response_intensity(amplitude,turns,propagator,response_parameters(x)),expected)
    finite=(obj(raw.detach()+h*direction)-obj(raw.detach()-h*direction))/(2*h)
    exact=raw.grad@direction
    assert torch.allclose(finite,exact,atol=2e-4,rtol=.03),(finite,exact)
    print(dict(passed=True,nominal_maxabs=float((got-expected).abs().max()),derivative=float(exact),finite_difference=float(finite),SDK_used=False,model_modified=False))


def run(a):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    import cv2
    import numpy as np
    import torch
    from PIL import Image
    import lab_caltech20_dvp_layerwise_20261019 as m
    from lab_caltech20_dvp_TRAIN_response_20261020 import CASES,phase_patterns,checker_phase,HARDWARE
    torch.set_num_threads(2);torch.manual_seed(73)
    assert not a.out.exists() and not m.LOCK.exists()
    report=m.read(a.probes/'report.json');assert report['state']=='complete' and report['response_frames']==216 and report['SDK_released'] and not report['TEST_used']
    assert m.session_hardware_digest(a.sequence)==HARDWARE
    assert report['protected']=={p:m.sha(p) for p in report['protected']}
    a.out.mkdir();start=time.monotonic();stability=[];pack=[];transfer=None
    m.save(a.out/'protocol.json',dict(source_git=m.git_head(),source_SHA256=m.sha(__file__),probe_report_SHA256=m.sha(a.probes/'report.json'),
        hardware_SHA256=HARDWARE,TRAIN_unique=40,TEST_used=False,VALID=0,fit_steps=160,batch=4,seed=73,Adam_lr=.04,CPU_threads=2,
        parameters=['phase_scale','phase_gamma','sigma_pixels','relative_background'],bounds=[[.25,2.],[.4,2.5],[0.,12.],[0.,3.]],
        split='even TRAIN index derivatives, odd TRAIN index response holdout; no dataset VALID or labels',
        response_admission='held TRAIN normalized field loss <=95% nominal; TRAIN also improves; repeats stable first',
        scope='Training observation surrogate only; no PT, SDK or deployed inference/hardware/LUT change'))
    bykey={(r['depth'],r['condition'],r['index']):r for r in report['records']}
    def image(row):
        assert m.sha(row['crop'])==row['crop_SHA256']
        return np.asarray(Image.open(row['crop'])).astype(np.float32)
    def coarse(x):return cv2.resize(x,(16,16),interpolation=cv2.INTER_AREA)
    for depth,stage,parent,digest in CASES:
        session=a.sequence/f'L{depth}_strict8pp_20261020';capture=session/'capture'/stage
        before=[];oldrows=[];effects={tag:[] for tag in ('flat','half_code','global_offset128')}
        old=np.load(capture/'train/ccd.npy',mmap_mode='r')
        assert m.sha(capture/'train/ccd.npy')==m.read(capture/'report.json')['arrays']['train']['SHA256']
        model,_=m.model_for(session/'fits'/parent/'best.pt',digest,depth,session/'source_reference.ts')
        current=model.propagator.transfer.detach().cpu()
        if transfer is not None:assert torch.equal(current,transfer)
        transfer=current.clone()
        plane=model.plane(stage);phase=torch.floor(torch.remainder(torch.angle(plane),2*np.pi)/(2*np.pi)*256).clamp(0,255)[20:498,20:498].detach().numpy().astype(np.uint8)
        patterns=dict(phase_patterns(phase));patterns.update({f'checker_{v}':checker_phase(v) for v in (0,64,128,192)})
        for i in range(20):
            b,after=image(bykey[depth,'nominal_before',i]),image(bykey[depth,'nominal_after',i])
            before.append(dict(index=i,PCC=m.correlation(b,after),coarse_PCC=m.correlation(coarse(b),coarse(after)),mean_ratio=float(after.mean()/max(b.mean(),1e-8))))
            oldrows.append(dict(index=i,PCC=m.correlation(b,old[i]),coarse_PCC=m.correlation(coarse(b),coarse(old[i])),mean_ratio=float(b.mean()/max(float(old[i].mean()),1e-8))))
            for tag in effects:effects[tag].append(m.correlation(b,image(bykey[depth,tag,i])))
            amp=np.asarray(Image.open(capture/'train'/f'{i:06d}_input.png')).astype(np.float32)/255.
            for tag in ('nominal_before','flat','half_code','global_offset128')+tuple(f'checker_{v}' for v in (0,64,128,192) if i<2):
                row=bykey[depth,tag,i]
                pack.append(dict(depth=depth,index=i,tag=tag,amplitude=amp,turns=patterns[tag].astype(np.float32)/256.,observed=image(row)))
        stable=all(r['coarse_PCC']>=.95 and .8<=r['mean_ratio']<=1.2 for r in before)
        stability.append(dict(depth=depth,within_connection_repeat_passed=stable,repeat=before,previous_capture_comparison=oldrows,
            phase_effect_raw_PCC_median={k:float(np.median(v)) for k,v in effects.items()},
            previous_capture_coarse_PCC_median=float(np.median([r['coarse_PCC'] for r in oldrows]))))
    m.save(a.out/'stability.json',dict(passed=all(r['within_connection_repeat_passed'] for r in stability),depths=stability,TEST_used=False))
    assert all(r['within_connection_repeat_passed'] for r in stability),'TRAIN repeat unstable: no observation fitting or retraining'
    # Deterministic split within TRAIN diagnostics; not the dataset VALID role.
    training=[v for v in pack if v['index']%2==0];held=[v for v in pack if v['index']%2==1]
    assert len(training)==len(held)==88
    def tensors(rows):return tuple(torch.from_numpy(np.stack([r[k] for r in rows])) for k in ('amplitude','turns','observed'))
    tr=tensors(training);hold=tensors(held);nominal=torch.tensor([1.,1.,0.,0.]);initial=torch.tensor([1.,1.,2.,.1])
    lo=torch.tensor([.25,.4,0.,0.]);hi=torch.tensor([2.,2.5,12.,3.]);raw=torch.nn.Parameter(torch.logit((initial-lo)/(hi-lo)))
    opt=torch.optim.Adam([raw],lr=.04);history=[]
    def evaluate(values,parameters):
        total=0.
        with torch.no_grad():
            for i in range(0,len(values[0]),4):
                aa,pp,oo=(v[i:i+4] for v in values)
                total+=float(normalized_loss(response_intensity(aa,pp,model.propagator,parameters),oo))*len(aa)
        return total/len(values[0])
    baseline=dict(TRAIN=evaluate(tr,nominal),held_TRAIN=evaluate(hold,nominal))
    for step in range(1,161):
        ix=torch.randperm(88)[:4];aa,pp,oo=(v[ix] for v in tr);opt.zero_grad(set_to_none=True)
        parameters=response_parameters(raw);loss=normalized_loss(response_intensity(aa,pp,model.propagator,parameters),oo)
        assert torch.isfinite(loss);loss.backward();assert raw.grad is not None and torch.isfinite(raw.grad).all()
        opt.step()
        if step%20==0:
            history.append(dict(step=step,TRAIN_batch_loss=float(loss.detach()),parameters=response_parameters(raw).detach().tolist(),elapsed_s=time.monotonic()-start))
            m.save(a.out/'progress.json',history[-1]);m.save(a.out/'history.json',history)
    fitted=response_parameters(raw).detach();final=dict(TRAIN=evaluate(tr,fitted),held_TRAIN=evaluate(hold,fitted))
    # Admission depends only on held-out TRAIN response, never a classification score.
    admitted=final['TRAIN']<baseline['TRAIN'] and final['held_TRAIN']<=baseline['held_TRAIN']*.95
    q=dict(state='complete',response_fit_admitted=admitted,baseline=baseline,fitted=final,parameters=dict(zip(('phase_scale','phase_gamma','sigma_pixels','relative_background'),fitted.tolist())),
        fit_steps=160,TRAIN_probe_fit_pairs=88,held_TRAIN_probe_pairs=88,TRAIN_unique=40,TEST_used=False,TEST_labels_used=False,VALID=0,
        observation_model_only=True,deployed_architecture_unchanged=True,physical_LUT_calibrated=False,LUT_modified=False,hardware_unchanged=True,
        probe_report_SHA256=m.sha(a.probes/'report.json'),hardware_SHA256=HARDWARE,source_git=m.git_head(),source_SHA256=m.sha(__file__),
        stability_SHA256=m.sha(a.out/'stability.json'),history_SHA256=m.sha(a.out/'history.json'),elapsed_s=time.monotonic()-start,
        interpretation='Empirical TRAIN-only four-parameter observation surrogate; improvement does not identify a unique phase/camera cause or establish classification accuracy. No model checkpoint or physical setting changed.')
    assert report['protected']=={p:m.sha(p) for p in report['protected']}
    m.save(a.out/'report.json',q);print(m.json.dumps(q),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--selftest',action='store_true')
    for key in ('sequence','probes','out'):p.add_argument('--'+key,type=Path)
    a=p.parse_args()
    if a.selftest:selftest()
    else:
        assert all(getattr(a,k) is not None for k in ('sequence','probes','out'))
        try:run(a)
        except BaseException as error:
            if a.out.exists():
                import json,traceback
                (a.out/'failure.json').write_text(json.dumps(dict(error=repr(error),traceback=traceback.format_exc(),automatic_retry=False),indent=2))
            raise
