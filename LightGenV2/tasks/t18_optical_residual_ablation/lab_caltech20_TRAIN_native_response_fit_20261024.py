"""TRAIN-only native-activation response fit; no inference/hardware changes.

One shared Gaussian observation width plus the existing 8x8 photometric map.
Fit only original TRAIN response pairs, validate on other TRAIN anchors/phases.
"""
import argparse,json,os,time
from pathlib import Path

RECIPE=dict(epochs=10,batch=4,steps=200,seconds=300,lr=.02,seed=73,CPU_threads=2,initial_sigma_pixels=.5)


def contrast_loss(x,y):
    x=x-x.mean((-2,-1),keepdim=True);y=y-y.mean((-2,-1),keepdim=True)
    x=x/x.square().mean((-2,-1),keepdim=True).sqrt().clamp_min(1e-6)
    y=y/y.square().mean((-2,-1),keepdim=True).sqrt().clamp_min(1e-6)
    return (x-y).square().mean()/2


def response(full_intensity,gain,background,sigma):
    import torch
    from lab_caltech20_TRAIN_flat_response_20261024 import apply_map
    n=full_intensity.shape[-1];f=torch.fft.fftfreq(n,dtype=full_intensity.dtype,device=full_intensity.device)
    kernel=torch.exp(-2*torch.pi**2*sigma.square()*(f[:,None].square()+f[None,:].square()))
    blurred=torch.fft.ifft2(torch.fft.fft2(full_intensity)*kernel).real.clamp_min(0)[:,20:498,20:498]
    return apply_map(blurred,gain,background)


def objective(predicted,observed,model):
    from torch.nn import functional as F
    native=contrast_loss(model.nonlinear(predicted,'global'),model.nonlinear(observed,'global'))
    pixel=contrast_loss(predicted,observed)
    coarse=contrast_loss(F.adaptive_avg_pool2d(predicted[:,None],(16,16)),F.adaptive_avg_pool2d(observed[:,None],(16,16)))
    return native+.25*pixel+.25*coarse


def selftest():
    import torch
    from torch.nn import functional as F
    torch.set_num_threads(2);torch.manual_seed(73)
    x=torch.rand(2,518,518);g=torch.ones(8,8,requires_grad=True);b=torch.ones(8,8,requires_grad=True)*.4;s=torch.tensor(.5,requires_grad=True)
    y=response(x,g,b,s);assert y.shape==(2,478,478) and torch.isfinite(y).all()
    zero=response(x,g.detach(),torch.zeros(8,8),torch.tensor(0.));crop=x[:,20:498,20:498];assert torch.allclose(zero,crop/crop.mean((-2,-1),keepdim=True),atol=3e-6,rtol=3e-6)
    weight=torch.linspace(0,1,478)[None,None,:];loss=(y*weight).mean();gradient=torch.autograd.grad(loss,(g,b,s));assert all(torch.isfinite(v).all() and float(v.norm())>0 for v in gradient)
    h=.01;fn=lambda z:(response(x,g.detach(),b.detach(),z)*weight).mean()
    finite=(fn(s.detach()+h)-fn(s.detach()-h))/(2*h);assert torch.allclose(finite,gradient[2],atol=2e-5,rtol=.05)
    assert abs(float(contrast_loss(crop,crop)))<1e-10
    print(json.dumps(dict(passed=True,nominal_identity=True,finite_response_gradients=True,sigma_gradient_finite_difference=True,TEST_used=False,SDK_used=False,GPU_used=False)))


def run(a):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    import numpy as np,torch
    from PIL import Image
    from torch.nn import functional as F
    import lab_caltech20_TRAIN_photometric_mask_20261024 as mask
    import lab_caltech20_tucam_TRAIN_response_20261024 as source
    from lab_caltech20_dvp_TRAIN_response_20261020 import phase_patterns
    base=mask.base;_,_,m,_=base.numerical_modules();p=m.read(a.profile);mask.validate_policy(p)
    assert m.sha(mask.__file__)=='dfb6ab85b3344301fd88f83fe3fbd4e774c969b5ac8115172e9bffbe69ae3607'
    assert m.sha(source.__file__)=='421a18ea3c8aecae7c07306146efb6a56a78d05baadfac3a51c6108bea718448'
    assert m.sha(base.HERE/'lab_caltech20_TRAIN_flat_response_20261024.py')==p['flat_engine_SHA256']
    diagnostic=base.ROOT/p['response_fit'];diagnostic=diagnostic.parent/'task_feature_audit_nativebatch'
    assert m.sha(diagnostic/'report.json')=='ea99480abe62129d541c142b6b79f7712148f9bcd311d30efcd323d17e99d6c5' and m.read(diagnostic/'independent_audit.json')['all40_input_canonical_exact']
    assert m.read(diagnostic.with_suffix('.return.json'))['returncode']==0
    assert not a.out.exists() and not m.LOCK.exists();a.out.mkdir();torch.set_num_threads(2);torch.manual_seed(73);start=time.monotonic()
    fp=base.HERE/'lab_caltech20_TRAIN_flat_response_profile_20261024.json';assert m.sha(fp)=='3c3e85db21beda6fb82b5f9abed1195020de7e111048eacec1fa437bdcf89a51'
    flat_policy=m.read(fp);response_policy=m.read(base.HERE/flat_policy['response_profile']);source.validate_policy(response_policy)
    model,phase,hw,protected,anchors=source.context(response_policy)
    pr=base.ROOT/p['response_fit'];pr=pr.parent/'probes';probe=m.read(pr/'report.json');assert m.sha(pr/'report.json')=='ed5f98770ce72fc4297d92f7125db0c41508af7bac22405b5dac0012e8973639'
    assert not probe['TEST_used'] and len(anchors)==40
    patterns=dict(phase_patterns(phase));lookup={(v['condition'],v['index']):v for v in probe['records']};pack=[]
    for anchor in anchors:
        amp=torch.from_numpy(np.asarray(Image.open(anchor['canonical'])).astype(np.float32)/255)
        for tag in ('flat','nominal_before','half_code','global_offset128'):
            row=lookup[tag,anchor['index']];assert row['sample_id']==anchor['sample_id'] and row['canonical_SHA256']==anchor['canonical_SHA256'] and row['split']==anchor['split']
            assert m.sha(row['crop'])==row['crop_SHA256'] and m.sha(row['phase'])==row['phase_SHA256']
            with torch.no_grad():
                turns=torch.from_numpy(patterns[tag].astype(np.float32)/256);field=amp.to(torch.complex64)*torch.exp(2j*torch.pi*turns)
                nominal=model.propagator(F.pad(field[None],(20,20,20,20))).abs().square()[0]
            observed=torch.from_numpy(np.asarray(Image.open(row['crop'])).astype(np.float32));observed=observed/observed.mean().clamp_min(1e-8)
            pack.append(dict(index=anchor['index'],sample_id=anchor['sample_id'],split=anchor['split'],condition=tag,nominal=nominal,observed=observed))
        m.save(a.out/'progress.json',dict(state='TRAIN_response_pair_cache',TRAIN_anchors=len(pack)//4,pairs=len(pack),TEST_used=False))
    assert len(pack)==160
    fitrows=[v for v in pack if v['split']=='fit'];held=[v for v in pack if v['split']=='held_TRAIN'];assert len(fitrows)==len(held)==80
    path=base.ROOT/p['response_fit']/'TRAIN_flat_gain_background.npz';assert m.sha(path)==p['response_map_SHA256']
    with np.load(path,allow_pickle=False) as z:g0=torch.from_numpy(z['gain'].copy());b0=torch.from_numpy(z['background'].copy())
    raw_g=torch.nn.Parameter(torch.logit(((g0-.05)/3.95).clamp(1e-6,1-1e-6)));raw_b=torch.nn.Parameter(torch.logit((b0/4).clamp(1e-6,1-1e-6)));raw_s=torch.nn.Parameter(torch.logit(torch.tensor(RECIPE['initial_sigma_pixels']/12)))
    def values():return .05+3.95*raw_g.sigmoid(),4*raw_b.sigmoid(),12*raw_s.sigmoid()
    def assess(rows,g,b,s):
        losses=[];pixel=[];coarse=[];native=[]
        with torch.no_grad():
            for begin in range(0,len(rows),4):
                chunk=rows[begin:begin+4];x=torch.stack([v['nominal'] for v in chunk]);y=torch.stack([v['observed'] for v in chunk]);pred=response(x,g,b,s)
                losses.append(float(objective(pred,y,model))*len(chunk));pixel.append(float(1-contrast_loss(pred,y))*len(chunk));native.append(float(1-contrast_loss(model.nonlinear(pred,'global'),model.nonlinear(y,'global')))*len(chunk));coarse.append(float(1-contrast_loss(F.adaptive_avg_pool2d(pred[:,None],(16,16)),F.adaptive_avg_pool2d(y[:,None],(16,16))))*len(chunk))
        return dict(loss=sum(losses)/len(rows),full478_PCC=sum(pixel)/len(rows),coarse16_PCC=sum(coarse)/len(rows),native_optical_PCC=sum(native)/len(rows),pairs=len(rows))
    groups={'fit':fitrows,'held':held};groups.update({'held_'+tag:[v for v in held if v['condition']==tag] for tag in patterns if tag!='nominal_after'})
    baseline={k:assess(v,g0,b0,torch.tensor(0.)) for k,v in groups.items()};protocol=dict(source_git=m.git_head(),source_SHA256=m.sha(__file__),profile_SHA256=m.sha(a.profile),recipe=RECIPE,
        TRAIN_fit_anchors=20,TRAIN_held_anchors=20,fit_pairs=80,held_pairs=80,TEST_used=False,classification_labels_used=False,new_CCD=0,SDK_used=False,GPU_used=False,
        parameters=129,initial_map_SHA256=p['response_map_SHA256'],diagnostic_report_SHA256=m.sha(diagnostic/'report.json'),probe_report_SHA256=m.sha(pr/'report.json'),
        loss='native optical activation contrast + .25 full pixel contrast + .25 coarse16 contrast',training_forward_only=True,no_warp=True,hardware_changed=False,model_checkpoint_written=False)
    m.save(a.out/'protocol.json',protocol);m.save(a.out/'baseline.json',baseline);rng=np.random.default_rng(73);opt=torch.optim.Adam([raw_g,raw_b,raw_s],lr=.02);steps=0;fitstart=time.monotonic();history=[]
    for epoch in range(1,11):
        order=rng.permutation(80);total=0.
        for begin in range(0,80,4):
            chunk=[fitrows[i] for i in order[begin:begin+4]];x=torch.stack([v['nominal'] for v in chunk]);y=torch.stack([v['observed'] for v in chunk]);opt.zero_grad(set_to_none=True)
            pred=response(x,*values());loss=objective(pred,y,model);assert torch.isfinite(loss);loss.backward();assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in (raw_g,raw_b,raw_s));opt.step();steps+=1;total+=float(loss.detach())*4
        row=dict(epoch=epoch,TRAIN_pairs=80,TRAIN_loss=total/80,updates=steps,sigma_pixels=float(values()[2].detach()),elapsed_s=time.monotonic()-fitstart);history.append(row);m.save(a.out/'history.json',history);m.save(a.out/'progress.json',row);print(json.dumps(row),flush=True)
        if time.monotonic()-fitstart>=300:break
    g,b,s=[v.detach() for v in values()];final={k:assess(v,g,b,s) for k,v in groups.items()}
    admitted=final['fit']['loss']<baseline['fit']['loss'] and final['held']['loss']<=.95*baseline['held']['loss']
    for tag in ('held','held_nominal_before'):
        admitted=admitted and final[tag]['native_optical_PCC']>=baseline[tag]['native_optical_PCC']+.03 and final[tag]['full478_PCC']>=baseline[tag]['full478_PCC'] and final[tag]['coarse16_PCC']>=baseline[tag]['coarse16_PCC']-.01
    admitted=admitted and all(final['held_'+tag]['native_optical_PCC']>=baseline['held_'+tag]['native_optical_PCC'] for tag in ('flat','half_code','global_offset128'))
    np.savez_compressed(a.out/'TRAIN_native_response.npz',gain=g.numpy(),background=b.numpy(),sigma_pixels=s.numpy(),fit_TRAIN_ids=np.array([v['sample_id'] for v in anchors if v['split']=='fit']))
    assert protected=={path:m.sha(path) for path in protected} and not m.LOCK.exists()
    report=dict(state='complete',native_response_admitted=bool(admitted),baseline=baseline,fitted=final,actual_epochs=epoch,actual_updates=steps,TRAIN_unique=40,fit_pairs=80,held_pairs=80,
        TEST_used=False,classification_labels_used=False,SDK_used=False,GPU_used=False,new_CCD=0,training_forward_only=True,new_inference_layers=0,hardware_changed=False,model_checkpoint_written=False,
        sigma_pixels=float(s),gain_range=[float(g.min()),float(g.max())],background_range=[float(b.min()),float(b.max())],protocol_SHA256=m.sha(a.out/'protocol.json'),map_SHA256=m.sha(a.out/'TRAIN_native_response.npz'),elapsed_s=time.monotonic()-start,classification_accuracy_updated=False)
    m.save(a.out/'report.json',report);m.save(a.out/'independent_audit.json',dict(passed=True,report_SHA256=m.sha(a.out/'report.json'),TRAIN_only=True,hardware_and_source_PT_frozen=True));print(json.dumps(report),flush=True)


if __name__=='__main__':
    q=argparse.ArgumentParser();q.add_argument('--profile',type=Path,required=True);q.add_argument('--out',type=Path,required=True);run(q.parse_args())
