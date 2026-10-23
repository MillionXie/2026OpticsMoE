"""One TRAIN-only phase-response fit with admitted photometry/blur frozen.

An empirical TRAIN forward model, NOT a hardware LUT or inference correction.
"""
import argparse,json,os,time
from pathlib import Path

RECIPE=dict(epochs=20,steps=400,batch=4,seconds=300,lr=.02,seed=73,CPU_threads=2,phase_pupil_parameters=25,phase_regularization=.0001)


def prediction(amp,turns,model,curve,coeff,g,b,sigma):
    import torch
    from lab_caltech20_dvp_TRAIN_spatial_response_20261020 import spatial_intensity
    from lab_caltech20_TRAIN_flat_response_20261024 import apply_map
    value=spatial_intensity(amp,turns,model.propagator,curve,sigma,amp.new_tensor(0.),coeff)
    return apply_map(value,g,b)


def selftest():
    import torch
    from torch.nn import functional as F
    from lab_caltech20_dvp_TRAIN_spatial_response_20261020 import monotonic_curve
    from lab_caltech20_TRAIN_native_response_fit_20261024 import response,contrast_loss
    torch.set_num_threads(2);torch.manual_seed(73);f=torch.fft.fftfreq(518);transfer=torch.exp(12j*(f[:,None].square()+f[None,:].square()))
    class Model:
        def propagator(self,x):return torch.fft.ifft2(torch.fft.fft2(x)*transfer)
    model=Model();amp=torch.rand(2,478,478);turns=torch.rand(478,478);g=torch.ones(8,8)*.7;b=torch.ones(8,8)*.4;s=torch.tensor(4.11)
    knots=torch.full((16,),float(torch.log(torch.expm1(torch.tensor(2*torch.pi/16)))),requires_grad=True);scale=torch.tensor(-.28768207245,requires_grad=True);pupil=torch.zeros(8,requires_grad=True);curve=monotonic_curve(knots,scale)
    expected=response(model.propagator(F.pad(amp.to(torch.complex64)*torch.exp(2j*torch.pi*turns)[None],(20,20,20,20))).abs().square(),g,b,s)
    got=prediction(amp,turns,model,curve,torch.pi*pupil.tanh(),g,b,s);error=float((got-expected).abs().max());assert torch.allclose(got,expected,atol=2e-5,rtol=2e-5)
    loss=contrast_loss(got,torch.rand_like(got));loss.backward();assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in (knots,scale,pupil))
    assert all(float(v.grad.norm())>0 for v in (knots,scale,pupil))
    print(json.dumps(dict(passed=True,nominal_response_identity_maxabs=error,finite_phase_pupil_gradients=True,TEST_used=False,SDK_used=False,GPU_used=False,new_inference_layers=0)))


def run(a):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    import numpy as np,torch
    from PIL import Image
    from lab_caltech20_dvp_TRAIN_spatial_response_20261020 import monotonic_curve
    from lab_caltech20_dvp_TRAIN_response_20261020 import phase_patterns
    import lab_caltech20_tucam_TRAIN_response_20261024 as source
    import lab_caltech20_TRAIN_native_response_fit_20261024 as native
    import lab_caltech20_TRAIN_photometric_mask_20261024 as mask
    base=mask.base;_,_,m,_=base.numerical_modules();p=m.read(a.profile);mask.validate_policy(p)
    assert m.sha(native.__file__)=='e79673be1ccc079768880877e7f42a05c28f295f5caebdb36e010f9e199fa8c8'
    assert m.sha(source.__file__)=='421a18ea3c8aecae7c07306146efb6a56a78d05baadfac3a51c6108bea718448'
    fp=base.HERE/'lab_caltech20_TRAIN_flat_response_profile_20261024.json';assert m.sha(fp)=='3c3e85db21beda6fb82b5f9abed1195020de7e111048eacec1fa437bdcf89a51'
    flat=m.read(fp);assert m.sha(base.HERE/'lab_caltech20_dvp_TRAIN_spatial_response_20261020.py')==flat['spatial_engine_SHA256']
    rp=m.read(base.HERE/flat['response_profile']);source.validate_policy(rp);model,phase,hw,protected,anchors=source.context(rp)
    folder=base.ROOT/p['response_fit'];folder=folder.parent/'native_feature_fit';r=m.read(folder/'report.json')
    assert m.sha(folder/'report.json')=='d883bd75f4cbfb822aeb2375b71bad4e266b3ee370935ddd18563e1b3a02afc6' and r['native_response_admitted'] and not r['TEST_used']
    assert m.sha(folder/'TRAIN_native_response.npz')=='2e8a1aa53f0acc4df02f4ef82609462fa2fba06e14709cce1168d9abb3c47e96'
    with np.load(folder/'TRAIN_native_response.npz',allow_pickle=False) as z:g=torch.from_numpy(z['gain'].copy());b=torch.from_numpy(z['background'].copy());sigma=torch.tensor(float(z['sigma_pixels']))
    probes=folder.parent/'probes';pr=m.read(probes/'report.json');assert m.sha(probes/'report.json')=='ed5f98770ce72fc4297d92f7125db0c41508af7bac22405b5dac0012e8973639' and not pr['TEST_used']
    assert not a.out.exists() and not m.LOCK.exists();a.out.mkdir();torch.set_num_threads(2);torch.manual_seed(73);start=time.monotonic();patterns=dict(phase_patterns(phase));lookup={(v['condition'],v['index']):v for v in pr['records']};pack=[]
    for anchor in anchors:
        amp=torch.from_numpy(np.asarray(Image.open(anchor['canonical'])).astype(np.float32)/255)
        for tag in ('flat','nominal_before','half_code','global_offset128'):
            row=lookup[tag,anchor['index']];assert row['sample_id']==anchor['sample_id'] and row['canonical_SHA256']==anchor['canonical_SHA256'] and row['split']==anchor['split']
            assert m.sha(row['crop'])==row['crop_SHA256'] and m.sha(row['phase'])==row['phase_SHA256']
            y=torch.from_numpy(np.asarray(Image.open(row['crop'])).astype(np.float32));y=y/y.mean().clamp_min(1e-8)
            pack.append(dict(index=anchor['index'],sample_id=anchor['sample_id'],split=anchor['split'],condition=tag,amp=amp,turns=torch.from_numpy(patterns[tag].astype(np.float32)/256),observed=y))
    fit=[v for v in pack if v['split']=='fit'];held=[v for v in pack if v['split']=='held_TRAIN'];assert len(fit)==len(held)==80
    knots=torch.nn.Parameter(torch.full((16,),float(torch.log(torch.expm1(torch.tensor(2*torch.pi/16))))));scale=torch.nn.Parameter(torch.tensor(-.28768207245));pupil=torch.nn.Parameter(torch.zeros(8));uniform=torch.linspace(0,2*torch.pi,17)
    def values():return monotonic_curve(knots,scale),torch.pi*pupil.tanh()
    def batch(rows,curve,coeff):return torch.cat([prediction(v['amp'][None],v['turns'],model,curve,coeff,g,b,sigma) for v in rows])
    def assess(rows):
        losses=[];pixel=[];coarse=[];act=[]
        from torch.nn import functional as F
        with torch.no_grad():
            for begin in range(0,len(rows),4):
                chunk=rows[begin:begin+4];y=torch.stack([v['observed'] for v in chunk]);x=batch(chunk,*values());losses.append(float(native.objective(x,y,model))*len(chunk));pixel.append(float(1-native.contrast_loss(x,y))*len(chunk));act.append(float(1-native.contrast_loss(model.nonlinear(x,'global'),model.nonlinear(y,'global')))*len(chunk));coarse.append(float(1-native.contrast_loss(F.adaptive_avg_pool2d(x[:,None],(16,16)),F.adaptive_avg_pool2d(y[:,None],(16,16))))*len(chunk))
        return dict(loss=sum(losses)/len(rows),full478_PCC=sum(pixel)/len(rows),native_optical_PCC=sum(act)/len(rows),coarse16_PCC=sum(coarse)/len(rows),pairs=len(rows))
    groups={'fit':fit,'held':held};groups.update({'held_'+tag:[v for v in held if v['condition']==tag] for tag in patterns if tag!='nominal_after'})
    baseline={k:assess(v) for k,v in groups.items()}
    assert abs(baseline['held_nominal_before']['native_optical_PCC']-r['fitted']['held_nominal_before']['native_optical_PCC'])<1e-4
    m.save(a.out/'protocol.json',dict(source_git=m.git_head(),source_SHA256=m.sha(__file__),profile_SHA256=m.sha(a.profile),recipe=RECIPE,TRAIN_fit_anchors=20,TRAIN_held_anchors=20,fit_pairs=80,held_pairs=80,
        native_response_report_SHA256=m.sha(folder/'report.json'),native_response_map_SHA256=m.sha(folder/'TRAIN_native_response.npz'),probe_report_SHA256=m.sha(probes/'report.json'),
        TEST_used=False,classification_labels_used=False,SDK_used=False,GPU_used=False,new_CCD=0,photometric_map_and_sigma_frozen=True,pupil_bounds_radians=[-float(torch.pi),float(torch.pi)],no_warp=True,model_checkpoint_written=False,hardware_changed=False,training_forward_only=True))
    m.save(a.out/'baseline.json',baseline);opt=torch.optim.Adam([knots,scale,pupil],lr=.02);rng=np.random.default_rng(73);steps=0;history=[];fitstart=time.monotonic()
    for epoch in range(1,21):
        order=rng.permutation(80);total=0.
        for begin in range(0,80,4):
            rows=[fit[i] for i in order[begin:begin+4]];y=torch.stack([v['observed'] for v in rows]);opt.zero_grad(set_to_none=True);curve,coeff=values();x=batch(rows,curve,coeff)
            loss=native.objective(x,y,model)+.0001*((curve-uniform).square().mean()+coeff.square().mean());assert torch.isfinite(loss);loss.backward();assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in (knots,scale,pupil));torch.nn.utils.clip_grad_norm_([knots,scale,pupil],1.);opt.step();steps+=1;total+=float(loss.detach())*4
        row=dict(epoch=epoch,TRAIN_pairs=80,TRAIN_loss=total/80,updates=steps,elapsed_s=time.monotonic()-fitstart);history.append(row);m.save(a.out/'history.json',history);m.save(a.out/'progress.json',row);print(json.dumps(row),flush=True)
        if time.monotonic()-fitstart>=300:break
    final={k:assess(v) for k,v in groups.items()};accepted=final['fit']['loss']<baseline['fit']['loss'] and final['held']['loss']<=.95*baseline['held']['loss']
    for tag in ('held','held_nominal_before'):accepted=accepted and final[tag]['native_optical_PCC']>=baseline[tag]['native_optical_PCC']+.03 and final[tag]['full478_PCC']>=baseline[tag]['full478_PCC'] and final[tag]['coarse16_PCC']>=baseline[tag]['coarse16_PCC']-.01
    accepted=accepted and all(final['held_'+tag]['native_optical_PCC']>=baseline['held_'+tag]['native_optical_PCC'] for tag in ('flat','half_code','global_offset128'))
    curve,coeff=[v.detach() for v in values()];assert protected=={q:m.sha(q) for q in protected} and not m.LOCK.exists()
    model_response=dict(phase_curve_radians=curve.tolist(),pupil_coefficients_radians=coeff.tolist(),gain=g.tolist(),background=b.tolist(),sigma_pixels=float(sigma))
    m.save(a.out/'TRAIN_phase_response.json',model_response)
    result=dict(state='complete',phase_native_response_admitted=bool(accepted),baseline=baseline,fitted=final,actual_epochs=epoch,actual_updates=steps,TRAIN_unique=40,fit_pairs=80,held_pairs=80,TEST_used=False,classification_labels_used=False,
        new_CCD=0,SDK_used=False,GPU_used=False,model_checkpoint_written=False,new_inference_layers=0,hardware_changed=False,LUT_changed=False,classification_accuracy_updated=False,training_forward_only=True,
        protocol_SHA256=m.sha(a.out/'protocol.json'),response_SHA256=m.sha(a.out/'TRAIN_phase_response.json'),elapsed_s=time.monotonic()-start)
    m.save(a.out/'report.json',result);m.save(a.out/'independent_audit.json',dict(passed=True,report_SHA256=m.sha(a.out/'report.json'),TRAIN_only=True,hardware_and_source_PT_frozen=True));print(json.dumps(result),flush=True)


if __name__=='__main__':
    q=argparse.ArgumentParser();q.add_argument('--profile',type=Path,required=True);q.add_argument('--out',type=Path,required=True);run(q.parse_args())
