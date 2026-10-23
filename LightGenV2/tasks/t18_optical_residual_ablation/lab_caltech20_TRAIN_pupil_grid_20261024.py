"""TRAIN-only shared complex-pupil response; current E2 is an unseen phase.

Not a hardware calibration/LUT, new inference layer, or classification result.
One fixed last-iterate fit; held TRAIN anchors/phases never select steps.
"""
import argparse,json,os,time
from pathlib import Path
RECIPE=dict(epochs=48,updates=960,batch=4,seconds=600,lr=.02,seed=73,CPU_threads=2,grid=16,parameters=529,regularization=.001)


def prediction(amp,turns,model,curve,phase_grid,amplitude_grid,g,b,sigma):
    import torch
    from torch.nn import functional as F
    from lab_caltech20_TRAIN_native_response_fit_20261024 import response
    position=(turns*16).clamp(0,16-1e-6);i=position.floor().long();fraction=position-i
    phase=curve[i]*(1-fraction)+curve[i+1]*fraction
    pupil=torch.pi*F.interpolate(phase_grid[None,None].tanh(),size=(478,478),mode='bilinear',align_corners=False)[0,0]
    pupil=pupil-pupil.mean()
    transmission=F.interpolate((.7*amplitude_grid.tanh()).exp()[None,None],size=(478,478),mode='bilinear',align_corners=False)[0,0]
    transmission=transmission/transmission.square().mean().sqrt()
    field=(amp*transmission).to(torch.complex64)*torch.exp(1j*(phase+pupil))
    return response(model.propagator(F.pad(field,(20,20,20,20))).abs().square(),g,b,sigma)


def selftest():
    import torch
    from torch.nn import functional as F
    from lab_caltech20_TRAIN_native_response_fit_20261024 import response,contrast_loss
    from lab_caltech20_dvp_TRAIN_spatial_response_20261020 import monotonic_curve
    torch.set_num_threads(2);torch.manual_seed(73);f=torch.fft.fftfreq(518);transfer=torch.exp(12j*(f[:,None].square()+f[None,:].square()))
    class Model:
        def propagator(self,x):return torch.fft.ifft2(torch.fft.fft2(x)*transfer)
    model=Model();amp=torch.rand(2,478,478);turns=torch.rand(478,478);g=torch.ones(8,8)*.7;b=torch.ones(8,8)*.4;sigma=torch.tensor(4.111456871)
    knots=torch.full((16,),float(torch.log(torch.expm1(torch.tensor(2*torch.pi/16)))),requires_grad=True);scale=torch.tensor(-.28768207245,requires_grad=True)
    pg=torch.zeros(16,16,requires_grad=True);ag=torch.zeros(16,16,requires_grad=True);curve=monotonic_curve(knots,scale)
    expected=response(model.propagator(F.pad(amp.to(torch.complex64)*torch.exp(2j*torch.pi*turns)[None],(20,20,20,20))).abs().square(),g,b,sigma)
    got=prediction(amp,turns,model,curve,pg,ag,g,b,sigma);error=float((got-expected).abs().max());assert torch.allclose(got,expected,atol=2e-5,rtol=2e-5)
    loss=contrast_loss(got,torch.rand_like(got));loss.backward();assert all(v.grad is not None and torch.isfinite(v.grad).all() and float(v.grad.norm())>0 for v in (knots,scale,pg,ag))
    print(json.dumps(dict(passed=True,nominal_response_identity_maxabs=error,finite_curve_phase_amplitude_gradients=True,TEST_used=False,SDK_used=False,GPU_used=False,new_inference_layers=0)))


def run(a):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    import numpy as np,torch
    from PIL import Image
    from torch.nn import functional as F
    from lab_caltech20_dvp_TRAIN_spatial_response_20261020 import monotonic_curve
    from lab_caltech20_dvp_TRAIN_response_20261020 import phase_patterns
    import lab_caltech20_tucam_TRAIN_response_20261024 as source
    import lab_caltech20_TRAIN_native_response_fit_20261024 as native
    import lab_caltech20_TRAIN_photometric_mask_20261024 as mask
    base=mask.base;_,_,m,_=base.numerical_modules();p=m.read(a.profile);mask.validate_policy(p)
    assert m.sha(source.__file__)=='421a18ea3c8aecae7c07306146efb6a56a78d05baadfac3a51c6108bea718448'
    assert m.sha(native.__file__)=='e79673be1ccc079768880877e7f42a05c28f295f5caebdb36e010f9e199fa8c8'
    flat=m.read(base.HERE/'lab_caltech20_TRAIN_flat_response_profile_20261024.json');rp=m.read(base.HERE/flat['response_profile']);source.validate_policy(rp)
    model,phase,hw,protected,anchors=source.context(rp);folder=(base.ROOT/p['response_fit']).parent/'native_feature_fit';r=m.read(folder/'report.json')
    assert m.sha(folder/'report.json')=='d883bd75f4cbfb822aeb2375b71bad4e266b3ee370935ddd18563e1b3a02afc6' and r['native_response_admitted'] and not r['TEST_used']
    assert m.sha(folder/'TRAIN_native_response.npz')=='2e8a1aa53f0acc4df02f4ef82609462fa2fba06e14709cce1168d9abb3c47e96'
    with np.load(folder/'TRAIN_native_response.npz',allow_pickle=False) as z:g=torch.from_numpy(z['gain'].copy());b=torch.from_numpy(z['background'].copy());sigma=torch.tensor(float(z['sigma_pixels']))
    probes=folder.parent/'probes';pr=m.read(probes/'report.json');assert m.sha(probes/'report.json')=='ed5f98770ce72fc4297d92f7125db0c41508af7bac22405b5dac0012e8973639' and not pr['TEST_used']
    current=folder.parents[1]/'e2_actual300_20261024';cap=current/'s_2/capture';cr=m.read(cap/'report.json');ca=m.read(cap/'independent_audit.json');cp=m.read(current/'protocol.json')['policy']
    assert ca['passed'] and ca['matched_samples']==2011 and ca['report_SHA256']==m.sha(cap/'report.json')=='84c24824e99045c4987061d77d9e0d7d1f905732baeaf07746fcefa0a2f0c9cd'
    candidate=base.ROOT/cp['candidate_fit']/'best.pt';assert m.sha(candidate)==cr['PT_SHA256']==cp['candidate_PT_SHA256']
    c=m.read(base.HERE/p['base_profile']);new_model,_=m.model_for(candidate,cp['candidate_PT_SHA256'],4,base.ROOT/c['reference'])
    plane=new_model.plane('2');new_phase=torch.floor(torch.remainder(torch.angle(plane),2*np.pi)/(2*np.pi)*256).clamp(0,255)[20:498,20:498].detach().numpy().astype(np.uint8)
    assert np.array_equal(m.phase_native(new_phase),np.asarray(Image.open(cap/'phase.bmp')))
    assert m.sha(cap/'phase.bmp')==cr['phase_SHA256']=='2e469eb91c507b777348d157346b14cdb657c2d09138385bedae2a6985458151'
    assert cr['hardware_SHA256']==rp['hardware_SHA256'] and cr['TRAIN']==1600 and cr['TEST']==411
    measured_path=cap/'train/ccd.npy';assert m.sha(measured_path)==cr['arrays']['train']['SHA256']=='e05e783e2e007058b0bc6753379a7d476cc5755a3cc32a8cd2ef87ba46ec4b77'
    measured=np.load(measured_path,mmap_mode='r');protected.update({str(q):m.sha(q) for q in (candidate,cap/'phase.bmp',cap/'report.json',measured_path)})
    assert not a.out.exists();a.out.mkdir();torch.set_num_threads(2);torch.manual_seed(73);start=time.monotonic();patterns=dict(phase_patterns(phase));lookup={(v['condition'],v['index']):v for v in pr['records']};pack=[];unseen=[]
    for anchor in anchors:
        amp=torch.from_numpy(np.asarray(Image.open(anchor['canonical'])).astype(np.float32)/255)
        for tag in ('flat','nominal_before','half_code','global_offset128'):
            row=lookup[tag,anchor['index']];assert row['sample_id']==anchor['sample_id'] and row['canonical_SHA256']==anchor['canonical_SHA256'] and row['split']==anchor['split']
            assert m.sha(row['crop'])==row['crop_SHA256'] and m.sha(row['phase'])==row['phase_SHA256'];y=torch.from_numpy(np.asarray(Image.open(row['crop'])).astype(np.float32));y=y/y.mean().clamp_min(1e-8)
            pack.append(dict(index=anchor['index'],sample_id=anchor['sample_id'],split=anchor['split'],condition=tag,amp=amp,turns=torch.from_numpy(patterns[tag].astype(np.float32)/256),observed=y))
        stem=cap/'train'/f"{anchor['index']:06d}";receipt=m.read(stem.with_suffix('.json'));input_path=stem.with_name(stem.name+'_input.png');observed_path=stem.with_suffix('.png')
        assert receipt['role']=='train' and receipt['sample_id']==anchor['sample_id'] and receipt['PT_SHA256']==cp['candidate_PT_SHA256'] and receipt['phase_SHA256']==cr['phase_SHA256']
        assert m.sha(input_path)==receipt['canonical_SHA256'];pixels=np.asarray(Image.open(observed_path));assert np.array_equal(pixels,measured[anchor['index']]);observed=torch.from_numpy(pixels.astype(np.float32));observed=observed/observed.mean().clamp_min(1e-8)
        unseen.append(dict(index=anchor['index'],sample_id=anchor['sample_id'],split='unseen_TRAIN_phase',condition='new_E2',amp=torch.from_numpy(np.asarray(Image.open(input_path)).astype(np.float32)/255),turns=torch.from_numpy(new_phase.astype(np.float32)/256),observed=observed))
    fit=[v for v in pack if v['split']=='fit'];held=[v for v in pack if v['split']=='held_TRAIN'];assert len(fit)==len(held)==80 and len(unseen)==40
    knots=torch.nn.Parameter(torch.full((16,),float(torch.log(torch.expm1(torch.tensor(2*torch.pi/16))))));scale=torch.nn.Parameter(torch.tensor(-.28768207245));pg=torch.nn.Parameter(torch.zeros(16,16));ag=torch.nn.Parameter(torch.zeros(16,16));uniform=torch.linspace(0,2*torch.pi,17)
    def values():return monotonic_curve(knots,scale),pg,ag
    def batch(rows):return torch.cat([prediction(v['amp'][None],v['turns'],model,*values(),g,b,sigma) for v in rows])
    def assess(rows):
        loss=pixel=coarse=act=0.
        with torch.no_grad():
            for begin in range(0,len(rows),4):
                chunk=rows[begin:begin+4];y=torch.stack([v['observed'] for v in chunk]);x=batch(chunk);n=len(chunk)
                loss+=float(native.objective(x,y,model))*n;pixel+=float(1-native.contrast_loss(x,y))*n;act+=float(1-native.contrast_loss(model.nonlinear(x,'global'),model.nonlinear(y,'global')))*n
                coarse+=float(1-native.contrast_loss(F.adaptive_avg_pool2d(x[:,None],(16,16)),F.adaptive_avg_pool2d(y[:,None],(16,16))))*n
        return dict(loss=loss/len(rows),full478_PCC=pixel/len(rows),native_optical_PCC=act/len(rows),coarse16_PCC=coarse/len(rows),pairs=len(rows))
    groups={'fit':fit,'held':held,'unseen_TRAIN_phase':unseen};groups.update({'held_'+tag:[v for v in held if v['condition']==tag] for tag in ('flat','nominal_before','half_code','global_offset128')})
    baseline={k:assess(v) for k,v in groups.items()};assert abs(baseline['held_nominal_before']['native_optical_PCC']-r['fitted']['held_nominal_before']['native_optical_PCC'])<1e-4
    m.save(a.out/'protocol.json',dict(source_git=m.git_head(),source_SHA256=m.sha(__file__),recipe=RECIPE,profile_SHA256=m.sha(a.profile),
        fit_TRAIN_anchors=20,held_TRAIN_anchors=20,fit_pairs=80,held_pairs=80,unseen_new_phase_TRAIN_pairs=40,new_phase_fitted=False,
        native_response_report_SHA256=m.sha(folder/'report.json'),native_response_map_SHA256=m.sha(folder/'TRAIN_native_response.npz'),probe_report_SHA256=m.sha(probes/'report.json'),new_E2_capture_report_SHA256=m.sha(cap/'report.json'),
        TEST_used=False,classification_labels_used=False,SDK_used=False,GPU_used=False,new_CCD=0,new_inference_layers=0,hardware_changed=False,model_checkpoint_written=False,training_forward_only=True,
        fit_selection='fixed_last_only_no_held_early_stopping',initialization='nominal_linear_phase_zero_pupil_unit_transmission_NO_rejected_parameters',regularized_complex_pupil_not_unique_physical_cause=True))
    m.save(a.out/'baseline.json',baseline);opt=torch.optim.Adam([knots,scale,pg,ag],lr=.02);rng=np.random.default_rng(73);steps=0;history=[];fitstart=time.monotonic()
    for epoch in range(1,49):
        order=rng.permutation(80);total=0.
        for begin in range(0,80,4):
            rows=[fit[i] for i in order[begin:begin+4]];y=torch.stack([v['observed'] for v in rows]);opt.zero_grad(set_to_none=True);x=batch(rows)
            tv=(pg[1:]-pg[:-1]).square().mean()+(pg[:,1:]-pg[:,:-1]).square().mean()+(ag[1:]-ag[:-1]).square().mean()+(ag[:,1:]-ag[:,:-1]).square().mean()
            penalty=(values()[0]-uniform).square().mean()+pg.tanh().square().mean()+ag.tanh().square().mean()+tv
            loss=native.objective(x,y,model)+.001*penalty;assert torch.isfinite(loss);loss.backward();assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in (knots,scale,pg,ag));torch.nn.utils.clip_grad_norm_([knots,scale,pg,ag],1.);opt.step();steps+=1;total+=float(loss.detach())*4
        row=dict(epoch=epoch,TRAIN_pairs=80,TRAIN_loss=total/80,updates=steps,elapsed_s=time.monotonic()-fitstart);history.append(row);m.save(a.out/'history.json',history);m.save(a.out/'progress.json',row);print(json.dumps(row),flush=True)
        if time.monotonic()-fitstart>=600:break
    final={k:assess(v) for k,v in groups.items()};accepted=final['fit']['loss']<baseline['fit']['loss']
    for tag in ('held','held_nominal_before','unseen_TRAIN_phase'):
        accepted=accepted and final[tag]['loss']<=.95*baseline[tag]['loss'] and final[tag]['native_optical_PCC']>=baseline[tag]['native_optical_PCC']+.03 and final[tag]['full478_PCC']>=baseline[tag]['full478_PCC'] and final[tag]['coarse16_PCC']>=baseline[tag]['coarse16_PCC']-.01
    accepted=accepted and all(final['held_'+tag]['native_optical_PCC']>=baseline['held_'+tag]['native_optical_PCC'] for tag in ('flat','half_code','global_offset128'))
    assert protected=={q:m.sha(q) for q in protected}
    m.save(a.out/'TRAIN_pupil_grid.json',dict(phase_curve_radians=values()[0].detach().tolist(),phase_grid_raw=pg.detach().tolist(),amplitude_grid_raw=ag.detach().tolist(),gain=g.tolist(),background=b.tolist(),sigma_pixels=float(sigma)))
    report=dict(state='complete',pupil_grid_response_admitted=bool(accepted),baseline=baseline,fitted=final,actual_epochs=epoch,actual_updates=steps,TRAIN_unique=40,fit_pairs=80,held_pairs=80,unseen_TRAIN_phase_pairs=40,
        TEST_used=False,classification_labels_used=False,new_CCD=0,SDK_used=False,GPU_used=False,model_checkpoint_written=False,new_inference_layers=0,hardware_changed=False,LUT_changed=False,classification_accuracy_updated=False,training_forward_only=True,
        protocol_SHA256=m.sha(a.out/'protocol.json'),response_SHA256=m.sha(a.out/'TRAIN_pupil_grid.json'),elapsed_s=time.monotonic()-start)
    m.save(a.out/'report.json',report);m.save(a.out/'independent_audit.json',dict(passed=True,report_SHA256=m.sha(a.out/'report.json'),TRAIN_only=True,hardware_and_source_PT_frozen=True));print(json.dumps(report),flush=True)


if __name__=='__main__':
    q=argparse.ArgumentParser();q.add_argument('--profile',type=Path,required=True);q.add_argument('--out',type=Path,required=True);run(q.parse_args())
