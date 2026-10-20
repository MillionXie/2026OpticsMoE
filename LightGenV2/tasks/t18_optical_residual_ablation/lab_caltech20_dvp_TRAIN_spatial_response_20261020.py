"""TRAIN-only spatial response diagnosis after scalar/background fit was weak.

One monotonic 17-knot response and eight low-order pupil polynomials. No image
warp, hardware LUT write, coordinates, new model checkpoint or TEST inference.
The fitted curve is an empirical training surrogate, not physical identification.
"""
import argparse
import os
from pathlib import Path
import time


def monotonic_curve(increments,scale_raw):
    import torch
    from torch.nn import functional as F
    steps=F.softplus(increments)
    total=2*torch.pi*(.25+1.75*scale_raw.sigmoid())
    return torch.cat([steps.new_zeros(1),steps.cumsum(0)/steps.sum()*total])


def wavefront_basis(value):
    import torch
    xy=torch.linspace(-1,1,478,device=value.device,dtype=value.dtype)
    y,x=torch.meshgrid(xy,xy,indexing='ij');r=x.square()+y.square()
    modes=torch.stack([x,y,2*r-1,x*y,x.square()-y.square(),x*(3*r-2),y*(3*r-2),6*r.square()-6*r+1])
    modes=modes-modes.mean((-2,-1),keepdim=True)
    return modes/modes.square().mean((-2,-1),keepdim=True).sqrt()


def spatial_intensity(amplitude,turns,propagator,curve,sigma,background,coefficients):
    import torch
    from torch.nn import functional as F
    position=(turns*16).clamp(0,16-1e-6);i=position.floor().long();fraction=position-i
    phase=curve[i]*(1-fraction)+curve[i+1]*fraction
    phase=phase+(coefficients[:,None,None]*wavefront_basis(amplitude)).sum(0)
    field=amplitude.to(torch.complex64)*torch.exp(1j*phase)
    intensity=propagator(F.pad(field,(20,20,20,20))).abs().square()
    f=torch.fft.fftfreq(518,device=amplitude.device,dtype=amplitude.dtype)
    kernel=torch.exp(-2*torch.pi**2*sigma.square()*(f[:,None].square()+f[None,:].square()))
    blurred=torch.fft.ifft2(torch.fft.fft2(intensity)*kernel).real.clamp_min(0)[:,20:498,20:498]
    return blurred+background*blurred.mean((-2,-1),keepdim=True)


def spatial_loss(predicted,observed):
    from torch.nn import functional as F
    def contrast(x):
        x=x-x.mean((-2,-1),keepdim=True)
        return x/x.square().mean((-2,-1),keepdim=True).sqrt().clamp_min(1e-6)
    loss=predicted.new_zeros(())
    for size,weight in ((8,.4),(16,.35),(32,.15)):
        p=contrast(F.adaptive_avg_pool2d(predicted[:,None],(size,size)))
        o=contrast(F.adaptive_avg_pool2d(observed[:,None],(size,size)))
        loss=loss+weight*(p-o).square().mean()
    p=predicted/predicted.square().mean((-2,-1),keepdim=True).sqrt().clamp_min(1e-6)
    o=observed/observed.square().mean((-2,-1),keepdim=True).sqrt().clamp_min(1e-6)
    return loss+.1*(p-o).square().mean()


def mask_review_rows(sequence,folder,propagator):
    """Only the one audited TRAIN diagnostic, never TEST or a deployment gate."""
    import numpy as np
    import torch
    from PIL import Image
    import lab_caltech20_dvp_layerwise_20261019 as m
    from lab_caltech20_dvp_TRAIN_mask_review_20261021 import CASES
    report=m.read(folder/'report.json');audit=m.read(folder/'independent_audit.json');protocol=m.read(folder/'protocol.json')
    assert report['state']=='complete' and report['response_frames']==60 and report['SDK_released'] and not report['TEST_inference'] and not report['stage_admission']
    assert audit['passed'] and audit['report_SHA256']==m.sha(folder/'report.json') and audit['protocol_SHA256']==m.sha(folder/'protocol.json')
    tag,namespace,fit,digest=CASES[1];assert tag=='response_trained'
    model,_=m.model_for(sequence/namespace/'fits'/fit/'best.pt',digest,4,sequence/namespace/'source_reference.ts')
    assert torch.equal(model.propagator.transfer,propagator.transfer)
    folder=folder/tag;phase=np.asarray(Image.open(folder/'phase.bmp'))
    # The saved phase is physical-native; reconstruct the exact logical codes.
    plane=model.plane('2');turns=torch.floor(torch.remainder(torch.angle(plane),2*torch.pi)/(2*torch.pi)*256).clamp(0,255)[20:498,20:498].numpy().astype(np.float32)/256
    assert np.array_equal(m.phase_native(np.rint(turns*256).astype(np.uint8)),phase)
    rows=[]
    for i in range(20):
        stem=folder/f'{i:06d}';r=m.read(stem.with_suffix('.json'));amp=stem.with_name(stem.name+'_input.png');crop=stem.with_suffix('.png')
        assert r['index']==i and r['PT_SHA256']==digest and m.sha(amp)==r['canonical_SHA256'] and m.sha(crop)==r['crop_SHA256']
        rows.append(dict(depth=4,index=i,tag='trained_mask_TRAIN_response',amplitude=np.asarray(Image.open(amp)).astype(np.float32)/255,turns=turns,observed=np.asarray(Image.open(crop)).astype(np.float32)))
    return rows,dict(report_SHA256=audit['report_SHA256'],audit_SHA256=m.sha(folder.parent/'independent_audit.json'),data_SHA256=protocol['data_SHA256'],unique_TRAIN=20,new_independent_samples=0,TEST_inference=False)


def selftest():
    import torch
    from torch.nn import functional as F
    torch.set_num_threads(2);torch.manual_seed(73)
    q=torch.zeros(16,requires_grad=True);scale=torch.tensor(-.287682,requires_grad=True)
    curve=monotonic_curve(q,scale)
    assert bool((curve[1:]>curve[:-1]).all()) and abs(float(curve[-1])-2*torch.pi)<1e-5
    inverse=torch.log(torch.expm1(curve.detach()[1:]-curve.detach()[:-1]))
    assert torch.allclose(monotonic_curve(inverse,scale.detach()),curve.detach(),atol=3e-6,rtol=1e-6)
    amplitude=torch.rand(1,478,478);turns=torch.rand(478,478)
    coeff=torch.zeros(8,requires_grad=True);sigma=torch.tensor(1.,requires_grad=True);b=torch.tensor(.2,requires_grad=True)
    f=torch.fft.fftfreq(518);transfer=torch.exp(30j*(f[:,None].square()+f[None,:].square()))
    prop=lambda x:torch.fft.ifft2(torch.fft.fft2(x)*transfer)
    output=spatial_intensity(amplitude,turns,prop,curve,sigma,b,coeff)
    loss=spatial_loss(output,amplitude.square());loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in (q,scale,coeff,sigma,b))
    assert torch.isfinite(output).all() and bool((output>=0).all())
    print(dict(passed=True,monotonic_phase_curve=True,finite_gradients=True,no_warp=True,SDK_used=False,TEST_used=False))


def run(a):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    import numpy as np
    import torch
    from torch.nn import functional as F
    from PIL import Image
    import lab_caltech20_dvp_layerwise_20261019 as m
    from lab_caltech20_dvp_TRAIN_response_20261020 import CASES,phase_patterns,checker_phase,HARDWARE
    from lab_caltech20_dvp_TRAIN_response_fit_20261020 import response_intensity
    from lab_caltech20_residual_terminal_cpu_20261019 import field_agreement
    torch.set_num_threads(2);torch.manual_seed(73)
    assert not a.out.exists() and not m.LOCK.exists()
    prior=m.read(a.response/'report.json');audit=m.read(a.probes/'independent_audit.json');probes=m.read(a.probes/'report.json')
    assert audit['passed'] and audit['report_SHA256']==m.sha(a.probes/'report.json')
    assert prior['response_fit_admitted'] and not prior['TEST_used']
    assert probes['protected']=={p:m.sha(p) for p in probes['protected']}
    index={(r['depth'],r['condition'],r['index']):r for r in probes['records']};rows=[]
    for depth,stage,parent,digest in CASES:
        session=a.sequence/f'L{depth}_strict8pp_20261020';capture=session/'capture'/stage
        model,_=m.model_for(session/'fits'/parent/'best.pt',digest,depth,session/'source_reference.ts')
        plane=model.plane(stage);phase=torch.floor(torch.remainder(torch.angle(plane),2*torch.pi)/(2*torch.pi)*256).clamp(0,255)[20:498,20:498].detach().numpy().astype(np.uint8)
        patterns=dict(phase_patterns(phase));patterns.update({f'checker_{v}':checker_phase(v) for v in (0,64,128,192)})
        for i in range(20):
            path=capture/'train'/f'{i:06d}_input.png';amp=np.asarray(Image.open(path)).astype(np.float32)/255
            for tag in ('nominal_before','flat','half_code','global_offset128')+tuple(f'checker_{v}' for v in (0,64,128,192) if i<2):
                record=index[depth,tag,i];assert m.sha(path)==record['canonical_SHA256'] and m.sha(record['crop'])==record['crop_SHA256']
                rows.append(dict(depth=depth,index=i,tag=tag,amplitude=amp,turns=patterns[tag].astype(np.float32)/256,observed=np.asarray(Image.open(record['crop'])).astype(np.float32)))
    mask_review=getattr(a,'mask_review',None);spatial_prior=getattr(a,'spatial_prior',None)
    assert bool(mask_review)==bool(spatial_prior),'A new-mask response refit requires both audited TRAIN review and exact original spatial prior'
    extra_proof=None
    if mask_review:
        extra,extra_proof=mask_review_rows(a.sequence,mask_review,model.propagator)
        assert extra_proof['data_SHA256']==m.read(a.sequence/'assets/data_manifest.json')['sha256']
        rows.extend(extra)
    def tensors(rows):return tuple(torch.from_numpy(np.stack([v[key] for v in rows])) for key in ('amplitude','turns','observed'))
    train=tensors([v for v in rows if v['index']%2==0]);held=tensors([v for v in rows if v['index']%2==1])
    assert len(train[0])==len(held[0])==(98 if mask_review else 88)
    pars=prior['parameters'];old=torch.tensor([pars[k] for k in ('phase_scale','phase_gamma','sigma_pixels','relative_background')])
    x=torch.linspace(0,1,17);steps=(x[1:].pow(pars['phase_gamma'])-x[:-1].pow(pars['phase_gamma']))*16
    knots=torch.nn.Parameter(torch.log(torch.expm1(steps)))
    scale=torch.nn.Parameter(torch.logit(torch.tensor((pars['phase_scale']-.25)/1.75)))
    sigma=torch.nn.Parameter(torch.logit(torch.tensor(pars['sigma_pixels']/12)))
    background=torch.nn.Parameter(torch.logit(torch.tensor(pars['relative_background']/3)))
    coefficients=torch.nn.Parameter(torch.zeros(8));parameters=[knots,scale,sigma,background,coefficients]
    if spatial_prior:
        fitted=m.read(spatial_prior/'report.json');assert fitted['state']=='complete' and fitted['spatial_response_admitted'] and not fitted['TEST_used']
        curve=torch.tensor(fitted['phase_curve_radians']);ratio=(curve[-1]/(2*torch.pi)-.25)/1.75
        with torch.no_grad():
            knots.copy_(torch.log(torch.expm1(curve[1:]-curve[:-1])))
            scale.copy_(torch.logit(ratio.clamp(1e-6,1-1e-6)));sigma.copy_(torch.logit(torch.tensor(fitted['sigma_pixels']/12).clamp(1e-6,1-1e-6)))
            background.copy_(torch.logit(torch.tensor(fitted['relative_background']/3).clamp(1e-6,1-1e-6)))
            coefficients.copy_(torch.atanh((torch.tensor(fitted['pupil_coefficients_radians'])/(12*torch.pi)).clamp(-1+1e-6,1-1e-6)))
    def physical():return monotonic_curve(knots,scale),12*sigma.sigmoid(),3*background.sigmoid(),12*torch.pi*coefficients.tanh()
    def evaluate(data,old_model=False):
        losses=[];pcc=[]
        with torch.no_grad():
            for i in range(0,len(data[0]),4):
                aa,pp,oo=(v[i:i+4] for v in data)
                predicted=response_intensity(aa,pp,model.propagator,old) if old_model else spatial_intensity(aa,pp,model.propagator,*physical())
                losses.append(float(spatial_loss(predicted,oo)));pcc.extend(field_agreement(predicted,oo)['coarse16_Pearson'])
        return dict(spatial_loss=float(np.mean(losses)),mean_coarse16_PCC=float(np.mean(pcc)),median_coarse16_PCC=float(np.median(pcc)))
    a.out.mkdir();m.save(a.out/'protocol.json',dict(source_git=m.git_head(),source_SHA256=m.sha(__file__),probe_report_SHA256=m.sha(a.probes/'report.json'),prior_report_SHA256=m.sha(a.response/'report.json'),
        TRAIN_fit_pairs=len(train[0]),held_TRAIN_pairs=len(held[0]),TEST_used=False,VALID=0,labels_used=False,steps=400,batch=4,Adam_lr=.02,seed=73,
        parameters=27,phase_knots=17,monotonic=True,pupil_polynomials=8,maximum_coefficient_radians=12*np.pi,
        image_warp=False,coordinates_changed=False,LUT_written=False,model_checkpoint_written=False,
        loss='multiscale contrast 8/16/32 weights .4/.35/.15 plus unit-RMS .1; brightness flattening cannot satisfy spatial criterion',
        mask_review=extra_proof,spatial_prior_report_SHA256=m.sha(spatial_prior/'report.json') if spatial_prior else None,
        new_mask_batch_sampling='2 original-response TRAIN pairs plus2 new-mask TRAIN pairs; no new independent samples' if mask_review else None,
        admission='new-mask held TRAIN spatial loss >=5% better AND PCC >=.03 better; old-response held TRAIN loss regression <=5%; pooled fit TRAIN improves' if mask_review else 'held TRAIN spatial loss >=5% better AND mean coarse16 PCC >=.03 better than prior; fit TRAIN loss also improves'))
    baseline=dict(TRAIN=evaluate(train,not bool(mask_review)),held_TRAIN=evaluate(held,not bool(mask_review)))
    if mask_review:
        baseline.update(old_held_TRAIN=evaluate(tuple(v[:88] for v in held)),new_mask_held_TRAIN=evaluate(tuple(v[88:] for v in held)))
    opt=torch.optim.Adam(parameters,lr=.02);history=[];started=time.monotonic()
    for step in range(1,401):
        ix=torch.cat([torch.randperm(88)[:2],88+torch.randperm(10)[:2]]) if mask_review else torch.randperm(88)[:4]
        aa,pp,oo=(v[ix] for v in train);opt.zero_grad(set_to_none=True)
        predicted=spatial_intensity(aa,pp,model.propagator,*physical());loss=spatial_loss(predicted,oo)+.0001*coefficients.square().sum()
        assert torch.isfinite(loss);loss.backward();assert all(torch.isfinite(v.grad).all() for v in parameters);opt.step()
        if step%50==0:
            row=dict(step=step,TRAIN_batch_spatial_loss=float(loss.detach()),elapsed_s=time.monotonic()-started)
            history.append(row);m.save(a.out/'history.json',history);m.save(a.out/'progress.json',row)
    final=dict(TRAIN=evaluate(train),held_TRAIN=evaluate(held));curve,psf,bg,coeff=physical()
    admitted=(final['TRAIN']['spatial_loss']<baseline['TRAIN']['spatial_loss'] and final['held_TRAIN']['spatial_loss']<=.95*baseline['held_TRAIN']['spatial_loss'] and final['held_TRAIN']['mean_coarse16_PCC']>=baseline['held_TRAIN']['mean_coarse16_PCC']+.03)
    if mask_review:
        final.update(old_held_TRAIN=evaluate(tuple(v[:88] for v in held)),new_mask_held_TRAIN=evaluate(tuple(v[88:] for v in held)))
        oldb,newb=baseline['old_held_TRAIN'],baseline['new_mask_held_TRAIN'];oldf,newf=final['old_held_TRAIN'],final['new_mask_held_TRAIN']
        admitted=(final['TRAIN']['spatial_loss']<baseline['TRAIN']['spatial_loss'] and oldf['spatial_loss']<=1.05*oldb['spatial_loss'] and newf['spatial_loss']<=.95*newb['spatial_loss'] and newf['mean_coarse16_PCC']>=newb['mean_coarse16_PCC']+.03)
    q=dict(state='complete',spatial_response_admitted=admitted,baseline=baseline,fitted=final,
        phase_curve_radians=curve.detach().tolist(),sigma_pixels=float(psf.detach()),relative_background=float(bg.detach()),pupil_coefficients_radians=coeff.detach().tolist(),
        steps=400,TEST_used=False,VALID=0,labels_used=False,training_surrogate_only=True,hardware_unchanged=True,LUT_written=False,physical_phase_calibrated=False,
        source_git=m.git_head(),source_SHA256=m.sha(__file__),protocol_SHA256=m.sha(a.out/'protocol.json'),prior_report_SHA256=m.sha(a.response/'report.json'),
        elapsed_s=time.monotonic()-started,model_checkpoint_written=False,unique_physical_cause_identified=False,mask_review=extra_proof,original_spatial_prior_report_SHA256=m.sha(spatial_prior/'report.json') if spatial_prior else None)
    assert probes['protected']=={p:m.sha(p) for p in probes['protected']}
    m.save(a.out/'report.json',q);print(m.json.dumps(q),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--selftest',action='store_true')
    for key in ('sequence','probes','response','out','mask-review','spatial-prior'):p.add_argument('--'+key,type=Path)
    a=p.parse_args()
    if a.selftest:selftest()
    else:
        assert all(getattr(a,k) for k in ('sequence','probes','response','out'))
        run(a)
