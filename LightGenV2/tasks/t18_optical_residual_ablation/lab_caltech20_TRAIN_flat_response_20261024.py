"""Flat-phase TRAIN photometric diagnosis with cross-phase TRAIN validation.

An empirical, low-resolution gain/background observation map is learned only
from 20 original TRAIN inputs under flat phase. It is NOT an image correction
for deployed inference, an optical LUT, a model checkpoint or a phase estimate.
"""
import argparse,json,os,time
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]


def validate_policy(c):
    assert c['schema']=='TUCam_E2_TRAIN_flat_photometric_v1'
    assert c['probes'].endswith('/e2_response_20261024/probes')
    assert c['probe_report_SHA256']=='ed5f98770ce72fc4297d92f7125db0c41508af7bac22405b5dac0012e8973639'
    assert c['fit_condition']=='flat' and c['TRAIN_fit_anchors']==20 and c['TRAIN_held_anchors']==20
    assert c['grid']==8 and c['ridge']==.05 and c['gain_bounds']==[.05,4.] and c['background_bounds']==[0.,4.]
    assert c['held_PCC_gain']==.03 and c['held_loss_ratio']==.95 and c['shared_residual_PCC']==.8
    assert c['TEST_used'] is False and c['VALID']==0 and c['new_CCD']==0 and c['new_inference_layers']==0
    assert c['intermediate_target']==305 and c['final_target']==340 and c['hardware_changed'] is False


def fit_gain_bias(predicted,observed,ridge=.05):
    """Exact 2-variable box-constrained ridge per coarse cell, TRAIN only.

min mean[(g*p+b-o)^2] + ridge*((g-1)^2+b^2), g in [.05,4], b in [0,4].
Interior plus all four edge minima contain the global solution of each QP.
"""
    p,o=np.asarray(predicted,np.float64),np.asarray(observed,np.float64)
    assert p.shape==o.shape and p.ndim==3 and len(p)>=4 and np.isfinite(p).all() and np.isfinite(o).all()
    assert ridge>0
    A=(p*p).mean(0)+ridge;B=p.mean(0);C=1+ridge
    D=(p*o).mean(0)+ridge;E=o.mean(0);den=A*C-B*B
    g=(D*C-B*E)/den;b=(A*E-B*D)/den
    candidates=[(g,b)]
    for bound in (.05,4.):candidates.append((np.full_like(g,bound),np.clip((E-B*bound)/C,0,4)))
    for bound in (0.,4.):candidates.append((np.clip((D-B*bound)/A,.05,4),np.full_like(b,bound)))
    losses=[]
    for gg,bb in candidates:
        loss=(A*gg*gg+2*B*gg*bb+C*bb*bb-2*D*gg-2*E*bb)
        losses.append(np.where((gg>=.05)&(gg<=4)&(bb>=0)&(bb<=4),loss,np.inf))
    which=np.stack(losses).argmin(0)
    gs=np.stack([v[0] for v in candidates]);bs=np.stack([v[1] for v in candidates])
    return np.take_along_axis(gs,which[None],axis=0)[0],np.take_along_axis(bs,which[None],axis=0)[0]


def apply_map(intensity,gain,background):
    import torch
    from torch.nn import functional as F
    assert intensity.ndim==3 and tuple(gain.shape)==tuple(background.shape)==(8,8)
    g=F.interpolate(gain[None,None],size=intensity.shape[-2:],mode='bilinear',align_corners=False)[0,0]
    b=F.interpolate(background[None,None],size=intensity.shape[-2:],mode='bilinear',align_corners=False)[0,0]
    return intensity/intensity.mean((-2,-1),keepdim=True).clamp_min(1e-8)*g+b


def run(profile,out):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    import cv2,torch
    from PIL import Image
    from torch.nn import functional as F
    import lab_caltech20_tucam_TRAIN_response_20261024 as source
    from lab_caltech20_dvp_TRAIN_response_fit_20261020 import response_intensity
    import lab_caltech20_dvp_TRAIN_response_fit_20261020 as nominal_kernel
    import lab_caltech20_dvp_TRAIN_spatial_response_20261020 as spatial
    from lab_caltech20_dvp_TRAIN_response_20261020 import phase_patterns,checker_phase
    from lab_caltech20_tucam_session_20261022 import logical
    import lab_caltech20_dvp_layerwise_20261019 as m
    c=json.loads(profile.read_text());validate_policy(c);torch.set_num_threads(2)
    assert not out.exists() and not m.LOCK.exists()
    assert m.sha(source.__file__)==c['response_entry_SHA256'] and m.sha(nominal_kernel.__file__)==c['nominal_engine_SHA256'] and m.sha(spatial.__file__)==c['spatial_engine_SHA256']
    p=m.read(HERE/c['response_profile']);source.validate_policy(p);model,phase,hw,protected,anchors=source.context(p)
    probes=ROOT/c['probes'];r=m.read(probes/'report.json');audit=probes.parent/'response_final_audit.json'
    assert m.sha(probes/'report.json')==c['probe_report_SHA256'] and m.sha(audit)==c['response_audit_SHA256'] and m.read(audit)['passed']
    assert r['protected']==protected and r['SDK_released'] and r['nominal_phase_restored_before_close']
    assert len(r['records'])==208 and not r['TEST_used']
    out.mkdir();start=time.monotonic();pack=[];bykey={(v['condition'],v['index']):v for v in r['records']}
    patterns=dict(phase_patterns(phase));patterns.update({f'checker_{v}':checker_phase(v) for v in (0,64,128,192)})
    warp=cv2.getPerspectiveTransform(np.float32(m.corners(Path(hw['coordinates']))),np.float32([[0,0],[477,0],[477,477],[0,477]]))
    nominal=torch.tensor([1.,1.,0.,0.])
    for anchor in anchors:
        amp=np.asarray(Image.open(anchor['canonical'])).astype(np.float32)/255
        tags=['nominal_before','flat','half_code','global_offset128']+([f'checker_{v}' for v in (0,64,128,192)] if anchor in anchors[:2] else [])
        for tag in tags:
            row=bykey[tag,anchor['index']]
            assert row['sample_id']==anchor['sample_id'] and row['split']==anchor['split'] and row['canonical_SHA256']==anchor['canonical_SHA256']
            for path,key in (('raw','raw_SHA256'),('crop','crop_SHA256'),('phase','phase_SHA256')):assert m.sha(row[path])==row[key]
            raw=np.asarray(Image.open(row['raw']));crop=np.asarray(Image.open(row['crop']))
            assert np.array_equal(crop,logical(raw,warp,65535,hw['orientation']))
            assert np.array_equal(np.asarray(Image.open(row['phase'])),m.phase_native(patterns[tag]))
            with torch.no_grad():pred=response_intensity(torch.from_numpy(amp)[None],torch.from_numpy(patterns[tag].astype(np.float32)/256),model.propagator,nominal)[0]
            observed=torch.from_numpy(crop.astype(np.float32));pred=pred/pred.mean().clamp_min(1e-8);observed=observed/observed.mean().clamp_min(1e-8)
            pack.append(dict(index=anchor['index'],sample_id=anchor['sample_id'],split=anchor['split'],condition=tag,predicted=pred,observed=observed))
        m.save(out/'progress.json',dict(state='replaying_TRAIN_response',TRAIN_anchors=len(pack)//4,response_pairs=len(pack),TEST_used=False,elapsed_s=time.monotonic()-start))
    assert len(pack)==168
    flat=[v for v in pack if v['condition']=='flat'];fitrows=[v for v in flat if v['split']=='fit'];heldrows=[v for v in flat if v['split']=='held_TRAIN'];assert len(fitrows)==len(heldrows)==20
    def pooled(rows,key,size):return F.adaptive_avg_pool2d(torch.stack([v[key] for v in rows])[:,None],(size,size))[:,0].numpy()
    pp=pooled(fitrows,'predicted',8);oo=pooled(fitrows,'observed',8);gain,bg=fit_gain_bias(pp,oo,.05)
    residual_fit=(oo-pp).mean(0);residual_held=(pooled(heldrows,'observed',8)-pooled(heldrows,'predicted',8)).mean(0)
    residual_PCC=m.correlation(residual_fit,residual_held)
    maps=out/'TRAIN_flat_gain_background.npz';np.savez_compressed(maps,gain=gain.astype(np.float32),background=bg.astype(np.float32),fit_TRAIN_ids=np.array([v['sample_id'] for v in fitrows]),grid=8,ridge=.05)
    m.save(out/'protocol.json',dict(source_git=m.git_head(),source_SHA256=m.sha(__file__),profile_SHA256=m.sha(profile),policy=c,probe_report_SHA256=m.sha(probes/'report.json'),response_audit_SHA256=m.sha(audit),
        fit_condition='flat',TRAIN_fit_anchors=20,TRAIN_held_anchors=20,derivative_examples=0,classification_labels_used=False,TEST_used=False,VALID=0,new_CCD=0,SDK_used=False,training_forward_only=True,hardware_unchanged=True,
        response_equation='gain8(x)*nominal_intensity/mean(nominal_intensity)+background8(x); fixed bilinear interpolation, no image warp',nominal_phase_only=True,unadmitted_prior_nuisance_fits_used=False))
    gt=torch.from_numpy(gain.astype(np.float32));bt=torch.from_numpy(bg.astype(np.float32))
    def assess(rows,corrected):
        losses=[];cc=[]
        for begin in range(0,len(rows),4):
            chunk=rows[begin:begin+4];pred=torch.stack([v['predicted'] for v in chunk]);obs=torch.stack([v['observed'] for v in chunk])
            if corrected:pred=apply_map(pred,gt,bt)
            losses.append(float(spatial.spatial_loss(pred,obs))*len(chunk))
            a=F.adaptive_avg_pool2d(pred[:,None],(16,16))[:,0].numpy();b=F.adaptive_avg_pool2d(obs[:,None],(16,16))[:,0].numpy()
            cc.extend(m.correlation(x,y) for x,y in zip(a,b))
        return dict(loss=sum(losses)/len(rows),coarse_PCC=float(np.mean(cc)),response_pairs=len(rows))
    groups={'fit_flat':fitrows,'held_flat':heldrows,'held_unseen_phase':[v for v in pack if v['split']=='held_TRAIN' and v['condition'] in ('nominal_before','half_code','global_offset128')]}
    groups.update({'held_'+tag:[v for v in pack if v['split']=='held_TRAIN' and v['condition']==tag] for tag in patterns if tag!='nominal_after'})
    baseline={k:assess(v,False) for k,v in groups.items()};fitted={k:assess(v,True) for k,v in groups.items()}
    admitted=residual_PCC>=.8 and fitted['fit_flat']['loss']<baseline['fit_flat']['loss']
    for tag in ('held_flat','held_unseen_phase','held_nominal_before'):
        admitted=admitted and fitted[tag]['loss']<=baseline[tag]['loss']*.95 and fitted[tag]['coarse_PCC']>=baseline[tag]['coarse_PCC']+.03
    admitted=admitted and all(fitted['held_'+tag]['coarse_PCC']>=baseline['held_'+tag]['coarse_PCC'] for tag in ('half_code','global_offset128'))
    assert protected=={q:m.sha(q) for q in protected} and not m.LOCK.exists()
    report=dict(state='complete',response_admitted=bool(admitted),hypothesis='sample-shared spatial photometric mismatch; NOT unique causal identification',shared_flat_residual_PCC=residual_PCC,
        baseline=baseline,fitted=fitted,map_SHA256=m.sha(maps),gain_range=[float(gain.min()),float(gain.max())],background_range=[float(bg.min()),float(bg.max())],protocol_SHA256=m.sha(out/'protocol.json'),
        TRAIN_unique=40,fit_flat_pairs=20,held_flat_pairs=20,held_unseen_phase_pairs=60,classification_labels_used=False,TEST_used=False,VALID=0,new_CCD=0,SDK_used=False,SDK_released=True,model_checkpoint_written=False,
        training_forward_only=True,inference_structure_unchanged=True,LUT_changed=False,coordinates_changed=False,exposure_changed=False,unique_physical_cause_identified=False,elapsed_s=time.monotonic()-start)
    m.save(out/'report.json',report);print(json.dumps(report),flush=True)


def selftest():
    rng=np.random.default_rng(73);p=rng.uniform(0,3,(40,8,8));g=rng.uniform(.1,3,(8,8));b=rng.uniform(.1,2,(8,8));o=p*g+b
    gg,bb=fit_gain_bias(p,o,1e-8);assert np.max(abs(gg-g))<1e-6 and np.max(abs(bb-b))<1e-6
    a,c=fit_gain_bias(p,100+100*p,.05);assert (a>=.05).all() and (a<=4).all() and (c>=0).all() and (c<=4).all()
    identity,zero=fit_gain_bias(p,p,.05);assert np.allclose(identity,1) and np.allclose(zero,0)
    print(json.dumps(dict(passed=True,synthetic_known_map_maxabs=float(max(np.max(abs(gg-g)),np.max(abs(bb-b)))),identity_exact=True,box_constraints=True,SDK_used=False,TEST_used=False)))


def torch_selftest():
    import torch
    torch.set_num_threads(2);torch.manual_seed(73)
    x=torch.rand(2,478,478,requires_grad=True);g=torch.ones(8,8);b=torch.zeros(8,8)
    y=apply_map(x,g,b);expected=x/x.mean((-2,-1),keepdim=True)
    assert torch.allclose(y,expected,atol=1e-7,rtol=1e-7)
    weight=torch.linspace(0,1,478)[None,None,:];(y*weight).mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all() and float(x.grad.norm())>0
    print(json.dumps(dict(passed=True,identity_relative_field=True,input_gradient_finite=True,input_gradient_norm=float(x.grad.norm()),new_inference_layers=0,SDK_used=False,TEST_used=False)))


if __name__=='__main__':
    q=argparse.ArgumentParser();q.add_argument('--selftest',action='store_true');q.add_argument('--torch-selftest',action='store_true');q.add_argument('--profile',type=Path);q.add_argument('--out',type=Path);a=q.parse_args()
    if a.selftest:selftest()
    elif a.torch_selftest:torch_selftest()
    else:run(a.profile,a.out)
