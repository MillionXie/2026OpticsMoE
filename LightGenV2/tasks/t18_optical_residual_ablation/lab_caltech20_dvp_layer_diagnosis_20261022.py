"""Read-only layer-local response and frozen-feature diagnosis of audited L4.

All deployment forwards return the existing measured intensity. Nominal local
responses use the SAME recorded input and phase, on fixed TRAIN identities only.
Linear probes are diagnostic, never installed as a new inference layer/model.
"""
import argparse,json,os,time,traceback
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image
from caltech20_alt246_biasfree_model import features,rms,energies
from lab_caltech20_rich_layerwise_20261018 import MeasuredPrefix

NAMES={'router':'router','0':'E1','1':'G1','2':'E2','3':'G2'}


def summaries(v):
    v=np.asarray(v,dtype=np.float64)
    assert np.isfinite(v).all() and v.size
    return dict(n=int(v.size),mean=float(v.mean()),min=float(v.min()),p10=float(np.quantile(v,.1)),median=float(np.median(v)),p90=float(np.quantile(v,.9)),max=float(v.max()))


def correlation(a,b):
    a=a.flatten(1).double();b=b.flatten(1).double()
    a=a-a.mean(1,keepdim=True);b=b-b.mean(1,keepdim=True)
    return (a*b).sum(1)/(a.norm(dim=1)*b.norm(dim=1)).clamp_min(1e-12)


def local_response(nominal,observed,model,target):
    """Descriptive statistics, not a fitted correction or calibration."""
    a=F.adaptive_avg_pool2d(nominal[:,None],(64,64))[:,0]
    b=F.adaptive_avg_pool2d(observed[:,None],(64,64))[:,0]
    result={'coarse64_identity_correlation':correlation(a,b).cpu().numpy()}
    for flip,label in (([-1],'flip_h'),([-2],'flip_v'),([-2,-1],'flip_hv')):
        result['coarse64_'+label+'_correlation']=correlation(a,b.flip(flip)).cpu().numpy()
    result['coarse64_transpose_correlation']=correlation(a,b.transpose(-1,-2)).cpu().numpy()
    aa=nominal/nominal.mean((-2,-1),keepdim=True).clamp_min(1e-8)
    bb=observed/observed.mean((-2,-1),keepdim=True).clamp_min(1e-8)
    result['mean_normalized_pixel_RMSE']=((aa-bb).square().mean((-2,-1)).sqrt()).cpu().numpy()
    oa=model.nonlinear(nominal,target);ob=model.nonlinear(observed,target)
    result['OEO_pixel_cosine']=F.cosine_similarity(oa.flatten(1),ob.flatten(1),dim=1).cpu().numpy()
    result['OEO_pool16_cosine']=F.cosine_similarity(features(oa),features(ob),dim=1).cpu().numpy()
    return result


def frozen_geometry(train_x,labels,ids,test_x,test_labels,penalty=.1):
    """Same-source-excluded NN and fixed TRAIN-only ridge diagnostic probe.

    Exact leave-source-group-out residuals exclude original and fixed augmented
    copies together. TEST never supplies geometry, ridge strength or targets.
    """
    x=np.asarray(train_x,np.float64);y=np.asarray(labels,np.int64);tx=np.asarray(test_x,np.float64)
    n,d=x.shape;assert len(y)==n and tx.shape[1]==d and penalty>0
    base=np.asarray([str(v).split('::TRAIN_shift')[0] for v in ids]);exclude=base[:,None]==base[None,:]
    unit=x/np.linalg.norm(x,axis=1,keepdims=True).clip(1e-12);similarity=unit@unit.T;similarity[exclude]=-np.inf
    nearest=similarity.argmax(1);centers=np.stack([x[y==k].mean(0) for k in range(20)])
    within=np.square(x-centers[y]).sum(1).mean();between=np.square(centers[y]-x.mean(0)).sum(1).mean()
    centered=x-x.mean(0);spectrum=np.linalg.eigvalsh(centered.T@centered).clip(0)[::-1];cum=np.cumsum(spectrum)/max(spectrum.sum(),1e-12)
    # Use a single common scaling, not per-column amplification of tiny axes.
    scale=max(float(np.sqrt(np.square(centered).mean())),1e-6);design=centered/scale;test=(tx-x.mean(0))/scale
    target=np.eye(20)[y];ym=target.mean(0);gram=design.T@design;lam=penalty*np.trace(gram)/max(d,1)+1e-8
    inverse=np.linalg.solve(gram+lam*np.eye(d),np.eye(d));w=inverse@design.T@(target-ym)
    fitted=design@w+ym;test_prediction=test@w+ym;loso=np.empty_like(fitted)
    for group in np.unique(base):
        ix=np.flatnonzero(base==group);h=design[ix]@inverse@design[ix].T+1/n
        residual=np.linalg.solve(np.eye(len(ix))-h,target[ix]-fitted[ix]);loso[ix]=target[ix]-residual
    pred=test_prediction.argmax(1);cm=np.bincount(np.asarray(test_labels)*20+pred,minlength=400).reshape(20,20)
    return dict(TRAIN_samples=n,dimensions=d,same_source_excluded=True,TRAIN_nearest_same_class_correct=int((y[nearest]==y).sum()),between_within_scatter=float(between/max(within,1e-12)),rank95=int(np.searchsorted(cum,.95)+1),rank99=int(np.searchsorted(cum,.99)+1),TRAIN_probe_correct=int((fitted.argmax(1)==y).sum()),TRAIN_leave_source_out_probe_correct=int((loso.argmax(1)==y).sum()),TEST_diagnostic_probe_correct=int((pred==test_labels).sum()),TEST_diagnostic_confusion=cm.tolist(),probe_regularization=float(lam),probe_penalty_fraction=penalty,probe_fitted_from_TRAIN_only=True,diagnostic_not_deployment_accuracy=True)


class ResponseObserver(MeasuredPrefix):
    def capture(self,amplitude,stage,weights=None,callback=None,ids=None):
        assert callback is None and stage in self.measured
        observed=self.measured[stage].float()
        field,canonical,phase=self.physical_fields(amplitude,stage,weights)
        self.observe(stage,amplitude,weights,field,canonical,phase,observed)
        return observed


def relation_summary(values,target,labels,ids):
    """TRAIN-only global check of the relation hypothesis, excluding source mates."""
    a=np.asarray(values,np.float64);b=np.asarray(target,np.float64)
    a=a/np.linalg.norm(a,axis=1,keepdims=True).clip(1e-12);b=b/np.linalg.norm(b,axis=1,keepdims=True).clip(1e-12)
    source=np.asarray([str(v).split('::TRAIN_shift')[0] for v in ids]);valid=source[:,None]!=source[None,:]
    ga=a@a.T;gb=b@b.T;same=np.asarray(labels)[:,None]==np.asarray(labels)[None,:]
    near=ga.copy();near[~valid]=-np.inf
    return dict(TRAIN_pair_relation_MSE=float(np.square(ga[valid]-gb[valid]).mean()),TRAIN_nearest_same_class_correct=int((labels[near.argmax(1)]==labels).sum()),TRAIN_sameclass_cosine=float(ga[valid&same].mean()),TRAIN_otherclass_cosine=float(ga[valid&~same].mean()),same_source_excluded=True)


def followup(sequence,fit,out):
    """After fit returns: did the TRAIN relation objective change actual features?"""
    import lab_caltech20_dvp_layerwise_20261019 as m
    from lab_caltech20_rich_tail_refine_20261018 import tail
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not out.exists();torch.set_num_threads(2)
    protocol=m.read(fit/'protocol.json');cfg=protocol['config'];report=m.read(fit/'report.json');audit=m.read(fit/'independent_audit.json')
    assert report['state']=='complete' and audit['passed'] and audit['report_SHA256']==m.sha(fit/'report.json')
    assert cfg['TRAIN_geometry_feature_key']=='G2_electronic' and report['strict_original_CPU'] and report['full_physical']
    source=Path(cfg['TRAIN_geometry_features']);assert m.sha(source)==cfg['TRAIN_geometry_features_SHA256']
    with np.load(source) as z:target=z['G2_electronic'].copy();old=z['G2_fused'].copy();labels=z['labels'].copy();ids=z['ids'].copy()
    pt=fit/'last.pt';assert m.sha(pt)==report['results']['last']['PT_SHA256'];model,_=m.model_for(pt,m.sha(pt),4,sequence/'L4_response_20261020/source_reference.ts')
    original=torch.load(cfg['parent_PT'],map_location='cpu',weights_only=False)['model'];state=m.state_cpu(model)
    assert all(torch.equal(v,state[k]) for k,v in original.items() if k not in protocol['allowed_keys'])
    inputs=Path(cfg['terminal_input_producer'])/'train_last_input.npy';ccd=Path(cfg['final_capture'])/'train/ccd.npy'
    assert m.sha(ccd)==protocol['prefix_SHA256']['3/train'];xx=np.load(inputs,mmap_mode='r');oo=np.load(ccd,mmap_mode='r')
    with np.load(fit/'last_train_predictions.npz') as z:assert np.array_equal(z['ids'],ids) and np.array_equal(z['labels'],labels);expected=z['probabilities'].copy()
    parts=[];hook=model.linear.register_forward_pre_hook(lambda module,args:parts.append(args[0].detach().cpu().numpy().copy()))
    try:
        with torch.inference_mode():
            for begin in range(0,1600,8):
                x=torch.from_numpy(np.asarray(xx[begin:begin+8]).copy());o=model.nonlinear(torch.from_numpy(np.asarray(oo[begin:begin+8]).copy()),'global')
                p=tail(model.electronic[3],model.linear,model.effective_alpha(3),x,o).softmax(1).numpy()
                assert np.max(np.abs(p-expected[begin:begin+8]))<=3e-6 and np.array_equal(p.argmax(1),expected[begin:begin+8].argmax(1))
    finally:hook.remove()
    new=np.concatenate(parts);result=dict(source_git=m.git_head(),source_SHA256=m.sha(__file__),TRAIN_only=True,TEST_used=False,TRAIN=1600,feature_target_SHA256=m.sha(source),train_last_input_SHA256=m.sha(inputs),TRAIN_CCD_SHA256=m.sha(ccd),fit_report_SHA256=m.sha(fit/'report.json'),last_PT_SHA256=m.sha(pt),last_probability_matches_full_native_audit=True,nonterminal_parameters_frozen=True,parent=relation_summary(old,target,labels,ids),last=relation_summary(new,target,labels,ids),frozen_electronic_target=relation_summary(target,target,labels,ids),not_a_performance_or_causal_claim=True,SDK_used=False,GPU_used=False)
    m.save(out,result);return result


def TRAIN_loss_balance(sequence,fit,out):
    """Fixed40 TRAIN gradient diagnostic, no update/TEST/example selection."""
    import lab_caltech20_dvp_layerwise_20261019 as m
    from lab_caltech20_rich_tail_refine_20261018 import tail
    from lab_caltech20_dvp_postCCD_SAM_20261021 import relational_TRAIN_geometry
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not out.exists();torch.set_num_threads(2)
    cfg=m.read(fit/'protocol.json')['config'];parent=Path(cfg['parent_PT']);assert m.sha(parent)==cfg['parent_SHA256']
    model,_=m.model_for(parent,cfg['parent_SHA256'],4,sequence/'L4_response_20261020/source_reference.ts');original=m.state_cpu(model)
    model.electronic[3].local.weight.requires_grad_(True);model.fusion_logits[3].requires_grad_(True);params=[model.electronic[3].local.weight,model.fusion_logits[3]]
    source=Path(cfg['TRAIN_geometry_features']);assert m.sha(source)==cfg['TRAIN_geometry_features_SHA256']
    with np.load(source) as z:target=torch.from_numpy(z[cfg['TRAIN_geometry_feature_key']][:40].copy());labels=torch.from_numpy(z['labels'][:40].copy());ids=z['ids'][:40].copy()
    assert torch.bincount(labels,minlength=20).tolist()==[2]*20
    x=np.load(Path(cfg['terminal_input_producer'])/'train_last_input.npy',mmap_mode='r');ccd=np.load(Path(cfg['final_capture'])/'train/ccd.npy',mmap_mode='r')
    cw=torch.tensor(m.read(fit/'TRAIN_emphasis_plan.json')['TRAIN_class_weights']);gradients={'CE':torch.zeros(10,dtype=torch.float64),'relation':torch.zeros(10,dtype=torch.float64)};losses={'CE':0.,'relation':0.};held={}
    hook=model.linear.register_forward_pre_hook(lambda module,args:held.update(features=args[0]))
    try:
        for begin in range(0,40,8):
            v=torch.from_numpy(np.asarray(x[begin:begin+8]).copy());o=model.nonlinear(torch.from_numpy(np.asarray(ccd[begin:begin+8]).copy()),'global')
            logits=tail(model.electronic[3],model.linear,model.effective_alpha(3),v,o)
            ce=F.cross_entropy(logits,labels[begin:begin+8],weight=cw,label_smoothing=cfg['tail']['label_smoothing']);relation=relational_TRAIN_geometry(held['features'],target[begin:begin+8])
            for name,value in (('CE',ce),('relation',relation)):
                g=torch.autograd.grad(value,params,retain_graph=True);gradients[name]+=torch.cat([z.detach().double().flatten() for z in g])/5;losses[name]+=float(value.detach())/5
    finally:hook.remove()
    assert all(torch.equal(v,m.state_cpu(model)[k]) for k,v in original.items())
    lr=torch.tensor([cfg['tail']['conv_lr']]*9+[cfg['tail']['alpha_lr']],dtype=torch.float64);a=gradients['CE']*lr;b=gradients['relation']*lr
    assert float(b.norm())>1e-12
    result=dict(source_git=m.git_head(),source_SHA256=m.sha(__file__),TRAIN_only=True,TRAIN_samples=40,TRAIN_each_class=2,TRAIN_ids=ids.tolist(),TEST_used=False,parameter_updates=0,all_parameters_unchanged=True,parent_SHA256=cfg['parent_SHA256'],target_SHA256=m.sha(source),TRAIN_losses=losses,gradients={k:v.tolist() for k,v in gradients.items()},CE_lr_scaled_norm=float(a.norm()),relation_lr_scaled_norm=float(b.norm()),coefficient_equalizing_lr_scaled_norm=float(a.norm()/b.norm()),gradient_cosine=float((a@b)/(a.norm()*b.norm()).clamp_min(1e-20)),current_geometry_coefficient=cfg['tail']['TRAIN_relational_geometry_weight'],SDK_used=False,GPU_used=False)
    m.save(out,result);return result


def run(a):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    import lab_caltech20_dvp_layerwise_20261019 as m
    import lab_caltech20_dvp_closedloop_20261021 as loop
    torch.set_num_threads(2);torch.manual_seed(73)
    cfg=m.read(a.profile);assert cfg['schema']=='L4_readonly_stage_response_v1' and cfg['TRAIN_nominal_samples']==200
    parent_cfg=m.read(Path(__file__).with_name(cfg['parent_profile']));parent=Path(parent_cfg['parent_PT'])
    assert m.sha(parent)==parent_cfg['parent_SHA256'] and not a.out.exists();a.out.mkdir(parents=True)
    data,dm=m.data_for(a.sequence);assert dm['sha256']==parent_cfg['data_SHA256']
    assert m.session_hardware_digest(a.sequence)==loop.HARDWARE
    session=a.sequence/'L4_response_20261020';base,_=m.model_for(parent,parent_cfg['parent_SHA256'],4,session/'source_reference.ts')
    state=m.state_cpu(base);model=ResponseObserver(4).eval();model.load_state_dict(state);model.propagator.transfer.copy_(base.propagator.transfer);model.requires_grad_(False)
    paths={k:session/'capture'/k for k in ('router','0','1')};paths.update({'2':a.sequence/'TRAIN_restored_same_mask_pair_20261021/L4/capture','3':Path(parent_cfg['final_capture'])})
    fields={};identities={};reports={};signals={};audits={}
    for key,path in paths.items():
        r=m.read(path/'report.json');assert r['state']=='complete' and r['physical_ccd']==2011 and (r.get('sdk_released') or r.get('SDK_released'))
        assert r['data_SHA256']==dm['sha256'] and r['phase_SHA256']==m.sha(path/'phase.bmp')
        auditfile=path/('independent_audit.json' if (path/'independent_audit.json').exists() else 'independent_capture_audit.json')
        audit=m.read(auditfile);assert audit['passed'] and audit['report_SHA256']==m.sha(path/'report.json');audits[key]=dict(path=str(auditfile),SHA256=m.sha(auditfile))
        reports[key]=r;fields[key]={};signals[key]={}
        for role,n in (('train',1600),('test',411)):
            p=path/role/'ccd.npy';assert m.sha(p)==r['arrays'][role]['SHA256'];identities[key+'/'+role]=m.sha(p);fields[key][role]=np.load(p,mmap_mode='r')
            assert fields[key][role].shape==(n,478,478);rows=[]
            for i in range(n):
                q=m.read(path/role/f'{i:06d}.json');assert q['sample_id']==str(data[role][2][i]) and q['index']==i and q['role']==role and q['phase_SHA256']==r['phase_SHA256']
                assert q['data_SHA256']==dm['sha256'] and q['PT_SHA256']==r['PT_SHA256'];rows.append(q)
            signals[key][role]={k:summaries([v[k] for v in rows]) for k in ('p99','mean','saturation_fraction')}
            signals[key][role]['p99_below15']=sum(q['p99']<15 for q in rows);signals[key][role]['above15pct_clipped']=sum(q['saturation_fraction']>.15 for q in rows)
    producer=Path(parent_cfg['feature_producer']);assert identities==m.read(producer/'protocol.json')['prefix_SHA256']
    protocol=dict(source_git=m.git_head(),source_SHA256=m.sha(__file__),profile=cfg,profile_SHA256=m.sha(a.profile),parent_PT_SHA256=m.sha(parent),data_SHA256=dm['sha256'],hardware_SHA256=loop.HARDWARE,prefix_SHA256=identities,audits=audits,TRAIN=1600,TEST=411,VALID=0,TRAIN_nominal_samples=200,nominal_TRAIN_only=True,all_model_parameters_frozen=True,all_deployment_forward_values_are_observed=True,SDK_used=False,GPU_used=False,recurring=False)
    m.save(a.out/'protocol.json',protocol);banks={};probs={};response={k:{} for k in paths};matched={k:0 for k in paths};phase_checked={};started=time.monotonic()
    context={}
    def observe(key,amplitude,weights,field,canonical,phase,actual):
        role=context['role'];begin=context['begin'];valid=context['valid'];name=NAMES[key]
        def add(label,x):banks[role].setdefault(label,[]).append(x[:valid].detach().cpu().numpy().copy())
        add(name+'_incident',features(canonical));add(name+'_rawCCD',features(actual))
        if key=='router':
            energy=energies(actual,model.router_masks);center=energy-energy.mean(1,keepdim=True);dense=(center/(center.square().mean(1,keepdim=True)+1e-8).sqrt()/2).softmax(1)
            route=dense/dense.square().sum(1,keepdim=True).sqrt().clamp_min(1e-8);add('router_weights',route)
        else:
            k=int(key);target=model.stages[k+1] if k<3 else 'global';size=478 if target=='global' else 224
            o=model.nonlinear(actual,target);e=model.electronic[k](amplitude,size);re,ro=rms(e),rms(o);mixed=(1-model.effective_alpha(k))*e/re+model.effective_alpha(k)*o/ro;fused=re*mixed/rms(mixed)
            add(name+'_OEO',features(o));add(name+'_electronic',features(e));add(name+'_fused',features(fused))
        if key not in phase_checked:
            pixels=(phase*256).to(torch.uint8).cpu().numpy();assert np.array_equal(m.phase_native(pixels),np.asarray(Image.open(paths[key]/'phase.bmp')))
            phase_checked[key]=True
        # All canonical inputs, TRAIN and TEST, must match the saved stimulus.
        for j in range(valid):
            i=begin+j;p=paths[key]/role/f'{i:06d}_input.png';q=m.read(paths[key]/role/f'{i:06d}.json')
            assert m.sha(p)==q['canonical_SHA256'];gray=np.rint(canonical[j].cpu().numpy().clip(0,1)*255).astype(np.uint8)
            assert np.array_equal(gray,np.asarray(Image.open(p))),('Captured input identity mismatch',key,role,i)
            matched[key]+=1
        if role=='train' and begin<cfg['TRAIN_nominal_samples']:
            nominal=model.propagator(field).abs().square()[:,20:498,20:498];target='expert' if key=='router' else model.stages[int(key)+1] if int(key)<3 else 'global'
            for stat,value in local_response(nominal[:valid],actual[:valid],model,target).items():response[key].setdefault(stat,[]).append(value)
    model.observe=observe
    with torch.inference_mode():
        for role,n in (('train',1600),('test',411)):
            banks[role]={};pieces=[]
            with np.load(parent.parent/('best_'+role+'_predictions.npz')) as z:assert np.array_equal(z['ids'],data[role][2]);expected=z['probabilities'].copy()
            for begin in range(0,n,8):
                images,valid,end=m.padded(data,role,begin);context.update(role=role,begin=begin,valid=valid)
                model.measured={}
                for key,byrole in fields.items():
                    v=np.asarray(byrole[role][begin:end]).copy()
                    if valid<8:v=np.concatenate((v,np.zeros((8-valid,478,478),v.dtype)))
                    model.measured[key]=torch.from_numpy(v)
                p=model(m.encode_images(images))[0][:valid].cpu().numpy();assert np.max(np.abs(p-expected[begin:end]))<=3e-6 and np.array_equal(p.argmax(1),expected[begin:end].argmax(1));pieces.append(p)
                m.save(a.out/'progress.json',dict(state='frozen_actual_prefix_response',role=role,samples=end,total=n,elapsed_s=time.monotonic()-started,phase_checked=list(phase_checked),matched_inputs=matched))
                assert time.monotonic()-started<cfg['maximum_seconds'],'Finite diagnosis budget exhausted; preserve partial evidence'
            probs[role]=np.concatenate(pieces);banks[role]={k:np.concatenate(v) for k,v in banks[role].items()}
            np.savez_compressed(a.out/(role+'_frozen_stage_features.npz'),ids=data[role][2],labels=data[role][1],probabilities=probs[role],**banks[role])
    assert matched=={k:2011 for k in paths} and set(phase_checked)==set(paths)
    assert all(torch.equal(v,m.state_cpu(model)[k]) for k,v in state.items())
    metrics={}
    for name,x in banks['train'].items():
        metrics[name]=frozen_geometry(x,data['train'][1],data['train'][2],banks['test'][name],data['test'][1],cfg['probe_penalty_fraction'])
        m.save(a.out/'progress.json',dict(state='TRAIN_only_geometry_and_probe',feature=name,completed=len(metrics),total=len(banks['train']),elapsed_s=time.monotonic()-started))
    responses={NAMES[k]:{name:summaries(np.concatenate(values)) for name,values in row.items()} for k,row in response.items()}
    report=dict(state='complete',deployment_unchanged_correct=int((probs['test'].argmax(1)==data['test'][1]).sum()),expected_deployment_correct=285,canonical_inputs_matched=matched,phase_exact=True,all_parameters_bitwise_frozen=True,original_native_probability_parity=True,signals={NAMES[k]:v for k,v in signals.items()},local_response_TRAIN200=responses,frozen_feature_diagnostics=metrics,features_SHA256={role:m.sha(a.out/(role+'_frozen_stage_features.npz')) for role in ('train','test')},protocol_SHA256=m.sha(a.out/'protocol.json'),elapsed_s=time.monotonic()-started,TEST_diagnostic_development=True,TEST_fitted_geometry=False,TEST_gradient=False,SDK_used=False,GPU_used=False,new_inference_layers=0,interpretation='Comparisons are descriptive local evidence, not a causal proof of a defective device/layer. Probes are never deployed, and local nominal substitution is never scored as complete physical inference. Original and shifted duplicates are excluded together in TRAIN probes. No transform, LUT, exposure, coordinate, mask or upstream electronics is changed.')
    assert report['deployment_unchanged_correct']==285;m.save(a.out/'report.json',report);m.save(a.out/'independent_audit.json',dict(passed=True,report_SHA256=m.sha(a.out/'report.json'),protocol_SHA256=m.sha(a.out/'protocol.json'),all2011_inputs_per_stage_match=True,all_states_frozen=True,native_all_roles_probability_parity=True));print(json.dumps(report),flush=True)


def selftest():
    torch.set_num_threads(2);torch.manual_seed(73)
    from lab_caltech20_rich_model_20261018 import DirectAlphaHybridMoE
    base=DirectAlphaHybridMoE(4).eval()
    with torch.no_grad():
        for p in base.fusion_logits:p.fill_(.8)
        x=torch.rand(2,224,224);expected=base(x)[0];measured={k:v['intensity'].clone() for k,v in base.last.items()};model=ResponseObserver(4).eval();model.load_state_dict(base.state_dict());model.propagator.transfer.copy_(base.propagator.transfer);model.measured=measured;called=[]
        model.observe=lambda stage,*args:called.append(stage);actual=model(x)[0];assert torch.equal(actual,expected) and called==['router','0','1','2','3']
        r=local_response(measured['3'],measured['3'],model,'global');assert np.allclose(r['coarse64_identity_correlation'],1) and np.max(r['mean_normalized_pixel_RMSE'])==0
    rng=np.random.default_rng(73);y=np.arange(80)%20;ids=np.array([str(i) for i in range(80)]);xx=np.eye(20)[y]+rng.normal(0,.01,(80,20));result=frozen_geometry(xx,y,ids,xx,y)
    assert result['TRAIN_leave_source_out_probe_correct']==80 and result['TEST_diagnostic_probe_correct']==80 and result['TRAIN_nearest_same_class_correct']==80
    print(json.dumps(dict(passed=True,all_measured_forward_bitwise_equal=True,local_response_identity=True,TRAIN_only_probe=True,SDK_used=False)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--sequence',type=Path);p.add_argument('--profile',type=Path);p.add_argument('--out',type=Path);p.add_argument('--fit',type=Path);p.add_argument('--balance',action='store_true');p.add_argument('--selftest',action='store_true');a=p.parse_args()
    if a.selftest:selftest();raise SystemExit(0)
    if a.fit:print(json.dumps((TRAIN_loss_balance if a.balance else followup)(a.sequence,a.fit,a.out)));raise SystemExit(0)
    try:run(a)
    except BaseException as e:
        if a.out.exists():(a.out/'failure.json').write_text(json.dumps(dict(error=repr(e),traceback=traceback.format_exc(),automatic_retry=False),indent=2))
        raise
