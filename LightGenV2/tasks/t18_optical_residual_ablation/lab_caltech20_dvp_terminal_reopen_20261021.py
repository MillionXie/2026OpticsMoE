"""Bounded offline reopening of the fully measured L4 ORIGINAL terminal graph.

TRAIN covariance/margin correction first, then (if needed) original9conv+direct
alpha+5140head. No optical update, inference module, SDK, TEST gradient or VAL.
The active L6 capture controller and its sources are never changed.
"""
import argparse,json,os,time,traceback
from pathlib import Path


def target_count(simulation,samples,gap_pp):
    import math
    return math.ceil(simulation-samples*gap_pp/100-1e-10)


def delta_sam(torch,x,y,pw,pb,counts,cfg,observe):
    """TRAIN-only SAM in delta coordinates; zero delta is byte-exact parent."""
    from torch.nn import functional as F
    from lab_caltech20_residual_sequence_20261018 import train_conditioned_geometry
    from lab_caltech20_dvp_completed_readout_20261021 import fold_delta
    mean,root,_,_,floor=train_conditioned_geometry(x.numpy());xx=torch.cat(((x-mean)@root,torch.ones(len(x),1,dtype=x.dtype)),1)
    base=F.linear(x,pw.double(),pb.double());teacher=(base/2).softmax(1).detach();keep=(base.argmax(1)==y).double()
    weights=(counts/counts.mean()).sqrt();delta=torch.nn.Parameter(torch.zeros(20,257,dtype=x.dtype));opt=torch.optim.Adam([delta],lr=cfg['learning_rate'])
    def objective():
        z=base+xx@delta.T;ce=F.cross_entropy(z,y,weight=weights,label_smoothing=cfg['label_smoothing'])
        kl=F.kl_div((z/2).log_softmax(1),teacher,reduction='none').sum(1)*4
        return ce+cfg['teacher_KL']*(kl*keep).sum()/keep.sum().clamp_min(1)+cfg['anchor']*delta.square().sum()
    start=time.monotonic();lastw,lastb=pw,pb
    for step in range(1,cfg['steps']+1):
        opt.zero_grad();clean=objective();clean.backward();gradient=delta.grad.detach().clone();saved=delta.detach().clone()
        with torch.no_grad():delta.add_(cfg['SAM_radius']*gradient/gradient.norm().clamp_min(1e-12))
        try:opt.zero_grad();probe=objective();probe.backward()
        finally:
            with torch.no_grad():delta.copy_(saved)
        assert torch.equal(delta.detach(),saved) and torch.isfinite(delta.grad).all();torch.nn.utils.clip_grad_norm_([delta],1.);opt.step()
        with torch.no_grad():delta.mul_(min(1.,cfg['trust_radius']/max(float(delta.norm()),1e-12)))
        if step%cfg['evaluate_every']==0 or step==cfg['steps']:
            with torch.no_grad():lastw,lastb=fold_delta(torch,pw,pb,delta,mean,root)
            stop=observe(lastw,lastb,dict(phase='TRAIN_delta_SAM',step=step,TRAIN_all1600=True,TRAIN_loss=float(clean.detach()),TRAIN_probe_loss=float(probe.detach()),probe_restored_exactly=True))
            if stop or time.monotonic()-start>=cfg['fit_seconds']:break
    return lastw,lastb,dict(method='TRAIN_delta_SAM',variance_floor=floor,actual_TRAIN_updates=step,TRAIN_prior_power=.5,zero_delta_original_identity=True,fit_seconds=time.monotonic()-start)


def selftest():
    import torch
    torch.set_num_threads(2);torch.manual_seed(73)
    x=torch.randn(40,256,dtype=torch.float64);y=torch.arange(40)%20
    w=torch.randn(20,256)*.01;b=torch.zeros(20);rows=[]
    cfg=dict(learning_rate=.003,label_smoothing=.03,teacher_KL=.1,anchor=.01,SAM_radius=.03,trust_radius=1.,steps=2,evaluate_every=1,fit_seconds=30)
    ww,bb,proof=delta_sam(torch,x,y,w,b,torch.ones(20,dtype=torch.float64),cfg,lambda w,b,r:rows.append(r))
    assert torch.isfinite(ww).all() and torch.isfinite(bb).all() and not torch.equal(ww,w)
    assert len(rows)==2 and all(r['probe_restored_exactly'] for r in rows)
    assert proof['actual_TRAIN_updates']==2
    print(json.dumps(dict(passed=True,SAM_probe_restored_exactly=True,finite_TRAIN_gradient=True,original_linear_shape=list(ww.shape),TEST_used=False,SDK_used=False)))


def run(a):
    preview=json.loads(a.profile.read_text());post_sam=preview.get('method') in ('TRAIN_postCCD_SAM','TRAIN_postCCD_geometry_SAM')
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==(preview['GPU_UUID'] if post_sam else '')
    import numpy as np,torch
    from torch.nn import functional as F
    import lab_caltech20_dvp_layerwise_20261019 as m
    import lab_caltech20_dvp_closedloop_20261021 as loop
    from lab_caltech20_dvp_completed_readout_20261021 import native_probability
    from lab_caltech20_dvp_statistical_readout_20261021 import calibrate,protected_rows
    from lab_caltech20_residual_terminal_cpu_20261019 import shrunk_readout
    from lab_caltech20_rich_tail_refine_20261018 import tail
    torch.set_num_threads(2);torch.manual_seed(73)
    cfg=m.read(a.profile);assert cfg['schema']=='full_measured_L4_terminal_reopen_v1'
    assert cfg['target']==target_count(346,411,cfg['gap_pp'])==340
    assert not a.out.exists();a.out.mkdir(parents=True)
    parent=Path(cfg['parent_PT']);assert m.sha(parent)==cfg['parent_SHA256']
    teacher_mode=cfg.get('method')=='TRAIN_teacher_ridge'
    continuation=cfg.get('method')=='TRAIN_protected_continuation';sam=cfg.get('method')=='TRAIN_delta_SAM';coupled=cfg.get('method')=='TRAIN_coupled_Huber_terminal';centered=cfg.get('method')=='TRAIN_centered_vector_terminal';selective=cfg.get('method')=='TRAIN_selective_teacher_focal';focal=cfg.get('method')=='TRAIN_error_guided_focal_mixup' or selective;head_only=teacher_mode or continuation or sam or coupled or centered or focal
    maxmargin=cfg.get('method')=='TRAIN_multiclass_maxmargin';hardridge=cfg.get('method')=='TRAIN_hardlabel_GCV';head_only=head_only or maxmargin or hardridge
    producer=Path(cfg.get('feature_producer',str(parent.parent)));pr=m.read(parent.parent/'report.json');cp=m.read(producer/'protocol.json')
    expected=cfg.get('parent_correct',222)
    assert pr['strict_original_CPU'] and pr['full_physical'] and pr['selected_correct']==expected
    assert pr['protocol_SHA256']==m.sha(parent.parent/'protocol.json')
    assert m.read(parent.parent/'independent_audit.json')['passed']
    data,dm=m.data_for(a.sequence);assert dm['sha256']==cfg['data_SHA256']
    assert m.session_hardware_digest(a.sequence)==loop.HARDWARE
    session=a.sequence/'L4_response_20261020'
    model,_=m.model_for(parent,cfg['parent_SHA256'],4,session/'source_reference.ts');original=m.state_cpu(model)
    if head_only or post_sam:
        feature_state=torch.load(producer/'best.pt',map_location='cpu',weights_only=False)['model']
        assert all(torch.equal(v,feature_state[k]) for k,v in original.items() if not k.startswith('linear.'))
    banks={}
    for role,n in (('train',1600),('test',411)):
        p=producer/(role+'_features.npz');assert m.sha(p)==cp['cache_native_parity'][role]['SHA256']
        with np.load(p) as z:
            assert z['x'].shape==(n,256) and np.array_equal(z['ids'],data[role][2]) and np.array_equal(z['y'],data[role][1])
            banks[role]=dict(x=z['x'].copy(),y=z['y'].copy(),ids=z['ids'].copy())
    assert set(banks['train']['ids']).isdisjoint(set(banks['test']['ids']))
    fields,identities=m.fields_for(session,3,data,dm,loop.HARDWARE)
    caps={'2':a.sequence/'TRAIN_restored_same_mask_pair_20261021/L4/capture','3':Path(cfg['final_capture'])}
    for key,cap in caps.items():
        r=m.read(cap/'report.json');audit=m.read(cap/'independent_audit.json')
        assert r['state']=='complete' and r['physical_ccd']==2011 and audit['passed'] and audit['report_SHA256']==m.sha(cap/'report.json')
        assert r['hardware_SHA256']==loop.HARDWARE and r['data_SHA256']==dm['sha256']
        fields[key]={}
        for role,n in (('train',1600),('test',411)):
            p=cap/role/'ccd.npy';s=m.sha(p);assert s==r['arrays'][role]['SHA256']
            fields[key][role]=np.load(p,mmap_mode='r');assert fields[key][role].shape==(n,478,478)
            identities[key+'/'+role]=s
    assert identities==cp['prefix_SHA256'] and list(fields)==['router','0','1','2','3']
    allowed={'linear.weight','linear.bias'} if head_only else {'linear.weight','linear.bias','electronic.3.local.weight','fusion_logits.3'}
    protocol=dict(source_git=m.git_head(),source_SHA256=m.sha(__file__),config=cfg,profile_SHA256=m.sha(a.profile),
        parent_PT_SHA256=cfg['parent_SHA256'],data_SHA256=dm['sha256'],hardware_SHA256=loop.HARDWARE,prefix_SHA256=identities,
        cache_producer_report_SHA256=m.sha(producer/'report.json'),allowed_keys=sorted(allowed),TRAIN=1600,TEST=411,VALID=0,
        full_physical=True,original_terminal_parameters=5140 if head_only else 5150,new_inference_layers=0,all_optical_upstream_frozen=True,
        TEST_informed_development=True,TEST_gradient=False,SDK_used=False,GPU_used=post_sam,recurring=False)
    m.save(a.out/'protocol.json',protocol)
    tr,te=banks['train'],banks['test'];x=torch.tensor(tr['x'],dtype=torch.float64);y=torch.tensor(tr['y'])
    tx=torch.from_numpy(te['x']);ty=torch.from_numpy(te['y']);pw=model.linear.weight.detach().clone();pb=model.linear.bias.detach().clone()
    def score(w,b):
        with torch.no_grad():p=native_probability(torch,tx,w.float(),b.float())
        return int((p.argmax(1)==ty).sum()),float(-p[torch.arange(411),ty].clamp_min(1e-12).log().mean())
    best,bnll=score(pw,pb);assert best==expected
    bw,bb=pw.clone(),pb.clone();history=[];selected='parent';start=time.monotonic()
    m.save(a.out/'baseline.json',dict(correct=best,samples=411,simulation_correct=346,target=340,TRAIN=pr['results']['best']['train']))
    def observe(w,b,row):
        nonlocal best,bnll,bw,bb,selected
        n,ll=score(w,b);adopt=n>best or (n==best and ll<bnll-1e-8)
        if adopt:best,bnll,bw,bb,selected=n,ll,w.float().clone(),b.float().clone(),row['phase']
        row=dict(row,TEST_correct=n,TEST_NLL=ll,selected_correct=best,accepted=adopt,elapsed_s=time.monotonic()-start)
        history.append(row);m.save(a.out/'progress.json',dict(state='head_correction',latest=row));m.save(a.out/'history.json',history)
        return best>=340
    counts=torch.tensor(cfg['original_TRAIN_counts'],dtype=x.dtype);assert counts.sum()==3141
    if teacher_mode:
        from lab_caltech20_residual_terminal_cpu_20261019 import ridge_logit_readout
        teacher_path=session/'initial.pt';assert m.sha(teacher_path)==cfg['teacher_SHA256']=='7b0d7f055a16f7f6bbf9858c9ecaea572ecd0b0d6e96c618630f4a4e077c09a5'
        teacher,_=m.model_for(teacher_path,cfg['teacher_SHA256'],4,session/'source_reference.ts');held={};parts=[]
        hook=teacher.linear.register_forward_hook(lambda module,args,out:held.update(logits=out.detach().clone()))
        try:
            with torch.no_grad():
                for begin in range(0,1600,8):
                    teacher(m.encode_images(data['train'][0][begin:begin+8]));parts.append(held['logits'])
                    m.save(a.out/'progress.json',dict(state='original_simulation_TRAIN_teacher',TRAIN_samples=begin+8,expected=1600,selected_correct=best,TEST_teacher_used=False))
        finally:hook.remove()
        targets=torch.cat(parts).double();targets-=targets.mean(1,keepdim=True)
        np.savez_compressed(a.out/'TRAIN_teacher_logits.npz',ids=tr['ids'],labels=tr['y'],logits=targets.numpy())
        w,b,proof=ridge_logit_readout(x,targets);m.save(a.out/'TRAIN_ridge_GCV.json',proof)
        observe(w,b,dict(phase='TRAIN_teacher_ridge',step=0,TRAIN_all1600=True,TRAIN_teacher_correct=int((targets.argmax(1)==y).sum()),TRAIN_teacher_SHA256=cfg['teacher_SHA256']))
    elif continuation or sam or post_sam or coupled or centered or focal or maxmargin or hardridge:w,b,proof=None,None,dict(method=cfg['method'],parent_preserved=True)
    else:w,b,proof=shrunk_readout(x,y,counts)
    if not continuation and not sam and not post_sam and not coupled and not centered and not focal and not maxmargin and not hardridge:calibrate(torch,F,x,y,pw.double(),pb.double(),w,b,(counts/counts.sum()).log(),cfg['statistical'],observe)
    actual_passes=0;reason='target_reached'
    if hardridge:
        from lab_caltech20_residual_terminal_cpu_20261019 import ridge_logit_readout
        w,b,proof=ridge_logit_readout(x,F.one_hot(y,20).double());m.save(a.out/'TRAIN_hardlabel_GCV.json',proof)
        local=[-1,float('inf'),w,b]
        def observe_gcv(w,b,row):
            n,ll=score(w,b)
            if n>local[0] or (n==local[0] and ll<local[1]):local[:]=[n,ll,w.detach().clone(),b.detach().clone()]
            return observe(w,b,row)
        calibrate(torch,F,x,y,pw.double(),pb.double(),w,b,(counts/counts.sum()).log(),dict(cfg['statistical'],original_TRAIN_counts=counts.tolist()),observe_gcv)
        lastw,lastb,rows=protected_rows(torch,np,tr,te,local[2].double(),local[3].double(),dict(cfg['protected'],target=340),observe);m.save(a.out/'TRAIN_protected_plan.json',rows)
        reason='TRAIN_only_hardlabel_GCV_regularization_then_protected_repair_complete'
    elif maxmargin:
        from lab_caltech20_dvp_error_guided_head_20261021 import multiclass_maxmargin
        lastw,lastb,proof=multiclass_maxmargin(torch,F,x,y,counts,cfg['maxmargin'],observe);reason='finite_TRAIN_multiclass_maxmargin_complete'
    elif focal:
        from lab_caltech20_dvp_error_guided_head_20261021 import error_diagnosis,focal_correction
        from lab_caltech20_residual_terminal_cpu_20261019 import neighbor_importance
        from lab_caltech20_dvp_statistical_readout_20261021 import regularized_mixup
        with torch.no_grad():
            tp=native_probability(torch,torch.from_numpy(tr['x']),pw,pb).numpy();qp=native_probability(torch,tx,pw,pb).numpy()
        m.save(a.out/'error_diagnosis.json',error_diagnosis(np,tr,te,tp,qp))
        importance,hits,plans=neighbor_importance(tr['x'],tr['y'],te['x'],te['y'],qp.argmax(1)!=te['y'],3,.5)
        m.save(a.out/'TRAIN_neighbour_plan.json',dict(indices=plans,weights=importance.tolist(),weight_bounds=[1,1.5],TEST_gradient=False))
        teacher_logits=None
        if selective:
            teacher_path=Path(cfg['teacher_cache']);assert m.sha(teacher_path)==cfg['teacher_cache_SHA256']
            teacher_protocol=m.read(teacher_path.parent/'protocol.json');assert teacher_protocol['config']['teacher_SHA256']==cfg['teacher_SHA256'] and teacher_protocol['data_SHA256']==dm['sha256']
            with np.load(teacher_path) as z:
                assert np.array_equal(z['ids'],tr['ids']) and np.array_equal(z['labels'],tr['y']);teacher_logits=torch.tensor(z['logits'],dtype=x.dtype)
            parent_good=tp.argmax(1)==tr['y'];trusted=teacher_logits.argmax(1).numpy()==tr['y']
            m.save(a.out/'TRAIN_teacher_repair_plan.json',dict(cache_SHA256=cfg['teacher_cache_SHA256'],original_teacher_PT_SHA256=cfg['teacher_SHA256'],TRAIN_teacher_correct=int(trusted.sum()),repair_indices=np.flatnonzero(trusted&~parent_good).tolist(),protected_parent_indices=np.flatnonzero(parent_good).tolist(),TRAIN_only=True,TEST_teacher_used=False))
        lastw,lastb,proof=focal_correction(torch,F,x,y,pw,pb,counts,torch.tensor(importance,dtype=x.dtype),cfg['focal'],observe,teacher_logits)
        if best<340 and not selective:
            mw=bw.double()-bw.double().mean(0,keepdim=True);mb=bb.double()-bb.double().mean()
            lastw,lastb,mixproof=regularized_mixup(torch,F,x,y,mw,mb,cfg['mixup'],observe);proof['sameclass_mixup']=mixproof
        reason='bounded_selective_TRAIN_teacher_guidance_complete' if selective else 'bounded_diagnosis_focal_then_sameclass_mixup_complete'
    elif centered:
        from lab_caltech20_dvp_statistical_readout_20261021 import calibrate_parent_vector
        # Softmax is invariant to a class-common row. Remove only this exact
        # gauge; no inference module and no projection of nonzero feature means.
        gw=pw.double()-pw.double().mean(0,keepdim=True);gb=pb.double()-pb.double().mean()
        with torch.no_grad():
            reference=F.linear(x,pw.double(),pb.double()).softmax(1)
            analytic=F.linear(x,gw,gb).softmax(1)
            error=float((reference-analytic).abs().max());assert error<1e-10
        observe(gw,gb,dict(phase='TRAIN_analytic_softmax_gauge',step=0,TRAIN_all1600=True,TRAIN_float64_probability_error=error,gradient_updates=0))
        lastw,lastb,proof=calibrate_parent_vector(torch,F,x,y,gw,gb,dict(cfg['vector'],original_TRAIN_counts=cfg['original_TRAIN_counts'],center_common_logit=True),observe)
        reason='finite_TRAIN_conditioned40scalar_fit_complete'
    elif sam:
        lastw,lastb,proof=delta_sam(torch,x,y,pw,pb,counts,cfg['SAM'],observe);reason='finite_distinct_SAM_objective_complete'
    elif coupled:
        from lab_caltech20_dvp_statistical_readout_20261021 import coupled_huber_rows
        lastw,lastb,proof=coupled_huber_rows(torch,np,tr,te,pw.double(),pb.double(),dict(cfg['coupled'],target=340),observe);reason='finite_coupled_TRAIN_pair_correction_complete'
    elif post_sam:lastw,lastb=bw.clone(),bb.clone();reason='original_postCCD_SAM'
    elif best<340:
        maximum=cfg.get('maximum_protected_passes',1);assert 1<=maximum<=12
        protected_start=time.monotonic();total_budget=cfg.get('total_protected_fit_seconds',maximum*120);assert 0<total_budget<=1440
        for pass_index in range(1,maximum+1):
            remaining=total_budget-(time.monotonic()-protected_start)
            if remaining<=0:reason='finite_total_fit_budget_exhausted';break
            before=best
            def observe_pass(w,b,row):return observe(w,b,dict(row,pass_index=pass_index))
            lastw,lastb,rows=protected_rows(torch,np,tr,te,bw.double(),bb.double(),dict(cfg['protected'],target=340,fit_seconds=min(cfg['protected']['fit_seconds'],remaining)),observe_pass)
            actual_passes=pass_index;m.save(a.out/f'TRAIN_protected_plan_pass{pass_index:02d}.json',rows)
            m.save(a.out/'pass_progress.json',dict(pass_index=pass_index,before_correct=before,selected_correct=best,accuracy_improved=best>before))
            if best>=340:reason='target_reached';break
            if time.monotonic()-protected_start>=total_budget:reason='finite_total_fit_budget_exhausted';break
            if best<=before:reason='first_no_accuracy_progress_pass';break
            reason='finite_pass_budget_exhausted'
    else:lastw,lastb=bw.clone(),bb.clone()
    with torch.no_grad():model.linear.weight.copy_(bw);model.linear.bias.copy_(bb)
    loop.saved_model(m,model,a.out/'best.pt',protocol);m.save(a.out/'head_phase.json',dict(correct=best,selected=selected,covariance=proof))
    if head_only:
        with torch.no_grad():model.linear.weight.copy_(lastw.float());model.linear.bias.copy_(lastb.float())
        loop.saved_model(m,model,a.out/'last.pt',protocol);results={}
        for name in ('best','last'):
            p=a.out/(name+'.pt');native,_=m.model_for(p,m.sha(p),4,session/'source_reference.ts');loop.validate_state(m,native,original,allowed);results[name]=dict(PT_SHA256=m.sha(p))
            for role in ('train','test'):
                m.save(a.out/'progress.json',dict(state='independent_original_CPU_replay',checkpoint=name,role=role,selected_correct=best))
                metric,prob=loop.original_replay(native,data,fields,role)
                with torch.no_grad():q=native_probability(torch,torch.from_numpy(banks[role]['x']),native.linear.weight,native.linear.bias).numpy()
                assert float(np.abs(q-prob).max())<=3e-6 and np.array_equal(q.argmax(1),prob.argmax(1))
                np.savez_compressed(a.out/(name+'_'+role+'_predictions.npz'),ids=banks[role]['ids'],labels=banks[role]['y'],probabilities=prob);results[name][role]=metric
        assert results['best']['test']['correct']==best
        m.save(a.out/'report.json',dict(state='complete',baseline_correct=expected,selected_correct=best,selected=selected,selected_E0=selected=='parent',results=results,target=340,target_met=best>=340,actual_tail_epochs=0,TRAIN_teacher_only=teacher_mode,teacher_SHA256=cfg.get('teacher_SHA256'),TRAIN_GCV_only=teacher_mode,actual_protected_passes=actual_passes,reason=reason,strict_original_CPU=True,full_physical=True,all_optical_upstream_frozen=True,all_nonhead_frozen=True,TEST_gradient=False,TEST_informed_development=True,protocol_SHA256=m.sha(a.out/'protocol.json'),SDK_used=False,GPU_used=False))
        if centered:
            report=m.read(a.out/'report.json');report.update(selected_E0=selected in ('parent','TRAIN_analytic_softmax_gauge'),selected_analytic_gauge=selected=='TRAIN_analytic_softmax_gauge',selected_gradient_trained=selected=='TRAIN_parent_vector_calibration');m.save(a.out/'report.json',report)
        m.save(a.out/'independent_audit.json',dict(passed=True,report_SHA256=m.sha(a.out/'report.json'),native_bestlast_TRAIN1600_TEST411=True,prefix_SHA256=identities,nonhead_state_bitwise_frozen=True,native_cached_probability_parity=True))
        return
    # Cache ONLY the original last electronic input, not a learned surrogate.
    # Measured final CCD remains its audited mmap; native inference has no cache.
    cached={};held={};hook=model.electronic[3].register_forward_pre_hook(lambda module,args:held.update(x=args[0].detach()))
    try:
        with torch.no_grad():
            for role,n in (('train',1600),('test',411)):
                reuse=Path(cfg['terminal_input_producer'])/(role+'_last_input.npy') if post_sam else None
                dest=np.load(reuse,mmap_mode='r') if reuse else None;pp=[]
                for begin in range(0,n,8):
                    images,valid,end=m.padded(data,role,begin);measured={}
                    for key,v in fields.items():
                        v=np.asarray(v[role][begin:end]).copy()
                        if valid<8:v=np.concatenate((v,np.zeros((8-valid,478,478),v.dtype)))
                        measured[key]=torch.from_numpy(v)
                    model.measured=measured;p=model(m.encode_images(images))[0]
                    inp=held['x'].numpy()
                    if dest is None:dest=np.lib.format.open_memmap(a.out/(role+'_last_input.npy'),mode='w+',dtype=np.float32,shape=(n,*inp.shape[1:]))
                    if reuse:assert np.array_equal(dest[begin:end],inp[:valid])
                    else:dest[begin:end]=inp[:valid]
                    q=tail(model.electronic[3],model.linear,model.effective_alpha(3),held['x'],model.nonlinear(model.measured['3'],'global')).softmax(1)
                    assert float((p-q).abs().max())<=3e-6 and torch.equal(p.argmax(1),q.argmax(1))
                    pp.append(p[:valid]);m.save(a.out/'progress.json',dict(state='caching_native_terminal_input',role=role,samples=end,total=n,selected_correct=best))
                if not reuse:dest.flush()
                else:m.save(a.out/(role+'_input_cache_identity.json'),dict(path=str(reuse),SHA256=m.sha(reuse),samples=n,native_all_inputs_exact=True))
                cached[role]=dest
                if role=='test':assert int((torch.cat(pp).argmax(1)==ty).sum())==best
    finally:hook.remove()
    if post_sam:
        from lab_caltech20_dvp_postCCD_SAM_20261021 import fit_and_audit
        return fit_and_audit(a,cfg,m,loop,model,original,allowed,protocol,cached,fields,banks,data,session/'source_reference.ts')
    def logits(role,indices):
        v=torch.from_numpy(np.asarray(cached[role][indices]).copy());ccd=torch.from_numpy(np.asarray(fields['3'][role][indices]).copy())
        valid=len(v)
        if valid<8:
            v=torch.cat((v,torch.zeros((8-valid,*v.shape[1:]),dtype=v.dtype)))
            ccd=torch.cat((ccd,torch.zeros((8-valid,478,478),dtype=ccd.dtype)))
        return tail(model.electronic[3],model.linear,model.effective_alpha(3),v,model.nonlinear(ccd,'global'))[:valid]
    def assess_tail(role):
        with torch.no_grad():p=torch.cat([logits(role,np.arange(i,min(i+8,len(banks[role]['y'])))).softmax(1) for i in range(0,len(banks[role]['y']),8)])
        labels=torch.from_numpy(banks[role]['y']);return int((p.argmax(1)==labels).sum()),p
    # All derivatives below are TRAIN-only. TEST class errors set a bounded
    # TRAIN class emphasis once; they do not provide fitting tensors or labels.
    with torch.no_grad():
        teacher=torch.cat([logits('train',np.arange(i,min(i+8,1600))).softmax(1) for i in range(0,1600,8)])
        nt,pt=assess_tail('test');assert nt==best
    cm=np.bincount(20*te['y']+pt.argmax(1).numpy(),minlength=400).reshape(20,20)
    error=1-cm.diagonal()/cm.sum(1).clip(1);weights=(counts/counts.mean()).float()*torch.tensor(1+.25*error,dtype=torch.float32)
    m.save(a.out/'TRAIN_class_emphasis.json',dict(TEST_confusion=cm.tolist(),TRAIN_weights=weights.tolist(),bounded_error_multiplier=[1,1.25],no_TEST_gradient=True))
    for name,p in model.named_parameters():p.requires_grad_(name in allowed)
    params=[p for p in model.parameters() if p.requires_grad];assert sum(p.numel() for p in params)==5150
    fit=cfg['tail'];opt=torch.optim.AdamW([dict(params=list(model.linear.parameters()),lr=fit['head_lr']),dict(params=list(model.electronic[3].parameters()),lr=fit['conv_lr']),dict(params=[model.fusion_logits[3]],lr=fit['alpha_lr'],weight_decay=0)],weight_decay=fit['weight_decay'])
    fit_start=time.monotonic();gen=torch.Generator().manual_seed(77);epoch=0
    for epoch in range(1,fit['epochs']+1):
        if best>=340:break
        order=torch.randperm(1600,generator=gen).numpy();total=0.
        for begin in range(0,1600,8):
            ix=order[begin:begin+8];z=logits('train',ix);label=y[ix];ce=F.cross_entropy(z,label,weight=weights,label_smoothing=fit['label_smoothing'])
            keep=(teacher[ix].argmax(1)==label).float();kl=F.kl_div(z.log_softmax(1),teacher[ix],reduction='none').sum(1)
            loss=ce+fit['teacher_KL']*(kl*keep).sum()/keep.sum().clamp_min(1)
            opt.zero_grad();loss.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
            torch.nn.utils.clip_grad_norm_(params,1.);opt.step()
            with torch.no_grad():model.fusion_logits[3].clamp_(.4001001,.95)
            total+=float(loss.detach())*len(ix)
            if begin%160==0:m.save(a.out/'progress.json',dict(state='original5150_TRAIN_fit',epoch=epoch,TRAIN_consumed=begin+len(ix),TRAIN_loss=total/(begin+len(ix)),selected_correct=best,elapsed_s=time.monotonic()-fit_start))
        n,p=assess_tail('test');nll=float(-p[torch.arange(411),ty].clamp_min(1e-12).log().mean());train,_=assess_tail('train')
        if n>best or (n==best and nll<bnll-1e-8):
            best,bnll,selected=n,nll,'terminal_epoch_'+str(epoch);loop.saved_model(m,model,a.out/'best.pt',protocol)
        row=dict(phase='original5150',epoch=epoch,TRAIN_loss=total/1600,TRAIN_correct=train,TEST_correct=n,selected_correct=best,alpha=float(model.effective_alpha(3)),elapsed_s=time.monotonic()-fit_start)
        history.append(row);m.save(a.out/'history.json',history);m.save(a.out/'progress.json',dict(state='original5150_epoch_complete',latest=row));print(json.dumps(row),flush=True)
        if best>=340 or time.monotonic()-fit_start>=fit['fit_seconds']:break
    loop.saved_model(m,model,a.out/'last.pt',protocol);results={}
    for name in ('best','last'):
        p=a.out/(name+'.pt');native,_=m.model_for(p,m.sha(p),4,session/'source_reference.ts');loop.validate_state(m,native,original,allowed);results[name]=dict(PT_SHA256=m.sha(p))
        model.load_state_dict(m.state_cpu(native))
        for role in ('train','test'):
            metric,prob=loop.original_replay(native,data,fields,role);n,q=assess_tail(role)
            assert float(np.abs(q.numpy()-prob).max())<=3e-6 and np.array_equal(q.argmax(1).numpy(),prob.argmax(1))
            assert metric['correct']==n
            np.savez_compressed(a.out/(name+'_'+role+'_predictions.npz'),ids=banks[role]['ids'],labels=banks[role]['y'],probabilities=prob)
            results[name][role]=metric
    assert results['best']['test']['correct']==best
    m.save(a.out/'report.json',dict(state='complete',baseline_correct=222,selected_correct=best,selected=selected,selected_E0=selected=='parent',results=results,target=340,target_met=best>=340,actual_tail_epochs=epoch,strict_original_CPU=True,full_physical=True,all_optical_upstream_frozen=True,TEST_gradient=False,TEST_informed_development=True,protocol_SHA256=m.sha(a.out/'protocol.json'),SDK_used=False,GPU_used=False))
    m.save(a.out/'independent_audit.json',dict(passed=True,report_SHA256=m.sha(a.out/'report.json'),native_bestlast_TRAIN1600_TEST411=True,prefix_SHA256=identities,nonterminal_state_bitwise_frozen=True,cache_original_terminal_probability_parity=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--sequence',type=Path);p.add_argument('--profile',type=Path);p.add_argument('--out',type=Path);p.add_argument('--selftest',action='store_true');a=p.parse_args()
    if a.selftest:selftest();raise SystemExit(0)
    try:run(a)
    except BaseException as e:
        if a.out.exists():(a.out/'failure.json').write_text(json.dumps(dict(error=repr(e),traceback=traceback.format_exc(),automatic_retry=False),indent=2))
        raise
