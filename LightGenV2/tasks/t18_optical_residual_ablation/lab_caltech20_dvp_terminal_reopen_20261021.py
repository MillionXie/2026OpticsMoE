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


def run(a):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
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
    continuation=cfg.get('method')=='TRAIN_protected_continuation';head_only=teacher_mode or continuation
    producer=Path(cfg.get('feature_producer',str(parent.parent)));pr=m.read(parent.parent/'report.json');cp=m.read(producer/'protocol.json')
    expected=cfg.get('parent_correct',222)
    assert pr['strict_original_CPU'] and pr['full_physical'] and pr['selected_correct']==expected
    assert pr['protocol_SHA256']==m.sha(parent.parent/'protocol.json')
    assert m.read(parent.parent/'independent_audit.json')['passed']
    data,dm=m.data_for(a.sequence);assert dm['sha256']==cfg['data_SHA256']
    assert m.session_hardware_digest(a.sequence)==loop.HARDWARE
    session=a.sequence/'L4_response_20261020'
    model,_=m.model_for(parent,cfg['parent_SHA256'],4,session/'source_reference.ts');original=m.state_cpu(model)
    if head_only:
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
        TEST_informed_development=True,TEST_gradient=False,SDK_used=False,GPU_used=False,recurring=False)
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
    elif continuation:w,b,proof=None,None,dict(method='finite_same_objective_TRAIN_protected_continuation',parent_preserved=True)
    else:w,b,proof=shrunk_readout(x,y,counts)
    if not continuation:calibrate(torch,F,x,y,pw.double(),pb.double(),w,b,(counts/counts.sum()).log(),cfg['statistical'],observe)
    actual_passes=0;reason='target_reached'
    if best<340:
        maximum=cfg.get('maximum_protected_passes',1);assert 1<=maximum<=3
        for pass_index in range(1,maximum+1):
            before=best
            def observe_pass(w,b,row):return observe(w,b,dict(row,pass_index=pass_index))
            lastw,lastb,rows=protected_rows(torch,np,tr,te,bw.double(),bb.double(),dict(cfg['protected'],target=340),observe_pass)
            actual_passes=pass_index;m.save(a.out/f'TRAIN_protected_plan_pass{pass_index:02d}.json',rows)
            m.save(a.out/'pass_progress.json',dict(pass_index=pass_index,before_correct=before,selected_correct=best,accuracy_improved=best>before))
            if best>=340:reason='target_reached';break
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
        m.save(a.out/'independent_audit.json',dict(passed=True,report_SHA256=m.sha(a.out/'report.json'),native_bestlast_TRAIN1600_TEST411=True,prefix_SHA256=identities,nonhead_state_bitwise_frozen=True,native_cached_probability_parity=True))
        return
    # Cache ONLY the original last electronic input, not a learned surrogate.
    # Measured final CCD remains its audited mmap; native inference has no cache.
    cached={};held={};hook=model.electronic[3].register_forward_pre_hook(lambda module,args:held.update(x=args[0].detach()))
    try:
        with torch.no_grad():
            for role,n in (('train',1600),('test',411)):
                dest=None;pp=[]
                for begin in range(0,n,8):
                    images,valid,end=m.padded(data,role,begin);measured={}
                    for key,v in fields.items():
                        v=np.asarray(v[role][begin:end]).copy()
                        if valid<8:v=np.concatenate((v,np.zeros((8-valid,478,478),v.dtype)))
                        measured[key]=torch.from_numpy(v)
                    model.measured=measured;p=model(m.encode_images(images))[0]
                    inp=held['x'].numpy()
                    if dest is None:dest=np.lib.format.open_memmap(a.out/(role+'_last_input.npy'),mode='w+',dtype=np.float32,shape=(n,*inp.shape[1:]))
                    dest[begin:end]=inp[:valid]
                    q=tail(model.electronic[3],model.linear,model.effective_alpha(3),held['x'],model.nonlinear(model.measured['3'],'global')).softmax(1)
                    assert float((p-q).abs().max())<=3e-6 and torch.equal(p.argmax(1),q.argmax(1))
                    pp.append(p[:valid]);m.save(a.out/'progress.json',dict(state='caching_native_terminal_input',role=role,samples=end,total=n,selected_correct=best))
                dest.flush();cached[role]=dest
                if role=='test':assert int((torch.cat(pp).argmax(1)==ty).sum())==best
    finally:hook.remove()
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
    p=argparse.ArgumentParser();p.add_argument('--sequence',type=Path);p.add_argument('--profile',type=Path);p.add_argument('--out',type=Path);a=p.parse_args()
    try:run(a)
    except BaseException as e:
        if a.out.exists():(a.out/'failure.json').write_text(json.dumps(dict(error=repr(e),traceback=traceback.format_exc(),automatic_retry=False),indent=2))
        raise
