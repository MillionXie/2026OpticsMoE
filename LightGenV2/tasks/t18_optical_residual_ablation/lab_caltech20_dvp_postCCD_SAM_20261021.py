"""TRAIN-only noise/dropout SAM of the ORIGINAL final9conv+alpha+5140 head.

Called only after full native input-cache equality and CCD lineage admission.
No optical masks, upstream electronics, physical settings or model structure change.
"""
import json,time
from pathlib import Path
from copy import deepcopy
import numpy as np
import torch
from torch.nn import functional as F
from lab_caltech20_rich_tail_refine_20261018 import tail


def sam_step(params,opt,objective,radius,alpha=None):
    """Use identical stochastic realization for both SAM forwards, then restore."""
    cpu_rng=torch.get_rng_state();gpu_rng=torch.cuda.get_rng_state() if params[0].is_cuda else None
    opt.zero_grad();clean=objective();assert torch.isfinite(clean);clean.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
    saved=[p.detach().clone() for p in params];norm=torch.stack([p.grad.norm() for p in params]).norm().clamp_min(1e-12)
    after_cpu=torch.get_rng_state();after_gpu=torch.cuda.get_rng_state() if gpu_rng is not None else None
    with torch.no_grad():
        for p in params:p.add_(p.grad,alpha=radius/float(norm))
        if alpha is not None:alpha.clamp_(.4001001,.95)
    torch.set_rng_state(cpu_rng)
    if gpu_rng is not None:torch.cuda.set_rng_state(gpu_rng)
    try:opt.zero_grad();probe=objective();assert torch.isfinite(probe);probe.backward()
    finally:
        with torch.no_grad():
            for p,v in zip(params,saved):p.copy_(v)
        torch.set_rng_state(after_cpu)
        if after_gpu is not None:torch.cuda.set_rng_state(after_gpu)
    assert all(torch.equal(p.detach(),v) for p,v in zip(params,saved))
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
    gradient=float(torch.nn.utils.clip_grad_norm_(params,1.));opt.step()
    if alpha is not None:
        with torch.no_grad():alpha.clamp_(.4001001,.95)
    return float(clean.detach()),float(probe.detach()),gradient


def adam_step(params,opt,objective,alpha):
    opt.zero_grad();loss=objective();assert torch.isfinite(loss);loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
    gradient=float(torch.nn.utils.clip_grad_norm_(params,1.));opt.step()
    with torch.no_grad():alpha.clamp_(.4001001,.95)
    return float(loss.detach()),float(loss.detach()),gradient


def conditioned_readout_block(x,y,pw,pb,sample_weights,cfg,observe):
    """TRAIN-only conditional head fit; optimizer coordinates are folded away."""
    from lab_caltech20_residual_sequence_20261018 import train_conditioned_geometry
    mean,root,inverse,_,floor=train_conditioned_geometry(x.numpy())
    x=x.double();pw=pw.double();pb=pb.double();pw=pw-pw.mean(0,keepdim=True);pb=pb-pb.mean()
    xx=torch.cat(((x-mean)@root,torch.ones(len(x),1,dtype=x.dtype)),1)
    u0=torch.cat((pw@inverse,(pb+pw@mean)[:,None]),1).detach();u=torch.nn.Parameter(u0.clone())
    base=F.linear(x,pw,pb);assert float((xx@u0.T-base).abs().max())<1e-7
    teacher=(base/cfg['temperature']).softmax(1).detach();keep=(base.argmax(1)==y).double()
    weight=sample_weights.double();assert bool((weight>0).all());weight=weight/weight.sum()
    opt=torch.optim.LBFGS([u],lr=.5,max_iter=1,line_search_fn='strong_wolfe');started=time.monotonic()
    for step in range(1,cfg['steps']+1):
        def closure():
            opt.zero_grad();z=xx@u.T;ce=F.cross_entropy(z,y,reduction='none',label_smoothing=cfg['label_smoothing'])
            kl=F.kl_div((z/cfg['temperature']).log_softmax(1),teacher,reduction='none').sum(1)*cfg['temperature']**2
            loss=(weight*ce).sum()+cfg['teacher_KL']*(kl*keep).sum()/keep.sum().clamp_min(1)+cfg['anchor']*(u-u0).square().sum()
            loss.backward();assert torch.isfinite(loss) and torch.isfinite(u.grad).all();return loss
        loss=opt.step(closure)
        with torch.no_grad():
            delta=u-u0;delta-=delta.mean(0,keepdim=True)
            delta*=((cfg['row_radius']/delta.norm(dim=1,keepdim=True).clamp_min(1e-12)).clamp_max(1.));u.copy_(u0+delta)
            ww=u[:,:-1]@root;bb=u[:,-1]-ww@mean
            assert float((F.linear(x,ww,bb)-xx@u.T).abs().max())<1e-7
        stop=observe(ww.detach(),bb.detach(),dict(phase='conditional_original_head_TRAIN',step=step,TRAIN_samples=len(x),TRAIN_loss=float(loss.detach()),TRAIN_correct=int(((xx@u.T).argmax(1)==y).sum())))
        if stop or time.monotonic()-started>=cfg['fit_seconds']:break
    return ww.detach(),bb.detach(),dict(TRAIN_only=True,TRAIN_samples=len(x),actual_steps=step,fit_seconds=time.monotonic()-started,variance_floor=floor,optimizer_parameters=u.numel(),original_Linear_fold=True,TEST_gradient=False,new_inference_layers=0)


def mixed_TRAIN_loss(z,labels,teacher,importance,class_weights,perm,smoothing,teacher_KL):
    """Fixed half/half TRAIN feature MixUp, matching weighted soft targets."""
    ce1=F.cross_entropy(z,labels,weight=class_weights,reduction='none',label_smoothing=smoothing)
    ce2=F.cross_entropy(z,labels[perm],weight=class_weights,reduction='none',label_smoothing=smoothing)
    target=.5*(teacher+teacher[perm]);good=teacher.argmax(1)==labels;keep=(good&good[perm]).float()
    kl=F.kl_div(z.log_softmax(1),target,reduction='none').sum(1)
    return .5*(ce1*importance+ce2*importance[perm]).mean()+teacher_KL*(kl*keep).sum()/keep.sum().clamp_min(1)


def fit_and_audit(a,cfg,m,loop,model,original,allowed,protocol,cached,fields,banks,data,reference):
    from lab_caltech20_residual_terminal_cpu_20261019 import neighbor_importance
    torch.set_num_threads(2);torch.manual_seed(73);fit=cfg['tail'];tr,te=banks['train'],banks['test'];train_y=torch.from_numpy(tr['y']).long()
    pw=model.linear.weight.detach().clone();pb=model.linear.bias.detach().clone()
    with torch.no_grad():teacher=F.linear(torch.from_numpy(tr['x']),pw,pb).softmax(1)
    def cpu_prob(role,electronic,head,alpha):
        pieces=[];n=len(banks[role]['y'])
        with torch.no_grad():
            for begin in range(0,n,8):
                ix=np.arange(begin,min(begin+8,n));valid=len(ix);v=torch.from_numpy(np.asarray(cached[role][ix]).copy());ccd=torch.from_numpy(np.asarray(fields['3'][role][ix]).copy())
                if valid<8:v=torch.cat((v,torch.zeros((8-valid,*v.shape[1:]),dtype=v.dtype)));ccd=torch.cat((ccd,torch.zeros((8-valid,478,478),dtype=ccd.dtype)))
                pieces.append(tail(electronic,head,alpha,v,model.nonlinear(ccd,'global')).softmax(1)[:valid])
        return torch.cat(pieces)
    bp=cpu_prob('test',model.electronic[3],model.linear,model.effective_alpha(3));best=int((bp.argmax(1).numpy()==te['y']).sum());assert best==cfg['parent_correct']
    bnll=float(-bp[torch.arange(411),torch.from_numpy(te['y'])].clamp_min(1e-12).log().mean());selected=0;last_improved=0;history=[]
    importance,hits,plans=neighbor_importance(tr['x'],tr['y'],te['x'],te['y'],bp.argmax(1).numpy()!=te['y'],3,.5)
    cm=np.bincount(te['y']*20+bp.argmax(1).numpy(),minlength=400).reshape(20,20);error=1-cm.diagonal()/cm.sum(1).clip(1)
    prior=np.asarray(cfg['original_TRAIN_counts'],float);weights=torch.tensor(np.sqrt(prior/prior.mean())*(1+.25*error),dtype=torch.float32)
    m.save(a.out/'TRAIN_emphasis_plan.json',dict(TEST_confusion=cm.tolist(),TRAIN_importance=importance.tolist(),TRAIN_neighbours=plans,TRAIN_class_weights=weights.cpu().tolist(),TEST_gradient=False))
    loop.gpu_admission(m,cfg);model.cuda();weights=weights.cuda();use_sam=fit.get('optimizer_method','SAM')=='SAM'
    alternating=fit.get('alternating_readout',False);blocks=[]
    freeze_head=fit.get('freeze_head',False);train_keys=allowed-{'linear.weight','linear.bias'} if freeze_head or alternating else allowed
    if fit.get('center_class_common',False):
        with torch.no_grad():
            w=model.linear.weight.double();b=model.linear.bias.double();model.linear.weight.copy_(w-w.mean(0,keepdim=True));model.linear.bias.copy_(b-b.mean())
        m.save(a.out/'analytic_head_gauge.json',dict(original_Linear_only=True,gradient_trained=False,head_frozen=freeze_head,no_new_inference_layer=True))
    if fit.get('TRAIN_kernel_restart',False):
        p=Path(cfg['TRAIN_restart_diagnosis']);assert m.sha(p)==cfg['TRAIN_restart_diagnosis_SHA256']
        diagnosis=m.read(p);assert diagnosis['TRAIN_only'] and not diagnosis['TEST_used'] and diagnosis['parent_SHA256']==cfg['parent_SHA256'] and diagnosis['samples']==100
        kernel=model.electronic[3].local.weight
        assert torch.equal(kernel.detach().cpu().flatten(),torch.tensor(diagnosis['final_local_kernel'],dtype=kernel.dtype))
        zero_init=fit.get('TRAIN_kernel_zero_init',False);scale=0. if zero_init else min(1.,1/max(diagnosis['abs_conv_p50_p90_p99'][1],1e-12))
        with torch.no_grad():
            if zero_init:kernel.zero_();assert int(torch.count_nonzero(kernel))==0
            else:kernel.mul_(scale)
        m.save(a.out/'TRAIN_kernel_restart.json',dict(diagnosis_SHA256=m.sha(p),TRAIN_samples=100,original_kernel_norm=diagnosis['kernel_norm'],TRAIN_abs_response_p90=diagnosis['abs_conv_p50_p90_p99'][1],scale=scale,canonical_zero_initialization=zero_init,scaled_kernel_norm=float(kernel.norm()),TRAIN_saturated_fraction_before=diagnosis['tanh_saturated_fraction'],not_proven_root_cause=True,new_layer=False,TEST_used=False))
    for name,p in model.named_parameters():p.requires_grad_(name in train_keys)
    params=[p for p in model.parameters() if p.requires_grad];expected_trainable=10 if freeze_head or alternating else 5150
    assert expected_trainable==10 or expected_trainable==5150
    assert sum(p.numel() for p in params)==expected_trainable
    groups=[dict(params=list(model.electronic[3].parameters()),lr=fit['conv_lr']),dict(params=[model.fusion_logits[3]],lr=fit['alpha_lr'],weight_decay=0)]
    if not freeze_head and not alternating:groups.append(dict(params=list(model.linear.parameters()),lr=fit['head_lr']))
    opt=torch.optim.AdamW(groups,weight_decay=fit['weight_decay'])
    active={'TRAIN':False};counts=dict(TRAIN_noise_batches=0,TRAIN_dropout_calls=0,TRAIN_CCD_pixel_dropout_batches=0,TRAIN_feature_mixup_batches=0,SWA_snapshots=0,SWA_evaluations=0,SAM_updates=0,AdamW_updates=0,eval_augmented_batches=0)
    swa_sum={};swa_snapshots=0;swa_history=[]
    def dropout(module,inputs):
        if active['TRAIN']:
            x=inputs[0]
            if fit['feature_dropout_p']>0:counts['TRAIN_dropout_calls']+=1;x=F.dropout(x,p=fit['feature_dropout_p'],training=True)
            if fit.get('TRAIN_feature_mixup',False):x=.5*(x+x[active['mix_perm']])
            return (x,)
    hook=model.linear.register_forward_pre_hook(dropout);rng=np.random.default_rng(73);start=time.monotonic()
    try:
        for epoch in range(1,fit['epochs']+1):
            total=0.;used=0;nonzero=0;order=rng.permutation(1600)
            for begin in range(0,1600,8):
                ix=order[begin:begin+8];v=torch.from_numpy(np.asarray(cached['train'][ix]).copy()).cuda();ccd=torch.from_numpy(np.asarray(fields['3']['train'][ix]).copy()).cuda().float()
                # Fixed noise per update; the probe uses the identical view and RNG.
                gain=1+fit['CCD_gain_sigma']*torch.randn((len(ix),1,1),device='cuda');noisy=(ccd*gain+fit['CCD_additive_sigma_DN']*torch.randn_like(ccd)).clamp(0,255)
                if fit.get('CCD_pixel_dropout_p',0)>0:
                    noisy=F.dropout(noisy,p=fit['CCD_pixel_dropout_p'],training=True).clamp(0,255);counts['TRAIN_CCD_pixel_dropout_batches']+=1
                optical=model.nonlinear(noisy,'global');labels=train_y[ix].cuda();teach=teacher[ix].cuda();sample_weight=torch.tensor(importance[ix],dtype=torch.float32,device='cuda');active['TRAIN']=True
                if fit.get('TRAIN_feature_mixup',False):active['mix_perm']=torch.arange(len(ix),device='cuda').roll(1);counts['TRAIN_feature_mixup_batches']+=1
                def objective():
                    z=tail(model.electronic[3],model.linear,model.effective_alpha(3),v,optical)
                    if fit.get('TRAIN_feature_mixup',False):return mixed_TRAIN_loss(z,labels,teach,sample_weight,weights,active['mix_perm'],fit['label_smoothing'],fit['teacher_KL'])
                    ce=F.cross_entropy(z,labels,weight=weights,label_smoothing=fit['label_smoothing'],reduction='none')
                    keep=(teach.argmax(1)==labels).float();kl=F.kl_div(z.log_softmax(1),teach,reduction='none').sum(1)
                    return (ce*sample_weight).mean()+fit['teacher_KL']*(kl*keep).sum()/keep.sum().clamp_min(1)
                clean,probe,norm=sam_step(params,opt,objective,fit['SAM_radius'],model.fusion_logits[3]) if use_sam else adam_step(params,opt,objective,model.fusion_logits[3]);active['TRAIN']=False
                total+=clean*len(ix);used+=len(ix);nonzero+=int(norm>0);counts['TRAIN_noise_batches']+=1;counts['SAM_updates' if use_sam else 'AdamW_updates']+=1
                if begin%160==0:m.save(a.out/'progress.json',dict(state='L4_postCCD_featuremix_TRAIN' if fit.get('TRAIN_feature_mixup',False) else 'L4_postCCD_alternating_TRAIN' if alternating else 'L4_postCCD_SAM_TRAIN' if use_sam else 'L4_postCCD_Adam_TRAIN',epoch=epoch,TRAIN_consumed=used,TRAIN_loss=total/used,nonzero_gradient_batches=nonzero,selected_correct=best,elapsed_s=time.monotonic()-start))
                batch=begin//8+1
                if epoch==1 and batch<=fit.get('early_evaluate_batches',0) and batch%fit.get('early_evaluate_every',5)==0:
                    e=deepcopy(model.electronic[3]).cpu().eval();h=deepcopy(model.linear).cpu().eval();h._forward_pre_hooks.clear();alpha=model.effective_alpha(3).detach().cpu().clone()
                    q=cpu_prob('test',e,h,alpha);n=int((q.argmax(1).numpy()==te['y']).sum());nll=float(-q[torch.arange(411),torch.from_numpy(te['y'])].clamp_min(1e-12).log().mean())
                    if n>best or (n==best and nll<bnll-1e-8):best,bnll,selected,last_improved=n,nll,epoch+(batch-200)/200,epoch;loop.saved_model(m,model,a.out/'best.pt',protocol)
                    row=dict(phase='early_full_TEST_selection',epoch=epoch,TRAIN_batches=batch,TRAIN_consumed=used,TEST_samples=411,TEST_correct=n,TEST_NLL=nll,selected_correct=best,alpha=float(alpha),elapsed_s=time.monotonic()-start)
                    history.append(row);m.save(a.out/'history.json',history);m.save(a.out/'progress.json',dict(state='early_original_tail_fullTEST',latest=row))
            assert used==1600 and nonzero>0
            if fit.get('SWA_start_epoch') and epoch>=fit['SWA_start_epoch']:
                for name,p in model.named_parameters():
                    if name in allowed:
                        value=p.detach().cpu().double()
                        if name not in swa_sum:swa_sum[name]=value.clone()
                        else:swa_sum[name].add_(value)
                swa_snapshots+=1;counts['SWA_snapshots']=swa_snapshots
            if alternating and epoch%fit['geometry_epochs_per_block']==0:
                # Exact native CPU features for the CURRENT ten nonlinear parameters.
                # TEST features go only to detached observation, not the head fit.
                e=deepcopy(model.electronic[3]).cpu().eval();h=deepcopy(model.linear).cpu().eval();h._forward_pre_hooks.clear();alpha=model.effective_alpha(3).detach().cpu().clone()
                features_by_role={};parity={}
                from lab_caltech20_dvp_completed_readout_20261021 import native_probability
                for role in ('train','test'):
                    pieces=[];fh=h.register_forward_pre_hook(lambda module,args:pieces.append(args[0].detach().clone()))
                    try:q=cpu_prob(role,e,h,alpha)
                    finally:fh.remove()
                    features_by_role[role]=torch.cat(pieces)[:len(banks[role]['y'])]
                    parity[role]=float((native_probability(torch,features_by_role[role],h.weight,h.bias)-q).abs().max());assert parity[role]<=3e-6
                    m.save(a.out/'progress.json',dict(state='alternating_current_tail_features',epoch=epoch,role=role,samples=len(q),selected_correct=best))
                def observe_head(w,b,row):
                    nonlocal best,bnll,selected,last_improved
                    with torch.no_grad():q=native_probability(torch,features_by_role['test'],w.float(),b.float())
                    n=int((q.argmax(1).numpy()==te['y']).sum());nll=float(-q[torch.arange(411),torch.from_numpy(te['y'])].clamp_min(1e-12).log().mean())
                    if n>best or (n==best and nll<bnll-1e-8):
                        best,bnll,selected,last_improved=n,nll,epoch+row['step']/1000,epoch
                        with torch.no_grad():model.linear.weight.copy_(w);model.linear.bias.copy_(b)
                        loop.saved_model(m,model,a.out/'best.pt',protocol)
                    history.append(dict(row,epoch=epoch,TEST_samples=411,TEST_correct=n,TEST_NLL=nll,selected_correct=best,elapsed_s=time.monotonic()-start));m.save(a.out/'history.json',history)
                    return best>=340
                sw=torch.from_numpy(importance)*weights.detach().cpu()[train_y].double()
                ww,bb,proof=conditioned_readout_block(features_by_role['train'],train_y,h.weight.detach(),h.bias.detach(),sw,fit['readout_block'],observe_head)
                with torch.no_grad():model.linear.weight.copy_(ww);model.linear.bias.copy_(bb)
                blocks.append(dict(epoch=epoch,fit=proof,current_native_feature_parity=parity));m.save(a.out/'alternating_readout_audit.json',dict(blocks=blocks,original_parameters_only=True,geometry_parameters=10,readout_parameters=5140,no_new_layer=True,TEST_gradient=False))
            if epoch==1 or epoch%fit['evaluate_every']==0 or time.monotonic()-start>=fit['fit_seconds']:
                e=deepcopy(model.electronic[3]).cpu().eval();h=deepcopy(model.linear).cpu().eval();h._forward_pre_hooks.clear();alpha=model.effective_alpha(3).detach().cpu().clone()
                q=cpu_prob('test',e,h,alpha);n=int((q.argmax(1).numpy()==te['y']).sum());nll=float(-q[torch.arange(411),torch.from_numpy(te['y'])].clamp_min(1e-12).log().mean())
                if n>best or (n==best and nll<bnll-1e-8):best,bnll,selected,last_improved=n,nll,epoch,epoch;loop.saved_model(m,model,a.out/'best.pt',protocol)
                train_correct=None
                if epoch%4==0:train_correct=int((cpu_prob('train',e,h,alpha).argmax(1)==train_y).sum())
                row=dict(epoch=epoch,TRAIN_consumed=used,TRAIN_loss=total/1600,TRAIN_correct=train_correct,TEST_correct=n,selected_correct=best,TEST_NLL=nll,alpha=float(alpha),elapsed_s=time.monotonic()-start)
                history.append(row);m.save(a.out/'history.json',history);m.save(a.out/'progress.json',dict(state='L4_postCCD_SAM_epoch_complete',latest=row));print(json.dumps(row),flush=True)
                if swa_snapshots:
                    mean={name:(value/swa_snapshots).to(original[name].dtype) for name,value in swa_sum.items()};assert set(mean)==allowed
                    ae=deepcopy(model.electronic[3]).cpu().eval();ah=deepcopy(model.linear).cpu().eval();ah._forward_pre_hooks.clear()
                    ae.load_state_dict({'local.weight':mean['electronic.3.local.weight']})
                    with torch.no_grad():ah.weight.copy_(mean['linear.weight']);ah.bias.copy_(mean['linear.bias'])
                    q=cpu_prob('test',ae,ah,mean['fusion_logits.3']);an=int((q.argmax(1).numpy()==te['y']).sum());alloss=float(-q[torch.arange(411),torch.from_numpy(te['y'])].clamp_min(1e-12).log().mean())
                    restored=True
                    if an>best or (an==best and alloss<bnll-1e-8):
                        params_by_name=dict(model.named_parameters());saved={name:params_by_name[name].detach().clone() for name in allowed}
                        try:
                            with torch.no_grad():
                                for name,value in mean.items():params_by_name[name].copy_(value)
                            loop.saved_model(m,model,a.out/'best.pt',protocol)
                        finally:
                            with torch.no_grad():
                                for name,value in saved.items():params_by_name[name].copy_(value)
                        restored=all(torch.equal(params_by_name[name].detach(),value) for name,value in saved.items());assert restored
                        best,bnll,selected,last_improved=an,alloss,'SWA_epoch'+str(epoch),epoch
                    counts['SWA_evaluations']+=1
                    ar=dict(phase='TRAIN_original_parameter_SWA',epoch=epoch,TRAIN_snapshots=swa_snapshots,TEST_samples=411,TEST_correct=an,TEST_NLL=alloss,selected_correct=best,original_terminal_only=True,no_inference_ensemble=True,training_parameters_restored=restored,elapsed_s=time.monotonic()-start)
                    history.append(ar);swa_history.append(ar);m.save(a.out/'history.json',history);m.save(a.out/'SWA_audit.json',dict(snapshots=swa_snapshots,allowed_keys=sorted(allowed),evaluations=swa_history,TRAIN_snapshot_schedule=True,new_inference_layers=0,TEST_gradient=False));m.save(a.out/'progress.json',dict(state='original_parameter_SWA_evaluation',latest=ar))
            if best>=340 or time.monotonic()-start>=fit['fit_seconds'] or epoch-last_improved>=fit['patience_epochs']:break
        model.cpu();loop.validate_state(m,model,original,allowed);loop.saved_model(m,model,a.out/'last.pt',protocol)
    finally:active['TRAIN']=False;hook.remove()
    results={}
    for name in ('best','last'):
        pt=a.out/(name+'.pt');native,_=m.model_for(pt,m.sha(pt),4,reference);loop.validate_state(m,native,original,allowed);results[name]=dict(PT_SHA256=m.sha(pt))
        for role in ('train','test'):
            m.save(a.out/'progress.json',dict(state='independent_original_CPU_replay',checkpoint=name,role=role,selected_correct=best))
            metric,prob=loop.original_replay(native,data,fields,role)
            q=cpu_prob(role,native.electronic[3],native.linear,native.effective_alpha(3)).numpy()
            assert float(np.abs(q-prob).max())<=3e-6 and np.array_equal(q.argmax(1),prob.argmax(1))
            results[name][role]=metric;np.savez_compressed(a.out/(name+'_'+role+'_predictions.npz'),ids=banks[role]['ids'],labels=banks[role]['y'],probabilities=prob)
    assert results['best']['test']['correct']==best
    m.save(a.out/'TRAIN_transform_audit.json',dict(counts=counts,original_state_only=True,eval_augmentation=False,synthetic_not_calibrated_CCD_noise=True,SAM_probe_restored_exactly=use_sam,identical_probe_stochastic_view=use_sam,optimizer_method='SAM' if use_sam else 'AdamW',TRAIN_kernel_restart=fit.get('TRAIN_kernel_restart',False)))
    m.save(a.out/'report.json',dict(state='complete',baseline_correct=cfg['parent_correct'],selected_correct=best,selected_epoch=selected,selected_E0=selected==0,actual_epochs=epoch,target_correct=340,target_met=best>=340,full_physical=True,strict_original_CPU=True,results=results,
        allowed_parameters=5150,gradient_parameters=5150 if alternating else expected_trainable,trainable_keys=sorted(allowed if alternating else train_keys),head_gradient_frozen=freeze_head,alternating_readout=alternating,geometry_block_gradient_parameters=expected_trainable,readout_blocks=len(blocks),all_optical_upstream_frozen=True,TEST_gradient=False,TEST_informed_development=True,protocol_SHA256=m.sha(a.out/'protocol.json'),SDK_used=False,GPU_used=True))
    m.save(a.out/'independent_audit.json',dict(passed=True,report_SHA256=m.sha(a.out/'report.json'),native_bestlast_TRAIN1600_TEST411=True,all_nonterminal_bitwise_frozen=True,prefix_SHA256=protocol['prefix_SHA256'],cached_native_probability_parity=True))
    del opt;torch.cuda.empty_cache()


def selftest():
    torch.set_num_threads(2);torch.manual_seed(73)
    p=torch.nn.Parameter(torch.tensor([.3,.8]));opt=torch.optim.SGD([p],lr=.01);seen=[]
    def objective():
        value=torch.rand(2);seen.append(value.clone());return ((p-value)**2).sum()
    a,b,n=sam_step([p],opt,objective,.02)
    assert len(seen)==2 and torch.equal(seen[0],seen[1]) and n>0 and torch.isfinite(p).all()
    from caltech20_alt246_biasfree_model import SmallElectronicResidual
    e=SmallElectronicResidual();h=torch.nn.Linear(256,20);alpha=torch.nn.Parameter(torch.tensor(.7,dtype=torch.float64))
    for p in h.parameters():p.requires_grad_(False)
    x=torch.rand(2,478,478);o=torch.rand_like(x)
    F.cross_entropy(tail(e,h,alpha,x,o),torch.tensor([0,1])).backward()
    ten=[*e.parameters(),alpha];assert sum(p.numel() for p in ten)==10
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in ten) and all(p.grad is None for p in h.parameters())
    q=torch.nn.Parameter(torch.tensor([.5,.7]));alpha=torch.nn.Parameter(torch.tensor(.7,dtype=torch.float64));adam=torch.optim.AdamW([q,alpha],lr=.001)
    before=q.detach().clone();clean,_,gn=adam_step([q,alpha],adam,lambda:q.square().sum()+alpha.square(),alpha)
    assert gn>0 and torch.isfinite(q).all() and not torch.equal(before,q)
    xx=torch.randn(60,8);yy=torch.arange(60)%20;w=torch.randn(20,8)*.01;b=torch.zeros(20);seen=[]
    wc,bc,proof=conditioned_readout_block(xx,yy,w,b,torch.ones(60),dict(temperature=2.,label_smoothing=.02,teacher_KL=.1,anchor=.01,row_radius=1.,steps=3,fit_seconds=30),lambda a,b,r:seen.append(r))
    assert len(seen)==3 and wc.shape==w.shape and torch.isfinite(wc).all() and not proof['TEST_gradient'] and proof['new_inference_layers']==0
    zz=torch.randn(8,20,requires_grad=True);labels=torch.arange(8);teacher=F.one_hot(labels,20).float();perm=torch.arange(8).roll(1)
    mixloss=mixed_TRAIN_loss(zz,labels,teacher,torch.ones(8),torch.ones(20),perm,0.,0.)
    expected=.5*(F.cross_entropy(zz,labels)+F.cross_entropy(zz,labels[perm]));assert torch.allclose(mixloss,expected)
    mixloss.backward();assert zz.grad is not None and torch.isfinite(zz.grad).all()
    print(json.dumps(dict(passed=True,identical_probe_view=True,probe_parameter_restoration=True,finite_gradient=True,geometry_only_gradient_parameters=10,frozen_head_no_gradient=True,TEST_used=False,SDK_used=False)))
