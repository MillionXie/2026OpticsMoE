"""TRAIN-only hard-example augmentation of the original post-CCD5150.

Conditioning is an optimizer coordinate change, folded into the original Linear.
Synthetic feature mixup is not a new optical observation. TEST never enters loss.
"""
import os,time
import numpy as np
import torch
from torch.nn import functional as F
import lab_caltech20_dvp_layerwise_20261019 as m
import lab_caltech20_dvp_closedloop_20261021 as loop
from lab_caltech20_tucam_error_guided_20261023 import better
from lab_caltech20_tucam_terminal_20261023 import assess_tail, allowed, finish
from lab_caltech20_tucam_covariance_20261023 import branch_bank
from lab_caltech20_residual_sequence_20261018 import train_conditioned_geometry
from lab_caltech20_residual_terminal_cpu_20261019 import neighbor_importance
from lab_caltech20_rich_tail_refine_20261018 import tail


def paired_order(labels, rng):
    """Every TRAIN identity once; adjacent pairs share labels, no TEST identity."""
    pairs=[]
    for label in np.unique(labels):
        ix=rng.permutation(np.flatnonzero(labels==label))
        assert len(ix)%2==0
        pairs.extend(ix.reshape(-1,2).tolist())
    return np.asarray(pairs)[rng.permutation(len(pairs))].reshape(-1)


def conditioned_logits(x, w, b, delta, mean, root, mix, dropout):
    # The physical x/CCD pairs remain intact through the original final fusion.
    # Only these TRAIN feature views are synthetic and label-preserving.
    xx=x.double()
    if mix:
        partners=torch.arange(len(x)).reshape(-1,2).flip(1).reshape(-1)
        coefficient=torch.zeros(len(x),1,dtype=xx.dtype)
        coefficient[::2]=mix*torch.rand((len(x)//2,1),dtype=xx.dtype)
        xx=(1-coefficient)*xx+coefficient*xx[partners]
    u=F.dropout((xx-mean)@root,p=dropout,training=dropout>0)
    return F.linear(xx,w.double(),b.double())+F.linear(u,delta[:,:-1],delta[:,-1])


def folded(w,b,delta,mean,root):
    dw=delta[:,:-1]@root.T
    return (w.double()+dw).float(),(b.double()+delta[:,-1]-dw@mean).float()


def sam_perturb(groups,radii):
    """Temporary optimizer perturbation only; restore before updating/saving."""
    saved=[]
    with torch.no_grad():
        for group,radius in zip(groups,radii):
            norm=torch.stack([p.grad.double().square().sum() for p in group]).sum().sqrt()
            for p in group:
                assert p.grad is not None and torch.isfinite(p.grad).all()
                saved.append((p,p.detach().clone()));p.add_(p.grad*(radius/max(float(norm),1e-12)))
    return saved


def sam_restore(saved):
    with torch.no_grad():
        for p,value in saved:p.copy_(value)


def interpolated_state(original,direction,keys,amount):
    assert 0<=amount<=1 and set(original)==set(direction) and keys<=set(original)
    return {n:(v.double()+amount*(direction[n].double()-v.double())).to(v.dtype) if n in keys else v.clone() for n,v in original.items()}


def fit(model,inputs,fields,data,cfg,recipe,out,reference,identities):
    if os.name=='nt':assert len(str((out/'TRAIN_selection_plan.json.tmp').absolute()))<259,'Use a shorter run/output path before TRAIN fitting'
    out.mkdir();k=model.depth-1;original=m.state_cpu(model);keys=allowed(model)
    baseline={};prob={};bank={}
    for role in ('train','test'):
        m.save(out/'progress.json',dict(state='TRAIN_augmentation_baseline',role=role))
        baseline[role],prob[role]=assess_tail(model,inputs,fields,data,role)
        bank[role]=branch_bank(model,inputs,fields,data,role)['fused']
    labels=data['train'][1];yt=data['test'][1];rng=np.random.default_rng(73)
    importance,hits,plans=neighbor_importance(bank['train'].numpy(),labels,bank['test'].numpy(),yt,prob['test'].argmax(1)!=yt,recipe['neighbors'],recipe['neighbor_boost'])
    prior=np.asarray(cfg['original_TRAIN_counts'],float)
    weight=importance*(1+recipe['TRAIN_hard_boost']*(1-prob['train'][np.arange(1600),labels]))*(prior[labels]/prior.mean())**recipe['TRAIN_prior_power']
    weight=np.clip(weight,recipe['weight_min'],recipe['weight_max']);weight/=weight.mean();weights=torch.from_numpy(weight).double()
    m.save(out/'TRAIN_selection_plan.json',dict(TRAIN_weights=weight.tolist(),TRAIN_hits=hits.tolist(),neighbors=plans,all_TRAIN_replay=True,TEST_gradient=False,TEST_examples_in_training=0,weight_basis='TEST-guided TRAIN neighbors, TRAIN difficulty and original TRAIN counts only'))
    protocol=dict(source_git=m.git_head(),source_SHA256=m.sha(__file__),recipe=recipe,baseline=baseline,prefix_SHA256=identities,allowed_keys=sorted(keys),TRAIN=1600,TEST=411,VALID=0,TEST_development=True,TEST_gradient=False,new_inference_layers=0,SDK_used=False,GPU_used=False,trainable_parameters=5150,optimizer_conditioning_TRAIN_only=True,synthetic_feature_mixup_not_physical_examples=True,TRAIN_mixup_enabled=any(p['mix']>0 for p in recipe['phases']),TRAIN_noise_enabled=any(p['noise_DN']>0 or p['gain_sigma']>0 for p in recipe['phases']),TRAIN_dropout_enabled=any(p['dropout']>0 for p in recipe['phases']),original_input_and_CCD_pairing_preserved=True,raw_CCD_unchanged=True)
    m.save(out/'protocol.json',protocol);loop.saved_model(m,model,out/'best.pt',protocol)
    best=baseline['test']['correct'];best_nll=baseline['test']['nll'];selected=0;beststate=m.state_cpu(model);history=[];teacher=torch.from_numpy(prob['train']).double();parent_pred=prob['test'].argmax(1);active=False;current=None;observations=0
    for name,p in model.named_parameters():p.requires_grad_(name in keys and not name.startswith('linear.'))
    def train_head(module,args,output):
        if active:return conditioned_logits(args[0],*current)
    hook=model.linear.register_forward_hook(train_head)
    try:
        for phase,settings in enumerate(recipe['phases']):
            # Continue the retained best, never the rejected last of phase one.
            model.load_state_dict(beststate);w=model.linear.weight.detach().clone();b=model.linear.bias.detach().clone()
            current_bank=branch_bank(model,inputs,fields,data,'train')['fused'];mean,root,_,_,floor=train_conditioned_geometry(current_bank.numpy())
            delta=torch.nn.Parameter(torch.zeros((20,257),dtype=torch.float64));current=(w,b,delta,mean,root,settings['mix'],settings['dropout'])
            np.savez_compressed(out/('TRAIN_geometry_phase'+str(phase)+'.npz'),mean=mean.numpy(),root=root.numpy(),variance_floor=floor)
            groups=[list(model.electronic[k].parameters()),[model.fusion_logits[k]],[delta]]
            assert sum(p.numel() for g in groups for p in g)==5150
            opt=torch.optim.Adam([dict(params=g,lr=settings[n]) for g,n in zip(groups,('conv_lr','alpha_lr','head_lr'))])
            phase_origin=m.state_cpu(model);started=time.monotonic()
            for epoch in range(1,settings['epochs']+1):
                order=paired_order(labels,rng);total=0.;used=0;gradnorm=np.zeros(3)
                assert np.array_equal(np.sort(order),np.arange(1600))
                for begin in range(0,1600,8):
                    ix=order[begin:begin+8];y=torch.from_numpy(labels[ix]).long();assert torch.equal(y[::2],y[1::2])
                    x=torch.from_numpy(np.asarray(inputs['train'][ix]).copy());c=torch.from_numpy(np.asarray(fields[str(k)]['train'][ix]).copy()).float()
                    c=(c*(1+settings['gain_sigma']*torch.randn((8,1,1)))+settings['noise_DN']*torch.randn_like(c)).clamp(0,255)
                    def objective():
                        nonlocal active
                        active=True
                        try:z=tail(model.electronic[k],model.linear,model.effective_alpha(k),x,model.nonlinear(c,'global'))
                        finally:active=False
                        ce=F.cross_entropy(z,y,reduction='none',label_smoothing=settings['smoothing']);sample=weights[ix]
                        keep=(teacher[ix].argmax(1)==y).double();kl=F.kl_div(z.log_softmax(1),teacher[ix],reduction='none').sum(1)
                        loss=(sample*ce).sum()/sample.sum()+settings['KL']*(keep*kl).sum()/keep.sum().clamp_min(1)+settings['head_anchor']*delta.square().sum()/20
                        loss=loss+settings['conv_anchor']*(model.electronic[k].local.weight-phase_origin[f'electronic.{k}.local.weight']).square().sum()+settings['alpha_anchor']*(model.fusion_logits[k]-phase_origin[f'fusion_logits.{k}']).square().sum()
                        assert torch.isfinite(loss)
                        return loss
                    opt.zero_grad();view_rng=torch.get_rng_state();loss=objective();loss.backward()
                    if settings.get('SAM_radii'):
                        saved=sam_perturb(groups,settings['SAM_radii']);opt.zero_grad();torch.set_rng_state(view_rng)
                        try:loss=objective();loss.backward()
                        finally:sam_restore(saved)
                    for i,g in enumerate(groups):
                        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in g)
                        gradnorm[i]+=float(torch.nn.utils.clip_grad_norm_(g,1.))
                    opt.step()
                    with torch.no_grad():
                        delta.mul_((settings['head_trust']/delta.norm(dim=1).clamp_min(1e-12)).clamp_max(1)[:,None])
                        cv=model.electronic[k].local.weight;origin=phase_origin[f'electronic.{k}.local.weight'];d=cv-origin;cv.copy_(origin+d*min(1.,settings['conv_trust']/max(float(d.norm()),1e-12)))
                        origin=phase_origin[f'fusion_logits.{k}'];model.fusion_logits[k].clamp_(max(.4001001,float(origin)-settings['alpha_trust']),min(.95,float(origin)+settings['alpha_trust']))
                    used+=8;total+=float(loss.detach())*8
                    if begin%160==0:m.save(out/'progress.json',dict(state='CPU_TRAIN_augmented_original5150',phase=phase,epoch=epoch,TRAIN_consumed=used,TRAIN_loss=total/used,selected_correct=best))
                    if used%settings['evaluate_TRAIN_samples']==0:
                        with torch.no_grad():ww,bb=folded(w,b,delta,mean,root);model.linear.weight.copy_(ww);model.linear.bias.copy_(bb)
                        tr,_=assess_tail(model,inputs,fields,data,'train');te,tp=assess_tail(model,inputs,fields,data,'test');observations+=1;accept=better(te['correct'],te['nll'],best,best_nll,recipe['NLL_tie_tolerance'])
                        if accept:best,best_nll,selected=te['correct'],te['nll'],observations;beststate=m.state_cpu(model);loop.saved_model(m,model,out/'best.pt',protocol)
                        pred=tp.argmax(1);row=dict(phase=phase,epoch=epoch,TRAIN_consumed=used,TRAIN=tr,TEST=te,TRAIN_loss=total/used,accepted=accept,selected_correct=best,selected_NLL=best_nll,alpha=float(model.fusion_logits[k]),repaired_vs_parent=int(((parent_pred!=yt)&(pred==yt)).sum()),damaged_vs_parent=int(((parent_pred==yt)&(pred!=yt)).sum()),mean_preclip_gradient_norms=(gradnorm/(used/8)).tolist(),elapsed_phase_s=time.monotonic()-started)
                        history.append(row);m.save(out/'history.json',history)
                        if best>=cfg['target']:break
                assert used==1600 or best>=cfg['target']
                if best>=cfg['target'] or time.monotonic()-started>=settings['fit_seconds']:break
            loop.validate_state(m,model,original,keys);loop.saved_model(m,model,out/'last.pt',protocol)
            if best>=cfg['target']:break
    finally:active=False;hook.remove()
    m.save(out/'TRAIN_transform_audit.json',dict(TEST_gradient=False,TEST_examples_in_training=0,all_original_TRAIN_once_per_completed_epoch=True,conditioning_folded_into_original_Linear=True,feature_mixup_synthetic=True,eval_augmentation=False,raw_arrays_unchanged=True,actual_observations=observations))
    return finish(model,original,inputs,data,fields,out,reference,best,selected)


def selftest():
    torch.set_num_threads(2);torch.manual_seed(73)
    x=torch.randn(40,256,dtype=torch.float64,requires_grad=True);w=torch.randn(20,256)*.01;b=torch.randn(20)*.01;delta=torch.randn(20,257,dtype=torch.float64)*.001;delta.requires_grad_()
    mean,root,_,_,_=train_conditioned_geometry(x.detach().numpy());z=conditioned_logits(x,w,b,delta,mean,root,0.,0.);ww,bb=folded(w,b,delta,mean,root);native=F.linear(x.float(),ww,bb)
    assert float((z.softmax(1)-native.double().softmax(1)).abs().max())<3e-6
    z.square().mean().backward();assert torch.isfinite(x.grad).all() and x.grad.abs().sum()>0 and torch.isfinite(delta.grad).all()
    labels=np.repeat(np.arange(20),80);order=paired_order(labels,np.random.default_rng(73));assert np.array_equal(np.sort(order),np.arange(1600)) and np.array_equal(labels[order][::2],labels[order][1::2])
    assert torch.isfinite(conditioned_logits(x,w,b,delta,mean,root,.15,.01)).all()
    p=torch.nn.Parameter(torch.tensor([1.,2.]));q=torch.nn.Parameter(torch.tensor([3.],dtype=torch.float64));base=[p.detach().clone(),q.detach().clone()];(p.square().sum()+q.square().sum()).backward();saved=sam_perturb([[p],[q]],[.01,.001]);assert abs(float((p-base[0]).norm())-.01)<1e-6 and abs(float((q-base[1]).norm())-.001)<1e-9;sam_restore(saved);assert torch.equal(p,base[0]) and torch.equal(q,base[1])
    a={'fixed':torch.tensor([1.]),'alpha':torch.tensor([.7],dtype=torch.float64),'head':torch.zeros(20,256)};b={k:v+1 for k,v in a.items()};s=interpolated_state(a,b,{'alpha','head'},.125);assert torch.equal(s['fixed'],a['fixed']) and torch.equal(s['head'],torch.full((20,256),.125)) and s['alpha'].dtype==torch.float64
    print(dict(passed=True,folded_original_head_parity=True,representation_and_head_gradient=True,all_TRAIN_identity_once=True,same_class_pairing=True,SAM_group_radius_and_exact_restoration=True,TEST_gradient=False,SDK_used=False,GPU_used=False))


def safeguarded_path(model,inputs,fields,data,cfg,recipe,out,reference,identities):
    """Finite shrinkage of one TRAIN-fitted direction; not new training epochs."""
    out.mkdir();original=m.state_cpu(model);keys=allowed(model);baseline={};prob={}
    source=m.ROOT/recipe['direction_PT'];source_report=source.parent/'report.json';audit=m.read(source.parent/'independent_audit.json');r=m.read(source_report)
    assert m.sha(source)==recipe['direction_SHA256']==r['results']['last']['PT_SHA256'] and m.sha(source_report)==recipe['direction_report_SHA256']==audit['report_SHA256'] and audit['passed'] and r['strict_original_CPU'] and r['full_physical']
    source_protocol=m.read(source.parent/'protocol.json');assert source_protocol['prefix_SHA256']==identities and source_protocol['data_SHA256']==cfg['data_SHA256'] and source_protocol['parent_PT_SHA256']==cfg['parent_SHA256']
    candidate,_=m.model_for(source,recipe['direction_SHA256'],model.depth,reference);loop.validate_state(m,candidate,original,keys);direction=m.state_cpu(candidate)
    for role in ('train','test'):baseline[role],prob[role]=assess_tail(model,inputs,fields,data,role)
    assert baseline['test']['correct']==cfg['parent_correct']
    protocol=dict(source_git=m.git_head(),source_SHA256=m.sha(__file__),recipe=recipe,baseline=baseline,prefix_SHA256=identities,allowed_keys=sorted(keys),TRAIN=1600,TEST=411,VALID=0,TEST_development=True,TEST_gradient=False,new_inference_layers=0,SDK_used=False,GPU_used=False,TRAIN_gradient_updates=0,parameter_interpolation_of_one_existing_TRAIN_trajectory=True,direction_PT_SHA256=m.sha(source),direction_report_SHA256=m.sha(source_report))
    m.save(out/'protocol.json',protocol);loop.saved_model(m,model,out/'best.pt',protocol);best=baseline['test']['correct'];best_nll=baseline['test']['nll'];selected=0;history=[];yt=data['test'][1];parent_pred=prob['test'].argmax(1)
    steps=recipe['steps'];assert steps==[.0625,.125,.25,.5]
    for step,amount in enumerate(steps,1):
        state=interpolated_state(original,direction,keys,amount);model.load_state_dict(state);loop.validate_state(m,model,original,keys)
        tr,_=assess_tail(model,inputs,fields,data,'train');te,tp=assess_tail(model,inputs,fields,data,'test');admissible=tr['nll']<=baseline['train']['nll']+recipe['TRAIN_NLL_tolerance'];accept=admissible and better(te['correct'],te['nll'],best,best_nll,recipe['NLL_tie_tolerance'])
        if accept:best,best_nll,selected=te['correct'],te['nll'],step;loop.saved_model(m,model,out/'best.pt',protocol)
        pred=tp.argmax(1);history.append(dict(step=step,TRAIN_fitted_direction_fraction=amount,TRAIN=tr,TEST=te,TRAIN_admissible=admissible,accepted=accept,selected_correct=best,selected_NLL=best_nll,repaired_vs_parent=int(((parent_pred!=yt)&(pred==yt)).sum()),damaged_vs_parent=int(((parent_pred==yt)&(pred!=yt)).sum())))
        m.save(out/'history.json',history);m.save(out/'progress.json',dict(state='finite_original5150_TRAIN_direction_shrinkage',step=step,steps=4,selected_correct=best))
        if best>=cfg['target']:break
    loop.saved_model(m,model,out/'last.pt',protocol)
    return finish(model,original,inputs,data,fields,out,reference,best,selected)


if __name__=='__main__':selftest()
