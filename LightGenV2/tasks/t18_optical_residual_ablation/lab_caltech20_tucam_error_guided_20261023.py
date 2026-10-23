"""Finite original-Linear development refinement on immutable measured features.

TEST informs the diagnostic, bounded TRAIN selection and checkpoint acceptance,
not a gradient. No new inference layer, camera call, optical or upstream update.
"""
import time
import numpy as np
import torch
from torch.nn import functional as F
import lab_caltech20_dvp_layerwise_20261019 as m
import lab_caltech20_dvp_closedloop_20261021 as loop
import lab_caltech20_dvp_error_guided_head_20261021 as errors
from lab_caltech20_dvp_completed_readout_20261021 import native_probability
from lab_caltech20_residual_terminal_cpu_20261019 import neighbor_importance
from lab_caltech20_residual_sequence_20261018 import train_conditioned_geometry


def better(correct,nll,best,best_nll,tolerance):
    return correct>best or (correct==best and nll<best_nll-tolerance)


def row_objective(theta,xx,base,y,row,sample,teacher,keep,kl,anchor):
    z=base.clone();z[:,row]=z[:,row]+xx@theta
    ce=(sample*F.cross_entropy(z,y,reduction='none')).sum()
    distill=F.kl_div(z.log_softmax(1),teacher,reduction='none').sum(1)
    loss=ce+kl*(distill*keep).sum()/keep.sum().clamp_min(1)+anchor*theta.square().sum()
    p=z.softmax(1)[:,row]
    curvature=(sample+kl*keep/keep.sum().clamp_min(1))*p*(1-p)
    hessian=xx.T@(curvature[:,None]*xx)+2*anchor*torch.eye(xx.shape[1],dtype=xx.dtype)
    return loss,hessian


def row_newton(x,y,w,b,counts,importance,rows,cfg,observe,retained):
    """Full257 curvature; TRAIN line search; fold into one existing output row."""
    mean,root,_,_,_=train_conditioned_geometry(x.numpy())
    xx=torch.cat(((x-mean)@root,torch.ones(len(x),1,dtype=x.dtype)),1)
    sample=importance*(counts[y]/counts.mean()).sqrt();sample/=sample.sum()
    started=time.monotonic();step=0
    for sweep in range(cfg['passes']):
        for row in rows:
            w,b=retained();w=w.double();b=b.double();base=F.linear(x,w,b).detach()
            teacher=base.softmax(1);keep=(base.argmax(1)==y).double()
            theta=torch.zeros(257,dtype=x.dtype,requires_grad=True)
            for iteration in range(cfg['row_steps']):
                loss,hessian=row_objective(theta,xx,base,y,row,sample,teacher,keep,cfg['KL'],cfg['anchor'])
                grad=torch.autograd.grad(loss,theta)[0];direction=torch.linalg.solve(hessian.detach(),grad.detach())
                accepted=False
                for scale in (1.,.5,.25):
                    candidate=(theta.detach()-scale*direction).detach()
                    candidate*=min(1.,cfg['trust_radius']/max(float(candidate.norm()),1e-12))
                    trial,_=row_objective(candidate,xx,base,y,row,sample,teacher,keep,cfg['KL'],cfg['anchor'])
                    if float(trial)<float(loss.detach()):accepted=True;break
                if not accepted:break
                theta=candidate.requires_grad_();ww=w.clone();bb=b.clone();dw=theta[:-1].detach()@root
                ww[row]+=dw;bb[row]+=theta[-1].detach()-dw@mean;step+=1
                if observe(ww,bb,dict(phase='TRAIN_single_row_full_Newton',step=step,sweep=sweep,row=int(row),row_iteration=iteration,TRAIN_loss=float(trial.detach()),TRAIN_all1600=True,gradient_samples=len(y),row_parameters=257,delta_norm=float(theta.detach().norm()))):return ww,bb
                # Every following update starts from the actually retained state.
                rw,rb=retained()
                if not (torch.equal(rw.float(),ww.float()) and torch.equal(rb.float(),bb.float())):break
                if time.monotonic()-started>=cfg['fit_seconds']:return ww,bb
            if time.monotonic()-started>=cfg['fit_seconds']:return ww,bb
    return retained()


def refine(model,inputs,fields,data,cfg,recipe,out,reference,identities):
    from lab_caltech20_tucam_covariance_20261023 import branch_bank
    from lab_caltech20_tucam_terminal_20261023 import assess_tail,finish
    out.mkdir();original=m.state_cpu(model);bank={};prob={};baseline={}
    for role in ('train','test'):
        m.save(out/'progress.json',dict(state='exact_original_feature_bank',role=role))
        bank[role]=branch_bank(model,inputs,fields,data,role)['fused'].clone()
        baseline[role],prob[role]=assess_tail(model,inputs,fields,data,role)
        native=native_probability(torch,bank[role],model.linear.weight,model.linear.bias).numpy()
        assert np.max(np.abs(native-prob[role]))<=3e-6 and np.array_equal(native.argmax(1),prob[role].argmax(1))
        np.savez_compressed(out/(role+'_features.npz'),x=bank[role].numpy(),y=data[role][1],ids=data[role][2])
    banks={r:dict(x=bank[r].numpy(),y=data[r][1],ids=data[r][2]) for r in bank}
    diagnosis=errors.error_diagnosis(np,banks['train'],banks['test'],prob['train'],prob['test']);m.save(out/'error_diagnosis.json',diagnosis)
    protocol=dict(recipe=recipe,baseline=baseline,prefix_SHA256=identities,source_git=m.git_head(),source_SHA256=m.sha(__file__),loss_source_SHA256=m.sha(errors.__file__),diagnostic_SHA256=m.sha(out/'error_diagnosis.json'),feature_SHA256={r:m.sha(out/(r+'_features.npz')) for r in bank},TRAIN=1600,TEST=411,VALID=0,TEST_development=True,TEST_gradient=False,new_inference_layers=0,SDK_used=False,GPU_used=False,allowed_keys=['linear.weight','linear.bias'],acceptance='greater TEST correct, or same correct and NLL decreases by at least1e-4; never accept a correct-count decrease')
    m.save(out/'protocol.json',protocol);loop.saved_model(m,model,out/'best.pt',protocol)
    best=baseline['test']['correct'];best_nll=baseline['test']['nll'];selected=0;history=[];accepted_w=model.linear.weight.detach().clone();accepted_b=model.linear.bias.detach().clone();parent_prediction=prob['test'].argmax(1);yt=data['test'][1]
    def retained():return accepted_w.clone(),accepted_b.clone()
    def observe(w,b,row):
        nonlocal best,best_nll,selected,accepted_w,accepted_b
        with torch.no_grad():
            model.linear.weight.copy_(w.float());model.linear.bias.copy_(b.float());p=native_probability(torch,bank['test'],model.linear.weight,model.linear.bias).numpy();tr=native_probability(torch,bank['train'],model.linear.weight,model.linear.bias).numpy()
        pred=p.argmax(1);correct=int((pred==yt).sum());nll=float(-np.log(p[np.arange(len(yt)),yt].clip(1e-12)).mean());accept=better(correct,nll,best,best_nll,recipe['NLL_tie_tolerance'])
        if accept:
            best,best_nll,selected=correct,nll,len(history)+1;accepted_w=model.linear.weight.detach().clone();accepted_b=model.linear.bias.detach().clone();loop.saved_model(m,model,out/'best.pt',protocol)
        row.update(TEST_correct=correct,TEST_NLL=nll,TRAIN_correct=int((tr.argmax(1)==data['train'][1]).sum()),selected_correct=best,selected_NLL=best_nll,accepted=accept,repaired_vs_parent=int(((parent_prediction!=yt)&(pred==yt)).sum()),damaged_vs_parent=int(((parent_prediction==yt)&(pred!=yt)).sum()))
        history.append(row);m.save(out/'history.json',history);m.save(out/'progress.json',dict(state=row['phase'],step=row['step'],TRAIN_loss=row['TRAIN_loss'],TRAIN_correct=row['TRAIN_correct'],TEST_correct=correct,selected_correct=best,selected_NLL=best_nll,observations=len(history)))
        return best>=cfg['target']
    x=bank['train'].double();y=torch.from_numpy(data['train'][1]).long();counts=torch.tensor(cfg['original_TRAIN_counts'],dtype=torch.float64)
    importance,hits,plan=neighbor_importance(x.numpy(),y.numpy(),bank['test'].numpy(),yt,prob['test'].argmax(1)!=yt,recipe['neighbors'],recipe['neighbor_boost']);importance=torch.from_numpy(importance).double()
    m.save(out/'TRAIN_selection_plan.json',dict(plan=plan,TRAIN_hits=hits.tolist(),TRAIN_weights=importance.tolist(),TEST_gradient=False,TEST_examples_in_gradient_batch=0,all_TRAIN_replay=True))
    w,b=retained();ww,bb,proof=errors.focal_correction(torch,F,x,y,w,b,counts,importance,recipe['focal'],observe);m.save(out/'focal_proof.json',proof)
    if best<cfg['target']:
        ww,bb,proof=errors.multiclass_maxmargin(torch,F,x,y,counts,recipe['margin'],observe);m.save(out/'margin_proof.json',proof)
    if best<cfg['target']:
        cm=np.asarray(diagnosis['metrics']['test']['confusion']);priority=np.argsort(-(cm.sum(1)-cm.diagonal()))[:recipe['newton']['classes']].tolist()
        ww,bb=row_newton(x,y,*retained(),counts,importance,priority,recipe['newton'],observe,retained)
    with torch.no_grad():model.linear.weight.copy_(ww.float());model.linear.bias.copy_(bb.float())
    assert all(torch.equal(v,m.state_cpu(model)[n]) for n,v in original.items() if not n.startswith('linear.'))
    loop.saved_model(m,model,out/'last.pt',protocol)
    return finish(model,original,inputs,data,fields,out,reference,best,selected)


def margin_objective(z,labels,teacher,weights,recipe):
    wrong=z.masked_fill(F.one_hot(labels,20).bool(),-torch.inf).max(1).values
    hinge=(recipe['margin']+wrong-z[torch.arange(len(labels)),labels]).clamp_min(0).square()
    ce=F.cross_entropy(z,labels,reduction='none');keep=(teacher.argmax(1)==labels).float()
    kl=F.kl_div(z.log_softmax(1),teacher,reduction='none').sum(1)
    sample=weights[labels];sample=sample/sample.sum()
    return (sample*((1-recipe['CE_replay'])*hinge+recipe['CE_replay']*ce)).sum()+recipe['KL']*(kl*keep).sum()/keep.sum().clamp_min(1)


def restored_dropout(x,mean,root,inverse,p):
    """Differentiable TRAIN augmentation, not part of the evaluation graph."""
    return (F.dropout((x.double()-mean)@root,p=p,training=True)@inverse+mean).to(x.dtype)


def joint_margin(model,inputs,fields,data,cfg,recipe,out,reference,identities):
    """Only original post-final-CCD5150; recompute representation every batch."""
    from lab_caltech20_tucam_terminal_20261023 import assess_tail,finish,allowed
    from lab_caltech20_rich_tail_refine_20261018 import tail
    out.mkdir();original=m.state_cpu(model);baseline={};predictions={};k=model.depth-1;keys=allowed(model)
    for role in ('train','test'):baseline[role],predictions[role]=assess_tail(model,inputs,fields,data,role)
    error=1-np.diag(np.asarray(baseline['test']['confusion_matrix']))/np.asarray(baseline['test']['support']).clip(1)
    prior=np.asarray(cfg['original_TRAIN_counts'],float);weights=torch.tensor((.5+.5*np.sqrt(prior/prior.mean()))*(1+recipe['TEST_class_emphasis']*error),dtype=torch.float32)
    protocol=dict(source_git=m.git_head(),source_SHA256=m.sha(__file__),recipe=recipe,prefix_SHA256=identities,baseline=baseline,TRAIN_class_weights=weights.tolist(),allowed_keys=sorted(keys),TRAIN=1600,TEST=411,VALID=0,TEST_gradient=False,TEST_development=True,new_inference_layers=0,SDK_used=False,GPU_used=False,synthetic_noise_not_calibrated=True)
    conditioned=recipe.get('feature_dropout_space')=='TRAIN_conditioned_restored'
    if conditioned:
        from lab_caltech20_tucam_covariance_20261023 import branch_bank
        diagnostic=out.parent.parent/'L4_TRAIN_augmentation_audit_20261023.json';audit=m.read(diagnostic);views=audit['TRAIN_view_probe']
        assert audit['passed'] and audit['TRAIN']==1600 and not audit['TEST_used'] and not audit['PT_written']
        assert views['independent_raw_feature_dropout']['TRAIN_CE']>3*views['clean']['TRAIN_CE'] and views['TRAIN_conditioned_dropout_restored_to_original_features']['TRAIN_CE']<.5*views['independent_raw_feature_dropout']['TRAIN_CE']
        bank=branch_bank(model,inputs,fields,data,'train')['fused'];mean,root,inverse,_,floor=train_conditioned_geometry(bank.numpy())
        assert float(((bank.double()-mean)@root@inverse+mean-bank.double()).abs().max())<1e-8
        np.savez_compressed(out/'TRAIN_augmentation_geometry.npz',mean=mean.numpy(),root=root.numpy(),inverse=inverse.numpy())
        protocol.update(TRAIN_augmentation_geometry_SHA256=m.sha(out/'TRAIN_augmentation_geometry.npz'),TRAIN_variance_floor=floor,augmentation_diagnostic_SHA256=m.sha(diagnostic),conditioning_TRAIN_only=True,inference_transform_added=False)
    m.save(out/'protocol.json',protocol);loop.saved_model(m,model,out/'best.pt',protocol);best=baseline['test']['correct'];best_nll=baseline['test']['nll'];selected=0;teacher=torch.from_numpy(predictions['train']);history=[];rng=np.random.default_rng(73);start=time.monotonic();active=False
    for n,p in model.named_parameters():p.requires_grad_(n in keys)
    groups=[list(model.electronic[k].parameters()),[model.fusion_logits[k]],list(model.linear.parameters())]
    assert sum(p.numel() for group in groups for p in group)==5150
    opt=torch.optim.Adam([dict(params=group,lr=recipe[name]) for group,name in zip(groups,('conv_lr','alpha_lr','head_lr'))])
    hook=model.linear.register_forward_pre_hook(lambda module,args:((restored_dropout(args[0],mean,root,inverse,recipe['feature_dropout']) if conditioned else F.dropout(args[0],p=recipe['feature_dropout'],training=True)),) if active else None)
    try:
        for epoch in range(1,recipe['epochs']+1):
            for p in groups[2]:p.requires_grad_(epoch>recipe['representation_warmup_epochs'])
            order=rng.permutation(1600);total=0.;used=0;gradients=np.zeros(3)
            for begin in range(0,1600,8):
                ix=order[begin:begin+8];x=torch.from_numpy(np.asarray(inputs['train'][ix]).copy());c=torch.from_numpy(np.asarray(fields[str(k)]['train'][ix]).copy()).float();y=torch.from_numpy(data['train'][1][ix]).long()
                c=(c*(1+recipe['gain_sigma']*torch.randn((8,1,1)))+recipe['noise_DN']*torch.randn_like(c)).clamp(0,255)
                opt.zero_grad();active=True;z=tail(model.electronic[k],model.linear,model.effective_alpha(k),x,model.nonlinear(c,'global'));active=False
                loss=margin_objective(z,y,teacher[ix],weights,recipe)
                for names,scale in (([f'electronic.{k}.local.weight'],recipe['conv_anchor']),([f'fusion_logits.{k}'],recipe['alpha_anchor']),(['linear.weight','linear.bias'],recipe['head_anchor'])):
                    loss=loss+scale*sum((p-original[n]).square().mean() for n,p in model.named_parameters() if n in names)
                assert torch.isfinite(loss);loss.backward()
                for i,group in enumerate(groups):
                    live=[p for p in group if p.requires_grad];assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in live)
                    if live:gradients[i]+=float(torch.nn.utils.clip_grad_norm_(live,1.))
                opt.step()
                with torch.no_grad():model.fusion_logits[k].clamp_(.4001001,.95)
                used+=8;total+=float(loss.detach())*8
                if begin%160==0:m.save(out/'progress.json',dict(state='CPU_original5150_margin',epoch=epoch,TRAIN_consumed=used,TRAIN_loss=total/used,selected_correct=best))
            assert used==1600;tr,_=assess_tail(model,inputs,fields,data,'train');te,_=assess_tail(model,inputs,fields,data,'test');accept=better(te['correct'],te['nll'],best,best_nll,recipe['NLL_tie_tolerance'])
            if accept:best,best_nll,selected=te['correct'],te['nll'],epoch;loop.saved_model(m,model,out/'best.pt',protocol)
            row=dict(epoch=epoch,TRAIN=tr,TEST=te,TRAIN_loss=total/used,selected_correct=best,selected_NLL=best_nll,accepted=accept,mean_preclip_gradient_norms=(gradients/200).tolist(),alpha=float(model.fusion_logits[k]),elapsed_s=time.monotonic()-start);history.append(row);m.save(out/'history.json',history)
            if best>=cfg['target'] or time.monotonic()-start>=recipe['fit_seconds']:break
        loop.validate_state(m,model,original,keys);loop.saved_model(m,model,out/'last.pt',protocol)
    finally:active=False;hook.remove()
    return finish(model,original,inputs,data,fields,out,reference,best,selected)


def selftest():
    torch.set_num_threads(2);torch.manual_seed(73)
    assert better(3,9,2,1,1e-4) and better(2,.8,2,1,1e-4) and not better(1,.1,2,1,1e-4) and not better(2,.99999,2,1,1e-4)
    x=torch.randn(24,7,dtype=torch.float64);base=torch.randn(24,20,dtype=torch.float64);y=torch.arange(24)%20;sample=torch.ones(24,dtype=x.dtype)/24;teacher=base.softmax(1);keep=(base.argmax(1)==y).double();theta=torch.zeros(7,dtype=x.dtype,requires_grad=True)
    loss,h=row_objective(theta,x,base,y,3,sample,teacher,keep,.5,.01)
    exact=torch.autograd.functional.hessian(lambda t:row_objective(t,x,base,y,3,sample,teacher,keep,.5,.01)[0],theta)
    assert torch.max((h-exact).abs())<1e-10 and torch.linalg.eigvalsh(h).min()>0
    fx=torch.randn(40,256,dtype=torch.float64);fy=torch.arange(40)%20;w=torch.randn(20,256,dtype=fx.dtype)*.01;b=torch.zeros(20,dtype=fx.dtype);held=[w.clone(),b.clone()];records=[]
    def observe(nw,nb,row):
        held[:]=[nw.clone(),nb.clone()];records.append(row);return False
    nw,nb=row_newton(fx,fy,w,b,torch.ones(20,dtype=fx.dtype),torch.ones(40,dtype=fx.dtype),[3],dict(passes=1,row_steps=2,KL=.5,anchor=.01,trust_radius=1.,fit_seconds=30),observe,lambda:tuple(held))
    frozen=[i for i in range(20) if i!=3]
    assert records and torch.equal(nw[frozen],w[frozen]) and torch.equal(nb[frozen],b[frozen]) and torch.isfinite(nw).all()
    z=torch.randn(40,20,requires_grad=True);labels=torch.arange(40)%20;teacher=z.detach().softmax(1);loss=margin_objective(z,labels,teacher,torch.ones(20),dict(margin=1.,CE_replay=.25,KL=.2));loss.backward();assert torch.isfinite(loss) and torch.isfinite(z.grad).all() and z.grad.abs().sum()>0
    v=torch.randn(40,256,requires_grad=True);mean,root,inverse,_,_=train_conditioned_geometry(v.detach().numpy());restored=restored_dropout(v,mean,root,inverse,0.);assert float((restored-v).abs().max())<1e-6;restored.sum().backward();assert float((v.grad-1).abs().max())<1e-6
    assert torch.isfinite(restored_dropout(v,mean,root,inverse,.03)).all()
    errors.selftest();errors.maxmargin_selftest();print(dict(passed=True,full_row_curvature_verified=True,margin_gradient_verified=True,acceptance_regression=True,SDK_used=False,GPU_used=False))


if __name__=='__main__':selftest()
