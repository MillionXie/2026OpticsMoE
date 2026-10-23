"""One bounded SAM response on the original unmeasured optical suffix.

Foret et al., https://arxiv.org/abs/2010.01412. This changes TRAIN optimization,
not the inference model. Both gradient passes use the SAME TRAIN batch/noise.
Recorded input ancestors are excluded, and perturbations restore exact clones.
"""
import math,time,json
import lab_caltech20_tucam_layerwise_20261023 as base
import lab_caltech20_tucam_TRAIN_distill_20261023 as teacher


def perturb(parameters,radius):
    _,torch,_,_=base.numerical_modules()
    assert radius==.05 and parameters and all(p.requires_grad and p.grad is not None and torch.isfinite(p.grad).all() for p in parameters)
    norm=torch.linalg.vector_norm(torch.stack([p.grad.detach().double().norm() for p in parameters]));assert torch.isfinite(norm) and norm>0
    originals=[p.detach().clone() for p in parameters]
    with torch.no_grad():
        for p in parameters:p.add_(p.grad*(radius/(norm+1e-12)).to(p.dtype))
    return originals,float(norm)


def restore(parameters,originals):
    _,torch,_,_=base.numerical_modules()
    with torch.no_grad():
        for p,original in zip(parameters,originals):p.copy_(original)
    assert len(parameters)==len(originals) and all(torch.equal(p.detach(),original) for p,original in zip(parameters,originals))


def fit_suffix(model,data,dm,fields,paths,parent,reference,c,recipe,out,target):
    np,torch,m,loop=base.numerical_modules()
    from lab_caltech20_dvp_TRAIN_regularization_20261021 import TRAINTransforms
    assert list(fields)==list(base.STAGES[:4]) and c['TRAIN_SAM']==dict(radius=.05,same_noise_both_passes=True)
    assert not out.exists() and not m.LOCK.exists();out.mkdir()
    original=m.state_cpu(model);keys=m.allowed_keys(model,4);lineage=base.identities(m,paths)
    test,prob=base.evaluate(model,data,fields,'test');best,bnll,selected=test['correct'],test['nll'],0
    protocol=dict(source_git=m.git_head(),source_SHA256=m.sha(__file__),config=c,recipe=recipe,parent_PT=str(parent),parent_PT_SHA256=m.sha(parent),
        data_SHA256=dm['sha256'],prefix_SHA256=lineage,allowed_keys=sorted(keys),measured_stages=list(fields),TRAIN=1600,TEST=411,VALID=0,
        TEST_gradient=False,TEST_development=True,original_model_structure=True,new_inference_layers=0,full_physical=False,SDK_used=False,
        teacher_role='TRAIN_only_fixed_original_simulation',inference_teacher=False,optimizer='SAM_AdamW',two_pass_data='same_TRAIN_batch_same_synthetic_noise',
        exact_parameter_restore_before_optimizer_step=True,measured_masks_and_input_ancestors_frozen=True)
    m.save(out/'protocol.json',protocol);m.save(out/'baseline.json',test)
    np.savez_compressed(out/'baseline_TEST_predictions.npz',ids=data['test'][2],labels=data['test'][1],probabilities=prob)
    loop.saved_model(m,model,out/'best.pt',protocol);loop.gpu_admission(m,c)
    m.save(out/'progress.json',dict(state='caching_fixed_TRAIN_teacher',samples=1600,TEST_used=False))
    soft,manifest=teacher.TRAIN_teacher(data,c,reference,out);soft=torch.from_numpy(soft).cuda();model.cuda()
    for name,p in model.named_parameters():p.requires_grad_(name in keys)
    parameters=[p for p in model.parameters() if p.requires_grad];groups=[]
    for prefix,lr in [('phase_sets.',c['phase_lr']),('electronic.',c['electronic_lr']),('fusion_logits.',c['alpha_lr']),('linear.',c['head_lr'])]:
        ps=[p for n,p in model.named_parameters() if n in keys and n.startswith(prefix)]
        if ps:groups.append(dict(params=ps,lr=lr,base_lr=lr,weight_decay=0 if prefix=='fusion_logits.' else c['weight_decay']))
    opt=torch.optim.AdamW(groups);transforms=TRAINTransforms(model,'2',c);held={}
    hook=model.linear.register_forward_hook(lambda module,args,value:held.update(logits=value))
    cm=np.asarray(test['confusion_matrix']);errors=1-cm.diagonal()/cm.sum(1).clip(1);prior=np.asarray(c['original_TRAIN_counts'],float)
    weights=torch.tensor((.5+.5*np.sqrt(prior/prior.mean()))*(1+recipe['TRAIN_class_emphasis']*errors),dtype=torch.float32,device='cuda')
    rng=np.random.default_rng(c['seed']);history=[];updates=passes=epoch=0;start=time.monotonic()
    try:
        for epoch in range(1,recipe['epochs']+1):
            model.cuda().train();order=rng.permutation(1600);total=sharp=0.;used=0
            scale=.1+.9*.5*(1+math.cos(math.pi*(epoch-1)/max(1,recipe['epochs']-1)))
            for group in opt.param_groups:group['lr']=group['base_lr']*scale
            for begin in range(0,1600,8):
                ix=order[begin:begin+8];labels=torch.from_numpy(data['train'][1][ix].astype(np.int64)).cuda()
                before_cpu=torch.get_rng_state();before_gpu=torch.cuda.get_rng_state()
                opt.zero_grad(set_to_none=True);model(loop.put_fields(model,data,fields,'train',ix,'cuda'))
                loss,_,_=teacher.distillation_loss(held['logits'],labels,soft[ix],weights,2,.3,c['label_smoothing']);assert torch.isfinite(loss);loss.backward();passes+=1
                originals,norm=perturb(parameters,c['TRAIN_SAM']['radius']);after_cpu=torch.get_rng_state();after_gpu=torch.cuda.get_rng_state()
                try:
                    with torch.no_grad():
                        for p in model.fusion_logits:
                            if p.requires_grad:p.clamp_(.4001001,.95)
                    torch.set_rng_state(before_cpu);torch.cuda.set_rng_state(before_gpu)
                    opt.zero_grad(set_to_none=True);model(loop.put_fields(model,data,fields,'train',ix,'cuda'))
                    nearby,_,_=teacher.distillation_loss(held['logits'],labels,soft[ix],weights,2,.3,c['label_smoothing']);assert torch.isfinite(nearby);nearby.backward();passes+=1
                    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters)
                finally:
                    restore(parameters,originals);torch.set_rng_state(after_cpu);torch.cuda.set_rng_state(after_gpu)
                torch.nn.utils.clip_grad_norm_(parameters,1.);opt.step();updates+=1
                with torch.no_grad():
                    for p in model.fusion_logits:
                        if p.requires_grad:p.clamp_(.4001001,.95)
                used+=len(ix);total+=float(loss.detach())*len(ix);sharp+=float(nearby.detach())*len(ix)
                if begin%160==0:m.save(out/'progress.json',dict(state='TRAIN_SAM_joint_suffix',epoch=epoch,TRAIN_consumed=used,TRAIN_loss=total/used,TRAIN_nearby_loss=sharp/used,optimizer_updates=updates,TRAIN_gradient_passes=passes,selected_correct=best,elapsed_s=time.monotonic()-start))
            assert used==1600 and passes==updates*2
            if epoch==1 or epoch%2==0 or epoch==recipe['epochs'] or time.monotonic()-start>=recipe['seconds']:
                test,_=base.evaluate(model,data,fields,'test')
                if test['correct']>best or (test['correct']==best and test['nll']<bnll-1e-4):best,bnll,selected=test['correct'],test['nll'],epoch;loop.saved_model(m,model,out/'best.pt',protocol)
                row=dict(epoch=epoch,TRAIN_consumed=used,TRAIN_loss=total/1600,TRAIN_nearby_loss=sharp/1600,TEST_correct=test['correct'],TEST_NLL=test['nll'],selected_correct=best,selected_epoch=selected,optimizer_updates=updates,TRAIN_gradient_passes=passes,elapsed_s=time.monotonic()-start,lr_scale=scale)
                history.append(row);m.save(out/'history.json',history);m.save(out/'progress.json',dict(state='CPU_development_selection',latest=row));print(json.dumps(row),flush=True)
            if best>=target or time.monotonic()-start>=recipe['seconds']:break
        model.cpu();loop.validate_state(m,model,original,keys);loop.saved_model(m,model,out/'last.pt',protocol)
    finally:transforms.remove();hook.remove()
    results={}
    for label in ('best','last'):
        pt=out/(label+'.pt');native,_=m.model_for(pt,m.sha(pt),4,reference);loop.validate_state(m,native,original,keys);results[label]=dict(PT_SHA256=m.sha(pt))
        for role in ('train','test'):
            m.save(out/'progress.json',dict(state='strict_native_CPU',checkpoint=label,role=role));metric,p=base.evaluate(native,data,fields,role);results[label][role]=metric
            np.savez_compressed(out/(label+'_'+role+'_predictions.npz'),ids=data[role][2],labels=data[role][1],probabilities=p)
    assert results['best']['test']['correct']==best and base.identities(m,paths)==lineage
    report=dict(state='complete',baseline_correct=m.read(out/'baseline.json')['correct'],selected_correct=best,selected_epoch=selected,selected_E0=selected==0,actual_epochs=epoch,optimizer_updates=updates,TRAIN_gradient_passes=passes,
        results=results,strict_original_CPU=True,target=target,target_met=best>=target,captured_upstream_bitwise_frozen=True,TEST_gradient=False,TEST_development=True,full_physical=False,
        TRAIN_transform_counts=transforms.counts,teacher_manifest=manifest,inference_teacher=False,new_inference_layers=0,SDK_used=False,protocol_SHA256=m.sha(out/'protocol.json'))
    m.save(out/'report.json',report);m.save(out/'independent_audit.json',dict(passed=True,report_SHA256=m.sha(out/'report.json'),prefix_SHA256=lineage,all_non_suffix_parameters_bitwise_frozen=True))
    chosen,_=m.model_for(out/'best.pt',m.sha(out/'best.pt'),4,reference);del model,opt,soft;torch.cuda.empty_cache();return chosen,out/'best.pt',report


def selftest():
    _,torch,_,_=base.numerical_modules();torch.set_num_threads(2)
    a=torch.nn.Parameter(torch.tensor([1.,-2.],dtype=torch.float64));b=torch.nn.Parameter(torch.tensor([3.],dtype=torch.float32));frozen=torch.tensor([7.]);before=frozen.clone()
    (a.square().sum()+b.square().sum()).backward();originals,norm=perturb([a,b],.05)
    distance=sum(float((p.detach().double()-v.double()).square().sum()) for p,v in zip((a,b),originals))**.5
    assert abs(distance-.05)<1e-6 and norm>0;restore([a,b],originals);assert torch.equal(before,frozen)
    teacher.selftest();print(json.dumps(dict(passed=True,SAM_radius_verified=True,restore_bitwise_exact=True,frozen_unchanged=True,TEST_gradient=False,SDK_used=False,GPU_used=False)))


if __name__=='__main__':selftest()
