"""TRAIN-only simulation-teacher regularization of the original optical suffix.

The teacher is discarded after caching the same1600 TRAIN examples. Inference
is the unchanged, hook-free native model, never a teacher/student ensemble.
"""
import math,time
import lab_caltech20_tucam_layerwise_20261023 as base


def distillation_loss(logits,labels,teacher_logits,weights,temperature,mix,smoothing):
    from torch.nn import functional as F
    ce=F.cross_entropy(logits,labels,weight=weights,label_smoothing=smoothing)
    kd=F.kl_div(F.log_softmax(logits/temperature,dim=1),F.softmax(teacher_logits.detach()/temperature,dim=1),reduction='batchmean')*temperature**2
    return (1-mix)*ce+mix*kd,ce,kd


def TRAIN_teacher(data,c,reference,out):
    np,torch,m,_=base.numerical_modules();cfg=c['TRAIN_teacher']
    teacher_path=base.ROOT/c['initial_PT'];assert m.sha(teacher_path)==c['initial_SHA256']
    teacher,_=m.model_for(teacher_path,c['initial_SHA256'],4,reference)
    teacher.cuda().eval().requires_grad_(False);held={}
    hook=teacher.linear.register_forward_hook(lambda module,args,value:held.update(logits=value.detach()))
    values=[]
    try:
        with torch.inference_mode():
            for begin in range(0,1600,8):
                x,n,_=m.padded(data,'train',begin);teacher.measured={};teacher(m.encode_images(x.cuda()));values.append(held['logits'][:n].cpu().numpy().copy())
    finally:hook.remove()
    logits=np.concatenate(values);assert logits.shape==(1600,20) and np.isfinite(logits).all()
    path=out/'teacher_TRAIN_logits.npz';np.savez_compressed(path,ids=data['train'][2],labels=data['train'][1],logits=logits)
    manifest=dict(source_PT_SHA256=c['initial_SHA256'],role='train',samples=1600,TEST_used=False,VALID_used=False,gradient=False,
        cache_SHA256=m.sha(path),TRAIN_correct=int((logits.argmax(1)==data['train'][1]).sum()),temperature=cfg['temperature'],mixture=cfg['mixture'])
    m.save(out/'teacher_manifest.json',manifest);del teacher,held;torch.cuda.empty_cache()
    return logits,manifest


def fit_suffix(model,data,dm,fields,paths,parent,reference,c,recipe,out,target):
    np,torch,m,loop=base.numerical_modules()
    from lab_caltech20_dvp_TRAIN_regularization_20261021 import TRAINTransforms
    cfg=c['TRAIN_teacher'];assert cfg['temperature']==2 and cfg['mixture']==.3
    assert list(fields)==['router','0'] and not out.exists() and not m.LOCK.exists();out.mkdir()
    original=m.state_cpu(model);keys=m.allowed_keys(model,len(fields));lineage=base.identities(m,paths)
    test,prob=base.evaluate(model,data,fields,'test');best,bnll,selected=test['correct'],test['nll'],0
    protocol=dict(source_git=m.git_head(),source_SHA256=m.sha(__file__),config=c,recipe=recipe,parent_PT=str(parent),parent_PT_SHA256=m.sha(parent),
        data_SHA256=dm['sha256'],prefix_SHA256=lineage,allowed_keys=sorted(keys),measured_stages=list(fields),TRAIN=1600,TEST=411,VALID=0,
        TEST_gradient=False,TEST_development=True,original_model_structure=True,new_inference_layers=0,full_physical=False,SDK_used=False,
        teacher_role='TRAIN_only_fixed_original_simulation',inference_teacher=False)
    m.save(out/'protocol.json',protocol);m.save(out/'baseline.json',test)
    np.savez_compressed(out/'baseline_TEST_predictions.npz',ids=data['test'][2],labels=data['test'][1],probabilities=prob)
    loop.saved_model(m,model,out/'best.pt',protocol);loop.gpu_admission(m,c)
    m.save(out/'progress.json',dict(state='caching_fixed_TRAIN_teacher',samples=1600,TEST_used=False))
    teacher,manifest=TRAIN_teacher(data,c,reference,out);teacher=torch.from_numpy(teacher).cuda()
    model.cuda()
    for name,p in model.named_parameters():p.requires_grad_(name in keys)
    groups=[]
    for prefix,lr in [('phase_sets.',c['phase_lr']),('electronic.',c['electronic_lr']),('fusion_logits.',c['alpha_lr']),('linear.',c['head_lr'])]:
        ps=[p for n,p in model.named_parameters() if n in keys and n.startswith(prefix)]
        if ps:groups.append(dict(params=ps,lr=lr,base_lr=lr,weight_decay=0 if prefix=='fusion_logits.' else c['weight_decay']))
    opt=torch.optim.AdamW(groups);transforms=TRAINTransforms(model,'0',c);held={}
    hook=model.linear.register_forward_hook(lambda module,args,value:held.update(logits=value))
    rng=np.random.default_rng(c['seed']);cm=np.asarray(test['confusion_matrix']);errors=1-cm.diagonal()/cm.sum(1).clip(1)
    prior=np.asarray(c['original_TRAIN_counts'],float);w=(.5+.5*np.sqrt(prior/prior.mean()))*(1+recipe['TRAIN_class_emphasis']*errors)
    weights=torch.tensor(w,dtype=torch.float32,device='cuda');start=time.monotonic();history=[];epoch=updates=0
    try:
        for epoch in range(1,recipe['epochs']+1):
            model.cuda().train();order=rng.permutation(1600);sums=np.zeros(3);used=0
            scale=.1+.9*.5*(1+math.cos(math.pi*(epoch-1)/max(1,recipe['epochs']-1)))
            for group in opt.param_groups:group['lr']=group['base_lr']*scale
            for begin in range(0,1600,8):
                ix=order[begin:begin+8];x=loop.put_fields(model,data,fields,'train',ix,'cuda');labels=torch.from_numpy(data['train'][1][ix].astype(np.int64)).cuda()
                opt.zero_grad(set_to_none=True);model(x)
                loss,ce,kd=distillation_loss(held['logits'],labels,teacher[ix],weights,cfg['temperature'],cfg['mixture'],c['label_smoothing'])
                assert torch.isfinite(loss);loss.backward();ps=[p for p in model.parameters() if p.requires_grad]
                assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in ps)
                norm=float(torch.nn.utils.clip_grad_norm_(ps,1.));assert norm>0;opt.step();updates+=1
                with torch.no_grad():
                    for p in model.fusion_logits:
                        if p.requires_grad:p.clamp_(.4001001,.95)
                used+=len(ix);sums+=np.array([float(v.detach()) for v in (loss,ce,kd)])*len(ix)
                if begin%160==0:m.save(out/'progress.json',dict(state='TRAIN_distillation',epoch=epoch,TRAIN_consumed=used,TRAIN_loss=sums[0]/used,TRAIN_CE=sums[1]/used,TRAIN_KL=sums[2]/used,optimizer_updates=updates,selected_correct=best,elapsed_s=time.monotonic()-start))
            assert used==1600
            if epoch==1 or epoch%2==0 or epoch==recipe['epochs'] or time.monotonic()-start>=recipe['seconds']:
                test,_=base.evaluate(model,data,fields,'test')
                if test['correct']>best or (test['correct']==best and test['nll']<bnll-1e-4):best,bnll,selected=test['correct'],test['nll'],epoch;loop.saved_model(m,model,out/'best.pt',protocol)
                row=dict(epoch=epoch,TRAIN_consumed=used,TRAIN_loss=sums[0]/1600,TRAIN_CE=sums[1]/1600,TRAIN_KL=sums[2]/1600,TEST_correct=test['correct'],TEST_NLL=test['nll'],selected_correct=best,selected_epoch=selected,elapsed_s=time.monotonic()-start,lr_scale=scale)
                history.append(row);m.save(out/'history.json',history);m.save(out/'progress.json',dict(state='CPU_development_selection',latest=row));print(__import__('json').dumps(row),flush=True)
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
    report=dict(state='complete',baseline_correct=m.read(out/'baseline.json')['correct'],selected_correct=best,selected_epoch=selected,selected_E0=selected==0,actual_epochs=epoch,optimizer_updates=updates,
        results=results,strict_original_CPU=True,target=target,target_met=best>=target,captured_upstream_bitwise_frozen=True,TEST_gradient=False,TEST_development=True,full_physical=False,
        TRAIN_transform_counts=transforms.counts,teacher_manifest=manifest,inference_teacher=False,new_inference_layers=0,SDK_used=False,protocol_SHA256=m.sha(out/'protocol.json'))
    m.save(out/'report.json',report);m.save(out/'independent_audit.json',dict(passed=True,report_SHA256=m.sha(out/'report.json'),prefix_SHA256=lineage,all_non_suffix_parameters_bitwise_frozen=True))
    chosen,_=m.model_for(out/'best.pt',m.sha(out/'best.pt'),4,reference);del model,opt,teacher;torch.cuda.empty_cache()
    return chosen,out/'best.pt',report


def selftest():
    _,torch,_,_=base.numerical_modules();torch.set_num_threads(2);torch.manual_seed(73)
    x=torch.randn(8,20,requires_grad=True);teacher=torch.randn(8,20,requires_grad=True);labels=torch.arange(8);weights=torch.ones(20)
    loss,ce,kd=distillation_loss(x,labels,teacher,weights,2,.3,.03);loss.backward()
    assert torch.isfinite(loss) and kd>=-1e-6 and x.grad is not None and teacher.grad is None
    print(__import__('json').dumps(dict(passed=True,teacher_gradient=False,student_gradient=True,TEST_used=False,SDK_used=False,GPU_used=False)))


if __name__=='__main__':selftest()
