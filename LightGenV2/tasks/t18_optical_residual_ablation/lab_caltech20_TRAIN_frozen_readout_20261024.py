"""Finite lawful native suffix losses; readout freeze is explicitly configured."""
import math,time,json
import lab_caltech20_tucam_layerwise_20261023 as base


def contrast_logits(features,weight,bias):
    from torch.nn import functional as F
    return F.linear(features,weight-weight.mean(0,keepdim=True),bias-bias.mean())


def TRAIN_backend_audit(model,data,fields,parent,c,out):
    """Read-only native CPU/CUDA plus contrast-coordinate TRAIN parity."""
    np,torch,m,loop=base.numerical_modules();assert not out.exists() and not m.LOCK.exists();out.mkdir()
    original=m.state_cpu(model);labels=data['train'][1][:80];assert np.bincount(labels,minlength=20).tolist()==[4]*20
    def collect(device):
        model.to(device).eval().requires_grad_(False);values=[];features=[];canonical=[];contrast=[]
        with torch.inference_mode():
            for begin in range(0,80,8):
                ix=np.arange(begin,begin+8);prob=model(loop.put_fields(model,data,fields,'train',ix,device))[0];f=model.readout['features']
                values.append(prob.cpu().numpy().copy());features.append(f.cpu().numpy().copy());canonical.append(torch.round(model.last['3']['amplitude']*255).to(torch.uint8).cpu().numpy().copy());contrast.append(contrast_logits(f,model.linear.weight,model.linear.bias).softmax(1).cpu().numpy().copy())
                m.save(out/'progress.json',dict(state='TRAIN_backend_parity',device=device,TRAIN_consumed=begin+8,TEST_used=False))
        return tuple(np.concatenate(v)for v in (values,features,canonical,contrast))
    cpu=collect('cpu');loop.gpu_admission(m,c);cuda=collect('cuda');model.cpu();assert all(torch.equal(v,model.state_dict()[k])for k,v in original.items())
    report=dict(state='complete',source_git=m.git_head(),source_SHA256=m.sha(__file__),parent_PT_SHA256=m.sha(parent),TRAIN=80,TEST_used=False,VALID=0,optimizer_updates=0,
        CPU_correct=int((cpu[0].argmax(1)==labels).sum()),CUDA_correct=int((cuda[0].argmax(1)==labels).sum()),CPU_CUDA_prediction_disagreements=int((cpu[0].argmax(1)!=cuda[0].argmax(1)).sum()),
        CPU_CUDA_probability_maxabs=float(np.max(np.abs(cpu[0]-cuda[0]))),CPU_CUDA_feature_maxabs=float(np.max(np.abs(cpu[1]-cuda[1]))),G2_canonical_changed_pixel_fraction=float(np.mean(cpu[2]!=cuda[2])),
        CPU_original_vs_TRAIN_contrast_probability_maxabs=float(np.max(np.abs(cpu[0]-cpu[3]))),CUDA_original_vs_TRAIN_contrast_probability_maxabs=float(np.max(np.abs(cuda[0]-cuda[3]))),
        CPU_contrast_correct=int((cpu[3].argmax(1)==labels).sum()),CUDA_contrast_correct=int((cuda[3].argmax(1)==labels).sum()),original_state_bitwise_unchanged=True,SDK_used=False,GPU_used=True)
    m.save(out/'report.json',report);torch.cuda.empty_cache();return report


def safe_distillation_loss(logits,labels,teacher,weights):
    import torch
    from torch.nn import functional as F
    ce=F.cross_entropy(logits,labels,weight=weights,label_smoothing=.05);t=teacher.detach()
    gate=(t.argmax(1)==labels)&(t.softmax(1).amax(1)>=.5)
    def normalized(value):
        value=value-value.mean(1,keepdim=True);return value/value.square().mean(1,keepdim=True).sqrt().clamp_min(.5)
    per=F.kl_div(F.log_softmax(normalized(logits)/2,dim=1),F.softmax(normalized(t)/2,dim=1),reduction='none').sum(1)*4
    kd=(per*gate).sum()/gate.sum().clamp_min(1);effective=torch.minimum(ce.new_tensor(.1),.25*ce.detach()/kd.detach().clamp_min(1e-8))
    return ce+effective*kd,ce,kd,effective,gate.sum()


def selftest():
    import torch
    from torch.nn import functional as F
    torch.set_num_threads(2);torch.manual_seed(73);x=torch.randn(12,6,dtype=torch.float64,requires_grad=True);w=torch.randn(3,6,dtype=torch.float64)+1000*torch.randn(1,6,dtype=torch.float64);b=torch.randn(3,dtype=torch.float64)+2000;y=torch.arange(12)%3
    a=F.linear(x,w,b);v=contrast_logits(x,w,b);ga=torch.autograd.grad(F.cross_entropy(a,y),x,retain_graph=True)[0];gv=torch.autograd.grad(F.cross_entropy(v,y),x)[0]
    error=float((a.softmax(1)-v.softmax(1)).abs().max());grad=float((ga-gv).abs().max());assert error<1e-10 and grad<1e-9
    student=torch.randn(12,3,requires_grad=True);teacher=(torch.nn.functional.one_hot(y,3).float()*8).requires_grad_();loss,ce,kd,coefficient,n=safe_distillation_loss(student,y,teacher,torch.ones(3));loss.backward()
    assert teacher.grad is None and student.grad is not None and torch.isfinite(student.grad).all() and n==12 and float(coefficient)<=.100001 and float((coefficient*kd/ce).detach())<=.250001
    wrong=torch.nn.functional.one_hot((y+1)%3,3).float()*8;total,only_ce,_,_,count=safe_distillation_loss(student.detach(),y,wrong,torch.ones(3));assert count==0 and torch.allclose(total,only_ce)
    z=torch.tensor([-6.,-3.,-1.,0.,1.,3.,6.],requires_grad=True);penalty=((z.abs()-2).relu()/2).square().mean();penalty.backward()
    assert torch.isfinite(z.grad).all() and z.grad[0]<0 and z.grad[-1]>0 and torch.equal(z.grad[2:5],torch.zeros(3))
    print(json.dumps(dict(passed=True,frozen_readout_TRAIN_contrast_probability_maxabs=error,gradient_maxabs=grad,readout_parameters_changed=False,correct_teacher_gate=True,teacher_gradient=False,KD_value_fraction_of_CE_at_most=.25,saturation_penalty_gradient_toward_sensitive_region=True,SDK_used=False,GPU_used=False,TEST_used=False)))


def fit_suffix(model,data,dm,fields,paths,parent,reference,c,recipe,out,target):
    np,torch,m,loop=base.numerical_modules()
    from torch.nn import functional as F
    from lab_caltech20_dvp_TRAIN_regularization_20261021 import TRAINTransforms
    assert list(fields)==['router','0','1','2'] and not out.exists() and not m.LOCK.exists();out.mkdir();original=m.state_cpu(model)
    safe=c.get('TRAIN_safe_teacher',False);metric=c.get('TRAIN_supervised_metric',False);desaturate=c.get('TRAIN_desaturation',False)
    assert all(isinstance(v,bool)for v in (safe,metric,desaturate)) and sum((safe,metric,desaturate))<=1;joint=safe or metric or desaturate
    keys=m.allowed_keys(model,4)
    if not joint:keys=keys-{'linear.weight','linear.bias'}
    assert keys and (joint or all(not k.startswith('linear.') for k in keys))
    lineage=base.identities(m,paths);test,prob=base.evaluate(model,data,fields,'test');best,bnll,selected=test['correct'],test['nll'],0
    protocol=dict(source_git=m.git_head(),source_SHA256=m.sha(__file__),parent_PT=str(parent),parent_PT_SHA256=m.sha(parent),data_SHA256=dm['sha256'],prefix_SHA256=lineage,config=c,recipe=recipe,allowed_keys=sorted(keys),
        TRAIN=1600,TEST=411,VALID=0,TEST_gradient=False,TEST_development=True,full_physical=False,SDK_used=False,new_inference_layers=0,original_5140_readout_bitwise_frozen=not joint,TRAIN_contrast_logits_mathematically_equivalent=True)
    teacher=None
    if safe:
        folder=base.ROOT/c['TRAIN_teacher_cache'];manifest=m.read(folder/'teacher_manifest.json');cache=folder/'teacher_TRAIN_logits.npz'
        assert m.sha(folder/'teacher_manifest.json')=='761611b3cf1986254e77efdc0fc0a0c5044fdb46f3fd89fbfe9cfa837b220a98'
        assert manifest['role']=='train' and manifest['samples']==1600 and not manifest['TEST_used'] and not manifest['VALID_used']
        assert manifest['source_PT_SHA256']==c['initial_SHA256']=='7b0d7f055a16f7f6bbf9858c9ecaea572ecd0b0d6e96c618630f4a4e077c09a5'
        assert m.sha(cache)==manifest['cache_SHA256']=='c73a318e3887b00200c51814790c7874754d64df9190edb044557347e25d6069'
        with np.load(cache,allow_pickle=False) as z:
            assert np.array_equal(z['ids'],data['train'][2]) and np.array_equal(z['labels'],data['train'][1]);teacher=torch.from_numpy(z['logits'].copy())
        assert teacher.shape==(1600,20) and torch.isfinite(teacher).all();protocol.update(TRAIN_safe_teacher=True,teacher_cache_SHA256=m.sha(cache),teacher_source_PT_SHA256=manifest['source_PT_SHA256'],teacher_gate='TRAIN-correct and rawconfidence>=.5',teacher_temperature=2,normalized_logits=True,KD_coefficient_max=.1,KD_value_fraction_of_CE_max=.25,inference_teacher=False)
        m.save(out/'teacher_manifest.json',manifest)
    families=None
    if metric:
        from lab_caltech20_TRAIN_metric_20261024 import family_paired_order,loss as metric_loss
        family_path=base.ROOT/c['sequence']/'assets/TRAIN_lineage.json';assert m.sha(family_path)==dm['lineage_SHA256'];records=m.read(family_path)['records'];assert [v['sample_id']for v in records]==data['train'][2].tolist();families=np.asarray([v['source_id']for v in records])
        protocol.update(TRAIN_supervised_metric=True,TRAIN_family_lineage_SHA256=m.sha(family_path),TRAIN_pairs='four classes, two independent source families per class, every1600 exactly once',metric_temperature=.2,metric_coefficient_max=.05,metric_value_fraction_of_CE_max=.25,inference_metric=False)
    if desaturate:protocol.update(TRAIN_desaturation=True,TRAIN_penalty='mean square relu(abs(original postE2 conv pre-tanh)-2)/2; stages2,3 averaged',penalty_coefficient_max=.01,penalty_value_fraction_of_CE_max=.1,inference_penalty=False,parameter_reset=False)
    m.save(out/'protocol.json',protocol);m.save(out/'baseline.json',test);np.savez_compressed(out/'baseline_TEST_predictions.npz',ids=data['test'][2],labels=data['test'][1],probabilities=prob);loop.saved_model(m,model,out/'best.pt',protocol)
    loop.gpu_admission(m,c);model.cuda()
    if teacher is not None:teacher=teacher.cuda()
    for name,v in model.named_parameters():v.requires_grad_(name in keys)
    groups=[]
    for prefix,lr in (('phase_sets.',c['phase_lr']),('electronic.',c['electronic_lr']),('fusion_logits.',c['alpha_lr'])):
        ps=[v for name,v in model.named_parameters() if name in keys and name.startswith(prefix)]
        if ps:groups.append(dict(params=ps,lr=lr,base_lr=lr,weight_decay=0 if prefix=='fusion_logits.' else .01))
    if joint:groups.append(dict(params=list(model.linear.parameters()),lr=c['head_lr'],base_lr=c['head_lr'],weight_decay=.01))
    opt=torch.optim.AdamW(groups);held={};hook=model.linear.register_forward_pre_hook(lambda module,args:held.update(features=args[0]));transforms=TRAINTransforms(model,'2',c)
    saturation_hooks=[]
    if desaturate:
        for k in (2,3):saturation_hooks.append(model.electronic[k].local.register_forward_hook(lambda module,args,value,k=k:held.update({f'pre_tanh_{k}':value})))
    prior=np.asarray(c['original_TRAIN_counts'],float);cm=np.asarray(test['confusion_matrix']);errors=1-cm.diagonal()/cm.sum(1).clip(1);weights=torch.tensor((.5+.5*np.sqrt(prior/prior.mean()))*(1+.75*errors),dtype=torch.float32,device='cuda')
    rng=np.random.default_rng(73);torch.manual_seed(73);history=[];updates=0;start=time.monotonic()
    try:
        for epoch in range(1,recipe['epochs']+1):
            model.cuda().train();order=family_paired_order(data['train'][1],families,rng) if metric else rng.permutation(1600);used=0;total=0.;kd_total=ce_total=coefficient_total=eligible=0.;saturation_stats={k:dict(abs_gt3=0.,derivative=0.)for k in (2,3)};scale=.1+.9*.5*(1+math.cos(math.pi*(epoch-1)/max(1,recipe['epochs']-1)))
            for group in opt.param_groups:group['lr']=group['base_lr']*scale
            for begin in range(0,1600,8):
                ix=order[begin:begin+8];x=loop.put_fields(model,data,fields,'train',ix,'cuda');labels=torch.from_numpy(data['train'][1][ix].astype(np.int64)).cuda();opt.zero_grad(set_to_none=True);model(x)
                logits=contrast_logits(held['features'],model.linear.weight,model.linear.bias)
                if safe:
                    loss,ce,kd,coefficient,n=safe_distillation_loss(logits,labels,teacher[ix],weights);ce_total+=float(ce.detach())*len(ix);kd_total+=float((coefficient*kd).detach())*len(ix);coefficient_total+=float(coefficient)*len(ix);eligible+=int(n)
                elif metric:
                    ce=F.cross_entropy(logits,labels,weight=weights,label_smoothing=.05);pair_loss=metric_loss(held['features'],labels);coefficient=torch.minimum(ce.new_tensor(.05),.25*ce.detach()/pair_loss.detach().clamp_min(1e-8));loss=ce+coefficient*pair_loss
                    ce_total+=float(ce.detach())*len(ix);kd_total+=float((coefficient*pair_loss).detach())*len(ix);coefficient_total+=float(coefficient)*len(ix)
                elif desaturate:
                    ce=F.cross_entropy(logits,labels,weight=weights,label_smoothing=.05);penalty=sum(((held[f'pre_tanh_{k}'].abs()-2).relu()/2).square().mean()for k in (2,3))/2
                    coefficient=torch.minimum(ce.new_tensor(.01),.1*ce.detach()/penalty.detach().clamp_min(1e-8));loss=ce+coefficient*penalty
                    ce_total+=float(ce.detach())*len(ix);kd_total+=float((coefficient*penalty).detach())*len(ix);coefficient_total+=float(coefficient)*len(ix)
                    for k in (2,3):
                        z=held[f'pre_tanh_{k}'].detach();saturation_stats[k]['abs_gt3']+=float((z.abs()>3).float().mean())*len(ix);saturation_stats[k]['derivative']+=float((1-z.tanh().square()).mean())*len(ix)
                else:loss=F.cross_entropy(logits,labels,weight=weights,label_smoothing=.05)
                assert torch.isfinite(loss);loss.backward()
                ps=[v for v in model.parameters() if v.requires_grad];assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in ps);torch.nn.utils.clip_grad_norm_(ps,1.);opt.step();updates+=1
                with torch.no_grad():
                    for v in model.fusion_logits:
                        if v.requires_grad:v.clamp_(.4001001,.95)
                used+=len(ix);total+=float(loss.detach())*len(ix)
                if begin%160==0:m.save(out/'progress.json',dict(state='TRAIN_original_suffix_joint' if joint else 'TRAIN_original_suffix_frozen_readout',epoch=epoch,TRAIN_consumed=used,TRAIN_loss=total/used,optimizer_updates=updates,selected_correct=best))
            assert used==1600
            if epoch==1 or epoch%2==0 or epoch==recipe['epochs'] or time.monotonic()-start>=recipe['seconds']:
                test,_=base.evaluate(model,data,fields,'test')
                if test['correct']>best or (test['correct']==best and test['nll']<bnll-1e-4):best,bnll,selected=test['correct'],test['nll'],epoch;loop.saved_model(m,model,out/'best.pt',protocol)
                row=dict(epoch=epoch,TRAIN_consumed=1600,TRAIN_loss=total/1600,TEST_correct=test['correct'],TEST_NLL=test['nll'],selected_correct=best,selected_epoch=selected,elapsed_s=time.monotonic()-start)
                if safe:row.update(TRAIN_CE=ce_total/1600,TRAIN_weighted_KD=kd_total/1600,KD_mean_coefficient=coefficient_total/1600,teacher_eligible_TRAIN=eligible)
                if metric:row.update(TRAIN_CE=ce_total/1600,TRAIN_weighted_metric=kd_total/1600,metric_mean_coefficient=coefficient_total/1600,TRAIN_independent_family_pairs=800)
                if desaturate:row.update(TRAIN_CE=ce_total/1600,TRAIN_weighted_saturation_penalty=kd_total/1600,penalty_mean_coefficient=coefficient_total/1600,TRAIN_original_tanh_statistics={str(k):{n:v/1600 for n,v in row.items()}for k,row in saturation_stats.items()})
                history.append(row);m.save(out/'history.json',history);m.save(out/'progress.json',row);print(json.dumps(row),flush=True)
            if best>=target or time.monotonic()-start>=recipe['seconds']:break
        model.cpu();loop.validate_state(m,model,original,keys);loop.saved_model(m,model,out/'last.pt',protocol)
    finally:
        transforms.remove();hook.remove()
        for handle in saturation_hooks:handle.remove()
    results={}
    for label in ('best','last'):
        pt=out/(label+'.pt');native,_=m.model_for(pt,m.sha(pt),4,reference);loop.validate_state(m,native,original,keys);results[label]=dict(PT_SHA256=m.sha(pt))
        for role in ('train','test'):
            m.save(out/'progress.json',dict(state='strict_native_CPU',checkpoint=label,role=role));metric,v=base.evaluate(native,data,fields,role);results[label][role]=metric;np.savez_compressed(out/(label+'_'+role+'_predictions.npz'),ids=data[role][2],labels=data[role][1],probabilities=v)
    assert results['best']['test']['correct']==best and base.identities(m,paths)==lineage
    report=dict(state='complete',baseline_correct=m.read(out/'baseline.json')['correct'],selected_correct=best,selected_epoch=selected,selected_E0=selected==0,actual_epochs=epoch,optimizer_updates=updates,results=results,
        strict_original_CPU=True,target=target,target_met=best>=target,captured_upstream_bitwise_frozen=True,original_5140_readout_bitwise_frozen=not joint,TRAIN_safe_teacher=safe,TRAIN_supervised_metric=metric,TRAIN_desaturation=desaturate,TEST_gradient=False,TEST_development=True,full_physical=False,new_inference_layers=0,SDK_used=False,
        TRAIN_transform_counts=transforms.counts,synthetic_CCD_noise_not_calibrated=True,protocol_SHA256=m.sha(out/'protocol.json'))
    m.save(out/'report.json',report);m.save(out/'independent_audit.json',dict(passed=True,report_SHA256=m.sha(out/'report.json'),prefix_SHA256=lineage,all_non_suffix_parameters_bitwise_frozen=True,original_readout_frozen=not joint))
    chosen,_=m.model_for(out/'best.pt',m.sha(out/'best.pt'),4,reference);del model,opt,teacher;torch.cuda.empty_cache();return chosen,out/'best.pt',report
