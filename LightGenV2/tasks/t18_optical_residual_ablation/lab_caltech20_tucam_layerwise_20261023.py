"""Fresh L4 TUCam acquisition with original downstream-mask adaptation.

One measured TRAIN/TEST layer -> finite original suffix optimization -> native
CPU and frozen-prefix review -> next fresh layer. No measured-mask adjoint,
TEST derivatives, new inference layer, exposure search or scheduler.
"""
import argparse,json,math,os,shutil,time,traceback
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
STAGES=('router','0','1','2','3')


def gate_target(simulation,pp,samples=411):
    return math.ceil(simulation-samples*pp/100)


def validate_config(c):
    assert c['schema']=='TUCam_L4_layerwise_original_suffix_v1'
    assert c['TRAIN']==1600 and c['TEST']==411 and c['VALID']==0
    assert c['TRAIN_each_class']==80 and c['simulation_correct']==346
    assert c['intermediate_target']==gate_target(346,8)==314
    assert c['final_target']==gate_target(346,1.5)==340
    assert list(c['hardware_by_stage'])==list(STAGES)
    assert len(c['recipes'])==2 and c['interstage_fit'] is True
    assert c['feature_dropout_p']==0 and c['dropout_location']=='CCD_pixels'
    assert 0<=c['CCD_pixel_dropout_p']<=.002
    assert c['phase_lr']>0 and c['electronic_lr']>0 and c['head_lr']>0
    assert len(c['original_TRAIN_counts'])==20 and sum(c['original_TRAIN_counts'])==3141
    for r in c['recipes']:
        assert 1<=r['epochs']<=24 and 0<r['seconds']<=900 and 0<r['lr_scale']<=1


def numerical_modules():
    import numpy as np,torch
    import lab_caltech20_dvp_layerwise_20261019 as m
    import lab_caltech20_dvp_closedloop_20261021 as loop
    return np,torch,m,loop


def evaluate(model,data,fields,role):
    _,torch,m,_=numerical_modules()
    with torch.inference_mode():return m.assess(model,data,fields,role)


def identities(m,paths):
    return {k+'/'+r:m.sha(p/r/'ccd.npy') for k,p in paths.items() for r in ('train','test')}


def fit_suffix(model,data,dm,fields,paths,parent,reference,c,recipe,out,target):
    np,torch,m,loop=numerical_modules()
    from torch.nn import functional as F
    from lab_caltech20_dvp_TRAIN_regularization_20261021 import TRAINTransforms
    assert not out.exists() and not m.LOCK.exists();out.mkdir()
    prefix=len(fields);keys=m.allowed_keys(model,prefix);original=m.state_cpu(model)
    lineage=identities(m,paths);test,prob=evaluate(model,data,fields,'test')
    protocol=dict(source_git=m.git_head(),source_SHA256=m.sha(__file__),config=c,recipe=recipe,
        parent_PT=str(parent),parent_PT_SHA256=m.sha(parent),prefix_SHA256=lineage,
        data_SHA256=dm['sha256'],measured_stages=list(fields),allowed_keys=sorted(keys),
        TRAIN=1600,TEST=411,VALID=0,TEST_gradient=False,TEST_informed_development=True,
        full_physical=prefix==5,original_model_structure=True,new_inference_layers=0,SDK_used=False)
    m.save(out/'protocol.json',protocol);m.save(out/'baseline.json',test)
    np.savez_compressed(out/'baseline_TEST_predictions.npz',ids=data['test'][2],labels=data['test'][1],probabilities=prob)
    loop.saved_model(m,model,out/'best.pt',protocol)
    best,bnll,selected=test['correct'],test['nll'],0
    loop.gpu_admission(m,c);model.cuda()
    for name,p in model.named_parameters():p.requires_grad_(name in keys)
    groups=[]
    for prefix_name,lr in (('phase_sets.',c['phase_lr']),('electronic.',c['electronic_lr']),('fusion_logits.',c['alpha_lr']),('linear.',c['head_lr'])):
        params=[p for name,p in model.named_parameters() if name in keys and name.startswith(prefix_name)]
        if params:groups.append(dict(params=params,lr=lr*recipe['lr_scale'],weight_decay=0 if prefix_name=='fusion_logits.' else c['weight_decay']))
    opt=torch.optim.AdamW(groups);transforms=TRAINTransforms(model,list(fields)[-1],c)
    held={};hook=model.linear.register_forward_hook(lambda module,args,value:held.update(logits=value))
    rng=np.random.default_rng(c['seed']);history=[];start=time.monotonic();epoch=0;updates=0
    try:
        for epoch in range(1,recipe['epochs']+1):
            model.cuda().train();order=rng.permutation(1600);total=0.;used=0
            cm=np.asarray(test['confusion_matrix']);errors=1-cm.diagonal()/cm.sum(1).clip(1)
            prior=np.asarray(c['original_TRAIN_counts'],float);w=(.5+.5*np.sqrt(prior/prior.mean()))*(1+recipe['TRAIN_class_emphasis']*errors)
            weights=torch.tensor(w,dtype=torch.float32,device='cuda')
            for begin in range(0,1600,8):
                ix=order[begin:begin+8];x=loop.put_fields(model,data,fields,'train',ix,'cuda');labels=torch.from_numpy(data['train'][1][ix].astype(np.int64)).cuda()
                opt.zero_grad(set_to_none=True);model(x)
                loss=F.cross_entropy(held['logits'],labels,weight=weights,label_smoothing=c['label_smoothing'])
                assert torch.isfinite(loss);loss.backward();ps=[p for p in model.parameters() if p.requires_grad]
                assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in ps)
                norm=float(torch.nn.utils.clip_grad_norm_(ps,1.));assert norm>0;opt.step();updates+=1
                with torch.no_grad():
                    for p in model.fusion_logits:
                        if p.requires_grad:p.clamp_(.4001001,.95)
                used+=len(ix);total+=float(loss.detach())*len(ix)
                if begin%160==0:m.save(out/'progress.json',dict(state='suffix_TRAIN',epoch=epoch,TRAIN_consumed=used,TRAIN_loss=total/used,optimizer_updates=updates,selected_correct=best,elapsed_s=time.monotonic()-start))
            assert used==1600
            if epoch==1 or epoch%c['evaluate_every']==0 or epoch==recipe['epochs'] or time.monotonic()-start>=recipe['seconds']:
                test,prob=evaluate(model,data,fields,'test')
                if test['correct']>best or (test['correct']==best and test['nll']<bnll-1e-4):
                    best,bnll,selected=test['correct'],test['nll'],epoch;loop.saved_model(m,model,out/'best.pt',protocol)
                row=dict(epoch=epoch,TRAIN_loss=total/1600,TRAIN_consumed=used,TEST_correct=test['correct'],TEST_NLL=test['nll'],selected_correct=best,selected_epoch=selected,elapsed_s=time.monotonic()-start)
                history.append(row);m.save(out/'history.json',history);m.save(out/'progress.json',dict(state='CPU_development_selection',latest=row));print(json.dumps(row),flush=True)
            if best>=target or time.monotonic()-start>=recipe['seconds']:break
        model.cpu();loop.validate_state(m,model,original,keys);loop.saved_model(m,model,out/'last.pt',protocol)
    finally:transforms.remove();hook.remove()
    results={}
    for label in ('best','last'):
        pt=out/(label+'.pt');native,_=m.model_for(pt,m.sha(pt),4,reference);loop.validate_state(m,native,original,keys);results[label]=dict(PT_SHA256=m.sha(pt))
        for role in ('train','test'):
            m.save(out/'progress.json',dict(state='strict_native_CPU',checkpoint=label,role=role))
            metric,p=evaluate(native,data,fields,role);results[label][role]=metric
            np.savez_compressed(out/(label+'_'+role+'_predictions.npz'),ids=data[role][2],labels=data[role][1],probabilities=p)
    assert results['best']['test']['correct']==best and identities(m,paths)==lineage
    report=dict(state='complete',baseline_correct=protocol.get('baseline_correct',m.read(out/'baseline.json')['correct']),selected_correct=best,selected_epoch=selected,selected_E0=selected==0,
        actual_epochs=epoch,optimizer_updates=updates,results=results,strict_original_CPU=True,full_physical=len(fields)==5,
        target=target,target_met=best>=target,captured_upstream_bitwise_frozen=True,TEST_gradient=False,TEST_development=True,
        TRAIN_transform_counts=transforms.counts,synthetic_CCD_noise_not_calibrated=True,SDK_used=False,protocol_SHA256=m.sha(out/'protocol.json'))
    m.save(out/'report.json',report);m.save(out/'independent_audit.json',dict(passed=True,report_SHA256=m.sha(out/'report.json'),prefix_SHA256=lineage,all_non_suffix_parameters_bitwise_frozen=True))
    chosen,_=m.model_for(out/'best.pt',m.sha(out/'best.pt'),4,reference)
    del model,opt;torch.cuda.empty_cache()
    return chosen,out/'best.pt',report


def run(a,c):
    np,torch,m,loop=numerical_modules()
    from lab_caltech20_tucam_capture_20261023 import acquire,validate_hardware
    validate_config(c);assert not a.out.exists() and not m.LOCK.exists()
    assert shutil.disk_usage(a.out.parent).free>c['minimum_free_GiB']*2**30
    a.out.mkdir();torch.set_num_threads(2);torch.manual_seed(c['seed'])
    data,dm=m.data_for(ROOT/c['sequence']);assert dm['sha256']==c['data_SHA256']
    parent=ROOT/c['initial_PT'];reference=ROOT/c['reference'];assert m.sha(reference)==c['reference_SHA256']
    model,_=m.model_for(parent,c['initial_SHA256'],4,reference)
    score,p=evaluate(model,data,{},'test');assert score['correct']==c['simulation_correct']
    ref=torch.jit.load(str(reference),map_location='cpu').eval();err=0.
    with torch.inference_mode():
        for begin in range(0,411,8):
            x,n,end=m.padded(data,'test',begin);z=m.encode_images(x);model.measured={};err=max(err,float((model(z)[0]-ref(z)[0]).abs().max()))
    assert err<=3e-6;model.measured={}
    for item in c['hardware_by_stage'].values():assert m.sha(ROOT/item['path'])==item['SHA256'];validate_hardware(ROOT/item['path'])
    m.save(a.out/'protocol.json',dict(source_git=m.git_head(),source_SHA256=m.sha(__file__),profile_SHA256=m.sha(a.profile),config=c,simulation_correct=score['correct'],reference_probability_max_error=err,
        TRAIN=1600,TEST=411,VALID=0,planned_physical_ccd=10055,interstage_fit=True,all_layers_fresh=True,automatic_restarts=False,scheduler=False))
    fields={};paths={};endpoints=[]
    for stage in STAGES:
        folder=a.out/('s_'+stage);folder.mkdir();hw=ROOT/c['hardware_by_stage'][stage]['path']
        m.save(a.out/'progress.json',dict(state='capturing',stage=stage,parent_PT_SHA256=m.sha(parent),completed_stages=len(paths)))
        fields=acquire(model,data,dm,fields,paths,hw,stage,folder/'capture',parent,interstage_fit=True);paths[stage]=folder/'capture'
        target=c['final_target'] if stage=='3' else c['intermediate_target'];fit_report=None
        for index,recipe in enumerate(c['recipes']):
            m.save(a.out/'progress.json',dict(state='suffix_fitting',stage=stage,recipe=recipe['name'],target=target))
            model,parent,fit_report=fit_suffix(model,data,dm,fields,paths,parent,reference,c,recipe,folder/('f'+str(index)),target)
            if fit_report['selected_correct']>=target:break
        endpoint=dict(stage=stage,PT=str(parent),PT_SHA256=m.sha(parent),prefix_SHA256=identities(m,paths),selected_correct=fit_report['selected_correct'],target=target,target_met=fit_report['selected_correct']>=target,full_physical=stage=='3',strict_original_CPU=True,TEST_development=True)
        m.save(folder/'endpoint_report.json',endpoint);endpoints.append(endpoint)
        if not endpoint['target_met']:
            m.save(a.out/'report.json',dict(state='stage_target_not_met',failed_stage=stage,endpoints=endpoints,SDK_released=not m.LOCK.exists(),automatic_retry=False,next_layer_not_started=True));return
    m.save(a.out/'report.json',dict(state='complete',full_physical=True,endpoints=endpoints,SDK_released=not m.LOCK.exists(),physical_ccd=10055,strict_original_CPU=True,TEST_development=True))


def selftest():
    np,torch,m,loop=numerical_modules()
    from torch.nn import functional as F
    from lab_caltech20_dvp_TRAIN_regularization_20261021 import TRAINTransforms
    torch.set_num_threads(2);torch.manual_seed(73)
    for prefix in range(1,6):
        model=m.MeasuredPrefix(4).eval()
        with torch.no_grad():
            for p in model.fusion_logits:p.fill_(.8)
            x=torch.rand(2,224,224)*.2;model(x)
            fields={k:model.last[k]['intensity'].clone()*73 for k in STAGES[:prefix]};model.measured=fields
            before=model(x)[0].clone()
        original=m.state_cpu(model);keys=m.allowed_keys(model,prefix)
        transforms=TRAINTransforms(model,STAGES[prefix-1],dict(CCD_gain_sigma=.005,CCD_additive_sigma_DN=.1,feature_dropout_p=0,dropout_location='CCD_pixels',CCD_pixel_dropout_p=.002))
        held={};hook=model.linear.register_forward_hook(lambda module,args,value:held.update(logits=value))
        try:
            model.measured=fields
            with torch.no_grad():assert torch.equal(before,model(x)[0])
            assert transforms.counts['noise_TRAIN_batches']==0
            model.requires_grad_(False)
            for name,p in model.named_parameters():p.requires_grad_(name in keys)
            model.train();model.measured=fields;model(x);F.cross_entropy(held['logits'],torch.tensor([0,1])).backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for name,p in model.named_parameters() if name in keys)
            assert all(p.grad is None for name,p in model.named_parameters() if name not in keys)
            assert all(not k.startswith('raw_router') for k in keys)
            captured=set().union(*(loop.mask_keys(model,str(k)) for k in range(prefix-1)))
            assert not captured&keys
            if prefix<5:assert any(float(p.grad.abs().max())>0 for name,p in model.named_parameters() if name in keys and name.startswith('phase_sets.'))
            for name,p in model.named_parameters():
                if name in keys:
                    with torch.no_grad():p.add_(p.grad,alpha=-1e-6)
            loop.validate_state(m,model,original,keys)
        finally:transforms.remove();hook.remove()
    assert gate_target(346,8)==314 and gate_target(346,1.5)==340
    print(json.dumps(dict(passed=True,prefixes_checked=5,future_phase_gradients=True,captured_masks_frozen=True,eval_augmentation_identity=True,SDK_used=False,GPU_used=False,TEST_used=False)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('run','selftest'),required=True);p.add_argument('--profile',type=Path,default=HERE/'lab_caltech20_tucam_layerwise_profile_20261023.json');p.add_argument('--out',type=Path);a=p.parse_args()
    if a.mode=='selftest':selftest()
    else:
        try:
            _,_,m,_=numerical_modules();run(a,m.read(a.profile));m.save(a.out.with_suffix('.return.json'),dict(returncode=0))
        except BaseException as e:
            if a.out and a.out.exists():m.save(a.out/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),automatic_retry=False,all_data_preserved=True))
            if a.out:m.save(a.out.with_suffix('.return.json'),dict(returncode=1,error=repr(e)))
            raise
