"""Fixed original simulation TRAIN teacher; adapt only unmeasured original suffix.

No SDK, new inference graph, TEST teacher/gradient, changed measured mask,
response correction, LUT or imaging setting. This is TEST-informed development.
"""
import argparse,json,os,time,traceback
from pathlib import Path

def teacher_loss(F,student,features,labels,teacher,teacher_features,cfg):
    """Only TRAIN tensors. Reference confidence filter uses TRAIN labels."""
    logp=student.clamp_min(1e-12).log();smooth=cfg['label_smoothing'];ce=(1-smooth)*F.nll_loss(logp,labels)-smooth*logp.mean()
    temp=cfg['temperature'];target=(teacher.clamp_min(1e-12).log()/temp).softmax(1).detach();keep=(teacher.argmax(1)==labels).to(logp.dtype)
    kl=F.kl_div((logp/temp).log_softmax(1),target,reduction='none').sum(1)*temp**2
    per=(features-teacher_features.detach()).square().mean(1)
    distill=(kl*keep).sum()/keep.sum().clamp_min(1);align=(per*keep).sum()/keep.sum().clamp_min(1)
    return ce+cfg['teacher_KL']*distill+cfg['feature_alignment']*align,dict(CE=float(ce.detach()),teacher_KL=float(distill.detach()),feature_alignment=float(align.detach()))

def reference_model(m,torch,path,depth,digest):
    assert m.sha(path)==digest;script=torch.jit.load(str(path),map_location='cpu').eval();state=script.state_dict();model=m.MeasuredPrefix(depth).eval();keys=set(model.state_dict())
    assert set(state)-keys=={'detectors','router_masks','propagator.transfer'}
    model.load_state_dict({k:v for k,v in state.items() if k in keys},strict=True)
    with torch.no_grad():
        model.detectors.copy_(state['detectors']);model.router_masks.copy_(state['router_masks']);model.propagator.transfer.copy_(state['propagator.transfer'])
    model.requires_grad_(False);return model,script

def run(a,depth):
    import numpy as np,torch
    from torch.nn import functional as F
    import lab_caltech20_dvp_layerwise_20261019 as m
    import lab_caltech20_dvp_closedloop_20261021 as loop
    from lab_caltech20_dvp_TRAIN_regularization_20261021 import TRAINTransforms
    torch.set_num_threads(2);torch.manual_seed(73+depth);cfg=m.read(a.profile);assert cfg['schema']=='DVP_original_reference_TRAIN_suffix_v1';assert not m.LOCK.exists()
    loop.gpu_admission(m,cfg);out=a.sequence/f'TRAIN_restored_same_mask_pair_20261021/L{depth}/TRAIN_reference_teacher_suffix_20261021';assert not out.exists();out.mkdir()
    folder=out.parent;c=cfg['parents'][str(depth)];parent=folder/c['relative'];assert m.sha(parent)==c['SHA256'];rp=m.read(parent.parent/'report.json');cp=m.read(parent.parent/'protocol.json');assert rp['state']=='complete' and rp['strict_original_CPU'] and rp['selected_correct']==c['correct'] and rp['protocol_SHA256']==m.sha(parent.parent/'protocol.json')
    data,dm=m.data_for(a.sequence);assert dm['sha256']==cfg['data_SHA256'] and m.session_hardware_digest(a.sequence)==loop.HARDWARE
    spec=loop.SPECS[depth];session=a.sequence/f'L{depth}_response_20261020';fields,identities=m.fields_for(session,spec['prefix']-1,data,dm,loop.HARDWARE);capture=folder/'capture';cr=m.read(capture/'report.json');assert m.read(capture/'same_stimulus_audit.json')['matched_samples']==2011 and m.read(capture/'independent_audit.json')['report_SHA256']==m.sha(capture/'report.json')
    fields[spec['stage']]={}
    for role in ('train','test'):
        p=capture/role/'ccd.npy';assert m.sha(p)==cr['arrays'][role]['SHA256'];fields[spec['stage']][role]=np.load(p,mmap_mode='r');identities[spec['stage']+'/'+role]=m.sha(p)
    assert identities==cp['prefix_SHA256'];ref=session/'source_reference.ts';model,_=m.model_for(parent,c['SHA256'],depth,ref);original=m.state_cpu(model);allowed=m.allowed_keys(model,spec['prefix']);assert not allowed&loop.mask_keys(model,spec['stage'])
    baseline,bp=loop.original_replay(model,data,fields,'test');assert baseline['correct']==c['correct'];teacher,script=reference_model(m,torch,ref,depth,cfg['teacher_SHA256'][str(depth)])
    with torch.no_grad():
        sample=m.encode_images(data['train'][0][:8]);expected=script(sample)[0];actual=teacher(sample)[0];parity=float((expected-actual).abs().max());assert parity<=3e-6 and torch.equal(expected.argmax(1),actual.argmax(1))
    del script;teacher.cuda();prob=[];features=[]
    for begin in range(0,1600,cfg['batch']):
        end=min(begin+cfg['batch'],1600)
        with torch.no_grad():
            p=teacher(m.encode_images(data['train'][0][begin:end].cuda()))[0];prob.append(p.cpu().numpy());features.append(teacher.readout['features'].cpu().numpy())
        if end%80==0:m.save(out/'progress.json',dict(state='fixed_simulation_TRAIN_teacher_cache',consumed=end,TEST_teacher=False))
    tp,tf=np.concatenate(prob),np.concatenate(features);teacher_correct=int((tp.argmax(1)==data['train'][1]).sum());np.savez_compressed(out/'fixed_reference_TRAIN_teacher.npz',ids=data['train'][2],labels=data['train'][1],probabilities=tp,features=tf)
    del teacher;torch.cuda.empty_cache()
    protocol=dict(source_git=m.git_head(),source_SHA256=m.sha(__file__),profile_SHA256=m.sha(a.profile),config=cfg,depth=depth,TRAIN=1600,TEST=411,VALID=0,TEST_gradient=False,TEST_teacher=False,TEST_informed_development=True,parent_PT_SHA256=c['SHA256'],parent_report_SHA256=m.sha(parent.parent/'report.json'),data_SHA256=dm['sha256'],prefix_SHA256=identities,hardware_SHA256=loop.HARDWARE,allowed_keys=sorted(allowed),all_measured_masks_and_upstream_frozen=True,original_structure=True,new_inference_layers=0,teacher_SHA256=m.sha(ref),teacher_script_parity_maxabs=parity,teacher_TRAIN_correct=teacher_correct,teacher_cache_SHA256=m.sha(out/'fixed_reference_TRAIN_teacher.npz'),synthetic_CCD_noise_not_calibrated=True,SDK=False,GPU_UUID=cfg['GPU_UUID'],new_CCD=0)
    m.save(out/'protocol.json',protocol);best,nll,selected=c['correct'],baseline['nll'],0;loop.saved_model(m,model,out/'best.pt',protocol);np.savez_compressed(out/'baseline_test_predictions.npz',ids=data['test'][2],labels=data['test'][1],probabilities=bp)
    model.requires_grad_(False)
    for name,p in model.named_parameters():p.requires_grad_(name in allowed)
    model.cuda();groups=[]
    for prefix,rate in (('phase_sets.',cfg['lr_phase']),('electronic.',cfg['lr_electronic']),('fusion_logits.',cfg['lr_alpha']),('linear.',cfg['lr_head'])):
        params=[p for n,p in model.named_parameters() if n in allowed and n.startswith(prefix)]
        if params:groups.append(dict(params=params,lr=rate,weight_decay=0.))
    optimizer=torch.optim.AdamW(groups);transforms=TRAINTransforms(model,spec['stage'],cfg);box={}
    handle=model.linear.register_forward_pre_hook(lambda module,inputs:box.update(features=inputs[0]));rng=np.random.default_rng(73+depth);history=[];stale=0;start=time.monotonic()
    try:
        for epoch in range(1,cfg['maximum_epochs']+1):
            model.cuda().train();order=rng.permutation(1600);total=0.;used=0
            for begin in range(0,1600,cfg['batch']):
                ix=order[begin:begin+cfg['batch']];labels=torch.from_numpy(data['train'][1][ix]).cuda();optimizer.zero_grad(set_to_none=True);p=model(loop.put_fields(model,data,fields,'train',ix,'cuda'))[0]
                loss,parts=teacher_loss(F,p,box['features'],labels,torch.from_numpy(tp[ix]).cuda(),torch.from_numpy(tf[ix]).cuda(),cfg)
                assert torch.isfinite(loss);loss.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for n,p in model.named_parameters() if n in allowed)
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.);optimizer.step()
                with torch.no_grad():
                    for p in model.fusion_logits:
                        if p.requires_grad:p.clamp_(.4001001,.95)
                used+=len(ix);total+=float(loss.detach())*len(ix)
                if used%80==0:m.save(out/'progress.json',dict(state='reference_TRAIN_suffix_fit',epoch=epoch,TRAIN_consumed=used,loss=total/used,parts=parts,elapsed_s=time.monotonic()-start))
            assert used==1600;loop.validate_state(m,model,original,allowed);metric,_=loop.original_replay(model,data,fields,'test');improved=metric['correct']>best or (metric['correct']==best and metric['nll']<nll-1e-8)
            if improved:best,nll,selected=metric['correct'],metric['nll'],epoch;loop.saved_model(m,model,out/'best.pt',protocol)
            stale=0 if improved else stale+1;row=dict(epoch=epoch,TRAIN_consumed=used,TRAIN_loss=total/used,TEST_correct=metric['correct'],TEST_NLL=metric['nll'],selected_correct=best,elapsed_s=time.monotonic()-start);history.append(row);m.save(out/'history.json',history);m.save(out/'progress.json',dict(state='suffix_epoch_complete',latest=row));print(json.dumps(dict(depth=depth,**row)),flush=True)
            if best>=cfg['targets'][str(depth)] or stale>=cfg['no_improvement_epochs'] or time.monotonic()-start>=cfg['maximum_fit_seconds']:break
    finally:handle.remove();transforms.remove()
    model.cpu();loop.validate_state(m,model,original,allowed);loop.saved_model(m,model,out/'last.pt',protocol);del optimizer;torch.cuda.empty_cache();results={}
    for name in ('best','last'):
        checked,_=m.model_for(out/(name+'.pt'),m.sha(out/(name+'.pt')),depth,ref);loop.validate_state(m,checked,original,allowed);metrics={}
        for role in ('train','test'):
            m.save(out/'replay_progress.json',dict(state='original_CPU_replay',checkpoint=name,role=role,started_ns=time.time_ns()));metric,p=loop.original_replay(checked,data,fields,role);metrics[role]=metric;np.savez_compressed(out/(name+'_'+role+'_predictions.npz'),ids=data[role][2],labels=data[role][1],probabilities=p)
        results[name]=dict(**metrics,PT_SHA256=m.sha(out/(name+'.pt')))
    assert results['best']['test']['correct']==best
    m.save(out/'TRAIN_transform_audit.json',dict(counts=transforms.counts,all_CCD_arrays_immutable=True,eval_transforms_disabled=True,hooks_removed=True,new_inference_modules=0))
    m.save(out/'report.json',dict(state='complete',baseline_correct=c['correct'],selected_correct=best,selected_epoch=selected,selected_E0=selected==0,actual_epochs=epoch,results=results,strict_original_CPU=True,full_physical=False,target=cfg['targets'][str(depth)],passed=best>=cfg['targets'][str(depth)],next_layer_started=False,protocol_SHA256=m.sha(out/'protocol.json'),freeze_verified=True))

def selftest():
    import torch
    from torch.nn import functional as F
    import lab_caltech20_dvp_layerwise_20261019 as m
    import lab_caltech20_dvp_closedloop_20261021 as loop
    torch.set_num_threads(2);torch.manual_seed(73);cfg=dict(label_smoothing=.03,temperature=2.,teacher_KL=.5,feature_alignment=.1);checks=[]
    for depth,prefix,stage in ((4,4,'2'),(6,2,'0')):
        model=m.MeasuredPrefix(depth).eval()
        with torch.no_grad():
            for p in model.fusion_logits:p.fill_(.8)
            x=torch.rand(2,224,224)*.1;teacher=model(x)[0];tf=model.readout['features'].clone();fields={k:v['intensity'].clone() for k,v in model.last.items() if k in (['router']+[str(i) for i in range(depth)])[:prefix]}
        model.measured=fields;before=m.state_cpu(model);allowed=m.allowed_keys(model,prefix);assert not allowed&loop.mask_keys(model,stage);model.requires_grad_(False)
        for n,p in model.named_parameters():p.requires_grad_(n in allowed)
        box={};handle=model.linear.register_forward_pre_hook(lambda module,inputs:box.update(f=inputs[0]));p=model(x)[0];loss,_=teacher_loss(F,p,box['f'],torch.tensor([0,1]),teacher,tf,cfg);loss.backward();handle.remove()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for n,p in model.named_parameters() if n in allowed);assert all(p.grad is None for n,p in model.named_parameters() if n not in allowed);loop.validate_state(m,model,before,allowed);checks.append(dict(depth=depth,only_original_unmeasured_suffix_gradients=True,original_state_keys=True))
    print(json.dumps(dict(passed=True,checks=checks,TEST_used=False,SDK=False)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--selftest',action='store_true');p.add_argument('--sequence',type=Path);p.add_argument('--profile',type=Path);p.add_argument('--depth',choices=('4','6','both'));a=p.parse_args()
    if a.selftest:selftest()
    else:
        for depth in ((4,6) if a.depth=='both' else (int(a.depth),)):
            try:run(a,depth)
            except BaseException as e:
                out=a.sequence/f'TRAIN_restored_same_mask_pair_20261021/L{depth}/TRAIN_reference_teacher_suffix_20261021'
                if out.exists():(out/'failure.json').write_text(json.dumps(dict(error=repr(e),traceback=traceback.format_exc(),automatic_retry=False),indent=2))
                raise
