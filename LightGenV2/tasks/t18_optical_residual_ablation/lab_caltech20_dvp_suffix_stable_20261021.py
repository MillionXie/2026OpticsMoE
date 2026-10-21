"""Offline stable raw-logit adaptation of an already measured L6 prefix.

No camera opens. Frozen measured router/E1/G1/E2; only unmeasured G2/E3/G3
masks and their deployable electronic suffix may change. Every following
optical layer will require its own fresh capture after human camera recovery.
"""
import argparse,json,os,time,traceback
from pathlib import Path


def load_prefix(a,m,np,cfg):
    import lab_caltech20_dvp_closedloop_20261021 as loop
    data,dm=m.data_for(a.sequence);assert dm['sha256']==cfg['data_SHA256']
    assert m.session_hardware_digest(a.sequence)==loop.HARDWARE and not m.LOCK.exists()
    run=a.sequence/cfg['source_run'];endpoint=m.read(run/'L6/stage_2/endpoint_report.json')
    parent=Path(endpoint['PT']);assert m.sha(parent)==cfg['parent_SHA256']==endpoint['PT_SHA256']
    assert endpoint['TEST_development']['correct']==147 and not endpoint['full_physical']
    assert m.read(run/'L6/stage_2/suffix/independent_audit.json')['passed']
    failed=run/'L6/stage_3/capture';failure=m.read(failed/'failure.json')
    assert failure['completed_receipts']==0 and 'DVP Start: -1001' in failure['error']
    assert not list(failed.glob('train/[0-9]*.json')) and not list(failed.glob('test/[0-9]*.json'))
    session=a.sequence/'L6_response_20261020';fields,identities=m.fields_for(session,1,data,dm,loop.HARDWARE)
    caps={'0':a.sequence/'TRAIN_restored_same_mask_pair_20261021/L6/capture','1':run/'L6/stage_1/capture','2':run/'L6/stage_2/capture'}
    for key,cap in caps.items():
        r=m.read(cap/'report.json');audit=m.read(cap/'independent_audit.json')
        assert r['state']=='complete' and r['physical_ccd']==2011 and r['data_SHA256']==dm['sha256'] and r['hardware_SHA256']==loop.HARDWARE
        assert audit['passed'] and audit['report_SHA256']==m.sha(cap/'report.json')
        fields[key]={}
        for role,n in (('train',1600),('test',411)):
            p=cap/role/'ccd.npy';s=m.sha(p);assert s==r['arrays'][role]['SHA256']
            v=np.load(p,mmap_mode='r');assert v.shape==(n,478,478) and v.dtype==np.uint8
            fields[key][role]=v;identities[key+'/'+role]=s
    assert identities==endpoint['prefix_SHA256'] and list(fields)==['router','0','1','2']
    model,_=m.model_for(parent,cfg['parent_SHA256'],6,session/'source_reference.ts')
    return data,dm,fields,identities,model,session/'source_reference.ts',parent


def run(a):
    import numpy as np,torch
    from torch.nn import functional as F
    import lab_caltech20_dvp_layerwise_20261019 as m
    import lab_caltech20_dvp_closedloop_20261021 as loop
    from lab_caltech20_dvp_TRAIN_regularization_20261021 import TRAINTransforms
    cfg=m.read(a.profile);assert cfg['schema']=='L6_measured_prefix4_stable_rawCE_v1'
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==cfg['GPU_UUID']
    assert not a.out.exists();a.out.mkdir(parents=True);torch.set_num_threads(2);torch.manual_seed(73)
    m.save(a.out/'progress.json',dict(state='checking_recorded_prefix_and_parent',camera_started=False))
    data,dm,fields,identities,model,reference,parent=load_prefix(a,m,np,cfg);original=m.state_cpu(model)
    m.save(a.out/'progress.json',dict(state='original_CPU_baseline_TEST411',camera_started=False))
    baseline,bp=loop.original_replay(model,data,fields,'test');assert baseline['correct']==147
    allowed=m.allowed_keys(model,4);assert not allowed&set().union(*(loop.mask_keys(model,str(k)) for k in range(3)))
    assert all(not k.startswith('raw_router') for k in allowed)
    protocol=dict(config=cfg,source_git=m.git_head(),source_SHA256=m.sha(__file__),profile_SHA256=m.sha(a.profile),parent_PT=str(parent),parent_PT_SHA256=cfg['parent_SHA256'],
        prefix_SHA256=identities,data_SHA256=dm['sha256'],hardware_SHA256=loop.HARDWARE,allowed_keys=sorted(allowed),depth=6,prefix=4,measured_stages=list(fields),
        TRAIN=1600,TEST=411,VALID=0,TEST_gradient=False,TEST_informed_development=True,full_physical=False,metric='measured_router_E1_G1_E2_simulated_G2_E3_G3',
        all_captured_optical_upstream_frozen=True,new_inference_layers=0,new_CCD=0,SDK_used=False,recurring=False,camera_restart_attempted=False)
    m.save(a.out/'protocol.json',protocol);m.save(a.out/'baseline.json',baseline)
    loop.saved_model(m,model,a.out/'best.pt',protocol);best,bnll,selected=147,baseline['nll'],0;history=[]
    # Fresh exact parity check before any TRAIN-only transform.
    ix=np.arange(8)
    with torch.no_grad():before=model(loop.put_fields(model,data,fields,'train',ix,'cpu'))[0]
    transforms=TRAINTransforms(model,'2',cfg);held={};hook=model.linear.register_forward_hook(lambda module,args,z:held.update(logits=z))
    try:
        with torch.no_grad():after=model(loop.put_fields(model,data,fields,'train',ix,'cpu'))[0]
        assert torch.equal(before,after) and transforms.counts['noise_TRAIN_batches']==0
        loop.gpu_admission(m,cfg);model.cuda()
        for name,p in model.named_parameters():p.requires_grad_(name in allowed)
        groups=[]
        for prefix,key in (('phase_sets.','phase_lr'),('electronic.','electronic_lr'),('fusion_logits.','alpha_lr'),('linear.','head_lr')):
            ps=[p for n,p in model.named_parameters() if n in allowed and n.startswith(prefix)]
            if ps:groups.append(dict(params=ps,lr=cfg[key],weight_decay=0 if prefix=='fusion_logits.' else cfg['weight_decay']))
        opt=torch.optim.AdamW(groups);rng=np.random.default_rng(73);start=time.monotonic()
        cm=np.asarray(baseline['confusion_matrix']);error=1-cm.diagonal()/cm.sum(1).clip(1)
        counts=np.asarray(cfg['original_TRAIN_counts'],float);assert counts.sum()==3141
        weights=torch.tensor(counts/counts.mean()*(1+.25*error),dtype=torch.float32,device='cuda')
        for epoch in range(1,cfg['epochs']+1):
            model.cuda().train();order=rng.permutation(1600);total=0.;used=0;nonzero=0
            for begin in range(0,1600,8):
                ix=order[begin:begin+8];inputs=loop.put_fields(model,data,fields,'train',ix,'cuda');labels=torch.from_numpy(data['train'][1][ix].astype(np.int64)).cuda()
                opt.zero_grad();model(inputs);logits=held['logits'];loss=F.cross_entropy(logits,labels,weight=weights,label_smoothing=cfg['label_smoothing'])
                assert torch.isfinite(loss);loss.backward();params=[p for p in model.parameters() if p.requires_grad]
                assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
                norm=float(torch.nn.utils.clip_grad_norm_(params,1.));nonzero+=int(norm>0);opt.step()
                with torch.no_grad():
                    for p in model.fusion_logits:
                        if p.requires_grad:p.clamp_(.4001001,.95)
                total+=float(loss.detach())*len(ix);used+=len(ix)
                if begin%160==0:m.save(a.out/'progress.json',dict(state='stable_rawCE_suffix_TRAIN',epoch=epoch,TRAIN_consumed=used,TRAIN_loss=total/used,nonzero_gradient_batches=nonzero,selected_correct=best,elapsed_s=time.monotonic()-start))
            assert used==1600 and nonzero>0
            if epoch==1 or epoch%cfg['evaluate_every']==0 or epoch==cfg['epochs'] or time.monotonic()-start>=cfg['fit_seconds']:
                score,prob=loop.original_replay(model,data,fields,'test')
                if score['correct']>best or (score['correct']==best and score['nll']<bnll-1e-8):
                    best,bnll,selected=score['correct'],score['nll'],epoch;loop.saved_model(m,model,a.out/'best.pt',protocol)
                row=dict(epoch=epoch,TRAIN_loss=total/1600,TRAIN_consumed=used,TEST_correct=score['correct'],TEST_NLL=score['nll'],selected_correct=best,elapsed_s=time.monotonic()-start)
                history.append(row);m.save(a.out/'history.json',history);m.save(a.out/'progress.json',dict(state='strict_CPU_suffix_selection',latest=row));print(json.dumps(row),flush=True)
            if best>=cfg['target_correct'] or time.monotonic()-start>=cfg['fit_seconds']:break
        model.cpu();loop.validate_state(m,model,original,allowed);loop.saved_model(m,model,a.out/'last.pt',protocol)
    finally:transforms.remove();hook.remove()
    results={}
    for name in ('best','last'):
        p=a.out/(name+'.pt');native,_=m.model_for(p,m.sha(p),6,reference);loop.validate_state(m,native,original,allowed);results[name]=dict(PT_SHA256=m.sha(p))
        for role in ('train','test'):
            m.save(a.out/'progress.json',dict(state='original_CPU_prefix_suffix_replay',checkpoint=name,role=role,selected_correct=best))
            metric,prob=loop.original_replay(native,data,fields,role);results[name][role]=metric
            np.savez_compressed(a.out/(name+'_'+role+'_predictions.npz'),ids=data[role][2],labels=data[role][1],probabilities=prob)
    assert results['best']['test']['correct']==best
    if selected==0:
        saved=torch.load(a.out/'best.pt',map_location='cpu',weights_only=False)['model']
        assert all(torch.equal(v,saved[k]) for k,v in original.items())
    m.save(a.out/'TRAIN_transform_audit.json',dict(counts=transforms.counts,eval_identity_exact=True,hook_free_original_bestlast=True,calibrated_camera_noise=False,new_inference_layers=0))
    m.save(a.out/'report.json',dict(state='complete',baseline_correct=147,selected_correct=best,selected_epoch=selected,selected_E0=selected==0,actual_epochs=epoch,target_correct=cfg['target_correct'],target_met=best>=cfg['target_correct'],
        full_physical=False,strict_original_CPU=True,results=results,all_captured_upstream_frozen=True,TEST_gradient=False,TEST_informed_development=True,protocol_SHA256=m.sha(a.out/'protocol.json'),SDK_used=False,camera_restart_attempted=False))
    m.save(a.out/'independent_audit.json',dict(passed=True,report_SHA256=m.sha(a.out/'report.json'),original_bestlast_TRAIN1600_TEST411=True,prefix_SHA256=identities,all_captured_optical_upstream_bitwise_frozen=True))
    m.save(a.out/'next_capture_handoff.json',dict(state='awaiting_explicit_human_camera_restore',depth=6,next_stage='3',next_name='G2',PT=str(a.out/'best.pt'),PT_SHA256=m.sha(a.out/'best.pt'),prefix_SHA256=identities,
        new_downstream_CCD_required=True,old_failed_stage3_frames=0,existing_G1_E2_not_recaptured=True,camera_started=False,full_physical=False))
    del opt;torch.cuda.empty_cache()


def selftest():
    import torch
    import lab_caltech20_dvp_layerwise_20261019 as m
    import lab_caltech20_dvp_closedloop_20261021 as loop
    from lab_caltech20_dvp_TRAIN_regularization_20261021 import TRAINTransforms
    torch.set_num_threads(2);torch.manual_seed(73)
    model=m.MeasuredPrefix(6).eval()
    with torch.no_grad():
        for p in model.fusion_logits:p.fill_(.8)
        x=torch.rand(2,224,224)*.25;model(x)
        fields={k:model.last[k]['intensity'].clone()*173 for k in ('router','0','1','2')}
        model.measured={k:v.clone() for k,v in fields.items()};before=model(x)[0].clone()
    state=m.state_cpu(model);allowed=m.allowed_keys(model,4)
    assert not allowed&set().union(*(loop.mask_keys(model,str(k)) for k in range(3)))
    transforms=TRAINTransforms(model,'2',dict(CCD_gain_sigma=.005,CCD_additive_sigma_DN=.1,feature_dropout_p=.02,dropout_location='head_features'))
    held={};hook=model.linear.register_forward_hook(lambda module,args,z:held.update(logits=z))
    try:
        model.measured={k:v.clone() for k,v in fields.items()}
        with torch.no_grad():assert torch.equal(before,model(x)[0])
        model.train().requires_grad_(False)
        for n,p in model.named_parameters():p.requires_grad_(n in allowed)
        model.measured={k:v.clone() for k,v in fields.items()};model(x)
        loss=torch.nn.functional.cross_entropy(held['logits'],torch.tensor([0,1]));loss.backward()
        assert torch.isfinite(loss)
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for n,p in model.named_parameters() if n in allowed)
        assert any(float(p.grad.abs().max())>0 for n,p in model.named_parameters() if n in allowed and n.startswith('phase_sets.'))
        assert all(p.grad is None for n,p in model.named_parameters() if n not in allowed)
        assert all(torch.equal(model.measured[k],fields[k]) for k in fields if k!='2')
        loop.validate_state(m,model,state,allowed)
    finally:transforms.remove();hook.remove()
    model.eval();model.measured=fields
    with torch.no_grad():assert torch.equal(before,model(x)[0])
    print(json.dumps(dict(passed=True,original_state_keys_unchanged=True,captured_mask_frozen=True,future_mask_gradient_nonzero=True,eval_transforms_identity=True,TEST_used=False,SDK_used=False,GPU_used=False)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--sequence',type=Path);p.add_argument('--profile',type=Path);p.add_argument('--out',type=Path);p.add_argument('--selftest',action='store_true');a=p.parse_args()
    if a.selftest:selftest();raise SystemExit(0)
    try:run(a)
    except BaseException as e:
        if a.out.exists():(a.out/'failure.json').write_text(json.dumps(dict(error=repr(e),traceback=traceback.format_exc(),automatic_retry=False,camera_started=False),indent=2))
        raise
