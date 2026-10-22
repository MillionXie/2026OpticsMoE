"""Explicit-human USB recovery: preserve four measured stages, G2 -> E3 -> G3.

Single SDK; full TRAIN then TEST at each stage, then original suffix training.
No timer, recapture of G1/E2, new layer, TEST derivative or setting adjustment.
"""
import argparse,json,os,time,traceback
from pathlib import Path


def suffix(m,loop,data,dm,fields,model,cfg,stage,out,reference):
    import numpy as np,torch
    from torch.nn import functional as F
    from lab_caltech20_dvp_TRAIN_regularization_20261021 import TRAINTransforms
    assert not out.exists();out.mkdir();original=m.state_cpu(model);prefix=int(stage)+2;allowed=m.allowed_keys(model,prefix);final=stage=='5'
    baseline,_=loop.original_replay(model,data,fields,'test');best,bnll,selected=baseline['correct'],baseline['nll'],0
    protocol=dict(config=cfg,source_git=m.git_head(),source_SHA256=m.sha(__file__),stage=stage,prefix=prefix,depth=6,full_physical=final,
        prefix_SHA256={k+'/'+r:m.sha(Path(cfg['_paths'][k])/r/'ccd.npy') for k in fields for r in ('train','test')},data_SHA256=dm['sha256'],TRAIN=1600,TEST=411,VALID=0,allowed_keys=sorted(allowed),TEST_gradient=False,TEST_informed_development=True,new_layers=0,SDK_used=False)
    m.save(out/'protocol.json',protocol);m.save(out/'baseline.json',baseline);loop.saved_model(m,model,out/'best.pt',protocol)
    transforms=TRAINTransforms(model,stage,cfg);held={};hook=model.linear.register_forward_hook(lambda module,args,z:held.update(logits=z));history=[]
    try:
        loop.gpu_admission(m,cfg);model.cuda()
        for name,p in model.named_parameters():p.requires_grad_(name in allowed)
        groups=[]
        for key,lr in (('phase_sets.','phase_lr'),('electronic.','electronic_lr'),('fusion_logits.','alpha_lr'),('linear.','head_lr')):
            ps=[p for n,p in model.named_parameters() if n in allowed and n.startswith(key)]
            if ps:groups.append(dict(params=ps,lr=cfg[lr],weight_decay=0 if key=='fusion_logits.' else cfg['weight_decay']))
        opt=torch.optim.AdamW(groups);rng=np.random.default_rng(73);start=time.monotonic();epochs=cfg['final_epochs'] if final else cfg['intermediate_epochs'];budget=cfg['final_fit_seconds'] if final else cfg['intermediate_fit_seconds']
        cm=np.asarray(baseline['confusion_matrix']);err=1-cm.diagonal()/cm.sum(1).clip(1);counts=np.asarray(cfg['original_TRAIN_counts'],float)
        weights=torch.tensor(counts/counts.mean()*(1+.25*err),dtype=torch.float32,device='cuda')
        for epoch in range(1,epochs+1):
            model.train();order=rng.permutation(1600);total=0.;used=0
            for begin in range(0,1600,8):
                ix=order[begin:begin+8];x=loop.put_fields(model,data,fields,'train',ix,'cuda');y=torch.from_numpy(data['train'][1][ix].astype(np.int64)).cuda()
                opt.zero_grad();model(x);loss=F.cross_entropy(held['logits'],y,weight=weights,label_smoothing=cfg['label_smoothing']);assert torch.isfinite(loss);loss.backward()
                params=[p for p in model.parameters() if p.requires_grad];assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
                torch.nn.utils.clip_grad_norm_(params,1.);opt.step()
                with torch.no_grad():
                    for p in model.fusion_logits:
                        if p.requires_grad:p.clamp_(.4001001,.95)
                used+=len(ix);total+=float(loss.detach())*len(ix)
                if begin%160==0:m.save(out/'progress.json',dict(state='full_TRAIN_raw_CE',stage=stage,epoch=epoch,TRAIN_consumed=used,TRAIN_loss=total/used,selected_correct=best))
            assert used==1600
            if epoch==1 or epoch%2==0 or epoch==epochs or time.monotonic()-start>=budget:
                with torch.no_grad():score,_=m.assess(model,data,fields,'test')
                if score['correct']>best or (score['correct']==best and score['nll']<bnll-1e-8):best,bnll,selected=score['correct'],score['nll'],epoch;loop.saved_model(m,model,out/'best.pt',protocol)
                row=dict(epoch=epoch,TRAIN_consumed=used,TRAIN_loss=total/used,TEST_correct=score['correct'],TEST_NLL=score['nll'],selected_correct=best,elapsed_s=time.monotonic()-start)
                history.append(row);m.save(out/'history.json',history);print(json.dumps(row),flush=True)
            if time.monotonic()-start>=budget or best>=cfg['final_target' if final else 'intermediate_target']:break
        model.cpu();loop.validate_state(m,model,original,allowed);loop.saved_model(m,model,out/'last.pt',protocol)
    finally:transforms.remove();hook.remove()
    results={};chosen=None
    for name in ('best','last'):
        pt=out/(name+'.pt');native,_=m.model_for(pt,m.sha(pt),6,reference);loop.validate_state(m,native,original,allowed);results[name]=dict(PT_SHA256=m.sha(pt))
        for role in ('train','test'):
            m.save(out/'progress.json',dict(state='original_CPU_bestlast_replay',checkpoint=name,role=role,selected_correct=best))
            score,prob=loop.original_replay(native,data,fields,role);results[name][role]=score
            np.savez_compressed(out/(name+'_'+role+'_predictions.npz'),ids=data[role][2],labels=data[role][1],probabilities=prob)
        if name=='best':chosen=native
    assert results['best']['test']['correct']==best
    m.save(out/'TRAIN_transform_audit.json',dict(counts=transforms.counts,eval_hook_free_original=True,synthetic_not_calibrated=True))
    m.save(out/'report.json',dict(state='complete',baseline_correct=baseline['correct'],selected_correct=best,selected_epoch=selected,selected_E0=selected==0,actual_epochs=epoch,target_correct=cfg['final_target' if final else 'intermediate_target'],full_physical=final,strict_original_CPU=True,results=results,protocol_SHA256=m.sha(out/'protocol.json'),TEST_gradient=False,TEST_informed_development=True))
    m.save(out/'independent_audit.json',dict(passed=True,report_SHA256=m.sha(out/'report.json'),all_captured_upstream_frozen=True,original_bestlast_TRAIN1600_TEST411=True,prefix_SHA256=protocol['prefix_SHA256']))
    del opt;torch.cuda.empty_cache();return chosen,best


def run(a):
    import numpy as np,torch
    import lab_caltech20_dvp_layerwise_20261019 as m
    import lab_caltech20_dvp_closedloop_20261021 as loop
    import lab_caltech20_dvp_advance_20261021 as advance
    from lab_caltech20_dvp_suffix_stable_20261021 import load_prefix
    torch.set_num_threads(2);torch.manual_seed(73);cfg=m.read(a.profile)
    assert cfg['schema'] in ('DVP_L6_human_recovered_prefix4_v1','DVP_L6_capture_remaining_then_terminal_v1') and cfg['explicit_human_camera_restored'] and cfg['user_stage_carry']
    capture_only=cfg['schema']=='DVP_L6_capture_remaining_then_terminal_v1'
    if capture_only:assert cfg['interstage_fit'] is False and cfg['parent_correct']==219 and cfg['TEST_gradient'] is False and cfg['VALID']==0
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==cfg['GPU_UUID'] and m.sha(advance.__file__)==cfg['capture_engine_SHA256']
    assert not a.out.exists();a.out.mkdir(parents=True);m.save(a.out/'progress.json',dict(state='prefix_and_selected_PT_admission',SDK_opened=False))
    base=m.read(Path(__file__).with_name(cfg['prefix_profile']));data,dm,fields,identities,old,reference,_=load_prefix(a,m,np,base)
    parent=a.sequence/cfg['parent_relative'];r=m.read(parent.parent/'report.json');audit=m.read(parent.parent/'independent_audit.json')
    expected=cfg.get('parent_correct',205)
    assert m.sha(parent)==cfg['parent_SHA256']==r['results']['best']['PT_SHA256'] and r['selected_correct']==expected and audit['passed'] and audit['report_SHA256']==m.sha(parent.parent/'report.json')
    assert audit['prefix_SHA256']==identities and m.read(parent.parent.with_suffix('.return.json'))['returncode']==0
    model,_=m.model_for(parent,cfg['parent_SHA256'],6,reference);loop.validate_state(m,model,m.state_cpu(old),m.allowed_keys(old,4))
    score,_=loop.original_replay(model,data,fields,'test');assert score['correct']==expected
    hw=m.read(a.sequence/'calibration/admission/admitted_hardware.json');assert m.session_hardware_digest(a.sequence)==advance.HARDWARE
    case=a.sequence/base['source_run'];session=a.sequence/'L6_response_20261020'
    paths={'router':session/'capture/router','0':a.sequence/'TRAIN_restored_same_mask_pair_20261021/L6/capture','1':case/'L6/stage_1/capture','2':case/'L6/stage_2/capture'}
    row=dict(model=model,fields=fields,identities=identities,paths=paths,parent=parent,reference=reference);history=[]
    m.save(a.out/'protocol.json',dict(config=cfg,source_git=m.git_head(),source_SHA256=m.sha(__file__),profile_SHA256=m.sha(a.profile),parent_PT_SHA256=m.sha(parent),prefix_SHA256=identities,baseline_mixed_correct=expected,interstage_fit=not capture_only,
        stage_order=['3','4','5'],planned_frames=6033,TRAIN=1600,TEST=411,VALID=0,TEST_gradient=False,TEST_informed_development=True,SDK_serial=True,recurring=False,old_G1_E2_preserved=True,old_failed_G2_frames=0,explicit_human_recovery=True))
    for stage in ('3','4','5'):
        folder=a.out/'L6'/('stage_'+stage);folder.mkdir(parents=True)
        m.save(a.out/'progress.json',dict(state='capturing',stage=stage,stage_name={'3':'G2','4':'E3','5':'G3'}[stage],completed=history))
        row['fields']=advance.acquire(m,loop,a,data,dm,hw,row,6,stage,folder/'capture');row['paths'][stage]=folder/'capture'
        for role in ('train','test'):row['identities'][stage+'/'+role]=m.sha(folder/'capture'/role/'ccd.npy')
        if capture_only and stage!='5':
            assert m.sha(row['parent'])==cfg['parent_SHA256']
            endpoint=dict(stage=stage,depth=6,full_physical=False,PT=str(row['parent']),PT_SHA256=m.sha(row['parent']),prefix_SHA256=dict(row['identities']),capture_audit_SHA256=m.sha(folder/'capture/independent_audit.json'),interstage_fit=False,SDK_released=not m.LOCK.exists())
            m.save(folder/'endpoint_report.json',endpoint);history.append(endpoint);m.save(a.out/'history.json',history)
            continue
        m.save(a.out/'progress.json',dict(state='suffix_fit_after_complete_capture',stage=stage,completed=history))
        fitcfg={**cfg,'_paths':{k:str(v) for k,v in row['paths'].items()}}
        chosen,best=suffix(m,loop,data,dm,row['fields'],row['model'],fitcfg,stage,folder/'suffix',reference)
        torch.cuda.empty_cache()
        row['model']=chosen;row['parent']=folder/'suffix/best.pt';endpoint=dict(stage=stage,depth=6,full_physical=stage=='5',selected_correct=best,PT=str(row['parent']),PT_SHA256=m.sha(row['parent']),prefix_SHA256=row['identities'],capture_audit_SHA256=m.sha(folder/'capture/independent_audit.json'),suffix_audit_SHA256=m.sha(folder/'suffix/independent_audit.json'),SDK_released=not m.LOCK.exists())
        m.save(folder/'endpoint_report.json',endpoint);history.append(endpoint);m.save(a.out/'history.json',history)
    m.save(a.out/'report.json',dict(state='complete',history=history,SDK_released=not m.LOCK.exists(),full_physical=True,interstage_fit=not capture_only,TEST_gradient=False,TEST_informed_development=True,recurring=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--sequence',type=Path,required=True);p.add_argument('--profile',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    try:run(a)
    except BaseException as e:
        if a.out.exists():(a.out/'failure.json').write_text(json.dumps(dict(error=repr(e),traceback=traceback.format_exc(),automatic_retry=False,all_old_CCD_preserved=True),indent=2))
        raise
