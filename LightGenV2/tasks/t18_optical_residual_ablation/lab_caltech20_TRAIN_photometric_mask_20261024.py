"""Admitted TRAIN response -> original E2/G2 mask learning -> fresh E2 only.

Calibrated hybrid TEST predictions select a TRAIN optimization checkpoint, not
a physical score. Only native replay after fresh CCD may pass the physical gate.
All PTs contain the unchanged native state schema; response maps never deploy.
"""
import argparse,json,os,shutil,time,traceback
from pathlib import Path
import lab_caltech20_tucam_layerwise_20261023 as base


def validate_policy(p):
    assert p['schema']=='TUCam_L4_admitted_photometric_E2_mask_v1'
    assert p['source_fit'].endswith('/e2z/s_2/fit') and p['source_correct']==273
    assert p['source_PT_SHA256']=='c511fc50b2c6a8763620c47cdc6e20556f8cc15a07e3fdcb549468943f3f2efd'
    assert p['response_fit'].endswith('/e2_response_20261024/flat_photometric')
    assert p['response_report_SHA256']=='6cf09552fc620262fa2eecaab695a64a1c36aeb859c03e327cccbd8634fcbdbf'
    assert p['response_map_SHA256']=='6735e4f409548677d38d080918bb24a03a0c31cdc4c9ce3cc8e6eef84425eaab'
    assert list(p['captures'])==['router','0','1','2']
    assert p['captures']['2'].endswith('/e2h/s_2/capture')
    assert p['mask_recipe']==dict(epochs=16,seconds=1200,phase_lr=.0002,electronic_lr=.0002,alpha_lr=.0002,head_lr=.0005,weight_decay=.01,label_smoothing=.03)
    assert p['observed_recipe']==dict(name='new_response_E2_actual_suffix_CE',epochs=36,seconds=1200,lr_scale=1,TRAIN_class_emphasis=.75)
    assert p['final_recipe']==dict(name='new_response_full_physical_original5150',epochs=120,seconds=1200,lr_scale=1,TRAIN_class_emphasis=.75)
    assert p['intermediate_target']==305 and p['final_target']==340 and p['TEST_gradient'] is False
    assert p['new_inference_layers']==0 and p['scheduler'] is False and p['new_G2_if_gate_unmet'] is False


def make_predictor(model,gain,bg):
    np,torch,m,_=base.numerical_modules()
    from lab_caltech20_TRAIN_response_operator_20261024 import TRAINPhotometricPrefix
    rng=torch.get_rng_state();work=TRAINPhotometricPrefix(m.state_cpu(model),gain,bg).cpu();torch.set_rng_state(rng)
    work.propagator.transfer.copy_(model.propagator.transfer.detach().cpu());return work


def predictive_score(model,gain,bg,data,fields,role):
    """Explicit CPU subclass avoids assess(GPU)'s native-class fallback."""
    _,torch,_,_=base.numerical_modules();cpu=make_predictor(model,gain,bg).eval()
    with torch.inference_mode():return base.evaluate(cpu,data,fields,role)


def admit(p,c,out):
    np,torch,m,loop=base.numerical_modules()
    response=base.ROOT/p['response_fit'];r=m.read(response/'report.json')
    assert m.sha(response/'report.json')==p['response_report_SHA256'] and r['response_admitted'] and r['state']=='complete' and not r['TEST_used']
    assert r['protocol_SHA256']==m.sha(response/'protocol.json') and r['map_SHA256']==m.sha(response/'TRAIN_flat_gain_background.npz')==p['response_map_SHA256']
    assert m.read(response.with_suffix('.return.json'))['returncode']==0 and not m.LOCK.exists()
    with np.load(response/'TRAIN_flat_gain_background.npz',allow_pickle=False) as z:
        gain=torch.from_numpy(z['gain'].copy());bg=torch.from_numpy(z['background'].copy());assert len(z['fit_TRAIN_ids'])==20
    source=base.ROOT/p['source_fit'];sr=m.read(source/'report.json');a=m.read(source/'independent_audit.json')
    parent=source/'best.pt';assert m.sha(parent)==p['source_PT_SHA256']==sr['results']['best']['PT_SHA256']
    assert sr['strict_original_CPU'] and sr['selected_correct']==273 and a['passed'] and a['report_SHA256']==m.sha(source/'report.json')
    data,dm=m.data_for(base.ROOT/c['sequence']);assert dm['sha256']==c['data_SHA256'];fields={};paths={}
    for stage,name in p['captures'].items():
        cap=base.ROOT/name;q=m.read(cap/'report.json');audit=m.read(cap/'independent_audit.json');proto=m.read(cap/'protocol.json')
        assert q['state']=='complete' and q['physical_ccd']==2011 and q['SDK_released'] and q['data_SHA256']==dm['sha256']
        assert audit['passed'] and audit['matched_samples']==2011 and audit['report_SHA256']==m.sha(cap/'report.json')
        assert audit['protocol_SHA256']==q['protocol_SHA256']==m.sha(cap/'protocol.json')
        assert q['hardware_SHA256']==c['hardware_by_stage'][stage]['SHA256'] and q['phase_SHA256']==m.sha(cap/'phase.bmp')
        if 'prefix_SHA256' in proto:assert proto['prefix_SHA256']==base.identities(m,paths)
        fields[stage]={}
        for role,n in (('train',1600),('test',411)):
            file=cap/role/'ccd.npy';assert m.sha(file)==q['arrays'][role]['SHA256'];value=np.load(file,mmap_mode='r')
            assert value.shape==(n,478,478) and value.dtype==np.uint8;fields[stage][role]=value
        paths[stage]=cap
    assert m.read(source/'protocol.json')['prefix_SHA256']==base.identities(m,paths)
    reference=base.ROOT/c['reference'];assert m.sha(reference)==c['reference_SHA256']
    model,_=m.model_for(parent,p['source_PT_SHA256'],4,reference);score,_=base.evaluate(model,data,fields,'test');assert score['correct']==273
    cap=paths['2'];old=m.read(cap/'report.json');old_parent=base.ROOT/'LightGenV2/tasks/t18_optical_residual_ablation/runs/hardware/caltech20_residual_TUCam_20261022/layerwise_L4_on_20261023/e2h/mask_fit/best.pt'
    old_model,_=m.model_for(old_parent,old['PT_SHA256'],4,reference);loop.validate_state(m,model,m.state_cpu(old_model),m.allowed_keys(model,4))
    saved_arrays=base.identities(m,paths);fields.pop('2');paths.pop('2')
    m.save(out/'admission.json',dict(passed=True,parent_correct=273,parent_PT_SHA256=m.sha(parent),all_old_8_arrays_SHA256=saved_arrays,
        response_report_SHA256=p['response_report_SHA256'],response_map_SHA256=p['response_map_SHA256'],response_TRAIN_fit_only=True,old_E2_used_only_for_source_baseline=True,
        training_fields=list(fields),old_E2_not_used_after_mask_changes=True,TRAIN=1600,TEST=411,VALID=0,SDK_used=False))
    return model,data,dm,fields,paths,parent,reference,gain,bg,old['phase_SHA256']


def fit_mask(model,data,dm,fields,paths,parent,reference,gain,bg,c,p,out):
    np,torch,m,loop=base.numerical_modules()
    from torch.nn import functional as F
    from lab_caltech20_dvp_TRAIN_regularization_20261021 import TRAINTransforms
    assert not out.exists() and not m.LOCK.exists();out.mkdir()
    original=m.state_cpu(model);keys=m.allowed_keys(model,3);lineage=base.identities(m,paths)
    score,prob=predictive_score(model,gain,bg,data,fields,'test');initial_train,_=predictive_score(model,gain,bg,data,fields,'train')
    protocol=dict(source_git=m.git_head(),source_SHA256=m.sha(__file__),policy=p,data_SHA256=dm['sha256'],parent_PT_SHA256=m.sha(parent),prefix_SHA256=lineage,
        allowed_keys=sorted(keys),TRAIN=1600,TEST=411,VALID=0,TEST_gradient=False,TEST_selection='calibrated_hybrid_prediction_NOT_physical_accuracy',
        response_map_SHA256=p['response_map_SHA256'],response_TRAIN_flat_only=True,new_inference_layers=0,hardware_changed=False)
    m.save(out/'protocol.json',protocol);m.save(out/'baseline.json',dict(calibrated_hybrid_prediction=score,TRAIN=initial_train,not_physical_accuracy=True))
    np.savez_compressed(out/'baseline_TEST_predictions.npz',ids=data['test'][2],labels=data['test'][1],probabilities=prob)
    best,bnll,selected=score['correct'],score['nll'],0;loop.saved_model(m,model,out/'best.pt',protocol)
    loop.gpu_admission(m,c);work=make_predictor(model,gain,bg).cuda();recipe=p['mask_recipe']
    for name,param in work.named_parameters():param.requires_grad_(name in keys)
    groups=[]
    for prefix,lr in [('phase_sets.',recipe['phase_lr']),('electronic.',recipe['electronic_lr']),('fusion_logits.',recipe['alpha_lr']),('linear.',recipe['head_lr'])]:
        ps=[v for n,v in work.named_parameters() if n in keys and n.startswith(prefix)]
        if ps:groups.append(dict(params=ps,lr=lr,base_lr=lr,weight_decay=0 if prefix=='fusion_logits.' else recipe['weight_decay']))
    opt=torch.optim.AdamW(groups);held={};hook=work.linear.register_forward_hook(lambda module,args,value:held.update(logits=value))
    transforms=TRAINTransforms(work,'1',{**c,'CCD_gain_sigma':.02,'CCD_additive_sigma_DN':.5,'CCD_pixel_dropout_p':.002})
    rng=np.random.default_rng(73);history=[];start=time.monotonic();updates=epoch=0
    prior=np.asarray(c['original_TRAIN_counts'],float);weights=torch.tensor(.5+.5*np.sqrt(prior/prior.mean()),dtype=torch.float32,device='cuda')
    try:
        for epoch in range(1,recipe['epochs']+1):
            work.cuda().train();order=rng.permutation(1600);total=0.;used=0
            scale=.1+.9*.5*(1+np.cos(np.pi*(epoch-1)/max(1,recipe['epochs']-1)))
            for group in opt.param_groups:group['lr']=group['base_lr']*scale
            for begin in range(0,1600,8):
                ix=order[begin:begin+8];x=loop.put_fields(work,data,fields,'train',ix,'cuda');labels=torch.from_numpy(data['train'][1][ix].astype(np.int64)).cuda()
                opt.zero_grad(set_to_none=True);work(x);loss=F.cross_entropy(held['logits'],labels,weight=weights,label_smoothing=.03)
                assert torch.isfinite(loss);loss.backward();ps=[v for v in work.parameters() if v.requires_grad]
                assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in ps)
                torch.nn.utils.clip_grad_norm_(ps,1.);opt.step();updates+=1
                with torch.no_grad():
                    for v in work.fusion_logits:
                        if v.requires_grad:v.clamp_(.4001001,.95)
                used+=len(ix);total+=float(loss.detach())*len(ix)
                if begin%160==0:m.save(out/'progress.json',dict(state='admitted_response_TRAIN_mask',epoch=epoch,TRAIN_consumed=used,TRAIN_loss=total/used,optimizer_updates=updates,selected_predictive_correct=best,not_physical_accuracy=True,elapsed_s=time.monotonic()-start))
            assert used==1600
            if epoch==1 or epoch%2==0 or epoch==recipe['epochs'] or time.monotonic()-start>=recipe['seconds']:
                metric,_=predictive_score(work,gain,bg,data,fields,'test')
                if metric['correct']>best or (metric['correct']==best and metric['nll']<bnll-1e-4):
                    best,bnll,selected=metric['correct'],metric['nll'],epoch;loop.saved_model(m,work,out/'best.pt',protocol)
                row=dict(epoch=epoch,TRAIN_consumed=1600,TRAIN_loss=total/1600,calibrated_TEST_predictive_correct=metric['correct'],selected_predictive_correct=best,selected_epoch=selected,not_physical_accuracy=True,elapsed_s=time.monotonic()-start)
                history.append(row);m.save(out/'history.json',history);m.save(out/'progress.json',row);print(json.dumps(row),flush=True)
            if (best>=305 and selected>0) or time.monotonic()-start>=recipe['seconds']:break
        loop.validate_state(m,work,original,keys);loop.saved_model(m,work,out/'last.pt',protocol)
    finally:transforms.remove();hook.remove()
    results={}
    for label in ('best','last'):
        pt=out/(label+'.pt');native,_=m.model_for(pt,m.sha(pt),4,reference);loop.validate_state(m,native,original,keys)
        results[label]=dict(PT_SHA256=m.sha(pt),original_native_state_load=True)
        for role in ('train','test'):
            metric,prob=predictive_score(native,gain,bg,data,fields,role);results[label]['calibrated_'+role]=metric
            np.savez_compressed(out/(label+'_'+role+'_predictions.npz'),ids=data[role][2],labels=data[role][1],probabilities=prob)
        nominal,_=base.evaluate(native,data,fields,'test');results[label]['nominal_unmeasured_TEST_counterfactual']=nominal
    assert results['best']['calibrated_test']['correct']==best and base.identities(m,paths)==lineage
    report=dict(state='complete',selected_predictive_correct=best,selected_epoch=selected,selected_E0=selected==0,actual_epochs=epoch,optimizer_updates=updates,results=results,
        initial_calibrated_TRAIN=initial_train,strict_original_state_and_CPU=True,full_physical=False,TEST_development=True,TEST_gradient=False,selection_not_physical_accuracy=True,
        response_TRAIN_only=True,new_inference_layers=0,upstream_bitwise_frozen=True,TRAIN_transform_counts=transforms.counts,protocol_SHA256=m.sha(out/'protocol.json'))
    m.save(out/'report.json',report);m.save(out/'independent_audit.json',dict(passed=True,report_SHA256=m.sha(out/'report.json'),prefix_SHA256=lineage,all_measured_ancestors_bitwise_frozen=True))
    completed=[v for v in out.iterdir() if v.is_file() and v.name!='completed_files_SHA.json']
    m.save(out/'completed_files_SHA.json',dict(files=[dict(name=v.name,SHA256=m.sha(v),bytes=v.stat().st_size) for v in sorted(completed)],original_state_only=True,not_physical_accuracy=True))
    chosen,_=m.model_for(out/'best.pt',m.sha(out/'best.pt'),4,reference);del work,opt;torch.cuda.empty_cache();return chosen,out/'best.pt',report


def run(a):
    np,torch,m,loop=base.numerical_modules()
    import lab_caltech20_tucam_capture_20261023 as capture
    import lab_caltech20_TRAIN_response_operator_20261024 as operator
    import lab_caltech20_TRAIN_flat_response_20261024 as flat
    import lab_caltech20_tucam_E2_mask_retake_20261023 as phase_source
    from lab_caltech20_tucam_E2_mask_retake_20261023 import phase_probe
    p=m.read(a.profile);validate_policy(p);assert m.sha(base.__file__)==p['suffix_engine_SHA256'] and m.sha(capture.__file__)==p['capture_engine_SHA256']
    assert m.sha(operator.__file__)==p['operator_SHA256'] and m.sha(base.HERE/p['base_profile'])==p['base_profile_SHA256']
    assert m.sha(flat.__file__)==p['flat_engine_SHA256'] and m.sha(phase_source.__file__)==p['phase_probe_engine_SHA256']
    proof=m.read(a.out.with_suffix('.operator_selftest.json'));assert proof['passed'] and proof['operator_SHA256']==p['operator_SHA256'] and not proof['SDK_used'] and not proof['GPU_used']
    c=m.read(base.HERE/p['base_profile']);base.validate_config(c);c['intermediate_target']=305
    assert not a.out.exists() and not m.LOCK.exists();a.out.mkdir();torch.set_num_threads(2);torch.manual_seed(73)
    m.save(a.out/'protocol.json',dict(source_git=m.git_head(),source_SHA256=m.sha(__file__),profile_SHA256=m.sha(a.profile),policy=p,scheduler=False,automatic_retry=False,TEST_gradient=False))
    model,data,dm,fields,paths,parent,reference,gain,bg,old_phase=admit(p,c,a.out);original=m.state_cpu(model)
    model,parent,fit=fit_mask(model,data,dm,fields,paths,parent,reference,gain,bg,c,p,a.out/'mask_fit')
    changed=[k for k in ('phase_sets.5','phase_sets.6','phase_sets.7','phase_sets.8') if not torch.equal(model.state_dict()[k],original[k])]
    new_phase=phase_probe(model,data,fields,'2',a.out/'candidate_E2_phase.bmp');loop.validate_state(m,model,original,m.allowed_keys(model,3))
    train=fit['results']['best']['calibrated_train'];before=fit['initial_calibrated_TRAIN']
    allowed=fit['selected_predictive_correct']>=305 and not fit['selected_E0'] and bool(changed) and new_phase!=old_phase and train['nll']<before['nll']
    m.save(a.out/'mask_deployment_audit.json',dict(passed=allowed,selected_predictive_correct=fit['selected_predictive_correct'],prediction_not_physical_gate=True,selected_E0=fit['selected_E0'],
        changed_E2_phase_keys=changed,new_phase_SHA256=new_phase,old_phase_SHA256=old_phase,original_state_CPU_and_ancestors_audited=True,TRAIN_response_NLL_improved=train['nll']<before['nll']))
    if not allowed:
        m.save(a.out/'report.json',dict(state='calibrated_mask_proposal_unqualified',new_CCD=0,G2_not_started=True,SDK_released=not m.LOCK.exists(),source_actual_correct_still273=True));return
    folder=a.out/'s_2';folder.mkdir();hardware=base.ROOT/c['hardware_by_stage']['2']['path'];assert m.sha(hardware)==c['hardware_by_stage']['2']['SHA256']
    assert shutil.disk_usage(base.ROOT).free>=120*1024**3
    fields=capture.acquire(model,data,dm,fields,paths,hardware,'2',folder/'capture',parent,interstage_fit=True);paths['2']=folder/'capture'
    assert m.read(paths['2']/'report.json')['phase_SHA256']==new_phase
    actual={**c,'phase_lr':.0002,'electronic_lr':.001,'alpha_lr':.001,'head_lr':.002}
    model,parent,result=base.fit_suffix(model,data,dm,fields,paths,parent,reference,actual,p['observed_recipe'],folder/'fit',305)
    m.save(folder/'endpoint_report.json',dict(actual_E2_correct=result['selected_correct'],PT=str(parent),PT_SHA256=m.sha(parent),strict_original_CPU=result['strict_original_CPU'],prefix_SHA256=base.identities(m,paths),full_physical=False))
    if result['selected_correct']<305:
        m.save(a.out/'report.json',dict(state='actual_E2_gate_unmet',actual_E2_correct=result['selected_correct'],strict_original_CPU=result['strict_original_CPU'],prefix_SHA256=base.identities(m,paths),G2_not_started=True,full_physical=False,TEST_development=True,SDK_released=not m.LOCK.exists(),old_E2_preserved=True));return
    # Original physical gate, never the surrogate TEST prediction, controls G2.
    assert result['strict_original_CPU'] and result['results']['best']['test']['correct']>=305
    folder=a.out/'s_3';folder.mkdir();hardware=base.ROOT/c['hardware_by_stage']['3']['path'];assert m.sha(hardware)==c['hardware_by_stage']['3']['SHA256']
    assert shutil.disk_usage(base.ROOT).free>=120*1024**3
    fields=capture.acquire(model,data,dm,fields,paths,hardware,'3',folder/'capture',parent,interstage_fit=True);paths['3']=folder/'capture'
    assert sum(v.numel() for n,v in model.named_parameters() if n in m.allowed_keys(model,5))==5150
    model,parent,final=base.fit_suffix(model,data,dm,fields,paths,parent,reference,actual,p['final_recipe'],folder/'fit',340)
    m.save(a.out/'report.json',dict(state='complete',actual_E2_correct=result['selected_correct'],full_physical_correct=final['selected_correct'],target=340,target_met=final['selected_correct']>=340,
        strict_original_CPU=final['strict_original_CPU'],prefix_SHA256=base.identities(m,paths),full_physical=True,TEST_development=True,SDK_released=not m.LOCK.exists(),old_E2_preserved=True,
        new_inference_layers=0,response_map_not_deployed=True,PT=str(parent),PT_SHA256=m.sha(parent)))


if __name__=='__main__':
    q=argparse.ArgumentParser();q.add_argument('--profile',type=Path,required=True);q.add_argument('--out',type=Path,required=True);a=q.parse_args()
    try:run(a)
    except BaseException as e:
        if a.out.exists():
            _,_,m,_=base.numerical_modules();m.save(a.out/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),automatic_retry=False,all_data_preserved=True))
        raise
