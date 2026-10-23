"""Human-authorized E2-mask retrain and ONLY its new downstream captures.

Retains the three validated measured layers, and the old unsuccessful E2.
Uses the published original-model fit/capture implementations; no new model,
surrogate camera, hardware calibration, retry loop or periodic task is added.
"""
import argparse,traceback,shutil
from pathlib import Path
import lab_caltech20_tucam_layerwise_20261023 as base
import lab_caltech20_tucam_TRAIN_distill_20261023 as teacher
import lab_caltech20_tucam_gap10_20261023 as gap
import lab_caltech20_tucam_suffix_response_20261023 as response


def validate_policy(p):
    assert p['schema'] in ('TUCam_L4_human_E2_mask_retake_v1','TUCam_L4_human_E2_mask_joint_recovery_v1','TUCam_L4_human_E2_mask_boundary_recovery_v1') and p['human_choice']=='keep305_retrain_E2_only'
    joint=p['schema']=='TUCam_L4_human_E2_mask_joint_recovery_v1'
    boundary=p['schema']=='TUCam_L4_human_E2_mask_boundary_recovery_v1';recovery=joint or boundary
    assert p['completed_controller'].endswith('/e2q' if boundary else '/e2p' if joint else '/e2a') and p['passed_G1_fit'].endswith('/g1s/s_1/fit')
    assert p['passed_G1_SHA256']=='8fa8544f54069e91e85dc6478519edc1b449bfbd9d17c47d0c48b32abe958e2b'
    assert list(p['reuse_captures'])==['router','0','1'] and p['old_E2_capture'].endswith('/g1s/s_2/capture')
    assert p['intermediate_target']==305 and p['final_target']==340 and p['mask_training_stop_target']==(305 if recovery else 411)
    assert p['learning_rates']==({} if boundary else dict(phase_lr=.00005,electronic_lr=.0001,alpha_lr=.0001,head_lr=.0005) if joint else dict(phase_lr=.0002,electronic_lr=.0002,alpha_lr=.0002,head_lr=0.))
    assert p['TRAIN_teacher']==dict(temperature=2,mixture=.3) and p['weight_decay']==.01
    assert p['TRAIN_noise']==dict(CCD_gain_sigma=.02,CCD_additive_sigma_DN=.5,CCD_pixel_dropout_p=.002)
    assert p['mask_recipe']==dict(name='E2_new_mask_local_TRAIN_boundary_recovery' if boundary else 'E2_TRAIN_supported_mask_joint_CE_recovery' if joint else 'E2_original_mask_focused_fixed_head',epochs=0 if boundary else 24,seconds=300 if boundary else 900 if joint else 1200,lr_scale=1,TRAIN_class_emphasis=.75)
    if recovery:
        assert p['candidate_mask_fit'].endswith('/e2q/mask_fit' if boundary else '/e2p/mask_fit') and p['candidate_checkpoint']==('best' if boundary else 'last')
        assert p['candidate_TRAIN_support']==dict(correct_not_lower=True,NLL_improvement=.0001)
    if boundary:
        import lab_caltech20_TRAIN_local_boundary_20261023 as local
        local.validate(p['TRAIN_local_boundary']);assert 'skip_classes' not in p['TRAIN_local_boundary']
        assert len(p['linear_engine_SHA256'])==len(p['TRAIN_local_engine_SHA256'])==64 and p['fresh_native_TRAIN_features'] is True
        assert p['native_feature_source']==p['candidate_mask_fit']
    assert p['observed_recipe']==dict(name='new_E2_observed_joint_adaptation',epochs=48,seconds=1500,lr_scale=1,TRAIN_class_emphasis=.75)
    assert p['observed_learning_rates']==dict(phase_lr=.0002,electronic_lr=.001,alpha_lr=.001,head_lr=.002)
    assert p['TEST_gradient'] is False and p['new_inference_layers']==0 and p['scheduler'] is False
    for key in ('base_profile_SHA256','suffix_engine_SHA256','teacher_engine_SHA256','capture_engine_SHA256'):
        assert len(p[key])==64


def phase_probe(model,data,fields,stage,dest):
    """Reproduce an existing native phase encoding with TRAIN only, no SDK."""
    np,torch,m,_=base.numerical_modules()
    from PIL import Image
    class Collected(Exception):pass
    packet={};model.cpu().eval();model.measured={};images,n,_=m.padded(data,'train',0)
    def collect(current,amplitudes,pixels,ids):
        assert current==stage;packet['phase']=m.phase_native(pixels);raise Collected()
    try:
        with torch.inference_mode():m.layer_forward(model,images,data['train'][2][:n],stage,{k:np.asarray(v['train'][:n]).copy() for k,v in fields.items()},collect)
    except Collected:pass
    assert packet['phase'].shape==(1200,1920);Image.fromarray(packet['phase']).save(dest);model.measured={};return m.sha(dest)


def admit(p,c,out):
    np,_,m,loop=base.numerical_modules();old=base.ROOT/p['completed_controller']
    assert m.read(old.with_suffix('.return.json'))['returncode']==0 and not m.LOCK.exists()
    r=m.read(old/'report.json');joint=p['schema']!='TUCam_L4_human_E2_mask_retake_v1'
    assert r['state']==('E2_mask_no_deployable_update' if joint else 'E2_native_linear_gate_unmet') and r['next_layer_not_started'] and r['SDK_released']
    assert [e['stage'] for e in r['endpoints']]==(['router','0','1'] if joint else ['router','0','1','2']) and r['endpoints'][-1]['selected_correct']==(305 if joint else 282)
    data,dm=m.data_for(base.ROOT/c['sequence']);assert dm['sha256']==c['data_SHA256'];fields={};paths={}
    for stage,name in p['reuse_captures'].items():
        cap=base.ROOT/name;q=m.read(cap/'report.json');a=m.read(cap/'independent_audit.json');proto=m.read(cap/'protocol.json')
        assert a['passed'] and a['matched_samples']==2011 and a['report_SHA256']==m.sha(cap/'report.json')
        assert a['protocol_SHA256']==m.sha(cap/'protocol.json')==q['protocol_SHA256'] and q['physical_ccd']==2011
        assert q['data_SHA256']==dm['sha256'] and q['hardware_SHA256']==c['hardware_by_stage'][stage]['SHA256'] and q['phase_SHA256']==m.sha(cap/'phase.bmp')
        assert q['PT_SHA256']==proto['PT_SHA256']
        if 'prefix_SHA256' in proto:assert proto['prefix_SHA256']==base.identities(m,paths)
        fields[stage]={}
        for role,n in (('train',1600),('test',411)):
            values=cap/role/'ccd.npy';assert m.sha(values)==q['arrays'][role]['SHA256'];array=np.load(values,mmap_mode='r')
            assert array.shape==(n,478,478) and array.dtype==np.uint8;fields[stage][role]=array
        paths[stage]=cap
    fitdir=base.ROOT/p['passed_G1_fit'];parent=fitdir/'best.pt';fit=m.read(fitdir/'report.json');audit=m.read(fitdir/'independent_audit.json');endpoint=m.read(fitdir.parent/'endpoint_report.json')
    assert m.sha(parent)==p['passed_G1_SHA256']==fit['results']['best']['PT_SHA256']==endpoint['PT_SHA256']
    assert fit['strict_original_CPU'] and fit['state']=='complete' and fit['selected_correct']==305 and audit['passed'] and audit['report_SHA256']==m.sha(fitdir/'report.json')
    assert base.identities(m,paths)==endpoint['prefix_SHA256'];reference=base.ROOT/c['reference'];assert m.sha(reference)==c['reference_SHA256']
    model,_=m.model_for(parent,p['passed_G1_SHA256'],4,reference);score,_=base.evaluate(model,data,fields,'test');assert score['correct']==305
    previous=base.ROOT/p['old_E2_capture'];old_report=m.read(previous/'report.json');old_audit=m.read(previous/'independent_audit.json')
    assert old_audit['passed'] and old_audit['report_SHA256']==m.sha(previous/'report.json') and old_report['PT_SHA256']==p['passed_G1_SHA256']
    old_phase=phase_probe(model,data,fields,'2',out/'original_E2_phase.bmp');assert old_phase==old_report['phase_SHA256']
    m.save(out/'admission.json',dict(passed=True,parent_PT_SHA256=m.sha(parent),parent_correct=305,reused_stages=list(fields),reused_frames=6033,
        prefix_SHA256=base.identities(m,paths),old_E2_preserved=str(previous),old_E2_phase_SHA256=old_phase,old_E2_used_for_fit=False,data_SHA256=dm['sha256'],SDK_used=False,TEST_gradient=False))
    return model,data,dm,fields,paths,parent,reference,r['endpoints'][:3],old_phase


def TRAIN_candidate(p,model,dm,paths,reference,out):
    """Admit a declared optimization checkpoint by TRAIN, never as a deployed score."""
    _,torch,m,loop=base.numerical_modules();source=base.ROOT/p['candidate_mask_fit'];r=m.read(source/'report.json');a=m.read(source/'independent_audit.json');proto=m.read(source/'protocol.json')
    assert r['state']=='complete' and r['strict_original_CPU'] and a['passed'] and a['report_SHA256']==m.sha(source/'report.json')
    assert proto['data_SHA256']==dm['sha256'] and proto['prefix_SHA256']==base.identities(m,paths) and r['actual_epochs']==24
    boundary=p['schema']=='TUCam_L4_human_E2_mask_boundary_recovery_v1';checkpoint=p['candidate_checkpoint']
    pt=source/(checkpoint+'.pt');assert m.sha(pt)==r['results'][checkpoint]['PT_SHA256'];candidate,_=m.model_for(pt,m.sha(pt),4,reference)
    original=m.state_cpu(model);loop.validate_state(m,candidate,original,m.allowed_keys(candidate,3) if boundary else m.allowed_keys(candidate,3)-{'linear.weight','linear.bias'})
    baseline=m.read(base.ROOT/p['passed_G1_fit']/'report.json')['results']['best']['train'];last=r['results']['best']['train'] if boundary else r['results']['last']['train']
    supported=last['correct']>=baseline['correct'] and last['nll']<baseline['nll']-p['candidate_TRAIN_support']['NLL_improvement']
    m.save(out/'candidate_TRAIN_admission.json',dict(passed=supported,source_fit=str(source),source_PT_SHA256=m.sha(pt),choice='completed_joint_best_development_checkpoint_NOT_yet_deployed' if boundary else 'predetermined_last_TRAIN_optimization_state_NOT_deployed_best',
        checkpoint=checkpoint,source_checkpoint_selected_on_TEST_development=boundary,parent_TRAIN=baseline['correct'],candidate_TRAIN=last['correct'],parent_TRAIN_NLL=baseline['nll'],candidate_TRAIN_NLL=last['nll'],TEST_used_for_candidate_admission=False,original_head_frozen=not boundary,measured_ancestors_frozen=True,old_E2_used=False,new_CCD=0))
    return candidate,pt,supported


def run(a):
    _,torch,m,loop=base.numerical_modules();p=m.read(a.profile);validate_policy(p)
    from lab_caltech20_tucam_capture_20261023 import acquire
    import lab_caltech20_tucam_capture_20261023 as capture
    assert m.sha(base.HERE/p['base_profile'])==p['base_profile_SHA256'] and m.sha(base.__file__)==p['suffix_engine_SHA256']
    assert m.sha(teacher.__file__)==p['teacher_engine_SHA256'] and m.sha(capture.__file__)==p['capture_engine_SHA256']
    proof=m.read(a.out.with_suffix('.mask_selftest.json'))
    assert proof['passed'] and proof['source_SHA256']==m.sha(__file__) and not proof['SDK_used'] and not proof['GPU_used']
    c=m.read(base.HERE/p['base_profile']);base.validate_config(c);c['intermediate_target']=305
    assert not a.out.exists() and not m.LOCK.exists();a.out.mkdir();torch.set_num_threads(2);torch.manual_seed(c['seed'])
    m.save(a.out/'protocol.json',dict(source_git=m.git_head(),source_SHA256=m.sha(__file__),profile_SHA256=m.sha(a.profile),policy=p,config=c,scheduler=False,automatic_retry=False,TEST_gradient=False))
    model,data,dm,fields,paths,parent,reference,endpoints,old_phase=admit(p,c,a.out);original=m.state_cpu(model)
    joint=p['schema']!='TUCam_L4_human_E2_mask_retake_v1';boundary=p['schema']=='TUCam_L4_human_E2_mask_boundary_recovery_v1'
    if joint:
        model,parent,supported=TRAIN_candidate(p,model,dm,paths,reference,a.out)
        if not supported:
            m.save(a.out/'report.json',dict(state='E2_mask_candidate_no_TRAIN_support',endpoints=endpoints,next_layer_not_started=True,new_physical_CCD=0,SDK_released=not m.LOCK.exists(),old_E2_preserved=True));return
    tuned={**c,**p['learning_rates'],**p['TRAIN_noise'],'TRAIN_teacher':p['TRAIN_teacher'],'weight_decay':p['weight_decay']}
    m.save(a.out/'progress.json',dict(state='E2_mask_joint_boundary_recovery' if joint else 'retraining_E2_masks_fixed_readout',completed_stages=3,physical_capture_not_started=True))
    fitter=base.fit_suffix if joint else teacher.fit_suffix
    if boundary:
        import lab_caltech20_tucam_TRAIN_linear_20261023 as linear
        assert m.sha(linear.__file__)==p['linear_engine_SHA256']
        tuned.update(TRAIN_readout='cached_TRAIN_local_boundary',TRAIN_local_boundary=p['TRAIN_local_boundary'],TRAIN_local_engine_SHA256=p['TRAIN_local_engine_SHA256'],native_feature_source=p['native_feature_source'],fresh_native_TRAIN_features=True)
        fitter=linear.fit_suffix
    model,parent,fit=fitter(model,data,dm,fields,paths,parent,reference,tuned,p['mask_recipe'],a.out/'mask_fit',p['mask_training_stop_target'])
    frozen_head=all(torch.equal(model.state_dict()[k],original[k]) for k in ('linear.weight','linear.bias'))
    if not joint:assert frozen_head
    for label in ('best','last'):
        pt=a.out/'mask_fit'/(label+'.pt');verified,_=m.model_for(pt,m.sha(pt),4,reference)
        loop.validate_state(m,verified,original,m.allowed_keys(verified,3) if joint else m.allowed_keys(verified,3)-{'linear.weight','linear.bias'})
        if not joint:assert all(torch.equal(verified.state_dict()[k],original[k]) for k in ('linear.weight','linear.bias'))
    changed=[k for k in ('phase_sets.5','phase_sets.6','phase_sets.7','phase_sets.8') if not torch.equal(model.state_dict()[k],original[k])]
    new_phase=phase_probe(model,data,fields,'2',a.out/'candidate_E2_phase.bmp')
    loop.validate_state(m,model,original,m.allowed_keys(model,3) if joint else m.allowed_keys(model,3)-{'linear.weight','linear.bias'})
    deployable=fit['selected_correct']>=305 and not fit['selected_E0'] and bool(changed) and new_phase!=old_phase
    m.save(a.out/'mask_deployment_audit.json',dict(passed=deployable,head_bitwise_frozen=frozen_head,changed_E2_phase_keys=changed,selected_correct=fit['selected_correct'],selected_E0=fit['selected_E0'],
        original_E2_phase_SHA256=old_phase,candidate_E2_phase_SHA256=new_phase,strict_original_CPU=fit['strict_original_CPU'],upstream_and_input_ancestors_frozen=True,new_CCD=0))
    if not deployable:
        m.save(a.out/'report.json',dict(state='E2_mask_no_deployable_update',endpoints=endpoints,next_layer_not_started=True,new_physical_CCD=0,SDK_released=not m.LOCK.exists(),old_E2_preserved=True));return
    endpoints[-1]=response.endpoint(m,'1',parent,fit,paths)
    for stage in ('2','3'):
        folder=a.out/('s_'+stage);folder.mkdir();target=340 if stage=='3' else 305
        m.save(a.out/'progress.json',dict(state='capturing',stage=stage,parent_PT_SHA256=m.sha(parent),completed_stages=len(fields)))
        hw=base.ROOT/c['hardware_by_stage'][stage]['path'];assert m.sha(hw)==c['hardware_by_stage'][stage]['SHA256']
        assert shutil.disk_usage(base.ROOT).free>=c['minimum_free_GiB']*1024**3,'Preserve all data: insufficient capture disk margin'
        fields=acquire(model,data,dm,fields,paths,hw,stage,folder/'capture',parent,interstage_fit=True);paths[stage]=folder/'capture'
        if stage=='2':assert m.read(paths[stage]/'report.json')['phase_SHA256']==new_phase
        observed={**c,**p['observed_learning_rates']}
        if stage=='2':
            model,parent,fit=base.fit_suffix(model,data,dm,fields,paths,parent,reference,observed,p['observed_recipe'],folder/'f0',target)
            if fit['selected_correct']<target:
                teacher_cfg={**c,'TRAIN_teacher':p['TRAIN_teacher'],'weight_decay':.01}
                recipe=dict(name='new_E2_TRAIN_teacher_regularized',epochs=36,seconds=1200,lr_scale=1,TRAIN_class_emphasis=.75)
                model,parent,fit=teacher.fit_suffix(model,data,dm,fields,paths,parent,reference,teacher_cfg,recipe,folder/'teacher',target)
        else:
            for i,recipe in enumerate(c['recipes']):
                model,parent,fit=base.fit_suffix(model,data,dm,fields,paths,parent,reference,c,recipe,folder/('f'+str(i)),target)
                if fit['selected_correct']>=target:break
            if fit['selected_correct']<target:
                updated,recipe,observed_gap=gap.diagnostic_recipe(fit,c)
                m.save(folder/'error_diagnosis.json',dict(TRAIN_TEST_gap=observed_gap,TEST_gradient=False,recipe=recipe))
                model,parent,fit=base.fit_suffix(model,data,dm,fields,paths,parent,reference,updated,recipe,folder/'extra',target)
        e=response.endpoint(m,stage,parent,fit,paths);endpoints.append(e);m.save(folder/'endpoint_report.json',e)
        if not e['target_met']:
            m.save(a.out/'report.json',dict(state='E2_mask_retake_stage_gate_unmet',endpoints=endpoints,next_layer_not_started=True,old_E2_preserved=True,SDK_released=not m.LOCK.exists()));return
    m.save(a.out/'report.json',dict(state='complete',endpoints=endpoints,full_physical=True,strict_original_CPU=True,TEST_development=True,old_E2_preserved=True,SDK_released=not m.LOCK.exists()))


def selftest():
    np,torch,m,_=base.numerical_modules();torch.set_num_threads(2);torch.manual_seed(73)
    model=m.MeasuredPrefix(4);keys=m.allowed_keys(model,3)
    assert {k for k in keys if k.startswith('phase_sets.')}=={'phase_sets.'+str(i) for i in range(5,10)}
    before={k:v.detach().clone() for k,v in model.linear.state_dict().items()};optimizer=torch.optim.AdamW(model.linear.parameters(),lr=0.,weight_decay=.01)
    features=torch.randn(8,256,requires_grad=True);labels=torch.arange(8)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True);loss=torch.nn.functional.cross_entropy(model.linear(features),labels);loss.backward();optimizer.step()
    assert torch.isfinite(features.grad).all() and features.grad.abs().sum()>0
    assert all(np.array_equal(v.detach().numpy().view(np.uint8),before[k].numpy().view(np.uint8)) for k,v in model.linear.state_dict().items())
    print(__import__('json').dumps(dict(passed=True,head_zero_LR_bitwise_frozen=True,gradient_through_frozen_head_to_features=True,E2_G2_masks_are_legal=True,SDK_used=False,GPU_used=False,TEST_used=False)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--profile',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    try:run(a);_,_,m,_=base.numerical_modules();m.save(a.out.with_suffix('.return.json'),dict(returncode=0))
    except BaseException as error:
        _,_,m,_=base.numerical_modules();m.save(a.out/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),automatic_retry=False,all_data_preserved=True));m.save(a.out.with_suffix('.return.json'),dict(returncode=1,error=repr(error)));raise
