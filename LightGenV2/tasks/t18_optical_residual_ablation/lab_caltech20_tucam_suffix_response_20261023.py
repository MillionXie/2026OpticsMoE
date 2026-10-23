"""Finite, audited diagnosis response after an existing measured prefix.

Reuses the published original suffix fitter. No inference algorithm is added,
no active controller is patched, and no captured-input ancestor is unfrozen.
"""
import argparse,traceback
from pathlib import Path
import lab_caltech20_tucam_layerwise_20261023 as base
import lab_caltech20_tucam_gap10_20261023 as gap


def validate_policy(p):
    fresh=p.get('fresh_E2_after_retake',False);assert isinstance(fresh,bool)
    assert not fresh or p['schema'] in ('TUCam_L4_E2_fusion_capacity_response_v1','TUCam_L4_E2_underfit_response_v1')
    assert p['schema'] in ('TUCam_L4_G1_underfit_response_v1','TUCam_L4_G1_teacher_response_v1','TUCam_L4_G1_linear_response_v1','TUCam_L4_G1_feature_response_v1','TUCam_L4_G1_fullbatch_response_v1','TUCam_L4_G1_CCDnoise_response_v1','TUCam_L4_G1_local_boundary_response_v1','TUCam_L4_E2_underfit_response_v1','TUCam_L4_E2_linear_response_v1','TUCam_L4_E2_fullbatch_response_v1','TUCam_L4_E2_teacher_response_v1','TUCam_L4_E2_fusion_capacity_response_v1','TUCam_L4_E2_local_boundary_response_v1','TUCam_L4_E2_SAM_response_v1','TUCam_L4_E2_feature_response_v1')
    e2=p['schema'] in ('TUCam_L4_E2_underfit_response_v1','TUCam_L4_E2_linear_response_v1','TUCam_L4_E2_fullbatch_response_v1','TUCam_L4_E2_teacher_response_v1','TUCam_L4_E2_fusion_capacity_response_v1','TUCam_L4_E2_local_boundary_response_v1','TUCam_L4_E2_SAM_response_v1','TUCam_L4_E2_feature_response_v1')
    assert p['parent_stage']==('2' if e2 else '1') and list(p['captures'])==(['router','0','1','2'] if e2 else ['router','0','1'])
    assert p['intermediate_target']==305 and p['final_target']==340
    assert p['maximum_parent_correct']==304
    if p['schema']=='TUCam_L4_E2_underfit_response_v1':
        if fresh:
            assert p['minimum_parent_correct']==p['expected_parent_correct']==270 and p['maximum_parent_TRAIN_accuracy']==.95
            assert p['completed_controller'].endswith('/e2v') and 'TRAIN_alpha_release' not in p and 'TRAIN_teacher' not in p
            assert p['TRAIN_teacher_conflict']==dict(temperature=2,mixture=.3,minimum_weighted_KL_over_CE=1.)
            assert len(p['expected_parent_SHA256'])==len(p['parent_TRAIN_history_SHA256'])==64
            assert p['learning_rates']==dict(phase_lr=.0002,electronic_lr=.001,alpha_lr=.001,head_lr=.0005)
            assert p['recipe']==dict(name='E2_fusion_conditioned_actual_TRAIN_CE_only',epochs=36,seconds=900,lr_scale=1,TRAIN_class_emphasis=.75)
            assert p['captures']['2'].endswith('/e2h/s_2/capture') and p['capture_parent_SHA256']=='4dc922d782d6a7842b9f01c287b791d88a2485a40c602813e562d3c0fa8cd0ba'
            assert p['E2_arrays_SHA256']==dict(train='3c42ab4f2095629bbfd6b42daa5f46775f63f3727f9101e0d993c074f481cf7e',test='48e826672b8c9978f38230ec0fa23984f32b684340f88b1da4e425bf5d03b745')
        else:
            assert p['minimum_parent_correct']==205 and p['maximum_parent_TRAIN_accuracy']==.80
            assert p['learning_rates']==dict(phase_lr=.0002,electronic_lr=.001,alpha_lr=.001,head_lr=.002)
            assert p['recipe']==dict(name='E2_observed_underfit_original_suffix',epochs=48,seconds=1500,lr_scale=1,TRAIN_class_emphasis=.75)
            assert p['capture_parent_SHA256']=='8fa8544f54069e91e85dc6478519edc1b449bfbd9d17c47d0c48b32abe958e2b'
            assert p['completed_controller'].endswith('/g1s')
            assert p['TRAIN_alpha_release']==dict(stage=2,samples=80,per_class=4,lower_initial_alpha=.70,minimum_TRAIN_NLL_improvement=.01)
    elif p['schema']=='TUCam_L4_E2_linear_response_v1':
        assert p['minimum_parent_correct']==222 and p['maximum_parent_TRAIN_accuracy']==.95
        assert p['learning_rates']=={} and p['TRAIN_readout']=='one_OAS_pooled_linear'
        assert len(p['linear_engine_SHA256'])==64 and p['completed_controller'].endswith('/e2u')
        assert p['recipe']==dict(name='E2_one_TRAIN_OAS_linear_fit',epochs=1,seconds=600,lr_scale=1,TRAIN_class_emphasis=.75)
        assert p['capture_parent_SHA256']=='8fa8544f54069e91e85dc6478519edc1b449bfbd9d17c47d0c48b32abe958e2b'
    elif p['schema']=='TUCam_L4_E2_fullbatch_response_v1':
        assert p['minimum_parent_correct']==p['expected_parent_correct']==257 and p['maximum_parent_TRAIN_accuracy']==1.
        assert p['minimum_parent_TRAIN_TEST_gap']==.03 and p['learning_rates']=={} and p['TRAIN_readout']=='cached_TRAIN_affine_LBFGS'
        assert len(p['linear_engine_SHA256'])==64 and p['native_feature_source'].endswith('/e2l/s_2/fit') and p['completed_controller'].endswith('/e2l')
        assert p['TRAIN_affine_optimizer']==dict(l2_parent_TRAIN_NLL_fraction=.1,label_smoothing=.03,outer_steps=80)
        assert p['recipe']==dict(name='E2_conditioned_TRAIN_affine_LBFGS',epochs=80,seconds=60,lr_scale=1,TRAIN_class_emphasis=.75)
    elif p['schema']=='TUCam_L4_E2_teacher_response_v1':
        assert p['minimum_parent_correct']==p['expected_parent_correct']==267 and p['maximum_parent_TRAIN_accuracy']==1.
        assert p['minimum_parent_TRAIN_TEST_gap']==.10 and p['completed_controller'].endswith('/e2b')
        assert p['learning_rates']==dict(phase_lr=.00005,electronic_lr=.0002,alpha_lr=.0002,head_lr=.0005)
        assert p['TRAIN_teacher']==dict(temperature=2,mixture=.3) and p['weight_decay']==.01 and len(p['teacher_engine_SHA256'])==64
        assert p['recipe']==dict(name='E2_TRAIN_only_fixed_teacher_cosine',epochs=36,seconds=1200,lr_scale=1,TRAIN_class_emphasis=.75)
    elif p['schema']=='TUCam_L4_E2_fusion_capacity_response_v1':
        if fresh:
            assert p['minimum_parent_correct']==p['expected_parent_correct']==254 and p['maximum_parent_TRAIN_accuracy']==.95 and 'minimum_parent_TRAIN_TEST_gap' not in p
            assert p['expected_parent_SHA256']=='585c65782a04d8abb8fa66976924da3a52f225239934d1293c6059102a6cf160'
            assert p['completed_controller'].endswith('/e2h') and p['captures']['2'].endswith('/e2h/s_2/capture')
            assert p['capture_parent_SHA256']=='4dc922d782d6a7842b9f01c287b791d88a2485a40c602813e562d3c0fa8cd0ba'
            assert p['E2_arrays_SHA256']==dict(train='3c42ab4f2095629bbfd6b42daa5f46775f63f3727f9101e0d993c074f481cf7e',test='48e826672b8c9978f38230ec0fa23984f32b684340f88b1da4e425bf5d03b745')
        else:
            assert p['minimum_parent_correct']==p['expected_parent_correct']==274 and p['maximum_parent_TRAIN_accuracy']==1.
            assert p['minimum_parent_TRAIN_TEST_gap']==.10 and p['completed_controller'].endswith('/e2k')
        assert p['TRAIN_fusion_capacity']==dict(stage=2,samples=1600,proposed_alpha=.4001001,estimators_per_setting=1)
        assert len(p['capacity_engine_SHA256'])==len(p['linear_engine_SHA256'])==len(p['teacher_engine_SHA256'])==64
        assert p['TRAIN_teacher']==dict(temperature=2,mixture=.3) and p['weight_decay']==.01
        assert p['learning_rates']==dict(phase_lr=.00005,electronic_lr=.0002,alpha_lr=.0002,head_lr=.0005)
        assert p['recipe']==dict(name='E2_TRAIN_supported_fusion_joint_teacher',epochs=24,seconds=900,lr_scale=1,TRAIN_class_emphasis=.75)
    elif p['schema']=='TUCam_L4_E2_local_boundary_response_v1':
        import lab_caltech20_TRAIN_local_boundary_20261023 as boundary
        assert p['minimum_parent_correct']==p['expected_parent_correct']==274 and p['maximum_parent_TRAIN_accuracy']==1.
        assert p['minimum_parent_TRAIN_TEST_gap']==.10 and p['completed_controller'].endswith('/e2c')
        assert p['learning_rates']=={} and p['TRAIN_readout']=='cached_TRAIN_local_boundary'
        assert p['native_feature_source'].endswith('/e2k/s_2/fit') and p['fresh_native_TRAIN_features'] is True
        boundary.validate(p['TRAIN_local_boundary']);assert 'skip_classes' not in p['TRAIN_local_boundary']
        assert len(p['linear_engine_SHA256'])==len(p['TRAIN_local_engine_SHA256'])==64
        assert p['recipe']==dict(name='E2_TRAIN_local_binary_rest_full_curvature',epochs=0,seconds=300,lr_scale=1,TRAIN_class_emphasis=.75)
    elif p['schema']=='TUCam_L4_E2_SAM_response_v1':
        assert p['minimum_parent_correct']==p['expected_parent_correct']==282 and p['maximum_parent_TRAIN_accuracy']==1.
        assert p['minimum_parent_TRAIN_TEST_gap']==.10 and p['completed_controller'].endswith('/e2r')
        assert p['TRAIN_SAM']==dict(radius=.05,same_noise_both_passes=True)
        assert p['TRAIN_teacher']==dict(temperature=2,mixture=.3) and p['weight_decay']==.01
        assert len(p['SAM_engine_SHA256'])==len(p['teacher_engine_SHA256'])==64
        assert p['learning_rates']==dict(phase_lr=.00005,electronic_lr=.0002,alpha_lr=.0002,head_lr=.0005)
        assert p['recipe']==dict(name='E2_TRAIN_SAM_original_joint_suffix',epochs=24,seconds=1200,lr_scale=1,TRAIN_class_emphasis=.75)
    elif p['schema']=='TUCam_L4_E2_feature_response_v1':
        assert p['minimum_parent_correct']==p['expected_parent_correct']==282 and p['maximum_parent_TRAIN_accuracy']==1.
        assert p['minimum_parent_TRAIN_TEST_gap']==.10 and p['completed_controller'].endswith('/e2m')
        assert p['learning_rates']=={} and p['TRAIN_readout']=='cached_TRAIN_mixup_dropout'
        assert p['native_feature_source'].endswith('/e2r/s_2/fit') and len(p['linear_engine_SHA256'])==64
        assert p['TRAIN_feature_aug']==dict(dropout=.1,beta=.4,lr=.0002,weight_decay=.01,label_smoothing=.03,cooldown=True)
        assert p['recipe']==dict(name='E2_best282_TRAIN_mixup_dropout_cooldown',epochs=48,seconds=60,lr_scale=1,TRAIN_class_emphasis=.75)
    elif p['schema']=='TUCam_L4_G1_underfit_response_v1':
        assert p['minimum_parent_correct']==250 and p['maximum_parent_TRAIN_accuracy']==.80
        assert p['learning_rates']==dict(phase_lr=.0002,electronic_lr=.001,alpha_lr=.001,head_lr=.002)
        assert p['recipe']==dict(name='G1_observed_underfit_original_suffix',epochs=48,seconds=1500,lr_scale=1,TRAIN_class_emphasis=.75)
    elif p['schema']=='TUCam_L4_G1_teacher_response_v1':
        assert p['minimum_parent_correct']==263 and p['maximum_parent_TRAIN_accuracy']==1.
        assert p['minimum_parent_TRAIN_TEST_gap']==.10 and p['expected_parent_correct']==281
        assert p['learning_rates']==dict(phase_lr=.00005,electronic_lr=.0002,alpha_lr=.0002,head_lr=.0005)
        assert p['TRAIN_teacher']==dict(temperature=2,mixture=.3) and p['weight_decay']==.01
        assert len(p['teacher_engine_SHA256'])==64
        assert p['recipe']==dict(name='G1_TRAIN_only_fixed_teacher_cosine',epochs=36,seconds=1200,lr_scale=1,TRAIN_class_emphasis=.75)
    elif p['schema']=='TUCam_L4_G1_linear_response_v1':
        assert p['minimum_parent_correct']==281 and p['maximum_parent_TRAIN_accuracy']==1.
        assert p['minimum_parent_TRAIN_TEST_gap']==.10 and p['expected_parent_correct']==298
        assert p['learning_rates']=={} and p['TRAIN_readout']=='one_OAS_pooled_linear'
        assert len(p['linear_engine_SHA256'])==64
        assert p['recipe']==dict(name='G1_one_TRAIN_OAS_linear_fit',epochs=1,seconds=600,lr_scale=1,TRAIN_class_emphasis=.75)
        post=p['post_linear_feature_aug']
        assert post['config']==dict(dropout=.1,beta=.4,lr=.001,weight_decay=.05,label_smoothing=.03)
        assert post['recipe']==dict(name='G1_native_TRAIN_feature_mixup_linear',epochs=48,seconds=60,lr_scale=1,TRAIN_class_emphasis=.75)
    elif p['schema']=='TUCam_L4_G1_feature_response_v1':
        assert p['minimum_parent_correct']==298 and p['maximum_parent_TRAIN_accuracy']==1.
        assert p['minimum_parent_TRAIN_TEST_gap']==.10 and p['expected_parent_correct']==298
        assert p['learning_rates']=={} and p['TRAIN_readout']=='cached_TRAIN_mixup_dropout'
        assert len(p['linear_engine_SHA256'])==64 and p['native_feature_source'].endswith('/g1l/s_1/fit')
        assert p['TRAIN_feature_aug']==dict(dropout=.1,beta=.4,lr=.0002,weight_decay=.01,label_smoothing=.03,cooldown=True)
        assert p['recipe']==dict(name='G1_best298_lowerLR_augmentation_cooldown',epochs=48,seconds=60,lr_scale=1,TRAIN_class_emphasis=.75)
    elif p['schema']=='TUCam_L4_G1_fullbatch_response_v1':
        assert p['minimum_parent_correct']==299 and p['maximum_parent_TRAIN_accuracy']==1.
        assert p['minimum_parent_TRAIN_TEST_gap']==.10 and p['expected_parent_correct']==299
        assert p['learning_rates']=={} and p['TRAIN_readout']=='cached_TRAIN_affine_LBFGS'
        assert len(p['linear_engine_SHA256'])==64 and p['native_feature_source'].endswith('/g1l/s_1/fit')
        assert p['TRAIN_affine_optimizer']==dict(l2_parent_TRAIN_NLL_fraction=.1,label_smoothing=.03,outer_steps=80)
        assert p['recipe']==dict(name='G1_illconditioned_TRAIN_fullbatch_affine_LBFGS',epochs=80,seconds=60,lr_scale=1,TRAIN_class_emphasis=.75)
    elif p['schema']=='TUCam_L4_G1_local_boundary_response_v1':
        import lab_caltech20_TRAIN_local_boundary_20261023 as boundary
        continuation='skip_classes' in p['TRAIN_local_boundary'];parent=304 if continuation else 301
        assert p['minimum_parent_correct']==parent and p['maximum_parent_TRAIN_accuracy']==1.
        assert p['minimum_parent_TRAIN_TEST_gap']==.10 and p['expected_parent_correct']==parent
        assert p['learning_rates']=={} and p['TRAIN_readout']=='cached_TRAIN_local_boundary'
        assert len(p['linear_engine_SHA256'])==len(p['TRAIN_local_engine_SHA256'])==64
        assert p['native_feature_source'].endswith('/g1r/s_1/fit' if continuation else '/g1n/s_1/fit') and p['fresh_native_TRAIN_features'] is (not continuation)
        boundary.validate(p['TRAIN_local_boundary'])
        assert p['recipe']==dict(name='G1_TRAIN_remaining_local_binary_rest_full_curvature' if continuation else 'G1_TRAIN_local_binary_rest_full_curvature',epochs=0,seconds=300,lr_scale=1,TRAIN_class_emphasis=.75)
    else:
        assert p['minimum_parent_correct']==299 and p['maximum_parent_TRAIN_accuracy']==1.
        assert p['minimum_parent_TRAIN_TEST_gap']==.10 and p['expected_parent_correct']==299
        assert p['learning_rates']==dict(phase_lr=.00005,electronic_lr=.0002,alpha_lr=.0002,head_lr=.0005)
        assert p['TRAIN_teacher']==dict(temperature=2,mixture=.3) and p['weight_decay']==.01
        assert len(p['teacher_engine_SHA256'])==64
        assert p['TRAIN_noise']==dict(CCD_gain_sigma=.02,CCD_additive_sigma_DN=.5,CCD_pixel_dropout_p=.002)
        assert p['recipe']==dict(name='G1_TRAIN_CCD_robust_original_suffix',epochs=36,seconds=1200,lr_scale=1,TRAIN_class_emphasis=.75)
    assert p['TEST_gradient'] is False and p['new_inference_layers']==0
    assert p['new_CCD_before_gate']==0 and p['scheduler'] is False
    for key in ('base_profile_SHA256','suffix_engine_SHA256','continuation_engine_SHA256'):
        assert len(p[key])==64


def TRAIN_loss_components(row,cfg):
    assert cfg==dict(temperature=2,mixture=.3,minimum_weighted_KL_over_CE=1.)
    ce=(1-cfg['mixture'])*row['TRAIN_CE'];kl=cfg['mixture']*row['TRAIN_KL']
    assert 0<ce<float('inf') and 0<=kl<float('inf') and kl/ce>=cfg['minimum_weighted_KL_over_CE'],'No TRAIN support for teacher-conflict hypothesis'
    return dict(weighted_TRAIN_CE=ce,weighted_TRAIN_KL=kl,ratio=kl/ce,decision_data='TRAIN_only',TEST_used_for_loss_diagnosis=False,remove_teacher_for_one_finite_joint_CE_fit=True)


def admit(p,c,out):
    np,_,m,loop=base.numerical_modules();old=base.ROOT/p['completed_controller']
    assert m.read(old.with_suffix('.return.json'))['returncode']==0 and not m.LOCK.exists()
    r=m.read(old/'report.json')
    expected_state='G1_native_linear_gate_unmet' if p['schema'] in ('TUCam_L4_G1_feature_response_v1','TUCam_L4_G1_fullbatch_response_v1','TUCam_L4_G1_CCDnoise_response_v1') or ('skip_classes' in p.get('TRAIN_local_boundary',{})) else 'G1_teacher_budget_complete_gate_unmet' if 'TRAIN_readout' in p else 'G1_underfit_budget_complete_gate_unmet' if 'TRAIN_teacher' in p else 'stage_target_not_met_after_diagnostic_fit'
    if p['schema']=='TUCam_L4_E2_linear_response_v1':expected_state='E2_underfit_budget_complete_gate_unmet'
    if p['schema']=='TUCam_L4_E2_fullbatch_response_v1':expected_state='E2_native_linear_gate_unmet'
    if p['schema']=='TUCam_L4_E2_teacher_response_v1':expected_state='E2_native_linear_gate_unmet'
    if p['schema']=='TUCam_L4_E2_SAM_response_v1':expected_state='E2_native_linear_gate_unmet'
    if p['schema']=='TUCam_L4_E2_feature_response_v1':expected_state='E2_teacher_budget_complete_gate_unmet'
    if p['schema']=='TUCam_L4_E2_fusion_capacity_response_v1':expected_state='E2_teacher_budget_complete_gate_unmet'
    if p.get('fresh_E2_after_retake',False):expected_state='E2_capacity_joint_gate_unmet' if 'TRAIN_teacher_conflict' in p else 'E2_mask_retake_stage_gate_unmet'
    if p['schema']=='TUCam_L4_E2_local_boundary_response_v1':
        expected_state='E2_capacity_no_TRAIN_support'
        capacity=m.read(old/'s_2/capacity/report.json')
        assert capacity['passed'] and not capacity['accepted'] and capacity['decision_data']=='TRAIN_only'
        assert capacity['parent_PT_SHA256']==r['endpoints'][-1]['PT_SHA256'] and capacity['all_nonhead_frozen']
    assert r['state']==expected_state
    assert r['next_layer_not_started'] and r['SDK_released']
    endpoints=r['endpoints'];assert [e['stage'] for e in endpoints]==list(p['captures'])
    assert all(e['target_met'] for e in endpoints[:-1]) and not endpoints[-1]['target_met']
    data,dm=m.data_for(base.ROOT/c['sequence']);assert dm['sha256']==c['data_SHA256']
    fields={};paths={}
    for stage,name in p['captures'].items():
        cap=base.ROOT/name;audit=m.read(cap/'independent_audit.json');q=m.read(cap/'report.json');protocol=m.read(cap/'protocol.json')
        assert audit['passed'] and audit['matched_samples']==2011
        assert audit['report_SHA256']==m.sha(cap/'report.json')
        assert audit['protocol_SHA256']==m.sha(cap/'protocol.json')==q['protocol_SHA256']
        assert q['physical_ccd']==2011 and q['data_SHA256']==dm['sha256']
        assert q['hardware_SHA256']==c['hardware_by_stage'][stage]['SHA256']
        assert q['phase_SHA256']==m.sha(cap/'phase.bmp')
        assert q['PT_SHA256']==protocol['PT_SHA256']
        if 'prefix_SHA256' in protocol:assert protocol['prefix_SHA256']==base.identities(m,paths)
        fields[stage]={}
        for role,n in (('train',1600),('test',411)):
            array=cap/role/'ccd.npy';assert m.sha(array)==q['arrays'][role]['SHA256']
            if stage=='2' and p.get('fresh_E2_after_retake',False):assert q['arrays'][role]['SHA256']==p['E2_arrays_SHA256'][role]
            values=np.load(array,mmap_mode='r');assert values.shape==(n,478,478) and values.dtype==np.uint8
            fields[stage][role]=values
        paths[stage]=cap
    e=endpoints[-1];parent=Path(e['PT']);assert m.sha(parent)==e['PT_SHA256']
    if p.get('fresh_E2_after_retake',False):assert e['selected_correct']==p['expected_parent_correct'] and e['PT_SHA256']==p['expected_parent_SHA256']
    assert base.identities(m,paths)==e['prefix_SHA256']
    fit=m.read(parent.parent/'report.json');audit=m.read(parent.parent/'independent_audit.json')
    assert fit['state']=='complete' and fit['strict_original_CPU'] and audit['passed']
    assert audit['report_SHA256']==m.sha(parent.parent/'report.json')
    assert fit['results']['best']['PT_SHA256']==e['PT_SHA256']
    assert p['minimum_parent_correct']<=e['selected_correct']<=p['maximum_parent_correct']
    if 'minimum_parent_TRAIN_TEST_gap' in p:
        b=fit['results']['best']
        assert b['train']['accuracy']<=p['maximum_parent_TRAIN_accuracy']
        assert b['train']['accuracy']-b['test']['accuracy']>=p['minimum_parent_TRAIN_TEST_gap']
        assert e['selected_correct']==p['expected_parent_correct']
    else:assert fit['results']['best']['train']['accuracy']<p['maximum_parent_TRAIN_accuracy']
    if 'TRAIN_teacher_conflict' in p:
        history=parent.parent/'history.json';assert m.sha(history)==p['parent_TRAIN_history_SHA256']
        diagnosis=TRAIN_loss_components(m.read(history)[-1],p['TRAIN_teacher_conflict'])
        m.save(out/'TRAIN_loss_diagnosis.json',dict(TRAIN_history_SHA256=m.sha(history),**diagnosis))
    reference=base.ROOT/c['reference'];assert m.sha(reference)==c['reference_SHA256']
    model,_=m.model_for(parent,e['PT_SHA256'],4,reference)
    # Compare to the EXACT PT that produced the last measured stage.
    cap_protocol=m.read(paths[p['parent_stage']]/'protocol.json');captured_parent=base.ROOT/p['capture_parent_PT']
    assert m.sha(captured_parent)==cap_protocol['PT_SHA256']==p['capture_parent_SHA256']
    origin,_=m.model_for(captured_parent,cap_protocol['PT_SHA256'],4,reference)
    loop.validate_state(m,model,m.state_cpu(origin),m.allowed_keys(model,len(fields)))
    score,_=base.evaluate(model,data,fields,'test')
    assert score['correct']==e['selected_correct']==fit['selected_correct']
    m.save(out/'admission.json',dict(parent_PT=str(parent),parent_SHA256=e['PT_SHA256'],parent_report_SHA256=m.sha(parent.parent/'report.json'),
        parent_correct=score['correct'],parent_TRAIN=fit['results']['best']['train']['correct'],data_SHA256=dm['sha256'],
        prefix_SHA256=base.identities(m,paths),captured_masks_and_input_ancestors_frozen=True,new_CCD=0,SDK_used=False,TEST_gradient=False))
    return model,data,dm,fields,paths,parent,reference,list(endpoints)


def endpoint(m,stage,parent,fit,paths):
    target=340 if stage=='3' else 305
    return dict(stage=stage,PT=str(parent),PT_SHA256=m.sha(parent),selected_correct=fit['selected_correct'],target=target,
        target_met=fit['selected_correct']>=target,prefix_SHA256=base.identities(m,paths),strict_original_CPU=True,full_physical=stage=='3',TEST_development=True)


def TRAIN_alpha_probe(model,data,fields,cfg,out):
    """One fixed TRAIN-only response check of a bounded existing alpha."""
    np,torch,m,loop=base.numerical_modules()
    from torch.nn import functional as F
    assert list(fields)==list(base.STAGES[:4]) and cfg==dict(stage=2,samples=80,per_class=4,lower_initial_alpha=.70,minimum_TRAIN_NLL_improvement=.01)
    labels=data['train'][1];ix=np.concatenate([np.flatnonzero(labels==k)[:cfg['per_class']] for k in range(20)])
    assert len(ix)==cfg['samples'] and len(np.unique(ix))==80 and np.bincount(labels[ix],minlength=20).tolist()==[4]*20
    model.cpu().eval();parameter=model.fusion_logits[cfg['stage']];original=parameter.detach().clone();losses=[]
    try:
        for value in (float(original),cfg['lower_initial_alpha']):
            with torch.inference_mode():
                parameter.fill_(value);total=0.
                for begin in range(0,len(ix),8):
                    batch=ix[begin:begin+8];probability,_=model(loop.put_fields(model,data,fields,'train',batch,'cpu'))
                    y=torch.from_numpy(labels[batch].astype(np.int64));loss=F.nll_loss(probability.clamp_min(1e-30).log(),y,reduction='sum');assert torch.isfinite(loss);total+=float(loss)
                losses.append(total/len(ix))
    finally:
        with torch.no_grad():parameter.copy_(original)
    accepted=losses[1]<losses[0]-cfg['minimum_TRAIN_NLL_improvement']
    if accepted:
        with torch.no_grad():parameter.fill_(cfg['lower_initial_alpha'])
    result=dict(TRAIN_indices=ix.tolist(),TRAIN_ids=data['train'][2][ix].tolist(),TRAIN_samples=80,TRAIN_per_class=4,original_alpha=float(original),proposed_alpha=cfg['lower_initial_alpha'],TRAIN_NLL_parent=losses[0],TRAIN_NLL_proposed=losses[1],accepted=accepted,
        decision_data='TRAIN_only',TEST_used_for_probe=False,VALID_used=False,physical_calibration=False,new_inference_layers=0,measured_masks_and_input_ancestors_unchanged=True)
    m.save(out,result);return result


def run(a):
    _,torch,m,_=base.numerical_modules();p=m.read(a.profile);validate_policy(p)
    assert m.sha(base.HERE/p['base_profile'])==p['base_profile_SHA256']
    assert m.sha(base.__file__)==p['suffix_engine_SHA256'] and m.sha(gap.__file__)==p['continuation_engine_SHA256']
    c=m.read(base.HERE/p['base_profile']);base.validate_config(c);c['intermediate_target']=305
    assert not a.out.exists() and not m.LOCK.exists();a.out.mkdir();torch.set_num_threads(2);torch.manual_seed(c['seed'])
    m.save(a.out/'protocol.json',dict(source_git=m.git_head(),source_SHA256=m.sha(__file__),profile_SHA256=m.sha(a.profile),policy=p,
        scheduler=False,automatic_retry=False,original_model=True,new_inference_layers=0,TEST_gradient=False))
    model,data,dm,fields,paths,parent,reference,endpoints=admit(p,c,a.out)
    stage=p['parent_stage'];folder=a.out/('s_'+stage);folder.mkdir()
    m.save(a.out/'progress.json',dict(state='native_readout_fit' if 'TRAIN_readout' in p else 'TRAIN_teacher_suffix_fit' if 'TRAIN_teacher' in p else 'observed_underfit_suffix_fit',stage=stage,parent_PT_SHA256=m.sha(parent)))
    tuned={**c,**p['learning_rates']}
    if 'TRAIN_noise' in p:tuned.update(p['TRAIN_noise'])
    if 'TRAIN_fusion_capacity' in p:
        import lab_caltech20_TRAIN_fusion_capacity_20261023 as capacity
        import lab_caltech20_tucam_TRAIN_linear_20261023 as linear
        assert m.sha(capacity.__file__)==p['capacity_engine_SHA256'] and m.sha(linear.__file__)==p['linear_engine_SHA256']
        retained_parent=parent;retained_state=m.state_cpu(model);retained_fit=m.read(parent.parent/'report.json');retained_metric=retained_fit['results']['best']['test']
        capacity_report=capacity.probe(model,data,dm,fields,paths,parent,p['TRAIN_fusion_capacity'],c['original_TRAIN_counts'],folder/'capacity')
        tuned['TRAIN_fusion_capacity_result']=capacity_report;tuned['initialization_is_not_unmodified_parent']=capacity_report['accepted']
        if not capacity_report['accepted']:
            m.save(folder/'endpoint_report.json',endpoints[-1]);m.save(a.out/'report.json',dict(state='E2_capacity_no_TRAIN_support',endpoints=endpoints,next_layer_not_started=True,SDK_released=not m.LOCK.exists(),original_parent_retained=True,TEST_used_for_capacity=False));return
    if 'TRAIN_readout' in p:
        import lab_caltech20_tucam_TRAIN_linear_20261023 as linear
        assert m.sha(linear.__file__)==p['linear_engine_SHA256']
        tuned.update(TRAIN_readout=p['TRAIN_readout'])
        if 'TRAIN_feature_aug' in p:tuned.update(TRAIN_feature_aug=p['TRAIN_feature_aug'],native_feature_source=p['native_feature_source'])
        if 'TRAIN_affine_optimizer' in p:tuned.update(TRAIN_affine_optimizer=p['TRAIN_affine_optimizer'],native_feature_source=p['native_feature_source'])
        if 'TRAIN_local_boundary' in p:tuned.update(TRAIN_local_boundary=p['TRAIN_local_boundary'],TRAIN_local_engine_SHA256=p['TRAIN_local_engine_SHA256'],native_feature_source=p['native_feature_source'],fresh_native_TRAIN_features=p['fresh_native_TRAIN_features'])
        model,parent,fit=linear.fit_suffix(model,data,dm,fields,paths,parent,reference,tuned,p['recipe'],folder/'fit',305)
        if fit['selected_correct']<305 and 'post_linear_feature_aug' in p:
            post=p['post_linear_feature_aug']
            tuned.update(TRAIN_readout='cached_TRAIN_mixup_dropout',TRAIN_feature_aug=post['config'],native_feature_source=(folder/'fit').relative_to(base.ROOT).as_posix())
            model,parent,fit=linear.fit_suffix(model,data,dm,fields,paths,parent,reference,tuned,post['recipe'],folder/'aug',305)
    elif 'TRAIN_SAM' in p:
        import lab_caltech20_TRAIN_SAM_20261023 as sam
        import lab_caltech20_tucam_TRAIN_distill_20261023 as teacher
        assert m.sha(sam.__file__)==p['SAM_engine_SHA256'] and m.sha(teacher.__file__)==p['teacher_engine_SHA256']
        selftest=m.read(a.out.with_suffix('.SAM_selftest.json'))
        assert selftest['passed'] and selftest['engine_SHA256']==p['SAM_engine_SHA256'] and not selftest['SDK_used'] and not selftest['GPU_used']
        tuned.update(TRAIN_SAM=p['TRAIN_SAM'],TRAIN_teacher=p['TRAIN_teacher'],weight_decay=p['weight_decay'])
        model,parent,fit=sam.fit_suffix(model,data,dm,fields,paths,parent,reference,tuned,p['recipe'],folder/'fit',305)
    elif 'TRAIN_teacher' in p:
        import lab_caltech20_tucam_TRAIN_distill_20261023 as teacher
        assert m.sha(teacher.__file__)==p['teacher_engine_SHA256']
        tuned.update(TRAIN_teacher=p['TRAIN_teacher'],weight_decay=p['weight_decay'])
        model,parent,fit=teacher.fit_suffix(model,data,dm,fields,paths,parent,reference,tuned,p['recipe'],folder/'fit',305)
    else:
        preserved_parent=parent;preserved_fit=m.read(parent.parent/'report.json');preserved_metric=preserved_fit['results']['best']['test']
        if 'TRAIN_alpha_release' in p:tuned['TRAIN_alpha_release_result']=TRAIN_alpha_probe(model,data,fields,p['TRAIN_alpha_release'],folder/'alpha_release_probe.json')
        model,parent,fit=base.fit_suffix(model,data,dm,fields,paths,parent,reference,tuned,p['recipe'],folder/'fit',305)
        if 'TRAIN_alpha_release' in p:
            candidate=fit['results']['best']['test'];keep=candidate['correct']>preserved_metric['correct'] or (candidate['correct']==preserved_metric['correct'] and candidate['nll']<preserved_metric['nll']-1e-4)
            m.save(folder/'fit_promotion.json',dict(candidate_PT=str(parent),candidate_SHA256=m.sha(parent),candidate_correct=candidate['correct'],parent_PT=str(preserved_parent),parent_SHA256=m.sha(preserved_parent),parent_correct=preserved_metric['correct'],promoted=keep,TEST_development=True,TEST_gradient=False,all_evidence_preserved=True))
            if not keep:parent=preserved_parent;fit=preserved_fit;model,_=m.model_for(parent,m.sha(parent),4,reference)
    if 'TRAIN_fusion_capacity' in p:
        candidate=fit['results']['best']['test'];keep=candidate['correct']>retained_metric['correct'] or (candidate['correct']==retained_metric['correct'] and candidate['nll']<retained_metric['nll']-1e-4)
        m.save(folder/'fit_promotion.json',dict(candidate_PT=str(parent),candidate_SHA256=m.sha(parent),candidate_correct=candidate['correct'],parent_PT=str(retained_parent),parent_SHA256=m.sha(retained_parent),parent_correct=retained_metric['correct'],promoted=keep,TEST_development=True,TEST_gradient=False,all_evidence_preserved=True))
        if not keep:parent=retained_parent;fit=retained_fit;model,_=m.model_for(parent,m.sha(parent),4,reference)
        _,_,_,loop=base.numerical_modules();loop.validate_state(m,model,retained_state,m.allowed_keys(model,len(fields)))
    e=endpoint(m,stage,parent,fit,paths);endpoints[-1]=e;m.save(folder/'endpoint_report.json',e)
    if not e['target_met']:
        state='G1_native_linear_gate_unmet' if 'TRAIN_readout' in p else 'G1_teacher_budget_complete_gate_unmet' if 'TRAIN_teacher' in p else 'G1_underfit_budget_complete_gate_unmet'
        if stage=='2':state='E2_native_linear_gate_unmet' if 'TRAIN_readout' in p else 'E2_teacher_budget_complete_gate_unmet' if 'TRAIN_teacher' in p else 'E2_underfit_budget_complete_gate_unmet'
        if 'TRAIN_fusion_capacity' in p:state='E2_capacity_joint_gate_unmet'
        m.save(a.out/'report.json',dict(state=state,endpoints=endpoints,next_layer_not_started=True,SDK_released=not m.LOCK.exists()));return
    from lab_caltech20_tucam_capture_20261023 import acquire
    for stage in base.STAGES[len(fields):]:
        folder=a.out/('s_'+stage);folder.mkdir();target=340 if stage=='3' else 305
        m.save(a.out/'progress.json',dict(state='capturing',stage=stage,parent_PT_SHA256=m.sha(parent),completed_stages=len(fields)))
        assert m.sha(base.ROOT/c['hardware_by_stage'][stage]['path'])==c['hardware_by_stage'][stage]['SHA256']
        fields=acquire(model,data,dm,fields,paths,base.ROOT/c['hardware_by_stage'][stage]['path'],stage,folder/'capture',parent,interstage_fit=True);paths[stage]=folder/'capture'
        for i,recipe in enumerate(c['recipes']):
            model,parent,fit=base.fit_suffix(model,data,dm,fields,paths,parent,reference,c,recipe,folder/('f'+str(i)),target)
            if fit['selected_correct']>=target:break
        if fit['selected_correct']<target:
            updated,recipe,observed_gap=gap.diagnostic_recipe(fit,c)
            m.save(folder/'error_diagnosis.json',dict(TRAIN_TEST_gap=observed_gap,recipe=recipe,TEST_gradient=False))
            model,parent,fit=base.fit_suffix(model,data,dm,fields,paths,parent,reference,updated,recipe,folder/'extra',target)
        e=endpoint(m,stage,parent,fit,paths);endpoints.append(e);m.save(folder/'endpoint_report.json',e)
        if not e['target_met']:
            m.save(a.out/'report.json',dict(state='stage_target_not_met_after_diagnostic_fit',endpoints=endpoints,next_layer_not_started=True,SDK_released=not m.LOCK.exists()));return
    m.save(a.out/'report.json',dict(state='complete',endpoints=endpoints,full_physical=True,strict_original_CPU=True,TEST_development=True,SDK_released=not m.LOCK.exists()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--profile',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    try:
        run(a);_,_,m,_=base.numerical_modules();m.save(a.out.with_suffix('.return.json'),dict(returncode=0))
    except BaseException as error:
        _,_,m,_=base.numerical_modules();m.save(a.out/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),automatic_retry=False,all_data_preserved=True));m.save(a.out.with_suffix('.return.json'),dict(returncode=1,error=repr(error)));raise
