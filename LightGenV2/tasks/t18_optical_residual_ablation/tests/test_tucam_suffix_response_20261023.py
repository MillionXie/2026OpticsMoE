import ast,hashlib,json
from pathlib import Path
import pytest

HERE=Path(__file__).resolve().parents[1]
ENTRY=HERE/'lab_caltech20_tucam_suffix_response_20261023.py'


def policy_validator():
    tree=ast.parse(ENTRY.read_text());f=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='validate_policy');ns={}
    exec(compile(ast.Module(body=[f],type_ignores=[]),str(ENTRY),'exec'),ns)
    return ns['validate_policy']


def test_registered_policy_and_immutable_active_source():
    p=json.loads((HERE/'lab_caltech20_tucam_G1_underfit_profile_20261023.json').read_text());policy_validator()(p)
    for key,name in [('base_profile_SHA256',p['base_profile']),('suffix_engine_SHA256','lab_caltech20_tucam_layerwise_20261023.py'),('continuation_engine_SHA256','lab_caltech20_tucam_gap10_20261023.py')]:
        assert hashlib.sha256((HERE/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==p[key]
    for key,value in [('TEST_gradient',True),('maximum_parent_TRAIN_accuracy',1.),('intermediate_target',304),('parent_stage','0')]:
        with pytest.raises(AssertionError):policy_validator()({**p,key:value})


def test_mixed_capture_roots_require_all_audits_and_exact_parent():
    s=ENTRY.read_text()
    for text in ("m.read(old.with_suffix('.return.json'))['returncode']==0", "audit['matched_samples']==2011", "audit['report_SHA256']==m.sha(cap/'report.json')", "base.identities(m,paths)==e['prefix_SHA256']", "m.allowed_keys(model,len(fields))", "assert score['correct']==e['selected_correct']==fit['selected_correct']", "assert fit['results']['best']['train']['accuracy']<p['maximum_parent_TRAIN_accuracy']", "q['phase_SHA256']==m.sha(cap/'phase.bmp')"):
        assert text in s
    assert 'loop.validate_state(m,model,m.state_cpu(origin)' in s


def test_gate_and_original_fitter_precede_new_capture():
    s=ENTRY.read_text().split('def run(a):',1)[1]
    assert s.index('base.fit_suffix(')<s.index("if not e['target_met']:")<s.index('fields=acquire(')
    assert 'next_layer_not_started=True' in s and 'interstage_fit=True' in s
    assert 'base.STAGES[len(fields):]' in s and 'scheduler=False' in s
    assert 'set_exposure(' not in s and 'PhysicalAdjoint(' not in s
    assert 'parent,reference,c,recipe,' in s


def test_teacher_requires_observed_gap_and_exact_native_parent():
    p=json.loads((HERE/'lab_caltech20_tucam_G1_distill_profile_20261023.json').read_text());policy_validator()(p)
    teacher=HERE/'lab_caltech20_tucam_TRAIN_distill_20261023.py'
    assert hashlib.sha256(teacher.read_bytes().replace(b'\r\n',b'\n')).hexdigest()==p['teacher_engine_SHA256']
    s=ENTRY.read_text()
    assert "b['train']['accuracy']-b['test']['accuracy']>=p['minimum_parent_TRAIN_TEST_gap']" in s
    assert "assert e['selected_correct']==p['expected_parent_correct']" in s
    assert "m.sha(teacher.__file__)==p['teacher_engine_SHA256']" in s
    assert s.index('teacher.fit_suffix(')<s.index("if not e['target_met']:")<s.index('fields=acquire(')


def test_finite_native_linear_and_TRAIN_only_cached_refinement():
    p=json.loads((HERE/'lab_caltech20_tucam_G1_linear_profile_20261023.json').read_text());policy_validator()(p)
    assert hashlib.sha256((HERE/'lab_caltech20_tucam_TRAIN_linear_20261023.py').read_bytes().replace(b'\r\n',b'\n')).hexdigest()==p['linear_engine_SHA256']
    assert p['expected_parent_correct']==298 and p['post_linear_feature_aug']['recipe']['epochs']==48
    s=ENTRY.read_text();assert "if fit['selected_correct']<305 and 'post_linear_feature_aug' in p:" in s
    assert "native_feature_source=(folder/'fit').relative_to(base.ROOT).as_posix()" in s
    assert s.index('linear.fit_suffix(')<s.index("if not e['target_met']:")<s.index('fields=acquire(')


def test_exact_native_best_cooldown_not_last_or_new_features():
    p=json.loads((HERE/'lab_caltech20_tucam_G1_feature_cooldown_profile_20261023.json').read_text());policy_validator()(p)
    assert p['expected_parent_correct']==298 and p['TRAIN_feature_aug']['cooldown'] is True
    assert p['native_feature_source'].endswith('/g1l/s_1/fit')


def test_registered_conditioning_response_has_no_TEST_L2_choice():
    p=json.loads((HERE/'lab_caltech20_tucam_G1_fullbatch_profile_20261023.json').read_text());policy_validator()(p)
    assert p['expected_parent_correct']==299 and p['TRAIN_affine_optimizer']['l2_parent_TRAIN_NLL_fraction']==.1
    assert p['recipe']['epochs']==80 and p['recipe']['seconds']==60


def test_CCD_robustness_is_TRAIN_configuration_not_hardware_change():
    p=json.loads((HERE/'lab_caltech20_tucam_G1_CCDnoise_profile_20261023.json').read_text());policy_validator()(p)
    assert set(p['TRAIN_noise'])=={'CCD_gain_sigma','CCD_additive_sigma_DN','CCD_pixel_dropout_p'}
    assert p['expected_parent_correct']==299 and p['TRAIN_teacher']=={'temperature':2,'mixture':.3}
    assert "if 'TRAIN_noise' in p:tuned.update(p['TRAIN_noise'])" in ENTRY.read_text()


def test_E2_underfit_uses_four_exact_measured_stages_and_joint_suffix():
    p=json.loads((HERE/'lab_caltech20_tucam_E2_underfit_profile_20261023.json').read_text());policy_validator()(p)
    assert p['parent_stage']=='2' and list(p['captures'])==['router','0','1','2']
    assert p['maximum_parent_TRAIN_accuracy']==.8 and p['minimum_parent_correct']==205
    assert p['recipe']['epochs']==48 and p['recipe']['seconds']==1500
    assert p['learning_rates']['phase_lr']>0 and p['learning_rates']['electronic_lr']>0 and p['learning_rates']['alpha_lr']>0
    assert "paths[p['parent_stage']]" in ENTRY.read_text()
    for key,value in [('parent_stage','1'),('maximum_parent_TRAIN_accuracy',1.),('intermediate_target',304)]:
        with pytest.raises(AssertionError):policy_validator()({**p,key:value})
    assert p['new_CCD_before_gate']==0 and p['scheduler'] is False


def test_alpha_probe_uses_fixed_balanced_TRAIN_and_preserves_parent():
    s=ENTRY.read_text();tree=ast.parse(s);f=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='TRAIN_alpha_probe')
    names={n.id.lower() for n in ast.walk(f) if isinstance(n,ast.Name)}
    assert not names.intersection({'test','test_labels','development'})
    assert "np.flatnonzero(labels==k)[:cfg['per_class']]" in s and "parameter.copy_(original)" in s
    assert 'TEST_used_for_probe=False' in s and "decision_data='TRAIN_only'" in s
    assert 'if not keep:parent=preserved_parent;fit=preserved_fit' in s
    p=json.loads((HERE/'lab_caltech20_tucam_E2_underfit_profile_20261023.json').read_text())
    assert p['TRAIN_alpha_release']['lower_initial_alpha']==.7 and p['TRAIN_alpha_release']['samples']==80


def test_E2_linear_is_one_TRAIN_estimator_after_joint_fit():
    p=json.loads((HERE/'lab_caltech20_tucam_E2_linear_profile_20261023.json').read_text());policy_validator()(p)
    assert p['parent_stage']=='2' and p['TRAIN_readout']=='one_OAS_pooled_linear'
    assert p['completed_controller'].endswith('/e2u') and p['learning_rates']=={}
    assert p['minimum_parent_correct']==222 and p['maximum_parent_TRAIN_accuracy']==.95
    engine=HERE/'lab_caltech20_tucam_TRAIN_linear_20261023.py'
    assert hashlib.sha256(engine.read_bytes().replace(b'\r\n',b'\n')).hexdigest()==p['linear_engine_SHA256']
    for key,value in [('parent_stage','1'),('maximum_parent_TRAIN_accuracy',1.)]:
        with pytest.raises(AssertionError):policy_validator()({**p,key:value})


def test_E2_fullbatch_reuses_exact_native_cache_and_TRAIN_only_L2():
    p=json.loads((HERE/'lab_caltech20_tucam_E2_fullbatch_profile_20261023.json').read_text());policy_validator()(p)
    assert p['expected_parent_correct']==257 and p['minimum_parent_TRAIN_TEST_gap']==.03
    assert p['native_feature_source'].endswith('/e2l/s_2/fit') and p['recipe']['epochs']==80 and p['recipe']['seconds']==60
    assert p['TRAIN_affine_optimizer']['l2_parent_TRAIN_NLL_fraction']==.1 and p['learning_rates']=={}


def test_E2_teacher_requires_native_gap_and_joint_original_parameters():
    p=json.loads((HERE/'lab_caltech20_tucam_E2_distill_profile_20261023.json').read_text());policy_validator()(p)
    assert p['expected_parent_correct']==267 and p['minimum_parent_TRAIN_TEST_gap']==.1
    assert p['TRAIN_teacher']==dict(temperature=2,mixture=.3) and p['recipe']['epochs']==36
    assert p['learning_rates']['phase_lr']>0 and p['learning_rates']['electronic_lr']>0 and p['learning_rates']['alpha_lr']>0
    engine=HERE/'lab_caltech20_tucam_TRAIN_distill_20261023.py'
    assert hashlib.sha256(engine.read_bytes().replace(b'\r\n',b'\n')).hexdigest()==p['teacher_engine_SHA256']


def test_fusion_capacity_is_TRAIN_only_fresh_two_settings_and_parent_fallback(monkeypatch):
    monkeypatch.syspath_prepend(str(HERE))
    import lab_caltech20_TRAIN_fusion_capacity_20261023 as cap
    a=dict(TRAIN_capacity_correct=1000,TRAIN_capacity_NLL=1.2)
    assert cap.capacity_better(a,dict(TRAIN_capacity_correct=1100,TRAIN_capacity_NLL=1.1))
    assert not cap.capacity_better(a,dict(TRAIN_capacity_correct=999,TRAIN_capacity_NLL=1.))
    assert not cap.capacity_better(a,dict(TRAIN_capacity_correct=1100,TRAIN_capacity_NLL=1.3))
    p=json.loads((HERE/'lab_caltech20_tucam_E2_capacity_profile_20261023.json').read_text());policy_validator()(p)
    engine=HERE/'lab_caltech20_TRAIN_fusion_capacity_20261023.py'
    assert hashlib.sha256(engine.read_bytes().replace(b'\r\n',b'\n')).hexdigest()==p['capacity_engine_SHA256']
    s=engine.read_text();assert "data['train']" in s and "data['test']" not in s
    assert 'TEST_used_for_capacity=False' in s and 'capacity_not_generalization=True' in s
    assert "model.load_state_dict(original)" in s and "loop.validate_state(m,model,original,allowed)" in s
    driver=ENTRY.read_text();assert 'E2_capacity_no_TRAIN_support' in driver and 'if not keep:parent=retained_parent;fit=retained_fit' in driver


def test_E2_local_boundary_uses_retained274_and_fresh_native_features(monkeypatch):
    monkeypatch.syspath_prepend(str(HERE))
    p=json.loads((HERE/'lab_caltech20_tucam_E2_local_boundary_profile_20261023.json').read_text());policy_validator()(p)
    assert p['parent_stage']=='2' and p['expected_parent_correct']==274 and len(p['captures'])==4
    assert p['native_feature_source'].endswith('/e2k/s_2/fit') and p['fresh_native_TRAIN_features']
    assert p['TRAIN_local_boundary']['maximum_classes']==8 and p['recipe']['epochs']==0
    with pytest.raises(AssertionError):policy_validator()({**p,'fresh_native_TRAIN_features':False})


def test_new_E2_underfit_capacity_requires_new_data_and_completed_retake():
    p=json.loads((HERE/'lab_caltech20_tucam_E2_retake_capacity_profile_20261023.json').read_text());policy_validator()(p)
    assert p['fresh_E2_after_retake'] and p['minimum_parent_correct']==p['expected_parent_correct']==254 and p['maximum_parent_TRAIN_accuracy']==.95
    assert p['completed_controller'].endswith('/e2h') and p['captures']['2'].endswith('/e2h/s_2/capture')
    assert len(p['E2_arrays_SHA256'])==2 and p['TRAIN_fusion_capacity']['proposed_alpha']==.4001001
    s=ENTRY.read_text();assert "if 'TRAIN_teacher_conflict' in p else 'E2_mask_retake_stage_gate_unmet'" in s and "q['arrays'][role]['SHA256']==p['E2_arrays_SHA256'][role]" in s
    for key,value in [('expected_parent_SHA256','0'*64),('expected_parent_correct',228),('capture_parent_SHA256','8fa8544f54069e91e85dc6478519edc1b449bfbd9d17c47d0c48b32abe958e2b'),('captures',{**p['captures'],'2':p['captures']['2'].replace('/e2h/','/g1s/')}),('intermediate_target',304)]:
        with pytest.raises(AssertionError):policy_validator()({**p,key:value})
    for key,name in [('capacity_engine_SHA256','lab_caltech20_TRAIN_fusion_capacity_20261023.py'),('linear_engine_SHA256','lab_caltech20_tucam_TRAIN_linear_20261023.py')]:
        assert hashlib.sha256((HERE/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==p[key]


def test_E2_SAM_joint_parameters_TRAIN_only_same_noise_and_restore():
    p=json.loads((HERE/'lab_caltech20_tucam_E2_SAM_profile_20261023.json').read_text());policy_validator()(p)
    assert p['expected_parent_correct']==282 and p['learning_rates']['phase_lr']>0 and p['learning_rates']['electronic_lr']>0
    engine=HERE/'lab_caltech20_TRAIN_SAM_20261023.py';s=engine.read_text()
    assert hashlib.sha256(engine.read_bytes().replace(b'\r\n',b'\n')).hexdigest()==p['SAM_engine_SHA256']
    for text in ("loop.put_fields(model,data,fields,'train',ix,'cuda')",'torch.set_rng_state(before_cpu)','torch.cuda.set_rng_state(before_gpu)',"finally:\n                    restore(parameters,originals)",'passes==updates*2','original=m.state_cpu(model)','loop.validate_state(m,native,original,keys)'):
        assert text in s
    assert "loop.put_fields(model,data,fields,'test'" not in s and 'new_inference_layers=0' in s
    with pytest.raises(AssertionError):policy_validator()({**p,'TRAIN_SAM':dict(radius=.1,same_noise_both_passes=True)})


def test_fresh_E2_CE_refinement_requires_TRAIN_loss_evidence_and_exact_best():
    p=json.loads((HERE/'lab_caltech20_tucam_E2_retake_CE_profile_20261023.json').read_text());policy_validator()(p)
    assert p['expected_parent_correct']==270 and p['completed_controller'].endswith('/e2v') and 'TRAIN_teacher' not in p
    assert p['recipe']['epochs']==36 and p['recipe']['seconds']==900 and p['learning_rates']['head_lr']==.0005
    tree=ast.parse(ENTRY.read_text());f=next(v for v in tree.body if isinstance(v,ast.FunctionDef) and v.name=='TRAIN_loss_components');ns={}
    exec(compile(ast.Module(body=[f],type_ignores=[]),str(ENTRY),'exec'),ns)
    result=ns['TRAIN_loss_components'](dict(TRAIN_CE=1.0648694178462028,TRAIN_KL=2.9407007628679276),p['TRAIN_teacher_conflict'])
    assert result['ratio']>1 and result['decision_data']=='TRAIN_only' and not result['TEST_used_for_loss_diagnosis']
    with pytest.raises(AssertionError):ns['TRAIN_loss_components'](dict(TRAIN_CE=2.,TRAIN_KL=.1),p['TRAIN_teacher_conflict'])
    accesses=[v.slice.value for v in ast.walk(f) if isinstance(v,ast.Subscript) and isinstance(v.value,ast.Name) and v.value.id=='row' and isinstance(v.slice,ast.Constant)]
    assert set(accesses)=={'TRAIN_CE','TRAIN_KL'}
    s=ENTRY.read_text();assert "m.sha(history)==p['parent_TRAIN_history_SHA256']" in s and "e['PT_SHA256']==p['expected_parent_SHA256']" in s
    for key,value in [('TRAIN_teacher',dict(temperature=2,mixture=.3)),('expected_parent_correct',269),('intermediate_target',304)]:
        with pytest.raises(AssertionError):policy_validator()({**p,key:value})


def test_fresh_E2_local_is_single_finite_pass_and_retained_exact_model(monkeypatch):
    monkeypatch.syspath_prepend(str(HERE))
    p=json.loads((HERE/'lab_caltech20_tucam_E2_retake_local_profile_20261024.json').read_text());policy_validator()(p)
    assert p['expected_parent_correct']==270 and p['completed_controller'].endswith('/e2w') and p['fresh_native_TRAIN_features']
    assert p['native_feature_source'].endswith('/e2v/s_2/fit') and p['TRAIN_local_boundary']['maximum_classes']==20
    assert p['TRAIN_local_boundary']['maximum_classes']*p['TRAIN_local_boundary']['steps_per_class']==1280
    assert 'skip_classes' not in p['TRAIN_local_boundary'] and p['recipe']['epochs']==0
    engine=HERE/'lab_caltech20_TRAIN_local_boundary_20261023.py'
    assert hashlib.sha256(engine.read_bytes().replace(b'\r\n',b'\n')).hexdigest()==p['TRAIN_local_engine_SHA256']
    s=ENTRY.read_text();assert "loop.validate_state(m,model,m.state_cpu(same),set())" in s and "assert fit['selected_E0']" in s
    for key,value in [('fresh_native_TRAIN_features',False),('expected_parent_correct',269),('TRAIN_local_boundary',{**p['TRAIN_local_boundary'],'maximum_classes':19})]:
        with pytest.raises(AssertionError):policy_validator()({**p,key:value})


def test_E2_feature_regularization_reuses_only_same_nonhead_native_cache():
    p=json.loads((HERE/'lab_caltech20_tucam_E2_feature_cooldown_profile_20261023.json').read_text());policy_validator()(p)
    assert p['parent_stage']=='2' and p['expected_parent_correct']==282 and p['completed_controller'].endswith('/e2m')
    assert p['native_feature_source'].endswith('/e2r/s_2/fit') and p['TRAIN_feature_aug']['cooldown']
    assert p['learning_rates']=={} and p['TRAIN_readout']=='cached_TRAIN_mixup_dropout'
    with pytest.raises(AssertionError):policy_validator()({**p,'TRAIN_feature_aug':{**p['TRAIN_feature_aug'],'dropout':.2}})
