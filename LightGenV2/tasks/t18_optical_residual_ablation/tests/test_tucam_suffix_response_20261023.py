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
