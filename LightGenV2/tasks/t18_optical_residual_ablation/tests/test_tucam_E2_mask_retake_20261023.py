import ast,json,hashlib
from pathlib import Path
import pytest

HERE=Path(__file__).resolve().parents[1]
ENTRY=HERE/'lab_caltech20_tucam_E2_mask_retake_20261023.py'


def validate(p):
    tree=ast.parse(ENTRY.read_text());f=next(v for v in tree.body if isinstance(v,ast.FunctionDef) and v.name=='validate_policy');ns={}
    exec(compile(ast.Module(body=[f],type_ignores=[]),str(ENTRY),'exec'),ns);ns['validate_policy'](p)


def test_human_choice_exact_prefix_and_published_engines():
    p=json.loads((HERE/'lab_caltech20_tucam_E2_mask_retake_profile_20261023.json').read_text());validate(p)
    for key,name in [('base_profile_SHA256',p['base_profile']),('suffix_engine_SHA256','lab_caltech20_tucam_layerwise_20261023.py'),('teacher_engine_SHA256','lab_caltech20_tucam_TRAIN_distill_20261023.py'),('capture_engine_SHA256','lab_caltech20_tucam_capture_20261023.py')]:
        assert hashlib.sha256((HERE/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==p[key]
    assert list(p['reuse_captures'])==['router','0','1'] and p['mask_training_stop_target']==411
    for key,value in [('intermediate_target',282),('learning_rates',{**p['learning_rates'],'head_lr':.0001}),('TEST_gradient',True)]:
        with pytest.raises(AssertionError):validate({**p,key:value})


def test_no_old_E2_training_and_real_changed_mask_before_retake():
    s=ENTRY.read_text();run=s.split('def run(a):',1)[1]
    assert "for stage,name in p['reuse_captures'].items()" in s and "old_E2_used_for_fit=False" in s
    assert "score['correct']==305" in s and "head_bitwise_frozen=frozen_head" in s
    assert "not fit['selected_E0'] and bool(changed) and new_phase!=old_phase" in s
    assert run.index("if not deployable:")<run.index('fields=acquire(')
    assert "m.allowed_keys(model,3)-{'linear.weight','linear.bias'}" in s
    assert "shutil.disk_usage(base.ROOT).free>=c['minimum_free_GiB']*1024**3" in s
    assert "assert m.read(paths[stage]/'report.json')['phase_SHA256']==new_phase" in s
    assert "for stage in ('2','3')" in s and "if not e['target_met']:" in s
    assert 'unlink(' not in s and 'set_exposure(' not in s and 'resize(' not in s


def test_phase_probe_is_TRAIN_only_native_encoding_without_SDK():
    s=ENTRY.read_text().split('def phase_probe(',1)[1].split('\ndef admit(',1)[0]
    assert "data['train']" in s and "data['test']" not in s and "model.measured={}" in s
    assert 'm.phase_native(pixels)' in s and 'Bench(' not in s


def test_CPU_readout_freeze_and_gradient_selftest_is_required():
    s=ENTRY.read_text()
    assert "proof['source_SHA256']==m.sha(__file__)" in s and "a.out.with_suffix('.mask_selftest.json')" in s
    assert 'head_zero_LR_bitwise_frozen=True' in s and 'gradient_through_frozen_head_to_features=True' in s
    assert "numpy().view(np.uint8)" in s and 'lr=0.,weight_decay=.01' in s


def test_joint_recovery_needs_TRAIN_supported_actual_mask_not_old_E2():
    p=json.loads((HERE/'lab_caltech20_tucam_E2_mask_joint_recovery_profile_20261023.json').read_text());validate(p)
    assert p['candidate_checkpoint']=='last' and p['candidate_TRAIN_support']['NLL_improvement']==.0001
    assert p['learning_rates']['head_lr']>0 and p['mask_training_stop_target']==305
    s=ENTRY.read_text().split('def TRAIN_candidate(',1)[1].split('\ndef run(',1)[0]
    assert "r['results']['last']['train']" in s and "r['results']['last']['test']" not in s
    assert 'TEST_used_for_candidate_admission=False' in s and "m.allowed_keys(candidate,3)-{'linear.weight','linear.bias'}" in s
    with pytest.raises(AssertionError):validate({**p,'candidate_checkpoint':'best'})


def test_boundary_recovery_fresh_current_mask_and_no_TEST_gradient(monkeypatch):
    monkeypatch.syspath_prepend(str(HERE))
    p=json.loads((HERE/'lab_caltech20_tucam_E2_mask_boundary_recovery_profile_20261023.json').read_text());validate(p)
    assert p['candidate_checkpoint']=='best' and p['completed_controller'].endswith('/e2q')
    assert p['fresh_native_TRAIN_features'] and p['native_feature_source']==p['candidate_mask_fit']
    assert p['learning_rates']=={} and p['mask_recipe']['epochs']==0 and p['TRAIN_local_boundary']['maximum_classes']==8
    s=ENTRY.read_text();assert 'source_checkpoint_selected_on_TEST_development=boundary' in s and 'fitter=linear.fit_suffix' in s
    with pytest.raises(AssertionError):validate({**p,'fresh_native_TRAIN_features':False})


def test_remaining_boundary_exact_plan_and_unchanged_native_features(monkeypatch):
    monkeypatch.syspath_prepend(str(HERE))
    p=json.loads((HERE/'lab_caltech20_tucam_E2_mask_remaining_profile_20261023.json').read_text());validate(p)
    assert p['completed_controller'].endswith('/e2g') and p['candidate_mask_fit'].endswith('/e2g/mask_fit')
    cfg=p['TRAIN_local_boundary'];assert cfg['maximum_classes']+len(cfg['skip_classes'])==20
    assert cfg['skip_classes']==[3,2,0,19,4,7,10,13] and p['expected_candidate_correct']==304
    assert p['parent_local_plan_SHA256']=='dd8390371550229009b21e343048381dd9c76d6d98add2568c26b4ddfe8fbb12'
    kernel=HERE/'lab_caltech20_TRAIN_local_boundary_20261023.py'
    assert hashlib.sha256(kernel.read_bytes().replace(b'\r\n',b'\n')).hexdigest()==p['TRAIN_local_engine_SHA256']
    s=ENTRY.read_text();assert "m.sha(plan)==p['parent_local_plan_SHA256']" in s and "m.read(plan)]==p['TRAIN_local_boundary']['skip_classes']" in s
    for key,value in [('expected_candidate_correct',305),('fresh_native_TRAIN_features',True),('TRAIN_local_boundary',{**cfg,'skip_classes':[3,2,0,4,9,15,7,12]})]:
        with pytest.raises(AssertionError):validate({**p,key:value})
