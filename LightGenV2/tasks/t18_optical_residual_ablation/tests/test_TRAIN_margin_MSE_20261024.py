import ast,json
from pathlib import Path
HERE=Path(__file__).resolve().parents[1]


def test_original_joint_suffix_only_and_TRAIN_teacher_gate():
    s=(HERE/'lab_caltech20_TRAIN_frozen_readout_20261024.py').read_text();tree=ast.parse(s)
    node=next(v for v in tree.body if isinstance(v,ast.FunctionDef) and v.name=='normalized_teacher_MSE');body=ast.get_source_segment(s,node)
    assert 'teacher.detach()' in body and 'gate=t.argmax(1)==labels' in body and 'test' not in body.lower()
    assert "TRAIN_teacher_gradient_fraction_max=.5" in s and '.500001' in s and 'or logit_mse' in s
    p=json.loads((HERE/'lab_caltech20_actual300_MSE285_profile_20261024.json').read_text());assert p['TRAIN_teacher_margin_MSE'] and p['expected_parent_correct']==285 and p['recipe']['epochs']==32


def test_future_local_repair_rebuilds_changed_suffix_features():
    s=(HERE/'lab_caltech20_actual_suffix_continue_20261024.py').read_text()
    local=next(v for v in s.splitlines() if "if p['method']=='local_remaining':" in v)
    assert "fresh_native_TRAIN_features=p.get('fresh_native_TRAIN_features',False)" in local
    s=(HERE/'lab_caltech20_tucam_TRAIN_linear_20261023.py').read_text()
    assert 'train_error)<=3e-6' in s and "assert np.array_equal(z['ids'],data['train'][2])" in s
    p=json.loads((HERE/'lab_caltech20_actual300_LOCALMSE_profile_20261024.json').read_text());assert p['fresh_native_TRAIN_features'] is True and p['expected_parent_correct']==295
    assert len(p['source_PT_SHA256'])==len(p['source_report_SHA256'])==64 and p['TRAIN_local_boundary']['maximum_classes']==20
