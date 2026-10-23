import ast,hashlib,json
from pathlib import Path
ENTRY=Path(__file__).resolve().parents[1]/'lab_caltech20_actual_suffix_continue_20261024.py'


def test_initial_OAS_profile_and_source_identities():
    node=next(n for n in ast.parse(ENTRY.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='validate_policy');ns={};exec(compile(ast.Module(body=[node],type_ignores=[]),str(ENTRY),'exec'),ns)
    p=json.loads((ENTRY.parent/'lab_caltech20_actual300_OAS_profile_20261024.json').read_text());ns['validate_policy'](p)
    assert p['method']=='OAS' and p['recipe']['epochs']==1
    for name,digest in p['source_SHA256'].items():assert hashlib.sha256((ENTRY.parent/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==digest


def test_fresh_E2_only_and_immutable_measured_upstream():
    s=ENTRY.read_text();ast.parse(s)
    assert "cap==run/'s_2/capture'" in s and "r['phase_SHA256']==p['E2_phase_SHA256']" in s
    assert "loop.validate_state(m,model,m.state_cpu(capture_model),m.allowed_keys(model,4))" in s
    assert "m.sha(f)==r['arrays'][role]['SHA256']" in s and "metric['correct']==p['expected_parent_correct']" in s
    assert "m.sha(source/'report.json')==sa['report_SHA256']==p['source_report_SHA256']" in s


def test_existing_finite_optimizers_only_and_true_gate_before_G2():
    s=ENTRY.read_text();assert "('OAS','affine_LBFGS','native_CE')" in s
    assert 'engine.fit_suffix(' in s and 'TRAIN_affine_optimizer=dict(l2_parent_TRAIN_NLL_fraction=.1,label_smoothing=.03,outer_steps=80)' in s
    assert s.index('assert admission.can_start_G2(result,300)')<s.index('fields=capture.acquire(')
    assert "if not admission.can_start_G2(result,300):" in s and 'G2_not_started=True' in s
    assert "==5150" in s and 'loss.backward' not in s
