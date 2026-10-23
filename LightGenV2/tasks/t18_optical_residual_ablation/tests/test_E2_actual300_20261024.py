import ast,hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parents[1]
ENTRY=HERE/'lab_caltech20_E2_actual300_20261024.py'


def pure(name):
    node=next(n for n in ast.parse(ENTRY.read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name)
    ns={};exec(compile(ast.Module(body=[node],type_ignores=[]),str(ENTRY),'exec'),ns);return ns[name]


def test_authorized_actual300_not_prediction_gate():
    p=json.loads((HERE/'lab_caltech20_E2_actual300_profile_20261024.json').read_text());pure('validate_policy')(p)
    gate=pure('can_start_G2')
    for n in (269,273,299,300,305):
        assert gate(dict(strict_original_CPU=True,selected_correct=n,results={'best':{'test':{'correct':n}}}),300)==(n>=300)
    assert not gate(dict(strict_original_CPU=False,selected_correct=411,results={'best':{'test':{'correct':411}}}),300)
    for name,digest in p['source_SHA256'].items():assert hashlib.sha256((HERE/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==digest


def test_original_graph_prefix_and_actual_before_G2():
    s=ENTRY.read_text();assert 'predictive_accuracy_prerequisite=False' in s
    assert "assert metric['correct']==269" in s and "assert metric['correct']>=300" not in s
    assert s.index('assert can_start_G2(result,300)')<s.index('fields=capture.acquire(')
    assert "m.allowed_keys(model,3)" in s and "{'linear.weight','linear.bias'}" in s
    assert "assert changed and new_phase!=old_phase" in s and "list(fields)==['router','0','1']" in s
    assert "m.sha(hardware)==c['hardware_by_stage'][stage]['SHA256']" in s
    assert 'base.fit_suffix(' in s and 'response_map_not_deployed=True' in s
    assert 'loss.backward' not in s and 'automatic_retry=False' in s
