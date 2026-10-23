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
    for text in ("m.read(old.with_suffix('.return.json'))['returncode']==0", "audit['matched_samples']==2011", "audit['report_SHA256']==m.sha(cap/'report.json')", "base.identities(m,paths)==e['prefix_SHA256']", "m.allowed_keys(model,len(fields))", "assert score['correct']==e['selected_correct']==fit['selected_correct']", "assert fit['results']['best']['train']['accuracy']<p['maximum_parent_TRAIN_accuracy']"):
        assert text in s
    assert 'loop.validate_state(m,model,m.state_cpu(origin)' in s


def test_gate_and_original_fitter_precede_new_capture():
    s=ENTRY.read_text().split('def run(a):',1)[1]
    assert s.index('base.fit_suffix(')<s.index("if not e['target_met']:")<s.index('fields=acquire(')
    assert 'next_layer_not_started=True' in s and 'interstage_fit=True' in s
    assert 'base.STAGES[len(fields):]' in s and 'scheduler=False' in s
    assert 'set_exposure(' not in s and 'PhysicalAdjoint(' not in s
    assert 'parent,reference,c,recipe,' in s
