import ast,hashlib,json
from pathlib import Path

HERE=Path(__file__).resolve().parents[1]
ENTRY=HERE/'lab_caltech20_tucam_E1_underfit_20261023.py'


def test_policy_and_immutable_engines():
    p=json.loads((HERE/'lab_caltech20_tucam_E1_underfit_profile_20261023.json').read_text())
    for key,name in [('base_profile_SHA256',p['base_profile']),('suffix_engine_SHA256','lab_caltech20_tucam_layerwise_20261023.py'),('continuation_engine_SHA256','lab_caltech20_tucam_gap10_20261023.py')]:
        assert hashlib.sha256((HERE/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==p[key]
    tree=ast.parse(ENTRY.read_text());f=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='validate_policy');ns={}
    exec(compile(ast.Module(body=[f],type_ignores=[]),str(ENTRY),'exec'),ns);ns['validate_policy'](p)
    assert p['intermediate_target']==305 and p['final_target']==340


def test_exact_audited_completed_prefix_and_underfit_required():
    s=ENTRY.read_text()
    for literal in ("audit['matched_samples']==2011","audit['report_SHA256']==m.sha(cap/'report.json')","base.identities(m,paths)==e['prefix_SHA256']","assert score['correct']==e['selected_correct']==fit['selected_correct']","fit['results']['best']['train']['accuracy']<policy['maximum_parent_TRAIN_accuracy']"):
        assert literal in s
    assert "m.read(old.with_suffix('.return.json'))['returncode']==0" in s
    assert "assert not a.out.exists() and not m.LOCK.exists()" in s


def test_original_fitter_and_strict_gate_precede_capture():
    s=ENTRY.read_text();start=s.index('def run(a):');s=s[start:]
    assert s.index('base.fit_suffix(')<s.index("if not e['target_met']:")<s.index('fields=acquire(')
    assert 'next_layer_not_started=True' in s and 'interstage_fit=True' in s
    assert 'base.STAGES[len(fields):]' in s
    assert 'scheduler=False' in s and 'automatic_retry=False' in s
    assert 'set_exposure(' not in s and 'PhysicalAdjoint(' not in s
    assert "tuned={**c,**p['learning_rates']}" in s
    assert 'parent,reference,c,recipe,' in s
