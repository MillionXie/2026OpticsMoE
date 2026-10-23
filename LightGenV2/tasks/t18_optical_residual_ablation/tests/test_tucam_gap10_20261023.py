import ast,hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parents[1]
ENTRY=HERE/'lab_caltech20_tucam_gap10_20261023.py'

def test_policy_identity_integer_targets():
    c=json.loads((HERE/'lab_caltech20_tucam_gap10_profile_20261023.json').read_text())
    # Git working-tree CRLF differs from Git blob: public SHA is normalized LF.
    raw=(HERE/c['base_profile']).read_bytes().replace(b'\r\n',b'\n')
    assert hashlib.sha256(raw).hexdigest()==c['base_profile_SHA256']
    assert c['intermediate_target']==305 and c['final_target']==340
    assert 305/411>=346/411-.1 and 304/411<346/411-.1

def test_preserve_existing_controller_and_no_retries():
    s=ENTRY.read_text();ast.parse(s)
    assert "while not old.with_suffix('.return.json').exists():" in s
    assert "if ret['returncode']!=0:" in s and 'predecessor_failed_needs_diagnosis' in s
    assert 'terminate(' not in s and 'kill(' not in s and 'set_exposure(' not in s
    assert 'interstage_fit=True' in s and 'scheduler=False' in s

def test_only_exact_audited_prefix_reused():
    s=ENTRY.read_text()
    assert "audit['matched_samples']==2011" in s and "assert m.sha(p)==q['arrays'][role]['SHA256']" in s
    assert "base.identities(m,paths)==e['prefix_SHA256']" in s
    assert "assert score['correct']==e['selected_correct']" in s
    assert "assert fit['strict_original_CPU']" in s

def test_every_new_stage_fits_before_endpoint_or_next_capture():
    s=ENTRY.read_text();tree=ast.parse(s);f=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
    loop=next(n for n in f.body if isinstance(n,ast.For));code=ast.get_source_segment(s,loop)
    assert code.index('fields=acquire(')<code.index('base.fit_suffix(')<code.index("'endpoint_report.json'")
    assert 'next_layer_not_started=True' in code
    assert "for stage in base.STAGES[len(fields):]:" in code

def test_diagnostic_branch_uses_actual_gap_not_random_search():
    import copy
    tree=ast.parse(ENTRY.read_text());f=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='diagnostic_recipe')
    ns={'copy':copy};exec(compile(ast.Module(body=[f],type_ignores=[]),str(ENTRY),'exec'),ns)
    parent={'phase_lr':.00005,'head_lr':.0005}
    for test_acc,expected in ((.6,'overfit_lowerLR_regularized'),(.9,'TRAIN_fit_focused_suffix')):
        r={'results':{'last':{'train':{'accuracy':.95},'test':{'accuracy':test_acc}}}}
        cfg,recipe,gap=ns['diagnostic_recipe'](r,parent)
        assert recipe['name']==expected and cfg['phase_lr']==parent['phase_lr'] and recipe['seconds']<=900
    assert parent=={'phase_lr':.00005,'head_lr':.0005}
