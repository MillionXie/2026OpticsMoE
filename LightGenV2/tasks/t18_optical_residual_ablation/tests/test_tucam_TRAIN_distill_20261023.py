import ast,hashlib,json
from pathlib import Path
import pytest

HERE=Path(__file__).resolve().parents[1]
ENTRY=HERE/'lab_caltech20_tucam_TRAIN_distill_20261023.py'


def test_teacher_cache_uses_only_TRAIN_original_teacher():
    s=ENTRY.read_text();tree=ast.parse(s);f=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='TRAIN_teacher');body=ast.get_source_segment(s,f)
    assert "m.padded(data,'train',begin)" in body and "data['test']" not in body
    assert "teacher.measured={}" in body and 'requires_grad_(False)' in body
    assert "source_PT_SHA256=c['initial_SHA256']" in body
    assert "teacher_logits.detach()" in s and 'inference_teacher=False' in s


def test_native_review_and_no_measured_mask_gradient():
    s=ENTRY.read_text()
    assert 'keys=m.allowed_keys(model,len(fields))' in s
    assert 'stage=measured_stage(fields)' in s and 'PhysicalAdjoint(' not in s
    assert 'TRAINTransforms(model,stage,c)' in s
    assert 'loop.validate_state(m,native,original,keys)' in s
    assert "for role in ('train','test'):" in s and 'base.identities(m,paths)==lineage' in s
    assert 'SDK_used=False' in s and 'new_inference_layers=0' in s
    assert 'model_for(out/\'best.pt\'' in s and 'transforms.remove();hook.remove()' in s


def test_contiguous_measured_prefix_selects_actual_last_CCD():
    tree=ast.parse(ENTRY.read_text());f=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='measured_stage')
    from types import SimpleNamespace
    ns={'base':SimpleNamespace(STAGES=('router','0','1','2','3'))}
    exec(compile(ast.Module(body=[f],type_ignores=[]),str(ENTRY),'exec'),ns)
    for stages,last in ((['router','0'],'0'),(['router','0','1'],'1'),(['router','0','1','2'],'2')):
        assert ns['measured_stage'](dict.fromkeys(stages))==last
    for stages in (['router'],['router','1'],['router','0','2'],['router','0','1','2','3']):
        with pytest.raises(AssertionError):ns['measured_stage'](dict.fromkeys(stages))


def test_registered_finite_teacher_profile():
    p=json.loads((HERE/'lab_caltech20_tucam_E1_distill_profile_20261023.json').read_text())
    assert p['TRAIN_teacher']=={'temperature':2,'mixture':.3}
    assert p['minimum_parent_TRAIN_TEST_gap']==.1
    assert p['recipe']['epochs']==36 and p['recipe']['seconds']==1200
    assert p['intermediate_target']==305 and p['final_target']==340
    assert p['TEST_gradient'] is False and p['new_inference_layers']==0
    assert hashlib.sha256(ENTRY.read_bytes().replace(b'\r\n',b'\n')).hexdigest()==p['teacher_engine_SHA256']
    driver=HERE/'lab_caltech20_tucam_E1_underfit_20261023.py';tree=ast.parse(driver.read_text());f=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='validate_policy');ns={}
    exec(compile(ast.Module(body=[f],type_ignores=[]),str(driver),'exec'),ns);ns['validate_policy'](p)


def test_teacher_requires_completed_overfit_parent_and_native_gate():
    s=(HERE/'lab_caltech20_tucam_E1_underfit_20261023.py').read_text()
    assert "b['train']['accuracy']-b['test']['accuracy']>=policy['minimum_parent_TRAIN_TEST_gap']" in s
    assert "m.sha(teacher.__file__)==p['teacher_engine_SHA256']" in s
    assert s.index('teacher.fit_suffix(')<s.index("if not e['target_met']:")<s.index('fields=acquire(')
