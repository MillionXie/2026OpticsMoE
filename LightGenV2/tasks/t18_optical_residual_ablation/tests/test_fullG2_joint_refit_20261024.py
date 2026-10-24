import ast
from pathlib import Path


def test_full_physical_joint_source_requires_complete_audited_parent():
    s=(Path(__file__).resolve().parents[1]/'lab_caltech20_fullG2_OAS_20261024.py').read_text();ast.parse(s)
    assert "m.read(source.with_suffix('.return.json'))['returncode']==0" in s
    assert "'fullG2_joint_TRAIN_OAS_20261024'" in s
    assert "fit=source/'fit'" in s and "audit['report_SHA256']==m.sha(fit/'report.json')" in s
    assert "else{'linear.weight','linear.bias'}" in s and "TEST_used=False" in s
    assert "loop.validate_state(m,model,m.state_cpu(captured),m.allowed_keys(model,5))" in s
    assert "if not better:model.load_state_dict(original)" in s


def test_full_physical_local_repair_uses_only_TRAIN_neighbors():
    root=Path(__file__).resolve().parents[1];s=(root/'lab_caltech20_fullG2_OAS_20261024.py').read_text()
    assert "maximum_classes=20,steps_per_class=64" in s and "'fresh_native_TRAIN_features':True" in s
    assert "assert result['full_physical']" in s and "epochs=1,seconds=180" in s
    engine=(root/'lab_caltech20_tucam_TRAIN_linear_20261023.py').read_text()
    assert engine.count('full_physical=len(fields)==5')==2
    assert "canonical_affine_logits(test,model.linear.weight,model.linear.bias)" in engine
    boundary=(root/'lab_caltech20_TRAIN_local_boundary_20261023.py').read_text()
    assert 'logits=canonical_affine_logits(' in boundary
    assert "source=root.parent/'fullG2_teacher_margin_20261024'" in s and "'fullG2_teacher_local_20261024'" in s


def test_population_prior_bias_is_only_TRAIN_original20_and_single_fit():
    s=(Path(__file__).resolve().parents[1]/'lab_caltech20_fullG2_OAS_20261024.py').read_text();ast.parse(s)
    assert "keys={'linear.bias'}if a.prior_bias" in s and 'max_iter=50,max_eval=65' in s
    assert "weight=model.linear.weight.detach().numpy().copy()" in s and "prior_exponent=1." in s
    assert "parameters=20" in s and 'TEST_used=False,VALID_used=False' in s
    assert "'TRAIN_readout':'cached_TRAIN_affine_LBFGS'" in s and 'epochs=80,seconds=60' in s
