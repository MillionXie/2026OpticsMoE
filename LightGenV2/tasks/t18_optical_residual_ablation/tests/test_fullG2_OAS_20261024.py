import ast,sys
from pathlib import Path
HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))


def test_same_TRAIN_estimator_and_original_affine_shape():
    from lab_caltech20_tucam_TRAIN_linear_20261023 import selftest
    selftest()


def test_all_five_physical_captures_and_frozen_parent_before_fit():
    s=(HERE/'lab_caltech20_fullG2_OAS_20261024.py').read_text();ast.parse(s)
    assert "for stage in base.STAGES:" in s and "ca['matched_samples']==2011" in s and "digest==protocol['prefix_SHA256'][stage+'/'+role]" in s
    assert "full['full_physical']" in s and "r['full_physical']" in s and "m.allowed_keys(model,5)" in s
    assert s.index('loop.validate_state(m,model,m.state_cpu(captured)')<s.index('TRAIN_linear_fit(features,')
    assert "keys={'linear.weight','linear.bias'}" in s and 'if not better:model.load_state_dict(original)' in s
    assert "for role in ('train','test')" in s and 'full_physical=True' in s and 'target=340' in s
    assert 'TEST_gradient=False' in s and 'analytical_fits=1' in s and 'hyperparameter_grid=False' in s and '.backward(' not in s
    assert '>=330' in s and "not m.LOCK.exists()" in s and "['returncode']==0" in s
