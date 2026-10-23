import importlib.util,sys
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
from lab_caltech20_TRAIN_teacher_ridge_20261024 import soft_targets,selftest


def test_wrong_teacher_is_not_a_training_target():
    assert selftest()['passed']
    y=np.arange(20);q,g=soft_targets(y,np.zeros((20,20)))
    assert not g.any() and np.array_equal(q,np.eye(20))


def test_only_TRAIN_estimator_no_TEST_or_Torch():
    s=(HERE/'lab_caltech20_TRAIN_teacher_ridge_20261024.py').read_text()
    assert 'import torch' not in s and 'test_labels' not in s and "system=prepare(x,targets,weights,families)" in s
    assert "targets[gate]=.5*hard[gate]+.5*soft[gate]" in s and 'estimate(system,regularization)' in s


def test_noise_component_audit_is_not_an_optimizer():
    import ast
    s=(HERE/'lab_caltech20_actual_readout_20261024.py').read_text();tree=ast.parse(s)
    n=next(v for v in tree.body if isinstance(v,ast.FunctionDef)and v.name=='TRAIN_perturbation_audit');segment=ast.get_source_segment(s,n)
    assert 'torch.optim' not in segment and "data['test']" not in segment and '.backward(' not in segment
    assert 'original_eval_identity_exact=True' in segment and 'gain_additive_only=' in segment and 'pixel_dropout_only=' in segment
