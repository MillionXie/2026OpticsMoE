import ast
from pathlib import Path
import numpy as np
import pytest
from LightGenV2.tasks.t06_video_quality_assessment import analyze_saved_batch_numerics as audit


def test_feature_delta_explains_linear_readout():
    weights = np.array([[1.,2.],[-1.,1.]])
    first = np.array([[1.,2.],[3.,4.],[2.,1.],[0.,3.]])
    changed = first+.125
    result = audit.decompose(dict(features=first,logits=first@weights.T),
        dict(features=changed,logits=changed@weights.T),weights,np.array([0.,100.]))
    assert result["unexplained_cross_batch_logit_residual_max_abs"] == 0
    assert result["fixed_features_replayed_batch_head_logit_max_abs_delta"] == 0
    assert result["feature_difference"]["max_abs"] == .125


def test_inversions_and_ties_are_not_conflated():
    result = audit.pair_order_changes(np.array([1.,2.,3.]),np.array([2.,1.,1.]))
    assert result == dict(strict_order_reversals=2,tie_state_changes=1,compared_pairs=3)


def test_numerical_evidence_refuses_nonfinite_or_misaligned():
    with pytest.raises(ValueError):
        audit.error_statistics(np.ones(2),np.array([1.,np.nan]))
    with pytest.raises(ValueError):
        audit.error_statistics(np.ones(2),np.ones(3))


def test_analysis_cannot_train_or_load_backbone():
    tree = ast.parse(Path(audit.__file__).read_text(encoding="utf-8"))
    calls = {node.func.attr for node in ast.walk(tree) if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute)}
    assert not {"backward","step","train","from_pretrained","cuda","to"}.intersection(calls)
