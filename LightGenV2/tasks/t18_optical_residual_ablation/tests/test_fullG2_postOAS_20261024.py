import ast
from pathlib import Path


def test_original_five_capture_head_fusion_only_finite_continue():
    p=Path(__file__).resolve().parents[1]/'lab_caltech20_fullG2_postOAS_20261024.py';s=p.read_text();ast.parse(s)
    assert 'm.allowed_keys(model,5)' in s and '==5150' in s and "for stage in base.STAGES:" in s
    assert "audit['matched_samples']==2011" in s and "sp['prefix_SHA256'][stage+'/'+role]" in s
    assert "epochs=12,seconds=120" in s and '>=240' in s and '>=225' in s
    assert "base.fit_suffix(" in s and 'new_inference_layers=0' in s and 'TEST_gradient=False' in s and '.backward(' not in s
