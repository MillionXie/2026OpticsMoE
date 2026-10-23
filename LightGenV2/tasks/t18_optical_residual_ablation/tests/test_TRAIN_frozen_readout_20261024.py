import ast
from pathlib import Path
P=Path(__file__).resolve().parents[1]/'lab_caltech20_TRAIN_frozen_readout_20261024.py'


def test_real_suffix_training_with_original_head_frozen():
    s=P.read_text();ast.parse(s)
    assert "keys=m.allowed_keys(model,4)-{'linear.weight','linear.bias'}" in s
    assert "('phase_sets.',c['phase_lr'])" in s and "('electronic.',c['electronic_lr'])" in s and "('fusion_logits.',c['alpha_lr'])" in s
    body=s.split('for begin in range(0,1600,8):',1)[1].split('assert used==1600',1)[0]
    assert "data['train'][1][ix]" in body and "data['test']" not in body
    assert "contrast_logits(held['features'],model.linear.weight,model.linear.bias)" in body
    assert 'original_5140_readout_bitwise_frozen=True' in s and 'loop.validate_state(m,native,original,keys)' in s


def test_equivalent_gradient_selftest_and_native_CPU_reports():
    s=P.read_text();assert 'gradient_maxabs' in s and 'error<1e-10 and grad<1e-9' in s
    assert 'transforms.remove();hook.remove()' in s and "for label in ('best','last')" in s and "for role in ('train','test')" in s
    assert 'TEST_gradient=False' in s and 'new_inference_layers=0' in s
