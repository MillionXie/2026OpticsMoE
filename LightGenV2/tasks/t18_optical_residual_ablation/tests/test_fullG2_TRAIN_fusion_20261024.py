import ast
from pathlib import Path


def test_TRAIN_only_fusion_loss_and_exact_parent_admission():
    s=(Path(__file__).resolve().parents[1]/'lab_caltech20_fullG2_TRAIN_fusion_20261024.py').read_text();ast.parse(s)
    assert "keys={'fusion_logits.3','linear.weight','linear.bias'}" in s and '==5141' in s
    assert 'assert maxerr<=3e-6' in s and "for step in range(1,protocol['alpha_steps_max']+1):" in s
    assert "score=separation(cached_features(stats['train'],alpha),labels)" in s
    assert "alpha.clamp_(.4001001,.95)" in s and "alpha_candidates_from_TEST=0" in s
    assert "results['last']=review('last')" in s and "if not better:model.load_state_dict(original)" in s
    assert "audit['matched_samples']==2011" in s and "loop.validate_state(m,native,original,keys)" in s


def test_native_fusion_keeps_exact_pixels_and_bounded_memory():
    s=(Path(__file__).resolve().parents[1]/'lab_caltech20_fullG2_TRAIN_fusion_20261024.py').read_text();ast.parse(s)
    assert "fullG2_TRAIN_fusion_native_20261024" in s and "psutil.virtual_memory().available>8*2**30" in s
    assert "del stats['test']" in s and "native_scatter_step(stats['train'],alpha,labels)" in s
    assert "v.backward(gradient[begin:begin+8])" in s and "native_pixel_features_bitwise=True" in s
