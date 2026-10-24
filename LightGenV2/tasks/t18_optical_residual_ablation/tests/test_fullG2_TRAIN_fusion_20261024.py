import ast
from pathlib import Path


def test_TRAIN_only_fusion_loss_and_exact_parent_admission():
    s=(Path(__file__).resolve().parents[1]/'lab_caltech20_fullG2_TRAIN_fusion_20261024.py').read_text();ast.parse(s)
    assert "keys={'fusion_logits.3','linear.weight','linear.bias'}" in s and '==5141' in s
    assert 'assert maxerr<=3e-6' in s and "for step in range(1,301):" in s
    assert "score=separation(cached_features(stats['train'],alpha),labels)" in s
    assert "alpha.clamp_(.4001001,.95)" in s and "alpha_candidates_from_TEST=0" in s
    assert "results['last']=review('last')" in s and "if not better:model.load_state_dict(original)" in s
    assert "audit['matched_samples']==2011" in s and "loop.validate_state(m,native,original,keys)" in s
