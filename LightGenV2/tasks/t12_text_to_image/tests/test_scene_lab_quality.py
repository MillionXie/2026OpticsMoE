import pytest
from LightGenV2.tasks.t12_text_to_image.scene_lab_quality import quality_gate,METRICS


def test_all_groups_metrics_and_negative_reward_sign():
    sim={g:{m:(-2 if m=='ImageReward_compact'else .8)for m in METRICS}for g in ('all','lamp','pillow')}
    exp={g:dict(v)for g,v in sim.items()};exp['lamp']['ImageReward_compact']=-2.08
    result=quality_gate(sim,exp);assert result['all_passed'] and len(result['rows'])==12
    exp['lamp']['ImageReward_compact']=-2.12
    assert not quality_gate(sim,exp)['all_passed']


def test_no_near_zero_percentage_or_nonfinite_score():
    sim={g:{m:0 for m in METRICS}for g in ('all','lamp','pillow')};exp={g:dict(v)for g,v in sim.items()}
    result=quality_gate(sim,exp);assert all(r['relative_drop']is None for r in result['rows'])
    exp['all']['CLIP_T']=float('nan')
    with pytest.raises(ValueError):quality_gate(sim,exp)
