import copy
import pytest
from LightGenV2.tasks.t12_text_to_image.scene_lab_full70_quality import quality_gate,paired_reports,METRICS,GROUPS


def scores(value):return {g:{m:value for m in METRICS}for g in GROUPS}


def test_three_metrics_nine_groups_and_no_reward_gate():
    r=quality_gate(scores(.5),scores(.48))
    assert len(r['rows'])==9 and r['all_nine_within_4percent']and r['all_nine_in_target_range']
    assert not r['ImageReward_in_acceptance']
    assert not quality_gate(scores(.5),scores(.479))['all_nine_within_4percent']


def test_naturally_higher_scores_are_reported_never_clipped():
    r=quality_gate(scores(.5),scores(.51))
    assert r['all_nine_within_4percent']and not r['all_nine_in_target_range']and r['natural_improvement_preserved']
    assert all(row['EXP']==.51 for row in r['rows'])
    with pytest.raises(ValueError):quality_gate(scores(0),scores(.5))


def test_mismatched_PT_or_partial_cohort_cannot_be_compared():
    sim=dict(complete=True,scope='TEST_selection',generated_count=500,checkpoint_sha256='a'*64,TEST_selection_protocol={'request':'fixed'})
    exp=dict(sim,actual_CCD_replay=True,physical_CCD_stage_calls=3000);paired_reports(sim,exp)
    for bad in [dict(exp,checkpoint_sha256='b'*64),dict(exp,generated_count=4),dict(exp,actual_CCD_replay=False),dict(exp,TEST_selection_protocol={'request':'changed'})]:
        with pytest.raises(ValueError):paired_reports(sim,bad)


def test_fixed_original_reference_is_explicit_and_only_one_overall_may_exceed():
    from LightGenV2.tasks.t12_text_to_image.scene_lab_full70_quality import fixed_reference_gate,fixed_reference_reports,FIXED_SIM
    sim=dict(complete=True,scope='TEST_selection',generated_count=500,checkpoint_sha256=FIXED_SIM,TEST_selection_protocol={'fixed':1})
    exp=dict(sim,checkpoint_sha256='b'*64,actual_CCD_replay=True,physical_CCD_stage_calls=3000,
             parent_physical_generator_SHA256=FIXED_SIM,adapted_existing_head=True,fixed_SIM_reference_SHA256=FIXED_SIM)
    fixed_reference_reports(sim,exp)
    with pytest.raises(ValueError):fixed_reference_reports(dict(sim,checkpoint_sha256='c'*64),exp)
    with pytest.raises(ValueError):paired_reports(sim,exp)
    e=scores(.49);e['all']['CLIP_T']=.501
    assert fixed_reference_gate(scores(.5),e)['overall_at_most_one_above_and_within_5percent']
    e['all']['CLIP_I']=.501
    assert not fixed_reference_gate(scores(.5),e)['overall_at_most_one_above_and_within_5percent']
