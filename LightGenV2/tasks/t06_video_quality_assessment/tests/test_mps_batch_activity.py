import pytest
from LightGenV2.tasks.t06_video_quality_assessment import measure_mps_batch_activity as module


def test_finite_plan_no_rental_no_warmup_and_no_paper_overwrite():
    c = module.configuration()
    assert c['no_rental_or_ds_kimi'] and c['no_formal_paper_overwrite']
    assert c['paired_caps'] == [100,50] and c['explicit_warmup_forwards'] == 0
    assert c['minimum_profiled_busy_percent'] == 50


@pytest.mark.parametrize('batch',[1,2,3,4,6,8,10,12,16])
def test_original200558_boundaries_and_all_tails(batch):
    plan = module.batches(batch)
    assert [i for a,b in plan for i in range(a,b)] == list(range(558))
    assert all(not a < 200 < b for a,b in plan)
    assert sum(b-a for a,b in plan if b <= 200) == 200
    assert max(b-a for a,b in plan) == batch


def test_pilot_fixed_videos_not_fixed_number_of_batches():
    for batch in (1,3,16):
        assert [i for a,b in module.batches(batch,True) for i in range(a,b)] == list(range(32))


def test_first_qualifying_batch_requires_every_repeat_not_highest_or_average():
    def report(x):
        return dict(status='complete',measurement=dict(lower_percent=x))
    attempts = {1:[report(49)],2:[report(51),report(49),report(59)],3:[report(50)]*3,4:[report(90)]*3}
    assert module.selected_batch(attempts,[1,2,3,4]) == 3
    assert not module.qualifies([report(99)])


def test_batch_cost_uses_original_nonpreprocessing_gaps():
    calls = [dict(preprocess_start=0,model_call_start=2,first_block_start=3,gpu_output=5),
             dict(preprocess_start=6,model_call_start=8,first_block_start=9,gpu_output=12)]
    intervals,compute,gap = module.original.charged_intervals(calls)
    assert intervals == [(3,5),(5,6),(9,12)] and compute == 5 and gap == 1


def test_authorized_resident_selection_has_three_warmed_repeats_not_cold_profile(monkeypatch):
    monkeypatch.setattr(module,'CONFIG',module.TASK/'configs/baselines/mps_resident_batch_20261021.json')
    c = module.configuration()
    assert (c['warmup_s'],c['measurement_s'],c['queue_depth']) == (5,30,8)
    assert c['cost_uses_no_warmup_and_distinct_videos'] and c['utilization_uses_warmup_and_same_inputs']
    def repeat(x):
        return dict(measurement=dict(mean_gpu_utilization_percent=x))
    assert module.resident_qualifies([repeat(51),repeat(52),repeat(50)])
    assert not module.resident_qualifies([repeat(99),repeat(49),repeat(99)])


def test_batch10_comparison_is_fixed_only_clip_yolo_without_other_models(monkeypatch):
    monkeypatch.setattr(module,'CONFIG',module.TASK/'configs/baselines/mps_batch10_comparison_20261021.json')
    c=module.configuration()
    assert c['run_models']==['clip','yolo'] and c['fixed_comparison_batch']==10
    assert c['candidate_batches']['clip']==c['candidate_batches']['yolo']==[10]
    assert c['explicit_warmup_forwards']==0 and c['paired_caps']==[100,50]
    assert (c['warmup_s'],c['measurement_s'],c['queue_depth'])==(5,30,8)


def test_batch6_comparison_preserves_boundaries_and_scope(monkeypatch):
    monkeypatch.setattr(module,'CONFIG',module.TASK/'configs/baselines/mps_batch6_comparison_20261021.json')
    c=module.configuration()
    assert c['run_models']==['clip','yolo'] and c['fixed_comparison_batch']==6
    assert c['candidate_batches']['clip']==c['candidate_batches']['yolo']==[6]
    assert c['paired_caps']==[100,50] and c['explicit_warmup_forwards']==0
    assert (c['warmup_s'],c['measurement_s'],c['queue_depth'])==(5,30,8)
    costs=[(a,b) for a,b in module.batches(6) if b<=200]
    assert len(costs)==34 and costs[-1]==(198,200)
    assert module.batches(6)[-1]==(554,558)


def test_batch8_clip_only_no_training_or_warm_cost(monkeypatch):
    monkeypatch.setattr(module,'CONFIG',module.TASK/'configs/baselines/mps_clip_batch8_20261022.json')
    c=module.configuration()
    assert c['run_models']==['clip'] and c['candidate_batches']['clip']==[8]
    assert c['explicit_warmup_forwards']==0
    assert len([x for x in module.batches(8) if x[1]<=200])==25
    assert module.batches(8)[-1]==(552,558)


def test_ready_input_boundary_counts_front_return_and_all_real_gaps():
    calls=[dict(preprocess_start=0,model_call_start=2,first_block_start=3,gpu_output=5,call_return=5.5),
           dict(preprocess_start=6,model_call_start=8,first_block_start=9,gpu_output=12,call_return=13)]
    intervals,model,gap=module.ready_input_intervals(calls)
    assert intervals==[(2,5.5),(5.5,6),(8,13)]
    assert model==8.5 and gap==.5
    assert sum(b-a for a,b in intervals)==9
    assert module.ready_input_intervals([])==([],0.,0.)
    with pytest.raises(ValueError):
        module.ready_input_intervals([dict(calls[0],call_return=4)])
    with pytest.raises(ValueError):
        module.ready_input_intervals([calls[0],dict(calls[1],preprocess_start=4)])
