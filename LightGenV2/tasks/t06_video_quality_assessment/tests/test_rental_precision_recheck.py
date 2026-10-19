import pytest
from LightGenV2.tasks.t06_video_quality_assessment import rental_precision_recheck as r


def test_contract():
    c=r.config()
    assert c['pacing_wait_excluded_from_pure_cost'] and c['utilization_separately_measured']
    assert c['no_training_offload_quantization_graph']


@pytest.mark.parametrize('batch',[1,2,3,6,10])
def test_tail_and_cost_boundary(batch):
    rows=r.plan(batch)
    assert sum(b-a for a,b in rows)==558
    assert sum(b-a for a,b in rows if b<=200)==200
    assert all(not a<200<b for a,b in rows)


def test_idle_and_capacity():
    assert r.idle_seconds(2,.5)==2
    assert r.idle_seconds(2,1)==0
    assert r.fp32_weight_bytes(27480134248)>80*1024**3
    with pytest.raises(ValueError):r.idle_seconds(1,0)
