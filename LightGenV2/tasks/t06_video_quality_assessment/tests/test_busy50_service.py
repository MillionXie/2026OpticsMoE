import math
import inspect
import pytest
from LightGenV2.tasks.t06_video_quality_assessment import measure_busy50_service as m


def test_bounded_plan_and_protected_anchors():
    c = m.config()
    assert sum(map(len, c['batches'].values())) == 12
    assert c['qwen_B1_paper_anchor_unchanged']
    assert c['paper_excluded_baselines'] == ['qwen80']
    assert c['not_equal_to_optical_spatial_occupancy']


@pytest.mark.parametrize('observed,direction', [(25, 1), (50, 0), (100, -1)])
def test_feedback(observed, direction):
    result = m.update_duty(.5, observed)
    assert (result > .5) if direction > 0 else ((result < .5) if direction < 0 else result == .5)


def test_low_busy_cannot_be_raised_with_fake_work():
    assert m.update_duty(1, 20) == 1
    assert m.imposed_wait(2, 1) == 0
    assert m.imposed_wait(2, .5) == 2


@pytest.mark.parametrize('value', [float('nan'), -1, 101])
def test_invalid_samples(value):
    with pytest.raises(ValueError):
        m.update_duty(.5, value)


def test_wait_and_same_window_power_enter_cost():
    cost = m.service_cost(40, 80, 2, 100)
    assert cost['service_time_16_videos_ms'] == 4000
    assert math.isclose(cost['estimated_energy_16_videos_j'], 1752.8)
    assert cost['completed_video_evaluations'] == 160
    assert cost['unique_videos'] == 2
    assert cost['not_dataset_once_latency']


def test_no_false_full558_accuracy_claim():
    assert 'No new full558 evaluation' in inspect.getsource(m.measure)
    assert 'firstB' in m.config()['quality']


def test_feedback_caps_and_does_not_rewrite_observations():
    assert m.update_duty(.05, 100) == .05
    assert m.update_duty(1, 0) == 1
    assert m.update_duty(.5, 50) == .5
    with pytest.raises(ValueError):
        m.imposed_wait(1, 0)


def test_same_reruns_are_not_claimed_unique():
    cost = m.service_cost(30, 8*100, 3, 90)
    assert cost['completed_video_evaluations'] == 2400
    assert cost['unique_videos'] == 3
    assert cost['includes_all_imposed_wait']
