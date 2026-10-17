import ast
import inspect
import math
from pathlib import Path
import pytest
from LightGenV2.tasks.t06_video_quality_assessment import rental_sequential_busy50 as m


def test_config_has_finite_ordered_safety_and_separate_costs():
    c = m.config()
    assert c['models'] == ['deepseek', 'kimi']
    assert c['batch_start'] == c['batch_step'] == 1
    assert c['safety_maximum_batch'] == 16
    assert c['maximum_suite_seconds'] == 3300
    assert c['minimum_initial_free_mib'] == 75000
    assert c['not_equal_to_optical_spatial_occupancy']
    assert c['old_results_immutable']


@pytest.mark.parametrize('observed', [0, 20, 50, 75, 100])
def test_controller_matches_approved_busy50_formula(observed):
    for duty in (.05, .5, 1):
        expected = min(1., max(.05, duty * min(1.5, max(.67, 50/max(observed, 1.)))))
        assert m.update_duty(duty, observed) == expected


@pytest.mark.parametrize('bad', [-1, 101, float('nan')])
def test_invalid_driver_observations_rejected(bad):
    with pytest.raises(ValueError): m.update_duty(.5, bad)


def test_low_utilization_not_raised_using_fake_work():
    assert m.update_duty(1, 20) == 1
    assert m.imposed_wait(2, 1) == 0
    assert m.imposed_wait(2, .5) == 2


def test_paired_service_energy_includes_idle_and_does_not_claim_dataset_cost():
    cost = m.service_cost(40, 80, 2, 100)
    assert cost['service_time_16_videos_ms'] == 4000
    assert math.isclose(cost['estimated_energy_16_videos_j'], 1752.8)
    assert cost['completed_video_evaluations'] == 160
    assert cost['unique_videos'] == 2
    assert cost['not_dataset_once_latency']


def test_pooled_strict_target_not_peak_or_rounding():
    assert not m.reached_target({'pooled_mean_utilization_percent': 98})
    assert not m.reached_target({'pooled_mean_utilization_percent': 97.9999})
    assert m.reached_target({'pooled_mean_utilization_percent': 98.0001})


def test_one_load_per_model_and_no_cache_clear_between_successful_batches():
    tree = ast.parse(inspect.getsource(m.worker))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    assert sum(isinstance(n.func, ast.Attribute) and n.func.attr == 'runtime' for n in calls) == 1
    loop = next(n for n in ast.walk(tree) if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == 'batch')
    assert 'runtime(' not in ast.unparse(loop)
    batch_try = next(n for n in loop.body if isinstance(n, ast.Try))
    assert 'empty_cache' not in ast.unparse(ast.Module(body=batch_try.body, type_ignores=[]))
    assert 'torch.compile' not in inspect.getsource(m)
    assert '.backward(' not in inspect.getsource(m)


def test_measurement_has_native_queue8_and_whole_windows():
    text = inspect.getsource(m.measure)
    assert 'range(8)' in text and 'torch.cuda.synchronize()' in text
    assert 'phase(f"measured{repeat}", 30, duty)' in text
    assert 'phase("calibration", 30, duty, True)' in text
    assert 'No new full558 SRCC' in text
    assert 'FirstBlockTimer' not in text
    assert 'selected = [s for s in list(sampler.samples) if start <=' in text
    assert len(m.sources()) == len(m.previous.sources())+2


def test_legacy_sources_are_immutable():
    assert m.old.sha(Path(m.previous.__file__)) == 'f6f88fc341b52e023671a770b351879a026d24321c3d6243769304af298be698'
    assert m.old.sha(Path(m.old.__file__)) == '00dbe21d9d4b89b8c8c8f0454549c5df3a4e1fd3b877cfdf6ba319eadc0780ba'


def test_capacity_probe_not_formal_cost_and_does_not_relax_worker_gate():
    text = inspect.getsource(m.capacity_check)
    assert 'args.model != "kimi"' in text
    assert 'if args.output.exists()' in text
    assert 'full558_accuracy=False' in text
    assert 'formal_utilization_or_cost=False' in text
    assert 'rtol=.01, atol=.125' in text
    assert 'free < required_weight_bytes + 8*1024**3' in text
    assert 'float(free) < 75000' in inspect.getsource(m.worker)
    assert 'measure(' not in text and '.backward(' not in text
