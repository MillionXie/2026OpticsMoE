import math
import inspect
from types import SimpleNamespace
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


def test_feedback_covers_compute_and_idle_not_only_tail():
    samples = [SimpleNamespace(host_monotonic_s=i*.1, utilization_percent=100 if i < 20 else 0)
               for i in range(40)]
    observation = m.feedback_observation(samples, 0, 4)
    assert observation['observed_percent'] == 50
    assert observation['sample_count'] == 40
    assert m.feedback_observation(samples, 3.9, 4) is None
    assert m.config()['protocol'].endswith('v2_complete_group_feedback')


def test_only_remaining_eight_are_planned():
    preserved = ['qwen_batch01', 'qwen_batch03', 'glm_batch01', 'glm_batch02']
    plan = m.planned_points(m.config(), dict(preserved_points=preserved))
    assert plan == [('small', 1), ('small', 3), ('clip', 1), ('clip', 10),
                    ('yolo', 1), ('yolo', 3), ('deepseek', 1), ('deepseek', 8)]
    assert len(m.planned_points(m.config())) == 12
    with pytest.raises(ValueError):
        m.planned_points(m.config(), dict(preserved_points=preserved[:3]))


def test_failed_cleanup_preserves_groups_csv_and_every_error(tmp_path):
    import json
    from LightGenV2.tasks.t06_video_quality_assessment import batch_telemetry
    def failure(*args, **kwargs):
        raise RuntimeError('injected reader or clock failure')
    sampler = SimpleNamespace(samples=[batch_telemetry.Sample(0, 1, 'measured1', 100, 50, 5000)],
        set_phase=lambda phase: None, stop=lambda **kwargs: None, check=failure)
    clocks = SimpleNamespace(stop=failure)
    guard = SimpleNamespace(stop=lambda: None, check=lambda: None, records=[{'pid': 42}])
    with pytest.raises(RuntimeError, match='raw evidence retained'):
        m.persist_measurement_evidence(tmp_path, sampler, clocks, guard, [{'forwards': 8}], [{'new_duty': .5}])
    assert (tmp_path/'telemetry.csv').read_text().count('\n') == 2
    assert json.loads((tmp_path/'forward_groups.json').read_text()) == [{'forwards': 8}]
    assert json.loads((tmp_path/'controller_updates.json').read_text()) == [{'new_duty': .5}]
    status = json.loads((tmp_path/'measurement_evidence_status.json').read_text())
    assert status['status'] == 'incomplete' and status['missing_values_never_imputed']
    assert [e['operation'] for e in status['errors']] == ['clock_stop', 'telemetry_integrity']


def test_continuation_rejects_other_directory_before_read(tmp_path):
    with pytest.raises(ValueError, match='canonical'):
        m.continuation_evidence(SimpleNamespace(resume_from=tmp_path/'other', output=tmp_path/'new'), m.config())


def test_continuation_binds_four_original_points_without_rewriting(tmp_path, monkeypatch):
    import json
    monkeypatch.setattr(m, 'TASK', tmp_path)
    monkeypatch.setattr(m, 'sources', lambda: {'arithmetic.py': 'unchanged',
                                              'batch_telemetry.py': 'new'})
    previous = tmp_path/'runs/simulation/lgvq_resident_busy50_a100_20261017/attempt02_20261018'
    previous.mkdir(parents=True)
    names = ['qwen_batch01', 'qwen_batch03', 'glm_batch01', 'glm_batch02']
    c = m.config()
    (previous/'suite_status.json').write_text(json.dumps(dict(status='failed', completed=names, protocol=c)))
    for name in names:
        p = previous/name; (p/'measurement').mkdir(parents=True)
        baseline = name.split('_')[0]
        identity = dict(checkpoint_sha256=m.costs.CHECKPOINTS[baseline], baseline=baseline,
            gpu_uuid=m.frozen.A100_UUID, manifest_sha256=m.frozen.MANIFEST_SHA, protocol=c,
            source_sha256={'arithmetic.py': 'unchanged', 'batch_telemetry.py': 'old'})
        (p/'identity.json').write_text(json.dumps(identity))
        (p/'report.json').write_text(json.dumps(dict(status='complete', repeats=[{}, {}, {}])))
        (p/'status.json').write_text('{}')
        for item in ['telemetry.csv','clock_telemetry.csv','forward_groups.json',
                     'controller_updates.json','process_audit.json','repeat01.json','repeat02.json','repeat03.json']:
            (p/'measurement'/item).write_text('raw fixture')
    args = SimpleNamespace(resume_from=previous, output=previous.with_name('attempt03_remaining_20261018'))
    evidence = m.continuation_evidence(args, c)
    assert evidence['preserved_points'] == names
    assert len(evidence['files_sha256']) == 45
    assert len(m.planned_points(c, evidence)) == 8
    monkeypatch.setattr(m, 'sources', lambda: {'arithmetic.py': 'changed', 'batch_telemetry.py': 'new'})
    with pytest.raises(RuntimeError, match='arithmetic/input'):
        m.continuation_evidence(args, c)
