import ast
from pathlib import Path

import pytest

from LightGenV2.tasks.t06_video_quality_assessment import glm46v_flash_temporal_baseline as run
from LightGenV2.tasks.t06_video_quality_assessment.prepare_vlm_assets import validate_profile


def source(name):
    tree = ast.parse(Path(run.__file__).read_text())
    return ast.unparse(next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name))


def test_original_protocol_except_native_hidden_width():
    c = run.config()
    assert c['frames'] == 4 and c['epochs'] == 50 and c['inference_batches'] == [1]
    assert c['thinking'] is False and c['hidden_width'] * 5 == 20480
    assert c['train_samples'] == 2250 and c['test_samples'] == 558
    assert c['backbone_trainable_parameters'] == 0
    assert c['selection'] == 'highest_TEST_SRCC_development_no_VAL'


def test_pinned_public_assets():
    c = run.config()
    validate_profile(c)
    assert c['model_revision'] == '411bb4d77144a3f03accbf4b780f5acb8b7cde4e'
    assert len(c['official_assets']) == 13
    assert sum(v['size'] for k, v in c['official_assets'].items() if k.endswith('.safetensors')) == 20585644048


def test_source_binding_and_finite_suite():
    assert all(len(v) == 64 for v in run.sources().values())
    s = source('suite')
    assert '10800' in s and '7200' in s and 'no automatic retry' in s
    assert 'a100_batch04' not in s and 'a100_batch08' not in s


def test_native_backbone_no_thinking_generation_or_patch():
    s = source('native_model')
    assert 'Glm4vForConditionalGeneration.from_pretrained' in s
    assert 'enable_thinking=False' in s and 'requires_grad_(False)' in s
    assert "'5.0.0'" in s and 'bfloat16' in s
    f = source('features')
    assert 'model.model(**inputs' in f and 'last_hidden_state' in f
    assert '.generate(' not in f


def test_four_frames_geometry_metadata_and_no_resampling():
    s = source('prepare_batch')
    assert "'do_sample_frames': False" in s and "'video_metadata'" in s
    assert "'do_resize': False" in s and 'len(d[0]) != 4' in s
    assert 'core.FRAME_FRACTIONS[4]' in s
    assert "grid[:, 0].eq(2)" in s


def test_original_single_loss_optimizer_no_scan():
    s = source('train_rows')
    assert 'range(1, 51)' in s and 'cross_entropy' in s
    assert 'lr=0.001, weight_decay=0.0' in s and 'T_max=50' in s
    assert 'FiveNativeTokenRows(initial)' in s
    assert '20480' in s and 'strict_cpu_metrics' in s


def test_accuracy_only_and_fresh_full_test():
    s = source('run')
    assert 'for i, row in enumerate(test)' in s
    assert 'formal_latency_energy_measured=False' in s and 'test_videos=558' in s
    assert 'runtime_sha256' in s and 'training_receipt.json' in s
    assert 'native_forward_last_logits_vs_feature_tap_max_abs' in s


def test_admission_not_small_model_threshold():
    assert '32000' in source('guard')


def test_recovery_keeps_original_absolute_budget():
    s = source('suite')
    assert 'min(time.time() + 10800, args.deadline_unix)' in s
    assert 'Original suite deadline already passed' in s


@pytest.mark.parametrize('status,header,length', [
    (200, 'bytes 4-7/12', '4'), (206, 'bytes 0-3/12', '4'),
    (206, 'bytes 4-7/13', '4'), (206, 'bytes 4-7/12', '5')])
def test_recovery_rejects_unmatched_range(status, header, length):
    from types import SimpleNamespace
    from LightGenV2.tasks.t06_video_quality_assessment.glm46v_assets import validate_range
    with pytest.raises(RuntimeError):
        validate_range(SimpleNamespace(status_code=status, headers={'Content-Range': header, 'Content-Length': length}), 4, 7, 12)


def test_recovery_accepts_exact_range_only():
    from types import SimpleNamespace
    from LightGenV2.tasks.t06_video_quality_assessment.glm46v_assets import validate_range
    validate_range(SimpleNamespace(status_code=206, headers={'Content-Range': 'bytes 4-7/12', 'Content-Length': '4'}), 4, 7, 12)


def test_transient_range_retry_bounded_and_does_not_retry_identity(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import requests
    from LightGenV2.tasks.t06_video_quality_assessment import glm46v_assets as assets
    calls = []
    class Response:
        status_code = 503
        headers = {}
        def close(self):
            pass
    def get(*args, **kwargs):
        calls.append(kwargs)
        return Response()
    clock = [0.0]
    monkeypatch.setattr(assets.time, 'time', lambda: clock[0])
    monkeypatch.setattr(assets.time, 'sleep', lambda delay: clock.__setitem__(0, clock[0]+delay))
    receipts = []
    monkeypatch.setattr(assets.shared.frozen, 'write_json', lambda path, value: receipts.append(value))
    fake = SimpleNamespace(get=get, HTTPError=requests.HTTPError, RequestException=requests.RequestException)
    with pytest.raises(requests.HTTPError):
        assets.range_request(fake, 'https://example.invalid', 4, 7, 12, assets.time.time()+300, tmp_path)
    assert len(calls) == 3
    assert [r['attempt'] for r in receipts] == [1, 2, 3]
    calls.clear()
    Response.status_code = 200
    with pytest.raises(RuntimeError):
        assets.range_request(fake, 'https://example.invalid', 4, 7, 12, assets.time.time()+300, tmp_path)
    assert len(calls) == 1


def test_refuse_old_outputs_before_loading(tmp_path):
    with pytest.raises(FileExistsError):
        run.shared.new_output(tmp_path)


def test_server_retry_after_is_respected_without_extending_deadline(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import requests
    from LightGenV2.tasks.t06_video_quality_assessment import glm46v_assets as assets
    clock = [0.0]
    monkeypatch.setattr(assets.time, 'time', lambda: clock[0])
    monkeypatch.setattr(assets.time, 'sleep', lambda delay: clock.__setitem__(0, clock[0]+delay))
    monkeypatch.setattr(assets.shared.frozen, 'write_json', lambda *args: None)
    limited = SimpleNamespace(status_code=429, headers={'Retry-After': '120'}, close=lambda: None)
    good = SimpleNamespace(status_code=206, headers={'Content-Range': 'bytes 4-7/12', 'Content-Length': '4'})
    responses = iter([limited, good])
    fake = SimpleNamespace(get=lambda *a, **k: next(responses), HTTPError=requests.HTTPError,
                           RequestException=requests.RequestException)
    assert assets.range_request(fake, 'https://example.invalid', 4, 7, 12, 300, tmp_path) is good
    assert clock[0] == 120
    clock[0] = 0
    fake.get = lambda *a, **k: limited
    with pytest.raises(TimeoutError):
        assets.range_request(fake, 'https://example.invalid', 4, 7, 12, 60, tmp_path)
    assert clock[0] == 0
