"""Three-worker admission remains bounded, owned and conditional on headroom."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from LightGenV2.tasks.t19_resnet20_baselines import train_easy100 as training
from LightGenV2.tasks.t19_resnet20_baselines.tests.test_small_text_easy100 import small_text_runner


def owned_commands(monkeypatch):
    command = (f'python -m LightGenV2.tasks.t19_resnet20_baselines.train_tasks '
               f'--gpu-uuid GPU-a --run-dir {training.TASK / "runs/simulation/owned"}')
    monkeypatch.setattr(Path, 'read_bytes', lambda path: command.encode())


def test_two_owned_contexts_admit_third_but_reject_foreign_and_fourth(monkeypatch):
    owned_commands(monkeypatch)
    apps = 'GPU-a, 101\nGPU-a, 102\nGPU-other, 999'
    assert training.validate_gpu_processes('GPU-a', apps, [101, 102], 3) == {101, 102}
    with pytest.raises(RuntimeError, match='two tasks'):
        training.validate_gpu_processes('GPU-a', apps, [101, 102])
    with pytest.raises(RuntimeError, match='outside this T19 queue'):
        training.validate_gpu_processes('GPU-a', apps, [101], 3)
    with pytest.raises(RuntimeError, match='three tasks'):
        training.validate_gpu_processes('GPU-a', apps + '\nGPU-a, 103', [101, 102, 103], 3)
    with pytest.raises(ValueError, match='two or three'):
        training.validate_gpu_processes('GPU-a', '', [], 4)


@pytest.mark.parametrize('active,state,allowed', [
    ({101, 102}, '18432,24576,50', True),   # Exactly 6144 MiB and 50% is admitted.
    ({101, 102}, '18432,24576,51', False),
    ({101, 102}, '18433,24576,20', False),
    ({101}, '18432,24576,95', True),        # Existing second-worker policy remains.
    ({101}, '18432,24576,96', False),
])
def test_third_worker_requires_low_utilization_and_memory(monkeypatch, active, state, allowed):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'GPU-a')
    observed = []

    def validate(uuid, apps, pids, cap):
        observed.append((uuid, set(pids), cap))
        return active

    monkeypatch.setattr(training, 'validate_gpu_processes', validate)
    monkeypatch.setattr(training.subprocess, 'check_output', lambda command, **kw:
        '\n'.join(f'GPU-a, {pid}' for pid in active)
        if '--query-compute-apps=gpu_uuid,pid' in command else state)
    if allowed:
        assert training.admit_gpu('GPU-a', sorted(active), 3) == active
    else:
        with pytest.raises(RuntimeError, match='headroom'):
            training.admit_gpu('GPU-a', sorted(active), 3)
    assert observed == [('GPU-a', active, 3)]


def test_stale_pid_identity_still_rejected_with_three_worker_limit(monkeypatch):
    monkeypatch.setattr(Path, 'read_bytes', lambda path: b'python unrelated.py')
    with pytest.raises(RuntimeError, match='verified T19 worker'):
        training.validate_gpu_processes('GPU-a', 'GPU-a, 101\nGPU-a, 102', [101, 102], 3)


def test_registry_accepts_explicit_three_and_preserves_old_default_two(tmp_path, monkeypatch):
    monkeypatch.setattr(training, 'TASK', tmp_path)
    run_root = tmp_path / 'runs'
    run_root.mkdir()
    status = run_root / 'queue_status.json'
    current = training.os.getpid()
    jobs = [{'status': 'running', 'gpu_uuid': 'GPU-a', 'pid': pid, 'name': name}
            for pid, name in ((current, 'current'), (101, 'peer1'), (102, 'peer2'))]
    status.write_text(json.dumps({'jobs': jobs}), encoding='utf-8')
    args = SimpleNamespace(queue_status_path=status, gpu_uuid='GPU-a',
                           run_dir=run_root / 'current', max_jobs_per_gpu=3)
    assert training.sharing_pids(args) == [101, 102]
    del args.max_jobs_per_gpu
    with pytest.raises(RuntimeError, match='not registered'):
        training.sharing_pids(args)


def test_wait_for_gpu_passes_explicit_capacity_and_waits_on_three_tasks(monkeypatch):
    calls = []
    monkeypatch.setattr(training, 'sharing_pids', lambda args: [101, 102])
    monkeypatch.setattr(training.time, 'sleep', lambda seconds: None)

    def admit(*args):
        calls.append(args)
        if len(calls) == 1:
            raise RuntimeError('Requested GPU already has three tasks')
        return {101, 102}

    monkeypatch.setattr(training, 'admit_gpu', admit)
    assert training.wait_for_gpu(SimpleNamespace(gpu_uuid='GPU-a', max_jobs_per_gpu=3)) == [101, 102]
    assert calls == [('GPU-a', [101, 102], 3)] * 2


def test_wait_default_retains_two_argument_mock_api(monkeypatch):
    calls = []
    monkeypatch.setattr(training, 'sharing_pids', lambda args: [101])
    monkeypatch.setattr(training, 'admit_gpu', lambda uuid, pids: calls.append((uuid, pids)))
    assert training.wait_for_gpu(SimpleNamespace(gpu_uuid='GPU-a')) == [101]
    assert calls == [('GPU-a', [101])]


def test_manifest_records_effective_three_workers_and_six_gpus(small_text_runner):
    fixture = small_text_runner
    fixture.args.max_jobs_per_gpu = 3
    fixture.args.max_physical_gpus = 6
    training.run(fixture.args)
    manifest = json.loads((fixture.run_dir / 'run_manifest.json').read_text())
    assert manifest['max_jobs_per_gpu'] == manifest['execution_profile']['max_jobs_per_gpu'] == 3
    assert manifest['execution_profile']['max_physical_gpus'] == 6
    assert manifest['status'] == 'complete'
