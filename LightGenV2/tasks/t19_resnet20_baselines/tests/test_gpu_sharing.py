from pathlib import Path

import pytest

from LightGenV2.tasks.t19_resnet20_baselines import train_easy100 as training


def test_second_owned_task_allowed_but_other_queue_process_rejected(monkeypatch):
    cmd = f'python -m LightGenV2.tasks.t19_resnet20_baselines.train_easy100 --gpu-uuid GPU-a --run-dir {training.TASK / "runs/simulation/first"}'
    monkeypatch.setattr(Path, 'read_bytes', lambda path: cmd.encode())
    assert training.validate_gpu_processes('GPU-a', 'GPU-a, 101\nGPU-other, 202', [101]) == {101}
    with pytest.raises(RuntimeError, match='outside this T19 queue'):
        training.validate_gpu_processes('GPU-a', 'GPU-a, 202', [101])


def test_third_task_and_stale_pid_identity_rejected(monkeypatch):
    with pytest.raises(RuntimeError, match='two tasks'):
        training.validate_gpu_processes('GPU-a', 'GPU-a, 101\nGPU-a, 102', [101, 102])
    monkeypatch.setattr(Path, 'read_bytes', lambda path: b'python other_job.py')
    with pytest.raises(RuntimeError, match='verified T19 worker'):
        training.validate_gpu_processes('GPU-a', 'GPU-a, 101', [101])


@pytest.mark.parametrize('state,allowed', [('3000,24576,20', True), ('21000,24576,20', False), ('3000,24576,99', False)])
def test_sharing_requires_memory_and_compute_headroom(monkeypatch, state, allowed):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'GPU-a')
    monkeypatch.setattr(training, 'validate_gpu_processes', lambda *args: {101})
    monkeypatch.setattr(training.subprocess, 'check_output', lambda command, **kw:
                        'GPU-a, 101' if '--query-compute-apps=gpu_uuid,pid' in command else state)
    if allowed:
        training.admit_gpu('GPU-a', [101])
    else:
        with pytest.raises(RuntimeError, match='headroom'):
            training.admit_gpu('GPU-a', [101])


def test_cpu_preflight_race_waits_for_foreign_job_without_starting_cuda(monkeypatch):
    from types import SimpleNamespace
    calls=[]
    monkeypatch.setattr(training,'sharing_pids',lambda args:[101])
    def admit(*args):
        calls.append(args)
        if len(calls)==1:raise RuntimeError('Requested GPU has a process outside this T19 queue')
    monkeypatch.setattr(training,'admit_gpu',admit)
    monkeypatch.setattr(training.time,'sleep',lambda seconds:None)
    assert training.wait_for_gpu(SimpleNamespace(gpu_uuid='GPU-a'))==[101]
    assert len(calls)==2


def test_admission_does_not_hide_invalid_configuration(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(training,'sharing_pids',lambda args:[])
    def invalid(*args):raise RuntimeError('Set CUDA_VISIBLE_DEVICES to the single requested GPU UUID')
    monkeypatch.setattr(training,'admit_gpu',invalid)
    with pytest.raises(RuntimeError,match='CUDA_VISIBLE_DEVICES'):
        training.wait_for_gpu(SimpleNamespace(gpu_uuid='GPU-a'))
