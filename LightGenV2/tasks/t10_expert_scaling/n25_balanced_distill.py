"""Finite full25 seed17 transfer of the frozen source-balanced KD recipe.

The optical model remains N25/k25. The TEST-exposed CNN teacher is reused,
while the original N25 checkpoint gets its own locked eight-view parent cache.
Only this seed17 pilot is authorized by this profile; it cannot extend itself.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

from . import reheat_refine as base
from . import cached_distill as kd
from . import seed_balanced_distill as shared
from .expert_tail_schedule import save, sha


SOURCE_SHA = '274a06192417a9226abd21074491a58995ebef95e08d01a5cb1dbcd1e1e3fd8e'
GPU_UUID = 'GPU-e8837b85-d55b-8e81-aaa5-ec1ac326932d'


def validate(c):
    assert c['seed'] == 17 and c['seeds'] == [17]
    assert c['model_experts'] == 25 and c['expected_parameter_count'] == 5634176
    assert c['seed_gpu'] == {'17': 1} and c['gpus'] == {'1': GPU_UUID}
    assert c['allow_shared_gpu'] is True and c['minimum_free_memory_mib'] == 6000
    assert c['rounds'] == 16 and c['check_rounds'] == [4, 8, 12, 16]
    assert c['batch'] == 16 and c['microbatch'] == 2 and c['samples_per_class'] == 128
    assert c['schedule_type'] == 'distill' and c['temperature'] == 2 and c['parent_weight'] == .5
    assert c['fixed_lambda_kd'] == .3203496288685833
    assert c['gradient_target_ratio'] == .5 and c['minimum_weighted_gradient_ratio'] == .05
    assert c['peak_lr'] == .0003 and c['end_lr'] == .00003 and c['ema_decay'] == .95
    assert c['explicit_test_training_authorization'] and c['test_images_enter_student_gradient']
    assert c['source_runs']['17']['sha256'] == SOURCE_SHA
    assert c['source_runs']['17']['actual_epoch'] == 79
    assert Path(c['source_runs']['17']['run']).name == 'N25_k25_s17'
    assert c['automatic_additional_seeds'] is False and c['automatic_extension'] is False
    assert c['teacher_cache_source_commit'] == '654183a3c16bd3cfef329a132fc15478e53f6bdb'
    assert c['locked_teacher_manifest_sha256'] == '47c7add1fd82c6aa05c4595cd885ee5f80c5eadd98fe3892b14236ebf1d8af61'


def bind_source_environment():
    commit = os.environ['T10_FULL_EXPERT_SOURCE_COMMIT']
    assert re.fullmatch('[0-9a-f]{40}', commit)
    for key in ['T10_SEED_FOLLOWUP_SOURCE_COMMIT', 'T10_REHEAT_SOURCE_COMMIT', 'T10_DISTILL_SOURCE_COMMIT']:
        if key in os.environ:
            assert os.environ[key] == commit, key
        os.environ[key] = commit
    return commit


def config_for_seed(c, seed, root, prepared=True):
    validate(c)
    assert seed == 17
    v = copy.deepcopy(c)
    v['recipes'] = ['source_balanced']
    v['recipe_gpu'] = {'source_balanced': 1}
    reference = root / 'reference_s17'
    original = c['source_runs']['17']
    row = dict(original, run=str(reference)) if prepared else copy.deepcopy(original)
    v['sources'] = {'student': row, 'source_balanced': row}
    v['original_source'] = copy.deepcopy(original)
    v['incumbent_run'] = str(reference)
    v['incumbent_sha256'] = v['parent_checkpoint_sha256'] = original['sha256']
    v['incumbent_result_name'] = 'result.json'
    v['incumbent_result_is_selected'] = True
    v['parent_cache'] = str(root / 'parent_s17')
    v['parent_cache_source_commit'] = os.environ['T10_FULL_EXPERT_SOURCE_COMMIT']
    v['calibration_run'] = str(root / 'calibration_s17')
    if prepared:
        receipt = json.loads((reference / 'preparation_receipt.json').read_text())
        assert receipt['seed'] == seed and receipt['source_checkpoint_sha256'] == original['sha256']
        assert receipt['source_commit'] == os.environ['T10_FULL_EXPERT_SOURCE_COMMIT']
        assert sha(Path(v['parent_cache']) / 'receipt.json') == receipt['parent_receipt_sha256']
        v['incumbent_predictions_sha256'] = receipt['predictions_sha256']
    return v


class Auxiliary(shared.Auxiliary):
    def __init__(self, c, calibrated=True):
        super().__init__(c, calibrated)
        self.identity.update(model_experts=25, model_top_k=25, expected_parameter_count=5634176,
            pilot_seed=17, automatic_additional_seeds=False,
            method_development_seed17_finalized=False,
            role='Frozen N49-discovered source-balanced recipe transferred to original N25 seed17; exposed-development fit.')

    def diagnostics(self, model, split, step, best_correct):
        result = super().diagnostics(model, split, step, best_correct)
        result['note'] = 'Finite N25/k25 seed17 pilot. Original N25 baseline remains a candidate. No matched N49 budget claim, additional seeds, or target-based extension.'
        return result


def assert_shared_gpu_policy(c, uuid, free_memory_mib, processes, root, require_capacity=True):
    """Only capacity and this task's occupancy are checked; foreign jobs stay intact."""
    assert uuid == c['gpus']['1']
    if require_capacity:
        assert free_memory_mib >= c['minimum_free_memory_mib'], 'Insufficient free GPU memory; preserve all processes and stop.'
    module = __package__ + '.n25_balanced_distill'
    for row in processes:
        if row['gpu_uuid'] == uuid:
            command = row['command'].replace('\\', '/')
            assert module not in command and str(root).replace('\\', '/') not in command, 'A GPU process from this task already exists; do not duplicate it.'


def check_gpu(c, root, require_capacity=True):
    uuid = subprocess.check_output(['nvidia-smi', '-i', '1', '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
    free = int(subprocess.check_output(['nvidia-smi', '-i', '1', '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).strip())
    raw = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid', '--format=csv,noheader,nounits'], text=True)
    rows = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        pid, gpu_uuid = [v.strip() for v in line.split(',', 1)]
        if gpu_uuid != uuid:
            continue
        try:
            command = Path('/proc', pid, 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
        except FileNotFoundError:
            continue
        rows.append(dict(pid=int(pid), gpu_uuid=gpu_uuid, command=command))
    assert_shared_gpu_policy(c, uuid, free, rows, root, require_capacity)
    return uuid


def pipeline(c, root, profile):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=check_gpu(c, root), CUDA_DEVICE_ORDER='PCI_BUS_ID', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    for mode in ['prepare', 'calibrate', 'worker']:
        check_gpu(c, root)
        with (root / f'seed17_{mode}.log').open('x') as stream:
            p = subprocess.Popen([sys.executable, '-u', '-m', __package__ + '.n25_balanced_distill', '--profile', str(profile), '--out', str(root), '--seed', '17', '--mode', mode], env=env, stdout=stream, stderr=subprocess.STDOUT)
            save(root / 'seed17_progress.json', dict(seed=17, state=mode, pipeline_pid=os.getpid(), child_pid=p.pid, gpu=1, shared_gpu=True, source_commit=os.environ['T10_FULL_EXPERT_SOURCE_COMMIT'], timestamp=time.time()))
            try:
                p.wait()
            finally:
                if p.poll() is None:
                    p.terminate()
                    p.wait(timeout=30)
        if p.returncode:
            raise RuntimeError(f'seed17 {mode} failed with {p.returncode}; artifacts preserved, no automatic retry')
    check_gpu(c, root, require_capacity=False)
    save(root / 'seed17_progress.json', dict(seed=17, state='complete', pipeline_pid=os.getpid(), child_pid=None, gpu=1, shared_gpu=True, timestamp=time.time()))


def supervise(c, root, profile):
    validate(c)
    commit, hashes = shared.published_sources(profile, c)
    check_gpu(c, root)
    source = Path(c['source_runs']['17']['run'])
    assert sha(source / 'best_checkpoint.pt') == SOURCE_SHA
    root.mkdir(parents=True, exist_ok=False)
    save(root / 'plan.json', dict(configuration=c, source_commit=commit, source_files=hashes, scheduler_pid=os.getpid(), host_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(), shared_gpu_authorization=True))
    p = None
    try:
        with (root / 'seed17_pipeline.log').open('x') as stream:
            p = subprocess.Popen([sys.executable, '-u', '-m', __package__ + '.n25_balanced_distill', '--profile', str(profile), '--out', str(root), '--seed', '17', '--mode', 'pipeline'], stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        while p.poll() is None:
            save(root / 'status.json', dict(state='running', scheduler_pid=os.getpid(), running={'17': p.pid}, failures={}, source_commit=commit, timestamp=time.time()))
            time.sleep(5)
        failures = {} if not p.returncode else {'17': dict(returncode=p.returncode, log=str(root / 'seed17_pipeline.log'))}
        save(root / 'status.json', dict(state='failed' if failures else 'complete', scheduler_pid=os.getpid(), running={}, failures=failures, source_commit=commit, seed17_pilot_only=True))
        if failures:
            raise RuntimeError(f'N25 pipeline failed: {failures}')
    except BaseException as error:
        save(root / 'status.json', dict(state='failed', scheduler_pid=os.getpid(), running={'17': p.pid} if p and p.poll() is None else {}, error=repr(error), source_commit=commit))
        raise
    finally:
        if p and p.poll() is None:
            os.killpg(p.pid, signal.SIGTERM)
            try:
                p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()
        save(root / 'release_check.json', dict(child_pids=[p.pid] if p else [], children_reaped=p is None or p.poll() is not None, authorized_gpu=1, shared_gpu=True, foreign_processes_untouched=True, timestamp=time.time()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seed', type=int, choices=[17], default=17)
    parser.add_argument('--mode', choices=['pipeline', 'prepare', 'calibrate', 'worker'])
    a = parser.parse_args()
    c = json.loads(a.profile.read_text())
    c['_profile'] = str(a.profile)
    validate(c)
    bind_source_environment()
    def interrupt(signum, frame):
        raise KeyboardInterrupt(str(signum))
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    if a.mode == 'pipeline':
        pipeline(c, a.out, a.profile)
        return
    if not a.mode:
        supervise(c, a.out, a.profile)
        return
    v = config_for_seed(c, a.seed, a.out, prepared=a.mode != 'prepare')
    if a.mode == 'prepare':
        shared.prepare(v, a.out, a.profile)
    elif a.mode == 'calibrate':
        out = Path(v['calibration_run'])
        kd.calibrate(v, out, a.profile, Auxiliary)
        receipt = json.loads((out / 'receipt.json').read_text())
        for batch in receipt['batches']:
            batch['split_origins'] = ['train' if i < 3496 else 'test' for i in batch['indices']]
        indices = {i for batch in receipt['batches'] for i in batch['indices']}
        assert len(indices) == 32 and sum(i < 3496 for i in indices) == 16
        receipt.update(seed=17, model_experts=25, model_top_k=25, expected_parameter_count=5634176,
            unique_calibration_samples=32, unique_train_samples=16,
            unique_original_train_samples=16, unique_test_samples=16)
        save(out / 'receipt.json', receipt)
    else:
        base.worker(v, a.out / 'student_s17', a.profile, 'source_balanced', Auxiliary(v))


if __name__ == '__main__':
    main()
