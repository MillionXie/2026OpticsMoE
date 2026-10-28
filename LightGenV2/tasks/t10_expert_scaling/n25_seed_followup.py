"""Prepared, evidence-gated fixed N25/k25 follow-ups for own seeds27 and37.

This entry cannot start without a passed independent seed17 fixed-check audit
showing a genuine correct-count gain above598. Seed27 may start on shared GPU3
while the pilot finishes; seed37 waits for the complete pilot to release GPU1.
No teacher retraining, optical-architecture change, or automatic extension.
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


PILOT_COMMIT = '85d5524b5b205bee5b6d279a2940ce07b6c1a048'
PILOT_PT = '274a06192417a9226abd21074491a58995ebef95e08d01a5cb1dbcd1e1e3fd8e'
SOURCE_SHAS = {
    '27': 'bc0ee540681cb1d6faa8c4d859a598a3188a5c7ee561ff8a1ae7a39a4abacf98',
    '37': '4f4e1b04f7091beff187b5840524794f273d462b5fdef1348621c34cc8729aa1'}
BASELINES = {'27': [3059, 578, 587], '37': [3045, 577, 586]}
GPU_UUIDS = {'1': 'GPU-e8837b85-d55b-8e81-aaa5-ec1ac326932d',
             '3': 'GPU-4d8bfdb9-8777-05a6-3811-ab18ff4eadfd'}


def validate(c):
    assert c['seeds'] == [27, 37] and c['seed_gpu'] == {'27': 3, '37': 1}
    assert c['model_experts'] == 25 and c['expected_parameter_count'] == 5634176
    assert c['gpus'] == GPU_UUIDS and c['allow_shared_gpu'] is True and c['minimum_free_memory_mib'] == 6000
    assert c['rounds'] == 16 and c['check_rounds'] == [4, 8, 12, 16]
    assert c['batch'] == 16 and c['microbatch'] == 2 and c['samples_per_class'] == 128
    assert c['schedule_type'] == 'distill' and c['temperature'] == 2 and c['parent_weight'] == .5
    assert c['fixed_lambda_kd'] == .3203496288685833
    assert c['gradient_target_ratio'] == .5 and c['minimum_weighted_gradient_ratio'] == .05
    assert c['peak_lr'] == .0003 and c['end_lr'] == .00003 and c['ema_decay'] == .95
    assert c['explicit_test_training_authorization'] and c['test_images_enter_student_gradient']
    assert c['automatic_extension'] is False and c['automatic_additional_seeds'] is False
    assert c['requires_pilot_evidence'] is True and c['profile_state'] == 'prepared_only_until_passed_pilot_evidence'
    assert c['pilot_source_commit'] == PILOT_COMMIT and c['pilot_original_checkpoint_sha256'] == PILOT_PT
    assert c['pilot_baseline_correct'] == 598 and c['pilot_minimum_new_candidate_correct'] == 599
    assert c['pilot_wait_timeout_seconds'] == 7200 and c['pilot_wait_poll_seconds'] == 10
    assert c['teacher_cache_source_commit'] == '654183a3c16bd3cfef329a132fc15478e53f6bdb'
    assert c['locked_teacher_manifest_sha256'] == '47c7add1fd82c6aa05c4595cd885ee5f80c5eadd98fe3892b14236ebf1d8af61'
    for seed, epoch in [(27, 89), (37, 90)]:
        row = c['source_runs'][str(seed)]
        assert row['sha256'] == SOURCE_SHAS[str(seed)] and row['actual_epoch'] == epoch
        assert Path(row['run']).name == f'N25_k25_s{seed}'
        assert [row['baseline_original_train_correct'], row['baseline_val_correct'], row['baseline_test_correct']] == BASELINES[str(seed)]


def bind_source_environment():
    commit = os.environ['T10_N25_SEED_FOLLOWUP_SOURCE_COMMIT']
    assert re.fullmatch('[0-9a-f]{40}', commit)
    for name in ['T10_SEED_FOLLOWUP_SOURCE_COMMIT', 'T10_REHEAT_SOURCE_COMMIT', 'T10_DISTILL_SOURCE_COMMIT']:
        if name in os.environ:
            assert os.environ[name] == commit, name
        os.environ[name] = commit
    return commit


def pilot_evidence_record(audit, c):
    """Require a locked new fixed-check student, never a baseline/NLL-only gain."""
    validate(c)
    assert audit['passed'] and audit['phase'] in ['round4', 'round8', 'final']
    assert audit['seed'] == 17 and audit['experts'] == audit['top_k'] == 25 and audit['parameter_count'] == 5634176
    assert audit['source_commit'] == PILOT_COMMIT
    for flag in ['all_manifest_SHA_verified', 'source20_Git_object_SHA_verified',
        'complete_teacher_logits_softmax_and_parent8views_verified',
        'calibration_frozen_lambda_actual_ratio_and_restored_Adam_verified',
        'actual_sampling_source512_each_class64_and_7union2TEST_coverage_verified',
        'stored_fixed_TEST_752_argmax_confusion_NLL_pairing_verified']:
        assert audit[flag], flag
    preparation = audit['preparation']
    assert preparation['source_checkpoint_sha256'] == PILOT_PT
    assert preparation['source_test_fit_correct'] == 598 and preparation['fixed_lambda'] == c['fixed_lambda_kd']
    student = audit['student']
    best = student['cpu_best']
    selected = best['selected_state']
    assert student['baseline_test_fit_correct'] == 598 and student['selected_test_fit_correct'] > 598
    assert best['parameter_count'] == 5634176 and selected['origin'] == 'adaptation'
    assert selected['round'] in [4, 8, 12, 16] and selected['round'] <= student['completed_rounds']
    assert selected['selected'] and selected['test_correct'] == student['selected_test_fit_correct']
    assert selected['test_correct'] > 598 and selected['state_field'] in ['model', 'live_model']
    assert best['model_tensor_sha256'] == selected['model_tensor_sha256']
    assert best['prediction_sha256'] == selected['predictions_sha256']
    origin = best['export_origin']
    assert origin['state_field'] == selected['state_field'] and origin['source_checkpoint_sha256'] == selected['checkpoint_sha256']
    matches = [row for row in student['checks'] if row['round'] == selected['round'] and row['field'] == selected['state_field']]
    assert len(matches) == 1
    checked = matches[0]
    assert checked['selected_at_check'] and checked['test_fit_correct'] == selected['test_correct']
    assert checked['model_tensor_sha256'] == selected['model_tensor_sha256']
    assert checked['source_last_checkpoint_sha256'] == selected['checkpoint_sha256']
    return dict(pilot_seed=17, source_commit=PILOT_COMMIT, original_checkpoint_sha256=PILOT_PT,
        baseline_test_fit_correct=598, phase=audit['phase'], selected_round=selected['round'],
        state_field=selected['state_field'], test_fit_correct=selected['test_correct'],
        best_checkpoint_sha256=best['checkpoint_sha256'], model_tensor_sha256=best['model_tensor_sha256'],
        source_last_checkpoint_sha256=selected['checkpoint_sha256'], predictions=selected['predictions'],
        predictions_sha256=selected['predictions_sha256'],
        gate='Fixed-check genuine correct-count improvement, not fallback or NLL-only tie.')


def validate_pilot_evidence(path, c):
    import numpy as np
    audit = json.loads(path.read_text())
    record = pilot_evidence_record(audit, c)
    root = Path(c['pilot_run'])
    plan = json.loads((root / 'plan.json').read_text())
    assert plan['source_commit'] == PILOT_COMMIT and len(plan['source_files']) == 20
    assert plan['configuration']['model_experts'] == 25 and plan['configuration']['expected_parameter_count'] == 5634176
    assert plan['configuration']['source_runs']['17']['sha256'] == PILOT_PT
    for name, value in plan['source_files'].items():
        assert sha(Path(name)) == value, 'Pilot production lock changed: ' + name
    profile_names = [name for name in plan['source_files'] if name.endswith('/configs/kather_n25_balanced_distill_20261028.json')]
    assert len(profile_names) == 1
    record['pilot_profile_sha256'] = plan['source_files'][profile_names[0]]
    record['pilot_source20_files'] = plan['source_files']
    assert json.loads((root / 'reference_s17/result.json').read_text())['test_correct'] == 598
    worker = root / 'student_s17'
    decisions = [json.loads(line) for line in (worker / 'development_decisions.jsonl').read_text().splitlines()]
    selected = next(row for row in decisions if row['round'] == record['selected_round'] and row['state_field'] == record['state_field'])
    assert selected['selected'] and selected['test_correct'] == record['test_fit_correct']
    assert selected['model_tensor_sha256'] == record['model_tensor_sha256']
    assert selected['checkpoint_sha256'] == record['source_last_checkpoint_sha256']
    assert selected['predictions'] == record['predictions'] and selected['predictions_sha256'] == record['predictions_sha256']
    locks = [json.loads(line) for line in (worker / 'development_locks.jsonl').read_text().splitlines()]
    lock = next(row for row in locks if row['round'] == record['selected_round'] and row['state_field'] == record['state_field'])
    assert all(lock[key] == selected[key] for key in lock)
    prediction = worker / record['predictions']
    assert sha(prediction) == record['predictions_sha256']
    with np.load(prediction, allow_pickle=False) as stored, np.load(root / 'reference_s17/selected_test_predictions.npz', allow_pickle=False) as baseline:
        assert np.array_equal(stored['ids'], baseline['ids']) and np.array_equal(stored['labels'], baseline['labels'])
        values = stored['scores']
        assert values.shape == (752, 8) and np.isfinite(values).all() and np.max(abs(values.sum(1) - 1)) < 1e-5
        assert np.array_equal(stored['predictions'], values.argmax(1))
        assert int((values.argmax(1) == stored['labels']).sum()) == record['test_fit_correct'] > 598
    record.update(evidence_path=str(path), evidence_sha256=sha(path))
    return record


def config_for_seed(c, seed, root, prepared=True):
    validate(c)
    assert seed in c['seeds']
    value = copy.deepcopy(c)
    value['seed'] = seed
    value['recipes'] = ['source_balanced']
    value['recipe_gpu'] = {'source_balanced': c['seed_gpu'][str(seed)]}
    original = c['source_runs'][str(seed)]
    reference = root / f'reference_s{seed}'
    row = dict(original, run=str(reference)) if prepared else copy.deepcopy(original)
    value['sources'] = {'student': row, 'source_balanced': row}
    value['original_source'] = copy.deepcopy(original)
    value['incumbent_run'] = str(reference)
    value['incumbent_sha256'] = value['parent_checkpoint_sha256'] = original['sha256']
    value['incumbent_result_name'] = 'result.json'
    value['incumbent_result_is_selected'] = True
    value['parent_cache'] = str(root / f'parent_s{seed}')
    value['parent_cache_source_commit'] = os.environ['T10_N25_SEED_FOLLOWUP_SOURCE_COMMIT']
    value['calibration_run'] = str(root / f'calibration_s{seed}')
    if prepared:
        receipt = json.loads((reference / 'preparation_receipt.json').read_text())
        assert receipt['seed'] == seed and receipt['source_checkpoint_sha256'] == original['sha256']
        assert receipt['source_commit'] == os.environ['T10_N25_SEED_FOLLOWUP_SOURCE_COMMIT']
        assert sha(Path(value['parent_cache']) / 'receipt.json') == receipt['parent_receipt_sha256']
        assert receipt['initial_test_correct'] == original['baseline_test_correct']
        assert receipt['initial_val_correct'] == original['baseline_val_correct']
        value['incumbent_predictions_sha256'] = receipt['predictions_sha256']
    return value


class Auxiliary(shared.Auxiliary):
    def __init__(self, c, calibrated=True):
        super().__init__(c, calibrated)
        self.identity.update(model_experts=25, model_top_k=25, expected_parameter_count=5634176,
            method_development_seed17_finalized=False, n25_pilot_gain_gate=c['_pilot_evidence_summary'],
            transfer_seed=c['seed'], role='Fixed N25 recipe transfer from this seed own original checkpoint; TEST-exposed fit.')

    def diagnostics(self, model, split, step, best_correct):
        result = super().diagnostics(model, split, step, best_correct)
        result['note'] = 'Fixed16 N25/k25 follow-up from this seed own original baseline. Seed17 gain admitted the recipe, not this candidate score; no target extension or copied parent.'
        return result


def own_task_command(command, root):
    command = command.replace('\\', '/')
    return (__package__ + '.') in command or '/t10_expert_scaling/' in command or str(root).replace('\\', '/') in command


def gpu_state(c, seed):
    index = c['seed_gpu'][str(seed)]
    uuid = subprocess.check_output(['nvidia-smi', '-i', str(index), '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
    assert uuid == c['gpus'][str(index)]
    free = int(subprocess.check_output(['nvidia-smi', '-i', str(index), '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).strip())
    raw = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid', '--format=csv,noheader,nounits'], text=True)
    processes = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        pid, device = [word.strip() for word in line.split(',', 1)]
        if device != uuid:
            continue
        try:
            command = Path('/proc', pid, 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
        except FileNotFoundError:
            continue
        processes.append(dict(pid=int(pid), gpu_uuid=device, command=command))
    return dict(index=index, uuid=uuid, free_memory_mib=free, processes=processes)


def assert_shared_gpu_policy(c, seed, state, root, require_capacity=True):
    assert state['uuid'] == c['gpus'][str(c['seed_gpu'][str(seed)])]
    if require_capacity:
        assert state['free_memory_mib'] >= c['minimum_free_memory_mib'], 'Insufficient memory; do not stop foreign processes.'
    assert not any(own_task_command(row['command'], root) for row in state['processes']), 'Another T10 GPU process remains on this card; do not duplicate it.'


def check_gpu(c, seed, root, require_capacity=True):
    state = gpu_state(c, seed)
    assert_shared_gpu_policy(c, seed, state, root, require_capacity)
    return state['uuid']


def pilot_release_ready(status, worker, progress, release, actual_own_pids_alive, own_gpu_processes):
    if status.get('state') == 'failed' or worker.get('state') == 'failed':
        raise RuntimeError('N25 pilot failed; preserve it and do not start seed37 automatically.')
    return (status.get('state') == 'complete' and not status.get('running') and not status.get('failures')
        and worker.get('state') == 'complete' and progress.get('state') == 'complete'
        and release.get('children_reaped') is True and not actual_own_pids_alive and not own_gpu_processes)


def is_original_pilot_command(command, root):
    # The original launch may have used a repository-relative --out argument.
    return (__package__ + '.n25_balanced_distill') in command and root.name in command.replace('\\', '/')


def pilot_pid_check(root):
    paths = ['plan.json', 'status.json', 'release_check.json', 'seed17_progress.json',
        'reference_s17/status.json', 'parent_s17/progress.json', 'calibration_s17/receipt.json',
        'student_s17/metadata.json', 'student_s17/status.json', 'student_s17/batch_progress.json']
    pids = set()
    for name in paths:
        path = root / name
        if not path.exists():
            continue
        value = json.loads(path.read_text())
        for key in ['pid', 'scheduler_pid', 'pipeline_pid', 'child_pid']:
            if value.get(key):
                pids.add(int(value[key]))
        pids.update(int(pid) for pid in value.get('child_pids', []))
        pids.update(int(pid) for pid in value.get('running', {}).values())
    actual = []
    for pid in sorted(pids):
        path = Path('/proc', str(pid), 'cmdline')
        try:
            command = path.read_bytes().replace(b'\0', b' ').decode(errors='replace')
        except FileNotFoundError:
            continue
        if is_original_pilot_command(command, root):
            actual.append(dict(pid=pid, command=command))
    return sorted(pids), actual


def wait_pilot_release(c, root):
    pilot = Path(c['pilot_run'])
    started = time.monotonic()
    while True:
        def read_or_empty(relative):
            path = pilot / relative
            return json.loads(path.read_text()) if path.exists() else {}
        status = read_or_empty('status.json')
        worker = read_or_empty('student_s17/status.json')
        progress = read_or_empty('seed17_progress.json')
        release = read_or_empty('release_check.json')
        known, alive = pilot_pid_check(pilot)
        state = gpu_state(c, 37)
        own = [row for row in state['processes'] if own_task_command(row['command'], root)]
        ready = pilot_release_ready(status, worker, progress, release, alive, own)
        if ready and state['free_memory_mib'] >= c['minimum_free_memory_mib']:
            save(root / 'pilot_gpu1_release_gate.json', dict(passed=True, pilot_run=str(pilot), pilot_state=status,
                known_pilot_pids=known, original_pilot_process_identities_absent=True,
                no_T10_compute_process_on_GPU1=True, free_memory_mib=state['free_memory_mib'],
                foreign_processes_untouched=True, timestamp=time.time()))
            return
        save(root / 'seed37_progress.json', dict(seed=37, state='waiting_for_pilot_release', pipeline_pid=os.getpid(),
            child_pid=None, gpu=1, pilot_state=status.get('state'), actual_pilot_pids_still_alive=alive,
            T10_compute_processes=own, free_memory_mib=state['free_memory_mib'], timestamp=time.time()))
        if time.monotonic() - started > c['pilot_wait_timeout_seconds']:
            raise RuntimeError('Pilot GPU1 release wait exceeded its finite bound; preserve artifacts, no automatic retry.')
        time.sleep(c['pilot_wait_poll_seconds'])


def pipeline(c, seed, root, profile, evidence_path):
    if seed == 37:
        wait_pilot_release(c, root)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=check_gpu(c, seed, root), CUDA_DEVICE_ORDER='PCI_BUS_ID', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    for mode in ['prepare', 'calibrate', 'worker']:
        check_gpu(c, seed, root)
        with (root / f'seed{seed}_{mode}.log').open('x') as stream:
            process = subprocess.Popen([sys.executable, '-u', '-m', __package__ + '.n25_seed_followup',
                '--profile', str(profile), '--out', str(root), '--pilot-evidence', str(evidence_path), '--seed', str(seed), '--mode', mode],
                env=env, stdout=stream, stderr=subprocess.STDOUT)
            save(root / f'seed{seed}_progress.json', dict(seed=seed, state=mode, pipeline_pid=os.getpid(),
                child_pid=process.pid, gpu=c['seed_gpu'][str(seed)], shared_gpu=True,
                source_commit=os.environ['T10_N25_SEED_FOLLOWUP_SOURCE_COMMIT'], timestamp=time.time()))
            try:
                process.wait()
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=30)
        if process.returncode:
            raise RuntimeError(f'seed{seed} {mode} failed with {process.returncode}; preserve artifacts, no automatic retry')
    check_gpu(c, seed, root, require_capacity=False)
    save(root / f'seed{seed}_progress.json', dict(seed=seed, state='complete', pipeline_pid=os.getpid(),
        child_pid=None, gpu=c['seed_gpu'][str(seed)], shared_gpu=True, timestamp=time.time()))


def supervise(c, root, profile, evidence_path):
    validate(c)
    evidence = validate_pilot_evidence(evidence_path, c)
    commit, hashes = shared.published_sources(profile, c)
    check_gpu(c, 27, root)
    for row in c['source_runs'].values():
        assert sha(Path(row['run']) / 'best_checkpoint.pt') == row['sha256']
    root.mkdir(parents=True, exist_ok=False)
    save(root / 'pilot_evidence_lock.json', evidence)
    save(root / 'plan.json', dict(configuration=c, source_commit=commit, source_files=hashes,
        pilot_evidence=evidence, scheduler_pid=os.getpid(), host_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()))
    processes, pids, failures = {}, [], {}
    try:
        for seed in c['seeds']:
            with (root / f'seed{seed}_pipeline.log').open('x') as stream:
                process = subprocess.Popen([sys.executable, '-u', '-m', __package__ + '.n25_seed_followup',
                    '--profile', str(profile), '--out', str(root), '--pilot-evidence', str(evidence_path), '--seed', str(seed), '--mode', 'pipeline'],
                    stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            processes[seed] = process
            pids.append(process.pid)
        while processes:
            save(root / 'status.json', dict(state='running', scheduler_pid=os.getpid(), running={str(seed): process.pid for seed, process in processes.items()},
                failures=failures, source_commit=commit, timestamp=time.time()))
            for seed, process in list(processes.items()):
                if process.poll() is not None:
                    if process.returncode:
                        failures[str(seed)] = dict(returncode=process.returncode, log=str(root / f'seed{seed}_pipeline.log'))
                    del processes[seed]
            if processes:
                time.sleep(5)
        save(root / 'status.json', dict(state='failed' if failures else 'complete', scheduler_pid=os.getpid(),
            running={}, failures=failures, source_commit=commit, no_automatic_extensions=True))
    except BaseException as error:
        save(root / 'status.json', dict(state='failed', scheduler_pid=os.getpid(),
            running={str(seed): process.pid for seed, process in processes.items()}, failures=failures, error=repr(error), source_commit=commit))
        raise
    finally:
        for process in processes.values():
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        for process in processes.values():
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        save(root / 'release_check.json', dict(child_pids=pids, children_reaped=True,
            authorized_seed_gpus=c['seed_gpu'], foreign_processes_untouched=True, timestamp=time.time()))
    if failures:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--pilot-evidence', type=Path, required=True)
    parser.add_argument('--seed', type=int, choices=[27, 37])
    parser.add_argument('--mode', choices=['pipeline', 'prepare', 'calibrate', 'worker'])
    arguments = parser.parse_args()
    if arguments.mode and arguments.seed is None:
        parser.error('--seed is required for a child mode')
    c = json.loads(arguments.profile.read_text())
    c['_profile'] = str(arguments.profile)
    validate(c)
    bind_source_environment()
    def interrupt(signum, frame):
        raise KeyboardInterrupt(str(signum))
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    if not arguments.mode:
        supervise(c, arguments.out, arguments.profile, arguments.pilot_evidence)
        return
    evidence = validate_pilot_evidence(arguments.pilot_evidence, c)
    locked = json.loads((arguments.out / 'pilot_evidence_lock.json').read_text())
    assert evidence == locked, 'Pilot evidence changed after admission.'
    c['_pilot_evidence_summary'] = evidence
    if arguments.mode == 'pipeline':
        pipeline(c, arguments.seed, arguments.out, arguments.profile, arguments.pilot_evidence)
        return
    value = config_for_seed(c, arguments.seed, arguments.out, prepared=arguments.mode != 'prepare')
    if arguments.mode == 'prepare':
        shared.prepare(value, arguments.out, arguments.profile)
        result = json.loads((arguments.out / f'reference_s{arguments.seed}/result.json').read_text())
        own_source = c['source_runs'][str(arguments.seed)]
        assert round(result['train']['accuracy'] * 3496) == own_source['baseline_original_train_correct']
        assert round(result['val']['accuracy'] * 752) == own_source['baseline_val_correct']
        assert result['test_correct'] == own_source['baseline_test_correct']
    elif arguments.mode == 'calibrate':
        output = Path(value['calibration_run'])
        kd.calibrate(value, output, arguments.profile, Auxiliary)
        receipt = json.loads((output / 'receipt.json').read_text())
        indices = {i for batch in receipt['batches'] for i in batch['indices']}
        assert len(indices) == 32 and sum(i < 3496 for i in indices) == 16
        for batch in receipt['batches']:
            batch['split_origins'] = ['train' if i < 3496 else 'test' for i in batch['indices']]
        receipt.update(seed=arguments.seed, model_experts=25, model_top_k=25, expected_parameter_count=5634176,
            unique_calibration_samples=32, unique_train_samples=16, unique_original_train_samples=16, unique_test_samples=16)
        save(output / 'receipt.json', receipt)
    else:
        base.worker(value, arguments.out / f'student_s{arguments.seed}', arguments.profile, 'source_balanced', Auxiliary(value))


if __name__ == '__main__':
    main()
