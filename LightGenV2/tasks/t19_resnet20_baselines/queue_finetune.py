"""Post-30 smoke/formal queue restricted to the user's existing four GPUs.

This uses a separate stage registry; it never replaces the original supervisor,
signals its workers, or expands the shared physical GPU whitelist.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .finetune_tasks import FINAL_EPOCHS
from .shared_idle_gpu import snapshot, validate
from .frozen_resnet18 import file_sha256
from .train_easy100 import TASK, ROOT, admit_gpu, git, require_published_source, write_json


def stage_command(job, suite, plan, uuid):
    base = TASK / 'runs' / ('smoke' if job['phase'] == 'smoke' else 'simulation')
    command = [sys.executable, '-u', '-m', 'LightGenV2.tasks.t19_resnet20_baselines.finetune_tasks',
               '--task', job['task'], '--plan', str(plan), '--run-dir', str(base / job['name']),
               '--gpu-uuid', uuid, '--queue-status-path', str(suite / 'queue_status.json'),
               '--max-physical-gpus', '4', '--max-jobs-per-gpu', '2', '--share-idle-existing-contexts']
    if job['phase'] == 'smoke':
        command += ['--smoke-batches', '4']
    else:
        command += ['--smoke-result', str(TASK / 'runs/smoke' / job['smoke_name'] / 'result.json')]
    return command


def main():
    import fcntl
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--suite', type=Path, required=True)
    parser.add_argument('--resume-queued-only', action='store_true',
                        help='Replace only a stopped supervisor whose six jobs never started')
    parser.add_argument('--retry-lsp-cleanup1', action='store_true',
                        help='One named LSP smoke retry after its documented early DataLoader cleanup failure')
    args = parser.parse_args()
    require_published_source()
    plan_path = args.plan.resolve(strict=True)
    suite = args.suite.resolve()
    if (not suite.is_relative_to((TASK / 'runs/simulation').resolve())
            or not plan_path.is_relative_to((TASK / 'configs').resolve())):
        raise ValueError('Fine-tuning queue paths must stay in this task')
    plan = json.loads(plan_path.read_text())
    if plan['max_physical_gpus'] != 4 or set(plan['candidate_gpu_indices']) != {2, 4, 5, 6}:
        raise ValueError('Cannot expand the latest four-card limit')
    if args.retry_lsp_cleanup1:
        if suite.name != 'lsp_finetune60_cleanup_retry1_20261024':
            raise ValueError('The single cleanup retry needs its own new named suite')
        original_suite = TASK / 'runs/simulation/frozen_finetune_fixed_last_20261024'
        original = json.loads((original_suite / 'queue_status.json').read_text())
        old_smoke = next(j for j in original['jobs'] if j['task'] == 'lsp' and j['phase'] == 'smoke')
        old_formal = next(j for j in original['jobs'] if j['task'] == 'lsp' and j['phase'] == 'formal')
        old_result = json.loads((TASK / 'runs/smoke' / old_smoke['name'] / 'result.json').read_text())
        old_log = (original_suite / (old_smoke['name'] + '.log')).read_text()
        if (old_smoke['status'] != 'failed' or old_formal['status'] != 'blocked_by_failed_smoke'
                or old_formal.get('pid') or old_result.get('test_evaluations') != 0
                or old_result.get('status') != 'complete'
                or 'DataLoader worker' not in old_log or '_shutdown_workers' not in old_log):
            raise ValueError('No matching completed-computation/failed-cleanup LSP attempt to retry')
    lock_name = '.t19_finetune_lsp_cleanup_retry1.lock' if args.retry_lsp_cleanup1 else '.t19_finetune_suite.lock'
    lock = (TASK / 'runs' / lock_name).open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if not args.resume_queued_only:
        suite.mkdir(parents=True, exist_ok=False)
    inventory = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid',
                                         '--format=csv,noheader,nounits'], text=True)
    uuids = {int(row.split(',')[0].strip()): row.split(',')[1].strip()
             for row in inventory.splitlines() if row.strip()}
    candidates = [uuids[index] for index in (6, 4, 5, 2)]
    if args.retry_lsp_cleanup1:
        # Prefer the unused original GPU5; after another task finishes, any of
        # the same four cards may admit this still-queued stage. Global T19
        # classification prevents sharing alongside another T19+other context.
        candidates = [uuids[index] for index in (5, 6, 4, 2)]
    jobs = []
    old = json.loads((suite / 'queue_status.json').read_text()) if args.resume_queued_only else None
    if old:
        if any(job['status'] != 'queued' or job.get('pid') for job in old['jobs']):
            raise ValueError('Queued-only takeover refuses all existing or started workers')
        jobs = old['jobs']
    else:
        for task, stop in FINAL_EPOCHS.items():
            if args.retry_lsp_cleanup1 and task != 'lsp':
                continue
            smoke_name = f'{task}_finetune{stop}_smoke_s42_20261024'
            if args.retry_lsp_cleanup1:
                smoke_name += '_cleanup1'
            for phase in ('smoke', 'formal'):
                name = smoke_name if phase == 'smoke' else f'{task}_finetune{stop}_formal_s42_20261024'
                if args.retry_lsp_cleanup1 and phase == 'formal':
                    name += '_cleanup1'
                base = TASK / 'runs' / ('smoke' if phase == 'smoke' else 'simulation')
                if (base / name).exists():
                    raise RuntimeError('Existing stage evidence found: inspect rather than restart or overwrite')
                jobs.append(dict(task=task, phase=phase, name=name, smoke_name=smoke_name, status='queued'))
    report = dict(status='running', pid=os.getpid(), jobs=jobs, gpu_uuids=candidates,
                  max_physical_gpus=4, max_jobs_per_gpu=2, git_commit=git('rev-parse', 'HEAD'),
                  profile_sha256=file_sha256(plan_path), started_unix=time.time(),
                  gpu_policy='shared_idle_stable10s_free16GiB_util50_oneT19_with_existing_contexts',
                  command=[sys.executable, '-m', __spec__.name, *sys.argv[1:]],
                  original_supervisor_and_workers_preserved=True)
    if args.retry_lsp_cleanup1:
        report['named_cleanup_retry'] = dict(attempt=1, original_smoke=old_smoke['name'],
            original_smoke_log_sha256=file_sha256(original_suite / (old_smoke['name'] + '.log')),
            training_model_loss_and_formal_worker_count_unchanged=True, additional_TEST_observations=0)
    def save():
        write_json(suite / 'queue_status.json', report)
    save()
    running = {}
    observations = {}
    try:
        while any(job['status'] in {'queued', 'running'} for job in jobs):
            for pid, (process, job, log) in list(running.items()):
                code = process.poll()
                if code is None:
                    continue
                log.close()
                del running[pid]
                base = TASK / 'runs' / ('smoke' if job['phase'] == 'smoke' else 'simulation')
                result_path = base / job['name'] / 'result.json'
                result = json.loads(result_path.read_text()) if result_path.exists() else {}
                job.update(status='complete' if code == 0 and result.get('status') == 'complete' else 'failed',
                           exit_code=code, finished_unix=time.time())
                if job['status'] == 'failed' and job['phase'] == 'smoke':
                    for formal in jobs:
                        if formal['task'] == job['task'] and formal['phase'] == 'formal':
                            formal['status'] = 'blocked_by_failed_smoke'
                save()
            for uuid in candidates:
                owned = [pid for pid, (_, job, _) in running.items() if job['gpu_uuid'] == uuid]
                if len(owned) >= 2:
                    continue
                ready = []
                for job in jobs:
                    if job['status'] != 'queued':
                        continue
                    smoke = next(j for j in jobs if j['task'] == job['task'] and j['phase'] == 'smoke')
                    if job['phase'] == 'smoke' or smoke['status'] == 'complete':
                        ready.append(job)
                if not ready:
                    continue
                try:
                    state = snapshot(uuid)
                    validate(state, owned, observations.get(uuid))
                except (RuntimeError, FileNotFoundError):
                    # Foreign work and busy cards wait; never signal it or take a CUDA context.
                    if 'state' in locals() and state['gpu_uuid'] == uuid:
                        previous = observations.get(uuid)
                        if previous is None or state['observed_monotonic'] - previous['observed_monotonic'] >= 15:
                            observations[uuid] = state
                    continue
                job = ready[0]
                command = stage_command(job, suite, plan_path, uuid)
                log = (suite / (job['name'] + '.log')).open('x')
                process = subprocess.Popen(command, cwd=ROOT,
                    env=dict(os.environ, CUDA_VISIBLE_DEVICES=uuid, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4'),
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                job.update(status='running', pid=process.pid, gpu_uuid=uuid, command=command,
                           started_unix=time.time())
                running[process.pid] = (process, job, log)
                save()
            time.sleep(5)
        report['status'] = 'complete' if all(j['status'] == 'complete' for j in jobs) else 'completed_with_failure'
    except BaseException as exc:
        # A supervisor problem must not interrupt healthy training children.
        report.update(status='supervisor_detached_workers_preserved', error=repr(exc))
        raise
    finally:
        report['last_updated_unix'] = time.time()
        save()
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


if __name__ == '__main__':
    main()
