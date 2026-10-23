"""Serialize each GPU's three seeds; one suite lock, at most three idle GPUs."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .train_easy100 import TASK, ROOT, admit_gpu, git, require_published_source, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu-uuids', nargs='+', required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--suite-id', required=True)
    args = parser.parse_args()
    if not 1 <= len(args.gpu_uuids) <= 3 or len(set(args.gpu_uuids)) != len(args.gpu_uuids):
        raise ValueError('One to three distinct GPU UUIDs required')
    if Path(args.suite_id).name != args.suite_id or args.suite_id in {'.', '..'}:
        raise ValueError('suite-id must be a single directory name')
    import fcntl  # Linux-only coordinator, no CUDA initialization.
    require_published_source()
    base = TASK / 'runs/simulation'
    base.mkdir(parents=True, exist_ok=True)
    lock = (base / '.t19_gpu_suite.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    suite = base / args.suite_id
    suite.mkdir(exist_ok=False)
    source_commit = git('rev-parse', 'HEAD')
    source_tree = git('rev-parse', 'HEAD:' + TASK.relative_to(ROOT).as_posix())
    # Confirm BOTH real-GPU contract checks succeeded, never use smoke metrics
    # as formal quality results. They exercise train/eval/save/reload lifecycle.
    for short in ('i2t', 't2i'):
        smoke = TASK / f'runs/smoke/easy100_{short}_s42_20261023_v2/result.json'
        evidence = json.loads(smoke.read_text())
        if evidence.get('status') != 'complete':
            raise RuntimeError(f'{short} GPU smoke has not completed')
    jobs = []
    for seed in (42, 43, 44):
        for direction in ('image_to_text', 'text_to_image'):
            name = f'easy100_{direction}_s{seed}_{args.suite_id}'
            jobs.append({'name': name, 'direction': direction, 'seed': seed, 'status': 'queued'})
    running, children = {}, []
    report = {'status': 'running', 'source_commit': source_commit, 'source_tree': source_tree, 'pid': os.getpid(),
              'gpu_uuids': args.gpu_uuids, 'max_gpus': 3, 'jobs': jobs,
              'started_unix': time.time(), 'scope': 'two_of_nine_tasks_three_seeds_each'}

    def save():
        write_json(suite / 'queue_status.json', report)

    def interrupt(signum, frame):
        raise KeyboardInterrupt(f'Signal {signum}')

    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    try:
        while any(job['status'] == 'queued' for job in jobs) or running:
            for uuid, (process, job, log) in list(running.items()):
                code = process.poll()
                if code is None:
                    continue
                log.close()
                job.update(status='complete' if code == 0 else 'failed', exit_code=code,
                           finished_unix=time.time())
                del running[uuid]
                if code:
                    raise RuntimeError(f'Worker {job["name"]} failed: {code}')
            if git('rev-parse', 'HEAD:' + TASK.relative_to(ROOT).as_posix()) != source_tree:
                raise RuntimeError('T19 source tree changed during suite; refusing mixed-source queue')
            for uuid in args.gpu_uuids:
                if uuid in running:
                    continue
                job = next((j for j in jobs if j['status'] == 'queued'), None)
                if job is None:
                    break
                old = os.environ.get('CUDA_VISIBLE_DEVICES')
                os.environ['CUDA_VISIBLE_DEVICES'] = uuid
                try:
                    admit_gpu(uuid)
                except RuntimeError as exc:
                    report.setdefault('waiting_for_idle', {})[uuid] = str(exc)
                    continue
                finally:
                    if old is None:
                        os.environ.pop('CUDA_VISIBLE_DEVICES', None)
                    else:
                        os.environ['CUDA_VISIBLE_DEVICES'] = old
                report.setdefault('waiting_for_idle', {}).pop(uuid, None)
                command = [sys.executable, '-m', 'LightGenV2.tasks.t19_resnet20_baselines.train_easy100',
                    '--direction', job['direction'], '--seed', str(job['seed']),
                    '--data-root', str(args.data_root.resolve()), '--run-dir', str(base / job['name']),
                    '--gpu-uuid', uuid]
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=uuid, OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1')
                log = (suite / (job['name'] + '.log')).open('x')
                process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                    stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
                job.update(status='running', pid=process.pid, gpu_uuid=uuid,
                           command=command, started_unix=time.time())
                running[uuid] = (process, job, log)
                children.append(process)
            save()
            if running or any(j['status'] == 'queued' for j in jobs):
                time.sleep(5)
        report['status'] = 'complete'
    except BaseException as exc:
        report.update(status='failed', error=repr(exc))
        for process, job, log in running.values():
            # Only process groups created by this coordinator are signalled.
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
            job['status'] = 'interrupted'
        for process, job, log in running.values():
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
            log.close()
        raise
    finally:
        own_pids = {p.pid for p in children}
        apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                       '--format=csv,noheader,nounits'], text=True)
        remaining = []
        for line in apps.splitlines():
            fields = line.split(',')
            if len(fields) != 2:
                continue
            pid = int(fields[1].strip())
            try:
                owned = pid in own_pids or os.getpgid(pid) in own_pids
            except ProcessLookupError:
                owned = False
            if owned:
                remaining.append(line)
        report.update(finished_unix=time.time(), remaining_owned_gpu_processes=remaining,
                      owned_gpu_processes_released=not remaining,
                      gpu_inventory_after_exit=subprocess.check_output(['nvidia-smi',
                          '--query-gpu=uuid,memory.used,utilization.gpu', '--format=csv,noheader'], text=True))
        save()
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


if __name__ == '__main__':
    main()
