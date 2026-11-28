"""Dispatch the fixed matched T25 experiment on the two designated laboratory GPUs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from LightGenV2.tasks.t24_continuous_spatial_routing.run_longrun_cohort import available_memory, read, write


TASK = Path(__file__).resolve().parent
PROFILE = TASK / 'configs' / 'matched_fiber_20261128.json'


def run_worker(stage, root, key, dataset, seed, arm, config):
    gpu = config['gpu_uuids'][str(seed)]
    out = root / key
    statefile = out / 'status.json'
    if statefile.exists() and read(statefile)['state'] == 'complete':
        print('Already complete', key, flush=True)
        return
    command = [sys.executable, '-m', f'{__package__}.train', '--dataset', dataset, '--seed', str(seed),
               '--arm', arm, '--epochs', str(config['epochs']), '--out', str(out), '--device', 'cuda']
    if stage == 'smoke':
        command.append('--smoke')
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': gpu, 'OMP_NUM_THREADS': '4', 'PYTHONUNBUFFERED': '1'}
    path = root / f'{key}.log'
    for attempt, minimum in enumerate((2800, 5600), 1):
        while available_memory()[gpu] < minimum:
            time.sleep(20)
        with path.open('a', encoding='utf8') as log:
            offset = log.tell()
            worker = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            print(f'{key} attempt={attempt} pid={worker.pid} gpu={gpu}', flush=True)
            code = worker.wait()
        if code == 0 and read(statefile)['state'] == 'complete':
            return
        with path.open(encoding='utf8', errors='replace') as log:
            log.seek(offset)
            reason = log.read()[-5000:]
        if 'out of memory' not in reason.lower() or attempt == 2:
            raise RuntimeError(f'{key} exit {code}: {reason[-2500:]}')


def run_seed(stage, root, config, seed):
    arms = config['arms'] if seed == 37 else list(reversed(config['arms']))
    jobs = [(d, a) for d in ('blood', 'kather') for a in arms]
    with ThreadPoolExecutor(max_workers=1 if stage == 'smoke' else 2) as pool:
        tasks = [pool.submit(run_worker, stage, root, f'{d}_s{seed}_{a}', d, seed, a, config) for d,a in jobs]
        for task in tasks:
            task.result()


def main():
    import fcntl
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['smoke','train'], required=True)
    args = parser.parse_args()
    config = read(PROFILE)
    stage = args.stage
    name = PROFILE.stem
    root = TASK / 'runs' / ('smoke' if stage == 'smoke' else 'simulation') / name
    root.mkdir(parents=True, exist_ok=True)
    if stage == 'train':
        passed = read(TASK/'runs'/'smoke'/name/'cohort_status.json')
        if passed['state'] != 'complete' or passed['source_commit'] != os.environ['T25_SOURCE_COMMIT']:
            raise ValueError('Require completed admissions for this training source')
    with (root / '.cohort.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status = dict(stage=stage, pid=os.getpid(), source_commit=os.environ['T25_SOURCE_COMMIT'],
                      datasets=config['datasets'], arms=config['arms'], seeds=config['seeds'], epochs=config['epochs'])
        write(root/'cohort_status.json', dict(status, state='running'))
        try:
            with ThreadPoolExecutor(max_workers=len(config['seeds'])) as pool:
                workers = [pool.submit(run_seed, stage, root, config, seed) for seed in config['seeds']]
                for worker in workers:
                    worker.result()
        except Exception as exc:
            write(root/'cohort_status.json',dict(status,state='failed',reason=str(exc)))
            raise
        write(root/'cohort_status.json',dict(status,state='complete'))


if __name__=='__main__':
    main()
