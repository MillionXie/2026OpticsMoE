"""Run the twelve matched continuations on their original lab GPUs to a shared VAL stopping point."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .stability import assess_cohort, load_protocol


TASK = Path(__file__).resolve().parent
PACKAGE = __package__


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def write(path, data):
    scratch = path.with_suffix('.tmp')
    scratch.write_text(json.dumps(data, indent=2) + '\n', encoding='utf8')
    scratch.replace(path)


def available_memory():
    output = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=uuid,memory.free', '--format=csv,noheader,nounits'], text=True)
    return {uuid.strip(): int(free) for uuid, free in (line.split(',') for line in output.splitlines())}


def run_stage(original, out, groups, target):
    waiting = []
    for seed, (uuid, names) in groups.items():
        for name in names:
            status = out / name / 'status.json'
            if status.exists():
                previous = read(status)
                if previous['state'] == 'complete' and previous['epoch'] == target:
                    continue
            waiting.append((name, uuid))
    active = {}
    requirements = {}
    failures = []
    while waiting or active:
        for name, (process, uuid, logfile, before) in list(active.items()):
            code = process.poll()
            if code is None:
                continue
            del active[name]
            if code:
                tail = logfile.read_text(encoding='utf8', errors='replace')[-6000:]
                # On shared cards, retry an interrupted allocation only when additional memory is available.
                if 'out of memory' in tail.lower() and name not in requirements:
                    requirements[name] = max(4200, min(before + 1000, 8500))
                    waiting.append((name, uuid))
                    print(f'{name} allocation failed. Retry requires {requirements[name]} MiB free.', flush=True)
                else:
                    failures.append((name, code, tail[-1200:]))
            else:
                status = read(out / name / 'status.json')
                if status['state'] != 'complete' or status['epoch'] != target:
                    failures.append((name, code, 'Worker exited without a completed target'))
        if failures and not active:
            raise RuntimeError(str(failures))
        if waiting and not failures:
            free = available_memory()
            for name, uuid in list(waiting):
                count = sum(job[1] == uuid for job in active.values())
                if count >= 2 or free[uuid] < requirements.get(name, 2350):
                    continue
                log = out / f'{name}_to_E{target}.log'
                env = {**os.environ, 'CUDA_VISIBLE_DEVICES': uuid, 'OMP_NUM_THREADS': '4', 'PYTHONUNBUFFERED': '1'}
                command = [sys.executable, '-m', f'{PACKAGE}.continue_matched', '--run', str(original / name),
                           '--out', str(out / name), '--target-epoch', str(target)]
                with log.open('a', encoding='utf8') as handle:
                    process = subprocess.Popen(command, env=env, stdout=handle, stderr=subprocess.STDOUT)
                active[name] = (process, uuid, log, free[uuid])
                waiting.remove((name, uuid))
                # Leave time for the new CUDA context to be reflected in nvidia-smi.
                free[uuid] = 0
                print(f'E{target} started {name} pid={process.pid} gpu={uuid}', flush=True)
        if waiting or active:
            time.sleep(12)


def run(out):
    protocol = load_protocol()
    original = TASK / 'runs' / 'simulation' / protocol['parent_cohort']
    groups = {}
    for seed in protocol['seeds']:
        modes = protocol['modes'] if seed == 37 else list(reversed(protocol['modes']))
        names = [f'{dataset}_s{seed}_{mode}' for mode in modes for dataset in ('blood', 'kather')]
        uuids = {read(original / name / 'metadata.json')['gpu_uuid'] for name in names}
        if len(uuids) != 1:
            raise ValueError(f'Expected one original GPU for seed {seed}: {uuids}')
        groups[seed] = (uuids.pop(), names)
    if len({uuid for uuid, _ in groups.values()}) != len(groups):
        raise ValueError('Seeds must use their distinct original authorized GPU devices')
    target = protocol['min_epochs']
    while True:
        report = out / f'VAL_decision_E{target}.json'
        if not report.exists():
            write(out / 'cohort_status.json', {'state': 'running', 'target_epoch': target, 'pid': os.getpid()})
            run_stage(original, out, groups, target)
            histories = {name: read(out / name / 'history.json') for _, names in groups.values() for name in names}
            decision = assess_cohort(histories, protocol, epoch=target)
            write(report, decision)
        else:
            decision = read(report)
        print(f"E{target} {decision['decision']}: {decision['runs_without_plateau']}", flush=True)
        if decision['decision'] != 'continue':
            write(out / 'cohort_status.json', {'state': 'complete', 'epoch': target, 'pid': os.getpid(),
                                               'decision': decision['decision']})
            return
        target = decision['next_epoch']


def main():
    import fcntl  # The lab controller runs on Linux.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    out = parser.parse_args().out.resolve()
    base = TASK / 'runs' / 'simulation'
    if out.parent != base or out.name == load_protocol()['parent_cohort']:
        parser.error('Choose a distinct T24 simulation cohort directory')
    out.mkdir(parents=True, exist_ok=True)
    with (out / '.cohort.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            run(out)
        except Exception as exc:
            write(out / 'cohort_status.json', {'state': 'failed', 'pid': os.getpid(), 'error': str(exc)})
            raise


if __name__ == '__main__':
    main()
