"""Run locked test evaluations sequentially while temporarily pausing training workers."""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def save(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(tmp, path)


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--selection-lock', type=Path, required=True)
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--test-source', type=Path)
    ap.add_argument('--pause-pids', type=int, nargs='*', default=[])
    ap.add_argument('--status', type=Path, required=True)
    ap.add_argument('--force', action='store_true', help='Evaluate every run even if test_result.json exists.')
    args = ap.parse_args()
    lock = json.loads(args.selection_lock.read_text())
    pending = list(lock['runs']) if args.force else [
        x for x in lock['runs'] if not (Path(x['run_dir'])/'test_result.json').exists()]
    paused = []
    completed = len(lock['runs']) - len(pending)
    failed = []

    def status(state, current=None):
        save(args.status, dict(state=state, total=len(lock['runs']), complete=completed,
                               remaining=len(pending), current=current, paused_training_pids=paused,
                               failed=failed))

    try:
        for pid in args.pause_pids:
            if alive(pid):
                os.kill(pid, signal.SIGSTOP)
                paused.append(pid)
        status('testing')
        while pending:
            item = pending.pop(0)
            status('testing', item)
            cmd = [sys.executable, '-u', '-m', 'LightGenV2.tasks.t10_expert_scaling.evaluate_test',
                   '--run', item['run_dir'], '--data', str(args.data),
                   '--selection-lock', str(args.selection_lock), '--batch', '2']
            if args.test_source:
                cmd.extend(['--test-source', str(args.test_source)])
            proc = subprocess.run(cmd)
            if proc.returncode:
                failed.append(dict(run=item, exit_code=proc.returncode))
                raise RuntimeError(f'Test evaluation failed: {item}')
            completed += 1
        status('complete')
    except BaseException as error:
        status('failed')
        save(args.status.with_name(args.status.stem + '_error.json'), dict(error=repr(error), failed=failed))
        raise
    finally:
        for pid in paused:
            if alive(pid):
                os.kill(pid, signal.SIGCONT)
        save(args.status.with_name(args.status.stem + '_release.json'),
             dict(training_workers_resumed=True, resumed_pids=[p for p in paused if alive(p)],
                  completed=completed, failed=failed, timestamp=time.time()))


if __name__ == '__main__':
    main()
