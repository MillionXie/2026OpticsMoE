"""One preregistered L16 paired ten-epoch fit restart, never test evaluation."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
PYTHON = '/home/guest3/miniconda3/envs/xml/bin/python'
GPUS = ['GPU-e8837b85-d55b-8e81-aaa5-ec1ac326932d', 'GPU-6dcca91a-8e08-1a50-9aa6-81defeaed50b']


def read(path): return json.loads(path.read_text())


def save(path, obj):
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj, indent=2)+'\n')
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    args = parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Parent must remain CPU only'
    assert read(args.base/'status.json')['state'] == 'fit_complete_awaiting_validation_review'
    assert read(args.base/'current_processes.json')['active'] == []
    for name in ['L16_rho0', 'L16_rho03']:
        result = read(args.base/name/'fit_result.json')
        assert result['epochs_completed'] == 30 and result['test_evaluations'] == 0
        # PID identities are recorded in metadata; do not start while an old worker is alive.
        pid = read(args.base/name/'metadata.json')['pid']
        cmdline = Path(f'/proc/{pid}/cmdline')
        assert not cmdline.exists() or str(args.base).encode() not in cmdline.read_bytes()
    assert sys.executable == PYTHON
    args.out.mkdir(parents=True, exist_ok=False)
    save(args.out/'metadata.json', dict(pid=os.getpid(), command=sys.argv, base=str(args.base),
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        policy='One L16 paired init .1 restart, 10 epochs, train/val only', cuda_visible_devices=''))
    active, finished = [], []
    try:
        for rho, name, gpu in zip([0., .3], ['rho0', 'rho03'], GPUS):
            cmd = [PYTHON, str(HERE/'medicinal_pilot.py'), '--profile', str(HERE/'coil20_L16_low_variance_profile.json'),
                '--out', str(args.out/name), '--data', str(args.data), '--gpu', gpu, '--rho', str(rho),
                '--depth', '16', '--epochs', '10', '--seconds', '1800', '--full-budget', '--micro', '1',
                '--batched', '1', '--probe', str(HERE/'runs/smoke/coil20_deeper_execution_20261011/probe.json')]
            log = (args.out/(name+'.log')).open('w')
            child = subprocess.Popen(cmd, env=dict(os.environ, CUDA_VISIBLE_DEVICES=gpu), stdout=log, stderr=subprocess.STDOUT)
            active.append((child, log, dict(name=name, rho=rho, gpu_uuid=gpu, pid=child.pid, command=cmd)))
        while active:
            save(args.out/'current_processes.json', dict(active=[j[2] for j in active]))
            save(args.out/'status.json', dict(state='paired_adjustment_running', completed=finished, test_evaluations=0))
            for job in list(active):
                child, log, identity = job
                code = child.poll()
                if code is None: continue
                log.close(); active.remove(job)
                assert code == 0, f"{identity['name']} failed with exit {code}"
                result = read(args.out/identity['name']/'fit_result.json')
                assert result['epochs_completed'] == 10 and result['test_evaluations'] == 0
                assert result['config']['maximum_epochs'] == 10
                finished.append(dict(identity, returncode=code))
            if active: time.sleep(5)
        save(args.out/'current_processes.json', dict(active=[]))
        save(args.out/'status.json', dict(state='fit_complete_awaiting_validation_review', completed=finished, test_evaluations=0))
    except BaseException as error:
        for child, log, identity in active:
            if child.poll() is None: child.terminate()
        for child, log, identity in active:
            try: child.wait(timeout=30)
            except subprocess.TimeoutExpired: pass
            log.close()
        save(args.out/'status.json', dict(state='failed', error=repr(error), test_evaluations=0))
        raise


if __name__ == '__main__': main()
