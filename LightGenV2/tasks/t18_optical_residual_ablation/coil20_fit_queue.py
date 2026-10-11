"""Six bounded deeper fit jobs; adopts saved L2, never reads or evaluates test."""
import argparse
import hashlib
import json
import os
import platform
from pathlib import Path
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
GPUS = ['GPU-e8837b85-d55b-8e81-aaa5-ec1ac326932d',
    'GPU-6dcca91a-8e08-1a50-9aa6-81defeaed50b', 'GPU-1b963983-7909-af6e-0528-f0f0661ab549']


def read(path): return json.loads(path.read_text())
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path, value):
    temporary = path.with_suffix('.tmp'); temporary.write_text(json.dumps(value, indent=2)); temporary.replace(path)


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Supervisor is CPU only'
    p = argparse.ArgumentParser(); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True); p.add_argument('--l2', type=Path, required=True)
    p.add_argument('--probe', type=Path, required=True); p.add_argument('--timing', type=Path, required=True)
    a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=False)
    data = read(a.data.parent/'data_manifest.json'); assert sha(a.data) == data['cache_sha256']
    timing = read(a.timing/'fit_result.json'); history = read(a.timing/'history.json')
    assert timing['epochs_completed'] == 2 and timing['depth'] == 16 and timing['test_evaluations'] == 0
    # Full train assessment every five epochs, separately from final selected-state assessment.
    projection = history[1]['epoch_seconds'] + (history[0]['epoch_seconds']-history[1]['epoch_seconds'])/5
    assert projection <= 180
    modes = {4:4, 8:4, 16:1}; probes = read(a.probe)['results']
    for depth, micro in modes.items():
        for rho in [0., .3]:
            assert any(x['depth'] == depth and x['rho'] == rho and x['microbatch'] == micro and x['expert_batched'] and x['passed'] for x in probes)
    selected = {}
    for name in ['rho0', 'rho03']:
        arm = a.l2/name; result = read(arm/'fit_result.json')
        assert result['epochs_completed'] == 30 and result['test_evaluations'] == 0
        assert result['data_sha256'] == data['cache_sha256']
        assert result['checkpoint_sha256'] == sha(arm/'best_checkpoint.pt')
        selected['L2_'+name] = str(arm.resolve())
        if name == 'rho0': assert result['accuracy_gate_passed'] and result['speed_gate_passed']
    command = [sys.executable, str(HERE/'medicinal_pilot.py')]
    jobs = [dict(depth=depth, rho=rho, name=f'L{depth}_'+('rho0' if rho == 0 else 'rho03'), micro=modes[depth])
        for depth, rho in [(16,0.),(16,.3),(8,0.),(8,.3),(4,0.),(4,.3)]]
    save(a.out/'metadata.json', dict(command=sys.argv, pid=os.getpid(), supervisor_source_sha256=sha(Path(__file__)),
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(), data_sha256=sha(a.data),
        reused_L2=selected, modes=modes, maximum_epochs=30, maximum_arm_seconds=5400,
        python=sys.version, platform=platform.platform(), CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'],
        L16_measured_ordinary_epoch_seconds=history[1]['epoch_seconds'],
        L16_projected_periodic_mean_seconds=projection, test_evaluations=0))
    pending, active, completed = jobs[:], {}, []
    started = time.time()
    try:
        while pending or active:
            for gpu in list(active):
                proc, job = active[gpu]
                if proc.poll() is not None:
                    assert proc.returncode == 0, f'Fit failed: {job}'
                    result = read(a.out/job['name']/'fit_result.json')
                    assert result['epochs_completed'] == 30 and result['test_evaluations'] == 0
                    selected[job['name']] = str((a.out/job['name']).resolve())
                    completed.append(dict(**job, returncode=proc.returncode)); del active[gpu]
            for gpu in GPUS:
                if gpu in active or not pending: continue
                job = pending.pop(0); arm = a.out/job['name']
                args = command + ['--profile', str(HERE/'coil20_pilot_profile.json'), '--out', str(arm),
                    '--data', str(a.data), '--gpu', gpu, '--rho', str(job['rho']), '--depth', str(job['depth']),
                    '--epochs', '30', '--seconds', '5400', '--micro', str(job['micro']), '--batched', '1',
                    '--full-budget', '--probe', str(a.probe)]
                env = os.environ.copy(); env['CUDA_VISIBLE_DEVICES'] = gpu
                log = (a.out/(job['name']+'.log')).open('w')
                process = subprocess.Popen(args, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                job.update(pid=process.pid, gpu_uuid=gpu, command=args); active[gpu] = (process, job)
            save(a.out/'current_processes.json', dict(active=[job for proc, job in active.values()], pending=pending))
            save(a.out/'status.json', dict(state='fit_running', completed=completed, pending=len(pending),
                active=len(active), seconds=time.time()-started, test_evaluations=0))
            time.sleep(5)
        save(a.out/'fit_manifest.json', dict(arms=selected, data_sha256=sha(a.data), test_evaluations=0,
            next_step='CPU audit and paired validation fit review; global selection lock required before any test'))
        save(a.out/'status.json', dict(state='fit_complete_awaiting_validation_review', completed=completed,
            seconds=time.time()-started, test_evaluations=0))
    except BaseException as error:
        for process, job in active.values():
            cmdline = Path(f'/proc/{process.pid}/cmdline')
            if cmdline.exists() and str(a.out).encode() in cmdline.read_bytes():
                os.kill(process.pid, signal.SIGTERM)
                try: process.wait(timeout=30)
                except subprocess.TimeoutExpired: pass
        save(a.out/'status.json', dict(state='failed', error=str(error), stopped_jobs=[job for proc, job in active.values()], test_evaluations=0))
        raise


if __name__ == '__main__': main()
