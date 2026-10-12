"""Four GPU slots, six fresh 100-epoch fits, and adoption of the completed L2 pair.

No test arrays or test inference. A failed child stops dispatch; healthy children
finish and retain their artifacts. There are no automatic retries.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
GPUS = [
    'GPU-e8837b85-d55b-8e81-aaa5-ec1ac326932d',
    'GPU-6dcca91a-8e08-1a50-9aa6-81defeaed50b',
    'GPU-afc19890-6209-ee4d-622d-e619da5bd5b2',
    'GPU-1b963983-7909-af6e-0528-f0f0661ab549',
]
KEYS = [f'L{d}_{r}' for d in (2, 4, 8, 16) for r in ('rho0', 'rho03')]


def read(path): return json.loads(path.read_text())
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def idle(gpu):
    output = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid',
        '--format=csv,noheader'], text=True)
    return gpu not in output.splitlines()


def verify_sources(sources):
    for name, digest in sources.items():
        assert sha(ROOT / name) == digest, f'Frozen training source changed: {name}'
        blob = subprocess.check_output(['git', 'show', 'HEAD:' + name], cwd=ROOT)
        assert hashlib.sha256(blob).hexdigest() == digest, name


def prepare(a):
    manifest = read(a.data.parent / 'data_manifest.json')
    profile = read(HERE / 'caltech20_profile.json')
    plan = read(HERE / 'caltech20_full_suite_plan.json')
    assert manifest['dataset'] == profile['dataset'] == 'Caltech101_prespecified20_train60'
    assert sha(a.data) == manifest['cache_sha256'] == plan['data_sha256']
    assert sha(a.data.parent / 'data_manifest.json') == plan['manifest_sha256']
    assert sha(HERE / 'caltech20_profile.json') == plan['profile_sha256']
    assert set(profile['gpu_uuids']) == set(GPUS) and len(GPUS) == 4
    assert manifest['previous_fit_or_validation_images_in_test'] == 0
    modes = {4: 4, 8: 1, 16: 1}
    probes = read(a.probe)['results']
    for depth, micro in modes.items():
        for rho in (0., .3):
            assert any(x['depth'] == depth and x['rho'] == rho and x['microbatch'] == micro
                and x['expert_batched'] and x['passed'] for x in probes)
    selected, adopted, frozen = {}, {}, None
    for name, rho in [('rho0', 0.), ('rho03', .3)]:
        arm = a.l2 / ('L2_' + name)
        result = read(arm / 'fit_result.json')
        assert result['depth'] == 2 and result['rho'] == rho
        assert result['epochs_completed'] == result['config']['requested_epochs'] == 100
        assert result['stop_reason'] == 'maximum_epochs' and not result['config']['early_stopping']
        assert result['test_evaluations'] == 0 and not result['test_used_for_selection']
        assert result['data_sha256'] == manifest['cache_sha256']
        assert result['checkpoint_sha256'] == sha(arm / 'best_checkpoint.pt') == plan['adopted_L2'][name]['best_sha256']
        assert result['history_sha256'] == sha(arm / 'history.json')
        assert result['sources'] == (frozen or result['sources'])
        frozen = result['sources']
        selected['L2_' + name] = str(arm.resolve())
        adopted[name] = dict(arm=str(arm.resolve()), fit_result_sha256=sha(arm / 'fit_result.json'),
            best_checkpoint_sha256=result['checkpoint_sha256'], training_git_commit=read(arm / 'metadata.json')['git_commit'])
    assert read(a.l2 / 'cpu_report_L2/pilot_audit.json')['cpu_checks_passed']
    verify_sources(frozen)
    timings = {}
    for name in ('rho0', 'rho03'):
        arm = a.l2 / ('L16_timing_' + name)
        result, history = read(arm / 'fit_result.json'), read(arm / 'history.json')
        assert result['depth'] == 16 and result['epochs_completed'] == 2 and result['test_evaluations'] == 0
        timings[name] = dict(ordinary_epoch_seconds=history[1]['epoch_seconds'],
            projected_periodic_mean_seconds=(history[0]['epoch_seconds'] + 4 * history[1]['epoch_seconds']) / 5,
            scope='Measured timing only; never reused as a formal checkpoint')
    return selected, adopted, frozen, timings, modes


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Supervisor is CPU only'
    p = argparse.ArgumentParser()
    for key in ('out', 'data', 'l2', 'probe'):
        p.add_argument('--' + key, type=Path, required=True)
    a = p.parse_args()
    selected, adopted, frozen, timings, modes = prepare(a)
    assert all(idle(gpu) for gpu in GPUS), 'A permitted GPU is occupied; no other process is killed'
    a.out.mkdir(parents=True, exist_ok=False)
    jobs = [dict(depth=d, rho=r, name=f'L{d}_' + ('rho0' if r == 0 else 'rho03'), micro=modes[d])
        for d, r in [(16, 0.), (16, .3), (8, 0.), (8, .3), (4, 0.), (4, .3)]]
    save(a.out / 'metadata.json', dict(command=sys.argv, pid=os.getpid(),
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        supervisor_source_sha256=sha(Path(__file__)), sources=frozen, adopted_L2=adopted,
        plan_sha256=sha(HERE / 'caltech20_full_suite_plan.json'), data_sha256=sha(a.data),
        execution_probe_sha256=sha(a.probe), measured_L16=timings, gpu_uuids=GPUS,
        modes=modes, maximum_epochs=100, maximum_arm_seconds=36000, test_evaluations=0))
    pending, active, completed, failures = jobs[:], {}, [], []
    started = time.time()
    while active or (pending and not failures):
        for gpu, (proc, job) in list(active.items()):
            code = proc.poll()
            if code is None: continue
            record = dict(job, exit_code=code)
            try:
                assert code == 0, f'Child exited with {code}'
                result = read(a.out / job['name'] / 'fit_result.json')
                assert result['epochs_completed'] == 100 and result['stop_reason'] == 'maximum_epochs'
                assert result['sources'] == frozen and result['test_evaluations'] == 0
                selected[job['name']] = str((a.out / job['name']).resolve())
                completed.append(record)
            except Exception as error:
                failures.append(dict(record, error=repr(error)))
            del active[gpu]
        if not failures:
            for gpu in GPUS:
                if gpu in active or not pending or not idle(gpu): continue
                try: verify_sources(frozen)
                except Exception as error:
                    failures.append(dict(error=repr(error))); break
                job = pending.pop(0)
                cmd = [sys.executable, str(HERE / 'medicinal_pilot.py'), '--profile', str(HERE / 'caltech20_profile.json'),
                    '--out', str(a.out / job['name']), '--data', str(a.data), '--gpu', gpu,
                    '--rho', str(job['rho']), '--depth', str(job['depth']), '--epochs', '100',
                    '--seconds', '36000', '--micro', str(job['micro']), '--batched', '1',
                    '--full-budget', '--probe', str(a.probe)]
                with (a.out / (job['name'] + '.log')).open('x') as log:
                    proc = subprocess.Popen(cmd, env=dict(os.environ, CUDA_VISIBLE_DEVICES=gpu),
                        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                job.update(pid=proc.pid, gpu_uuid=gpu, command=cmd)
                active[gpu] = (proc, job)
        save(a.out / 'current_processes.json', dict(active=[j for c, j in active.values()], pending=pending))
        save(a.out / 'status.json', dict(state='fit_failure_draining' if failures else 'fit_running',
            completed=completed, failures=failures, pending=len(pending), active=len(active),
            seconds=time.time() - started, test_evaluations=0))
        if active or (pending and not failures): time.sleep(5)
    if failures:
        save(a.out / 'status.json', dict(state='fit_failed_no_automatic_retry', completed=completed,
            failures=failures, pending=pending, active=0, seconds=time.time() - started, test_evaluations=0))
        raise RuntimeError(f'No automatic retry; preserved failures: {failures}')
    assert set(selected) == set(KEYS)
    save(a.out / 'fit_manifest.json', dict(arms=selected, data_sha256=sha(a.data), test_evaluations=0,
        next_step='CPU audit all eight, seal pair/selection/global locks, then once per selected EMA test'))
    save(a.out / 'status.json', dict(state='fit_complete_awaiting_validation_review', completed=completed,
        failures=[], seconds=time.time() - started, test_evaluations=0))


if __name__ == '__main__': main()
