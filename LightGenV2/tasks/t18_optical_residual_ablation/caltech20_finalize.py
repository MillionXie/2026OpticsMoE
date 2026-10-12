"""Caltech20 common100: audit all eight, lock EMA choices, test once per arm."""
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
KEYS = [f'L{d}_{rho}' for d in (2, 4, 8, 16) for rho in ('rho0', 'rho03')]
GPUS = [
    'GPU-e8837b85-d55b-8e81-aaa5-ec1ac326932d',
    'GPU-6dcca91a-8e08-1a50-9aa6-81defeaed50b',
    'GPU-afc19890-6209-ee4d-622d-e619da5bd5b2',
    'GPU-1b963983-7909-af6e-0528-f0f0661ab549',
]


def read(path): return json.loads(path.read_text())
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value, exclusive=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive:
        with path.open('x') as stream:
            json.dump(value, stream, indent=2)
            stream.flush(); os.fsync(stream.fileno())
    else:
        tmp = path.with_suffix(path.suffix + '.tmp')
        tmp.write_text(json.dumps(value, indent=2) + '\n'); tmp.replace(path)


def verify_arm(arm):
    result, history, metadata = [read(arm / n) for n in ('fit_result.json', 'history.json', 'metadata.json')]
    assert result['test_evaluations'] == 0 and 'test' not in result['metrics']
    assert not result['test_used_for_selection'] and result['selection_split'] == 'val'
    assert result['candidate'] == 'base'
    assert result['epochs_completed'] == result['config']['requested_epochs'] == len(history) == 100
    assert result['stop_reason'] == 'maximum_epochs' and not result['config']['early_stopping']
    assert [r['epoch'] for r in history] == list(range(1, 101))
    assert all('test' not in row for row in history)
    best = min(history, key=lambda row: row['val']['balanced_nll'])
    assert best['epoch'] == result['selected_epoch'] and best['val'] == result['metrics']['val']
    assert sha(arm / 'history.json') == result['history_sha256']
    assert sha(arm / 'best_checkpoint.pt') == result['checkpoint_sha256']
    assert result['config'] == read(arm / 'config.json') and result['sources'] == metadata['sources']
    assert result['config']['microbatch'] == result['config']['evaluation_batch_size'] == (4 if result['depth'] <= 4 else 1)
    for path, digest in result['sources'].items():
        blob = subprocess.check_output(['git', 'show', metadata['git_commit'] + ':' + path], cwd=ROOT)
        assert hashlib.sha256(blob).hexdigest() == digest, path
    return result


def verify_data(data, plan):
    manifest = read(data.parent / 'data_manifest.json')
    assert manifest['dataset'] == plan['dataset'] == 'Caltech101_prespecified20_train60'
    assert sha(data) == manifest['cache_sha256'] == plan['data_sha256']
    assert sha(data.parent / 'data_manifest.json') == plan['manifest_sha256']
    assert len(manifest['classes']) == 20 and len(set(manifest['classes'])) == 20
    assert manifest['total_records'] == 3888 and manifest['previous_fit_or_validation_images_in_test'] == 0
    for split in ('train', 'val', 'test'):
        assert len(manifest['split_ids'][split]) == sum(manifest['supports'][split])
    assert manifest['supports']['train'] == [60] * 20 and manifest['supports']['val'] == [10] * 20
    assert sum(manifest['supports']['test']) == 2488
    assert sha(HERE / 'caltech20_profile.json') == plan['profile_sha256']
    return manifest


def seal(args):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    assert read(args.base / 'status.json')['state'] == 'fit_complete_awaiting_validation_review'
    assert read(args.base / 'current_processes.json')['active'] == []
    plan_path = HERE / 'caltech20_full_suite_plan.json'
    plan = read(plan_path); manifest = verify_data(args.data, plan)
    assert read(args.base / 'metadata.json')['plan_sha256'] == sha(plan_path)
    arms = read(args.base / 'fit_manifest.json')['arms']
    assert set(arms) == set(KEYS), 'All eight formal arms are required'
    audits, choices, results = {}, {}, {}
    for depth in (2, 4, 8, 16):
        pair = [Path(arms[f'L{depth}_{name}']) for name in ('rho0', 'rho03')]
        fits = [verify_arm(arm) for arm in pair]
        assert [r['depth'] for r in fits] == [depth, depth]
        assert [r['rho'] for r in fits] == [0., .3]
        assert all(r['data_sha256'] == manifest['cache_sha256'] for r in fits)
        assert fits[0]['sources'] == fits[1]['sources'] == read(args.base / 'metadata.json')['sources']
        assert fits[0]['config']['classes'] == manifest['classes']
        assert {k for k in set(fits[0]['config']) | set(fits[1]['config'])
            if fits[0]['config'].get(k) != fits[1]['config'].get(k)} == {'residual_rho'}
        assert all(fits[0][k] == fits[1][k] for k in ('initial_parameter_sha256', 'orders', 'transforms', 'parameters'))
        if depth == 2:
            for name, fit in zip(('rho0', 'rho03'), fits):
                assert fit['checkpoint_sha256'] == plan['adopted_L2'][name]['best_sha256']
        report = args.out / 'candidate_audits' / f'L{depth}_base'
        cmd = [sys.executable, str(HERE / 'audit_medicinal_pilot.py'), '--run', str(args.base),
            '--depth', str(depth), '--arm0', str(pair[0]), '--arm03', str(pair[1]),
            '--data', str(args.data), '--out', str(report)]
        with (args.out / f'audit_L{depth}.log').open('x') as log:
            subprocess.run(cmd, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''), stdout=log,
                stderr=subprocess.STDOUT, check=True)
        assert read(report / 'pilot_audit.json')['cpu_checks_passed']
        audits[f'L{depth}_base'] = sha(report / 'pilot_audit.json')
        choices[str(depth)] = dict(selected_candidate='base', rule='Per-arm first exact minimum EMA validation balanced NLL; common100, no new candidates',
            candidates=dict(base=dict(arms=[str(p.resolve()) for p in pair], epochs=100,
                mean_best_val_balanced_nll=sum(f['metrics']['val']['balanced_nll'] for f in fits) / 2)))
        results.update({f'L{depth}_{n}': f for n, f in zip(('rho0', 'rho03'), fits)})
    # Locks are written only after every saved-PT/CSV CPU audit has succeeded.
    entries = {}
    for depth in (2, 4, 8, 16):
        pair_lock = args.out / f'pair_lock_L{depth}.json'
        save(pair_lock, choices[str(depth)], exclusive=True)
        for name in ('rho0', 'rho03'):
            key = f'L{depth}_{name}'; arm = Path(arms[key]); fit = results[key]
            entry = dict(arm=str(arm.resolve()), depth=depth, rho=fit['rho'], candidate='base',
                selected_epoch=fit['selected_epoch'], selected_state_kind='ema', selection_split='val',
                test_evaluations_before_lock=0, test_used_for_selection=False, pair_lock_sha256=sha(pair_lock))
            for field, filename in ARTIFACTS.items(): entry[field] = sha(arm / filename)
            save(args.out / key / 'selection_lock.json', entry, exclusive=True)
            entries[key] = dict(entry, selection_lock_sha256=sha(args.out / key / 'selection_lock.json'))
    save(args.out / 'global_selection_lock.json', dict(entries=entries, choices=choices, candidate_audits=audits,
        data_sha256=sha(args.data), manifest_sha256=sha(args.data.parent / 'data_manifest.json'),
        classes=manifest['classes'], supports=manifest['supports'], plan_sha256=sha(plan_path),
        lock_time_unix=time.time(), source_sha256=sha(Path(__file__)),
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        scope='Count-conditioned Caltech101 20-category subset, common100, seed17; validation-guided exploratory task adaptation; no established instance/photographer independence'), exclusive=True)
    save(args.out / 'status.json', dict(state='selection_locked_test_not_started', test_evaluations=0))


ARTIFACTS = dict(fit_result_sha256='fit_result.json', checkpoint_sha256='best_checkpoint.pt',
    last_checkpoint_sha256='last_checkpoint.pt', history_sha256='history.json', config_sha256='config.json',
    metadata_sha256='metadata.json', gradients_sha256='gradients.json', smoke_sha256='smoke.json',
    train_predictions_sha256='train_predictions.csv', val_predictions_sha256='val_predictions.csv')


def verify_global(out, data):
    lock = read(out / 'global_selection_lock.json')
    assert set(lock['entries']) == set(KEYS), 'All eight choices required before test'
    assert lock['source_sha256'] == sha(Path(__file__)), 'Changed sealed finalizer source'
    plan = read(HERE / 'caltech20_full_suite_plan.json')
    assert lock['plan_sha256'] == sha(HERE / 'caltech20_full_suite_plan.json')
    manifest = verify_data(data, plan)
    assert manifest['classes'] == lock['classes'] and manifest['supports'] == lock['supports']
    assert sha(data) == lock['data_sha256'] and sha(data.parent / 'data_manifest.json') == lock['manifest_sha256']
    for key, entry in lock['entries'].items():
        depth = entry['depth']; expected_rho = 0. if key.endswith('_rho0') else .3
        assert key == f'L{depth}_' + ('rho0' if expected_rho == 0 else 'rho03') and entry['rho'] == expected_rho
        assert entry['selected_state_kind'] == 'ema' and entry['selection_split'] == 'val'
        assert entry['test_evaluations_before_lock'] == 0 and not entry['test_used_for_selection']
        local = out / key / 'selection_lock.json'
        assert sha(local) == entry['selection_lock_sha256']
        assert read(local) == {k: v for k, v in entry.items() if k != 'selection_lock_sha256'}
        pair = out / f'pair_lock_L{depth}.json'
        assert sha(pair) == entry['pair_lock_sha256'] and read(pair) == lock['choices'][str(depth)]
        choice = lock['choices'][str(depth)]
        assert choice['selected_candidate'] == entry['candidate'] == 'base'
        assert choice['candidates']['base']['epochs'] == 100
        assert entry['arm'] == choice['candidates']['base']['arms'][0 if entry['rho'] == 0. else 1]
        for field, name in ARTIFACTS.items():
            assert sha(Path(entry['arm']) / name) == entry[field], f'Changed locked artifact: {key}/{name}'
        audit = out / 'candidate_audits' / f'L{depth}_base' / 'pilot_audit.json'
        assert sha(audit) == lock['candidate_audits'][f'L{depth}_base'] and read(audit)['cpu_checks_passed']
    return lock


def claim_test(out, key, global_digest):
    assert key in KEYS
    save(out / key / 'test_started.json', dict(global_selection_lock_sha256=global_digest,
        pid=os.getpid(), time_unix=time.time(), policy='Exclusive durable marker; failure forbids automatic retry'), exclusive=True)


def load_test_arrays(data, manifest):
    import numpy as np
    import torch
    with np.load(data, allow_pickle=False) as cache:
        images, labels, ids = [cache['test_' + n].copy() for n in ('images', 'labels', 'ids')]
    assert images.dtype == np.uint8 and images.shape == (len(manifest['split_ids']['test']), 150, 150, 3)
    assert ids.tolist() == manifest['split_ids']['test']
    assert np.bincount(labels, minlength=len(manifest['classes'])).tolist() == manifest['supports']['test']
    return torch.from_numpy(images.transpose(0, 3, 1, 2).copy()), torch.from_numpy(labels).long(), ids


def worker(args):
    lock = verify_global(args.out, args.data)
    entry = lock['entries'][args.key]; arm = Path(entry['arm']); fit = verify_arm(arm)
    assert os.environ['CUDA_VISIBLE_DEVICES'] == args.gpu and args.gpu in GPUS
    assert not (args.out / args.key / 'test_started.json').exists(), 'No test retry'
    assert all(sha(ROOT / p) == h for p, h in fit['sources'].items()), 'Changed inference dependency'
    import medicinal_pilot as pilot
    execution, o, torch = pilot.execution, pilot.o, pilot.torch
    o.f.gpu_budget.ALLOWED.update(GPUS)
    policy = o.f.gpu_budget.configure(torch, args.gpu, shared=True, memory_gib=3)
    torch.set_num_threads(4)
    profile = read(HERE / 'caltech20_profile.json')
    checkpoint = torch.load(arm / 'best_checkpoint.pt', map_location='cpu', weights_only=False)
    assert checkpoint['selected_state_kind'] == 'ema' and checkpoint['selection_split'] == 'val'
    assert checkpoint['epoch'] == entry['selected_epoch'] and checkpoint['config'] == fit['config']
    assert checkpoint['sources'] == fit['sources'] and checkpoint['data_sha256'] == lock['data_sha256']
    model = execution.build(entry['depth'], profile, fit['config'], batched=fit['config']['expert_batched'])
    model.load_state_dict(checkpoint['model'], strict=True); model.eval()
    # Claim before first test-array materialization; failed claims/evaluations remain.
    claim_test(args.out, args.key, sha(args.out / 'global_selection_lock.json'))
    data = load_test_arrays(args.data, read(args.data.parent / 'data_manifest.json'))
    metric, rows = pilot.evaluate(model, data, fit['config']['evaluation_batch_size'])
    o.r.csvwrite(args.out / args.key / 'test_predictions.csv', rows)
    save(args.out / args.key / 'result.json', dict(depth=entry['depth'], rho=entry['rho'], candidate='base',
        selected_epoch=entry['selected_epoch'], selected_state_kind='ema', selection_split='val',
        metrics=dict(fit['metrics'], test=metric), parameters=fit['parameters'],
        test_evaluations=1, test_used_for_selection=False,
        global_selection_lock_sha256=sha(args.out / 'global_selection_lock.json'),
        checkpoint_sha256=entry['checkpoint_sha256'], data_sha256=lock['data_sha256'],
        test_predictions_sha256=sha(args.out / args.key / 'test_predictions.csv'),
        inference_source_sha256=sha(Path(__file__)), gpu_policy=policy, scope=lock['scope']))


def idle(gpu):
    return gpu not in subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid',
        '--format=csv,noheader'], text=True).splitlines()


def suite(args):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    verify_global(args.out, args.data)
    assert not any((args.out / key / 'test_started.json').exists() for key in KEYS), 'No automatic test retries'
    pending, active, completed, failures = list(reversed(KEYS)), {}, [], []
    while active or (pending and not failures):
        for gpu, (child, key) in list(active.items()):
            code = child.poll()
            if code is None: continue
            try:
                assert code == 0
                result = read(args.out / key / 'result.json')
                assert result['test_evaluations'] == 1 and not result['test_used_for_selection']
                completed.append(key)
            except Exception as error:
                failures.append(dict(key=key, pid=child.pid, exit_code=code, error=repr(error)))
            del active[gpu]
        if not failures:
            for gpu in GPUS:
                if gpu in active or not pending or not idle(gpu): continue
                key = pending.pop(0)
                cmd = [sys.executable, str(Path(__file__).resolve()), '--phase', 'worker',
                    '--out', str(args.out), '--data', str(args.data), '--key', key, '--gpu', gpu]
                with (args.out / key / 'test.log').open('x') as log:
                    child = subprocess.Popen(cmd, env=dict(os.environ, CUDA_VISIBLE_DEVICES=gpu),
                        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                active[gpu] = (child, key)
        save(args.out / 'current_processes.json', dict(active=[dict(key=k, gpu_uuid=g, pid=c.pid) for g, (c, k) in active.items()], pending=pending))
        save(args.out / 'status.json', dict(state='test_failure_draining' if failures else 'selected_test_running', completed=completed, failures=failures))
        if active or (pending and not failures): time.sleep(3)
    save(args.out / 'status.json', dict(state='test_failed_no_automatic_retry' if failures else 'tests_complete_awaiting_cpu_audit', completed=completed, failures=failures))
    assert not failures, 'Test failure: keep all markers and artifacts, no automatic retry'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--phase', choices=('lock', 'suite', 'worker'), required=True)
    p.add_argument('--base', type=Path)
    p.add_argument('--out', type=Path, required=True); p.add_argument('--data', type=Path, required=True)
    p.add_argument('--key', choices=KEYS); p.add_argument('--gpu', choices=GPUS)
    a = p.parse_args()
    if a.phase == 'lock':
        assert a.base is not None
        a.out.mkdir(parents=True, exist_ok=False); seal(a)
    elif a.phase == 'suite': suite(a)
    else: worker(a)


if __name__ == '__main__': main()
