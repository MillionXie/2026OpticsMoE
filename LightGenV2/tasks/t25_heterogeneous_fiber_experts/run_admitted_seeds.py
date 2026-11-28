"""Dispatch a T25 seed as soon as its eight original GPU admissions are complete."""
import hashlib
import math
import os
from pathlib import Path
import re
import sys
import time
import traceback

from .run_cohort import PROFILE, TASK, read, run_seed, write


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def verify_manifest(directory, names):
    manifest = read(directory / 'manifest.json')
    require(set(manifest) == set(names), f'{directory}: unexpected manifest entries')
    for name in names:
        require(sha(directory / name) == manifest[name], f'{directory.name}: bad SHA for {name}')
    return manifest


def global_smoke(root, train_source):
    path = root / 'cohort_status.json'
    if not path.exists():
        return None
    receipt = read(path)
    if receipt['state'] == 'failed':
        raise RuntimeError(f"Original smoke cohort failed: {receipt.get('reason', 'see controller.log')}")
    require(receipt['stage'] == 'smoke' and receipt['source_commit'] == train_source
            and receipt['state'] in ('running', 'complete'), 'Unexpected original smoke cohort identity')
    return receipt['state']


def smoke_process_alive(pid):
    # Individual status is atomically published just before CUDA process exit.
    try:
        command = (Path('/proc') / str(int(pid)) / 'cmdline').read_bytes().split(b'\0')
    except (FileNotFoundError, ProcessLookupError):
        return False
    return b'--smoke' in command and f'{__package__}.train'.encode() in command


def admit_seed(root, profile, seed, train_source):
    expected = [(d, a, root/f'{d}_s{seed}_{a}') for d in profile['datasets'] for a in profile['arms']]
    require(len(expected) == 8, 'Expected eight admissions per seed')
    for _, _, directory in expected:
        status = directory/'status.json'
        if not status.exists() or read(status)['state'] != 'complete':
            return None
    sources = {}
    hashes, initialization = {}, {}
    gpu = profile['gpu_uuids'][str(seed)]
    for dataset, arm, directory in expected:
        status = read(directory/'status.json')
        require(status['smoke'] is True, f'{directory.name}: wrong status role')
        manifest = verify_manifest(directory, ('config.json','metadata.json','admission.json'))
        m, cfg, admission = (read(directory/n) for n in ('metadata.json','config.json','admission.json'))
        base_file = TASK.parent/profile['base_configs'][dataset]
        base = read(base_file)
        slots = profile['fiber_slots_zero_based'][arm]
        slots = slots[str(seed)] if isinstance(slots, dict) else slots
        cores = [392 if arm == 'uniform_half' else 784 if i in slots else 0 for i in range(4)]
        require(cfg == {**base, 'arm':arm, 'seed':seed, 'fiber_slots':slots, 'fiber_core_counts':cores,
                        'fiber_model':profile['fiber_model'], 'epochs':profile['epochs'], 'protocol_sha256':sha(PROFILE),
                        'assigned_gpu_uuid':gpu, 'data_config_sha256':sha(base_file)}, f'{directory.name}: profile differs')
        require(m['smoke'] is True and m['device'] == 'cuda' and m['config'] == cfg and m['seed'] == seed
                and m['dataset'] == cfg['dataset'] and m['config_file_sha256'] == manifest['config.json']
                and m['protocol_sha256'] == cfg['protocol_sha256'] and m['arm'] == admission['arm'] == arm
                and m['source_commit'] == admission['source_commit'] == train_source
                and m['gpu_uuid'] == admission['gpu_uuid'] == gpu and m['pid'] == status['pid']
                and m['fiber_core_counts'] == admission['fiber_core_counts'] == cores, f'{directory.name}: smoke identity')
        init = m['initial_state_sha256']
        require(re.fullmatch(r'[0-9a-f]{64}', init) and init == admission['initial_state_sha256']
                and initialization.setdefault(dataset, init) == init, f'{directory.name}: initialization differs')
        require(admission['domain'] == 'original TRAIN only'
                and sorted(admission['labels']) == list(range(base['num_classes']))
                and len(set(admission['ids'])) == base['num_classes']
                and admission['diagnostic_logits_bitwise'] is True and admission['shared_router_bitwise'] is True
                and (arm != 'diffraction' or admission['diffraction_control_logits_bitwise'] is True)
                and math.isfinite(admission['loss']), f'{directory.name}: failed TRAIN qualification')
        for field in ('gradient_l2','update_max_abs'):
            require(len(admission[field]) == 4 and all(math.isfinite(v) and v > 0 for v in admission[field].values()),
                    f'{directory.name}: invalid {field}')
        if not sources:
            sources = m['source_sha256']
            prefix = TASK.relative_to(TASK.parents[2]).as_posix()
            require(all(f'{prefix}/{f}' in sources for f in ('train.py','model.py')), 'Missing optical/trainer sources')
            for rel, digest in sources.items():
                file = (TASK.parents[2] / rel).resolve()
                require(file.is_relative_to(TASK.parents[2] / 'LightGenV2') and sha(file) == digest,
                        f'{directory.name}: on-disk training source differs at {rel}')
        require(m['source_sha256'] == sources, f'{directory.name}: training source versions differ')
        if smoke_process_alive(status['pid']):
            return None
        hashes[directory.name] = sha(directory/'manifest.json')
    return {'gpu_uuid':gpu, 'initial_state_sha256':initialization, 'smoke_manifest_sha256':hashes}


def verify_existing_formals(root, smoke_root, seed, profile, source, *, must_exist=False):
    names = ('config.json','metadata.json','history.json','selection_lock.json','result.json',
             'best_checkpoint.pt','last_checkpoint.pt','predictions_train.npz','predictions_val.npz','predictions_test.npz')
    for d in profile['datasets']:
        for a in profile['arms']:
            folder = root/f'{d}_s{seed}_{a}'
            status = folder/'status.json'
            if not status.exists() or read(status)['state'] != 'complete':
                require(not must_exist, f'{folder.name}: formal result missing')
                continue
            manifest = verify_manifest(folder, names)
            m = read(folder/'metadata.json')
            smoke = read(smoke_root/folder.name/'metadata.json')
            lock = read(folder/'selection_lock.json')
            require(m['smoke'] is False and m['seed'] == seed and m['arm'] == a and m['source_commit'] == source
                    and all(m[k] == smoke[k] for k in ('config','source_sha256','gpu_uuid','initial_state_sha256'))
                    and lock['best_checkpoint_sha256'] == manifest['best_checkpoint.pt']
                    and lock['target_epoch'] == read(status)['epoch'] == profile['epochs'],
                    f'{folder.name}: completed formal identity differs from admission')


def dispatch(root, smoke, profile, train_source, scheduler_source):
    seeds = profile['seeds']
    require(len(seeds) == len({profile['gpu_uuids'][str(s)] for s in seeds}) == 2,
            'Per-seed pools require two distinct physical GPUs')
    status = dict(stage='train', pid=os.getpid(), source_commit=train_source, scheduler_commit=scheduler_source,
                  datasets=profile['datasets'], arms=profile['arms'], seeds=seeds, epochs=profile['epochs'])
    admissions, active, finished = {}, {}, []
    def publish(state, **fields):
        write(root/'cohort_status.json', dict(status, state=state, admissions=admissions,
              active_seed_pids={s:p for p,s in active.items()}, completed_seeds=finished, **fields))
    publish('running')
    try:
        while len(finished) < len(seeds):
            smoke_state = global_smoke(smoke, train_source)
            for pid, seed in list(active.items()):
                waited, code = os.waitpid(pid, os.WNOHANG)
                if waited:
                    del active[pid]
                    require(os.waitstatus_to_exitcode(code) == 0, f'Seed {seed} formal worker pool failed')
                    verify_existing_formals(root, smoke, seed, profile, train_source, must_exist=True)
                    finished.append(seed)
                    publish('running')
            for seed in seeds:
                if smoke_state is None or seed in admissions:
                    continue
                receipt = admit_seed(smoke, profile, seed, train_source)
                if not receipt:
                    require(smoke_state != 'complete', f'Smoke cohort complete but seed {seed} admissions missing')
                    continue
                verify_existing_formals(root, smoke, seed, profile, train_source)
                admissions[seed] = receipt
                sys.stdout.flush()
                pid = os.fork()  # The Linux child inherits the parent's locked file descriptor.
                if pid == 0:
                    try:
                        run_seed('train', root, profile, seed)
                        os._exit(0)
                    except BaseException:
                        traceback.print_exc()
                        os._exit(1)
                active[pid] = seed
                print(f'admitted seed={seed} pool pid={pid} gpu={receipt["gpu_uuid"]}', flush=True)
                publish('running')
            if len(finished) < len(seeds):
                time.sleep(5)
        publish('complete')
    except BaseException as exc:
        publish('failed', reason=str(exc))
        raise


def main():
    import fcntl
    train_source = os.environ['T25_SOURCE_COMMIT']
    scheduler_source = os.environ['T25_SCHEDULER_COMMIT']
    require(all(re.fullmatch(r'[0-9a-f]{40}', v) for v in (train_source, scheduler_source)), 'Require full commit IDs')
    root = TASK/'runs'/'simulation'/PROFILE.stem
    smoke = TASK/'runs'/'smoke'/PROFILE.stem
    root.mkdir(parents=True, exist_ok=True)
    with (root/'.cohort.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        dispatch(root, smoke, read(PROFILE), train_source, scheduler_source)


if __name__ == '__main__':
    main()
