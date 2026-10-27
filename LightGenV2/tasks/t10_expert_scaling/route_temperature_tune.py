"""Bounded validation-selected routing-temperature pilot; shared trainers stay frozen."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time

from .expert_tail_schedule import save, sha


def validate_profile(cfg):
    if cfg['gpu'] != 3 or cfg['gpu_uuid'] != 'GPU-4d8bfdb9-8777-05a6-3811-ab18ff4eadfd':
        raise ValueError('Only authorized GPU3')
    if cfg['temperatures'] != [.02, .01] or cfg['extra_epochs'] != 12 or cfg['slots'] != 2:
        raise ValueError('The declared pilot is two temperatures, twelve epochs each')
    if cfg['pilot_seed'] != 17 or cfg['rollout_seeds'] != [27, 37]:
        raise ValueError('Seed identities changed')
    if cfg['effective_batch'] != 16 or cfg['microbatch'] != 2:
        raise ValueError('Keep the existing batch contract')
    if cfg['required_validation_gain_correct'] != 4 or cfg['validation_samples'] != 752:
        raise ValueError('Validation-only rollout criterion changed')
    if cfg.get('temperature_schedule') != 'linear_from_0.05_over_12_epochs':
        raise ValueError('Use gradual input-power adaptation, not an abrupt temperature change')


def annealed_temperature(target, step, epochs=12):
    if target <= 0 or not 1 <= step <= epochs or epochs < 2:
        raise ValueError('Invalid temperature schedule')
    return .05 + (target - .05) * (step - 1) / (epochs - 1)


def select_pilot(results, baseline_accuracy, cfg):
    """Ignore any TEST fields. A minimum four validation examples gates rollout."""
    best = min(results, key=lambda x: (-x['val']['accuracy'], x['val']['macro_nll']))
    gain = (best['val']['accuracy'] - baseline_accuracy) * cfg['validation_samples']
    return best, gain >= cfg['required_validation_gain_correct'] - 1e-3


def model_factory(protocol, arguments, classes, temperature):
    from .model import ScalingOptics

    class TemperatureOptics(ScalingOptics):
        def route(self, source, dense=False, ste=True):
            _, original_probability, mask = super().route(source, dense=dense, ste=ste)
            # softmax(log(softmax(score/0.05))*0.05/T) == softmax(score/T).
            probability = (original_probability.clamp_min(1e-12).log() * (.05 / self.routing_temperature)).softmax(1)
            return probability.clamp_min(1e-12).sqrt(), probability, mask

    if int(arguments['experts']) != 49 or int(arguments['top_k']) != 49:
        raise ValueError('This study preserves all 49 active experts')
    model = TemperatureOptics(protocol, 49, 49, arguments['arch'], classes,
                              int(arguments['layers']), int(arguments['padding']))
    model.routing_temperature = temperature
    return model


def source_identity(cfg, seed):
    root = Path(cfg['source_root'])
    p = root / f'N49_k49_s{seed}'
    m = json.loads((p / 'metadata.json').read_text())
    result = json.loads((p / 'result.json').read_text())
    history = json.loads((p / 'history.json').read_text())
    if int(m['arguments']['epochs']) != 120 or history[-1]['epoch'] != 120:
        raise ValueError('Require a completed original 120-epoch source')
    if int(m['arguments']['seed']) != seed or m['arguments']['arch'] != 'moe_oeo':
        raise ValueError('Source identity changed')
    digest = sha(p / 'best_checkpoint.pt')
    if digest != result['checkpoint_sha256']:
        raise ValueError('Source selected model changed')
    expected = cfg['source_best_sha256'].get(str(seed))
    if expected and expected != digest:
        raise ValueError('Predeclared source hash changed')
    return p, m, result, digest


def verify_sources():
    commit = os.environ['T10_TEMPERATURE_SOURCE_COMMIT']
    repo = Path(subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], text=True).strip())
    paths = [Path(__file__), *[Path(__file__).with_name(n) for n in
             ('model.py', 'train.py', 'plan.py', 'diagnose_topk.py')],
             Path(__file__).parent / 'configs/study.json',
             repo / 'LightGenV2/common/optical_contract.py']
    hashes = {}
    for p in paths:
        relative = p.resolve().relative_to(repo).as_posix()
        if p.read_bytes() != subprocess.check_output(['git', 'show', commit + ':' + relative]):
            raise ValueError('Published source mismatch: ' + relative)
        hashes[relative] = sha(p)
    return commit, hashes


def worker(cfg, out, seed, temperature, profile_path, expected_source_sha=None):
    import numpy as np
    import torch
    from .train import data, images, objective, evaluate
    from .diagnose_topk import classification_margin_loss, save_route_audit

    commit, sources = verify_sources()
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    source, original, source_result, digest = source_identity(cfg, seed)
    if expected_source_sha is not None and digest != expected_source_sha:
        raise ValueError('Rollout source changed after recipe lock')
    out.mkdir(parents=True, exist_ok=False)
    cache = Path(original['arguments']['data'])
    if sha(cache) != original['data_sha256']:
        raise ValueError('Training data changed')
    splits, manifest = data(cache)  # Reads TRAIN/VAL only.
    protocol = copy.deepcopy(original['protocol'])
    protocol['model']['routing_temperature'] = temperature
    tc = protocol['training_proposal']
    model = model_factory(protocol, original['arguments'], len(manifest['classes']), .05).cuda()
    state = torch.load(source / 'best_checkpoint.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(state['model'])
    del state
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=0.)
    metadata = copy.deepcopy(original)
    metadata.pop('diagnosis', None)
    metadata.update(git_commit=commit, host_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                    protocol=protocol, source_run=str(source), source_best_sha256=digest,
                    source_selected_epoch=source_result['best_epoch'], command=sys.argv,
                    profile_sha256=sha(profile_path), source_files=sources,
                    cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), pid=os.getpid(),
                    temperature_tune=dict(target_temperature=temperature,
                                          temperature_schedule=cfg['temperature_schedule'],
                                          extra_epochs=12, lr_start=cfg['lr'],
                                          lr_end=cfg['lr']*.1, optimizer_reset=True,
                                          live_and_ema_initialized_from_selected_ema=True,
                                          source_training_budget=120, not_exact_resume=True,
                                          margin_weight=cfg['margin_weight'], test_read=False))
    metadata['arguments'].update(out=str(out), epochs=132, lr=cfg['lr'])
    save(out / 'metadata.json', metadata)
    initial, _ = evaluate(ema, splits['val'], 2)
    if abs(initial['accuracy'] - source_result['val']['accuracy']) > 1.001 / cfg['validation_samples']:
        raise ValueError('The unchanged-temperature warm start does not reproduce source VAL')
    save(out / 'initial_val.json', initial)
    best_key = (-initial['accuracy'], initial['macro_nll'])
    best_epoch = 120

    def payload(epoch, val):
        return dict(model=ema.state_dict(), live_model=model.state_dict(),
                    optimizer=optimizer.state_dict(), epoch=epoch, val=val, metadata=metadata,
                    routing_temperature=ema.routing_temperature)

    torch.save(payload(120, initial), out / 'best_checkpoint.pt')
    x, y, _ = splits['train']
    counts = torch.bincount(y, minlength=len(manifest['classes'])).float().cuda()
    weights = len(y) / (len(manifest['classes']) * counts)
    history = []
    started = time.time()
    for step in range(1, 13):
        epoch = 120 + step
        model.routing_temperature = ema.routing_temperature = annealed_temperature(temperature, step)
        model.train()
        order = torch.randperm(len(y), generator=torch.Generator().manual_seed(seed + epoch*100003))
        lr = cfg['lr'] * (.1 + .9*(1 + math.cos(math.pi*(step-1)/11))/2)
        for group in optimizer.param_groups:
            group['lr'] = lr
        total, batches = 0., 0
        for start in range(0, len(order), 16):
            effective = order[start:start+16]
            optimizer.zero_grad(set_to_none=True)
            for offset in range(0, len(effective), 2):
                ix = effective[offset:offset+2]
                output = model(images(x, ix, epoch, seed, True), dense=False)
                labels = y[ix].cuda()
                loss = objective(model, output, labels, weights, tc)
                loss = loss + cfg['margin_weight'] * classification_margin_loss(output['probabilities'], labels)
                loss = loss * len(ix) / len(effective)
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite loss')
                loss.backward()
                total += float(loss.detach())
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            with torch.no_grad():
                for ep, p in zip(ema.parameters(), model.parameters()):
                    ep.lerp_(p, 1-tc['ema_decay'])
            batches += 1
        val, _ = evaluate(ema, splits['val'], 2)
        entry = dict(epoch=epoch, temperature=ema.routing_temperature,
                     train_objective=total/batches, val=val, lr=lr,
                     gradient_norm=float(norm), elapsed_seconds=time.time()-started)
        if step in (1, 5, 10, 12):
            entry['train'], _ = evaluate(ema, splits['train'], 2)
        history.append(entry)
        save(out / 'history.json', history)
        torch.save(payload(epoch, val), out / 'last_checkpoint.pt')
        key = (-val['accuracy'], val['macro_nll'])
        if key < best_key:
            best_key, best_epoch = key, epoch
            torch.save(payload(epoch, val), out / 'best_checkpoint.pt')
        save(out / 'status.json', dict(state='training', epoch=epoch, total_epochs=132,
                                       best_epoch=best_epoch, pid=os.getpid()))
        print(json.dumps(entry), flush=True)
    selected = torch.load(out / 'best_checkpoint.pt', map_location='cpu', weights_only=False)
    ema.load_state_dict(selected['model'])
    ema.routing_temperature = selected['routing_temperature']
    train, _ = evaluate(ema, splits['train'], 2)
    val, _ = evaluate(ema, splits['val'], 2)
    for label in ('train', 'val'):
        save_route_audit(ema, splits[label], out / f'selected_{label}_routes.npz')
    result = dict(seed=seed, temperature=ema.routing_temperature, target_temperature=temperature,
                  best_epoch=best_epoch, train=train, val=val,
                  checkpoint_sha256=sha(out / 'best_checkpoint.pt'), source_best_sha256=digest,
                  original_temperature_val=source_result['val'], test_read=False)
    save(out / 'result.json', result)
    save(out / 'status.json', dict(state='validation_selected', **result))


def evaluate_selected(out, cfg):
    import numpy as np
    import torch
    from .train import evaluate

    commit, sources = verify_sources()
    m = json.loads((out / 'metadata.json').read_text())
    result = json.loads((out / 'result.json').read_text())
    digest = sha(out / 'best_checkpoint.pt')
    if digest != result['checkpoint_sha256'] or (out / 'test_result.json').exists():
        raise ValueError('Changed checkpoint or TEST already evaluated')
    save(out / 'test_selection_lock.json', dict(checkpoint_sha256=digest, selection='validation_accuracy_then_nll'))
    cache = Path(m['arguments']['data'])
    manifest_path = cache.parent / 'data_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if sha(cache) != m['data_sha256'] or sha(cache) != manifest['cache_sha256']:
        raise ValueError('TEST data identity changed')
    with np.load(cache, allow_pickle=False) as archive:
        x = archive['test_images'].copy()
        y = archive['test_labels'].reshape(-1).copy()
        ids = archive['test_ids'].astype(str).copy()
    if ids.tolist() != manifest['split_ids']['test'] or set(ids) & (set(manifest['split_ids']['train']) | set(manifest['split_ids']['val'])):
        raise ValueError('TEST split overlap or identity mismatch')
    split = (torch.from_numpy(x.transpose(0, 3, 1, 2).copy()), torch.from_numpy(y.astype(np.int64)), ids)
    model = model_factory(m['protocol'], m['arguments'], len(manifest['classes']), result['temperature']).cuda()
    model.load_state_dict(torch.load(out / 'best_checkpoint.pt', map_location='cpu', weights_only=False)['model'])
    metrics, probabilities = evaluate(model, split, 2)
    np.savez_compressed(out / 'test_predictions.npz', ids=ids, labels=y,
                        scores=probabilities.numpy(), predictions=probabilities.argmax(1).numpy())
    save(out / 'test_result.json', dict(test=metrics, checkpoint_sha256=digest,
         predictions_sha256=sha(out / 'test_predictions.npz'), data_sha256=sha(cache),
         data_manifest_sha256=sha(manifest_path), source_commit=commit, source_files=sources,
         role='validation_selected_temperature_development_followup', temperature=result['temperature']))
    del model
    torch.cuda.empty_cache()


def supervise(cfg, out, profile):
    validate_profile(cfg)
    commit, sources = verify_sources()
    uuid = subprocess.check_output(['nvidia-smi', '-i', '3', '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
    if uuid != cfg['gpu_uuid']:
        raise ValueError('GPU3 identity changed')
    out.mkdir(parents=True, exist_ok=False)
    source, _, baseline, digest = source_identity(cfg, 17)
    save(out / 'plan.json', dict(profile=cfg, profile_sha256=sha(profile), source_commit=commit,
                                 source_files=sources, baseline_source=str(source), baseline_sha256=digest,
                                 pid=os.getpid(), role='bounded_temperature_pilot_then_validation_gated_rollout'))
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=uuid, CUDA_DEVICE_ORDER='PCI_BUS_ID', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    # This parent evaluates only after a child exits; two compute slots maximum.
    os.environ.update(CUDA_VISIBLE_DEVICES=uuid, CUDA_DEVICE_ORDER='PCI_BUS_ID')
    running, pids = {}, []

    def interrupted(signum, frame):
        raise KeyboardInterrupt(str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)

    def launch(seed, temperature, directory, source_sha=None):
        if int(subprocess.check_output(['nvidia-smi', '-i', '3', '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True)) < 6000:
            raise RuntimeError('Insufficient free GPU3 memory; preserve the run and report')
        log = (out / (directory.name + '.log')).open('x')
        command = [sys.executable, '-u', '-m', __package__ + '.route_temperature_tune',
                '--profile', str(profile), '--out', str(directory), '--worker', '--seed', str(seed),
                '--temperature', str(temperature)]
        if source_sha:
            command.extend(['--source-sha', source_sha])
        proc = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        running[proc.pid] = (proc, log, directory)
        pids.append(proc.pid)

    def collect(phase, test_when_complete=False):
        completed = []
        while running:
            for pid, (proc, log, directory) in list(running.items()):
                if proc.poll() is None:
                    continue
                log.close()
                del running[pid]
                if proc.returncode or not (directory / 'result.json').exists():
                    raise RuntimeError('Trial failed: ' + str(directory))
                completed.append(directory)
                if test_when_complete:
                    evaluate_selected(directory, cfg)
            save(out / 'status.json', dict(state=phase, scheduler_pid=os.getpid(),
                 running={str(pid):str(v[2]) for pid,v in running.items()}, completed=[str(p) for p in completed]))
            if running:
                time.sleep(10)
        return completed

    try:
        for t in cfg['temperatures']:
            launch(17, t, out / f'pilot_T{t:g}_s17')
        pilots = collect('pilot_training')
        results = [dict(json.loads((p / 'result.json').read_text()), run=str(p)) for p in pilots]
        chosen, rollout = select_pilot(results, baseline['val']['accuracy'], cfg)
        save(out / 'recipe_selection.json', dict(selected=chosen, rollout=rollout, candidates=results,
             baseline_val=baseline['val'], criterion='At least four additional correct VAL samples; no TEST selection'))
        # Lock the recipe before any TEST. Even unsuccessful pilots remain visible.
        evaluate_selected(Path(chosen['run']), cfg)
        finals = [Path(chosen['run'])]
        if rollout:
            locks = []
            for seed in cfg['rollout_seeds']:
                p, _, _, h = source_identity(cfg, seed)
                locks.append(dict(seed=seed, source=str(p), best_sha256=h))
            save(out / 'rollout_sources_lock.json', dict(sources=locks, target_temperature=chosen['target_temperature']))
            for locked in locks:
                seed = locked['seed']
                launch(seed, chosen['target_temperature'],
                       out / f'rollout_T{chosen["target_temperature"]:g}_s{seed}', locked['best_sha256'])
            finals.extend(collect('rollout_training', test_when_complete=True))
        rows = []
        for p in finals:
            result = json.loads((p / 'result.json').read_text())
            result.update(test=json.loads((p / 'test_result.json').read_text())['test'], run=str(p))
            rows.append(result)
        save(out / 'summary.json', dict(role='validation_selected_temperature_development', rollout=rollout,
                                        target_temperature=chosen['target_temperature'], rows=rows))
        save(out / 'status.json', dict(state='complete', scheduler_pid=os.getpid(), rollout=rollout, running={}))
    except BaseException as error:
        save(out / 'status.json', dict(state='failed', error=repr(error), scheduler_pid=os.getpid()))
        raise
    finally:
        for proc, log, _ in running.values():
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
            log.close()
        save(out / 'release_check.json', dict(child_pids=pids, children_reaped=True, timestamp=time.time(), gpu_uuid=uuid))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--profile', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--worker', action='store_true')
    ap.add_argument('--seed', type=int)
    ap.add_argument('--temperature', type=float)
    ap.add_argument('--source-sha')
    a = ap.parse_args()
    cfg = json.loads(a.profile.read_text())
    validate_profile(cfg)
    if a.worker:
        try:
            worker(cfg, a.out, a.seed, a.temperature, a.profile, a.source_sha)
        except BaseException as error:
            if a.out.exists():
                save(a.out / 'status.json', dict(state='failed', error=repr(error), pid=os.getpid()))
            raise
    else:
        supervise(cfg, a.out, a.profile)


if __name__ == '__main__':
    main()
