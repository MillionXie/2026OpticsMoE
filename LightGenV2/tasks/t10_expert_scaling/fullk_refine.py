"""Bounded full-k training-method studies with unmodified optical inference."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import random
import signal
import shutil
import subprocess
import sys
import time

from .expert_tail_schedule import save, sha


def validate(cfg):
    if cfg['gpu_uuid'] != 'GPU-4d8bfdb9-8777-05a6-3811-ab18ff4eadfd':
        raise ValueError('Only GPU3 is authorized')
    robust = cfg.get('study_kind') == 'training_robustness'
    if cfg['epochs'] != (4 if robust else 6) or cfg['freeze_router_epochs'] != (0 if robust else 2) or cfg['patience'] != 3:
        raise ValueError('Keep the bounded, staged optimization budget')
    if cfg['pilot_seed'] != 17 or cfg['rollout_seeds'] != [27, 37]:
        raise ValueError('Seed identity changed')
    recipes = ['color', 'phase_color'] if robust else ['staged', 'distill']
    if cfg['recipes'] != recipes or cfg['batch'] != 16 or cfg['microbatch'] != 2:
        raise ValueError('Paired training contract changed')
    if cfg['required_validation_gain_correct'] != 4:
        raise ValueError('Rollout is selected using validation only')
    if cfg['lr'] != (1e-4 if robust else 5e-5) or cfg['group_lr_scales'] != {'expert': 1., 'global': .5, 'router': .1}:
        raise ValueError('Keep paired conservative parameter-group learning rates')
    if cfg['kd_weight'] != (0. if robust else .3) or cfg['kd_temperature'] != 2. or cfg['lr_end_fraction'] != .2:
        raise ValueError('Distillation or learning-rate budget changed')
    if robust and (cfg['color_gain_range'] != [.9, 1.1] or cfg['phase_noise_std'] != .02):
        raise ValueError('Keep bounded TRAIN-only image/phase perturbations')


def color_augmentation(batch, indices, epoch, seed, cfg):
    """RGB**gain is channel-wise optical-density scaling; black/white stay fixed."""
    low, high = cfg['color_gain_range']
    parameters = []
    for index in indices.tolist():
        rng = random.Random((seed * 1000003 + epoch * 1000033 + index) * 1000037 + 2718281)
        parameters.append([rng.uniform(low, high) for _ in range(3)])
    gain = batch.new_tensor(parameters)[:, :, None, None]
    return batch.clamp(0, 1).pow(gain)


def choose(rows, baseline):
    selected = min(rows, key=lambda v: (-v['val']['accuracy'], v['val']['macro_nll']))
    return selected, (selected['val']['accuracy'] - baseline) * 752 >= 4 - 1e-3


def sources():
    commit = os.environ['T10_REFINE_SOURCE_COMMIT']
    root = Path(subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], text=True).strip())
    paths = [Path(__file__), *[Path(__file__).with_name(n) for n in
             ['model.py', 'train.py', 'plan.py', 'diagnose_topk.py']],
             Path(__file__).parent / 'configs/study.json', root / 'LightGenV2/common/optical_contract.py']
    hashes = {}
    for p in paths:
        relative = p.resolve().relative_to(root).as_posix()
        if p.read_bytes() != subprocess.check_output(['git', 'show', commit + ':' + relative]):
            raise ValueError('Published source mismatch: ' + relative)
        hashes[relative] = sha(p)
    return commit, hashes


def identity(cfg, seed, teacher=False):
    prefix = 'teacher' if teacher else 'student'
    p = Path(cfg[prefix + '_root']) / f'N49_k{24 if teacher else 49}_s{seed}'
    metadata = json.loads((p / 'metadata.json').read_text())
    result = json.loads((p / 'result.json').read_text())
    history = json.loads((p / 'history.json').read_text())
    expected_epoch = 90 if teacher else 120
    if history[-1]['epoch'] != expected_epoch or metadata['arguments']['seed'] != seed:
        raise ValueError('Warm-start identity changed')
    if (metadata['arguments']['arch'], metadata['arguments']['experts'], metadata['arguments']['top_k']) != ('moe_oeo', 49, 24 if teacher else 49):
        raise ValueError('Source architecture identity changed')
    digest = sha(p / 'best_checkpoint.pt')
    if digest != result['checkpoint_sha256'] or digest != cfg[prefix + '_sha256'][str(seed)]:
        raise ValueError('Source selected weight changed')
    return p, metadata, result, digest


def kd_loss(student_prob, teacher_prob, labels, temperature):
    import torch
    import torch.nn.functional as F
    teacher = (teacher_prob.clamp_min(1e-12).log() / temperature).softmax(1).detach()
    student_log = (student_prob.clamp_min(1e-12).log() / temperature).log_softmax(1)
    index = torch.arange(len(labels), device=labels.device)
    eligible = ((teacher_prob.argmax(1) == labels) &
                (teacher_prob[index, labels] > student_prob.detach()[index, labels])).detach()
    per_sample = F.kl_div(student_log, teacher, reduction='none').sum(1) * temperature**2
    return (per_sample * eligible).mean(), eligible.float().mean()


def worker(cfg, out, profile, seed, recipe):
    import numpy as np
    import torch
    from .model import ScalingOptics
    from .train import data, images, evaluate, objective
    from .diagnose_topk import classification_margin_loss, save_route_audit

    commit, hashes = sources()
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    source, original, baseline, digest = identity(cfg, seed)
    args = original['arguments']
    cache = Path(args['data'])
    if sha(cache) != original['data_sha256']:
        raise ValueError('Student data changed')
    splits, manifest = data(cache)  # Only TRAIN and VAL arrays are read.
    protocol = copy.deepcopy(original['protocol'])
    tc = protocol['training_proposal']
    if cfg.get('study_kind') == 'training_robustness':
        tc['phase_noise_std'] = cfg['phase_noise_std'] if recipe == 'phase_color' else 0.
    model = ScalingOptics(protocol, 49, 49, 'moe_oeo', len(manifest['classes']), args['layers'], args['padding']).cuda()
    model.load_state_dict(torch.load(source / 'best_checkpoint.pt', map_location='cpu', weights_only=False)['model'])
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    teacher, teacher_identity = None, None
    if recipe == 'distill':
        tp, tm, tr, th = identity(cfg, seed, teacher=True)
        if tm['data_sha256'] != original['data_sha256'] or tm['geometry'] != original['geometry']:
            raise ValueError('Teacher/student data or physical geometry differ')
        teacher = ScalingOptics(tm['protocol'], 49, 24, 'moe_oeo', len(manifest['classes']), args['layers'], args['padding']).cuda()
        teacher.load_state_dict(torch.load(tp / 'best_checkpoint.pt', map_location='cpu', weights_only=False)['model'])
        teacher.eval().requires_grad_(False)
        teacher_identity = dict(run=str(tp), checkpoint_sha256=th, selected_epoch=tr['best_epoch'],
                                selected_val=tr['val'], usage='TRAIN augmented views only, never at inference')
    optimizer = torch.optim.AdamW([
        {'params': list(model.expert_phases), 'lr_scale': cfg['group_lr_scales']['expert'], 'group_name': 'expert'},
        {'params': list(model.global_phases), 'lr_scale': cfg['group_lr_scales']['global'], 'group_name': 'global'},
        {'params': [model.router_phase], 'lr_scale': cfg['group_lr_scales']['router'], 'group_name': 'router'}], lr=cfg['lr'], weight_decay=0.)
    out.mkdir(parents=True, exist_ok=False)
    metadata = copy.deepcopy(original)
    metadata.update(git_commit=commit, host_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                    protocol=protocol, command=sys.argv, source_files=hashes, profile_sha256=sha(profile),
                    source_run=str(source), source_best_sha256=digest, source_selected_epoch=baseline['best_epoch'],
                    teacher=teacher_identity, refinement=dict(recipe=recipe, configuration=cfg,
                    optimizer_reset=True, live_and_ema_from_selected_ema=True, exact_resume=False,
                    inference='Unmodified ScalingOptics N49/k49, original router temperature0.05'),
                    pid=os.getpid(), cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'])
    metadata.pop('diagnosis', None)
    metadata['arguments'].update(out=str(out), epochs=120 + cfg['epochs'], lr=cfg['lr'])
    save(out / 'metadata.json', metadata)
    initial, _ = evaluate(ema, splits['val'], 2)
    if abs(initial['accuracy'] - baseline['val']['accuracy']) > 1.001 / 752:
        raise ValueError('Unchanged forward does not reproduce source validation')
    save(out / 'initial_val.json', initial)
    best_key, best_epoch, best_nll_seen = (-initial['accuracy'], initial['macro_nll']), 120, initial['macro_nll']
    best_accuracy_seen = initial['accuracy']
    stale = 0

    def payload(epoch, val):
        return dict(model=ema.state_dict(), live_model=model.state_dict(), optimizer=optimizer.state_dict(),
                    epoch=epoch, metadata=metadata, val=val)

    torch.save(payload(120, initial), out / 'best_checkpoint.pt')
    x, y, _ = splits['train']
    weights = len(y) / (len(manifest['classes']) * torch.bincount(y, minlength=len(manifest['classes'])).float().cuda())
    history, started, last_progress = [], time.time(), 0.
    for step in range(1, cfg['epochs'] + 1):
        epoch = 120 + step
        model.train()
        model.router_phase.requires_grad_(step > cfg['freeze_router_epochs'])
        order = torch.randperm(len(y), generator=torch.Generator().manual_seed(seed + epoch * 100003))
        end = cfg['lr_end_fraction']
        lr = cfg['lr'] * (end + (1 - end) * (1 + math.cos(math.pi * (step - 1) / (cfg['epochs'] - 1))) / 2)
        for group in optimizer.param_groups:
            group['lr'] = lr * group['lr_scale']
        totals = dict(supervised=0., distillation=0., teacher_forwards=0, eligible_fraction=0.)
        batches = 0
        for start in range(0, len(order), 16):
            effective = order[start:start + 16]
            optimizer.zero_grad(set_to_none=True)
            for offset in range(0, len(effective), 2):
                ix = effective[offset:offset + 2]
                batch = images(x, ix, epoch, seed, True)
                if cfg.get('study_kind') == 'training_robustness':
                    batch = color_augmentation(batch, ix, epoch, seed, cfg)
                output = model(batch)
                labels = y[ix].cuda()
                supervised = objective(model, output, labels, weights, tc)
                supervised = supervised + .2 * classification_margin_loss(output['probabilities'], labels)
                distillation = supervised.new_zeros(())
                fraction = len(ix) / len(effective)
                if teacher is not None:
                    with torch.no_grad():
                        target = teacher(batch)['probabilities']
                    auxiliary, eligible = kd_loss(output['probabilities'], target, labels, cfg['kd_temperature'])
                    distillation = cfg['kd_weight'] * auxiliary
                    totals['teacher_forwards'] += 1
                    totals['eligible_fraction'] += float(eligible) * fraction
                loss = (supervised + distillation) * fraction
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite objective')
                loss.backward()
                totals['supervised'] += float(supervised.detach()) * fraction
                totals['distillation'] += float(distillation.detach()) * fraction
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            with torch.no_grad():
                for ep, parameter in zip(ema.parameters(), model.parameters()):
                    ep.lerp_(parameter, 1 - tc['ema_decay'])
            batches += 1
            if time.time() - last_progress >= 60:
                save(out / 'batch_progress.json', dict(epoch=epoch, optimizer_batches=batches,
                     batches_per_epoch=math.ceil(len(order) / 16), gradient_norm=float(norm),
                     supervised=totals['supervised'] / batches, kd=totals['distillation'] / batches,
                     timestamp=time.time(), pid=os.getpid()))
                last_progress = time.time()
        train, _ = evaluate(ema, splits['train'], 2)
        val, _ = evaluate(ema, splits['val'], 2)
        gradients = {name: dict(norm=float(p.grad.norm()), finite=bool(torch.isfinite(p.grad).all()))
                     for name, p in model.named_parameters() if p.grad is not None}
        entry = dict(epoch=epoch, train=train, val=val, learning_rates={g['group_name']: g['lr'] for g in optimizer.param_groups},
                     router_frozen=not model.router_phase.requires_grad, gradient_norm=float(norm), gradients=gradients,
                     supervised_loss=totals['supervised'] / batches, kd_loss=totals['distillation'] / batches,
                     kd_eligible_fraction=totals['eligible_fraction'] / batches,
                     phase_noise_std=model.phase_noise_std,
                     teacher_forwards=totals['teacher_forwards'], elapsed_seconds=time.time() - started)
        history.append(entry)
        save(out / 'history.json', history)
        torch.save(payload(epoch, val), out / 'last_checkpoint.pt')
        key = (-val['accuracy'], val['macro_nll'])
        if key < best_key:
            best_key, best_epoch = key, epoch
            torch.save(payload(epoch, val), out / 'best_checkpoint.pt')
        improved = val['accuracy'] > best_accuracy_seen or val['macro_nll'] < best_nll_seen
        stale = 0 if improved else stale + 1
        best_accuracy_seen = max(best_accuracy_seen, val['accuracy'])
        best_nll_seen = min(best_nll_seen, val['macro_nll'])
        save(out / 'status.json', dict(state='training', epoch=epoch, best_epoch=best_epoch, stale=stale, pid=os.getpid()))
        print(json.dumps(entry), flush=True)
        if stale >= cfg['patience']:
            break
    ema.load_state_dict(torch.load(out / 'best_checkpoint.pt', map_location='cpu', weights_only=False)['model'])
    train, _ = evaluate(ema, splits['train'], 2)
    val, _ = evaluate(ema, splits['val'], 2)
    for name in ['train', 'val']:
        save_route_audit(ema, splits[name], out / f'selected_{name}_routes.npz')
    result = dict(seed=seed, recipe=recipe, best_epoch=best_epoch, completed_extra_epochs=len(history), train=train, val=val,
                  initial_val=initial, checkpoint_sha256=sha(out / 'best_checkpoint.pt'), source_best_sha256=digest,
                  test_read=False, run=str(out), source_commit=commit)
    save(out / 'result.json', result)
    save(out / 'status.json', dict(state='validation_selected', **result))


def test_selected(out):
    import numpy as np
    import torch
    from .model import ScalingOptics
    from .train import evaluate
    commit, hashes = sources()
    m = json.loads((out / 'metadata.json').read_text())
    result = json.loads((out / 'result.json').read_text())
    if (out / 'test_result.json').exists() or sha(out / 'best_checkpoint.pt') != result['checkpoint_sha256']:
        raise ValueError('Already tested or changed checkpoint')
    save(out / 'test_selection_lock.json', dict(checkpoint_sha256=result['checkpoint_sha256'], criterion='VALaccuracy then NLL'))
    if result['best_epoch'] == 120:
        source = Path(m['source_run'])
        old_result = json.loads((source / 'test_result.json').read_text())
        if old_result['checkpoint_sha256'] != m['source_best_sha256'] or old_result['data_sha256'] != m['data_sha256']:
            raise ValueError('Original selected TEST identity changed')
        original_state = torch.load(source / 'best_checkpoint.pt', map_location='cpu', weights_only=False)['model']
        selected_state = torch.load(out / 'best_checkpoint.pt', map_location='cpu', weights_only=False)['model']
        if original_state.keys() != selected_state.keys() or not all(torch.equal(original_state[k], selected_state[k]) for k in original_state):
            raise ValueError('Source candidate is not tensor-identical')
        for name, digest in old_result['evaluation_dependencies'].items():
            if sha(Path(__file__).with_name(name)) != digest:
                raise ValueError('Original inference dependency changed')
        if sha(source / 'test_predictions.npz') != old_result['predictions_sha256']:
            raise ValueError('Original TEST predictions changed')
        shutil.copy2(source / 'test_predictions.npz', out / 'test_predictions.npz')
        reused = copy.deepcopy(old_result)
        reused.update(checkpoint_sha256=result['checkpoint_sha256'], source_commit=commit, source_files=hashes,
                      evaluation_role='Reused original TEST after exact weight tensor and inference identity check',
                      reused_from=str(source), reused_source_checkpoint_sha256=m['source_best_sha256'])
        save(out / 'test_result.json', reused)
        save(out / 'test_reuse_receipt.json', dict(source=str(source), identical_weight_tensors=True,
             unchanged_inference_dependencies=True, gpu_forward_executed=False,
             source_predictions_sha256=old_result['predictions_sha256']))
        return
    cache = Path(m['arguments']['data'])
    manifest_path = cache.parent / 'data_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if sha(cache) != m['data_sha256'] or sha(cache) != manifest['cache_sha256']:
        raise ValueError('TEST source identity changed')
    with np.load(cache, allow_pickle=False) as z:
        x, y, ids = z['test_images'], z['test_labels'].reshape(-1), z['test_ids'].astype(str)
    if ids.tolist() != manifest['split_ids']['test'] or set(ids) & (set(manifest['split_ids']['train']) | set(manifest['split_ids']['val'])):
        raise ValueError('TEST sample identity or overlap')
    split = (torch.from_numpy(x.transpose(0, 3, 1, 2).copy()), torch.from_numpy(y.astype(np.int64)), ids)
    model = ScalingOptics(m['protocol'], 49, 49, 'moe_oeo', len(manifest['classes']), m['arguments']['layers'], m['arguments']['padding']).cuda()
    model.load_state_dict(torch.load(out / 'best_checkpoint.pt', map_location='cpu', weights_only=False)['model'])
    metrics, probabilities = evaluate(model, split, 2)
    np.savez_compressed(out / 'test_predictions.npz', ids=ids, labels=y, scores=probabilities.numpy(), predictions=probabilities.argmax(1).numpy())
    save(out / 'test_result.json', dict(test=metrics, checkpoint_sha256=result['checkpoint_sha256'],
         predictions_sha256=sha(out / 'test_predictions.npz'), data_sha256=sha(cache), data_manifest_sha256=sha(manifest_path),
         source_commit=commit, source_files=hashes, role='TRAIN_only_teacher_or_staged_development_no_teacher_at_inference'))
    del model
    torch.cuda.empty_cache()


def supervise(cfg, out, profile):
    validate(cfg)
    commit, hashes = sources()
    if subprocess.check_output(['nvidia-smi', '-i', '3', '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip() != cfg['gpu_uuid']:
        raise ValueError('GPU3 identity changed')
    out.mkdir(parents=True, exist_ok=False)
    _, _, baseline, _ = identity(cfg, 17)
    save(out / 'plan.json', dict(configuration=cfg, profile_sha256=sha(profile), pid=os.getpid(), source_commit=commit,
                                source_files=hashes, role='Post-test development, unmodified inference'))
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=cfg['gpu_uuid'], CUDA_DEVICE_ORDER='PCI_BUS_ID', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    os.environ.update(CUDA_VISIBLE_DEVICES=cfg['gpu_uuid'], CUDA_DEVICE_ORDER='PCI_BUS_ID')
    running, children = {}, []

    def interrupt(signum, frame):
        raise KeyboardInterrupt(str(signum))
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)

    def launch(seed, recipe):
        if int(subprocess.check_output(['nvidia-smi', '-i', '3', '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True)) < 7000:
            raise RuntimeError('Insufficient free GPU3 memory for the next bounded job')
        directory = out / f'{recipe}_s{seed}'
        log = directory.with_suffix('.log').open('x')
        child = subprocess.Popen([sys.executable, '-u', '-m', __package__ + '.fullk_refine', '--profile', str(profile),
                '--out', str(directory), '--worker', '--seed', str(seed), '--recipe', recipe], env=env,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        running[child.pid] = (child, log, directory)
        children.append(child.pid)

    def collect(phase, test_when_complete=False):
        complete = []
        while running:
            for pid, (child, log, directory) in list(running.items()):
                if child.poll() is None:
                    continue
                log.close()
                del running[pid]
                if child.returncode or not (directory / 'result.json').exists():
                    raise RuntimeError('Trial failed: ' + str(directory))
                complete.append(json.loads((directory / 'result.json').read_text()))
                if test_when_complete:
                    test_selected(directory)
            save(out / 'status.json', dict(state=phase, scheduler_pid=os.getpid(),
                 running={str(pid): str(v[2]) for pid, v in running.items()}, completed=complete))
            if running:
                time.sleep(10)
        return complete

    try:
        for recipe in cfg['recipes']:
            launch(17, recipe)
        candidates = collect('pilot_training')
        selected, rollout = choose(candidates, baseline['val']['accuracy'])
        save(out / 'recipe_selection.json', dict(selected=selected, rollout=rollout, candidates=candidates,
             criterion='VALaccuracy/NLL; >=4 additional correct VAL examples for rollout; no TEST selection'))
        test_selected(Path(selected['run']))
        final = [selected]
        if rollout:
            locks = []
            for seed in cfg['rollout_seeds']:
                _, _, _, h = identity(cfg, seed)
                _, _, _, th = identity(cfg, seed, teacher=True)
                locks.append(dict(seed=seed, student_sha256=h, teacher_sha256=th))
            save(out / 'rollout_sources_lock.json', dict(recipe=selected['recipe'], sources=locks))
            for seed in cfg['rollout_seeds']:
                launch(seed, selected['recipe'])
            final.extend(collect('rollout_training', test_when_complete=True))
        for row in final:
            row['test'] = json.loads((Path(row['run']) / 'test_result.json').read_text())['test']
        save(out / 'summary.json', dict(role='Post-test development, unmodified full-k inference', rows=final, rollout=rollout))
        save(out / 'status.json', dict(state='complete', scheduler_pid=os.getpid(), running={}))
    except BaseException as error:
        save(out / 'status.json', dict(state='failed', error=repr(error), scheduler_pid=os.getpid()))
        raise
    finally:
        for child, log, _ in running.values():
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
            log.close()
        save(out / 'release_check.json', dict(child_pids=children, children_reaped=True, gpu_uuid=cfg['gpu_uuid'], timestamp=time.time()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--recipe', choices=['staged', 'distill', 'color', 'phase_color'])
    args = parser.parse_args()
    cfg = json.loads(args.profile.read_text())
    validate(cfg)
    if args.worker:
        try:
            worker(cfg, args.out, args.profile, args.seed, args.recipe)
        except BaseException as error:
            if args.out.exists():
                save(args.out / 'status.json', dict(state='failed', error=repr(error), pid=os.getpid()))
            raise
    else:
        supervise(cfg, args.out, args.profile)


if __name__ == '__main__':
    main()
