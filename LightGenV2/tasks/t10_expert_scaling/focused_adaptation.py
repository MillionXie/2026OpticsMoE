"""TRAIN-only class-balanced hard-example replay with less repeated evaluation."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import sys
import time

from .expert_tail_schedule import save, sha
from .fullk_refine import color_augmentation, kd_loss


def validate(cfg):
    if cfg['gpus'] != {'1': 'GPU-e8837b85-d55b-8e81-aaa5-ec1ac326932d', '3': 'GPU-4d8bfdb9-8777-05a6-3811-ab18ff4eadfd'}:
        raise ValueError('Only authorized GPUs1/3')
    if cfg['parallel_jobs'] != 3 or cfg['seed'] != 17 or cfg['recipes'] != ['color', 'phase_color', 'distill']:
        raise ValueError('Only the three seed17 methods')
    if cfg['recipe_gpu'] != {'color': 3, 'phase_color': 3, 'distill': 1}:
        raise ValueError('Resource allocation changed')
    if (cfg['samples_per_class'], cfg['hard_per_class'], cfg['refresh_rounds'], cfg['max_rounds']) != (128,64,5,20):
        raise ValueError('Focused sampling or finite budget changed')
    if cfg['adaptation_window_new_epochs'] != 10 or cfg['patience_after_window'] != 6:
        raise ValueError('Do not return to three-round early stopping')


def focused_indices(scores, labels, seed, round_index, per_class=128, hard=64):
    import numpy as np
    scores=np.asarray(scores); labels=np.asarray(labels).reshape(-1)
    if scores.shape != (len(labels),8) or not np.isfinite(scores).all():
        raise ValueError('Invalid TRAIN-only scores')
    selected=[]; hard_selected=[]
    for c in range(8):
        ix=np.flatnonzero(labels==c)
        if len(ix)<per_class or hard>=per_class:
            raise ValueError('Insufficient class coverage')
        other=scores[ix].copy(); other[:,c]=-np.inf
        margin=scores[ix,c]-other.max(1)
        ranked=ix[np.lexsort((ix,margin))]
        difficult=ranked[:hard]; rest=ranked[hard:].copy()
        np.random.default_rng(seed+c*100003).shuffle(rest)
        n=per_class-hard; offset=((round_index-1)*n)%len(rest)
        replay=np.take(rest,np.arange(offset,offset+n)%len(rest))
        selected.extend(difficult.tolist()+replay.tolist()); hard_selected.extend(difficult.tolist())
    selected=np.array(selected,dtype=np.int64)
    np.random.default_rng(seed+round_index*1000033).shuffle(selected)
    assert len(selected)==1024 and len(np.unique(selected))==1024
    return selected,np.array(hard_selected,dtype=np.int64)


def round_budget(source_full_epochs,cap=20):
    remaining=(20-source_full_epochs)*3496
    rounds=min(cap,remaining//1024)
    if rounds<10:raise ValueError('Insufficient remaining image budget for ten focused rounds')
    return remaining,rounds


def learning_rate(cfg, new_step, new_epochs):
    warmup = cfg['warmup_new_epochs']
    if new_step <= warmup:
        factor = .2 + .8 * (new_step - 1) / (warmup - 1)
    else:
        fraction = (new_step - warmup) / (new_epochs - warmup)
        factor = cfg['lr_end_fraction'] + (1 - cfg['lr_end_fraction']) * (1 + math.cos(math.pi * fraction)) / 2
    return cfg['new_stage_lr'] * factor


def updated_patience(cfg, new_stage_epoch, stale, improved):
    # The observation window is deliberately NOT counted toward patience.
    return 0 if new_stage_epoch <= cfg['adaptation_window_new_epochs'] or improved else stale + 1


def accumulate_phases(sums, named_parameters):
    import torch
    for name, parameter in named_parameters:
        phasor = torch.polar(torch.ones_like(parameter), parameter.detach())
        if name not in sums:
            sums[name] = phasor
        else:
            sums[name].add_(phasor)


def verify_sources():
    commit = os.environ['T10_FOCUSED_SOURCE_COMMIT']
    root = Path(subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], text=True).strip())
    paths = [Path(__file__), *[Path(__file__).with_name(n) for n in
             ['extended_adaptation.py', 'fullk_refine.py', 'model.py', 'train.py', 'plan.py', 'diagnose_topk.py']],
             Path(__file__).parent / 'configs/study.json', Path(__file__).parent/'configs/kather_n49_focused_adaptation_20261027.json', root / 'LightGenV2/common/optical_contract.py']
    hashes = {}
    for p in paths:
        name = p.resolve().relative_to(root).as_posix()
        if p.read_bytes() != subprocess.check_output(['git', 'show', commit + ':' + name]):
            raise ValueError('Published-source mismatch: ' + name)
        hashes[name] = sha(p)
    return commit, hashes


def lock_sources(cfg):
    if sha(Path(cfg['reference_run'])/'best_checkpoint.pt') != cfg['reference_sha256']:
        raise ValueError('Reference model changed')
    locks={}
    for recipe,directory in cfg['source_runs'].items():
        p=Path(directory); metadata=json.loads((p/'metadata.json').read_text()); history=json.loads((p/'history.json').read_text())
        extra=history[-1]['epoch']-120
        if not 4<=extra<20 or metadata['arguments']['seed']!=17 or metadata['arguments']['top_k']!=49:
            raise ValueError('Unexpected source continuation budget')
        round_budget(extra,cfg['max_rounds'])
        locks[recipe]=dict(run=directory,extra_epochs=extra,source_epoch=history[-1]['epoch'],metadata_sha256=sha(p/'metadata.json'),
            history_sha256=sha(p/'history.json'),best_sha256=sha(p/'best_checkpoint.pt'),last_sha256=sha(p/'last_checkpoint.pt'))
    return locks


def worker(cfg, lock, out, profile, recipe):
    import numpy as np
    import torch
    from .model import ScalingOptics
    from .train import data, images, evaluate, objective
    from .diagnose_topk import classification_margin_loss, save_route_audit
    commit, hashes = verify_sources()
    torch.set_num_threads(4)
    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    random.seed(cfg['seed'])
    source = Path(lock['run'])
    for name, key in [('metadata.json', 'metadata_sha256'), ('history.json', 'history_sha256'), ('best_checkpoint.pt', 'best_sha256'), ('last_checkpoint.pt', 'last_sha256')]:
        if sha(source / name) != lock[key]:
            raise ValueError('Continuation source changed after lock')
    original = json.loads((source / 'metadata.json').read_text())
    args = original['arguments']
    protocol = copy.deepcopy(original['protocol'])
    protocol['model']['feature_layers_provisional'] = args['layers']
    tc = protocol['training_proposal']
    tc.update(phase_noise_std=cfg['phase_noise_std'] if recipe == 'phase_color' else 0.,
              learning_rate=cfg['new_stage_lr'], checkpoint_selection='validation_accuracy_then_macro_nll')
    cache = Path(args['data'])
    if sha(cache) != original['data_sha256']:
        raise ValueError('Data identity changed')
    splits, manifest = data(cache)

    def make_model(k=49):
        return ScalingOptics(protocol, 49, k, 'moe_oeo', len(manifest['classes']), args['layers'], args['padding']).cuda()

    model = make_model()
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    average = copy.deepcopy(ema)
    state = torch.load(source / 'last_checkpoint.pt', map_location='cpu', weights_only=False)
    if state['epoch'] != 120 + lock['extra_epochs']:
        raise ValueError('Last checkpoint epoch differs from source history')
    model.load_state_dict(state['live_model'])
    ema.load_state_dict(state['model'])
    optimizer = torch.optim.AdamW([
        {'params': list(model.expert_phases), 'lr_scale': 1., 'group_name': 'expert'},
        {'params': list(model.global_phases), 'lr_scale': .5, 'group_name': 'global'},
        {'params': [model.router_phase], 'lr_scale': .1, 'group_name': 'router'}], lr=cfg['new_stage_lr'], weight_decay=0.)
    optimizer.load_state_dict(state['optimizer'])
    source_rng=state.get('rng')
    del state
    teacher = None
    if recipe == 'distill':
        teacher_path = Path(cfg['teacher_run'])
        if sha(teacher_path / 'best_checkpoint.pt') != cfg['teacher_sha256']:
            raise ValueError('Teacher changed')
        tm = json.loads((teacher_path / 'metadata.json').read_text())
        if tm['data_sha256'] != original['data_sha256']:
            raise ValueError('Teacher and student data differ')
        teacher = make_model(24).eval().requires_grad_(False)
        teacher.load_state_dict(torch.load(teacher_path / 'best_checkpoint.pt', map_location='cpu', weights_only=False)['model'])
    out.mkdir(parents=True, exist_ok=False)
    metadata = copy.deepcopy(original)
    for legacy in ['refinement', 'diagnosis', 'temperature_tune', 'teacher', 'extended_adaptation']:
        metadata.pop(legacy, None)
    metadata.update(command=sys.argv, git_commit=commit, source_files=hashes, protocol=protocol,
                    host_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                    source_run=str(source), continuation_source_lock=lock, reference_run=cfg['reference_run'],
                    source_last_sha256=lock['last_sha256'], source_best_sha256=lock['best_sha256'],
                    teacher_run=cfg['teacher_run'] if teacher is not None else None,
                    teacher_sha256=cfg['teacher_sha256'] if teacher is not None else None,
                    focused_adaptation=dict(recipe=recipe,configuration=cfg,optimizer_and_live_ema_preserved=True,source_rng_restored=source_rng is not None,
                    train_selection='TRAIN-only low-margin64 and rotating uniform replay64 per class',not_full_training_epochs=True),
                    profile_sha256=sha(profile), pid=os.getpid(), cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'])
    metadata['arguments'].update(out=str(out), epochs=None, lr=cfg['new_stage_lr'])
    save(out / 'metadata.json', metadata)
    shutil.copy2(source / 'best_checkpoint.pt', out / 'best_checkpoint.pt')
    best_state = torch.load(out / 'best_checkpoint.pt', map_location='cpu', weights_only=False)
    average.load_state_dict(best_state['model'])
    best_epoch = best_state['epoch']
    expected_best=best_state['val']['accuracy']
    del best_state
    inherited, _ = evaluate(average, splits['val'], 2)
    resume_val, _ = evaluate(ema, splits['val'], 2)
    expected_last = json.loads((source / 'history.json').read_text())[-1]['val']['accuracy']
    if abs(inherited['accuracy'] - expected_best) > 1.001 / 752 or abs(resume_val['accuracy'] - expected_last) > 1.001 / 752:
        raise ValueError('Selected or last EMA source does not reproduce its recorded validation')
    save(out / 'initial_validation.json', dict(inherited_best=inherited, resumed_last_ema=resume_val))
    best_key = (-inherited['accuracy'], inherited['macro_nll'])
    best_accuracy, best_nll = inherited['accuracy'], inherited['macro_nll']
    best_kind = 'inherited_best'
    stale, sums, average_count = 0, {}, 0
    x, y, _ = splits['train']
    weights = len(y) / (len(manifest['classes']) * torch.bincount(y, minlength=len(manifest['classes'])).float().cuda())
    history, started, last_progress = [], time.time(), 0.
    max_images,new_epochs=round_budget(lock['extra_epochs'],cfg['max_rounds'])
    probe_indices=torch.cat([torch.nonzero(y==c).flatten()[:64] for c in range(8)])
    probe=(x[probe_indices],y[probe_indices],splits['train'][2][probe_indices.numpy()])
    t0=time.time(); full_train, scores=evaluate(ema,splits['train'],cfg['eval_batch'])
    initial_scoring_seconds=time.time()-t0
    np.savez_compressed(out/'initial_train_predictions.npz',ids=splits['train'][2],labels=y.numpy(),scores=scores.numpy())
    save(out/'train_selection_identity.json',dict(split='train',ids_sha256=__import__('hashlib').sha256('\n'.join(splits['train'][2]).encode()).hexdigest(),
        data_sha256=original['data_sha256'],scores_sha256=sha(out/'initial_train_predictions.npz'),scoring_seconds=initial_scoring_seconds,
        sample_budget=max_images,max_rounds=new_epochs,samples_per_round=1024,source_full_epochs=lock['extra_epochs']))
    if source_rng is not None:
        torch.set_rng_state(source_rng['torch_cpu']); torch.cuda.set_rng_state(source_rng['torch_cuda'])
        np.random.set_state(source_rng['numpy']); random.setstate(source_rng['python'])

    def payload(candidate, epoch, val, kind):
        return dict(model=candidate.state_dict(), live_model=model.state_dict(), ema_model=ema.state_dict(),
                    optimizer=optimizer.state_dict(), epoch=epoch, val=val, selected_kind=kind, metadata=metadata,
                    phase_average_sums=sums, phase_average_count=average_count,
                    rng=dict(torch_cpu=torch.get_rng_state(), torch_cuda=torch.cuda.get_rng_state(),
                             numpy=np.random.get_state(), python=random.getstate()))

    for new_step in range(1,new_epochs+1):
        epoch=lock['source_epoch']+new_step
        total_epoch=lock['extra_epochs']+new_step*1024/3496
        scoring_seconds=0.
        if new_step>1 and (new_step-1)%cfg['refresh_rounds']==0:
            t0=time.time();full_train,scores=evaluate(ema,splits['train'],cfg['eval_batch']);scoring_seconds=time.time()-t0
            np.savez_compressed(out/'refreshed_train_predictions.npz',ids=splits['train'][2],labels=y.numpy(),scores=scores.numpy(),round=new_step)
        model.train()
        lr = learning_rate(cfg, new_step, new_epochs)
        for group in optimizer.param_groups:
            group['lr'] = lr * group['lr_scale']
        kd_weight = cfg['kd_weight_start'] + (cfg['kd_weight_target'] - cfg['kd_weight_start']) * min(1., (new_step - 1) / (cfg['kd_warmup_new_epochs'] - 1))
        selected,hard_selected=focused_indices(scores.numpy(),y.numpy(),cfg['seed'],new_step,cfg['samples_per_class'],cfg['hard_per_class'])
        order=torch.from_numpy(selected)
        save(out/'current_train_selection.json',dict(round=new_step,train_indices=selected.tolist(),hard_indices=hard_selected.tolist(),
             ids=splits['train'][2][selected].tolist(),class_counts=torch.bincount(y[order],minlength=8).tolist(),
             label_source='TRAIN only; no TEST sample or label enters selection'))
        torch.cuda.synchronize(); training_started=time.time()
        totals = dict(supervised=0., kd=0., eligible=0.)
        batches = 0
        for start in range(0, len(order), 16):
            effective = order[start:start + 16]
            optimizer.zero_grad(set_to_none=True)
            for offset in range(0, len(effective), 2):
                ix = effective[offset:offset + 2]
                batch = color_augmentation(images(x, ix, epoch, cfg['seed'], True), ix, epoch, cfg['seed'], cfg)
                output = model(batch)
                labels = y[ix].cuda()
                supervised = objective(model, output, labels, weights, tc) + .2 * classification_margin_loss(output['probabilities'], labels)
                auxiliary = supervised.new_zeros(())
                eligible = supervised.new_zeros(())
                if teacher is not None:
                    with torch.no_grad():
                        target = teacher(batch)['probabilities']
                    auxiliary, eligible = kd_loss(output['probabilities'], target, labels, cfg['kd_temperature'])
                    auxiliary = kd_weight * auxiliary
                fraction = len(ix) / len(effective)
                loss = (supervised + auxiliary) * fraction
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite objective')
                loss.backward()
                totals['supervised'] += float(supervised.detach()) * fraction
                totals['kd'] += float(auxiliary.detach()) * fraction
                totals['eligible'] += float(eligible) * fraction
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            with torch.no_grad():
                for ep, parameter in zip(ema.parameters(), model.parameters()):
                    ep.lerp_(parameter, 1 - tc['ema_decay'])
            batches += 1
            if time.time() - last_progress >= 60:
                save(out / 'batch_progress.json', dict(epoch=epoch, adaptation_epoch=total_epoch, optimizer_batches=batches,
                     batches_per_epoch=math.ceil(len(order) / 16), focused_round=new_step, samples_per_round=1024, gradient_norm=float(norm),
                     supervised=totals['supervised'] / batches, kd=totals['kd'] / batches,
                     timestamp=time.time(), pid=os.getpid()))
                last_progress = time.time()
        torch.cuda.synchronize(); training_seconds=time.time()-training_started
        t0=time.time();train,_=evaluate(ema,probe,cfg['eval_batch']);probe_seconds=time.time()-t0
        t0=time.time();val,val_scores=evaluate(ema,splits['val'],cfg['eval_batch']);validation_seconds=time.time()-t0
        np.savez_compressed(out/'latest_val_predictions.npz',ids=splits['val'][2],labels=splits['val'][1].numpy(),scores=val_scores.numpy())
        candidates = [('ema', ema, val)]
        average_val = None
        if new_step >= cfg['swa_start_new_stage_epoch']:
            with torch.no_grad():
                accumulate_phases(sums, ema.named_parameters())
                average_count += 1
                for name, p in average.named_parameters():
                    p.copy_(sums[name].angle())
            average_val, _ = evaluate(average, splits['val'], 2)
            candidates.append(('circular_swa', average, average_val))
        improved = any(v['accuracy'] > best_accuracy or v['macro_nll'] < best_nll for _, _, v in candidates)
        for kind, candidate, metrics in candidates:
            key = (-metrics['accuracy'], metrics['macro_nll'])
            if key < best_key:
                best_key, best_epoch, best_kind = key, epoch, kind
                torch.save(payload(candidate, epoch, metrics, kind), out / 'best_checkpoint.pt')
        best_accuracy = max([best_accuracy] + [v['accuracy'] for _, _, v in candidates])
        best_nll = min([best_nll] + [v['macro_nll'] for _, _, v in candidates])
        stale = updated_patience(cfg, new_step, stale, improved)
        gradients = {name: dict(norm=float(p.grad.norm()), finite=bool(torch.isfinite(p.grad).all()))
                     for name, p in model.named_parameters() if p.grad is not None}
        entry = dict(epoch=epoch, adaptation_epoch=total_epoch, new_stage_epoch=new_step, focused_round=new_step,
                     train_probe_samples=512, train_samples=1024, training_seconds=training_seconds,probe_seconds=probe_seconds,
                     validation_seconds=validation_seconds,scoring_seconds=scoring_seconds, optimized_images=new_step*1024,
                     full_train_accuracy_on_last_refresh=full_train['accuracy'],train=train, val=val,
                     circular_swa_val=average_val, circular_swa_count=average_count, gradient_norm=float(norm),
                     gradients=gradients, supervised_loss=totals['supervised'] / batches, kd_loss=totals['kd'] / batches,
                     kd_eligible_fraction=totals['eligible'] / batches, kd_weight=kd_weight if teacher is not None else 0.,
                     learning_rates={g['group_name']: g['lr'] for g in optimizer.param_groups},
                     patience=stale, elapsed_seconds=time.time() - started)
        history.append(entry)
        save(out / 'history.json', history)
        torch.save(payload(ema, epoch, val, 'last_ema'), out / 'last_checkpoint.pt')
        save(out / 'status.json', dict(state='training', pid=os.getpid(), epoch=epoch, adaptation_epoch=total_epoch,
             max_focused_rounds=new_epochs, best_epoch=best_epoch, best_kind=best_kind, patience=stale))
        print(json.dumps(entry), flush=True)
        if stale >= cfg['patience_after_window']:
            break
    selected = torch.load(out / 'best_checkpoint.pt', map_location='cpu', weights_only=False)
    average.load_state_dict(selected['model'])
    del selected
    train, _ = evaluate(average, splits['train'], 2)
    val, _ = evaluate(average, splits['val'], 2)
    for name in ['train', 'val']:
        save_route_audit(average, splits[name], out / f'selected_{name}_routes.npz')
    result = dict(recipe=recipe, seed=17, source_adaptation_epochs=lock['extra_epochs'],
                  focused_rounds=len(history), optimized_train_images=len(history)*1024, total_full_epoch_equivalent=history[-1]['adaptation_epoch'],
                  best_epoch=best_epoch, best_kind=best_kind, train=train, val=val, run=str(out),
                  checkpoint_sha256=sha(out / 'best_checkpoint.pt'), test_read=False, source_commit=commit)
    save(out / 'result.json', result)
    save(out / 'status.json', dict(state='validation_selected', **result))


def test_selected(out, cfg):
    import numpy as np
    import torch
    from .model import ScalingOptics
    from .train import evaluate
    commit, hashes = verify_sources()
    m = json.loads((out / 'metadata.json').read_text())
    result = json.loads((out / 'result.json').read_text())
    if (out / 'test_result.json').exists() or sha(out / 'best_checkpoint.pt') != result['checkpoint_sha256']:
        raise ValueError('Already tested or changed selection')
    save(out / 'test_selection_lock.json', dict(checkpoint_sha256=result['checkpoint_sha256'], selection='VALaccuracy then NLL'))
    reference = Path(cfg['reference_run'])
    if sha(reference / 'best_checkpoint.pt') != cfg['reference_sha256']:
        raise ValueError('Reference weight changed')
    current = torch.load(out / 'best_checkpoint.pt', map_location='cpu', weights_only=False)['model']
    original = torch.load(reference / 'best_checkpoint.pt', map_location='cpu', weights_only=False)['model']
    identical = current.keys() == original.keys() and all(torch.equal(current[k], original[k]) for k in current)
    if identical:
        old = json.loads((reference / 'test_result.json').read_text())
        if old['checkpoint_sha256'] != cfg['reference_sha256'] or old['data_sha256'] != m['data_sha256']:
            raise ValueError('Cached TEST identity changed')
        for name, digest in old['evaluation_dependencies'].items():
            if sha(Path(__file__).with_name(name)) != digest:
                raise ValueError('Cached inference dependency changed')
        if sha(reference / 'test_predictions.npz') != old['predictions_sha256']:
            raise ValueError('Cached predictions changed')
        shutil.copy2(reference / 'test_predictions.npz', out / 'test_predictions.npz')
        old.update(checkpoint_sha256=result['checkpoint_sha256'], reused_from=str(reference),
                   reused_source_checkpoint_sha256=cfg['reference_sha256'], source_commit=commit,
                   evaluation_role='Tensor-identical original model; no GPU inference')
        save(out / 'test_result.json', old)
        save(out / 'test_reuse_receipt.json', dict(identical_weight_tensors=True, gpu_forward_executed=False, source=str(reference)))
        return
    del original
    cache = Path(m['arguments']['data'])
    manifest_path = cache.parent / 'data_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if sha(cache) != m['data_sha256'] or sha(cache) != manifest['cache_sha256']:
        raise ValueError('TEST data changed')
    with np.load(cache, allow_pickle=False) as z:
        x, y, ids = z['test_images'], z['test_labels'].reshape(-1), z['test_ids'].astype(str)
    if ids.tolist() != manifest['split_ids']['test'] or set(ids) & (set(manifest['split_ids']['train']) | set(manifest['split_ids']['val'])):
        raise ValueError('TEST identities or overlap changed')
    split = (torch.from_numpy(x.transpose(0, 3, 1, 2).copy()), torch.from_numpy(y.astype(np.int64)), ids)
    model = ScalingOptics(m['protocol'], 49, 49, 'moe_oeo', len(manifest['classes']), m['arguments']['layers'], m['arguments']['padding']).cuda()
    model.load_state_dict(current)
    metrics, probabilities = evaluate(model, split, 2)
    np.savez_compressed(out / 'test_predictions.npz', ids=ids, labels=y, scores=probabilities.numpy(), predictions=probabilities.argmax(1).numpy())
    save(out / 'test_result.json', dict(test=metrics, checkpoint_sha256=result['checkpoint_sha256'],
         predictions_sha256=sha(out / 'test_predictions.npz'), data_sha256=sha(cache), data_manifest_sha256=sha(manifest_path),
         source_commit=commit, source_files=hashes, role='TRAIN-only focused replay; TEST-informed development diagnosis'))
    del model
    torch.cuda.empty_cache()


def process_identity(pid,root):
    p=Path('/proc')/str(pid)
    if not p.exists():return None
    command=(p/'cmdline').read_bytes()
    if str(root).encode() not in command or b'extended_adaptation' not in command:
        raise RuntimeError('Unrelated PID; never signal it')
    return dict(pid=int(pid),start_ticks=(p/'stat').read_text().split()[21],command=command.decode(errors='replace'))


def watchdog(receipt):
    info=json.loads(receipt.read_text());time.sleep(info['timeout_seconds'])
    for v in info['identities']:
        p=Path('/proc')/str(v['pid'])
        if p.exists() and (p/'stat').read_text().split()[21]==v['start_ticks'] and v['command'].encode()==(p/'cmdline').read_bytes():
            os.kill(v['pid'],signal.SIGCONT)


def handoff(cfg,out):
    root=Path(cfg['active_source_root']);state=json.loads((root/'status.json').read_text())
    if state['state']!='training' or len(state['running'])!=3:
        raise RuntimeError('Source state changed; inspect instead of duplicate scheduling')
    identities=[process_identity(state['scheduler_pid'],root)]+[process_identity(pid,root) for pid in state['running']]
    if any(v is None for v in identities):raise RuntimeError('Source process disappeared')
    save(out/'handoff_snapshot.json',dict(source_status=state,identities=identities,timestamp=time.time()))
    guard_file=out/'handoff_watchdog.json';save(guard_file,dict(identities=identities,timeout_seconds=1500))
    guard=subprocess.Popen([sys.executable,'-m',__package__+'.focused_adaptation','--watchdog',str(guard_file)],start_new_session=True,
                           stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    frozen=[];completed=False
    try:
        os.kill(state['scheduler_pid'],signal.SIGSTOP)
        baseline={int(pid):(Path(d)/'last_checkpoint.pt').stat().st_mtime_ns for pid,d in state['running'].items()}
        previous={};deadline=time.time()+1200
        while len(frozen)<3:
            if time.time()>deadline:raise RuntimeError('Boundary handoff deadline; restore original training')
            for pid,d in state['running'].items():
                pid=int(pid)
                if pid in frozen:continue
                process_identity(pid,root)
                q=Path(d);f=q/'last_checkpoint.pt';stat=f.stat();stamp=(stat.st_mtime_ns,stat.st_size)
                if stat.st_mtime_ns>baseline[pid] and previous.get(pid)==stamp:
                    os.kill(pid,signal.SIGSTOP)
                    import torch
                    try:
                        checkpoint=torch.load(f,map_location='cpu',weights_only=False)
                        h=json.loads((q/'history.json').read_text());assert checkpoint['epoch']==h[-1]['epoch']
                        assert all(k in checkpoint for k in ['live_model','model','optimizer','rng'])
                        del checkpoint
                    except BaseException:
                        os.kill(pid,signal.SIGCONT);previous.pop(pid,None);continue
                    frozen.append(pid)
                    save(out/'handoff_progress.json',dict(frozen_pids=frozen,source_epoch=h[-1]['epoch'],timestamp=time.time()))
                previous[pid]=stamp
            time.sleep(1)
        locks=lock_sources(cfg);save(out/'source_locks.json',dict(sources=locks))
        for pid in frozen:
            process_identity(pid,root);os.kill(pid,signal.SIGTERM);os.kill(pid,signal.SIGCONT)
        process_identity(state['scheduler_pid'],root);os.kill(state['scheduler_pid'],signal.SIGTERM);os.kill(state['scheduler_pid'],signal.SIGCONT)
        deadline=time.time()+45
        while time.time()<deadline:
            if not any((Path('/proc')/str(v['pid'])).exists() for v in identities):break
            time.sleep(1)
        else:raise RuntimeError('Old process cleanup requires inspection; no competing launch')
        completed=True
        receipt=dict(state='superseded_at_checkpoint_boundary',new_run=str(out),source_locks=locks,identities=identities,
            source_status=state,timestamp=time.time(),checkpoint_rng_optimizer_ema_preserved=True,
            possible_discarded_work='At most the short next-batch window between completed checkpoint write and process freeze; source partial gradients were never called an exact mid-batch resume.')
        save(root/'focused_handoff_receipt.json',receipt);save(root/'status.json',dict(state='superseded',scheduler_pid=state['scheduler_pid'],new_run=str(out)))
        save(out/'handoff_ack.json',receipt)
        return locks
    finally:
        if not completed:
            for v in identities:
                p=Path('/proc')/str(v['pid'])
                if p.exists() and (p/'stat').read_text().split()[21]==v['start_ticks']:os.kill(v['pid'],signal.SIGCONT)
        guard.terminate();guard.wait(timeout=5)


def supervise(cfg,out,profile):
    validate(cfg);commit,hashes=verify_sources();out.mkdir(parents=True,exist_ok=False)
    save(out/'plan.json',dict(configuration=cfg,source_commit=commit,source_files=hashes,pid=os.getpid(),profile_sha256=sha(profile)))
    save(out/'status.json',dict(state='waiting_for_checkpoint_handoff',scheduler_pid=os.getpid(),running={}))
    locks=handoff(cfg,out)
    for index,expected in cfg['gpus'].items():
        if subprocess.check_output(['nvidia-smi','-i',index,'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()!=expected:
            raise RuntimeError('GPU UUID changed')
    env=dict(os.environ,CUDA_DEVICE_ORDER='PCI_BUS_ID',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
    running={};completed=[];children=[]
    def stop(signum,frame):raise KeyboardInterrupt(str(signum))
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        for recipe in cfg['recipes']:
            gpu=cfg['recipe_gpu'][recipe]
            if int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))<7000:
                raise RuntimeError('Insufficient authorized GPU memory')
            directory=out/(recipe+'_s17');log=directory.with_suffix('.log').open('x')
            proc=subprocess.Popen([sys.executable,'-u','-m',__package__+'.focused_adaptation','--profile',str(profile),'--out',str(directory),
                 '--worker','--recipe',recipe,'--lock',str(out/'source_locks.json')],env=dict(env,CUDA_VISIBLE_DEVICES=cfg['gpus'][str(gpu)]),
                 stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            running[proc.pid]=(proc,log,directory);children.append(proc.pid)
        while running:
            for pid,(proc,log,directory) in list(running.items()):
                if proc.poll() is None:continue
                log.close();del running[pid]
                if proc.returncode:raise RuntimeError('Worker failed '+str(directory))
                completed.append(json.loads((directory/'result.json').read_text()))
            save(out/'status.json',dict(state='training',scheduler_pid=os.getpid(),running={str(k):str(v[2]) for k,v in running.items()},completed=completed))
            if running:time.sleep(10)
        selected=min(completed,key=lambda v:(-v['val']['accuracy'],v['val']['macro_nll']))
        save(out/'recipe_selection.json',dict(selected=selected,candidates=completed,criterion='VALaccuracy then NLL; TRAIN-only sampling'))
        save(out/'status.json',dict(state='testing',scheduler_pid=os.getpid(),running={}))
        os.environ['CUDA_VISIBLE_DEVICES']=cfg['gpus']['1'];test_selected(Path(selected['run']),cfg)
        selected['test']=json.loads((Path(selected['run'])/'test_result.json').read_text())['test']
        save(out/'summary.json',dict(selected=selected,candidates=completed,role='Single-seed TEST-observed targeted development'))
        save(out/'status.json',dict(state='complete',scheduler_pid=os.getpid(),running={}))
    finally:
        for proc,log,d in running.values():
            if proc.poll() is None:os.killpg(proc.pid,signal.SIGTERM);proc.wait(timeout=30)
            log.close()
        save(out/'release_check.json',dict(child_pids=children,children_reaped=True,authorized_gpus=cfg['gpus'],timestamp=time.time()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--watchdog',type=Path)
    parser.add_argument('--profile', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--recipe', choices=['color', 'phase_color', 'distill'])
    parser.add_argument('--lock', type=Path)
    args = parser.parse_args()
    if args.watchdog:
        watchdog(args.watchdog);return
    cfg = json.loads(args.profile.read_text())
    validate(cfg)
    if args.worker:
        try:
            lock = json.loads(args.lock.read_text())['sources'][args.recipe]
            worker(cfg, lock, args.out, args.profile, args.recipe)
        except BaseException as error:
            if args.out.exists():
                save(args.out / 'status.json', dict(state='failed', error=repr(error), pid=os.getpid()))
            raise
    else:
        try:
            supervise(cfg, args.out, args.profile)
        except BaseException as error:
            if args.out.exists():
                save(args.out / 'status.json', dict(state='failed', scheduler_pid=os.getpid(), error=repr(error)))
            raise


if __name__ == '__main__':
    main()
