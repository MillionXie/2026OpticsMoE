"""Paired, single-seed continuation of immutable historical top-k runs.

Keep geometry, loss, augmentation, Adam moments and EMA unchanged. Add 30
epochs with a predeclared 2e-4 -> 2e-5 cosine schedule. Select on validation
macro-NLL, including the original best. Test is evaluated once after selection.
"""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

import numpy as np
import torch

from .model import ScalingOptics, encode_rgb
from .train import data, evaluate, images, objective, save, sha


def selection_key(metrics, metric):
    if metric == 'accuracy':
        return (-metrics['accuracy'], metrics['macro_nll'])
    return (metrics['macro_nll'],)


def classification_margin_loss(probabilities, labels, margin=.2):
    logp = probabilities.clamp_min(1e-12).log()
    correct = logp.gather(1, labels[:, None]).squeeze(1)
    wrong = logp.scatter(1, labels[:, None], float('-inf')).max(1).values
    return (margin+wrong-correct).clamp_min(0).mean()


def load_balance_loss(probabilities, mask, k):
    """Load-aware auxiliary: N/k * sum_e(f_e * p_e), uniform baseline=1.

    f is the observed top-k inclusion fraction (detached); p is the differentiable
    probability averaged over the entire effective batch. This is a surrogate,
    not a claim that hard-mask variance itself has a derivative.
    """
    frequency = mask.detach().mean(0)
    return probabilities.shape[1]/k * (frequency*probabilities.mean(0)).sum()


@torch.no_grad()
def save_route_audit(model, split, path):
    """Per-sample router records, without augmentations or full-field propagation."""
    model.eval()
    x, y, ids = split
    probabilities, selections = [], []
    for start in range(0, len(y), 16):
        ix = torch.arange(start, min(start+16, len(y)))
        source = encode_rgb(images(x, ix, 0, 0), model.router_side)
        _, probability, mask = model.route(source, ste=False)
        probabilities.append(probability.cpu())
        selections.append(mask.cpu())
    p, m = torch.cat(probabilities).numpy(), torch.cat(selections).numpy()
    np.savez_compressed(path, ids=ids, labels=y.numpy(), route_probabilities=p, route_mask=m)
    _, frequencies = np.unique(m, axis=0, return_counts=True)
    save(path.with_suffix('.json'), dict(
        samples=len(y), selection_load=m.mean(0).tolist(), mean_probability=p.mean(0).tolist(),
        distinct_selected_sets=len(frequencies), most_common_set_fraction=float(frequencies.max()/len(y)),
        selection_by_class=[m[y.numpy()==i].mean(0).tolist() for i in range(model.classes)]))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--test-source', type=Path)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--batch', type=int, default=2)
    ap.add_argument('--hard-balance-weight', type=float, default=0.,
                    help='Additional load-aware routing auxiliary over effective batch16')
    ap.add_argument('--margin-weight', type=float, default=0.)
    ap.add_argument('--capture-weight', type=float)
    ap.add_argument('--selection-metric', choices=['macro_nll', 'accuracy'], default='macro_nll')
    a = ap.parse_args()
    if a.out.resolve() == a.source.resolve():
        raise ValueError('Source runs are immutable')
    if (a.epochs < 1 or a.lr <= 0 or a.hard_balance_weight < 0 or a.margin_weight < 0
            or (a.capture_weight is not None and a.capture_weight < 0)):
        raise ValueError('Invalid continuation budget')
    a.out.mkdir(parents=True, exist_ok=False)
    try:
        run(a)
    except BaseException as error:
        save(a.out/'status.json', dict(state='failed', error=repr(error), pid=os.getpid()))
        raise
    finally:
        torch.cuda.empty_cache()


def run(a):
    torch.set_num_threads(4)
    original = json.loads((a.source/'metadata.json').read_text())
    result = json.loads((a.source/'result.json').read_text())
    args = original['arguments']
    seed = int(args['seed'])
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cache = Path(args['data'])
    if sha(cache) != original['data_sha256']:
        raise ValueError('Training cache identity changed')
    if sha(a.source/'best_checkpoint.pt') != result['checkpoint_sha256']:
        raise ValueError('Historical selected checkpoint identity changed')
    splits, manifest = data(cache)
    cfg = copy.deepcopy(original['protocol'])
    tc = cfg['training_proposal']
    if a.capture_weight is not None:
        tc['capture_loss_weight'] = a.capture_weight
    model = ScalingOptics(cfg, int(args['experts']), int(args['top_k']), args['arch'],
                          len(manifest['classes']), int(args['layers']), int(args['padding'])).cuda()
    ema = copy.deepcopy(model).eval()
    ema.requires_grad_(False)
    state = torch.load(a.source/'last_checkpoint.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(state['live_model'])
    ema.load_state_dict(state['model'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0)
    optimizer.load_state_dict(state['optimizer'])
    origin_epoch = int(state['epoch'])
    del state
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    metadata = copy.deepcopy(original)
    metadata.update(git_commit=commit, protocol=cfg, pid=os.getpid(), device=torch.cuda.get_device_name(),
                    cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                    source_run=str(a.source.resolve()), source_last_sha256=sha(a.source/'last_checkpoint.pt'),
                    source_best_sha256=result['checkpoint_sha256'],
                    diagnosis=dict(extra_epochs=a.epochs, lr_start=a.lr, lr_end=a.lr*.1,
                                   optimizer_moments_preserved=True, test_role='development_followup',
                                   hard_balance_weight=a.hard_balance_weight,
                                   hard_balance_formula='N/k sum(detached_selection_frequency * mean_probability)',
                                   hard_balance_batch=16, original_soft_balance_preserved=True),
                    command=sys.argv)
    metadata['diagnosis'].update(margin_weight=a.margin_weight, log_probability_margin=.2,
                                selection_metric=a.selection_metric, capture_weight=tc['capture_loss_weight'])
    metadata['arguments'].update(out=str(a.out), epochs=origin_epoch+a.epochs)
    save(a.out/'metadata.json', metadata)
    shutil.copy2(a.source/'best_checkpoint.pt', a.out/'best_checkpoint.pt')
    best = selection_key(result['val'], a.selection_metric)
    best_epoch = result['best_epoch']
    x, y, _ = splits['train']
    counts = torch.bincount(y, minlength=len(manifest['classes'])).float().cuda()
    weights = len(y)/(len(manifest['classes'])*counts)
    history = []
    started = time.time()
    # Explicitly compare EMA and live models; legacy logs only evaluated EMA.
    for label, m in [('source_last_ema', ema), ('source_last_live', model)]:
        train_metrics, _ = evaluate(m, splits['train'], a.batch)
        val_metrics, _ = evaluate(m, splits['val'], a.batch)
        save(a.out/f'{label}.json', dict(train=train_metrics, val=val_metrics))
    for step in range(1, a.epochs+1):
        epoch = origin_epoch+step
        model.train()
        order = torch.randperm(len(y), generator=torch.Generator().manual_seed(seed+epoch*100003))
        lr = a.lr*(.1+.9*(1+math.cos(math.pi*(step-1)/max(1, a.epochs-1)))/2)
        for group in optimizer.param_groups:
            group['lr'] = lr
        total = 0.
        steps = 0
        auxiliary_total = 0.
        auxiliary_gradient_total = 0.
        classification_router_gradient_total = 0.
        for start in range(0, len(order), 16):
            effective = order[start:start+16]
            optimizer.zero_grad(set_to_none=True)
            for offset in range(0, len(effective), a.batch):
                ix = effective[offset:offset+a.batch]
                out = model(images(x, ix, epoch, seed, True), dense=False)
                labels = y[ix].cuda()
                loss = objective(model, out, labels, weights, tc)
                if a.margin_weight:
                    loss = loss+a.margin_weight*classification_margin_loss(out['probabilities'], labels)
                loss = loss*len(ix)/len(effective)
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite loss')
                loss.backward()
                total += float(loss.detach())
            if a.hard_balance_weight:
                # Recompute only the small router for the same augmented images.
                # All microbatches feed one batch16 load statistic; no full optical
                # graph needs to be retained across microbatch backward calls.
                classification_router_gradient_total += float(model.router_phase.grad.norm())
                source = encode_rgb(images(x, effective, epoch, seed, True), model.router_side)
                _, rp, mask = model.route(source, dense=False)
                auxiliary = a.hard_balance_weight*load_balance_loss(rp, mask, model.k)
                auxiliary_grad = torch.autograd.grad(auxiliary, model.router_phase, retain_graph=True)[0]
                auxiliary_gradient_total += float(auxiliary_grad.detach().norm())
                auxiliary.backward()
                auxiliary_total += float(auxiliary.detach())
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            with torch.no_grad():
                for ep, p in zip(ema.parameters(), model.parameters()):
                    ep.lerp_(p, 1-tc['ema_decay'])
            steps += 1
        vm, _ = evaluate(ema, splits['val'], a.batch)
        entry = dict(epoch=epoch, train_objective=total/steps, val=vm, lr=lr,
                     gradient_norm=float(norm), elapsed_seconds=time.time()-started)
        if a.hard_balance_weight:
            entry.update(load_auxiliary_mean=auxiliary_total/steps,
                         total_objective_mean=(total+auxiliary_total)/steps,
                         load_auxiliary_router_gradient_mean=auxiliary_gradient_total/steps,
                         classification_router_gradient_mean=classification_router_gradient_total/steps)
        if step % 5 == 0 or step == 1:
            entry['train'], _ = evaluate(ema, splits['train'], a.batch)
        history.append(entry)
        save(a.out/'history.json', history)
        payload = dict(model=ema.state_dict(), live_model=model.state_dict(),
                       optimizer=optimizer.state_dict(), epoch=epoch, metadata=metadata, val=vm)
        torch.save(payload, a.out/'last_checkpoint.pt')
        if selection_key(vm, a.selection_metric) < best:
            best, best_epoch = selection_key(vm, a.selection_metric), epoch
            torch.save(payload, a.out/'best_checkpoint.pt')
        save(a.out/'status.json', dict(state='training', epoch=epoch, total_epochs=origin_epoch+a.epochs,
                                       best_epoch=best_epoch, pid=os.getpid()))
        print(json.dumps(entry), flush=True)
    if a.hard_balance_weight:
        save_route_audit(ema, splits['val'], a.out/'last_val_routes.npz')
    chosen = torch.load(a.out/'best_checkpoint.pt', map_location='cpu', weights_only=False)
    ema.load_state_dict(chosen['model'])
    train_metrics, _ = evaluate(ema, splits['train'], a.batch)
    val_metrics, _ = evaluate(ema, splits['val'], a.batch)
    if a.hard_balance_weight:
        for split in ('train', 'val'):
            save_route_audit(ema, splits[split], a.out/f'selected_{split}_routes.npz')
    selected = dict(best_epoch=best_epoch, train=train_metrics, val=val_metrics,
                    checkpoint_sha256=sha(a.out/'best_checkpoint.pt'), git_commit=commit,
                    elapsed_seconds=time.time()-started, test_read=False)
    save(a.out/'result.json', selected)
    lock = a.out/'test_selection_lock.json'
    save(lock, dict(selection=f'validation_{a.selection_metric}_including_original_best',
                    runs=[dict(run_dir=str(a.out.resolve()), checkpoint_sha256=selected['checkpoint_sha256'])]))
    cmd = [sys.executable, '-m', 'LightGenV2.tasks.t10_expert_scaling.evaluate_test',
           '--run', str(a.out), '--data', str(cache), '--selection-lock', str(lock), '--batch', str(a.batch)]
    if a.test_source:
        cmd += ['--test-source', str(a.test_source)]
    subprocess.run(cmd, check=True)
    save(a.out/'status.json', dict(state='complete', pid=os.getpid(), **selected))


if __name__ == '__main__':
    main()
