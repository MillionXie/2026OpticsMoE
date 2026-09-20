"""Offline joint four-pathology-dataset training for a standard D2NN."""
import argparse
import json
import math
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from .cross_dataset import balanced_subset, load_dataset
from .data import sha
from .model import OpticalD2NN, loss
from .run import save


def joint_epoch_indices(lengths, per_task, steps, rng):
    """Return deterministic task-balanced indices, reshuffling on every wrap."""
    pools = [rng.permutation(n) for n in lengths]
    offsets = [0] * len(lengths)
    for _ in range(steps):
        batch = []
        for task, n in enumerate(lengths):
            chosen = []
            while len(chosen) < per_task:
                remaining = n - offsets[task]
                take = min(per_task - len(chosen), remaining)
                chosen.extend(pools[task][offsets[task]:offsets[task] + take].tolist())
                offsets[task] += take
                if offsets[task] == n:
                    pools[task] = rng.permutation(n)
                    offsets[task] = 0
            batch.append(np.asarray(chosen, dtype=np.int64))
        yield batch


@torch.no_grad()
def evaluate(model, images, labels, batch_size):
    model.eval(); probabilities = []
    device = next(model.parameters()).device
    for start in range(0, len(labels), batch_size):
        probabilities.append(model(images[start:start+batch_size].to(device))['probabilities'].cpu())
    p = torch.cat(probabilities); pred = p.argmax(1)
    confusion = torch.bincount(labels * 2 + pred, minlength=4).reshape(2, 2)
    recall = confusion.diag().float() / confusion.sum(1).clamp_min(1)
    metrics = {
        'accuracy': float((pred == labels).float().mean()),
        'balanced_accuracy': float(recall.mean()),
        'nll': float(torch.nn.functional.nll_loss(p.clamp_min(1e-12).log(), labels)),
        'recall_non_tumor': float(recall[0]),
        'recall_tumor': float(recall[1]),
        'confusion': confusion.tolist(),
    }
    return metrics, p


def train_epoch(model, optimizer, tasks, per_task, steps, rng):
    model.train(); total = correct = count = 0
    for ids_by_task in joint_epoch_indices([len(t['y']) for t in tasks], per_task, steps, rng):
        xb = torch.cat([task['x'][ids] for task, ids in zip(tasks, ids_by_task)])
        yb = torch.cat([task['y'][ids] for task, ids in zip(tasks, ids_by_task)])
        order = torch.from_numpy(rng.permutation(len(yb)))
        xb, yb = xb[order], yb[order]
        device = next(model.parameters()).device
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        output = model(xb); value = loss(output, yb)
        if not torch.isfinite(value):
            raise RuntimeError('Nonfinite loss')
        value.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        total += value.item() * len(yb)
        correct += int((output['probabilities'].argmax(1) == yb).sum())
        count += len(yb)
    return {'nll': total / count, 'accuracy_online': correct / count, 'samples': count}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    for name in 'abcd':
        parser.add_argument('--task-' + name, type=Path, required=True)
        parser.add_argument('--task-' + name + '-manifest', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--pilot', action='store_true')
    args = parser.parse_args(); cfg = json.loads(args.config.read_text())
    if cfg['batch_size'] % 4:
        raise ValueError('batch_size must be divisible by four for task-balanced batches')
    if args.pilot:
        cfg['epochs'] = 1; cfg['steps_per_epoch'] = min(4, cfg['steps_per_epoch'])
    args.out.mkdir(parents=True, exist_ok=False)
    save(args.out / 'status.json', {'state': 'preparing'})
    try:
        torch.manual_seed(cfg['seed']); np.random.seed(cfg['seed']); torch.set_num_threads(4)
        rng = np.random.default_rng(cfg['seed']); raw = []; manifests = []
        for name in 'abcd':
            data, manifest = load_dataset(getattr(args, 'task_' + name),
                                          getattr(args, 'task_' + name + '_manifest'))
            raw.append(data); manifests.append(manifest)
        tasks = []
        for i, (name, data) in enumerate(zip('ABCD', raw)):
            train_ids = balanced_subset(data['train_labels'], cfg['train_per_class_' + name], cfg['seed'] + i)
            val_ids = balanced_subset(data['val_labels'], cfg['val_per_class_' + name], cfg['seed'] + 10 + i)
            tasks.append({
                'x': torch.from_numpy(data['train_images'][train_ids]),
                'y': torch.from_numpy(data['train_labels'][train_ids]).long(),
                'vx': torch.from_numpy(data['val_images'][val_ids]),
                'vy': torch.from_numpy(data['val_labels'][val_ids]).long(),
                'train_ids': data['train_ids'][train_ids],
                'val_ids': data['val_ids'][val_ids],
            })
        per_task = cfg['batch_size'] // 4
        minimum_steps = max(math.ceil(len(task['y']) / per_task) for task in tasks)
        if cfg['steps_per_epoch'] < minimum_steps and not args.pilot:
            raise ValueError(f"steps_per_epoch={cfg['steps_per_epoch']} does not cover the largest task ({minimum_steps})")
        save(args.out / 'config.json', cfg)
        save(args.out / 'split.json', {
            name: {'train_ids': task['train_ids'].tolist(), 'val_ids': task['val_ids'].tolist()}
            for name, task in zip('ABCD', tasks)
        } | {'test_images_read': False})
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
        save(args.out / 'metadata.json', {
            'command': sys.argv, 'commit': commit, 'python': sys.version,
            'torch': torch.__version__, 'platform': platform.platform(),
            'device': args.device, 'manifests': manifests,
            'data_sha256': {name: sha(getattr(args, 'task_' + name.lower())) for name in 'ABCD'},
            'scope': 'joint D2NN pilot' if args.pilot else 'joint D2NN validation-selected experiment',
            'test_images_read': False,
            'sampling': {'tasks_per_batch': 4, 'samples_per_task': per_task,
                         'steps_per_epoch': cfg['steps_per_epoch']},
        })
        model = OpticalD2NN(cfg).to(args.device)
        optimizer = torch.optim.Adam([
            {'params': [model.phase_1], 'lr': cfg['lr_expert']},
            {'params': [model.phase_2], 'lr': cfg['lr_shared']},
        ])
        parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        history = []; best = -1.; best_epoch = None
        for epoch in range(1, cfg['epochs'] + 1):
            train = train_epoch(model, optimizer, tasks, per_task, cfg['steps_per_epoch'], rng)
            validation = {name: evaluate(model, task['vx'], task['vy'], cfg['eval_batch_size'])[0]
                          for name, task in zip('ABCD', tasks)}
            score = float(np.mean([m['balanced_accuracy'] for m in validation.values()]))
            row = {'epoch': epoch, 'train': train, 'validation': validation,
                   'selection_score': score}
            history.append(row)
            state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                     'config': cfg, 'epoch': epoch, 'validation': validation,
                     'selection_score': score}
            torch.save(state, args.out / 'last_checkpoint.pt')
            if score > best:
                best = score; best_epoch = epoch
                torch.save(state, args.out / 'best_checkpoint.pt')
            save(args.out / 'history.json', history)
            save(args.out / 'status.json', {'state': 'training', 'epoch': epoch,
                                             'best_epoch': best_epoch,
                                             'best_selection_score': best})
            print(json.dumps({'epoch': epoch, 'train': train,
                              'val_bal_acc': {k: v['balanced_accuracy'] for k, v in validation.items()},
                              'selection_score': score}), flush=True)
        selected = torch.load(args.out / 'best_checkpoint.pt', map_location=args.device, weights_only=False)
        model.load_state_dict(selected['model'])
        results = {'selected_epoch': selected['epoch'],
                   'selection_score': selected['selection_score'],
                   'headline_metric': 'balanced_accuracy',
                   'parameter_count': parameter_count,
                   'test_images_read': False,
                   'selection_rule': 'maximum mean validation balanced accuracy across A/B/C/D'}
        for name, task in zip('ABCD', tasks):
            train_metrics, _ = evaluate(model, task['x'], task['y'], cfg['eval_batch_size'])
            val_metrics, probabilities = evaluate(model, task['vx'], task['vy'], cfg['eval_batch_size'])
            results[name] = {'train': train_metrics, 'validation': val_metrics}
            np.savez_compressed(args.out / f'{name}_validation_predictions.npz',
                                ids=task['val_ids'], labels=task['vy'].numpy(),
                                probabilities=probabilities.numpy())
        save(args.out / 'metrics.json', results)
        save(args.out / 'status.json', {'state': 'complete', 'selected_epoch': selected['epoch']})
    except BaseException as error:
        save(args.out / 'status.json', {'state': 'failed', 'error': repr(error)})
        raise


if __name__ == '__main__':
    main()
