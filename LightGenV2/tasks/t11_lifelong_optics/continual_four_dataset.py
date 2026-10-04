"""A->B->C->D optical continual learning with fixed old experts and replay."""
import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from .continual_three_dataset import optimizer_for, train_epoch
from .cross_dataset import balanced_subset, evaluate, load_dataset
from .model import OpticalMoE
from .run import save


def masks(groups=4):
    group_masks = [[g * 4 <= i < (g + 1) * 4 for i in range(groups * 4)] for g in range(groups)]
    prefix_masks = [[i < (g + 1) * 4 for i in range(groups * 4)] for g in range(groups)]
    return group_masks, prefix_masks


GROUP_MASKS, PREFIX_MASKS = masks()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    for name in 'abcd':
        parser.add_argument('--task-' + name, type=Path, required=True)
        parser.add_argument('--task-' + name + '-manifest', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--pilot', action='store_true')
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    if cfg.get('num_experts') != 16:
        raise ValueError('The four-task protocol requires a fixed 16-expert geometry')
    if args.pilot:
        for key in ('epochs_A', 'epochs_warmup_B', 'epochs_B', 'epochs_warmup_C', 'epochs_C', 'epochs_warmup_D', 'epochs_D'):
            cfg[key] = 1
    args.out.mkdir(parents=True, exist_ok=False)
    save(args.out / 'status.json', {'state': 'preparing'})
    try:
        torch.manual_seed(cfg['seed']); np.random.seed(cfg['seed']); torch.set_num_threads(4)
        rng = np.random.default_rng(cfg['seed'])
        raw = []; manifests = []
        for name in 'abcd':
            data, manifest = load_dataset(getattr(args, 'task_' + name), getattr(args, 'task_' + name + '_manifest'))
            raw.append(data); manifests.append(manifest)
        tasks = []
        for i, (name, data) in enumerate(zip('ABCD', raw)):
            train_ids = balanced_subset(data['train_labels'], cfg['train_per_class_' + name], cfg['seed'] + i)
            val_ids = balanced_subset(data['val_labels'], cfg['val_per_class_' + name], cfg['seed'] + 10 + i)
            tasks.append({
                'x': torch.from_numpy(data['train_images'][train_ids]), 'y': torch.from_numpy(data['train_labels'][train_ids]).long(),
                'vx': torch.from_numpy(data['val_images'][val_ids]), 'vy': torch.from_numpy(data['val_labels'][val_ids]).long(),
                'train_ids': data['train_ids'][train_ids], 'val_ids': data['val_ids'][val_ids]
            })
        replays = []
        for i, task in enumerate(tasks):
            ids = balanced_subset(task['y'].numpy(), cfg['replay_per_class'], cfg['seed'] + 20 + i)
            replays.append((task['x'][ids], task['y'][ids], task['train_ids'][ids]))
        save(args.out / 'config.json', cfg)
        save(args.out / 'split.json', {name: {'train_ids': task['train_ids'].tolist(), 'val_ids': task['val_ids'].tolist(), 'replay_ids': replays[i][2].tolist()} for i, (name, task) in enumerate(zip('ABCD', tasks))})
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
        save(args.out / 'metadata.json', {'command': sys.argv, 'commit': commit, 'python': sys.version, 'torch': torch.__version__, 'platform': platform.platform(), 'device': args.device, 'manifests': manifests, 'scope': 'four-task pilot' if args.pilot else 'four-task validation-selected experiment', 'test_images_read': False})
        model = OpticalMoE(cfg).to(args.device)
        geometry = {key: list(value.shape) for key, value in model.state_dict().items()}
        history = []; audit = []; snapshots = {}
        for group, name in enumerate('ABCD'):
            if group:
                model.configure_group(group, warmup=True)
                frozen = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
                optimizer = optimizer_for(model, cfg, False)
                for epoch in range(1, cfg['epochs_warmup_' + name] + 1):
                    value = train_epoch(model, optimizer, tasks[group]['x'], tasks[group]['y'], rng.permutation(len(tasks[group]['y'])), cfg['batch_size'], rng, warmup=True)
                    old = {old_name: evaluate(model, tasks[j]['vx'], tasks[j]['vy'], cfg['batch_size'], PREFIX_MASKS[group - 1])[0] for j, old_name in enumerate('ABCD'[:group])}
                    new = evaluate(model, tasks[group]['vx'], tasks[group]['vy'], cfg['batch_size'], warmup=True)[0]
                    row = {'stage': 'warmup_' + name, 'epoch': epoch, 'loss': value, 'old_prefix': old, 'new_uniform': new}
                    history.append(row); save(args.out / 'history.json', history)
                    print(json.dumps({'stage': 'warmup_' + name, 'epoch': epoch, 'loss': value, 'old_bal_acc': {k: v['balanced_accuracy'] for k, v in old.items()}, 'new_bal_acc': new['balanced_accuracy']}), flush=True)
                for n, value in model.named_parameters():
                    if n in frozen and not torch.equal(value, frozen[n]): raise RuntimeError('Frozen changed ' + n)
                audit.append({'stage': 'warmup_' + name, 'frozen_unchanged': list(frozen), 'geometry_unchanged': geometry == {k: list(v.shape) for k, v in model.state_dict().items()}})
            model.configure_group(group, warmup=False)
            frozen = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
            optimizer = optimizer_for(model, cfg, True)
            stage = args.out / name; stage.mkdir(); best = -1.0
            if group == 0:
                replay_spec = (); current = cfg['batch_size']
            elif group == 1:
                replay_spec = ((replays[0][0], replays[0][1], cfg['replay_A_in_B']),); current = cfg['current_batch_B']
            else:
                count = cfg['replay_each_old_in_' + name]
                replay_spec = tuple((replays[j][0], replays[j][1], count) for j in range(group))
                current = cfg['current_batch_' + name]
            for epoch in range(1, cfg['epochs_' + name] + 1):
                value = train_epoch(model, optimizer, tasks[group]['x'], tasks[group]['y'], rng.permutation(len(tasks[group]['y'])), current, rng, replay_spec)
                seen = {task_name: evaluate(model, tasks[j]['vx'], tasks[j]['vy'], cfg['batch_size'])[0] for j, task_name in enumerate('ABCD'[:group + 1])}
                score = float(np.mean([metric['balanced_accuracy'] for metric in seen.values()]))
                row = {'stage': name, 'epoch': epoch, 'loss': value, 'seen': seen, 'selection_score': score}; history.append(row)
                state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'config': cfg, 'stage': name, 'epoch': epoch, 'validation': row}
                torch.save(state, stage / 'last_checkpoint.pt')
                if score > best:
                    best = score; torch.save(state, stage / 'best_checkpoint.pt')
                save(args.out / 'history.json', history); save(args.out / 'status.json', {'state': 'training', 'stage': name, 'epoch': epoch})
                print(json.dumps({'stage': name, 'epoch': epoch, 'loss': value, 'bal_acc': {k: v['balanced_accuracy'] for k, v in seen.items()}, 'score': score}), flush=True)
            for n, value in model.named_parameters():
                if n in frozen and not torch.equal(value, frozen[n]): raise RuntimeError('Frozen changed ' + n)
            audit.append({'stage': name, 'frozen_unchanged': list(frozen), 'geometry_unchanged': geometry == {k: list(v.shape) for k, v in model.state_dict().items()}})
            state = torch.load(stage / 'best_checkpoint.pt', map_location=args.device, weights_only=False)
            model.load_state_dict(state['model'])
            snapshots[name] = {'epoch': state['epoch'], 'selection_score': state['validation']['selection_score'], 'metrics': state['validation']['seen']}
        results = {'snapshots': snapshots, 'headline_metric': 'balanced_accuracy'}
        for j, name in enumerate('ABCD'):
            variants = [('all', None), ('own_group', GROUP_MASKS[j]), ('learned_prefix', PREFIX_MASKS[j])]
            if j: variants.append(('previous_prefix', PREFIX_MASKS[j - 1]))
            for label, mask in variants:
                results[name + '_' + label] = evaluate(model, tasks[j]['vx'], tasks[j]['vy'], cfg['batch_size'], mask)[0]
        for j, name in enumerate('ABC'):
            results[name + '_BWT_after_D'] = results[name + '_all']['balanced_accuracy'] - snapshots[name]['metrics'][name]['balanced_accuracy']
        results['interpretation'] = 'Masks alter coherent interference; own-group and prefix results are diagnostics.'
        save(args.out / 'metrics.json', results); save(args.out / 'audit.json', audit); save(args.out / 'status.json', {'state': 'complete'})
    except BaseException as error:
        save(args.out / 'status.json', {'state': 'failed', 'error': repr(error)}); raise


if __name__ == '__main__':
    main()
