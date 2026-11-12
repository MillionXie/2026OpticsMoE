"""Adapt a smaller readout to relative-intensity detector features on TRAIN."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

if __package__:
    from . import _paths
    from .checkpoint_utils import file_sha256
    from .data import batch, groups, load_inputs
    from .run import correlation, regression_metrics, save_checkpoint, write
    from .simplify_checkpoint import fold_final_projection, prune_hidden_units
else:
    import _paths
    from checkpoint_utils import file_sha256
    from data import batch, groups, load_inputs
    from run import correlation, regression_metrics, save_checkpoint, write
    from simplify_checkpoint import fold_final_projection, prune_hidden_units

if __package__ and __package__ != 'tools':
    from ..model import load_lightgen
else:
    from model import load_lightgen


@torch.inference_mode()
def collect_features(source, target, payload, split, device):
    indices, valid = groups(payload, split)
    features, predictions, targets, ids = [], [], [], []
    captured = {}
    handle = target.readout.register_forward_pre_hook(
        lambda module, inputs: captured.update(value=inputs[0]))
    try:
        for start in range(0, len(indices), 4):
            ii, vv = indices[start:start+4], valid[start:start+4]
            item = batch(payload, ii, vv, device)
            teacher = source(item['vision']) if split == 'train' else None
            target(item['vision'])
            features.append(captured['value'][item['valid']].cpu())
            if teacher is not None:
                predictions.append(teacher['prediction'][item['valid']].cpu())
            targets.append(item['target'][item['valid']].cpu())
            ids.extend(payload['ids'][index] for index in ii[vv].tolist())
    finally:
        handle.remove()
    result = {'features': torch.cat(features), 'target': torch.cat(targets), 'ids': ids}
    if predictions:
        result['teacher'] = torch.cat(predictions)
    return result


def write_predictions(path, data, prediction, epoch):
    import csv
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['sample_id', 'target', 'prediction', 'epoch'])
        writer.writerows(zip(data['ids'], data['target'].cpu().tolist(), prediction, [epoch]*len(prediction)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--paths', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--cache', type=Path)
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--learning-rate', type=float, default=0.0003)
    parser.add_argument('--teacher-weight', type=float, default=10)
    parser.add_argument('--minimum-srcc', type=float, default=-1)
    parser.add_argument('--maximum-srcc', type=float, default=1)
    parser.add_argument('--seed', type=int, default=713)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if not -1 <= args.minimum_srcc <= args.maximum_srcc <= 1:
        raise ValueError('Invalid development-SRCC selection interval')
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.mkdir(parents=True)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    source, source_metadata = load_lightgen(args.weights)
    fused = fold_final_projection(source)
    source_hash = file_sha256(args.weights)
    converted_metadata = {**source_metadata, 'role': 'FUSED_INITIALIZATION',
                          'parent_checkpoint_sha256': source_hash}
    save_checkpoint(args.out/'folded_initialization.pt', fused.checkpoint(**converted_metadata))
    relative = fold_final_projection(source, 'relative_intensity')
    source = source.to(args.device).eval()
    relative = relative.to(args.device).eval()
    relative.requires_grad_(False)
    cache_path = args.cache or args.out/'features.pt'
    if cache_path.exists():
        cache = torch.load(cache_path, weights_only=True, map_location='cpu')
        if cache['source_checkpoint_sha256'] != source_hash:
            raise ValueError('Feature-cache checkpoint differs')
    else:
        payload = load_inputs(json.loads(args.paths.read_text('utf-8')))
        cache = {'source_checkpoint_sha256': source_hash,
                 'dataset': payload['identity'],
                 'config': asdict(relative.cfg),
                 'train': collect_features(source, relative, payload, 'train', args.device),
                 'test': collect_features(source, relative, payload, 'test', args.device)}
        save_checkpoint(cache_path, cache)
    if cache['config'] != asdict(relative.cfg):
        raise ValueError('Feature-cache model configuration differs')
    data = {split: {name: value.to(args.device) if isinstance(value, torch.Tensor) else value
                    for name, value in cache[split].items()} for split in ('train', 'test')}
    if args.hidden < relative.cfg.readout_hidden:
        model, selected = prune_hidden_units(relative, data['train']['features'], args.hidden)
    elif args.hidden == relative.cfg.readout_hidden:
        model, selected = relative.cpu(), list(range(args.hidden))
    else:
        raise ValueError('Requested readout is wider than its source')
    del source, relative
    model = model.to(args.device).eval()
    model.requires_grad_(False)
    model.readout.requires_grad_(True)
    fixed = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()
             if not name.startswith('readout.')}
    optimizer = torch.optim.AdamW(model.readout.parameters(), lr=args.learning_rate, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=args.learning_rate*.1)
    metadata = {'role': 'ELECTRONIC_SIMPLIFICATION_DEVELOPMENT',
                'parent_training_metadata': source_metadata,
                'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=_paths.ROOT, text=True).strip(),
                'source_SHA256': {str(path.relative_to(_paths.ROOT)): file_sha256(path)
                                  for path in [_paths.ROOT/'model.py', Path(__file__), Path(__file__).with_name('simplify_checkpoint.py')]},
                'dataset': cache['dataset'], 'parent_checkpoint_sha256': source_hash,
                'model_config': asdict(model.cfg), 'model_parameters': sum(p.numel() for p in model.parameters()),
                'frozen_except': 'readout', 'hidden_selection': selected,
                'test_used_for_selection': True, 'test_used_for_gradient': False,
                'training_groups': 'fixed original TRAIN order with cyclic padding excluded',
                'command': sys.argv, 'python': platform.python_version(), 'torch': str(torch.__version__),
                'GPU': torch.cuda.get_device_name() if args.device.startswith('cuda') else 'CPU',
                'seed': args.seed, 'epochs': args.epochs, 'learning_rate': args.learning_rate,
                'teacher_weight': args.teacher_weight, 'GPU_resource_mode': 'nonexclusive_local_desktop'}
    metadata['selection_interval'] = [args.minimum_srcc, args.maximum_srcc]
    metadata['config'] = {key: metadata[key] for key in
                          ('seed', 'epochs', 'learning_rate', 'teacher_weight', 'frozen_except', 'training_groups')}
    metadata['config']['model'] = asdict(model.cfg)
    metadata['optimization_parameters'] = sum(p.numel() for p in model.readout.parameters())
    write(args.out/'metadata.json', metadata)
    history, best = [], -float('inf')
    started = time.time()
    train = data['train']
    for epoch in range(args.epochs+1):
        losses = []
        if epoch:
            model.readout.train()
            order = torch.randperm(len(train['ids']), device=args.device)
            for index in order.split(128):
                logits = model.readout(train['features'][index])
                prediction = logits.softmax(-1) @ model.level_scores
                normalized = (prediction-model.target_mean)/model.target_std
                target = (train['target'][index]-model.target_mean)/model.target_std
                teacher = (train['teacher'][index]-model.target_mean)/model.target_std
                loss = (F.smooth_l1_loss(normalized, target) + 1.3*(1-correlation(normalized, target))
                        + args.teacher_weight*F.smooth_l1_loss(normalized, teacher))
                labels = torch.bucketize(train['target'][index], model.level_boundaries)
                loss = loss + .15*F.cross_entropy(logits, labels)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError('Nonfinite readout loss')
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.readout.parameters(), 5)
                optimizer.step()
                losses.append(float(loss.detach()))
            scheduler.step()
        if epoch % 5 == 0 or epoch == 1 or epoch == args.epochs:
            model.eval()
            with torch.inference_mode():
                test = data['test']
                prediction = (model.readout(test['features']).softmax(-1) @ model.level_scores).cpu().tolist()
            metrics = regression_metrics(prediction, test['target'].cpu().tolist())
            row = {'epoch': epoch, 'loss': float(np.mean(losses)) if losses else None,
                   'metrics': metrics, 'elapsed_seconds': time.time()-started}
            history.append(row)
            eligible = args.minimum_srcc <= metrics['srcc'] <= args.maximum_srcc
            row['eligible_for_selection'] = eligible
            if eligible and metrics['srcc'] > best:
                best = metrics['srcc']
                checkpoint = model.checkpoint(**{**metadata, 'epoch': epoch, 'metrics': metrics})
                save_checkpoint(args.out/'best_checkpoint.pt', checkpoint)
                write_predictions(args.out/'test_predictions.csv', cache['test'], prediction, epoch)
                write(args.out/'metrics.json', metrics)
            write(args.out/'history.json', history)
            write(args.out/'status.json', {'state': 'running', 'epoch': epoch,
                                          'best_srcc': best if np.isfinite(best) else None})
            print(json.dumps(row), flush=True)
    for name, expected in fixed.items():
        if not torch.equal(model.state_dict()[name].cpu(), expected):
            raise ValueError('A frozen tensor changed: '+name)
    if not np.isfinite(best):
        raise ValueError('No evaluated epoch meets the requested SRCC interval')
    save_checkpoint(args.out/'last_checkpoint.pt', model.checkpoint(**{**metadata, 'epoch': args.epochs}))
    write(args.out/'status.json', {'state': 'completed', 'best_srcc': best,
                                 'epochs': args.epochs, 'frozen_tensors_unchanged': True})


if __name__ == '__main__':
    main()
