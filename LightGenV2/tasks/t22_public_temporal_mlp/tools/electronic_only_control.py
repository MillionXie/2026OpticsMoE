"""Train an electronic-capacity control after removing routing, experts and globals."""
from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

if __package__:
    from . import _paths
    from .analyze_global_ablation import ablation_metrics
    from .checkpoint_utils import file_sha256, tensor_fingerprint
    from .data import load_inputs
    from .run import correlation, save_checkpoint, write
else:
    import _paths
    from analyze_global_ablation import ablation_metrics
    from checkpoint_utils import file_sha256, tensor_fingerprint
    from data import load_inputs
    from run import correlation, save_checkpoint, write

if __package__ and __package__ != 'tools':
    from ..model import load_lightgen
else:
    from model import load_lightgen


class ElectronicControl(nn.Module):
    """Retain the parent's adapters and bridge, with a selected electronic head."""

    def __init__(self, parent, hidden_units):
        super().__init__()
        self.vision_adapter = deepcopy(parent.vision_adapter)
        self.language_adapter = deepcopy(parent.language_adapter)
        self.frame_merger = deepcopy(parent.frame_merger)
        self.frame_position = nn.Parameter(parent.frame_position.detach().clone())
        self.sequence_position = nn.Parameter(parent.sequence_position.detach().clone())
        for name in ('prompt_embedding', 'target_mean', 'target_std',
                     'level_scores', 'level_boundaries'):
            self.register_buffer(name, getattr(parent, name).detach().clone())
        self.readout = nn.Sequential(
            nn.Flatten(-2), nn.Linear(42*192, len(hidden_units)),
            nn.ReLU(), nn.Linear(len(hidden_units), 5),
        )
        selected = torch.tensor(hidden_units, dtype=torch.long, device=parent.readout[1].weight.device)
        with torch.no_grad():
            self.readout[1].weight.copy_(parent.readout[1].weight[selected])
            self.readout[1].bias.copy_(parent.readout[1].bias[selected])
            self.readout[3].weight.copy_(parent.readout[3].weight[:, selected])
            self.readout[3].bias.copy_(parent.readout[3].bias)

    def encode(self, vision_tokens):
        if tuple(vision_tokens.shape[1:]) != (4, 49, 1024):
            raise ValueError('Expected one four-frame feature tensor per video')
        vision = self.vision_adapter(vision_tokens.float())
        frame_summary = torch.cat((vision.mean(-2), vision.amax(-2)), dim=-1)
        frames = self.frame_merger(frame_summary) + self.frame_position[:, 0]
        prompt = self.language_adapter(self.prompt_embedding.float())
        prompt = prompt[None].expand(len(frames), -1, -1)
        sequence = torch.cat((frames, prompt), dim=-2)
        return sequence + self.sequence_position[:, :sequence.shape[-2]]

    def forward(self, vision_tokens):
        logits = self.readout(self.encode(vision_tokens))
        prediction = logits.softmax(-1) @ self.level_scores
        return logits, prediction


def checkpoint(model, metadata, epoch, metrics=None):
    record = {**metadata, 'epoch': epoch}
    if metrics is not None:
        record['metrics'] = metrics
    return {'schema': 1, 'architecture': metadata['architecture'],
            'model': model.state_dict(), 'metadata': record}


def teacher_for_train(cache, payload, train_indices):
    records = cache['train']
    ids = records['ids']
    expected = [payload['ids'][index] for index in train_indices]
    if len(ids) != 2250 or len(set(ids)) != 2250 or set(ids) != set(expected):
        raise ValueError('Teacher identities must match the 2250 TRAIN videos')
    lookup = {sample_id: index for index, sample_id in enumerate(ids)}
    values = records['teacher']
    targets = records['target']
    if values.shape != (2250,) or not bool(torch.isfinite(values).all()):
        raise ValueError('Invalid TRAIN teacher predictions')
    ordered = torch.tensor([lookup[sample_id] for sample_id in expected])
    if not torch.equal(targets[ordered], payload['target'][train_indices]):
        raise ValueError('Teacher TRAIN labels differ from the manifest')
    return values[ordered].float()


@torch.no_grad()
def initialize_head(model, parent, vision, candidates, hidden_width, batch_size):
    parent_head = parent.readout
    hidden_sum = torch.zeros(256, dtype=torch.float64, device=vision.device)
    hidden_square_sum = torch.zeros_like(hidden_sum)
    for start in range(0, len(vision), batch_size):
        features = model.encode(vision[start:start+batch_size])
        hidden = parent_head[2](parent_head[1](features.flatten(-2)))
        hidden_sum += hidden.double().sum(0)
        hidden_square_sum += hidden.double().square().sum(0)
    hidden_mean = (hidden_sum / len(vision)).float()
    hidden_units = candidates
    if hidden_width < len(candidates):
        rms = (hidden_square_sum / len(vision)).sqrt().float()
        importance = rms * parent_head[3].weight.square().sum(0).sqrt()
        candidate_indices = torch.tensor(candidates, device=vision.device)
        selected = candidate_indices[importance[candidate_indices].topk(hidden_width).indices]
        hidden_units = selected.sort().values.cpu().tolist()
        model = ElectronicControl(parent, hidden_units).to(vision.device).eval()
    removed = torch.ones(256, dtype=torch.bool, device=vision.device)
    removed[torch.tensor(hidden_units, device=vision.device)] = False
    correction = parent_head[3].weight[:, removed] @ hidden_mean[removed]
    model.readout[3].bias.add_(correction)
    return model, hidden_units, correction.cpu().tolist()


@torch.inference_mode()
def evaluate(model, vision, target, batch_size):
    model.eval()
    predictions = []
    for start in range(0, len(vision), batch_size):
        _, prediction = model(vision[start:start+batch_size])
        predictions.extend(prediction.cpu().tolist())
    return ablation_metrics(predictions, target.cpu().tolist()), predictions


def write_predictions(path, ids, targets, predictions, epoch):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['sample_id', 'target', 'prediction', 'epoch'])
        writer.writerows(zip(ids, targets.cpu().tolist(), predictions, [epoch]*len(ids)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, default=_paths.ROOT/'assets/initialization/simplification_parent.pt')
    parser.add_argument('--selection', type=Path, default=_paths.ROOT/'runs/simulation/current/LightGen.pt')
    parser.add_argument('--unit-candidates', choices=('all_parent', 'selection'), default='all_parent',
                        help='Choose all 256 parent hidden units or the recorded selected-unit pool')
    parser.add_argument('--teacher-cache', type=Path, required=True)
    parser.add_argument('--paths', type=Path, default=_paths.ROOT/'paths.local.json')
    parser.add_argument('--out', type=Path, default=_paths.ROOT/'runs/simulation/electronic_only_control_20261113')
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--learning-rate', type=float, default=0.0001)
    parser.add_argument('--seed', type=int, default=713)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 2 or args.learning_rate <= 0:
        raise ValueError('Invalid training budget')
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.mkdir(parents=True)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    started = time.time()
    parent_hash = file_sha256(args.parent)
    if parent_hash != '8c34a360227e258f86acb738fbb0da02af59ab38430a52986df0a09e9beec359':
        raise ValueError('Expected the recorded 256-unit parent checkpoint')
    parent, _ = load_lightgen(args.parent)
    selection_hash = None
    if args.unit_candidates == 'all_parent':
        candidates = list(range(256))
    else:
        selection = torch.load(args.selection, map_location='cpu', weights_only=True)
        selection_metadata = selection['metadata']
        candidates = selection_metadata.get('hidden_selection')
        if candidates is None:
            candidates = selection_metadata['initialization']['hidden_units']
        selection_hash = file_sha256(args.selection)
    if (len(set(candidates)) != len(candidates)
            or not all(0 <= index < 256 for index in candidates)
            or not 0 < args.hidden <= len(candidates)):
        raise ValueError('Hidden width must fit the distinct recorded parent-unit candidates')
    payload = load_inputs(json.loads(args.paths.read_text('utf-8')))
    train_indices = [i for i, split in enumerate(payload['splits']) if split == 'train']
    test_indices = [i for i, split in enumerate(payload['splits']) if split == 'test']
    cache = torch.load(args.teacher_cache, map_location='cpu', weights_only=True)
    if cache['source_checkpoint_sha256'] != parent_hash:
        raise ValueError('Teacher cache belongs to a different parent')
    if cache['dataset']['manifest']['sha256'] != payload['identity']['manifest']['sha256']:
        raise ValueError('Teacher manifest differs')
    teacher = teacher_for_train(cache, payload, train_indices).to(args.device)
    train_vision = payload['vision'][train_indices].to(args.device)
    test_vision = payload['vision'][test_indices].to(args.device)
    train_target = payload['target'][train_indices].to(args.device)
    test_target = payload['target'][test_indices].to(args.device)
    test_ids = [payload['ids'][index] for index in test_indices]
    model = ElectronicControl(parent, candidates).to(args.device).eval()
    parent.readout.to(args.device)
    model, hidden_units, correction = initialize_head(
        model, parent, train_vision, candidates, args.hidden, args.batch_size)
    assert sum(p.numel() for p in model.parameters()) == 684101 + 8070*args.hidden
    dataset_identity = payload['identity']
    del parent, cache, payload
    model.requires_grad_(True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.epochs, eta_min=args.learning_rate*.1)
    metadata = {
        'role': 'ELECTRONIC_CAPACITY_CONTROL_ORIGINAL_TEST_DEVELOPMENT',
        'architecture': f't22_electronic_capacity_control_mean_max_fc{args.hidden}_five_levels',
        'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=_paths.ROOT, text=True).strip(),
        'source_SHA256': {str(path.relative_to(_paths.ROOT)): file_sha256(path) for path in
                          (Path(__file__), _paths.ROOT/'model.py', Path(__file__).with_name('data.py'),
                           Path(__file__).with_name('run.py'))},
        'parent_checkpoint_sha256': parent_hash,
        'selection_checkpoint_sha256': selection_hash,
        'teacher_cache_sha256': file_sha256(args.teacher_cache),
        'teacher_usage': 'Only cache[train][teacher], paired to the exact 2250 TRAIN IDs',
        'dataset': dataset_identity,
        'initialization': {
            'electronic_modules': 'parent vision_adapter, language_adapter, frame_merger and position tensors',
            'head': 'selected rows/columns of the parent 8064-to-256-to-5 head',
            'hidden_units': hidden_units,
            'hidden_candidates': candidates,
            'unit_candidate_mode': args.unit_candidates,
            'hidden_selection_rule': 'retain all candidates when widths match; narrower heads select TRAIN RMS activation times parent output-column L2 norm within the recorded candidates',
            'TRAIN_missing_unit_logit_mean_correction': correction,
        },
        'input_contract': 'cached frozen-Qwen features, 4x49x1024 per video',
        'model_parameters': sum(p.numel() for p in model.parameters()),
        'optimization_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'trained_modules': [name for name, _ in model.named_parameters()],
        'selection': 'highest evaluated original TEST development SRCC without an upper bound',
        'test_used_for_selection': True, 'test_used_for_gradient': False,
        'reference_difference': 'all retained electronic modules trainable; head input8064 versus selected reference4032; no router/expert/global modules',
        'configuration': {'epochs': args.epochs, 'batch_size': args.batch_size,
                          'hidden_width': args.hidden,
                          'unit_candidates': args.unit_candidates,
                          'learning_rate': args.learning_rate, 'weight_decay': 0.0001,
                          'correlation_weight': 1.3, 'teacher_weight': 10, 'quality_ce_weight': 0.15,
                          'seed': args.seed},
        'initial_model_state_sha256': tensor_fingerprint(model.state_dict()),
        'command': sys.argv, 'python': platform.python_version(), 'torch': str(torch.__version__),
        'platform': platform.platform(),
        'GPU': torch.cuda.get_device_name() if args.device.startswith('cuda') else 'CPU',
        'GPU_resource_mode': 'authorized nonexclusive local desktop',
    }
    write(args.out/'metadata.json', metadata)
    history = []
    best = -float('inf')
    best_epoch = None
    for epoch in range(args.epochs+1):
        losses = []
        epoch_started = time.time()
        if epoch:
            model.train()
            order = torch.randperm(len(train_target), device=args.device)
            for index in order.split(args.batch_size):
                logits, prediction = model(train_vision[index])
                normalized = (prediction-model.target_mean)/model.target_std
                target = (train_target[index]-model.target_mean)/model.target_std
                soft_target = (teacher[index]-model.target_mean)/model.target_std
                loss = (F.smooth_l1_loss(normalized, target) + 1.3*(1-correlation(normalized, target))
                        + 10*F.smooth_l1_loss(normalized, soft_target))
                loss += 0.15*F.cross_entropy(logits, torch.bucketize(train_target[index], model.level_boundaries))
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError('Nonfinite electronic-control loss')
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                if not bool(torch.isfinite(norm)):
                    raise FloatingPointError('Nonfinite electronic-control gradient')
                optimizer.step()
                losses.append(float(loss.detach()))
            scheduler.step()
            if args.device.startswith('cuda'):
                torch.cuda.synchronize()
        epoch_seconds = time.time()-epoch_started
        row = {'epoch': epoch, 'loss': float(np.mean(losses)) if losses else None,
               'epoch_seconds': epoch_seconds, 'elapsed_seconds': time.time()-started,
               'learning_rate': optimizer.param_groups[0]['lr']}
        if epoch % 5 == 0 or epoch == 1 or epoch == args.epochs:
            metrics, predictions = evaluate(model, test_vision, test_target, args.batch_size)
            row['test_development'] = metrics
            score = metrics['srcc']
            if score is not None and score > best:
                best, best_epoch = score, epoch
                save_checkpoint(args.out/'best_checkpoint.pt', checkpoint(model, metadata, epoch, metrics))
                write_predictions(args.out/'test_predictions.csv', test_ids, test_target, predictions, epoch)
                write(args.out/'metrics.json', metrics)
            print(json.dumps(row), flush=True)
        elif epoch == 2:
            print(json.dumps({'timing_estimate': 'full 400-epoch budget',
                              'epoch_seconds': epoch_seconds,
                              'estimated_remaining_seconds': (args.epochs-epoch)*epoch_seconds}), flush=True)
        history.append(row)
        write(args.out/'history.json', history)
        write(args.out/'status.json', {'state': 'running', 'epoch': epoch, 'epochs': args.epochs,
                                      'best_srcc': best if np.isfinite(best) else None,
                                      'best_epoch': best_epoch, 'elapsed_seconds': time.time()-started})
        if epoch and (epoch % 20 == 0 or epoch == args.epochs):
            last = checkpoint(model, metadata, epoch)
            last['optimizer'] = optimizer.state_dict()
            last['scheduler'] = scheduler.state_dict()
            save_checkpoint(args.out/'last_checkpoint.pt', last)
    if best_epoch is None:
        raise ValueError('No evaluated model had a defined correlation')
    write(args.out/'status.json', {'state': 'completed', 'epochs': args.epochs,
                                  'best_srcc': best, 'best_epoch': best_epoch,
                                  'elapsed_seconds': time.time()-started})
    print(json.dumps({'state': 'completed', 'best_srcc': best, 'best_epoch': best_epoch}), flush=True)


if __name__ == '__main__':
    main()
