"""Randomly initialized two-FC control on fixed Qwen-derived embeddings."""
from __future__ import annotations

import argparse
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


INPUT_WIDTHS = {'flatten_concat': 278528, 'pool_4032': 4032}
PARAMETER_NAMES = {'fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias'}


def fixed_embedding_vector(vision_tokens, prompt_embedding, mode):
    """Apply explicit fixed shape processing, with detached embedding inputs."""
    if tuple(vision_tokens.shape[1:]) != (4, 49, 1024):
        raise ValueError('Expected [videos,4,49,1024] fixed visual embeddings')
    if tuple(prompt_embedding.shape) != (38, 2048):
        raise ValueError('Expected the fixed [38,2048] prompt embedding')
    vision = vision_tokens.detach().float()
    prompt = prompt_embedding.detach().float()
    if mode == 'flatten_concat':
        visual_vector = vision.flatten(1)
        text_vector = prompt.flatten()
    elif mode == 'pool_4032':
        # First average each frame's 49 spatial tokens, then average its 1024
        # channels into 96 bins. Each text token's 2048 channels are averaged
        # into 96 bins. These operations discard spatial/channel information.
        visual_vector = F.adaptive_avg_pool1d(
            vision.mean(-2).reshape(-1, 1, 1024), 96).reshape(len(vision), 4*96)
        text_vector = F.adaptive_avg_pool1d(prompt[:, None], 96).flatten()
    else:
        raise ValueError('Unknown embedding input mode')
    return torch.cat((visual_vector, text_vector[None].expand(len(vision), -1)), dim=-1)


class EmbeddingReadout(nn.Module):
    """Train exactly FC1W/b and FC2W/b, using TRAIN-derived MOS statistics."""

    def __init__(self, prompt_embedding, mode, train_targets, hidden_width=32):
        super().__init__()
        if mode not in INPUT_WIDTHS or hidden_width <= 0:
            raise ValueError('Invalid readout configuration')
        targets = torch.as_tensor(train_targets, dtype=torch.float32).detach().cpu()
        if targets.ndim != 1 or len(targets) < 2 or not bool(torch.isfinite(targets).all()):
            raise ValueError('Finite one-dimensional TRAIN labels are required')
        lower, upper = targets.min(), targets.max()
        if not bool(upper > lower):
            raise ValueError('TRAIN labels must have a positive range')
        if tuple(prompt_embedding.shape) != (38, 2048):
            raise ValueError('Prompt dimensions differ from the fixed-input contract')
        self.input_mode = mode
        self.hidden_width = hidden_width
        self.register_buffer('prompt_embedding', prompt_embedding.detach().clone())
        self.register_buffer('target_mean', targets.mean())
        self.register_buffer('target_std', targets.std(unbiased=False))
        self.register_buffer('level_scores', torch.linspace(float(lower), float(upper), 5))
        self.register_buffer('level_boundaries', torch.linspace(float(lower), float(upper), 6)[1:-1])
        self.fc1 = nn.Linear(INPUT_WIDTHS[mode], hidden_width)
        self.fc2 = nn.Linear(hidden_width, 5)
        if set(dict(self.named_parameters())) != PARAMETER_NAMES:
            raise ValueError('Unexpected learnable parameter outside the two FC layers')

    def forward(self, vision_tokens):
        vector = fixed_embedding_vector(vision_tokens, self.prompt_embedding, self.input_mode)
        logits = self.fc2(F.relu(self.fc1(vector)))
        prediction = logits.softmax(-1) @ self.level_scores
        return logits, prediction


def checkpoint(model, metadata, epoch, metrics=None):
    record = {**metadata, 'epoch': epoch}
    if metrics is not None:
        record['metrics'] = metrics
    return {
        'schema': 1,
        'architecture': f't22_fixed_embeddings_{model.input_mode}_random_fc{model.hidden_width}_five_levels',
        'config': {'input_mode': model.input_mode, 'input_width': INPUT_WIDTHS[model.input_mode],
                   'hidden_width': model.hidden_width},
        'model': model.state_dict(), 'metadata': record,
    }


@torch.inference_mode()
def evaluate(model, vision, target, batch_size):
    model.eval()
    prediction = []
    for start in range(0, len(vision), batch_size):
        _, result = model(vision[start:start+batch_size])
        prediction.extend(result.cpu().tolist())
    return ablation_metrics(prediction, target.cpu().tolist()), prediction


def write_predictions(path, ids, targets, predictions, epoch):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['sample_id', 'target', 'prediction', 'epoch'])
        writer.writerows(zip(ids, targets.cpu().tolist(), predictions, [epoch]*len(ids)))


def train(args):
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    started = time.time()
    paths = json.loads(args.paths.read_text('utf-8'))
    payload = load_inputs(paths)
    train_indices = [i for i, split in enumerate(payload['splits']) if split == 'train']
    test_indices = [i for i, split in enumerate(payload['splits']) if split == 'test']
    train_target = payload['target'][train_indices].to(args.device)
    test_target = payload['target'][test_indices].to(args.device)
    model = EmbeddingReadout(payload['prompt'], args.input_mode, train_target, args.hidden).to(args.device)
    train_vision = payload['vision'][train_indices].to(args.device)
    test_vision = payload['vision'][test_indices].to(args.device)
    test_ids = [payload['ids'][i] for i in test_indices]
    dataset_identity = payload['identity']
    del payload
    parameter_names = list(dict(model.named_parameters()))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.epochs, eta_min=args.learning_rate*.1)
    frozen_buffers = {name: value.detach().cpu().clone() for name, value in model.named_buffers()}
    metadata = {
        'role': 'RANDOM_TWO_FC_FIXED_EMBEDDINGS_CONTROL_ORIGINAL_TEST_DEVELOPMENT',
        'initialization': 'torch.nn.Linear default random initialization, with the recorded seed',
        'task_checkpoint_initialization': None,
        'teacher_distillation': False,
        'input_mode': args.input_mode, 'input_width': INPUT_WIDTHS[args.input_mode],
        'input_processing': (
            'flatten all 4x49x1024 visual and 38x2048 text values, then concatenate'
            if args.input_mode == 'flatten_concat' else
            'information-losing fixed averages: visual frame spatial mean49x1024->1024 then adaptive channel average1024->96; text token2048->96; concatenate42x96'
        ),
        'dataset': dataset_identity,
        'fixed_Qwen_component_identity': {
            'vision_cache': dataset_identity['vision_cache']['sha256'],
            'language_cache': dataset_identity['language_cache']['sha256'],
        },
        'MOS_statistics_source': '2250 TRAIN temporal labels, computed afresh for this run',
        'target_mean': float(model.target_mean), 'target_std': float(model.target_std),
        'level_scores': model.level_scores.cpu().tolist(),
        'level_boundaries': model.level_boundaries.cpu().tolist(),
        'model_parameters': sum(p.numel() for p in model.parameters()),
        'gradient_parameter_names': parameter_names,
        'initial_model_state_sha256': tensor_fingerprint(model.state_dict()),
        'initial_parameter_sha256': tensor_fingerprint(dict(model.named_parameters())),
        'test_used_for_gradient': False, 'test_used_for_selection': True,
        'selection': 'highest evaluated original TEST development SRCC without an upper bound',
        'configuration': {'epochs': args.epochs, 'seed': args.seed, 'hidden_width': args.hidden,
                          'batch_size': args.batch_size, 'learning_rate': args.learning_rate,
                          'weight_decay': 0.0001, 'correlation_weight': 1.3,
                          'quality_ce_weight': 0.15, 'teacher_weight': 0},
        'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=_paths.ROOT, text=True).strip(),
        'source_SHA256': {str(path.relative_to(_paths.ROOT)): file_sha256(path) for path in
                          (Path(__file__), Path(__file__).with_name('data.py'),
                           Path(__file__).with_name('run.py'), Path(__file__).with_name('analyze_global_ablation.py'))},
        'command': sys.argv, 'python': platform.python_version(), 'torch': str(torch.__version__),
        'platform': platform.platform(),
        'GPU': torch.cuda.get_device_name() if args.device.startswith('cuda') else 'CPU',
        'GPU_resource_mode': 'authorized nonexclusive local desktop',
    }
    write(args.out/'metadata.json', metadata)
    history, best, best_epoch = [], -float('inf'), None
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
                loss = (F.smooth_l1_loss(normalized, target)
                        + 1.3*(1-correlation(normalized, target))
                        + 0.15*F.cross_entropy(logits, torch.bucketize(train_target[index], model.level_boundaries)))
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError('Nonfinite embedding-readout loss')
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if set(name for name, parameter in model.named_parameters() if parameter.grad is not None) != PARAMETER_NAMES:
                    raise ValueError('Gradient parameter contract differs from the two FC layers')
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                if not bool(torch.isfinite(norm)):
                    raise FloatingPointError('Nonfinite embedding-readout gradient')
                optimizer.step()
                losses.append(float(loss.detach()))
            scheduler.step()
            if args.device.startswith('cuda'):
                torch.cuda.synchronize()
        row = {'epoch': epoch, 'loss': float(np.mean(losses)) if losses else None,
               'epoch_seconds': time.time()-epoch_started, 'elapsed_seconds': time.time()-started,
               'learning_rate': optimizer.param_groups[0]['lr']}
        if epoch % 5 == 0 or epoch == 1 or epoch == args.epochs:
            metrics, predictions = evaluate(model, test_vision, test_target, args.batch_size)
            row['test_development'] = metrics
            if metrics['srcc'] is not None and metrics['srcc'] > best:
                best, best_epoch = metrics['srcc'], epoch
                save_checkpoint(args.out/'best_checkpoint.pt', checkpoint(model, metadata, epoch, metrics))
                write(args.out/'metrics.json', metrics)
                write_predictions(args.out/'test_predictions.csv', test_ids, test_target, predictions, epoch)
            print(json.dumps(row), flush=True)
        history.append(row)
        write(args.out/'history.json', history)
        write(args.out/'status.json', {'state': 'running', 'epoch': epoch, 'epochs': args.epochs,
                                      'best_srcc': best if np.isfinite(best) else None, 'best_epoch': best_epoch})
        if epoch and (epoch % 20 == 0 or epoch == args.epochs):
            last = checkpoint(model, metadata, epoch)
            last['optimizer'], last['scheduler'] = optimizer.state_dict(), scheduler.state_dict()
            save_checkpoint(args.out/'last_checkpoint.pt', last)
    if best_epoch is None:
        raise ValueError('No evaluated model had defined correlation')
    if any(not torch.equal(value.cpu(), frozen_buffers[name]) for name, value in model.named_buffers()):
        raise ValueError('A fixed embedding or TRAIN-derived statistic changed')
    write(args.out/'status.json', {'state': 'completed', 'epochs': args.epochs,
                                  'best_srcc': best, 'best_epoch': best_epoch,
                                  'elapsed_seconds': time.time()-started, 'fixed_buffers_unchanged': True})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-mode', choices=tuple(INPUT_WIDTHS), default='flatten_concat')
    parser.add_argument('--paths', type=Path, default=_paths.ROOT/'paths.local.json')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--hidden', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--learning-rate', type=float, default=0.0001)
    parser.add_argument('--seed', type=int, default=713)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 2 or args.hidden < 1 or args.learning_rate <= 0:
        raise ValueError('Invalid training configuration')
    args.out = args.out or _paths.ROOT/'runs/simulation/embedding_readout_control_20261113'/args.input_mode
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.mkdir(parents=True)
    try:
        train(args)
    except BaseException as error:
        write(args.out/'status.json', {'state': 'failed', 'type': type(error).__name__,
                                      'error': str(error), 'time': time.time()})
        raise


if __name__ == '__main__':
    main()
