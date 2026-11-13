"""Fine-tune retained phase masks and the 32-unit readout with spatial losses."""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
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
    from .data import load_inputs, groups, batch, sha256
    from .phase_regularization import model_phase_penalties, smooth_phase_gradients, smooth_phase_parameters, phase_statistics
    from .run import evaluate, write, save_checkpoint, correlation, router_regularization
    from .spatial_locality import capture_modulated_fields, spatial_locality_loss, audit_spatial_locality
    from .training_policy import restrict_training_to
    from ..model import load_lightgen
else:
    import _paths
    from data import load_inputs, groups, batch, sha256
    from phase_regularization import model_phase_penalties, smooth_phase_gradients, smooth_phase_parameters, phase_statistics
    from run import evaluate, write, save_checkpoint, correlation, router_regularization
    from spatial_locality import capture_modulated_fields, spatial_locality_loss, audit_spatial_locality
    from training_policy import restrict_training_to
    from model import load_lightgen


def candidate_score(metrics, spatial, config):
    if not config['minimum_srcc'] <= metrics['srcc'] <= config['maximum_srcc']:
        return None
    largest_deviation = max(metrics[name + '_routing']['maximum_slot_share_deviation']
                            for name in ('frame', 'video'))
    if largest_deviation > config['maximum_slot_share_deviation']:
        return None
    global_deviation = max(abs(value - .25)
                           for name in ('frame', 'video')
                           for value in metrics[name + '_routing']['selection_share'])
    if global_deviation > config['maximum_global_share_deviation']:
        return None
    return -sum(spatial[name]['foreign_fraction'] + spatial[name]['dark_source_fraction']
                + config.get('spatial_guard_weight', 0.) * spatial[name]['guard_fraction']
                for name in ('vision', 'language')) / 2


def assess(model, payload, device, config):
    # Use a fresh canonical parameter state for evaluation, matching checkpoint
    # loading rather than the optimizer's training-only freezing flags.
    canonical = copy.deepcopy(model).requires_grad_(True).eval()
    metrics = evaluate(canonical, payload, device, config['eval_batch_fields'])
    spatial, _ = audit_spatial_locality(canonical, payload, device, split='train',
                                       field_count=config['audit_train_fields'])
    del canonical
    return metrics, spatial


def train(args):
    config = json.loads(args.config.read_text('utf-8'))
    args.out.mkdir(parents=True, exist_ok=False)
    write(args.out / 'config.json', config)
    torch.set_num_threads(4)
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    paths = json.loads(args.paths.read_text('utf-8'))
    payload = load_inputs(paths)
    model, parent_metadata = load_lightgen(args.weights, args.device)
    if model.cfg.readout_hidden != 32 or model.cfg.expert_mode != 'linear' or not model.cfg.fused_readout:
        raise ValueError('Expected the approved linear-expert model with its 32-unit readout')
    if parent_metadata['dataset']['manifest']['sha256'] != payload['identity']['manifest']['sha256']:
        raise ValueError('Parent checkpoint and data identities differ')
    front = torch.load(paths['front'], map_location='cpu', weights_only=True)
    for name, value in model.front.state_dict().items():
        if not torch.equal(value.cpu(), front['front'][name]):
            raise ValueError('Frozen frontend identity differs: ' + name)
    if not torch.equal(model.prompt_embedding.cpu(), payload['prompt']):
        raise ValueError('Frozen prompt identity differs')
    teacher = copy.deepcopy(model).requires_grad_(False).eval()
    prefixes = ['frame_global.raw_global_phase', 'video_global.raw_global_phase', 'readout']
    encoders = ('frame_global.width_to_field', 'frame_global.tokens_to_field', 'video_global.width_to_field')
    if config.get('train_field_encoders', False):
        prefixes.extend(encoders)
    restrict_training_to(model, prefixes)
    parameter_groups = [
        {'params': [model.frame_global.raw_global_phase, model.video_global.raw_global_phase],
         'lr': config['phase_learning_rate'], 'name': 'phase'},
        {'params': model.readout.parameters(), 'lr': config['readout_learning_rate'], 'name': 'readout'},
    ]
    if config.get('train_field_encoders', False):
        parameters = [value for name, value in model.named_parameters()
                      if any(name.startswith(prefix + '.') for prefix in encoders)]
        parameter_groups.append({'params': parameters, 'lr': config['encoder_learning_rate'], 'name': 'field_encoders'})
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
        lambda epoch: .05 + .95 * (1 + math.cos(math.pi * epoch / config['epochs'])) / 2)
    anchors = [stage.phase().detach().clone() for stage in (model.frame_global, model.video_global)]
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=_paths.ROOT, text=True).strip()
    metadata = {'role': 'spatial_locality_refinement_development',
                'source_commit': os.environ.get('LIGHTGEN_SOURCE_COMMIT', head), 'checkout_HEAD': head,
                'source_sha256': {name: sha256(_paths.TOOLS / name)
                                  for name in ('refine_spatial_locality.py', 'spatial_locality.py')},
                'command': sys.argv, 'pid': os.getpid(), 'config': config,
                'initialization': {'checkpoint_sha256': sha256(args.weights), 'parent_epoch': parent_metadata.get('epoch')},
                'dataset': payload['identity'], 'front_source': front['source'],
                'python': platform.python_version(), 'torch': str(torch.__version__),
                'gpu': torch.cuda.get_device_name(0) if args.device.startswith('cuda') else 'CPU',
                'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
                'trainable_parameters': [name for name, value in model.named_parameters() if value.requires_grad],
                'test_used_for_selection': True, 'test_used_for_gradient': False,
                'training_forward_mode': 'evaluation mode with gradients enabled, preserving deterministic optics'}
    write(args.out / 'metadata.json', metadata)
    metrics, spatial = assess(model, payload, args.device, config)
    baseline_test, baseline_arrays = audit_spatial_locality(teacher, payload, args.device, split='test', field_count=35)
    write(args.out / 'baseline_spatial_test.json', baseline_test)
    np.savez_compressed(args.out / 'baseline_source_to_detector.npz', **baseline_arrays)
    history = [{'epoch': 0, 'metrics': metrics, 'spatial': spatial}]
    best_score = candidate_score(metrics, spatial, config)
    best_epoch = 0
    if best_score is not None:
        save_checkpoint(args.out / 'best_checkpoint.pt', model.checkpoint(**metadata, epoch=0, metrics=metrics, spatial=spatial))
    started = time.monotonic()
    for epoch in range(1, config['epochs'] + 1):
        model.eval()
        indices, valid = groups(payload, 'train', config['seed'] + epoch)
        losses, spatial_values = [], []
        with capture_modulated_fields(model) as captured:
            for step, start in enumerate(range(0, len(indices), config['batch_fields'])):
                item = batch(payload, indices[start:start + config['batch_fields']],
                             valid[start:start + config['batch_fields']], args.device)
                optimizer.zero_grad(set_to_none=True)
                result = model(item['vision'])
                keep = item['valid']
                target = ((item['target'] - model.target_mean) / model.target_std)[keep]
                prediction = result['normalized_prediction'][keep]
                with torch.no_grad():
                    teacher_prediction = teacher(item['vision'])['normalized_prediction'][keep]
                loss = F.smooth_l1_loss(prediction, target)
                loss = loss + config['correlation_weight'] * (1 - correlation(prediction, target))
                loss = loss + config['teacher_weight'] * F.smooth_l1_loss(prediction, teacher_prediction)
                slots = [(step * config['source_slots_per_step'] + offset) % 16
                         for offset in range(config['source_slots_per_step'])]
                local = spatial_locality_loss(model, captured, keep, slots,
                                              guard_weight=config.get('spatial_guard_weight', 0.))
                loss = loss + config['spatial_weight'] * local
                regularization, _ = router_regularization(result['routing'], keep)
                loss = loss + config['router_importance_weight'] * regularization['importance']
                phase_terms = model_phase_penalties(model)
                loss = loss + config['phase_curvature_weight'] * phase_terms['curvature']
                anchor = torch.stack([(1 - (stage.phase() - initial).cos()).mean()
                                      for stage, initial in zip((model.frame_global, model.video_global), anchors)]).mean()
                loss = loss + config['phase_anchor_weight'] * anchor
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError('Nonfinite spatial-refinement loss')
                loss.backward()
                smooth_phase_gradients(model, config['phase_gradient_smoothing_sigma'])
                gradient_norm = torch.nn.utils.clip_grad_norm_([value for value in model.parameters() if value.requires_grad], 5)
                if not bool(torch.isfinite(gradient_norm)):
                    raise FloatingPointError('Nonfinite spatial-refinement gradient')
                optimizer.step()
                smooth_phase_parameters(model, config.get('phase_parameter_smoothing_sigma', 0.))
                losses.append(float(loss.detach()))
                spatial_values.append(float(local.detach()))
                captured.clear()
                write(args.out / 'status.json', {'state': 'training', 'epoch': epoch, 'step': step + 1,
                      'steps_per_epoch': math.ceil(len(indices) / config['batch_fields']),
                      'best_epoch': best_epoch, 'elapsed_seconds': time.monotonic() - started})
        scheduler.step()
        row = {'epoch': epoch, 'loss': float(np.mean(losses)), 'training_foreign_loss': float(np.mean(spatial_values)),
               'elapsed_seconds': time.monotonic() - started}
        if epoch == 1 or epoch % config['eval_interval'] == 0 or epoch == config['epochs']:
            metrics, spatial = assess(model, payload, args.device, config)
            row.update(metrics=metrics, spatial=spatial)
            score = candidate_score(metrics, spatial, config)
            if score is not None and (best_score is None or score > best_score):
                best_score, best_epoch = score, epoch
                save_checkpoint(args.out / 'best_checkpoint.pt',
                                model.checkpoint(**metadata, epoch=epoch, metrics=metrics, spatial=spatial))
        save_checkpoint(args.out / 'last_checkpoint.pt', model.checkpoint(**metadata, epoch=epoch))
        history.append(row)
        write(args.out / 'history.json', history)
        print(json.dumps(row), flush=True)
    selected, selected_metadata = load_lightgen(args.out / 'best_checkpoint.pt', args.device)
    final = evaluate(selected, payload, args.device, config['eval_batch_fields'], args.out,
                     selected_metadata['epoch'])
    spatial, arrays = audit_spatial_locality(selected, payload, args.device, split='test', field_count=35)
    write(args.out / 'spatial_test_metrics.json', spatial)
    np.savez_compressed(args.out / 'source_to_detector_test.npz', **arrays)
    write(args.out / 'phase_statistics.json', {name: phase_statistics(getattr(selected, name))
          for name in ('frame_global', 'video_global')})
    save_checkpoint(args.out / 'best_checkpoint.pt', selected.checkpoint(**metadata,
                    epoch=selected_metadata['epoch'], metrics=final, spatial_test=spatial))
    write(args.out / 'status.json', {'state': 'completed', 'best_epoch': selected_metadata['epoch'],
          'SRCC': final['srcc'], 'spatial': spatial, 'elapsed_seconds': time.monotonic() - started})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--paths', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    try:
        train(args)
    except BaseException as error:
        if args.out.is_dir():
            write(args.out / 'failure.json', {'type': type(error).__name__, 'error': str(error)})
        raise


if __name__ == '__main__':
    main()
