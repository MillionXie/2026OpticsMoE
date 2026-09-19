"""Reload a four-task checkpoint and reproduce validation metrics without training."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .continual_four_dataset import GROUP_MASKS, PREFIX_MASKS
from .cross_dataset import evaluate, load_dataset
from .data import sha
from .evaluate_three_dataset import select_by_identity
from .model import OpticalMoE
from .run import save


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    for name in 'abcd':
        parser.add_argument('--task-' + name, type=Path, required=True)
        parser.add_argument('--task-' + name + '-manifest', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    cfg = json.loads((args.run / 'config.json').read_text())
    split = json.loads((args.run / 'split.json').read_text())
    recorded = json.loads((args.run / 'metrics.json').read_text())
    checkpoint = args.run / 'D' / 'best_checkpoint.pt'
    state = torch.load(checkpoint, map_location=args.device, weights_only=False)
    model = OpticalMoE(cfg).to(args.device); model.load_state_dict(state['model'])
    results = {'checkpoint': str(checkpoint), 'checkpoint_sha256': sha(checkpoint), 'checkpoint_stage': state.get('stage'), 'checkpoint_epoch': state.get('epoch'), 'headline_metric': 'balanced_accuracy', 'test_images_read': False, 'interpretation': 'Expert masks alter coherent interference; masked results are diagnostics, not additive knowledge estimates.'}
    for j, name in enumerate('ABCD'):
        data, _ = load_dataset(getattr(args, 'task_' + name.lower()), getattr(args, 'task_' + name.lower() + '_manifest'))
        ids = select_by_identity(data['val_ids'], split[name]['val_ids'], 'task ' + name + ' validation')
        images = torch.from_numpy(data['val_images'][ids]); labels = torch.from_numpy(data['val_labels'][ids]).long()
        variants = [('all', None), ('own_group', GROUP_MASKS[j]), ('learned_prefix', PREFIX_MASKS[j])]
        if j: variants.append(('previous_prefix', PREFIX_MASKS[j - 1]))
        for label, mask in variants:
            metrics, _, _ = evaluate(model, images, labels, cfg['batch_size'], mask)
            results[name + '_' + label] = metrics
    for name in 'ABC':
        results[name + '_BWT_after_D'] = results[name + '_all']['balanced_accuracy'] - recorded['snapshots'][name]['metrics'][name]['balanced_accuracy']
    for name in 'ABCD':
        expected, actual = recorded[name + '_all'], results[name + '_all']
        if actual['confusion'] != expected['confusion'] or abs(actual['balanced_accuracy'] - expected['balanced_accuracy']) > 1e-12:
            raise RuntimeError('Reloaded ' + name + ' metrics do not match the recorded run')
    save(args.run / 'reevaluation_four.json', results)
    print(json.dumps({key: value['balanced_accuracy'] for key, value in results.items() if isinstance(value, dict) and 'balanced_accuracy' in value}, indent=2))


if __name__ == '__main__':
    main()
