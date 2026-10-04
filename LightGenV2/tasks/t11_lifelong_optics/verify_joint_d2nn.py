"""Independently verify saved joint-D2NN predictions and checkpoint identity."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import torch

from .joint_d2nn import binary_metrics
from .run import save


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args(); run = args.run
    status = json.loads((run / 'status.json').read_text())
    metadata = json.loads((run / 'metadata.json').read_text())
    metrics = json.loads((run / 'metrics.json').read_text())
    history = json.loads((run / 'history.json').read_text())
    if status.get('state') != 'complete' or metadata.get('test_images_read') is not False:
        raise ValueError('Run is incomplete or its test-access contract is invalid')
    checkpoint = torch.load(run / 'best_checkpoint.pt', map_location='cpu', weights_only=False)
    if checkpoint['epoch'] != metrics['selected_epoch']:
        raise ValueError('Selected epoch and checkpoint disagree')
    verified = {}
    for name in 'ABCD':
        with np.load(run / f'{name}_validation_predictions.npz', allow_pickle=False) as values:
            p = torch.from_numpy(values['probabilities'])
            y = torch.from_numpy(values['labels']).long()
        verified[name] = binary_metrics(p, y)
        if verified[name]['confusion'] != metrics[name]['validation']['confusion']:
            raise ValueError(f'{name} confusion matrix mismatch')
        if abs(verified[name]['balanced_accuracy'] - metrics[name]['validation']['balanced_accuracy']) > 1e-7:
            raise ValueError(f'{name} balanced accuracy mismatch')
    mean = float(np.mean([verified[name]['balanced_accuracy'] for name in 'ABCD']))
    if abs(mean - metrics['selection_score']) > 1e-7:
        raise ValueError('Selection score mismatch')
    output = {
        'state': 'verified',
        'source_commit': metadata['commit'],
        'verification_commit': subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'selected_epoch': checkpoint['epoch'],
        'selection_score': mean,
        'best_checkpoint_sha256': sha256(run / 'best_checkpoint.pt'),
        'last_checkpoint_sha256': sha256(run / 'last_checkpoint.pt'),
        'test_images_read': False,
        'label_contract': {'0': 'tumor', '1': 'non_tumor_or_normal'},
        'validation': verified,
        'history_best': max((row['selection_score'], row['epoch']) for row in history),
        'note': 'The source run swapped only the names of its two per-class recall fields; confusion, accuracy, balanced accuracy, NLL, checkpoint selection, and weights were unaffected.',
    }
    save(run / 'verification.json', output)
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
