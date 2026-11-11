"""Evaluate one parameter-averaged model with the existing inference graph."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
import torch

if __package__:
    from .data import load_inputs, sha256
    from .model import load_lightgen
    from .run import ROOT, ensure_idle_gpu, evaluate, save_checkpoint, write
else:
    from data import load_inputs, sha256
    from model import load_lightgen
    from run import ROOT, ensure_idle_gpu, evaluate, save_checkpoint, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoints', type=Path, nargs='+', required=True)
    parser.add_argument('--paths', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if len(args.checkpoints) < 2:
        raise ValueError('Parameter averaging requires at least two checkpoints')
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError('Refusing to overwrite run: ' + str(out))

    paths = json.loads(args.paths.read_text('utf-8'))
    payload = load_inputs(paths)
    model, first_metadata = load_lightgen(args.checkpoints[0])
    parameter_names = set(dict(model.named_parameters()))
    original = model.state_dict()
    sums = {name: value.double().clone() for name, value in original.items()
            if name in parameter_names}
    parents = []
    for index, path in enumerate(args.checkpoints):
        parent, metadata = load_lightgen(path)
        if parent.cfg != model.cfg:
            raise ValueError('Parent model configurations differ')
        for name in ['manifest', 'vision_cache', 'language_cache']:
            if metadata['dataset'][name]['sha256'] != payload['identity'][name]['sha256']:
                raise ValueError('Parent data identity differs: ' + name)
        for name, value in parent.state_dict().items():
            if name in parameter_names:
                if index:
                    sums[name].add_(value.double())
            elif not torch.equal(value, original[name]):
                raise ValueError('Parent inference buffers differ: ' + name)
        parents.append({'path': str(path), 'sha256': sha256(path),
                        'coefficient': 1 / len(args.checkpoints),
                        'source_commit': metadata['source_commit']})

    for name, value in sums.items():
        original[name] = (value / len(args.checkpoints)).to(original[name].dtype)
    model.load_state_dict(original, strict=True)
    admission = ensure_idle_gpu() if args.device.startswith('cuda') else 'CPU'
    model = model.to(args.device).eval()
    torch.set_num_threads(4)
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    metadata = {
        'role': 'PUBLIC_SURROGATE_DEVELOPMENT',
        'method': 'uniform_parameter_average',
        'parents': parents,
        'test_used_for_selection': True,
        'test_used_for_gradient': False,
        'validation_used': False,
        'dataset': payload['identity'],
        'config': first_metadata['config'],
        'model_config': asdict(model.cfg),
        'model_parameters': sum(value.numel() for value in model.parameters()),
        'source_commit': os.environ.get('LIGHTGEN_SOURCE_COMMIT', head),
        'checkout_HEAD': head,
        'source_SHA256': {path.name: sha256(path) for path in ROOT.glob('*.py')},
        'python': platform.python_version(),
        'torch': str(torch.__version__),
        'platform': platform.platform(),
        'gpu_admission': admission,
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'command': sys.argv,
    }
    out.mkdir(parents=True)
    write(out / 'metadata.json', metadata)
    write(out / 'status.json', {'state': 'running', 'method': metadata['method']})
    metrics = evaluate(model, payload, args.device, 4, out)
    config = metadata['config']
    metrics['reference_srcc'] = config['reference_srcc']
    metrics['absolute_srcc_difference'] = abs(metrics['srcc'] - config['reference_srcc'])
    metrics['within_target'] = (
        metrics['absolute_srcc_difference'] <= config['allowed_absolute_srcc_difference'])
    cycles = [evaluate(model, payload, args.device, 4, slot_shift=index)['srcc']
              for index in range(16)]
    write(out / 'slot_cycle_audit.json', {'srcc_by_shift': cycles,
                                       'mean_srcc': float(np.mean(cycles)),
                                       'minimum_srcc': float(np.min(cycles))})
    checkpoint = model.checkpoint(**metadata, metrics=metrics)
    save_checkpoint(out / 'best_checkpoint.pt', checkpoint)
    save_checkpoint(out / 'LightGen.pt', checkpoint)
    write(out / 'metrics.json', metrics)
    write(out / 'status.json', {'state': 'completed', **metrics})
    print(json.dumps(metrics), flush=True)


if __name__ == '__main__':
    main()
