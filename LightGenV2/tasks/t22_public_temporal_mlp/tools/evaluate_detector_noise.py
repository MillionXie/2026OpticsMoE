"""Measure frozen-checkpoint sensitivity to explicitly defined detector noise."""
from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
import json
import math
import os
from pathlib import Path
import platform
import subprocess

import numpy as np
import torch
from unittest.mock import patch

if __package__:
    from . import _paths
    from .data import batch, groups, load_inputs, sha256
    from .run import evaluate, write
    from .spatial_locality import STAGES
    from ..model import load_lightgen
else:
    import _paths
    from data import batch, groups, load_inputs, sha256
    from run import evaluate, write
    from spatial_locality import STAGES
    from model import load_lightgen


def perturb_intensity(intensity, kind, level, reference, generator):
    """Add read noise or sample photon counts in the detector intensity domain."""
    if not math.isfinite(reference) or reference <= 0:
        raise ValueError('A positive finite reference intensity is required')
    if not math.isfinite(level) or level < 0:
        raise ValueError('Noise level must be finite and nonnegative')
    if kind == 'gaussian':
        if level == 0:
            return intensity
        noise = torch.randn(intensity.shape, generator=generator,
                            dtype=intensity.dtype, device=intensity.device)
        return (intensity + level * reference * noise).clamp_min(0)
    if kind == 'poisson':
        if level == 0:
            raise ValueError('Photon count at the reference intensity must be positive')
        gain = level / reference
        return torch.poisson(intensity.clamp_min(0) * gain, generator=generator) / gain
    raise ValueError('Unsupported detector noise model')


@contextmanager
def detector_noise(model, kind, level, reference, seed):
    """Perturb both detector stages before their existing mean normalization."""
    with ExitStack() as stack:
        for index, (branch, attribute) in enumerate(STAGES.items()):
            stage = getattr(model, attribute)
            generator = torch.Generator(device=stage.raw_global_phase.device).manual_seed(seed + 100 * index)
            original = stage.process_detector

            def process(intensity, original=original, generator=generator, branch=branch):
                observed = perturb_intensity(intensity, kind, level, reference[branch], generator)
                return original(observed)

            stack.enter_context(patch.object(stage, 'process_detector', process))
        yield


@torch.no_grad()
def reference_intensities(model, payload, device, batch_fields):
    """Use only TRAIN images to fix one mean intensity for each detector stage."""
    totals = {branch: 0. for branch in STAGES}
    counts = {branch: 0 for branch in STAGES}
    current_valid = None
    indices, valid = groups(payload, 'train')
    model.eval()
    with ExitStack() as stack:
        for branch, attribute in STAGES.items():
            stage = getattr(model, attribute)
            original = stage.process_detector

            def capture(intensity, original=original, branch=branch):
                means = intensity.mean((-2, -1))[current_valid]
                totals[branch] += float(means.double().sum())
                counts[branch] += means.numel()
                return original(intensity)

            stack.enter_context(patch.object(stage, 'process_detector', capture))
        for start in range(0, len(indices), batch_fields):
            item = batch(payload, indices[start:start + batch_fields], valid[start:start + batch_fields], device)
            current_valid = item['valid']
            model(item['vision'])
    if counts != {'vision': 2250 * 4, 'language': 2250}:
        raise ValueError('Reference images or padding counts differ')
    values = {branch: totals[branch] / counts[branch] for branch in STAGES}
    if any(not math.isfinite(value) or value <= 0 for value in values.values()):
        raise ValueError('Detector intensity reference is dark or nonfinite')
    return values


def summarize(records, clean):
    rows = []
    for kind in ('gaussian', 'poisson'):
        levels = sorted({row['level'] for row in records if row['kind'] == kind})
        for level in levels:
            values = [row['metrics']['srcc'] for row in records if row['kind'] == kind and row['level'] == level]
            rows.append({'kind': kind, 'level': level, 'trials': len(values),
                         'mean_srcc': float(np.mean(values)), 'minimum_srcc': min(values),
                         'maximum_srcc': max(values), 'mean_delta_srcc': float(np.mean(values)) - clean})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--paths', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--trials', type=int, default=5)
    parser.add_argument('--batch-fields', type=int, default=4)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if args.trials < 1 or args.batch_fields < 1:
        raise ValueError('Trials and field batches must be positive')
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    try:
        checkpoint_hash = sha256(args.weights)
        model, metadata = load_lightgen(args.weights, args.device)
        model.requires_grad_(False).eval()
        payload = load_inputs(json.loads(args.paths.read_text('utf-8')))
        if payload['identity']['manifest']['sha256'] != metadata['dataset']['manifest']['sha256']:
            raise ValueError('Checkpoint data identity differs')
        reference = reference_intensities(model, payload, args.device, args.batch_fields)
        clean_dir = args.out / 'clean'
        clean_dir.mkdir()
        clean = evaluate(model, payload, args.device, args.batch_fields, clean_dir, 0)
        protocol = {
            'checkpoint_sha256': checkpoint_hash, 'data_identity': payload['identity'],
            'videos': 558, 'optical_fields': 35, 'padding_excluded': True,
            'reference_intensity': reference, 'reference_source': '2250 TRAIN images, without label use',
            'injection_location': 'Both detector regions after squared-magnitude detection and before existing mean normalization',
            'gaussian_definition': 'I_observed = max(I + r * I_reference * Normal(0,1), 0)',
            'poisson_definition': 'I_observed = Poisson(I * N / I_reference) * I_reference / N',
            'reference_is_fixed_across_test_images': True,
            'noise_is_resampled_each_observation': True, 'hardware_calibrated': False,
            'trials': args.trials, 'seed_base': 29301,
            'source_commit': os.environ.get('LIGHTGEN_SOURCE_COMMIT', subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'], cwd=_paths.ROOT, text=True).strip()),
            'source_sha256': {name: sha256(_paths.source_path(name))
                              for name in ('model.py', 'evaluate_detector_noise.py', 'run.py')},
            'environment': {'python': platform.python_version(), 'torch': str(torch.__version__),
                            'device': torch.cuda.get_device_name(0) if args.device.startswith('cuda') else 'CPU'},
        }
        write(args.out / 'protocol.json', protocol)
        records = []
        for kind, levels in (('gaussian', (.01, .05, .10)), ('poisson', (100., 1000., 10000.))):
            for level in levels:
                for trial in range(args.trials):
                    seed = 29301 + trial
                    directory = args.out / f'{kind}_{level:g}_trial{trial+1}'
                    directory.mkdir()
                    with detector_noise(model, kind, level, reference, seed):
                        metrics = evaluate(model, payload, args.device, args.batch_fields, directory, 0)
                    records.append({'kind': kind, 'level': level, 'trial': trial + 1, 'seed': seed,
                                    'metrics': metrics})
                    write(args.out / 'comparison.json', {'clean': clean, 'conditions': records,
                          'summary': summarize(records, clean['srcc'])})
                    write(args.out / 'status.json', {'state': 'running', 'completed_conditions': len(records)})
                    print(json.dumps({'kind': kind, 'level': level, 'trial': trial + 1,
                                      'srcc': metrics['srcc']}), flush=True)
        if sha256(args.weights) != checkpoint_hash:
            raise ValueError('Source checkpoint changed during the audit')
        write(args.out / 'status.json', {'state': 'completed', 'completed_conditions': len(records)})
    except BaseException as error:
        write(args.out / 'status.json', {'state': 'failed', 'type': type(error).__name__, 'error': str(error)})
        raise


if __name__ == '__main__':
    main()
