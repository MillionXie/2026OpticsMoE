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


def perturb_intensity(intensity, kind, level, reference, generator, *, straight_through=False):
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
        observed = torch.poisson(intensity.clamp_min(0) * gain, generator=generator) / gain
        if straight_through:
            # Exact Poisson samples in forward, identity surrogate gradient in training.
            return observed.detach() + (intensity - intensity.detach())
        return observed
    if kind == 'combined':
        if level == 0:
            return intensity
        # At I_reference, read and shot noise each contribute half the total variance.
        observed = perturb_intensity(intensity, 'poisson', 2 / level**2, reference,
                                     generator, straight_through=straight_through)
        return perturb_intensity(observed, 'gaussian', level / math.sqrt(2), reference, generator)
    raise ValueError('Unsupported detector noise model')


@contextmanager
def detector_noise(model, kind, level, reference, seed, *, straight_through=False):
    """Perturb both detector stages before their existing mean normalization."""
    with ExitStack() as stack:
        for index, (branch, attribute) in enumerate(STAGES.items()):
            stage = getattr(model, attribute)
            generator = torch.Generator(device=stage.raw_global_phase.device).manual_seed(seed + 100 * index)
            original = stage.process_detector

            def process(intensity, original=original, generator=generator, branch=branch):
                observed = perturb_intensity(intensity, kind, level, reference[branch], generator,
                                             straight_through=straight_through)
                return original(observed)

            stack.enter_context(patch.object(stage, 'process_detector', process))
        yield


def noise_level(kind, fraction):
    """Express shot noise as the count giving the requested RMS at I_reference."""
    if not math.isfinite(fraction) or fraction <= 0:
        raise ValueError('Reference noise fraction must be positive and finite')
    return 1 / fraction**2 if kind == 'poisson' else fraction


def detector_noise_metrics(model, payload, device, batch_fields, config, out=None):
    records = []
    for kind in config.get('evaluation_kinds', ('gaussian', 'poisson', 'combined')):
        for fraction in config['evaluation_fractions']:
            for trial in range(config['selection_trials']):
                seed = config['selection_seed_base'] + trial
                directory = None
                if out is not None:
                    directory = Path(out) / f'{kind}_{fraction:g}_trial{trial+1}'
                    directory.mkdir(parents=True, exist_ok=False)
                with detector_noise(model, kind, noise_level(kind, fraction), config['reference_intensity'], seed):
                    metrics = evaluate(model, payload, device, batch_fields, directory, 0)
                records.append({'kind': kind, 'fraction': fraction, 'trial': trial + 1,
                                'seed': seed, 'srcc': metrics['srcc']})
    return records


def detector_noise_score(records, config):
    """Use the least margin over every configured kind, level and random seed."""
    expected = config.get('evaluation_kinds', ('gaussian', 'poisson', 'combined'))
    if len(config['evaluation_fractions']) != len(config['minimum_srcc']) or not config['evaluation_fractions']:
        raise ValueError('Each noise fraction requires one target SRCC')
    margins = []
    for kind in expected:
        for fraction, target in zip(config['evaluation_fractions'], config['minimum_srcc']):
            scores = [row['srcc'] for row in records if row['kind'] == kind and row['fraction'] == fraction]
            if len(scores) != config['selection_trials'] or not all(math.isfinite(value) for value in scores):
                raise ValueError('Incomplete detector noise selection conditions')
            margins.append(min(scores) - target)
    return min(margins)


@torch.no_grad()
def reference_intensities(model, payload, device, batch_fields):
    """Use 2250 TRAIN videos to fix one mean intensity for each detector stage."""
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
    for kind in ('gaussian', 'poisson', 'combined'):
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
    parser.add_argument('--profile', type=Path, help='Use fixed reference intensities and noise fractions from a training profile')
    parser.add_argument('--seed-base', type=int, default=29301)
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
        profile = json.loads(args.profile.read_text('utf-8'))['detector_noise'] if args.profile else None
        reference = (profile['reference_intensity'] if profile else
                     reference_intensities(model, payload, args.device, args.batch_fields))
        clean_dir = args.out / 'clean'
        clean_dir.mkdir()
        clean = evaluate(model, payload, args.device, args.batch_fields, clean_dir, 0)
        protocol = {
            'checkpoint_sha256': checkpoint_hash, 'data_identity': payload['identity'],
            'videos': 558, 'optical_fields': 35, 'padding_excluded': True,
            'reference_intensity': reference,
            'reference_source': (profile['reference_source'] if profile else
                                 '2250 TRAIN videos with four frames each, without label use'),
            'injection_location': 'Both detector regions after squared-magnitude detection and before existing mean normalization',
            'gaussian_definition': 'I_observed = max(I + r * I_reference * Normal(0,1), 0)',
            'poisson_definition': 'I_observed = Poisson(I * N / I_reference) * I_reference / N',
            'reference_is_fixed_across_test_images': True,
            'noise_is_resampled_each_observation': True, 'hardware_calibrated': False,
            'combined_definition': 'Independent read and shot noise, each contributing half the variance at I_reference',
            'trials': args.trials, 'seed_base': args.seed_base,
            'source_commit': os.environ.get('LIGHTGEN_SOURCE_COMMIT', subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'], cwd=_paths.ROOT, text=True).strip()),
            'source_sha256': {name: sha256(_paths.source_path(name))
                              for name in ('model.py', 'evaluate_detector_noise.py', 'run.py')},
            'environment': {'python': platform.python_version(), 'torch': str(torch.__version__),
                            'device': torch.cuda.get_device_name(0) if args.device.startswith('cuda') else 'CPU'},
        }
        write(args.out / 'protocol.json', protocol)
        records = []
        conditions = ([(kind, tuple(noise_level(kind, fraction) for fraction in profile['evaluation_fractions']))
                       for kind in profile['evaluation_kinds']] if profile else
                      [('gaussian', (.01, .05, .10)), ('poisson', (100., 1000., 10000.))])
        if args.profile:
            protocol['profile_sha256'] = sha256(args.profile)
            protocol['noise_fractions'] = profile['evaluation_fractions']
            write(args.out / 'protocol.json', protocol)
        for kind, levels in conditions:
            for level in levels:
                for trial in range(args.trials):
                    seed = args.seed_base + trial
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
