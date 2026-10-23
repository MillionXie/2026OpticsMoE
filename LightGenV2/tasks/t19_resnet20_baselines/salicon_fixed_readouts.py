"""Two predeclared SALICON references without a learned saliency task head.

TRAIN mean density is an image-independent statistical prior, not a ResNet.
Frozen ImageNet feature energy is an image-dependent fixed readout, not a
saliency-trained model. Both use all 5,000 existing public-val TEST targets;
neither is selected or tuned against a desired score window.
"""
import argparse
import json
import os
from pathlib import Path
import platform
import sys
import time

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .continue_lgvq50 import shutdown_loader, structured_sha256
from .frozen_resnet18 import FrozenResNet18, assert_encoder_unchanged, encoder_state_sha256, file_sha256
from .task_problems import Problem
from .train_easy100 import TASK, git, require_published_source, write_json, save_checkpoint


def energy_density(energy, size=224):
    """Fixed bilinear readout of mean squared last-stage activations."""
    if energy.ndim != 4 or energy.shape[1] != 1 or not torch.isfinite(energy).all() or (energy < 0).any():
        raise ValueError('Expected finite nonnegative [B,1,H,W] feature energy')
    values = F.interpolate(energy.float(), (size, size), mode='bilinear', align_corners=False) + 1e-8
    return values / values.flatten(1).sum(1)[:, None, None, None]


def cc_per_image(predicted, target):
    x, y = predicted.float().flatten(1), target.float().flatten(1)
    x, y = x - x.mean(1, keepdim=True), y - y.mean(1, keepdim=True)
    denominator = x.square().sum(1).sqrt() * y.square().sum(1).sqrt()
    return (x * y).sum(1) / denominator.clamp_min(1e-8)


def run(args):
    require_published_source()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('This diagnostic requires CUDA_VISIBLE_DEVICES empty')
    run_dir = args.run_dir.resolve()
    if not run_dir.is_relative_to((TASK / 'runs/simulation').resolve()):
        raise ValueError('Diagnostic must remain inside task simulation runs')
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest = dict(status='preflight', pid=os.getpid(), started_unix=time.time(),
        command=[sys.executable, '-m', __spec__.name, *sys.argv[1:]], git_commit=git('rev-parse', 'HEAD'),
        source_sha256={name: file_sha256(TASK / name) for name in (
            'salicon_fixed_readouts.py', 'task_problems.py', 'other_data.py', 'frozen_resnet18.py')},
        device='cpu', threads=4, python=platform.python_version(), torch=torch.__version__,
        optimizer_steps=0, learned_task_head_parameters=0, test_passes_per_reference=1,
        definitions={'train_mean_prior': 'mean of all10000 normalized TRAIN densities; no image input at TEST',
            'frozen_feature_energy': 'mean squared512-channel final-stage ImageNet RN18 activations; '
                'bilinear7x7_to224; add1e-8; normalize mass; no fitted readout or saliency weights'},
        historical_test_observed=True, score_window_used_for_selection=False)
    write_json(run_dir / 'run_manifest.json', manifest)
    loaders = []
    began = time.monotonic()
    try:
        torch.set_num_threads(4)
        problem = Problem('salicon')
        if (len(problem.train), len(problem.test)) != (10000, 5000) or problem.settings.augmentation_enabled:
            raise ValueError('Diagnostic requires the original counts and deterministic disabled augmentation')
        manifest['data_identity'] = problem.identity
        manifest['status'] = 'building_train_prior'
        write_json(run_dir / 'run_manifest.json', manifest)
        options = dict(batch_size=16, num_workers=2, multiprocessing_context='spawn', persistent_workers=True,
                       generator=torch.Generator().manual_seed(42), shuffle=False, pin_memory=False)
        train = DataLoader(problem.train, **options)
        loaders.append(train)
        total = torch.zeros(1, 1, 224, 224, dtype=torch.float64)
        train_ids = []
        for step, batch in enumerate(train):
            total += batch['density'].double().sum(0, keepdim=True)
            train_ids.extend(batch['sample_id'])
            if step % 25 == 0:
                write_json(run_dir / 'progress.json', dict(phase='train_mean_prior', samples=len(train_ids)))
        if len(train_ids) != 10000 or len(set(train_ids)) != 10000:
            raise RuntimeError('TRAIN prior identities incomplete')
        prior = (total / len(train_ids)).float()
        prior = prior / prior.sum()
        model = FrozenResNet18().cpu().eval()
        frozen_sha = encoder_state_sha256(model)
        manifest.update(status='evaluating_fixed_readouts', pretrained_provenance=model.pretrained_provenance,
                        encoder_state_sha256=frozen_sha, frozen_feature_model_parameters=sum(p.numel() for p in model.parameters()),
                        prior_training_ids_sha256=structured_sha256(train_ids))
        write_json(run_dir / 'run_manifest.json', manifest)
        test = DataLoader(problem.test, **options)
        loaders.append(test)
        records, energies, test_ids = [], [], []
        with torch.no_grad():
            for step, batch in enumerate(test):
                features = model.features(batch['image'])[-1]
                energy = features.float().square().mean(1, keepdim=True)
                predicted = energy_density(energy)
                targets = batch['density']
                image_cc = cc_per_image(predicted, targets)
                prior_cc = cc_per_image(prior.expand_as(targets), targets)
                energies.append(energy)
                test_ids.extend(batch['sample_id'])
                records.extend(dict(sample_id=sid, frozen_feature_energy_cc=float(a), train_mean_prior_cc=float(b))
                               for sid, a, b in zip(batch['sample_id'], image_cc, prior_cc))
                if step % 25 == 0:
                    write_json(run_dir / 'progress.json', dict(phase='test_fixed_readouts', samples=len(test_ids)))
        if (len(records) != 5000 or len(set(test_ids)) != 5000 or set(train_ids) & set(test_ids)):
            raise RuntimeError('Reference TEST identities incomplete or overlap TRAIN')
        manifest['frozen_encoder_final_audit'] = assert_encoder_unchanged(model, frozen_sha)
        saved_maps = dict(train_prior=prior, train_ids=train_ids, test_ids=test_ids,
                          feature_energy_7x7=torch.cat(energies))
        save_checkpoint(run_dir / 'fixed_readout_maps.pt', saved_maps)
        write_json(run_dir / 'fixed_weight_predictions.json', records)
        metrics = {key: sum(row[key] for row in records) / len(records)
                   for key in ('frozen_feature_energy_cc', 'train_mean_prior_cc')}
        write_json(run_dir / 'result.json', dict(status='complete', metrics=metrics,
            samples=5000, optimizer_steps=0, learned_task_head_parameters=0,
            maps_sha256=file_sha256(run_dir / 'fixed_readout_maps.pt'),
            maps_content_sha256=structured_sha256(saved_maps), test_passes_per_reference=1,
            elapsed_seconds=time.monotonic() - began, definitions=manifest['definitions'],
            score_window_used_for_selection=False))
        manifest['status'] = 'complete'
    except BaseException as exc:
        manifest.update(status='failed', error=repr(exc))
        raise
    finally:
        manifest['finished_unix'] = time.time()
        write_json(run_dir / 'run_manifest.json', manifest)
        for loader in loaders:
            shutdown_loader(loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    run(parser.parse_args())
