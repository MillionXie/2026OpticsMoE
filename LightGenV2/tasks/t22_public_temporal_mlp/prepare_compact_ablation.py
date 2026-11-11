"""Prepare matched 256-hidden initializations using only original TRAIN inputs."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import subprocess

import torch
from torch.nn import functional as F

if __package__:
    from .data import batch, groups, load_inputs, sha256
    from .model import LightGenTemporal, load_lightgen
    from .run import ROOT, ensure_idle_gpu, save_checkpoint
else:
    from data import batch, groups, load_inputs, sha256
    from model import LightGenTemporal, load_lightgen
    from run import ROOT, ensure_idle_gpu, save_checkpoint


class LinearFit:
    """Streaming uncentered ridge fit. The expert has no intercept."""

    def __init__(self, inputs, outputs, device):
        self.gram = torch.zeros(inputs, inputs, dtype=torch.float64, device=device)
        self.cross = torch.zeros(inputs, outputs, dtype=torch.float64, device=device)
        self.target_energy = torch.zeros((), dtype=torch.float64, device=device)
        self.count = 0

    def add(self, inputs, outputs):
        x, y = inputs.double(), outputs.double()
        self.gram.add_(x.T @ x)
        self.cross.add_(x.T @ y)
        self.target_energy.add_(y.square().sum())
        self.count += len(x)

    def solve(self, ridge_fraction):
        penalty = self.gram.diag().mean() * ridge_fraction
        regularized = self.gram + torch.eye(
            len(self.gram), device=self.gram.device, dtype=torch.float64) * penalty
        coefficients = torch.linalg.solve(regularized, self.cross)
        error = (self.target_energy - 2 * (coefficients * self.cross).sum()
                 + (coefficients * (self.gram @ coefficients)).sum()).clamp_min(0)
        record = {'rows': self.count, 'ridge_fraction': ridge_fraction,
                  'ridge_penalty': float(penalty),
                  'relative_squared_error': float(error / self.target_energy.clamp_min(1e-20))}
        return coefficients.float(), record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--paths', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch-fields', type=int, default=4)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError('Initialization directory already exists')
    if args.device.startswith('cuda'):
        ensure_idle_gpu()
    torch.set_num_threads(4)
    torch.manual_seed(163)
    parent, parent_metadata = load_lightgen(args.source)
    if parent.cfg.readout_mode != 'five_levels' or parent.cfg.readout_hidden != 1024:
        raise ValueError('Expected the audited 1024-hidden five-level parent')
    paths = json.loads(args.paths.read_text('utf-8'))
    payload = load_inputs(paths)
    if payload['identity']['manifest']['sha256'] != parent_metadata['dataset']['manifest']['sha256']:
        raise ValueError('Parent manifest identity differs')
    # The original T06 pruning helper uses this downstream-weighted influence.
    influence = parent.readout[3].weight.norm(dim=0) * parent.readout[1].weight.norm(dim=1)
    keep = influence.argsort(descending=True)[:256].to(args.device)
    parent.to(args.device).eval()
    expert_fits = {name: [LinearFit(192, 192, args.device) for _ in range(4)]
                   for name in ('frame', 'video')}
    head_fit = LinearFit(257, 5, args.device)
    context = {}
    handles = []

    def expert_hook(stage, expert_index):
        def collect(module, inputs, output):
            valid = context['valid']
            x = inputs[0][valid].reshape(-1, 192)
            y = output[valid].reshape(-1, 192)
            expert_fits[stage][expert_index].add(x, y)
        return collect

    def head_hook(module, inputs, output):
        valid = context['valid']
        # Reuse retained neurons, then solve the five-logit output on TRAIN.
        flat = inputs[0][valid].flatten(-2)
        hidden = F.relu(F.linear(flat, parent.readout[1].weight[keep],
                                 parent.readout[1].bias[keep]))
        augmented = torch.cat((hidden, torch.ones_like(hidden[:, :1])), dim=-1)
        head_fit.add(augmented, output[valid])

    for stage in ('frame', 'video'):
        for index, expert in enumerate(getattr(parent, stage + '_experts').experts):
            handles.append(expert.register_forward_hook(expert_hook(stage, index)))
    handles.append(parent.readout.register_forward_hook(head_hook))
    indices, valid = groups(payload, 'train')
    with torch.inference_mode():
        for start in range(0, len(indices), args.batch_fields):
            item = batch(payload, indices[start:start + args.batch_fields],
                         valid[start:start + args.batch_fields], args.device)
            context['valid'] = item['valid']
            parent(item['vision'])
    for handle in handles:
        handle.remove()
    cfg = replace(parent.cfg, readout_hidden=256, orm_mode='amplitude',
                  alpha_min=2 / 3, alpha_max=2 / 3, alpha_eval=2 / 3)
    nonlinear = LightGenTemporal(cfg)
    state = parent.cpu().state_dict()
    common = {name: value for name, value in state.items() if not name.startswith('readout.')}
    missing, unexpected = nonlinear.load_state_dict(common, strict=False)
    if set(missing) != {'readout.1.weight', 'readout.1.bias', 'readout.3.weight', 'readout.3.bias'} or unexpected:
        raise ValueError('Unexpected compact initialization difference')
    keep = keep.cpu()
    coefficients, head_record = head_fit.solve(1e-5)
    with torch.no_grad():
        nonlinear.readout[1].weight.copy_(state['readout.1.weight'][keep])
        nonlinear.readout[1].bias.copy_(state['readout.1.bias'][keep])
        nonlinear.readout[3].weight.copy_(coefficients[:-1].T.cpu())
        nonlinear.readout[3].bias.copy_(coefficients[-1].cpu())
    linear = LightGenTemporal(replace(cfg, expert_mode='linear'))
    nonlinear_state = nonlinear.state_dict()
    common = {name: value for name, value in nonlinear_state.items()
              if not name.startswith(('frame_experts.', 'video_experts.'))}
    missing, unexpected = linear.load_state_dict(common, strict=False)
    expected = {stage + '_experts.experts.' + str(i) + '.weight'
                for stage in ('frame', 'video') for i in range(4)}
    if set(missing) != expected or unexpected:
        raise ValueError('Unexpected linear initialization difference')
    fit_records = {}
    with torch.no_grad():
        for stage, fits in expert_fits.items():
            fit_records[stage] = []
            for expert, fit in zip(getattr(linear, stage + '_experts').experts, fits):
                coefficients, record = fit.solve(1e-5)
                expert.weight.copy_(coefficients.T.cpu())
                fit_records[stage].append(record)
    common_equal = all(torch.equal(value, linear.state_dict()[name])
                       for name, value in common.items())
    if not common_equal:
        raise ValueError('A common tensor differs between ablation initializations')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    metadata = {
        'role': 'UNMEASURED_COMPACT_ABLATION_INITIALIZATION',
        'source_commit': os.environ.get('LIGHTGEN_SOURCE_COMMIT', commit),
        'source_SHA256': {p.name: sha256(p) for p in ROOT.glob('*.py')},
        'dataset': parent_metadata['dataset'],
        'parent_checkpoint': {'path': str(args.source), 'sha256': sha256(args.source)},
        'fitting_split': 'train', 'fitting_videos': 2250, 'TEST_fitting_videos': 0,
        'retained_neuron_indices': keep.tolist(),
        'head_fit': head_record, 'linear_expert_fits': fit_records,
        'common_initial_tensors_bitwise_equal': common_equal,
        'measured_metrics_available': False,
        'orm': 'alpha*D*U + (1-alpha)*U, alpha fixed at 2/3 in training and evaluation',
    }
    args.output_dir.mkdir(parents=True)
    for name, model in [('mlp', nonlinear), ('linear', linear)]:
        save_checkpoint(args.output_dir / (name + '.pt'),
                        model.checkpoint(**metadata, model_config=asdict(model.cfg)))
    (args.output_dir / 'preparation.json').write_text(json.dumps(metadata, indent=2) + '\n', 'utf-8')
    print(json.dumps({'output_dir': str(args.output_dir), 'head_fit': head_record,
                      'parameters': {name: sum(p.numel() for p in model.parameters())
                                     for name, model in [('mlp', nonlinear), ('linear', linear)]}}, indent=2))


if __name__ == '__main__':
    main()
