"""Measure single-video optical energy reaching the other video regions."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import torch

if __package__:
    from . import _paths
    from .data import load_inputs, groups, batch, sha256
    from .run import write
    from ..model import load_lightgen
else:
    import _paths
    from data import load_inputs, groups, batch, sha256
    from run import write
    from model import load_lightgen


STAGES = {'vision': 'frame_global', 'language': 'video_global'}


@contextmanager
def capture_modulated_fields(model, *, include_incident=False):
    """Retain the two actual full-field propagation inputs, including gradients."""
    captured = {}
    handles = []

    def hook(name):
        def capture(module, arguments):
            captured[name] = arguments[0]
        return capture

    for name, attribute in STAGES.items():
        stage = getattr(model, attribute)
        handles.append(stage.propagation.register_forward_pre_hook(hook(name)))
        if include_incident:
            def incident_hook(module, arguments, branch=name):
                captured[branch + '_incident'] = module.fields(arguments[0])
            handles.append(stage.register_forward_pre_hook(incident_hook))
    try:
        yield captured
    finally:
        for handle in handles:
            handle.remove()
        captured.clear()


def video_region_masks(stage, field):
    masks = field.real.new_zeros(len(stage.origins), *field.shape[-2:])
    for slot, (row, column) in enumerate(stage.origins):
        masks[slot, row:row + 115, column:column + 115] = 1
    return masks


def isolated_video_response(stage, modulated_field, slots):
    """Propagate selected sources across the whole original coherent canvas.

    Rows index source video slots and columns index detector video regions.
    This auxiliary diagnostic preserves the original propagation operator.
    """
    slots = torch.as_tensor(slots, dtype=torch.long, device=modulated_field.device)
    masks = video_region_masks(stage, modulated_field)
    sources = modulated_field[:, None] * masks[slots][None]
    leading = sources.shape[:2]
    # Auxiliary propagation bypasses capture hooks, leaving the actual joint
    # forward in capture_modulated_fields unchanged.
    propagated = stage.propagation.forward(sources.flatten(0, 1))
    intensity = propagated.abs().square().float().reshape(*leading, *sources.shape[-2:])
    total_energy = intensity.sum((-2, -1))
    total = total_energy.clamp_min(1e-12)
    energy = torch.einsum('bshw,vhw->bsv', intensity, masks)
    fractions = energy / total[..., None]
    own = fractions.gather(-1, slots[None, :, None].expand(len(fractions), -1, 1)).squeeze(-1)
    foreign = (fractions.sum(-1) - own).clamp_min(0)
    input_energy = sources.abs().square().sum((-2, -1)).clamp_min(1e-12)
    guard = torch.where(total_energy > 0, (1 - fractions.sum(-1)).clamp_min(0),
                        torch.zeros_like(total_energy))
    return {'fractions': fractions, 'foreign_fraction': foreign, 'own_fraction': own,
            'guard_fraction': guard, 'filter_transmission': total_energy / input_energy,
            'dark_source_fraction': (total_energy == 0).to(total_energy)}


def spatial_locality_loss(model, captured, valid, slots, *, guard_weight=0., foreign_threshold=0.):
    if not 0 <= foreign_threshold <= 1:
        raise ValueError('Cross-video energy threshold must lie in [0,1]')
    penalties = []
    for name, attribute in STAGES.items():
        response = isolated_video_response(getattr(model, attribute), captured[name], slots)
        weights = valid[:, slots].to(response['foreign_fraction'])
        penalty = ((response['foreign_fraction'] - foreign_threshold).clamp_min(0)
                   + guard_weight * response['guard_fraction']
                   + response['dark_source_fraction'])
        penalties.append((penalty * weights).sum() / weights.sum().clamp_min(1))
    return torch.stack(penalties).mean()


@torch.no_grad()
def audit_spatial_locality(model, payload, device, *, split='test', field_count=8):
    indices, valid = groups(payload, split)
    chosen = np.unique(np.linspace(0, len(indices) - 1, min(field_count, len(indices))).astype(int))
    matrices = {name: torch.zeros(16, 16, dtype=torch.float64) for name in STAGES}
    counts = torch.zeros(16, dtype=torch.float64)
    measures = ('foreign_fraction', 'own_fraction', 'guard_fraction', 'filter_transmission', 'dark_source_fraction')
    per_video = {name: {key: [] for key in ('fractions', *measures)} for name in STAGES}
    sample_ids = []
    model.eval()
    with capture_modulated_fields(model) as captured:
        for field_number in chosen:
            item = batch(payload, indices[field_number:field_number + 1], valid[field_number:field_number + 1], device)
            model(item['vision'])
            keep = item['valid'][0].cpu()
            counts += keep
            sample_ids.extend(payload['ids'][int(index)] for index in item['indices'][0][keep])
            for name, attribute in STAGES.items():
                for start in range(0, 16, 4):
                    slots = list(range(start, start + 4))
                    response = isolated_video_response(getattr(model, attribute), captured[name], slots)
                    matrices[name][slots] += response['fractions'][0].cpu().double() * keep[slots, None]
                    for key in per_video[name]:
                        per_video[name][key].append(response[key][0].cpu().double()[keep[slots]])
    summary = {'split': split, 'fields': chosen.tolist(), 'videos': len(sample_ids),
               'definition': 'Single-video illumination across the full canvas with the trained modulation and propagation'}
    arrays = {'sample_ids': np.asarray(sample_ids)}
    for name in STAGES:
        matrix = matrices[name] / counts.clamp_min(1)[:, None]
        arrays[name + '_source_to_detector_fraction'] = matrix.numpy()
        for key, values in per_video[name].items():
            arrays[name + '_per_video_' + key] = torch.cat(values).numpy()
        summary[name] = {key: float(arrays[name + '_per_video_' + key].mean()) for key in measures}
        summary[name]['maximum_source_foreign_fraction'] = float((matrix.sum(-1) - matrix.diagonal()).max())
        summary[name]['maximum_video_foreign_fraction'] = float(arrays[name + '_per_video_foreign_fraction'].max())
    return summary, arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--paths', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--split', choices=('train', 'test'), default='test')
    parser.add_argument('--fields', type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(4)
    checkpoint_hash = sha256(args.weights)
    model, _ = load_lightgen(args.weights, args.device)
    payload = load_inputs(json.loads(args.paths.read_text('utf-8')))
    summary, arrays = audit_spatial_locality(model, payload, args.device, split=args.split, field_count=args.fields)
    if sha256(args.weights) != checkpoint_hash:
        raise ValueError('Checkpoint changed during the spatial audit')
    summary['checkpoint_sha256'] = checkpoint_hash
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=_paths.ROOT, text=True).strip()
    summary['source_commit'] = os.environ.get('LIGHTGEN_SOURCE_COMMIT', head)
    summary['checkout_HEAD'] = head
    summary['source_sha256'] = sha256(Path(__file__))
    args.out.mkdir(parents=True, exist_ok=False)
    write(args.out / 'metrics.json', summary)
    np.savez_compressed(args.out / 'source_to_detector.npz', **arrays)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
