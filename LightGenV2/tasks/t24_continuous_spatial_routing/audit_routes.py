"""Frozen validation-only interventions on router CCD counts and input placements."""
import argparse
import os
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import torch

from .compare import location_metrics
from .quantize_positions import load_validation, read_json
from .train import REPO, TASK, require_authorized_gpu, sha256, state_digest, write


def derangement(n, seed):
    """Uniformly sample a permutation conditional on having zero fixed points."""
    if n < 2:
        raise ValueError('Derangement requires at least two images')
    rng = np.random.default_rng(seed)
    identity = np.arange(n)
    while True:
        donor = rng.permutation(n)
        if (donor != identity).all():
            return donor


def unordered_distance(a, b):
    """Minimum mean Euclidean displacement over the two assignments, per image."""
    aligned = np.linalg.norm(a - b, axis=2).mean(axis=1)
    reversed_pair = np.linalg.norm(a - b[:, ::-1], axis=2).mean(axis=1)
    return np.minimum(aligned, reversed_pair), reversed_pair < aligned


def rank_matched_donors(positions, gates, donor):
    # Constrained addresses rank smoothed CCD readings, while actual gates
    # use raw readings, so native slot order can differ from the gate rank.
    strong_gate_slot = gates.argmax(axis=1)
    swap = strong_gate_slot[donor] != strong_gate_slot
    replacement = positions[donor].copy()
    replacement[swap] = replacement[swap, ::-1]
    return replacement, swap


def photon_field(field, photons, rng):
    """Independent Poisson counts per real CCD pixel; E[counts/photons] = |field|²."""
    intensity = field.abs().square()
    counts = torch.poisson(intensity * photons, generator=rng)
    # locate() consumes only abs().square() of this transient detector field.
    return (counts / photons).sqrt().to(field.dtype)


def stats(values):
    return {k: float(v) for k, v in zip(('mean', 'median', 'p90', 'p99'),
                                        [np.mean(values), *np.percentile(values, (50, 90, 99))])}


def compare_positions(a, b):
    distance, swapped = unordered_distance(a, b)
    return {'minimum_assignment_mean_distance_px': stats(distance),
            'fraction_over_1px': float((distance > 1).mean()),
            'fraction_over_16px': float((distance > 16).mean()),
            'closer_after_slot_swap_fraction': float(swapped.mean())}


def effective_pixels(p):
    return p.sum((-2, -1)).square() / p.square().sum((-2, -1)).clamp_min(1e-30)


def shifted_router_input(amplitude):
    """Shift each 112² tile one column right; replicate its own left edge."""
    # The horizontal tile pairs share the same two column maps at every row.
    idx = torch.arange(224, device=amplitude.device)
    idx = idx - (idx.remainder(112) > 0).long()
    shifted = amplitude.index_select(-1, idx)
    return shifted / shifted.flatten(1).norm(dim=1)[:, None, None].clamp_min(1e-12)


@torch.inference_mode()
def evaluate_intervention(model, fields, batch, device, *, photons=None, noise_seed=None, positions=None,
                          shift_router=False):
    """Use the trained forward end-to-end. Intervene exclusively inside locate()."""
    locate, multiplex = model.locate, model.multiplex
    face_sum = torch.zeros(478, 478, dtype=torch.float64, device=device)
    neff = []
    batches = {key: [] for key in ('scores', 'positions', 'power', 'router_capture')}
    rng = torch.Generator(device=device).manual_seed(noise_seed) if photons is not None else None
    if int(positions is not None) + int(photons is not None) + int(shift_router) > 1:
        raise ValueError('Use only one intervention at a time')

    def routed(amplitude):
        if shift_router:
            return locate(shifted_router_input(amplitude))
        if rng is not None:
            def router_noise(_, inputs, output):
                return photon_field(output, photons, rng)
            # Installed only while processing the router; the two main stages are noiseless.
            with model.propagator.register_forward_hook(router_noise):
                return locate(amplitude)
        coords, power, capture = locate(amplitude)
        if positions is not None:
            coords = torch.from_numpy(positions[start:start + len(amplitude)]).to(device)
        return coords, power, capture

    def measured(amplitude, coords, power):
        entrance = multiplex(amplitude, coords, power)
        irradiance = entrance.abs().square()
        face_sum.add_(irradiance.sum(dim=0))
        neff.append(effective_pixels(irradiance).cpu().numpy())
        return entrance

    with patch.object(model, 'locate', routed), patch.object(model, 'multiplex', measured):
        for start in range(0, len(fields), batch):
            output = model(fields[start:start + batch].to(device))
            batches['scores'].append(output['logits'].softmax(1).cpu().numpy())
            for key in ('positions', 'power', 'router_capture'):
                batches[key].append(output[key].cpu().numpy())
    arrays = {key: np.concatenate(parts) for key, parts in batches.items()}
    if any(not np.isfinite(v).all() for v in arrays.values()):
        raise ValueError('Nonfinite intervention output')
    total = face_sum.sum()
    if not torch.isclose(total, torch.tensor(len(fields), device=device, dtype=total.dtype), rtol=1e-5):
        raise ValueError('Incident power normalization mismatch')
    strip = face_sum[224:254].sum() + face_sum[:, 224:254].sum() - face_sum[224:254, 224:254].sum()
    usage = dict(geometric_placements=location_metrics(arrays['positions']),
                 mean_per_image_effective_illuminated_pixels=float(np.concatenate(neff).mean()),
                 population_effective_illuminated_pixels=float(effective_pixels(face_sum)),
                 mean_power_fraction_in_original_cross_gap=float(strip / total))
    return arrays, usage


def outcomes(variant, baseline, y):
    pred = variant['scores'].argmax(axis=1)
    clean = baseline['scores'].argmax(axis=1)
    return {'n': len(y), 'correct': int((pred == y).sum()), 'accuracy': float((pred == y).mean()),
            'changed_predictions': int((pred != clean).sum()),
            'gained_correct': int(((pred == y) & (clean != y)).sum()),
            'lost_correct': int(((pred != y) & (clean == y)).sum()),
            'paired_delta_pp': float(100 * ((pred == y).sum() - (clean == y).sum()) / len(y))}


def audit(run, out, device, photons):
    from .train import build_model
    manifest = read_json(run / 'manifest.json')
    for name in ('metadata.json', 'best_checkpoint.pt', 'predictions_val.npz', 'selection_lock.json'):
        if sha256(run / name) != manifest[name]:
            raise ValueError('Run hash mismatch: ' + name)
    meta = read_json(run / 'metadata.json')
    lock = read_json(run / 'selection_lock.json')
    config, mode = meta['config'], meta['mode']
    if read_json(run / 'status.json')['state'] != 'complete' or lock['mode'] != mode or lock[
            'best_checkpoint_sha256'] != manifest['best_checkpoint.pt']:
        raise ValueError('A completed, locked model is required')
    cuda_uuid = require_authorized_gpu(config['authorized_gpu_uuids']) if device.type == 'cuda' else None
    torch.set_num_threads(4)
    model = build_model(mode, config['num_classes']).eval()
    payload = torch.load(run / 'best_checkpoint.pt', map_location='cpu', weights_only=False)
    if payload['epoch'] != lock['selected_epoch'] or payload['config'] != config or payload['mode'] != mode:
        raise ValueError('Checkpoint/selection identity mismatch')
    model.load_state_dict(payload['model'], strict=True)
    model.to(device)
    param_hash = state_digest(model)
    relevant = ['model.py'] + (['constrained_routing.py'] if mode == 'constrained' else [])
    for name in relevant:
        key = (TASK / name).relative_to(REPO).as_posix()
        if sha256(TASK / name) != meta['source_sha256'][key]:
            raise ValueError('Trained model dependency differs: ' + name)
    fields, truth, ids = load_validation(config, device)
    truth = truth.numpy()
    batch = config['batch']
    baseline, usage = evaluate_intervention(model, fields, batch, device)
    with np.load(run / 'predictions_val.npz', allow_pickle=False) as cached:
        if not (np.array_equal(cached['ids'], ids) and np.array_equal(cached['labels'], truth)
                and np.array_equal(cached['predictions'], baseline['scores'].argmax(1))
                and np.allclose(cached['power'], baseline['power'], rtol=2e-4, atol=1e-5)
                and np.allclose(cached['positions'], baseline['positions'], atol=2e-3, rtol=0)):
            raise ValueError('Canonical validation replay mismatch (IDs, labels, predicted classes, gates or positions)')
        max_delta = float(np.max(np.abs(cached['scores'] - baseline['scores'])))
    result = {'run': run.name, 'dataset': config['dataset'], 'mode': mode, 'seed': config['seed'],
              'split': 'val', 'gpu_uuid': cuda_uuid, 'command': sys.argv, 'torch_version': str(torch.__version__),
              'best_checkpoint_sha256': manifest['best_checkpoint.pt'], 'selected_epoch': lock['selected_epoch'],
              'val_predictions_sha256': manifest['predictions_val.npz'], 'data_manifest_sha256': config['manifest_sha256'],
              'audit_sha256': sha256(Path(__file__)), 'trained_source_commit': meta['source_commit'],
              'canonical_max_score_difference': max_delta, 'optimization_steps': 0,
              'photons_per_unit_incident_energy': photons,
              'noise_model': 'Ideal full-router CCD independent Poisson per pixel, two independent captures, no other noise. Actual summed expected counts scale with captured CCD power.',
              'assignment': 'Minimum mean Euclidean distance over both permutations of the two positions.',
              'shuffle': 'Uniform validation derangement. Donor high-power position goes to recipient high-power slot. Powers/amplitude remain those of the recipient. Ties favor slot zero.',
              'eff_pixels': 'Inverse participation ratio of normalized 478x478 entrance intensity; reported per image and after accumulation over images.',
              'clean': dict(outcomes(baseline, baseline, truth), usage=usage), 'shot_noise': [], 'shuffled': []}
    np.savez_compressed(out / 'canonical.npz', ids=ids, labels=truth, **baseline)
    captures = []
    for noise_seed in (1069, 1999):
        noisy, noisy_usage = evaluate_intervention(model, fields, batch, device, photons=photons, noise_seed=noise_seed)
        result['shot_noise'].append(dict(outcomes(noisy, baseline, truth), seed=noise_seed, usage=noisy_usage,
                                          vs_clean=compare_positions(noisy['positions'], baseline['positions'])))
        captures.append(noisy['positions'])
        np.savez_compressed(out / f'noise_{noise_seed}.npz', **noisy)
    result['independent_same_image_noise_pair'] = compare_positions(*captures)
    shifted, shifted_usage = evaluate_intervention(model, fields, batch, device, shift_router=True)
    result['router_tile_shift'] = dict(outcomes(shifted, baseline, truth),
            intervention='Shift four 112x112 input tiles right one column independently with left-edge replication. Renormalize router input to unit power. Main image is unchanged.',
            vs_clean=compare_positions(shifted['positions'], baseline['positions']), usage=shifted_usage)
    np.savez_compressed(out / 'router_tile_shift.npz', **shifted)
    targets = np.rint(baseline['positions'])
    rounded, rounded_usage = evaluate_intervention(model, fields, batch, device, positions=targets)
    if not (np.array_equal(rounded['positions'], targets)
            and np.array_equal(rounded['power'], baseline['power'])
            and np.array_equal(rounded['router_capture'], baseline['router_capture'])):
        raise ValueError('Integer actuator replay changed a gate/capture or misplaced an input')
    result['integer_actuator'] = dict(outcomes(rounded, baseline, truth),
            intervention='Round both coordinates of each original landing position to the nearest integer pixel (ties to even). Original gate powers and main image are retained.',
            vs_clean=compare_positions(rounded['positions'], baseline['positions']), usage=rounded_usage)
    np.savez_compressed(out / 'integer_actuator.npz', **rounded)
    for permutation_seed in (3701, 4423, 5701):
        donor = derangement(len(ids), permutation_seed)
        donor_positions, swapped = rank_matched_donors(baseline['positions'], baseline['power'], donor)
        shuffled, shuffled_usage = evaluate_intervention(model, fields, batch, device, positions=donor_positions)
        if not (np.array_equal(shuffled['positions'], donor_positions)
                and np.array_equal(shuffled['power'], baseline['power'])
                and np.array_equal(shuffled['router_capture'], baseline['router_capture'])):
            raise ValueError('Shuffled position/gate identity mismatch')
        result['shuffled'].append(dict(outcomes(shuffled, baseline, truth), permutation_seed=permutation_seed,
                                       donor_native_order_swaps=int(swapped.sum()),
                                       different_image_distance=compare_positions(donor_positions, baseline['positions']),
                                       usage=shuffled_usage))
        np.savez_compressed(out / f'shuffle_{permutation_seed}.npz', donor_rows=donor, swapped=swapped, **shuffled)
    if state_digest(model) != param_hash:
        raise RuntimeError('Evaluation changed a parameter')
    full_passes = 3 + len(result['shot_noise']) + len(result['shuffled'])
    result['budget'] = dict(full_VAL_classification_passes=full_passes,
                            classified_image_presentations=full_passes * len(ids))
    write(out / 'result.json', result)
    write(out / 'manifest.json', {p.name: sha256(p) for p in sorted(out.iterdir()) if p.is_file() and p.name != 'status.json'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--photons', type=float, default=1e6)
    args = parser.parse_args()
    run, out = args.run.resolve(), args.out.resolve()
    if not run.is_relative_to(TASK / 'runs' / 'simulation') or not out.is_relative_to(TASK / 'runs' / 'smoke'):
        parser.error('Expected a T24 simulation run and fresh T24/runs/smoke output directory')
    if not np.isfinite(args.photons) or args.photons <= 0:
        parser.error('Photons must be finite and positive')
    out.mkdir(parents=True, exist_ok=False)
    pid = os.getpid()
    write(out / 'status.json', {'state': 'running', 'pid': pid})
    try:
        audit(run, out, torch.device(args.device), args.photons)
    except BaseException as exc:
        write(out / 'status.json', {'state': 'failed', 'error': str(exc), 'pid': pid})
        raise
    write(out / 'status.json', {'state': 'complete', 'pid': pid})


if __name__ == '__main__':
    main()
