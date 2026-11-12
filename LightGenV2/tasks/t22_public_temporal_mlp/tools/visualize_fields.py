"""Trace the selected linear model and plot its two simulated global fields."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle
import numpy as np
import torch
from torch.nn import functional as F

if __package__:
    from . import _paths
else:
    import _paths

if __package__ and __package__ != 'tools':
    from .plot_style import save_figure
    from ..infer import frames_from_video, iter_video_fields
    from ..model import load_lightgen
    from .phase_plotting import phase_display_norm,phase_display_values,add_phase_colorbar,PHASE_DISPLAY
else:
    from plot_style import save_figure
    from infer import frames_from_video, iter_video_fields
    from model import load_lightgen
    from phase_plotting import phase_display_norm,phase_display_values,add_phase_colorbar,PHASE_DISPLAY

ROOT = _paths.ROOT


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for piece in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(piece)
    return digest.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n', 'utf-8')


def input_canvas(stage, fields):
    canvas = np.zeros((518, 518), dtype=np.float32)
    for slot, (y, x) in enumerate(stage.origins):
        if stage.frame_level:
            for frame, (dy, dx) in enumerate([(0, 0), (0, 59), (59, 0), (59, 59)]):
                canvas[y+dy+14:y+dy+41, x+dx+14:x+dx+41] = fields[slot, frame]
        else:
            canvas[y+29:y+85, x+29:x+85] = fields[slot]
    return canvas


def detector_regions(stage, intensity):
    patches = []
    offsets = [(0, 0), (0, 59), (59, 0), (59, 59)] if stage.frame_level else [(0, 0)]
    side = 56 if stage.frame_level else 115
    for y, x in stage.origins:
        pieces = []
        for dy, dx in offsets:
            patch = intensity[y+dy:y+dy+side, x+dx:x+dx+side]
            pieces.append(patch)
        patches.append(pieces)
    patches = np.asarray(patches, dtype=np.float32)
    if not stage.frame_level:
        patches = patches[:, 0]
    processed = stage.process_detector(torch.from_numpy(patches).clamp_min(0)).numpy()
    mosaic = np.full((518, 518), np.nan, dtype=np.float32)
    for slot, (y, x) in enumerate(stage.origins):
        for frame, (dy, dx) in enumerate(offsets):
            value = processed[slot, frame] if stage.frame_level else processed[slot]
            mosaic[y+dy:y+dy+side, x+dx:x+dx+side] = value
    return patches, mosaic


def trace(model, vision):
    captured = {'frame': {}, 'video': {}}
    handles = []

    def before_global(name):
        def capture(module, inputs):
            captured[name]['tokens'] = inputs[0].detach().clone()
        return capture

    def before_propagation(name):
        def capture(module, inputs):
            captured[name]['modulated_field'] = inputs[0][0].detach().cpu().numpy().copy()
        return capture

    def after_propagation(name):
        def capture(module, inputs, output):
            captured[name]['propagated_field'] = output[0].detach().cpu().numpy().copy()
            captured[name]['intensity'] = output.abs().square()[0].float().cpu().numpy().copy()
        return capture

    def before_ccd_norm(name):
        def capture(module, inputs):
            captured[name]['pooled'] = inputs[0].detach().cpu().clone()
        return capture

    def before_experts(name):
        def capture(module, inputs):
            captured[name]['expert_tokens'] = inputs[0].detach().clone()
            captured[name]['expert_weights'] = inputs[1].detach().clone()
        return capture

    def after_experts(name):
        def capture(module, inputs, output):
            captured[name]['expert_output'] = output.detach().clone()
        return capture

    for name in captured:
        stage = getattr(model, name + '_global')
        experts = getattr(model, name + '_experts')
        handles.extend([
            stage.register_forward_pre_hook(before_global(name)),
            stage.propagation.register_forward_pre_hook(before_propagation(name)),
            stage.propagation.register_forward_hook(after_propagation(name)),
            stage.norm.register_forward_pre_hook(before_ccd_norm(name)),
            experts.register_forward_pre_hook(before_experts(name)),
            experts.register_forward_hook(after_experts(name))])
    try:
        with torch.inference_mode():
            result = model(vision)
    finally:
        for handle in handles:
            handle.remove()
    audits = {}
    for name, values in captured.items():
        stage = getattr(model, name + '_global')
        experts = getattr(model, name + '_experts')
        with torch.inference_mode():
            fields = stage.fields(values['tokens'])[0].cpu().numpy()
            tile_modulation = stage.modulation().cpu().numpy()
            phases = stage.phase(wrapped=True).cpu().numpy()
            operators = torch.stack([expert.weight for expert in experts.experts])
            effective = torch.einsum('...k,koi->...oi', values['expert_weights'], operators)
            equivalent = torch.einsum('...ti,...oi->...to', values['expert_tokens'], effective)
            expert_error = float((equivalent - values['expert_output']).abs().max())
            if not torch.allclose(equivalent, values['expert_output'], rtol=2e-5, atol=1e-5):
                raise ValueError('Linear weighted-operator identity failed for ' + name)
        incident = input_canvas(stage, fields)
        mask = np.ones((518, 518), dtype=np.complex64)
        phase_map = np.full((518, 518), np.nan, dtype=np.float32)
        for slot, (y, x) in enumerate(stage.origins):
            mask[y+2:y+113, x+2:x+113] = tile_modulation[slot]
            phase_map[y+2:y+113, x+2:x+113] = phases[slot]
        field_error = float(np.max(np.abs(incident * mask - values['modulated_field'])))
        if field_error > 2e-5:
            raise ValueError('Reconstructed incident field differs from actual propagation input')
        patches, processed_mosaic = detector_regions(stage, values['intensity'])
        processed = stage.process_detector(torch.from_numpy(patches).clamp_min(0))
        token_count = 49 if stage.frame_level else 4+stage.cfg.prompt_length
        pooled = F.adaptive_avg_pool2d(processed.reshape(-1, 1, *patches.shape[-2:]), (token_count, 96))[:, 0]
        pooling_error = float((pooled - values['pooled']).abs().max())
        if pooling_error > 5e-5:
            raise ValueError('CCD preprocessing differs from captured forward values')
        values.update(incident_amplitude=incident, intended_phase=phase_map,
                      effective_modulation=mask, encoded_fields=fields,
                      detector_patches=patches, processed_mosaic=processed_mosaic)
        audits[name] = {'linear_expert_weighted_operator_max_error': expert_error,
                        'incident_times_ORM_to_actual_V_max_error': field_error,
                        'reconstructed_CCD_pool_to_actual_pool_max_error': pooling_error}
    return result, captured, audits


def plot_overview(stage, values, out):
    intensity = values['intensity']
    upper = max(float(np.percentile(intensity, 99.8)), 1e-8)
    lower = max(float(intensity.max()) * 1e-6, 1e-12)
    if lower >= upper:
        upper = lower * 10
    modulated = values['modulated_field']
    propagated = values['propagated_field']
    input_phase = np.where(values['incident_amplitude'] > 0, np.angle(modulated), np.nan)
    output_phase = np.where(intensity > intensity.max() * 1e-6, np.angle(propagated), np.nan)
    legacy_processing = stage.cfg.detector_processing == 'legacy_log_clip'
    processed_title = 'CCD mean / clip / log1p' if legacy_processing else 'CCD mean-normalized intensity'
    processed_upper = (np.log(9) if legacy_processing else
                       max(float(np.nanmax(values['processed_mosaic'])), 1e-8))
    panels = [
        ('Incident amplitude |U|', values['incident_amplitude'], 'magma', {}),
        ('Intended SLM phase', phase_display_values(values['intended_phase']), 'twilight', {'norm': phase_display_norm()}),
        ('Post-ORM amplitude |V|', np.abs(modulated), 'magma', {}),
        ('Post-ORM phase arg(V)', input_phase, 'twilight', {'vmin': -np.pi, 'vmax': np.pi}),
        ('Detector intensity |O|², linear', intensity, 'inferno', {'vmin': 0, 'vmax': upper}),
        ('Detector intensity |O|², log', intensity, 'inferno', {'norm': LogNorm(lower, upper)}),
        ('Propagated phase arg(O)', output_phase, 'twilight', {'vmin': -np.pi, 'vmax': np.pi}),
        (processed_title, values['processed_mosaic'], 'viridis', {'vmin': 0, 'vmax': processed_upper})]
    fig, axes = plt.subplots(2, 4, figsize=(8.4, 4.7), constrained_layout=True)
    for ax, (title, array, cmap, limits) in zip(axes.flat, panels):
        artist = ax.imshow(array, cmap=cmap, interpolation='nearest', **limits)
        ax.add_patch(Rectangle((19.5, 19.5), 478, 478, fill=False, edgecolor='#46cbd5', lw=.8))
        ax.set(title=title, xlabel='x (pixel)', ylabel='y (pixel)')
        if title.startswith('Intended SLM phase'):
            add_phase_colorbar(fig,artist,ax)
        else:
            fig.colorbar(artist, ax=ax, shrink=.6, aspect=14, pad=.02)
    label = 'Vision global, 16 videos × 4 frames' if stage.frame_level else 'Vision-language global, 16 video fields'
    fig.suptitle(label + '\n518 × 518 canvas, 478 × 478 active region')
    save_figure(fig,out)
    plt.close(fig)
    return {'intensity_display_upper_percentile': 99.8, 'intensity_display_upper': upper,
            'log_intensity_display_lower': lower, 'phase_visibility_threshold_fraction': 1e-6,
            'detector_processing': stage.cfg.detector_processing,
            'processed_intensity_display_upper': processed_upper,
            'SLM_phase_display':PHASE_DISPLAY}


def plot_inputs(pixels, ids, out):
    fig, axes = plt.subplots(4, 4, figsize=(7.2, 7.2), constrained_layout=True)
    images = pixels.permute(0, 1, 3, 4, 2).numpy()
    for slot, ax in enumerate(axes.flat):
        mosaic = np.concatenate((np.concatenate(images[slot, :2], axis=1),
                                 np.concatenate(images[slot, 2:], axis=1)), axis=0)
        ax.imshow(mosaic)
        ax.set_title(f'Slot {slot:02d}  |  {ids[slot][:12]}', fontsize=8)
        ax.axis('off')
    fig.suptitle('Actual cropped video frames; each slot shows t1 t2 / t3 t4', fontsize=8)
    save_figure(fig,out)
    plt.close(fig)


def plot_detail(pixels, values, result, slot, sample_id, out):
    fig, axes = plt.subplots(3, 4, figsize=(8.4, 6.1), constrained_layout=True)
    fractions = np.linspace(.1, .9, 4)
    raw = values['frame']['detector_patches'][slot]
    high = max(float(np.percentile(raw, 99.8)), 1e-8)
    low = max(float(raw.max()) * 1e-6, 1e-12)
    high = max(high, low * 10)
    for frame in range(4):
        axes[0, frame].imshow(pixels[slot, frame].permute(1, 2, 0).numpy())
        axes[0, frame].set_title(f'Frame {frame+1}, time fraction {fractions[frame]:.4f}')
        axes[0, frame].axis('off')
        artist = axes[1, frame].imshow(raw[frame], cmap='inferno', norm=LogNorm(low, high))
        axes[1, frame].set_title(f'Frame-global CCD region {frame+1}')
        fig.colorbar(artist, ax=axes[1, frame], shrink=.7)
    routes = result['routing']['frame']['probabilities'][0, slot].cpu().numpy()
    artist = axes[2, 0].imshow(routes, vmin=0, vmax=1, cmap='Blues')
    axes[2, 0].set(title='Vision softmax probabilities', xlabel='Expert', ylabel='Frame',
                   xticks=range(4), yticks=range(4))
    fig.colorbar(artist, ax=axes[2, 0], shrink=.7)
    video_raw = values['video']['detector_patches'][slot]
    artist = axes[2, 1].imshow(video_raw, cmap='inferno')
    axes[2, 1].set_title('Video-global CCD region, linear')
    fig.colorbar(artist, ax=axes[2, 1], shrink=.7)
    weights = result['routing']['video']['probabilities'][0, slot].cpu().numpy()
    axes[2, 2].bar(range(4), weights, color='#3174a4')
    axes[2, 2].set(title='Fusion softmax probabilities', xlabel='Expert', ylim=(0, 1))
    probabilities = result['quality_probabilities'][0, slot].cpu().numpy()
    axes[2, 3].bar(range(5), probabilities, color='#3174a4')
    axes[2, 3].set(title=f'MOS = {float(result["prediction"][0, slot]):.4f}',
                   xticks=range(5), xticklabels=['Bad', 'Poor', 'Fair', 'Good', 'Excellent'], ylim=(0, 1))
    axes[2, 3].tick_params(axis='x', rotation=30)
    fig.suptitle(f'Slot {slot:02d} | {sample_id}\nSelected linear model, traced optical fields', fontsize=8)
    save_figure(fig,out)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', type=Path, default=ROOT/'configs/selected.json')
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--paths', type=Path, help='Internal cached-feature dataset configuration')
    inputs.add_argument('--videos', type=Path, nargs='+', help='Native video input, standalone inference')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--group', type=int, default=0)
    parser.add_argument('--detail-slot', type=int, default=0)
    parser.add_argument('--skip-input-preview', action='store_true',
                        help='Export cached fields when the original video previews are unavailable')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    profile = json.loads(args.profile.read_text('utf-8'))
    weights = ROOT / profile['checkpoint']
    if sha256(weights) != profile['checkpoint_sha256']:
        raise ValueError('Selected checkpoint SHA differs')
    torch.set_num_threads(4)
    model, metadata = load_lightgen(weights, args.device)
    if model.cfg.expert_mode != 'linear' or model.cfg.orm_mode != 'amplitude':
        raise ValueError('Selected linear architecture required')
    if args.group < 0 or not 0 <= args.detail_slot < 16:
        raise ValueError('Invalid evaluation group or detail slot')
    if args.paths:
        if __package__ and __package__ != 'tools':
            from .data import batch, groups, load_inputs
        else:
            from data import batch, groups, load_inputs
        payload = load_inputs(json.loads(args.paths.read_text('utf-8')))
        if payload['identity']['manifest']['sha256'] != metadata['dataset']['manifest']['sha256']:
            raise ValueError('Selected dataset manifest differs')
        indices, valid = groups(payload, 'test')
        if args.group >= len(indices):
            raise ValueError('Evaluation group exceeds dataset')
        index = indices[args.group:args.group+1]
        item = batch(payload, index, valid[args.group:args.group+1], args.device)
        vision = item['vision']
        keep = valid[args.group].tolist()
        ids = [payload['ids'][i] for i in index[0]]
        sources = [payload['video_paths'][i] for i in index[0]]
        targets = [float(payload['target'][i]) for i in index[0]]
        pixels = (None if args.skip_input_preview else
                  torch.stack([frames_from_video(path) for path in sources]))
        data_identity = payload['identity']
        feature_source = 'original_TEST_cached_features'
    else:
        if args.group >= (len(args.videos)+15)//16:
            raise ValueError('Video group exceeds supplied input')
        iterator = iter_video_fields(args.videos)
        for _ in range(args.group+1):
            chunk, packed_pixels = next(iterator)
        pixels = packed_pixels[0]
        prefix = args.videos[:15]
        padded_sources = list(chunk) + [prefix[i % len(prefix)] for i in range(16-len(chunk))]
        sources = [str(path) for path in padded_sources]
        ids = [Path(path).stem for path in sources]
        keep = [i < len(chunk) for i in range(16)]
        targets = [None]*16
        data_identity = {'videos': {str(path): sha256(path) for path in set(padded_sources)}}
        with torch.inference_mode():
            vision = model.front(packed_pixels.to(args.device))
        feature_source = 'native_video_pixels'
    with torch.inference_mode():
        baseline = model(vision)['prediction'].clone()
        result, captured, audits = trace(model, vision)
    if not torch.equal(baseline, result['prediction']):
        raise ValueError('Trace hooks changed model predictions')
    args.out.mkdir(parents=True)
    display = {}
    for name, values in captured.items():
        stage = getattr(model, name + '_global')
        arrays = {key: value for key, value in values.items() if isinstance(value, np.ndarray)}
        np.savez_compressed(args.out/(name + '_global_arrays.npz'), **arrays)
        display[name] = plot_overview(stage, values, args.out/(name + '_global.png'))
    if not args.skip_input_preview:
        plot_inputs(pixels, ids, args.out/'input_frames.png')
        plot_detail(pixels, captured, result, args.detail_slot, ids[args.detail_slot], args.out/'slot_detail.png')
    records = []
    for slot in range(16):
        records.append({'slot': slot, 'sample_id': ids[slot],
                        'valid': keep[slot], 'target': targets[slot],
                        'prediction': float(result['prediction'][0, slot]),
                        'video_path': sources[slot]})
    with (args.out/'group_predictions.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    write(args.out/'routing.json', {name: result['routing'][name]['weights'][0].cpu().tolist()
                                  for name in ('frame', 'video')})
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = 'standalone_directory'
    record = {'role': 'SELECTED_LINEAR_SIMULATED_OPTICAL_FIELDS', 'group_index': args.group,
              'weights_sha256': sha256(weights), 'profile': profile, 'inputs': records,
              'data_identity': data_identity, 'feature_source': feature_source, 'audits': audits,
              'input_preview_exported': not args.skip_input_preview,
              'trace_prediction_bitwise_equal_to_unhooked_forward': True,
              'display': display, 'intensity_unit': 'arbitrary, after per-field input RMS normalization',
              'phase_unit': 'radians', 'array_shape': [518, 518], 'active_bounds_yx': [20, 498, 20, 498],
              'readout_region_origins_yx': model.frame_global.origins,
              'source_commit': os.environ.get('LIGHTGEN_SOURCE_COMMIT', commit),
              'source_SHA256': {p.name: sha256(p) for p in (ROOT/'model.py', ROOT/'infer.py', Path(__file__))},
              'environment': {'python': platform.python_version(), 'torch': str(torch.__version__),
                              'matplotlib': matplotlib.__version__, 'device': args.device},
              'files': {p.name: sha256(p) for p in args.out.iterdir() if p.is_file()}}
    write(args.out/'visualization_manifest.json', record)
    print(json.dumps({'out': str(args.out), 'audits': audits}), flush=True)


if __name__ == '__main__':
    main()
