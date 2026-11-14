"""Result plotting and field capture used by the complete infer.py evaluation."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
import numpy as np
import torch

BRANCHES = ('vision', 'language')
PREDICTION_COLOR = '#6486a3'


def configure_figures():
    """Use Arial 7 pt, with 8 pt titles, for every exported figure."""
    font_manager.findfont(font_manager.FontProperties(family='Arial'), fallback_to_default=False)
    plt.rcParams.update({
        'font.family': 'Arial', 'font.size': 7, 'axes.labelsize': 7,
        'axes.titlesize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
        'legend.fontsize': 7, 'figure.titlesize': 8, 'axes.linewidth': .6,
    })


def save_figure(figure, path):
    figure.savefig(path, dpi=300, bbox_inches='tight', pad_inches=.05)
    plt.close(figure)


def normalize_display(array, percentile=100):
    """Set an explicitly recorded upper limit for a native-pixel linear view."""
    scale = np.float32(max(float(np.percentile(array, percentile)), 1e-12))
    return np.clip(array / scale, 0, 1).astype(np.float32), scale


def plot_field(array, path, title, *, phase=False, quantity='Intensity'):
    figure, axis = plt.subplots(figsize=(3.2, 3.0), layout='constrained')
    if phase:
        image = axis.imshow(array, cmap='twilight_shifted', vmin=-np.pi, vmax=np.pi,
                            interpolation='nearest', resample=False)
        colorbar = figure.colorbar(image, ax=axis, shrink=.68, pad=.035, fraction=.045)
        colorbar.set_ticks([-np.pi, 0, np.pi], labels=['−π', '0', 'π'])
        colorbar.set_label('Phase (rad)')
    else:
        image = axis.imshow(array, cmap='inferno', norm=Normalize(0, 1),
                            interpolation='nearest', resample=False,
                            extent=(-.5, 517.5, 517.5, -.5))
        colorbar = figure.colorbar(image, ax=axis, shrink=.68, pad=.035, fraction=.045)
        colorbar.set_label('Normalized ' + quantity.lower())
    axis.set_title(title)
    axis.add_patch(Rectangle((19.5, 19.5), 478, 478, fill=False, edgecolor='#46cbd5', linewidth=.6))
    axis.set_xlabel('x (pixel)')
    axis.set_ylabel('y (pixel)')
    axis.set_xticks([0, 259, 517])
    axis.set_yticks([0, 259, 517])
    save_figure(figure, path)


def capture_optical_fields(model, normalized_rgb):
    """Observe the actual forward without changing its tensors or operators."""
    captured = {branch: {} for branch in BRANCHES}
    handles = []

    def capture_tokens(branch):
        def hook(module, arguments):
            captured[branch]['tokens'] = arguments[0].detach()
        return hook

    def capture_propagation(branch):
        def hook(module, arguments, output):
            captured[branch]['modulated'] = arguments[0].detach()
            captured[branch]['propagated'] = output.detach()
        return hook

    for branch in BRANCHES:
        stage = getattr(model, branch + '_global')
        handles.append(stage.register_forward_pre_hook(capture_tokens(branch)))
        handles.append(stage.propagation.register_forward_hook(capture_propagation(branch)))
    try:
        output = model(normalized_rgb=normalized_rgb)
    finally:
        for handle in handles:
            handle.remove()
    return output, captured


def plot_input_frames(normalized_rgb, output):
    images = ((normalized_rgb[0].float() + 1) * 127.5).round().clamp(0, 255).to(torch.uint8)
    images = images.permute(0, 1, 3, 4, 2).cpu().numpy()
    figure, axes = plt.subplots(4, 4, figsize=(6.6, 7.0), layout='constrained')
    for slot, axis in enumerate(axes.flat):
        frames = images[slot, :, ::4, ::4]
        mosaic = np.concatenate((np.concatenate(frames[:2], 1),
                                 np.concatenate(frames[2:], 1)), 0)
        axis.imshow(mosaic)
        axis.set_title(f'Video {slot + 1}')
        axis.set_axis_off()
    figure.suptitle('Input video frames')
    save_figure(figure, output)


def write_records(path, records):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def export_evaluation(records, metrics, directory):
    """Save every actual video prediction and the quality comparison figure."""
    configure_figures()
    directory.mkdir(parents=True, exist_ok=True)
    write_records(directory / '01_predictions.csv', records)
    target = np.asarray([record['target_MOS'] for record in records])
    prediction = np.asarray([record['predicted_MOS'] for record in records])
    figure, axis = plt.subplots(figsize=(3.3, 3.1), layout='constrained')
    axis.scatter(target, prediction, s=9, alpha=.6, color=PREDICTION_COLOR, linewidths=0)
    lower, upper = min(target.min(), prediction.min()), max(target.max(), prediction.max())
    axis.plot([lower, upper], [lower, upper], color='#888888', linewidth=.7)
    axis.set_xlabel('Human temporal MOS')
    axis.set_ylabel('Predicted temporal MOS')
    score = f"{metrics['SRCC']:.6f}" if metrics['SRCC'] is not None else 'undefined'
    axis.set_title(f"{len(records)} videos with SRCC {score}")
    axis.spines[['top', 'right']].set_visible(False)
    save_figure(figure, directory / '03_prediction_vs_mos.png')


@torch.inference_mode()
def export_visualizations(model, records, normalized_rgb, captured, directory, *, field_number):
    """Export the field already captured during the complete evaluation."""
    if len(records) != 16 or tuple(normalized_rgb.shape[:3]) != (1, 16, 4):
        raise ValueError('Visualizations require exactly 16 videos in one optical field')
    configure_figures()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    names = np.asarray([record['sample_id'] for record in records])
    valid = np.asarray([not record['is_padding'] for record in records])
    for branch in BRANCHES:
        stage = getattr(model, branch + '_global')
        record = captured[branch]
        incident = stage.input_field(record['tokens']).to(torch.complex64)
        modulation = stage.modulation_field(incident)
        if not torch.equal(incident * modulation, record['modulated']):
            raise ValueError('Captured modulation differs from the forward in ' + branch)
        phase_tiles = torch.angle(torch.exp(1j * stage.phase())).cpu().numpy()
        phase_map = np.zeros((518, 518), dtype=np.float32)
        for slot, (row, column) in enumerate(stage.origins):
            phase_map[row + 2:row + 113, column + 2:column + 113] = phase_tiles[slot]
        first_number = 5 if branch == 'vision' else 8
        input_prefix = directory / f'{first_number:02d}_{branch}_input_field'
        mask_prefix = directory / f'{first_number + 1:02d}_{branch}_mask'
        output_prefix = directory / f'{first_number + 2:02d}_{branch}_output_field'
        np.savez_compressed(str(mask_prefix) + '.npz',
                            phase=phase_map, phase_tiles=phase_tiles,
                            modulation=modulation[0].cpu().numpy(),
                            slot_origins=np.asarray(stage.origins), video_names=names,
                            valid=valid, field_number=field_number)
        plot_field(phase_map, str(mask_prefix) + '.png', branch.capitalize() + ' phase mask', phase=True)
        input_field = incident[0].cpu().numpy()
        input_amplitude = np.abs(input_field)
        input_intensity = input_amplitude ** 2
        display_amplitude, input_scale = normalize_display(input_amplitude)
        np.savez_compressed(str(input_prefix) + '.npz',
                            field=input_field, amplitude=input_amplitude, intensity=input_intensity,
                            display_amplitude=display_amplitude, display_scale=input_scale,
                            modulated_field=record['modulated'][0].cpu().numpy(), video_names=names,
                            valid=valid, field_number=field_number)
        plot_field(display_amplitude, str(input_prefix) + '.png',
                   branch.capitalize() + ' input field', quantity='Amplitude')
        output_field = record['propagated'][0].cpu().numpy()
        output_intensity = record['propagated'].abs().square().float()[0].cpu().numpy()
        display_intensity, output_scale = normalize_display(output_intensity, percentile=99.8)
        np.savez_compressed(str(output_prefix) + '.npz',
                            field=output_field, intensity=output_intensity,
                            display_intensity=display_intensity, display_scale=output_scale,
                            display_percentile=99.8,
                            transfer=stage.propagation.transfer.cpu().numpy(), video_names=names,
                            valid=valid, field_number=field_number)
        plot_field(display_intensity, str(output_prefix) + '.png',
                   branch.capitalize() + ' output field')
    plot_input_frames(normalized_rgb, directory / '04_input_frames.png')
