"""Export complete vision and language masks and optical fields as NPZ and PNG."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import Normalize
import numpy as np
import torch

if __package__:
    from .infer import PROJECT, iter_video_fields, save_predictions, videos_in_directory
    from .model import load_lightgen2
else:
    from infer import PROJECT, iter_video_fields, save_predictions, videos_in_directory
    from model import load_lightgen2


BRANCHES = ('vision', 'language')
EXPERT_COLORS = ('#6486a3', '#d4a478', '#8eaa93', '#b495b5')


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


def plot_field(array, path, title, *, phase=False, quantity='Intensity'):
    figure, axis = plt.subplots(figsize=(3.2, 3.0), layout='constrained')
    if phase:
        image = axis.imshow(array, cmap='twilight_shifted', vmin=-np.pi, vmax=np.pi)
        colorbar = figure.colorbar(image, ax=axis, shrink=.68, pad=.035, fraction=.045)
        colorbar.set_ticks([-np.pi, 0, np.pi], labels=['−π', '0', 'π'])
        colorbar.set_label('Phase (rad)')
    else:
        norm = Normalize(vmin=0, vmax=max(float(array.max()), 1e-12))
        image = axis.imshow(array, cmap='inferno', norm=norm)
        colorbar = figure.colorbar(image, ax=axis, shrink=.68, pad=.035, fraction=.045)
        colorbar.set_label(quantity + ' (a.u.)')
    axis.set_title(title)
    axis.set_xlabel('x (pixel)')
    axis.set_ylabel('y (pixel)')
    axis.set_xticks([0, 259, 517])
    axis.set_yticks([0, 259, 517])
    save_figure(figure, path)


def capture_optical_fields(model, pixels):
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
        output = model(pixels=pixels)
    finally:
        for handle in handles:
            handle.remove()
    return output, captured


def plot_input_frames(pixels, output):
    images = pixels[0].permute(0, 1, 3, 4, 2).cpu().numpy()
    figure, axes = plt.subplots(4, 4, figsize=(6.6, 7.0), layout='constrained')
    for slot, axis in enumerate(axes.flat):
        frames = images[slot, :, ::4, ::4]
        mosaic = np.concatenate((np.concatenate(frames[:2], 1),
                                 np.concatenate(frames[2:], 1)), 0)
        axis.imshow(mosaic)
        axis.set_title(f'video_{slot + 1:02d} · slot {slot}')
        axis.set_axis_off()
    figure.suptitle('Sixteen input videos with four sampled frames each')
    save_figure(figure, output)


def export_routing(result, directory):
    arrays = {}
    figure, axes = plt.subplots(1, 2, figsize=(5.0, 2.1), layout='constrained')
    for branch, axis in zip(BRANCHES, axes):
        routing = result['routing'][branch]
        selected = routing['selected'].cpu().numpy()[0]
        probabilities = routing['probabilities'].cpu().numpy()[0]
        counts = selected.reshape(-1, 4).sum(0)
        share = 100 * counts / counts.sum()
        arrays[branch + '_selected'] = selected
        arrays[branch + '_probabilities'] = probabilities
        arrays[branch + '_selection_count'] = counts
        arrays[branch + '_share_percent'] = share
        axis.bar(range(4), share, color=EXPERT_COLORS, width=.65)
        axis.set_xticks(range(4), ['E1', 'E2', 'E3', 'E4'])
        axis.set_ylim(0, 60)
        axis.set_ylabel('Optical expert share (%)')
        axis.set_title(branch.capitalize() + ' branch')
        axis.spines[['top', 'right']].set_visible(False)
        for index, value in enumerate(share):
            axis.text(index, value + 1, f'{value:.1f}', ha='center')
    np.savez_compressed(directory / 'optical_expert_share.npz', **arrays)
    save_figure(figure, directory / 'optical_expert_share.png')


@torch.inference_mode()
def export_visualizations(model, video_paths, pixels, directory):
    """Export one complete 16-video field and its per-video predictions."""
    if len(video_paths) != 16 or tuple(pixels.shape[:3]) != (1, 16, 4):
        raise ValueError('Visualizations require exactly 16 videos in one optical field')
    configure_figures()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    result, captured = capture_optical_fields(model, pixels)
    names = np.asarray([Path(path).name for path in video_paths])
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
        prefix = directory / branch
        np.savez_compressed(str(prefix) + '_mask.npz',
                            phase=phase_map, phase_tiles=phase_tiles,
                            modulation=modulation[0].cpu().numpy(),
                            slot_origins=np.asarray(stage.origins), video_names=names)
        plot_field(phase_map, str(prefix) + '_mask.png', branch.capitalize() + ' phase mask', phase=True)
        input_field = incident[0].cpu().numpy()
        input_amplitude = np.abs(input_field)
        input_intensity = input_amplitude ** 2
        np.savez_compressed(str(prefix) + '_input_field.npz',
                            field=input_field, amplitude=input_amplitude, intensity=input_intensity,
                            modulated_field=record['modulated'][0].cpu().numpy(), video_names=names)
        plot_field(input_amplitude, str(prefix) + '_input_field.png',
                   branch.capitalize() + ' input field', quantity='Amplitude')
        output_field = record['propagated'][0].cpu().numpy()
        output_intensity = record['propagated'].abs().square().float()[0].cpu().numpy()
        np.savez_compressed(str(prefix) + '_output_field.npz',
                            field=output_field, intensity=output_intensity,
                            transfer=stage.propagation.transfer.cpu().numpy(), video_names=names)
        plot_field(output_intensity, str(prefix) + '_output_field.png', branch.capitalize() + ' output field')
    plot_input_frames(pixels, directory / 'input_frames.png')
    export_routing(result, directory)
    predictions = result['prediction'][0].cpu().tolist()
    records = [{'video': name, 'field': 0, 'slot': slot, 'MOS': prediction}
               for slot, (name, prediction) in enumerate(zip(names.tolist(), predictions))]
    save_predictions(directory / 'predictions.json', records)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, default=PROJECT / 'weights/LightGen2.pt')
    parser.add_argument('--video-dir', type=Path, default=PROJECT / 'demo_videos')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', type=Path, default=PROJECT / 'outputs/visualizations')
    args = parser.parse_args()
    paths = videos_in_directory(args.video_dir)
    if len(paths) != 16:
        raise ValueError('Supply a directory containing exactly 16 videos')
    _, pixels = next(iter_video_fields(paths))
    model, _ = load_lightgen2(args.weights, args.device)
    export_visualizations(model, paths, pixels.to(args.device), args.output)
    print('Saved vision and language NPZ/PNG files to ' + str(args.output))


if __name__ == '__main__':
    main()
