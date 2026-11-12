"""Redraw the single selected model from retained numerical records and weights."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import torch

if __package__:
    from . import analyze_routing, explain_model, illustrate_routing, visualize_fields
    from .analyze_phase_effect import plot_phases
    from .data import batch, groups, load_inputs, sha256
    from .evaluate_phase_robustness import plot_curves, plot_masks
    from .model import load_lightgen
    from .phase_plotting import PHASE_DISPLAY, add_phase_colorbar, phase_display_norm
    from .phase_regularization import illumination_support
    from .plot_style import BODY_SIZE_PT, FONT_FAMILY, TITLE_SIZE_PT, save_figure
else:
    import analyze_routing
    import explain_model
    import illustrate_routing
    import visualize_fields
    from analyze_phase_effect import plot_phases
    from data import batch, groups, load_inputs, sha256
    from evaluate_phase_robustness import plot_curves, plot_masks
    from model import load_lightgen
    from phase_plotting import PHASE_DISPLAY, add_phase_colorbar, phase_display_norm
    from phase_regularization import illumination_support
    from plot_style import BODY_SIZE_PT, FONT_FAMILY, TITLE_SIZE_PT, save_figure


ROOT = Path(__file__).resolve().parent
STAGES = ('frame_global', 'video_global')


def read_json(path):
    return json.loads(Path(path).read_text('utf-8-sig'))


def plot_phase_training(initial, trained, output, arrays_path, statistics_path):
    arrays = {}
    statistics = {}
    for name in STAGES:
        initial_phase = getattr(initial, name).phase(wrapped=True).detach().cpu().numpy()
        trained_phase = getattr(trained, name).phase(wrapped=True).detach().cpu().numpy()
        difference = np.arctan2(np.sin(trained_phase - initial_phase),
                                np.cos(trained_phase - initial_phase))
        support = illumination_support(getattr(trained, name).frame_level).numpy()
        arrays[name + '_initial'] = initial_phase
        arrays[name + '_trained'] = trained_phase
        arrays[name + '_wrapped_update'] = difference
        statistics[name] = {
            'all_16_tiles_RMS_rad': float(np.sqrt(np.mean(difference ** 2))),
            'illuminated_support_RMS_rad': float(np.sqrt(np.mean(difference[:, support] ** 2))),
            'slot0_RMS_rad': float(np.sqrt(np.mean(difference[0] ** 2))),
            'maximum_absolute_update_rad': float(np.abs(difference).max()),
        }
    limit = max(float(np.abs(arrays[name + '_wrapped_update']).max()) for name in STAGES)
    limit = max(limit, 1e-6)
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.4), layout='constrained')
    for row, name in enumerate(STAGES):
        stage = 'Vision' if row == 0 else 'Vision-language'
        for column, suffix in enumerate(('_initial', '_trained')):
            image = axes[row, column].imshow(
                arrays[name + suffix][0], cmap='twilight', norm=phase_display_norm())
            axes[row, column].set_title(stage + (' initialization' if column == 0 else ' trained'))
        add_phase_colorbar(fig, image, list(axes[row, :2]))
        image = axes[row, 2].imshow(arrays[name + '_wrapped_update'][0],
                                   cmap='RdBu_r', vmin=-limit, vmax=limit)
        axes[row, 2].set_title('Wrapped training update')
        fig.colorbar(image, ax=axes[row, 2], shrink=.6, aspect=14,
                     label='Update (rad)', ticks=[-limit, 0, limit], format='%.2f')
    for ax in axes.flat:
        ax.set(xticks=[], yticks=[])
    fig.suptitle('Actual run initialization and selected weights, physical video slot 0')
    save_figure(fig, output)
    plt.close(fig)
    np.savez_compressed(arrays_path, **arrays)
    statistics_path.write_text(json.dumps({
        'phase_unit': 'radians',
        'difference': 'atan2(sin(trained-initial), cos(trained-initial))',
        'statistics_scope': 'all 16 physical phase tiles unless explicitly marked slot0',
        'image_scope': 'physical video slot 0',
        'initialization_sha256': sha256(ROOT / 'assets/initialization/initialization.pt'),
        'selected_sha256': sha256(ROOT / 'runs/simulation/current/LightGen.pt'),
        'statistics': statistics,
    }, indent=2) + '\n', 'utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paths', type=Path, default=ROOT / 'paths.local.json')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    font_path = font_manager.findfont(font_manager.FontProperties(family=FONT_FAMILY),
                                     fallback_to_default=False)
    profile = read_json(ROOT / 'configs/selected.json')
    checkpoint = ROOT / profile['checkpoint']
    if sha256(checkpoint) != profile['checkpoint_sha256']:
        raise ValueError('Selected checkpoint identity mismatch')
    current = checkpoint.parent
    output = current / 'figures'
    output.mkdir(exist_ok=True)
    model, metadata = load_lightgen(checkpoint, args.device)
    initial, _ = load_lightgen(ROOT / 'assets/initialization/initialization.pt', 'cpu')
    torch.set_num_threads(4)

    # Retained field arrays provide the exact numerical source of these panels.
    field_sources = {}
    for name in ('frame', 'video'):
        path = current / 'fields' / (name + '_global_arrays.npz')
        with np.load(path) as data:
            values = {key: data[key] for key in data.files}
        visualize_fields.plot_overview(getattr(model, name + '_global'), values,
                                       output / (name + '_global.png'))
        field_sources[str(path.relative_to(ROOT))] = sha256(path)

    route = read_json(current / 'routing/routing_analysis.json')
    if route['weights_sha256'] != profile['checkpoint_sha256']:
        raise ValueError('Routing record belongs to a different checkpoint')
    shares = {name: np.asarray(route[name]['share_percent']) / 100 for name in ('frame', 'video')}
    analyze_routing.plot_shares(shares, output / 'optical_expert_share.png')
    mass = {name: np.asarray(route[name]['routing_weight_mass_percent']) / 100
            for name in ('frame', 'video')}
    analyze_routing.plot_shares(mass, output / 'mean_routing_weight.png', 'Mean routing weight (%)')
    slot_shares = {name: np.zeros((16, 4)) for name in ('frame', 'video')}
    with (current / 'routing/expert_share_by_slot.csv').open(encoding='utf-8-sig') as stream:
        for row in csv.DictReader(stream):
            slot_shares[row['stage']][int(row['physical_slot']), int(row['expert'][1:]) - 1] = (
                float(row['share_percent']) / 100)
    analyze_routing.plot_slot_shares(slot_shares, output / 'expert_share_by_slot.png')

    payload = load_inputs(read_json(args.paths))
    if metadata['dataset']['manifest']['sha256'] != payload['identity']['manifest']['sha256']:
        raise ValueError('Checkpoint and input manifest identities differ')
    indices, valid = groups(payload, 'test')
    item = batch(payload, indices[:1], valid[:1], args.device)
    with torch.inference_mode():
        result, captured, _ = visualize_fields.trace(model, item['vision'])
    tokens = captured['frame']['expert_tokens'][0, 0, 0].cpu()
    gates = captured['frame']['expert_weights'][0, 0, 0].cpu().numpy()
    analyze_routing.plot_expert_operation(model, tokens, gates, output / 'expert_matrix_and_tokens.png')
    analyze_routing.plot_virtual_activation(tokens, gates, output / 'virtual_expert_activation.png')
    analyze_routing.plot_router_sensitivity(model, tokens.to(args.device), output / 'router_sensitivity.png')
    analyze_routing.plot_layout(output / 'global_layout.png')
    analyze_routing.plot_phase_support(model, output / 'phase_illumination_support.png')

    example = read_json(current / 'routing_example/routing_example.json')
    if example['checkpoint_SHA256'] != profile['checkpoint_sha256']:
        raise ValueError('Routing example belongs to a different checkpoint')
    inputs = ROOT / 'assets/example'
    identity = read_json(inputs / 'input_identity.json')
    if identity['sample_id'] != payload['ids'][int(indices[0, 0])]:
        raise ValueError('Example image and cached input identities differ')
    for filename, expected in example['image_SHA256'].items():
        if sha256(inputs / filename) != expected:
            raise ValueError('Example image changed: ' + filename)
    for key, actual in (
        ('frame_probabilities', result['routing']['frame']['probabilities'][0, 0]),
        ('joint_probabilities', result['routing']['video']['probabilities'][0, 0]),
    ):
        if not np.allclose(actual.cpu().numpy(), example[key], rtol=1e-4, atol=1e-5):
            raise ValueError('Retained routing example differs from current forward pass')
    illustrate_routing.plot_example(
        identity, inputs, np.asarray(example['frame_probabilities']),
        np.asarray(example['frame_gates']), np.asarray(example['joint_probabilities']),
        np.asarray(example['joint_gates']), output / 'routing_example.png')

    effects = read_json(current / 'phase_effect/phase_effect.json')
    with np.load(current / 'phase_effect/phase_arrays.npz') as data:
        arrays = {'Current': {name: data['Current_' + name] for name in (
            'frame_global', 'frame_global_centered', 'video_global', 'video_global_centered')}}
    plot_phases(effects, arrays, output / 'phase_texture.png')
    robust = read_json(current / 'robustness/comparison.json')
    with np.load(current / 'robustness/slot0_phase_arrays.npz') as data:
        phases = {'Current': {name: data['Current_' + name] for name in STAGES}}
    plot_curves(robust, output / 'robustness_curves.png')
    plot_masks(robust, phases, output / 'phase_masks.png')
    plot_phase_training(initial, model, output / 'phase_training_update.png',
                        current / 'fields/phase_training_arrays.npz',
                        current / 'fields/phase_training_statistics.json')
    explain_model.draw_network(output / 'network_overview.png')
    explain_model.draw_frame(output / 'frame_dimensions.png')
    record = {
        'checkpoint_sha256': sha256(checkpoint),
        'font_family': FONT_FAMILY,
        'font_file': font_path,
        'body_size_pt': BODY_SIZE_PT,
        'title_size_pt': TITLE_SIZE_PT,
        'dpi': 300,
        'phase_display': PHASE_DISPLAY,
        'phase_colorbar_ticks_rad': [0, math.pi, 2 * math.pi],
        'field_array_sources': field_sources,
        'initialization_sha256': sha256(ROOT / 'assets/initialization/initialization.pt'),
        'script_sha256': sha256(Path(__file__)),
        'files': {path.name: sha256(path) for path in sorted(output.glob('*.png'))},
    }
    (output / 'render_manifest.json').write_text(json.dumps(record, indent=2) + '\n', 'utf-8')
    print(json.dumps({'figures': len(record['files']), 'directory': str(output)}))


if __name__ == '__main__':
    main()
