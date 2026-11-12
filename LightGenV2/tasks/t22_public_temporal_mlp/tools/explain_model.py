"""Draw the configured serial model and its two simulated global stages."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

if __package__:
    from . import _paths
    from .plot_style import save_figure
else:
    import _paths
    from plot_style import save_figure

if __package__ and __package__ != 'tools':
    from ..model import ModelConfig, load_lightgen
else:
    from model import ModelConfig, load_lightgen


LEGACY_CONFIG = ModelConfig(
    readout_hidden=256,
    fusion_router_input='all_tokens',
    expert_mode='linear',
    orm_mode='amplitude',
)


def box(ax, x, y, width, height, text, color):
    ax.add_patch(FancyBboxPatch(
        (x, y), width, height, boxstyle='round,pad=0.08',
        facecolor=color, edgecolor='#526575', linewidth=1,
    ))
    ax.text(x+width/2, y+height/2, text, ha='center', va='center', fontsize=7)


def arrow(ax, start, end):
    ax.annotate('', xy=end, xytext=start,
                arrowprops={'arrowstyle': '->', 'color': '#34495e', 'lw': 1.5})


def draw_network(path, config=None):
    config = config or LEGACY_CONFIG
    width = config.width
    sequence_tokens = config.frames + config.prompt_length
    joint_width = 96 if config.fused_readout else width
    readout_input = sequence_tokens * joint_width
    router_tokens = sequence_tokens if config.fusion_router_input == 'all_tokens' else config.frames
    activation = 'ReLU' if config.mlp_activation == 'relu' else 'GELU'
    output_width = 5 if config.readout_mode == 'five_levels' else 1
    score_description = ('softmax -> five-level expected MOS' if config.readout_mode == 'five_levels'
                         else 'standardized score -> MOS')
    detector_label = ('mean-normalized intensity' if config.detector_processing == 'relative_intensity'
                      else 'mean / clip / log1p')
    detector_projection = '' if config.fused_readout else f' / Linear96->{width}'
    expert_label = 'linear experts' if config.expert_mode == 'linear' else 'MLP experts'

    fig, ax = plt.subplots(figsize=(7.2, 5.0), layout='constrained')
    ax.set(xlim=(0, 13), ylim=(0, 9))
    ax.axis('off')
    box(ax, .4, 7.7, 8.2, .8,
        'Four RGB frames -> frozen Qwen patch / position\n'
        'Spatial mean pooling -> 49 x 1024 per frame\n'
        f'LN / linear adapter -> 49 x {width}', '#e4edf9')
    box(ax, .4, 6.15, 8.2, 1.0,
        'VISION STAGE   (frame_*)\n'
        f'Each frame: router -> Top2 {expert_label} -> optical global\n'
        f'Four separate frames retain shape 4 x 49 x {width}', '#d9e8f8')
    arrow(ax, (4.5, 7.7), (4.5, 7.2))
    box(ax, .4, 4.8, 8.2, .8,
        f'Frame Merger: mean/max -> LN{2*width} -> Linear {2*width}->{width} -> {activation}\n'
        f'One video: 4 x {width} visual tokens', '#e4edf9')
    arrow(ax, (4.5, 6.15), (4.5, 5.65))
    box(ax, 9.2, 4.8, 3.2, .8,
        f'Prompt -> LN2048 / Linear\n{config.prompt_length} x {width} text tokens', '#fff0d6')
    box(ax, .4, 3.55, 12, .7,
        f'Concatenate four visual tokens and {config.prompt_length} text tokens -> {sequence_tokens} x {width}',
        '#f2e5f4')
    arrow(ax, (4.5, 4.8), (4.5, 4.3))
    arrow(ax, (10.8, 4.8), (10.8, 4.3))
    box(ax, .4, 1.9, 12, 1.1,
        'VISION-LANGUAGE STAGE   (video_*)\n'
        f'Router reads {router_tokens} tokens; experts process {sequence_tokens} image/text tokens\n'
        f'Router: LN -> Flatten {router_tokens*width} -> FC{config.router_hidden} -> {activation} -> FC4\n'
        f'Global -> CCD {detector_label} / pool / LN96 / ReLU{detector_projection} -> {sequence_tokens} x {joint_width}', '#f2e5f4')
    arrow(ax, (6.4, 3.55), (6.4, 3.05))
    box(ax, .4, .5, 12, .8,
        f'Flatten {readout_input} -> Linear {config.readout_hidden} -> {activation} -> Linear {output_width}\n'
        + score_description, '#e5f1e7')
    arrow(ax, (6.4, 1.9), (6.4, 1.35))
    ax.set_title('Two serial global operations per forward pass\n'
                 'The prompt enters after the first global and the Frame Merger', fontsize=8)
    save_figure(fig, path)
    plt.close(fig)


def draw_frame(path, config=None):
    config = config or LEGACY_CONFIG
    width = config.width
    activation = 'ReLU' if config.mlp_activation == 'relu' else 'GELU'
    fig, ax = plt.subplots(figsize=(7.2, 5.5), layout='constrained')
    ax.set(xlim=(0, 13), ylim=(0, 10))
    ax.axis('off')
    box(ax, 3.5, 8.7, 6, .8,
        'Each frame has 7 x 7 spatial token positions\n'
        f'49 tokens, each with {width} channels -> X[49,{width}]', '#e4edf9')
    box(ax, .4, 6.7, 5.4, 1.2,
        'Router\n'
        f'LN -> Flatten {49*width} -> FC{config.router_hidden} -> {activation} -> FC4\n'
        'softmax over four -> Top2 -> L1 renormalize', '#fff0d6')
    expert_description = (
        'Four independent linear experts\n'
        f'Each computes X[49,{width}] @ Wk.T[{width},{width}]\n'
        if config.expert_mode == 'linear' else
        'Four independent MLP experts\n'
        f'LN -> Linear {config.expert_hidden} -> {activation} -> Linear {width}\n'
    )
    box(ax, 6.5, 6.7, 6.1, 1.2,
        expert_description + f'Each output Yk has shape [49,{width}]', '#e5f1e7')
    arrow(ax, (5, 8.7), (3.1, 7.95))
    arrow(ax, (8.5, 8.7), (9.5, 7.95))
    box(ax, 2.2, 5.2, 8.6, .8,
        'Y = g1 Y1 + g2 Y2 + g3 Y3 + g4 Y4\n'
        f'Two nonzero gates, shared by all 49 tokens of this frame -> 49 x {width}', '#f2e5f4')
    arrow(ax, (3.1, 6.7), (4.2, 6.05))
    arrow(ax, (9.5, 6.7), (8.7, 6.05))
    box(ax, 2.2, 3.8, 8.6, .8,
        f'Project each token using Linear({width},27) -> ReLU\n'
        f'49 x {width} -> 49 x 27', '#e4edf9')
    arrow(ax, (6.5, 5.2), (6.5, 4.65))
    box(ax, 2.2, 2.35, 8.6, .9,
        'Transpose 49 x 27 -> 27 x 49\n'
        'Apply Linear(49,27) -> ReLU along the token axis\n'
        'Transpose back to a 27 x 27 nonnegative amplitude grid', '#e4edf9')
    arrow(ax, (6.5, 3.8), (6.5, 3.3))
    processing = ('mean normalize' if config.detector_processing == 'relative_intensity'
                  else 'mean / clip / log1p')
    box(ax, 2.2, .65, 8.6, 1.1,
        'RMS normalize -> pack all 64 frame fields -> coherent propagation\n'
        f'CCD region -> {processing} -> pool 49 x 96\n'
        f'LN96 -> ReLU -> Linear(96,{width}) -> 49 x {width}', '#d9e8f8')
    arrow(ax, (6.5, 2.35), (6.5, 1.85))
    ax.set_title('Frame token processing and optical encoding\n'
                 'Four frames share parameters and make separate routing decisions', fontsize=8)
    save_figure(fig, path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--weights', type=Path, help='Use the dimensions and detector processing of this checkpoint')
    args = parser.parse_args()
    config = load_lightgen(args.weights)[0].cfg if args.weights else LEGACY_CONFIG
    args.out.mkdir(parents=True, exist_ok=True)
    draw_network(args.out/'network_overview.png', config)
    draw_frame(args.out/'frame_dimensions.png', config)


if __name__ == '__main__':
    main()
