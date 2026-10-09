"""Reproducible six-stage figure; exact metrics and checkpoint caveats retained."""
from pathlib import Path
import argparse
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).parent / 'reports' / 'cumulative_20261009')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 7, 'axes.labelsize': 7,
                         'xtick.labelsize': 7, 'ytick.labelsize': 7, 'axes.linewidth': .6,
                         'svg.fonttype': 'none', 'pdf.fonttype': 42, 'savefig.facecolor': 'white'})
    labels = ['Simulation reference', 'Hardware baseline', 'Sensor-aware training',
              'Sensor- and zero-order-aware training',
              'Sensor-, zero-order- and\ngeometry-aware training', 'Electronic adaptation†']
    values = [89.50, 54.90, 59.80, 69.15, 73.15, 89.10]
    positions = [6.1, 4.8, 3.8, 2.8, 1.8, .65]
    palette = ['#92989E', '#0072B2', '#D55E00']
    fig, ax = plt.subplots(figsize=(183/25.4, 112/25.4))
    fig.subplots_adjust(left=.365, right=.97, top=.86, bottom=.25)
    ax.barh(positions, values, height=.52, color=[palette[0]]+[palette[1]]*4+[palette[2]], edgecolor='none')
    for y, value in zip(positions, values):
        ax.text(value+1.3, y, f'{value:.2f}', va='center', ha='left', fontsize=7)
    ax.set_yticks(positions, labels)
    ax.set_xlim(0, 100)
    ax.set_ylim(-.05, 6.8)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xlabel('Changed-cell accuracy (%)', labelpad=7)
    ax.tick_params(axis='y', length=0, pad=8)
    ax.tick_params(axis='x', length=3, width=.6)
    for side in ['top', 'right', 'left']:
        ax.spines[side].set_visible(False)
    fig.legend(handles=[Patch(facecolor=c, label=label) for c, label in zip(
        palette, ['Simulation', 'Hardware evaluation', 'Electronic adaptation'])],
        loc='upper center', bbox_to_anchor=(.5, .985), ncol=3, frameon=False,
        handlelength=1.1, columnspacing=1.7, fontsize=7)
    notes = [
        '† Adaptation uses the frozen 73.15% optical core and its real CCD features; only the original decoder is trained.',
        '89.50% is the no-trick reference. Pre-adaptation simulation of the geometry-aware core is 91.20%.',
        'TEST development set: 1,000 scenes; TEST-selected checkpoints. Staged configurations, not isolated ablations.',
    ]
    for y, note in zip([.118, .081, .044], notes):
        fig.text(.035, y, note, fontsize=5.8)
    for ext in ['png', 'svg', 'pdf']:
        fig.savefig(args.output / f'openmoji_cumulative_8910.{ext}', dpi=600)
    plt.close(fig)
    metadata = {'labels': labels, 'values_percent': values, 'metric': 'changed_cell_accuracy',
                'notes': notes, 'adapted_run': 'editor16_align_decoder_consistency010_gpu120_20261009',
                'best_sha256': '0d9aa4c381e49386dd53ee4bd2388a1ec77b1411e26c76755351a9017de85a52',
                'selected_epoch': 30, 'test_scenes': 1000, 'test_selected_development': True}
    (args.output/'figure_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
