"""Draw measured results only; no fitted/imagined accuracy or image enhancement."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main(args):
    args.output.mkdir(parents=True, exist_ok=True)
    report = json.loads((args.training / 'final_report.json').read_text(encoding='utf-8'))
    history = json.loads((args.training / 'history.json').read_text(encoding='utf-8'))
    audit = json.loads((args.audit / 'audit.json').read_text(encoding='utf-8'))
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), layout='constrained')
    values = [audit['metrics']['square_64d']['hit_at_1'], report['metrics']['hit_at_1'],
              report['remove_optical_same_weights']['hit_at_1'], report['phase_pixels_shuffled_seed42']['hit_at_1']]
    bars = axes[0].bar(range(4), np.array(values) * 100, color=['#D55E00', '#0072B2', '#999999', '#CC79A7'])
    axes[0].bar_label(bars, fmt='%.2f', padding=3)
    axes[0].set(xticks=range(4), xticklabels=['Qwen\nsquare 64D', 'Optical\nalpha > 0.4', 'Same weights\nremove optical', 'Same weights\nshuffle phase'],
                ylabel='Test Hit@1 (%)', ylim=(0, 103), title='a  Fixed-checkpoint comparison')
    # Negative epochs are inherited conversion diagnostics, not this continuation.
    rows = [r for r in history if 'test' in r and r['epoch'] >= 0]
    axes[1].plot([r['epoch'] for r in rows], [100*r['test']['hit_at_1'] for r in rows], '-o', label='EMA / initial', color='#0072B2')
    live = [r for r in rows if 'test_live' in r]
    axes[1].plot([r['epoch'] for r in live], [100*r['test_live']['hit_at_1'] for r in live], '-s', label='Live', color='#D55E00')
    axes[1].axvline(report['selected_epoch'], linestyle='--', color='#555555', label='Selected epoch')
    axes[1].set(xlabel='Continuation epoch', ylabel='Test Hit@1 (%)', title='b  30-epoch continuation (test-selected)')
    axes[1].legend(fontsize=9)
    fig.savefig(args.output / '05_results_training.png', dpi=170)
    plt.close(fig)
    with (args.verification / 'ccd_readout_audit.csv').open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    keys = [(m, s) for m in ('vision', 'language') for s in ('expert', 'global')]
    fig, ax = plt.subplots(figsize=(8, 4.2), layout='constrained')
    summary = {}
    for offset, key, label, color in [(-.18, 'pooled_intensity_retained', 'Pooled intensity retained', '#0072B2'),
                                      (.18, 'decoded_squared_norm_retained', 'Decoded squared norm retained', '#D55E00')]:
        values = [100*np.mean([float(r[key]) for r in rows if (r['modality'], r['stage']) == pair]) for pair in keys]
        bars = ax.bar(np.arange(4)+offset, values, width=.34, label=label, color=color)
        ax.bar_label(bars, fmt='%.1f', padding=3)
        summary[key] = dict(zip(['_'.join(k) for k in keys], values))
    ax.set(xticks=range(4), xticklabels=['V expert', 'V global', 'L expert', 'L global'],
           ylabel='Retained fraction (%)', ylim=(0, 113), title='40 test products, first view each; existing decoder')
    ax.legend(fontsize=9, loc='upper right')
    fig.savefig(args.output / '06_ccd_readout.png', dpi=170)
    plt.close(fig)
    (args.output / 'plot_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('training', 'audit', 'verification', 'output'):
        p.add_argument('--'+key, type=Path, required=True)
    main(p.parse_args())
