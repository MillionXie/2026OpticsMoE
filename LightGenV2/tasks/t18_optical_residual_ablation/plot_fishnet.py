"""Render completed FishNet audit records without model or dataset inference."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def export(fig, path):
    for suffix in ['png', 'svg', 'pdf']:
        target = path.with_suffix('.' + suffix)
        fig.savefig(target, dpi=300, bbox_inches='tight')
        if suffix == 'svg':
            target.write_text('\n'.join(line.rstrip() for line in target.read_text(encoding='utf-8').splitlines()) + '\n', encoding='utf-8')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reports', type=Path, default=Path(__file__).parent / 'reports/fishnet_20261011')
    parser.add_argument('--depth', type=int, choices=[2, 4, 6], action='append',
                        help='Render only these learning-curve depths; CSV summaries still include all audited arms')
    a = parser.parse_args()
    audits = [json.loads(p.read_text(encoding='utf-8')) for p in sorted(a.reports.glob('L*_audit.json'))]
    assert audits, 'No completed, audited training arms'
    audits.sort(key=lambda q: (q['result']['depth'], q['result']['rho']))
    plt.rcParams.update({'font.family': ['Arial', 'DejaVu Sans'], 'font.size': 9,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.fonttype': 'none', 'pdf.fonttype': 42})
    curves, summary = [], []
    for audit in audits:
        r = audit['result']
        assert r['epochs_completed'] == 100 and len(audit['curve']) == 100
        for row in audit['curve']:
            curves.append({'depth': r['depth'], 'rho': r['rho'], **row})
        row = {k: r[k] for k in ['depth', 'rho', 'parameters', 'selected_epoch', 'selected_state_kind',
                                 'epochs_completed', 'checkpoint_sha256']}
        row['scope'] = 'test-selected DEVELOPMENT'
        for split, metric in [('train', r['metrics']['train']), ('val', r['metrics']['val']),
                              ('test', r['test_development'])]:
            for key in ['accuracy', 'balanced_accuracy', 'nll', 'balanced_nll']:
                row[split + '_' + key] = metric[key]
        summary.append(row)
    for name, rows in [('curve_data.csv', curves), ('stage_summary.csv', summary)]:
        fields = list(dict.fromkeys(k for row in rows for k in row))
        with (a.reports / name).open('w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader(); writer.writerows(rows)
    palette = {'train': '#c46c73', 'val': '#3d7f9d', 'test_ema': '#3d8a68', 'test_raw': '#858585'}
    labels = {'train': 'Train EMA', 'val': 'Validation EMA',
              'test_ema': 'Test-development EMA', 'test_raw': 'Test-development raw'}
    for depth in sorted({q['result']['depth'] for q in audits}):
        if a.depth and depth not in a.depth:
            continue
        arms = [q for q in audits if q['result']['depth'] == depth]
        fig, axes = plt.subplots(len(arms), 2, figsize=(8.2, 2.8 * len(arms)), squeeze=False)
        for index, audit in enumerate(arms):
            r = audit['result']; rows = audit['curve']
            for split in palette:
                valid = [q for q in rows if split + '_accuracy' in q]
                x = [q['epoch'] for q in valid]
                style = ':' if split == 'test_raw' else '-'
                alpha = .5 if split == 'test_raw' else 1.
                axes[index, 0].plot(x, [100 * q[split + '_accuracy'] for q in valid],
                                    color=palette[split], linestyle=style, alpha=alpha,
                                    linewidth=1.3, label=labels[split])
                axes[index, 1].plot(x, [q[split + '_balanced_nll'] for q in valid],
                                    color=palette[split], linestyle=style, alpha=alpha,
                                    linewidth=1.3, label=labels[split])
            best = r['test_development']
            axes[index, 0].scatter([r['selected_epoch']], [100 * best['accuracy']],
                                    color='black', s=24, zorder=5, label='Selected checkpoint')
            axes[index, 1].scatter([r['selected_epoch']], [best['balanced_nll']], color='black', s=24, zorder=5)
            title = f"L{depth}, amplitude mixing coefficient {r['rho']:g}; selected epoch {r['selected_epoch']} ({r['selected_state_kind'].upper()})"
            axes[index, 0].set_title(title, loc='left', fontsize=9)
            axes[index, 1].set_title('Balanced negative log-likelihood', loc='left', fontsize=9)
            axes[index, 0].set_ylabel('Accuracy (%)'); axes[index, 0].set_ylim(0, 102)
            axes[index, 1].set_ylabel('Balanced NLL'); axes[index, 1].set_ylim(bottom=0)
            for ax in axes[index]:
                ax.set_xlabel('Epoch'); ax.set_xlim(1, 100); ax.grid(axis='y', alpha=.16)
            axes[index, 0].legend(frameon=False, fontsize=7, loc='lower right')
        fig.suptitle('FishNet v1 | seed 17 | 100 epochs | test-selected DEVELOPMENT', fontsize=10)
        fig.tight_layout()
        export(fig, a.reports / f'learning_curves_L{depth}')
    if all(any(q['result']['depth'] == d and q['result']['rho'] == rho for q in audits)
           for d in [2, 4, 6] for rho in [0., .3]):
        fig, ax = plt.subplots(figsize=(3.7, 3.1))
        for rho, color, label in [(0., '#c46c73', 'No unmodulated component'),
                                   (.3, '#3d7f9d', 'Amplitude mixture = 0.3')]:
            selected = sorted([q for q in audits if q['result']['rho'] == rho], key=lambda q: q['result']['depth'])
            x = [q['result']['depth'] for q in selected]
            y = [100 * q['result']['test_development']['accuracy'] for q in selected]
            ax.plot(x, y, marker='o', color=color, linewidth=1.7, markersize=4, label=label)
        ax.set_xticks([2, 4, 6]); ax.set_xlabel('Main optical layers'); ax.set_ylabel('Development accuracy (%)')
        ax.set_title('FishNet v1, seed 17', loc='left'); ax.legend(frameon=False, fontsize=7)
        ax.grid(axis='y', alpha=.16); fig.tight_layout()
        export(fig, a.reports / 'depth_accuracy')
    print(json.dumps({'completed_arms': len(audits), 'output': str(a.reports)}))


if __name__ == '__main__':
    main()
