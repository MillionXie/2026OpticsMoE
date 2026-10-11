"""Plot only real audited full60 train/val histories and locked single test points."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
COLORS = {'train': '#c66b75', 'val': '#3684a0', 'test': '#222222'}


def export_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def save(fig, dest):
    for ext in ('png', 'svg', 'pdf'):
        fig.savefig(dest.with_suffix('.' + ext), dpi=180, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reports', type=Path, default=HERE / 'reports/butterfly60_20261011')
    parser.add_argument('--depth', type=int, choices=[2, 4, 8, 16])
    parser.add_argument('--fit-only', action='store_true')
    args = parser.parse_args()
    files = sorted(args.reports.glob('*_fit_audit.json' if args.fit_only else '*_audit.json'))
    if not args.fit_only:
        files = [f for f in files if not f.name.endswith('_fit_audit.json')]
    audits = [json.loads(f.read_text(encoding='utf-8')) for f in files]
    assert audits and all(x['schema'] == 'butterfly60_validation_v1' and x['audit_passed'] for x in audits)
    assert len({x['run_id'] for x in audits}) == 1, 'Never mix distinct runs or datasets'
    assert all(x['result']['selection_split'] == 'val' and not x['result']['test_used_for_selection'] for x in audits)
    assert all(x['result']['test_evaluations'] == int(not args.fit_only) for x in audits)
    curves, summaries = [], []
    for audit in audits:
        result = audit['result']
        prefix = dict(depth=result['depth'], rho=result['rho'], candidate=result['candidate'])
        for point in audit['history']:
            row = dict(prefix, epoch=point['epoch'], lr=point['lr'], train_objective=point['train_loss'])
            for split in ('train', 'val'):
                if split in point:
                    row.update({f'{split}_{key}': value for key, value in point[split].items()})
            curves.append(row)
        row = dict(prefix, parameters=result['parameters'], epochs_completed=result['epochs_completed'],
                   selected_epoch=result['selected_epoch'], checkpoint_sha256=result['checkpoint_sha256'])
        for split, metric in result['metrics'].items():
            row.update({f'{split}_{key}': metric[key] for key in ('accuracy', 'balanced_accuracy', 'macro_f1', 'nll', 'balanced_nll')})
        summaries.append(row)
    export_csv(args.reports / ('fit_curve_data.csv' if args.fit_only else 'curve_data.csv'), curves)
    export_csv(args.reports / ('fit_stage_summary.csv' if args.fit_only else 'stage_summary.csv'), summaries)
    depths = [args.depth] if args.depth else sorted({x['result']['depth'] for x in audits})
    for depth in depths:
        arms = sorted([x for x in audits if x['result']['depth'] == depth], key=lambda x: (x['result']['candidate'], x['result']['rho']))
        assert arms, ('No real audited arm', depth)
        fig, axes = plt.subplots(len(arms), 2, figsize=(12, 3.5 * len(arms)), squeeze=False)
        scope = 'train/val fit only; test not read' if args.fit_only else 'validation selected; one test per locked EMA'
        fig.suptitle(f'Butterfly-10K | 60 species | seed 17 | L{depth} | {scope}', fontsize=14)
        for i, audit in enumerate(arms):
            result, history = audit['result'], audit['history']
            for j, key in enumerate(('accuracy', 'balanced_nll')):
                ax = axes[i, j]
                scale = 100 if key == 'accuracy' else 1
                for split in ('train', 'val'):
                    points = [point for point in history if split in point]
                    ax.plot([p['epoch'] for p in points], [p[split][key] * scale for p in points],
                            color=COLORS[split], label=f'{split} EMA', linewidth=1.8)
                ax.axvline(result['selected_epoch'], color='#999999', linestyle=':', linewidth=.8, label='Selected epoch')
                if not args.fit_only:
                    ax.scatter([result['selected_epoch']], [result['metrics']['test'][key] * scale],
                               color=COLORS['test'], label='One test after all locks', s=30, zorder=4)
                ax.set_title(f"rho={result['rho']}; {result['candidate']}; selected E{result['selected_epoch']} / {result['epochs_completed']}", fontsize=10)
                ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy (%)' if key == 'accuracy' else 'Balanced NLL')
                ax.grid(alpha=.18); ax.legend(fontsize=8)
                ax.spines[['top', 'right']].set_visible(False)
                if key == 'accuracy':
                    ax.set_ylim(0, 102)
        fig.tight_layout(rect=(0, 0, 1, .97))
        save(fig, args.reports / (f'fit_learning_curves_L{depth}' if args.fit_only else f'learning_curves_L{depth}'))
    if not args.fit_only and {x['result']['depth'] for x in audits} == {2, 4, 8, 16}:
        assert len(audits) == 8
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        for rho, style in ((0., '-o'), (.3, '--s')):
            arms = sorted([x['result'] for x in audits if x['result']['rho'] == rho], key=lambda x: x['depth'])
            assert len(arms) == 4
            for ax, key, title in zip(axes, ('accuracy', 'balanced_accuracy', 'balanced_nll'),
                                     ('Accuracy (%)', 'Macro recall (%)', 'Balanced NLL')):
                for split in ('train', 'val', 'test'):
                    scale = 1 if key == 'balanced_nll' else 100
                    ax.plot([x['depth'] for x in arms], [x['metrics'][split][key] * scale for x in arms], style,
                            color=COLORS[split], linewidth=1.3, label=f'{split}, rho={rho}')
                ax.set_xticks([2, 4, 8, 16]); ax.set_xlabel('Main optical layers'); ax.set_ylabel(title)
                ax.grid(alpha=.18); ax.spines[['top', 'right']].set_visible(False)
        axes[-1].legend(fontsize=7)
        fig.suptitle('Full60 | validation-selected bounded fit protocol | seed 17 | no error bars')
        fig.tight_layout(); save(fig, args.reports / 'depth_comparison')
    print(json.dumps(dict(audited_arms=len(audits), plotted_depths=depths, fit_only=args.fit_only)))


if __name__ == '__main__':
    main()
