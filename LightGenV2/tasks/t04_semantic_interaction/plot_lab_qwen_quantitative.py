"""Audit and plot the sealed lab .8655 / frozen-Qwen .8120 aggregate reports.

No model inference, training, hardware access, selection or raw-image processing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

GROUPS = ('overall', 'add', 'replace', 'move', 'remove')
METRICS = {
    'changed_cell_accuracy': 'Changed-cell accuracy',
    'edit_grid_iou': 'Edit-grid IoU',
    'object_f1': 'Object F1',
    'scene_exact_match': 'Scene exact match',
    'foreground_category_accuracy': 'Foreground accuracy',
    'cell_accuracy': 'All-cell accuracy',
    'preserved_cell_accuracy': 'Preserved-cell accuracy',
    'task_accuracy': 'Operation accuracy',
}
OURS_SHA = '03cb861c3ac344556601eb3eb6d7d1a22b77a54d2e7e68e85d77ee30fb09eb21'
QWEN_SHA = '0be775796d5f64df3fab6359eb354ec6666dfc29329484a620b619a6f7ec7cf8'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(evidence):
    ours = json.loads((evidence/'ours_lab_full_report.json').read_text())
    qwen = json.loads((evidence/'qwen_evaluation.json').read_text())
    manifest = json.loads((evidence/'source_manifest.json').read_text())
    for file, item in manifest['files'].items():
        assert sha(evidence/file) == item['sha256'], ('Source SHA changed', file)
    assert ours['status'] == 'complete' and ours['checkpoint_sha256'] == OURS_SHA
    assert manifest['qwen_checkpoint_sha256'] == QWEN_SHA
    assert ours['test_count'] == qwen['test_samples'] == 1000
    assert ours['ccd_images'] == 6000 and qwen['selected_epoch'] == 30
    assert abs(ours['metrics']['overall']['changed_cell_accuracy']-.8655) < 1e-12
    assert abs(qwen['metrics']['overall']['changed_cell_accuracy']-.8120) < 1e-12
    assert 'no downstream retraining' in ours['notes']
    for report in (ours, qwen):
        assert set(report['metrics']) == set(GROUPS)
        for group in GROUPS:
            assert report['metrics'][group]['samples'] == (1000 if group == 'overall' else 250)
            assert all(0 <= report['metrics'][group][key] <= 1 for key in METRICS)
        for key in METRICS:
            mean = sum(report['metrics'][g][key] for g in GROUPS[1:])/4
            assert abs(mean-report['metrics']['overall'][key]) < 1e-7, ('Group mean mismatch', key)
    predictions = [json.loads(line) for line in (evidence/'qwen_test_predictions.jsonl').read_text().splitlines()]
    assert len(predictions) == len({r['sample_id'] for r in predictions}) == 1000
    for group in GROUPS:
        records = predictions if group == 'overall' else [r for r in predictions if r['task'] == group]
        assert len(records) == qwen['metrics'][group]['samples']
        for key in METRICS:
            assert abs(sum(r[key] for r in records)/len(records)-qwen['metrics'][group][key]) < 1e-9
    rows = []
    for group in GROUPS:
        for metric, label in METRICS.items():
            a, b = ours['metrics'][group][metric], qwen['metrics'][group][metric]
            rows.append({'group': group, 'samples': 1000 if group == 'overall' else 250,
                         'metric': metric, 'label': label, 'ours': a, 'qwen': b,
                         'difference_pp': 100*(a-b), 'relative_change_percent': 100*(a/b-1) if b else None})
    return {'rows': rows, 'sources': manifest, 'ours': ours, 'qwen': qwen,
            'protocol': {
                'ours_domain': 'laboratory CCD, frozen epoch45, no downstream retraining',
                'baseline_domain': 'digital full frozen Qwen, trained shared readout, selected epoch30',
                'statistics': 'single-run aggregate comparison; no uncertainty/significance claims',
                'test_role': 'TEST-selected development results, not independent held-out generalization',
                'pairing_limit': 'Lab aggregate has no TEST manifest SHA or sample-ID prediction file; no paired per-case audit performed',
                'baseline_variant': 'original sealed evaluation; not the later A100 export with one secondary F1 drift',
            }}


def plot(result, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 9, 'axes.labelsize': 9,
                         'axes.titlesize': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.linewidth': .6,
                         'svg.fonttype': 'none', 'savefig.facecolor': 'white'})
    colors = ('#0072B2', '#D55E00')
    labels = ('Ours (laboratory)', 'Qwen baseline (digital)')
    def bars(ax, keys, group=None, metric=None):
        records = [next(r for r in result['rows'] if r['group'] == (group or key)
                        and r['metric'] == (key if group else metric)) for key in keys]
        x = np.arange(len(keys)); width = .34
        for j, name in enumerate(('ours', 'qwen')):
            values = [r[name]*100 for r in records]
            artists = ax.bar(x+(j-.5)*width, values, width, color=colors[j], label=labels[j], zorder=3)
            for i, (artist, value) in enumerate(zip(artists, values)):
                pair = (records[i]['ours']*100, records[i]['qwen']*100)
                label_y = value+2
                if group is None and abs(pair[0]-pair[1]) < 4:
                    label_y = max(pair)+2+j*5
                ax.text(artist.get_x()+artist.get_width()/2, label_y, f'{value:.2f}',
                        ha='center', va='bottom', fontsize=8)
        ax.set_ylim(0, 112)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_ylabel('Score (%)')
        ax.grid(axis='y', color='#e3e8ed', linewidth=.5, zorder=0)
        ax.tick_params(axis='x', length=0)
        return x

    fig, ax = plt.subplots(figsize=(7.2, 3.75))
    chosen = list(METRICS)[:5]
    x = bars(ax, chosen, 'overall')
    ax.set_xticks(x, ['Changed-cell\naccuracy', 'Edit-grid\nIoU', 'Object F1',
                      'Scene exact\nmatch', 'Foreground\naccuracy'])
    ax.set_title('Text-guided editing: overall test performance', loc='left', pad=31)
    ax.legend(loc='upper left', bbox_to_anchor=(0, 1.105), ncol=2, frameon=False, fontsize=8.5)
    fig.subplots_adjust(left=.085, right=.985, top=.78, bottom=.22)
    fig.text(.085, .045, '1,000 test scenes per system; TEST-selected development results; single run.',
             fontsize=8, color='#53616c')
    for ext in ('png', 'svg'):
        fig.savefig(output/f'01_overall_comparison.{ext}', dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(8.1, 3.55))
    for ax, keys_metric, title in zip(axes, ('changed_cell_accuracy', 'edit_grid_iou'),
                                    ('a  Changed-cell accuracy', 'b  Edit-grid IoU')):
        x = bars(ax, GROUPS[1:], metric=keys_metric)
        ax.set_xticks(x, ['Add', 'Replace', 'Move', 'Remove'])
        ax.set_title(title, loc='left', pad=10)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc='upper center', bbox_to_anchor=(.5, .995), ncol=2,
               frameon=False, fontsize=9)
    fig.subplots_adjust(left=.075, right=.99, top=.80, bottom=.20, wspace=.24)
    fig.text(.075, .045, '250 test scenes per operation per system; single-run development comparison.',
             fontsize=8, color='#53616c')
    for ext in ('png', 'svg'):
        fig.savefig(output/f'02_operation_comparison.{ext}', dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    task = Path(__file__).resolve().parent
    parser.add_argument('--evidence', type=Path, default=task/'runs/hardware/lab08655_qwen08120_comparison_20261015')
    parser.add_argument('--output', type=Path, default=task/'reports/lab_qwen_quantitative_20261015')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    result = audit(args.evidence)
    result['source_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    result['plotter_sha256'] = sha(__file__)
    result['evidence_dir'] = str(args.evidence.resolve())
    (args.output/'plot_data.json').write_text(json.dumps(result, indent=2)+'\n')
    plot(result, args.output)
    print(json.dumps({'output': str(args.output), 'overall': [r for r in result['rows'] if r['group']=='overall']}, indent=2))


if __name__ == '__main__':
    main()
