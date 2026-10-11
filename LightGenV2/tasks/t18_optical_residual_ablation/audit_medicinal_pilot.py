"""Independent CPU receipts and scientific plots from saved pilot artifacts only."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import numpy as np


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path): return json.loads(path.read_text())


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU audit only'
    import torch
    from butterfly_ablation import metrics
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True); p.add_argument('--data', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--depth', type=int, choices=[2,4,8,16], default=2)
    p.add_argument('--arm0', type=Path); p.add_argument('--arm03', type=Path)
    a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    manifest = read(a.data.parent/'data_manifest.json')
    classes = len(manifest['classes'])
    assert digest(a.data) == manifest['cache_sha256']
    summaries, histories, identities, configurations, sources = [], [], [], [], []
    for name in ['rho0', 'rho03']:
        arm = (a.arm0 if name == 'rho0' else a.arm03) or a.run/name
        result = read(arm/'fit_result.json'); history = read(arm/'history.json')
        best = torch.load(arm/'best_checkpoint.pt', map_location='cpu', weights_only=False)
        last = torch.load(arm/'last_checkpoint.pt', map_location='cpu', weights_only=False)
        selected = min(history, key=lambda h: h['val']['balanced_nll'])
        assert result['test_evaluations'] == 0 and not result['test_used_for_selection']
        assert 'test' not in result['metrics'] and all('test' not in h for h in history)
        assert result['epochs_completed'] == len(history) == last['epoch'] <= result['config']['requested_epochs'] <= 100
        assert [h['epoch'] for h in history] == list(range(1, len(history)+1))
        assert selected['epoch'] == best['epoch'] == result['selected_epoch']
        assert selected['val'] == best['validation'] == result['metrics']['val']
        assert result['checkpoint_sha256'] == digest(arm/'best_checkpoint.pt')
        assert result['history_sha256'] == digest(arm/'history.json')
        assert result['config'] == best['config'] == last['config'] == read(arm/'config.json')
        assert result['sources'] == best['sources'] == last['sources'] == read(arm/'metadata.json')['sources']
        assert result['data_sha256'] == best['data_sha256'] == last['data_sha256'] == manifest['cache_sha256']
        commit = read(arm/'metadata.json')['git_commit']
        assert all(hashlib.sha256(subprocess.check_output(['git', 'show', commit+':'+path])).hexdigest() == sha
            for path, sha in result['sources'].items())
        assert result['depth'] == a.depth
        assert sum(v.numel() for k, v in best['model'].items() if k.endswith('raw_phase')) == result['parameters'] == 10000+439848*(a.depth//2)
        assert result['initial_parameter_sha256'] == last['initial_parameter_sha256'] == read(arm/'smoke.json')['initial_parameter_sha256']
        assert result['orders'] == last['orders'] and result['transforms'] == last['transforms']
        gradients = read(arm/'gradients.json')
        assert len(gradients) == len(history) and all(len(g['norms']) == 1+5*a.depth and all(np.isfinite(v) and v > 0 for v in g['norms'].values()) for g in gradients)
        for split in ['train', 'val']:
            with np.load(a.data, allow_pickle=False) as z:
                labels, ids = z[split+'_labels'].copy(), z[split+'_ids'].copy()
            rows = list(csv.DictReader((arm/(split+'_predictions.csv')).open()))
            assert [row['sample_id'] for row in rows] == ids.tolist() == manifest['split_ids'][split]
            assert [int(row['label_true']) for row in rows] == labels.tolist()
            probability = np.array([[float(row[f'score{k}']) for k in range(classes)] for row in rows])
            assert probability.argmax(1).tolist() == [int(row['label_pred']) for row in rows]
            computed = metrics(labels, probability)
            saved = result['metrics'][split]
            for key in ['support', 'confusion_matrix']: assert computed[key] == saved[key]
            for key in ['accuracy', 'balanced_accuracy', 'nll', 'balanced_nll', 'macro_f1']:
                assert np.isclose(computed[key], saved[key], atol=1e-10, rtol=1e-10)
        tm, vm = result['metrics']['train'], result['metrics']['val']
        summary = dict(rho=result['rho'], epochs=result['epochs_completed'], best_epoch=result['selected_epoch'],
            train_accuracy=tm['accuracy'], val_accuracy=vm['accuracy'], train_macro=tm['balanced_accuracy'], val_macro=vm['balanced_accuracy'],
            train_nll=tm['nll'], val_nll=vm['nll'], train_balanced_nll=tm['balanced_nll'], val_balanced_nll=vm['balanced_nll'],
            selected_train_val_gap_pp=100*(tm['accuracy']-vm['accuracy']), seconds=result['seconds'],
            mean_epoch_seconds=float(np.mean([h['epoch_seconds'] for h in history])),
            accuracy_gate_passed=result['accuracy_gate_passed'], speed_gate_passed=result['speed_gate_passed'],
            stop_reason=result['stop_reason'], checkpoint_sha256=result['checkpoint_sha256'],
            last_checkpoint_sha256=digest(arm/'last_checkpoint.pt'), test_evaluations=0,
            last_train_accuracy=history[-1]['train']['accuracy'], last_val_accuracy=history[-1]['val']['accuracy'])
        summaries.append(summary); histories.append(history)
        identities.append(result['initial_parameter_sha256']); configurations.append(result['config']); sources.append(result['sources'])
    assert identities[0] == identities[1] and sources[0] == sources[1]
    assert {k for k in configurations[0] if configurations[0][k] != configurations[1][k]} == {'residual_rho'}
    length = min(map(len, histories))
    results = [read(arm/'fit_result.json') for arm in [a.arm0 or a.run/'rho0', a.arm03 or a.run/'rho03']]
    assert results[0]['orders'][:length] == results[1]['orders'][:length]
    assert results[0]['transforms'][:length] == results[1]['transforms'][:length]
    report = dict(scope='Validation-reused feasibility pilot only, not independent test or formal residual benefit',
        run=str(a.run.resolve()), cache_sha256=manifest['cache_sha256'], manifest_sha256=digest(a.data.parent/'data_manifest.json'),
        sources=sources[0], initial_parameter_sha256=identities[0], cpu_checks_passed=True,
        paired_overlap_epochs=length, summaries=summaries, test_evaluations=0)
    (a.out/'pilot_audit.json').write_text(json.dumps(report, indent=2))
    with (a.out/'pilot_summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0])); writer.writeheader(); writer.writerows(summaries)
    curves = []
    for rho, history in zip([0., .3], histories):
        for h in history:
            curves.append(dict(rho=rho, epoch=h['epoch'], val_accuracy=h['val']['accuracy'], val_macro=h['val']['balanced_accuracy'],
                val_balanced_nll=h['val']['balanced_nll'], train_accuracy=h.get('train', {}).get('accuracy', ''),
                train_macro=h.get('train', {}).get('balanced_accuracy', ''), train_balanced_nll=h.get('train', {}).get('balanced_nll', ''), epoch_seconds=h['epoch_seconds']))
    with (a.out/'curve_data.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(curves[0])); writer.writeheader(); writer.writerows(curves)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for rho, history, summary, color in zip([0., .3], histories, summaries, ['#2365a5', '#ce6c23']):
        train = [h for h in history if 'train' in h]
        axes[0].plot([h['epoch'] for h in history], [100*h['val']['accuracy'] for h in history], color=color, label=f'Val rho={rho}')
        axes[0].plot([h['epoch'] for h in train], [100*h['train']['accuracy'] for h in train], 'o--', color=color, markersize=3, label=f'Train rho={rho}')
        axes[1].plot([h['epoch'] for h in history], [h['val']['balanced_nll'] for h in history], color=color, label=f'Val rho={rho}')
        axes[1].plot([h['epoch'] for h in train], [h['train']['balanced_nll'] for h in train], 'o--', color=color, markersize=3, label=f'Train rho={rho}')
        for ax in axes: ax.axvline(summary['best_epoch'], color=color, alpha=.25, linestyle=':')
    axes[0].axhline(70, color='#888888', linewidth=.8, linestyle=':')
    axes[0].set(ylabel='Accuracy (%)', ylim=(0, 100)); axes[1].set(ylabel='Balanced NLL')
    for ax in axes: ax.set(xlabel='Epoch'); ax.grid(alpha=.15); ax.legend(fontsize=8)
    fig.suptitle(f'{configurations[0]["dataset"]} | {classes} classes | L{a.depth} | seed17 | train / val only')
    for ext in ['png', 'svg', 'pdf']: fig.savefig(a.out/(f'learning_curves_L{a.depth}.'+ext), dpi=180)
    svg = a.out/f'learning_curves_L{a.depth}.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    plt.close(fig)
    print(json.dumps(dict(cpu_checks_passed=True, summaries=summaries), indent=2))


if __name__ == '__main__': main()
