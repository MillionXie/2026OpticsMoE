"""Read-only training diagnostics; optionally watch the existing owned PID.

Never touches checkpoints or optimizer state. Per-mode historical metrics are
captured only when latest JSON agrees with a completed history entry; unavailable
old per-mode values are not reconstructed from a global average.
"""
import argparse
import csv
import json
import os
import time
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def current_snapshot(run, history):
    if not history:
        return None
    row = history[-1]
    dev, train = read(run/'val_latest.json'), read(run/'train_latest.json')
    checks = [(row['val_psnr'], dev['mean_per_image_psnr_db']),
              (row['val_lpips'], dev['mean_lpips']),
              (row['train_psnr'], train['mean_per_image_psnr_db'])]
    if any(abs(a-b) > 1e-8 for a, b in checks):
        return None
    keys = ['count', 'mean_per_image_psnr_db', 'mean_lpips',
            'mean_object_crop_lpips', 'by_mode', 'lpips_by_mode',
            'object_crop_lpips_by_mode', 'by_category_mode']
    return {'step': row['step'], 'epoch': row['epoch'],
            'train_diagnostic': {k: train[k] for k in keys},
            'development_test': {k: dev[k] for k in keys}}


def render(run, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    history = read(run/'history.json')
    if not history:
        return
    output.mkdir(parents=True, exist_ok=True)
    try:
        snap = current_snapshot(run, history)
        if snap:
            (output/f"step_{snap['step']:05d}.json").write_text(json.dumps(snap, indent=2))
    except (FileNotFoundError, json.JSONDecodeError):
        pass  # Evaluation may be writing the next result. Try next observation.
    before = read(run/'val_before_training.json')
    xs = [r['epoch'] for r in history]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(xs, [r['train_psnr'] for r in history], 'o-', label='TRAIN: fixed 12-pair diagnostic')
    ax.plot([0]+xs, [before['mean_per_image_psnr_db']]+[r['val_psnr'] for r in history], 'o-', label='Development TEST: all 96 pairs')
    ax.set(title='PSNR (higher is better)', ylabel='dB')
    ax.legend(fontsize=8)
    ax = axes[0, 1]
    ax.plot([0]+xs, [before['mean_lpips']]+[r['val_lpips'] for r in history], 'o-', label='Full-image LPIPS')
    ax.plot([0]+xs, [before['mean_object_crop_lpips']]+[r['val_object_crop_lpips'] for r in history], 'o-', label='Object-crop LPIPS')
    ax.set(title='Development TEST perception (lower is better)', ylabel='LPIPS')
    ax.legend(fontsize=8)
    snaps = sorted([read(p) for p in output.glob('step_*.json')], key=lambda s: s['step'])
    for mode, color in zip(('background', 'object', 'joint'), ('C0', 'C1', 'C2')):
        for ax, metric in zip(axes[1], ('lpips_by_mode', 'object_crop_lpips_by_mode')):
            ax.plot([0]+[s['epoch'] for s in snaps],
                    [before[metric][mode]]+[s['development_test'][metric][mode] for s in snaps],
                    'o-', color=color, label=mode)
    axes[1, 0].set(title='Per-mode full-image LPIPS', ylabel='LPIPS')
    axes[1, 1].set(title='Per-mode object-crop LPIPS', ylabel='LPIPS')
    for ax in axes.flat:
        ax.set_xlabel('Epoch in the current continuation')
        ax.grid(alpha=.2)
    for ax in axes[1]:
        ax.legend(fontsize=8)
    fig.suptitle('960 TRAIN / 96 development TEST, lamps + pillows\nTEST is used for model selection; not an independent generalization test', fontsize=11)
    fig.savefig(output/'learning_curves.png', dpi=160)
    plt.close(fig)
    with (output/'global_history.csv').open('w', newline='', encoding='utf-8') as f:
        fields = ['step', 'epoch', 'train_loss', 'train_psnr', 'val_psnr', 'val_lpips', 'val_object_crop_lpips']
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader(); w.writerows(history)
    last = history[-1]
    summary = {'observed_unix': time.time(), 'last_epoch': last['epoch'],
        'last_development_test': {k: v for k, v in last.items() if k.startswith('val_')},
        'train_psnr_diagnostic_pairs': 12, 'development_test_pairs': before['count'],
        'psnr_gap_diagnostic_db': last['train_psnr']-last['val_psnr'],
        'best_development_lpips_epoch': min(history, key=lambda r: r['val_lpips'])['epoch'],
        'per_mode_snapshots_started_at_epoch': snaps[0]['epoch'] if snaps else None,
        'limitations': ['TRAIN curve is the existing fixed 12-pair diagnostic, not the full 960-pair TRAIN mean.',
             'Single-minibatch train_loss is not an epoch-mean loss; do not infer convergence from it alone.',
             'Missing old per-mode snapshots are unavailable, not interpolated or invented.',
             'TRAIN/TEST SKU and conditional target diversity differ; a PSNR gap alone does not prove overfitting.',
             'TEST-based targeted refinement remains development optimization, not independent evaluation.']}
    (output/'diagnosis.json').write_text(json.dumps(summary, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--watch-pid', type=int)
    a = p.parse_args()
    starttime = None
    if a.watch_pid:
        starttime = Path(f'/proc/{a.watch_pid}/stat').read_text().split()[21]
        command = Path(f'/proc/{a.watch_pid}/cmdline').read_bytes().replace(b'\0', b' ').decode()
        if 'qwen_vl_spatial_run' not in command or str(a.run) not in command:
            raise ValueError('Only watch this run\'s existing training PID')
    while True:
        try:
            render(a.run, a.output)
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        proc = Path(f'/proc/{a.watch_pid}/stat')
        if not a.watch_pid or not proc.exists() or proc.read_text().split()[21] != starttime:
            break
        time.sleep(30)
    (a.output/'observer_exit.json').write_text(json.dumps({'complete': True,
            'observed_pid': a.watch_pid, 'observer_pid': os.getpid(), 'gpu_used': False}))


if __name__ == '__main__':
    main()
