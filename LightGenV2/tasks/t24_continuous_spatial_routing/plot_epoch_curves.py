"""Plot the matched T24 original training records and companion epoch diagnostics."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np

from .compare import DATASETS, read_json, require


MODES = {'fixed': 'Four fixed corners', 'continuous': 'Two free peaks',
         'constrained': 'Constrained pair'}
SEEDS = (37, 53)
TRAIN_COLOR, TEST_COLOR = '#247D73', '#AC521F'
VAL_COLOR, LOSS_COLOR = '#405577', '#575A61'


def verified_files(root, names):
    manifest = read_json(root / 'manifest.json')
    for name in names:
        actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
        require(actual == manifest[name], f'{root.name}/{name} differs from manifest')
    return {k: manifest[k] for k in names}


def split_metrics(stats, n):
    require(stats['n'] == n and 0 <= stats['correct'] <= n, 'Invalid split count')
    accuracy = stats['correct'] / n
    require(abs(stats['accuracy'] - accuracy) < 1e-6, 'Inconsistent correct count / accuracy')
    return {'n': n, 'correct': int(stats['correct']), 'accuracy_percent': 100 * accuracy}


def load_series(original, companion, dataset, seed, mode):
    for path in (original, companion):
        require(read_json(path / 'status.json')['state'] == 'complete', f'{path} incomplete')
    original_sha = verified_files(original, ['metadata.json', 'history.json', 'selection_lock.json',
                                             'result.json', 'best_checkpoint.pt', 'last_checkpoint.pt',
                                             'predictions_test.npz'])
    companion_sha = verified_files(companion, ['result.json'])
    meta = read_json(original / 'metadata.json')
    result = read_json(original / 'result.json')
    lock = read_json(original / 'selection_lock.json')
    history = read_json(original / 'history.json')
    replay = read_json(companion / 'result.json')
    config = meta['config']
    title, sizes = DATASETS[dataset]
    require(meta['mode'] == result['mode'] == lock['mode'] == replay['mode'] == mode
            and config['dataset'] == replay['dataset'] == title.lower()
            and config['seed'] == result['seed'] == lock['seed'] == replay['seed'] == seed,
            f'{original} run identity mismatch')
    require(original_sha['best_checkpoint.pt'] == lock['best_checkpoint_sha256']
            == result['best_checkpoint_sha256'] == replay['original_best_checkpoint_sha256']
            and result['selected_epoch'] == lock['selected_epoch'] == replay['original_selected_epoch']
            and lock['TEST_selection_calls'] == 0, f'{original} original checkpoint or selection mismatch')
    require(original_sha['history.json'] == replay['original_history_sha256']
            and original_sha['last_checkpoint.pt'] == replay['original_last_checkpoint_sha256']
            and meta['source_commit'] == replay['original_source_commit']
            and meta['initial_state_sha256'] == replay['initial_state_sha256']
            and meta['source_sha256'] == replay['original_source_sha256']
            and replay['replay_TEST_selection_calls'] == 0
            and replay['verification']['all_split_ids_labels_and_order_verified']
            and replay['verification']['ordered_minibatch_hashes_verified'],
            f'{original} companion used different original history, final state or training code')
    require(config['expected_counts'] == dict(zip(('train', 'val', 'test'), sizes)), 'Original split sizes')
    require(8 == config['epochs'] == len(replay['epochs']) == len(history), 'Original/replay budget differs')
    selected = lock['selected_epoch']
    require(selected == max(history, key=lambda r: (r['validation']['correct'], -r['validation']['cross_entropy']))['epoch'],
            'Selected epoch does not match original VAL history')
    with np.load(original / 'predictions_test.npz', allow_pickle=False) as z:
        correct = int(np.count_nonzero(z['scores'].argmax(1) == z['labels']))
        require(len(z['labels']) == sizes[2] and result['test']['correct'] == correct,
                f'{original} locked TEST predictions mismatch')
    rows = []
    for epoch, (initial, repeated) in enumerate(zip(history, replay['epochs']), 1):
        require(epoch == initial['epoch'] == repeated['epoch'], 'Missing epoch or unexpected ordering')
        rows.append({'epoch': epoch, 'original_online_train_loss': initial['train_loss'],
                     'original_val': split_metrics(initial['validation'], sizes[1]),
                     'companion_train': split_metrics(repeated['replay']['train'], sizes[0]),
                     'companion_val': split_metrics(repeated['replay']['val'], sizes[1]),
                     'companion_test': split_metrics(repeated['replay']['test'], sizes[2]),
                     'direct_or_replay_sources': repeated['source'],
                     'direct_or_replay_train': split_metrics(repeated['train'], sizes[0]),
                     'direct_or_replay_test': split_metrics(repeated['test'], sizes[2])})
    return {'mode': mode, 'seed': seed, 'run': original.name, 'original_source_commit': meta['source_commit'],
            'original_sha256': original_sha, 'companion_sha256': companion_sha,
            'original_split_manifest_sha256': config['manifest_sha256'],
            'original_initial_state_sha256': meta['initial_state_sha256'],
            'selected_epoch': selected, 'trajectory_status': replay['verification']['trajectory_status'],
            'original_locked_test': split_metrics(result['test'], sizes[2]), 'epochs': rows}


def draw(title, runs, out):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'svg.fonttype': 'none',
                         'svg.hashsalt': 't24-original-and-companion-epochs-v1'})
    fig = plt.figure(figsize=(11.1, 8.8))
    grid = fig.add_gridspec(2, 3, left=.095, right=.978, top=.845, bottom=.09, hspace=.59, wspace=.24)
    max_loss = max(x['original_online_train_loss'] for r in runs for x in r['epochs'])
    min_loss = min(x['original_online_train_loss'] for r in runs for x in r['epochs'])
    all_acc = [q[k]['accuracy_percent'] for r in runs for q in r['epochs']
               for k in ('original_val', 'companion_train', 'companion_test')]
    all_acc.extend(r['original_locked_test']['accuracy_percent'] for r in runs)
    all_acc.extend(r['epochs'][-1]['direct_or_replay_test']['accuracy_percent'] for r in runs)
    lower = max(0, 10 * np.floor((min(all_acc) - 2) / 10))
    upper = min(100, 10 * np.ceil((max(all_acc) + 3) / 10))
    loss_pad = max(.05, .1 * (max_loss - min_loss))
    for r in runs:
        i, j = SEEDS.index(r['seed']), list(MODES).index(r['mode'])
        panel = grid[i, j].subgridspec(2, 1, height_ratios=(3.15, 1), hspace=.15)
        ac = fig.add_subplot(panel[0, 0])
        loss = fig.add_subplot(panel[1, 0], sharex=ac)
        epochs = [row['epoch'] for row in r['epochs']]
        for key, color, line in (('companion_train', TRAIN_COLOR, '-'), ('companion_test', TEST_COLOR, '--'),
                                 ('original_val', VAL_COLOR, ':')):
            ac.plot(epochs, [row[key]['accuracy_percent'] for row in r['epochs']],
                    ls=line, color=color, lw=1.8, marker='.', markersize=4)
        ac.scatter(r['selected_epoch'], r['original_locked_test']['accuracy_percent'],
                   marker='*', s=125, c='#181818', edgecolors='white', linewidths=.55, zorder=5)
        final = r['epochs'][-1]
        if final['epoch'] != r['selected_epoch']:
            require('last' in final['direct_or_replay_sources']['test'].lower(),
                    'Missing direct original final-checkpoint TEST evaluation')
            ac.scatter(final['epoch'], final['direct_or_replay_test']['accuracy_percent'],
                       marker='D', s=36, facecolors='none', edgecolors='#181818', linewidths=1.1, zorder=5)
        loss.plot(epochs, [row['original_online_train_loss'] for row in r['epochs']],
                  color=LOSS_COLOR, marker='.', lw=1.4, ms=3)
        ac.set_ylim(lower, upper)
        loss.set_ylim(max(0, min_loss - loss_pad), max_loss + loss_pad)
        ac.yaxis.set_major_locator(MaxNLocator(nbins=4))
        loss.yaxis.set_major_locator(MaxNLocator(nbins=2))
        ac.set_title(f"{MODES[r['mode']]}  ·  seed {r['seed']}", loc='left', fontweight='semibold')
        for ax in (ac, loss):
            ax.spines[['right', 'top']].set_visible(False)
            ax.tick_params(length=3, width=.6, labelsize=8)
            ax.grid(axis='y', alpha=.18)
            ax.set_xlim(.8, max(epochs) + .24)
        plt.setp(ac.get_xticklabels(), visible=False)
        loss.set_xticks(epochs)
        if j == 0:
            ac.set_ylabel('Accuracy (%)')
            loss.set_ylabel('Loss')
        if i == 1:
            loss.set_xlabel('Epoch')
    fig.suptitle(f'{title}   |   Eight-epoch routing comparison', fontsize=16, fontweight='semibold', y=.973)
    handles = [Line2D([], [], color=TRAIN_COLOR, label='Replay TRAIN'),
               Line2D([], [], color=TEST_COLOR, ls='--', label='Replay TEST'),
               Line2D([], [], color=VAL_COLOR, ls=':', label='Original VAL'),
               Line2D([], [], ls='', marker='*', color='#181818', markersize=10, label='Original locked TEST'),
               Line2D([], [], ls='', marker='D', color='#181818', mfc='none', markersize=6, label='Original final TEST¹'),
               Line2D([], [], color=LOSS_COLOR, label='Original online TRAIN loss')]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .94), ncol=6, frameon=False,
               columnspacing=1, handlelength=1.9, fontsize=8)
    verified = sum(r['trajectory_status'] == 'matched_all_observable_checks' for r in runs)
    fig.text(.095, .032, 'TRAIN and TEST lines use full splits measured after each replay epoch. '
             'Stars show original checkpoints selected by validation.', fontsize=8.5)
    fig.text(.095, .015, f'¹ Final original checkpoint re-evaluated if different from the selected epoch. '
             f'Replays matching all observable checks: {verified}/{len(runs)}.',
             fontsize=8.5)
    svg = out.with_suffix('.svg')
    fig.savefig(svg, metadata={'Date': None})
    lines = svg.read_text(encoding='utf-8').splitlines()
    svg.write_bytes(('\n'.join(line.rstrip() for line in lines) + '\n').encode('utf-8'))
    fig.savefig(out.with_suffix('.png'), dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, required=True, help='Original twelve matched runs')
    parser.add_argument('--replays', type=Path, required=True, help='Companion runs using the same directory names')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    report = {'loss_definition': 'Original sample-weighted mean minibatch training objective during updates.',
              'accuracy_definition': 'Each companion point evaluates a whole split at its end-of-epoch weights.',
              'locked_test_definition': 'Original TEST correctness at the original best VAL-selected checkpoint.',
              'plot_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'datasets': {}}
    for dataset, (title, _) in DATASETS.items():
        all_series = []
        for seed in SEEDS:
            for mode in MODES:
                name = f'{dataset}_s{seed}_{mode}'
                all_series.append(load_series(args.runs / name, args.replays / name, dataset, seed, mode))
        report['datasets'][dataset] = all_series
    args.out.mkdir(parents=True, exist_ok=True)
    for dataset, series in report['datasets'].items():
        draw(DATASETS[dataset][0], series, args.out / f'{dataset}_epochs')
    (args.out / 'results.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
