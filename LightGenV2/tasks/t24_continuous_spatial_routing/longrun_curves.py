"""Verify and plot the matched continuations of the twelve eight-epoch T24 runs."""
import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np

from .compare import DATASETS, read_json, require
from .plot_epoch_curves import (LOSS_COLOR, MODES, SEEDS, TEST_COLOR, TRAIN_COLOR, VAL_COLOR,
                                load_series, split_metrics, verified_files)
from .stability import assess_cohort, load_protocol


SPLITS = ('train', 'val', 'test')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def locked_predictions(original, extended, result, sizes):
    """Recount all three splits; bind paired predictions to the archived ordered IDs/labels."""
    predictions = {}
    archived = verified_files(original, [f'predictions_{split}.npz' for split in SPLITS])
    for split, n in zip(SPLITS, sizes):
        name = f'predictions_{split}.npz'
        with np.load(original / name, allow_pickle=False) as before, np.load(extended / name, allow_pickle=False) as after:
            y, p = after['labels'], after['predictions']
            require(len(np.unique(after['ids'])) == n and y.shape == p.shape == (n,)
                    and np.array_equal(after['ids'], before['ids'])
                    and np.array_equal(y, before['labels']), f'{extended.name}/{split}: sample identity or order differs')
            scores = after['scores']
            require(scores.ndim == 2 and scores.shape[0] == n and np.isfinite(scores).all()
                    and np.array_equal(scores.argmax(1), p)
                    and np.count_nonzero(p == y) == result[split]['correct'],
                    f'{extended.name}/{split}: incorrect locked predictions or count')
            predictions[split] = (after['ids'], y, p)
    return archived, predictions


def joined_series(original, replay, long, dataset, seed, mode):
    start = load_series(original, replay, dataset, seed, mode)
    archived_meta, archived_history = (read_json(original / name) for name in ('metadata.json', 'history.json'))
    files = ['metadata.json', 'history.json', 'config.json', 'best_checkpoint.pt', 'last_checkpoint.pt',
             'selection_lock.json', 'result.json', *(f'predictions_{s}.npz' for s in SPLITS)]
    hashes = verified_files(long, files)
    meta, history, config, lock, result, status = (read_json(long / name) for name in
        ('metadata.json', 'history.json', 'config.json', 'selection_lock.json', 'result.json', 'status.json'))
    lineage = meta['lineage']
    require(lineage['original_manifest_sha256'] == sha(original / 'manifest.json')
            and lineage['original_last_sha256'] == start['original_sha256']['last_checkpoint.pt']
            and lineage['original_best_sha256'] == start['original_sha256']['best_checkpoint.pt']
            and lineage['original_history_sha256'] == start['original_sha256']['history.json']
            and lineage['parent_source_commit'] == start['original_source_commit']
            and Path(lineage['original_run']).name == original.name, f'{long.name}: E8 parent identity mismatch')
    target = len(history)
    require(target > 8 and status['state'] == 'complete' and status['epoch'] == status['target_epoch'] == target
            and target == config['epochs'] == result['target_epoch'] == lock['target_epoch'] == meta['target_epoch']
            and config == meta['config'] == {**archived_meta['config'], 'epochs': target}
            and hashes['config.json'] == meta['config_file_sha256']
            and meta['initial_state_sha256'] == start['original_initial_state_sha256']
            and meta['data'] == archived_meta['data']
            and meta['parent_epoch'] == 8 and meta['parent_config'] == archived_meta['config']
            and history[:8] == archived_history, f'{long.name}: inconsistent continuation data/config/budget')
    best = max(history, key=lambda r: (r['validation']['correct'], -r['validation']['cross_entropy']))['epoch']
    require(all(r['epoch'] == i for i, r in enumerate(history, 1))
            and lock['mode'] == result['mode'] == meta['mode'] == mode and lock['seed'] == result['seed'] == seed
            and hashes['best_checkpoint.pt'] == result['best_checkpoint_sha256'] == lock['best_checkpoint_sha256']
            and meta['source_commit'] == lock['source_commit'] == result['source_commit']
            and result['selected_epoch'] == lock['selected_epoch'] == status['selected_epoch'] == best
            and lock['TEST_selection_calls'] == result['TEST_selection_calls'] == 0
            and result['end_of_epoch_TEST_diagnostic_calls_since_parent'] == target - 8,
            f'{long.name}: selected checkpoint or diagnostic budget disagrees with history')
    sizes = DATASETS[dataset][1]
    locked = {s: split_metrics(result[s], n) for s, n in zip(SPLITS, sizes)}
    original_prediction_hashes, predictions = locked_predictions(original, long, result, sizes)
    early = start['epochs']
    for s in ('train', 'test'):
        require(early[-1]['direct_or_replay_sources'][s] in
                ('original_best_archived_predictions', 'original_last_direct_evaluation'), f'Missing direct E8 {s}')
    rows = []
    rng = np.random.default_rng(seed)
    batch = config['batch']
    for i, r in enumerate(history):
        order = rng.permutation(sizes[0]).astype(np.int64)
        require(r['train_steps'] == (sizes[0] + batch - 1) // batch
                and hashlib.sha256(order.tobytes()).hexdigest() == r['minibatch_order_sha256']
                and np.isfinite(r['train_loss']) and np.isfinite(r['validation']['cross_entropy']),
                f'{long.name} E{i+1}: invalid loss, updates, or shuffle order')
        row = {'epoch': r['epoch'], 'train_steps': r['train_steps'], 'online_train_loss': r['train_loss'],
               'val': split_metrics(r['validation'], sizes[1]), 'val_cross_entropy': r['validation']['cross_entropy']}
        if i < 8:
            e = early[i]
            row.update(train=e['direct_or_replay_train'], test=e['direct_or_replay_test'],
                       sources=e['direct_or_replay_sources'], replay_train=e['companion_train'], replay_test=e['companion_test'])
        else:
            row.update(train=split_metrics(r['train'], sizes[0]), test=split_metrics(r['test'], sizes[2]),
                       sources={s: 'continuation_end_of_epoch' for s in SPLITS})
        rows.append(row)
    for s in SPLITS:
        require(locked[s]['correct'] == rows[best-1][s]['correct'], f'{long.name}:{s} locked E{best} differs')
    out = {'run': long.name, 'mode': mode, 'seed': seed, 'epochs': rows, 'selected_epoch': best, 'locked': locked,
           'budget': {'epochs': target, 'continuation_epochs': target - 8,
                      'optimizer_updates': sum(r['train_steps'] for r in rows),
                      'continuation_updates': sum(r['train_steps'] for r in rows[8:]),
                      'training_image_presentations': target * sizes[0], 'continuation_TEST_diagnostics': target - 8,
                      'TEST_selection_calls': 0},
           'original_source_commit': start['original_source_commit'], 'continuation_source_commit': meta['source_commit'],
           'initial_state_sha256': start['original_initial_state_sha256'], 'data_manifest_sha256': config['manifest_sha256'],
           'archived_epoch8_checkpoint_sha256': lineage['original_last_sha256'], 'lineage': lineage,
           'original_sha256': {**start['original_sha256'], **original_prediction_hashes},
           'replay_sha256': start['companion_sha256'], 'replay_trajectory_status': start['trajectory_status'],
           'continuation_sha256': hashes}
    return out, history, predictions


def paired_summary(entries, archives):
    """Summaries retain seeds as distinct runs with repeated use of the same images."""
    pairwise = []
    for seed in SEEDS:
        names = {r['mode']: r['run'] for r in entries if r['seed'] == seed}
        for reference, candidate in combinations(MODES, 2):
            paired = {}
            for split in SPLITS:
                ids, labels, pred_a = archives[names[reference]][split]
                other_ids, other_labels, pred_b = archives[names[candidate]][split]
                require(np.array_equal(ids, other_ids) and np.array_equal(labels, other_labels), 'Unpaired samples')
                a, b = pred_a == labels, pred_b == labels
                gained = int(np.count_nonzero(~a & b))
                lost = int(np.count_nonzero(a & ~b))
                paired[split] = {'n': len(labels), 'both_correct': int(np.count_nonzero(a & b)),
                                 'both_wrong': int(np.count_nonzero(~a & ~b)), 'candidate_only_correct': gained,
                                 'reference_only_correct': lost, 'net_correct': gained - lost,
                                 'delta_pp': 100. * (gained-lost) / len(labels)}
            pairwise.append({'seed': seed, 'reference': reference, 'candidate': candidate, 'splits': paired})
    averages = {}
    for mode in MODES:
        selected = [r for r in entries if r['mode'] == mode]
        averages[mode] = {s: float(np.mean([r['locked'][s]['accuracy_percent'] for r in selected])) for s in SPLITS}
    return {'within_seed_pairs': pairwise, 'mean_percentage_over_two_seeds': averages,
            'total_optimizer_updates': sum(r['budget']['optimizer_updates'] for r in entries),
            'total_continuation_updates': sum(r['budget']['continuation_updates'] for r in entries),
            'total_training_image_presentations': sum(r['budget']['training_image_presentations'] for r in entries)}


def draw(title, runs, decision, out):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'svg.fonttype': 'none',
                         'svg.hashsalt': 't24-matched-long-run-curves-v1'})
    fig = plt.figure(figsize=(12, 8))
    grid = fig.add_gridspec(2, 3, left=.08, right=.982, top=.845, bottom=.1, hspace=.48, wspace=.21)
    limit = len(runs[0]['epochs'])
    losses = [r['online_train_loss'] for e in runs for r in e['epochs']]
    vals = [r[s]['accuracy_percent'] for e in runs for r in e['epochs'] for s in SPLITS]
    vals += [r[s]['accuracy_percent'] for e in runs for r in e['epochs'][:8] for s in ('replay_train', 'replay_test')]
    acc_min = max(0, 10 * np.floor((min(vals)-1)/10))
    loss_high = max(losses)*1.07
    for run in runs:
        i, j = SEEDS.index(run['seed']), list(MODES).index(run['mode'])
        panel = grid[i, j].subgridspec(2, 1, height_ratios=(3.1, 1), hspace=.12)
        ac = fig.add_subplot(panel[0, 0])
        loss = fig.add_subplot(panel[1, 0], sharex=ac)
        rows = run['epochs']
        for split, color in [('train', TRAIN_COLOR), ('test', TEST_COLOR)]:
            ac.plot(range(1, 9), [r['replay_'+split]['accuracy_percent'] for r in rows[:8]],
                    color=color, lw=1.2, ls=':')
            ac.plot(range(8, limit + 1), [r[split]['accuracy_percent'] for r in rows[7:]], color=color, lw=1.15)
            ac.scatter(8, rows[7][split]['accuracy_percent'], s=15, color=color, edgecolor='white', lw=.3, zorder=4)
        ac.plot(range(1, limit + 1), [r['val']['accuracy_percent'] for r in rows], color=VAL_COLOR, lw=1.1)
        selected = run['selected_epoch']
        ac.scatter(selected, run['locked']['val']['accuracy_percent'], marker='*', s=105, c=VAL_COLOR,
                   edgecolors='white', linewidths=.6, zorder=5)
        ac.scatter(selected, run['locked']['test']['accuracy_percent'], marker='D', s=29, c=TEST_COLOR,
                   edgecolors='white', linewidths=.55, zorder=5)
        loss.plot(range(1, limit + 1), [r['online_train_loss'] for r in rows], color=LOSS_COLOR, lw=1.25)
        ac.set_ylim(acc_min, 101)
        loss.set_ylim(0, loss_high)
        ac.yaxis.set_major_locator(MaxNLocator(nbins=4))
        loss.yaxis.set_major_locator(MaxNLocator(nbins=3))
        for ax in (ac, loss):
            ax.set_xlim(1, limit+.9)
            ax.axvspan(1, 8, color='#CBC7BC', alpha=.22, linewidth=0, zorder=0)
            ax.spines[['right', 'top']].set_visible(False)
            ax.tick_params(length=3, width=.65, labelsize=8)
            ax.grid(axis='y', alpha=.19)
        ac.tick_params(labelbottom=False)
        loss.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        ac.set_title(f"{MODES[run['mode']]} · seed {run['seed']}", loc='left', fontsize=10, fontweight='semibold')
        if j == 0:
            ac.set_ylabel('Accuracy (%)')
            loss.set_ylabel('Loss')
        if i == 1:
            loss.set_xlabel('Epoch')
    fig.suptitle(f'{title}  |  Matched training through epoch {limit}', fontsize=17, fontweight='semibold', y=.975)
    legend = [Line2D([], [], color=c, label=s) for s, c in [('TRAIN', TRAIN_COLOR), ('VAL', VAL_COLOR),
              ('TEST', TEST_COLOR), ('Online TRAIN loss', LOSS_COLOR)]]
    legend += [Line2D([], [], ls='', marker=m, color=c, markersize=n, label=s)
               for m, c, n, s in [('*', VAL_COLOR, 11, 'Best VAL'), ('D', TEST_COLOR, 6, 'TEST at best VAL')]]
    fig.legend(handles=legend, loc='upper center', bbox_to_anchor=(.5,.94), ncol=6, frameon=False, columnspacing=1.4)
    fig.text(.08, .044, 'Shading marks E1–8. Dotted TRAIN/TEST lines show a seeded replay. '
             'The measured continuation starts at the original E8 checkpoint (dots).', fontsize=8)
    hit = sum(r['plateau']['plateau'] for r in runs)
    fig.text(.08, .023, f'{hit}/6 runs meet the predeclared VAL stability thresholds at E{limit}.  '
             f'All-dataset decision: {decision.replace("_", " ")}.  TEST curves are diagnostic.', fontsize=8)
    svg = out.with_suffix('.svg')
    fig.savefig(svg, metadata={'Date': None})
    lines = svg.read_text(encoding='utf-8').splitlines()
    svg.write_bytes(('\n'.join(line.rstrip() for line in lines) + '\n').encode('utf-8'))
    fig.savefig(out.with_suffix('.png'), dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key, help_text in [('runs', 'Original E1–8 runs'), ('replays', 'Companion E1–8 diagnostics'),
                           ('long', 'Twelve completed continuation run directories under this path'),
                           ('out', 'Directory for SVGs, PNGs and results.json')]:
        parser.add_argument('--'+key, type=Path, required=True, help=help_text)
    args = parser.parse_args()
    protocol = load_protocol()
    histories, archives, datasets = {}, {}, {}
    for dataset in DATASETS:
        entries = []
        for seed in SEEDS:
            for mode in MODES:
                name = f'{dataset}_s{seed}_{mode}'
                entry, hist, preds = joined_series(args.runs / name, args.replays / name, args.long / name,
                                                   dataset, seed, mode)
                entries.append(entry)
                histories[name], archives[name] = hist, preds
        require(len({e['data_manifest_sha256'] for e in entries}) == 1, f'{dataset}: mixed split manifests')
        for seed in SEEDS:
            require(len({e['initial_state_sha256'] for e in entries if e['seed'] == seed}) == 1, 'Unmatched initial weights')
        datasets[dataset] = {'runs': entries, 'summary': paired_summary(entries, archives)}
    require(len({len(h) for h in histories.values()}) == 1, 'All 12 runs must share a common endpoint')
    target = len(next(iter(histories.values())))
    require(protocol['min_epochs'] <= target <= protocol['max_epochs']
            and (target-protocol['min_epochs']) % protocol['decision_interval'] == 0, 'Invalid scheduled endpoint')
    decisions = [assess_cohort(histories, protocol, epoch=ep) for ep in
                 range(protocol['min_epochs'], target+1, protocol['decision_interval'])]
    require(all(d['decision'] == 'continue' for d in decisions[:-1]), 'History exceeds earliest joint stopping point')
    final = decisions[-1]
    args.out.mkdir(parents=True, exist_ok=True)
    for dataset, data in datasets.items():
        for entry in data['runs']:
            entry['plateau'] = final['runs'][entry['run']]
        path = args.out / f'{dataset}_longrun'
        draw(DATASETS[dataset][0], data['runs'], final['decision'], path)
        data['figure_sha256'] = {ext: sha(path.with_suffix('.'+ext)) for ext in ('svg','png')}
    report = {'protocol': protocol, 'decision_history': decisions, 'datasets': datasets,
              'total_optimizer_updates_all_runs': sum(d['summary']['total_optimizer_updates'] for d in datasets.values()),
              'total_continuation_updates_all_runs': sum(d['summary']['total_continuation_updates'] for d in datasets.values()),
              'epoch_evidence': 'VAL and online training loss are from the original trajectory throughout. E1–8 dotted '
                  'TRAIN/TEST curves are companion-replay diagnostics. Direct original checkpoint evaluations are labeled '
                  'per epoch. E8 dots anchor measured E9+ full-split end-of-epoch curves.',
              'selection': 'Best VAL correct count, then minimum VAL cross entropy, then earliest exact tie.',
              'plot_source_sha256': sha(Path(__file__))}
    (args.out / 'results.json').write_bytes((json.dumps(report, indent=2, allow_nan=False)+'\n').encode('utf-8'))


if __name__ == '__main__':
    main()
