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
TAIL_EPOCHS = 10
MODE_COLORS = {'fixed': '#495B72', 'continuous': '#BA571C', 'constrained': '#16756A'}


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


def endpoint_summary(entries):
    """Compare all methods at the same epoch, with an a priori ten-epoch TEST diagnostic."""
    ends = {r['epochs'][-1]['epoch'] for r in entries}
    require(len(ends) == 1, 'Unmatched epoch in paired endpoint comparison')
    last = ends.pop()
    first = last - TAIL_EPOCHS + 1
    require(first >= 9, 'The ten-epoch diagnostic requires ten measured continuation epochs')
    by_mode = {}
    for mode in MODES:
        by_seed = {}
        for r in entries:
            if r['mode'] != mode:
                continue
            tail = r['epochs'][-TAIL_EPOCHS:]
            require([x['epoch'] for x in tail] == list(range(first, last+1))
                    and all(x['sources']['test'] == 'continuation_end_of_epoch' for x in tail),
                    'Missing observed TEST metric in the common terminal window')
            require(r['seed'] not in by_seed, 'Duplicate run at common endpoint')
            by_seed[r['seed']] = {'final': {s: tail[-1][s] for s in SPLITS},
                                  'last10_test_mean_percent': float(np.mean([x['test']['accuracy_percent'] for x in tail]))}
        require(set(by_seed) == set(SEEDS), 'Missing a paired seed at common endpoint')
        by_mode[mode] = {
            'seeds': by_seed,
            'final_mean_percent': {s: float(np.mean([x['final'][s]['accuracy_percent'] for x in by_seed.values()]))
                                   for s in SPLITS},
            'last10_test_mean_percent': float(np.mean([x['last10_test_mean_percent'] for x in by_seed.values()]))}
    deltas = {}
    for candidate in ('continuous', 'constrained'):
        records = {}
        for seed in SEEDS:
            a, b = (by_mode[m]['seeds'][seed]['final']['test'] for m in ('fixed', candidate))
            require(a['n'] == b['n'], 'Unequal paired TEST denominators')
            delta = b['correct'] - a['correct']
            records[seed] = {'test_n': a['n'], 'net_correct': delta, 'delta_pp': 100*delta/a['n']}
        deltas[candidate+'_minus_fixed'] = {
            'seeds': records, 'mean_delta_pp': float(np.mean([d['delta_pp'] for d in records.values()]))}
    return {'epoch': last, 'last10_epoch_range_inclusive': [first, last], 'by_mode': by_mode,
            'same_seed_final_test_deltas': deltas,
            'method': 'Full-split accuracies observed at the final common epoch. Tail TEST averages use all ten '
                      'observed end-of-epoch accuracies with equal weight; seeds have equal weight. These diagnostics '
                      'use the same held images repeatedly and do not select the checkpoint or stopping epoch.'}


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
            'selected_checkpoint_summary_basis': 'Both fields above use each run’s best VAL checkpoint.',
            'common_endpoint': endpoint_summary(entries),
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
    save_figure(fig, out)


def save_figure(fig, out):
    svg = out.with_suffix('.svg')
    fig.savefig(svg, metadata={'Date': None})
    lines = svg.read_text(encoding='utf-8').splitlines()
    svg.write_bytes(('\n'.join(line.rstrip() for line in lines) + '\n').encode('utf-8'))
    fig.savefig(out.with_suffix('.png'), dpi=140)
    plt.close(fig)


def draw_test_overview(datasets, out):
    """Display the complete mean trajectory alongside the prespecified endpoint and tail window."""
    fig = plt.figure(figsize=(11.3, 5.75))
    fig.subplots_adjust(top=.79, bottom=.30, left=.075, right=.98, wspace=.2)
    for index, (dataset, data) in enumerate(datasets.items(), 1):
        ax = fig.add_subplot(1, len(datasets), index)
        endpoint = data['summary']['common_endpoint']
        last = endpoint['epoch']
        first = endpoint['last10_epoch_range_inclusive'][0]
        ax.axvspan(1, 8, color='#d2d0ca', alpha=.42, zorder=0, lw=0)
        ax.axvspan(first-.5, last, color='#cfe4ed', alpha=.45, zorder=0, lw=0)
        for mode in MODES:
            rows = [r['epochs'] for r in data['runs'] if r['mode'] == mode]
            replay = np.mean([[r['replay_test']['accuracy_percent'] for r in run[:8]] for run in rows], axis=0)
            actual = np.mean([[r['test']['accuracy_percent'] for r in run[7:]] for run in rows], axis=0)
            summary = endpoint['by_mode'][mode]
            require(np.isclose(actual[-1], summary['final_mean_percent']['test'])
                    and np.isclose(actual[-TAIL_EPOCHS:].mean(), summary['last10_test_mean_percent']),
                    'Overview values differ from paired endpoint summary')
            color = MODE_COLORS[mode]
            ax.plot(range(1, 9), replay, color=color, lw=1.55, ls=':')
            ax.plot(range(8, last+1), actual, color=color, lw=1.45, alpha=.94)
            ax.scatter(8, actual[0], c=color, s=22, marker='o', edgecolor='white', linewidth=.4, zorder=4)
            ax.scatter(last, actual[-1], c=color, s=30, marker='s', edgecolor='white', linewidth=.5, zorder=4)
        ax.set_title(DATASETS[dataset][0], loc='left', fontweight='semibold', fontsize=12)
        ax.set_xlim(1, last+.012*last)
        ax.set_xlabel('Epoch')
        if index == 1:
            ax.set_ylabel('Mean TEST accuracy (%)')
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.18)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=7, integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.tick_params(labelsize=8, length=3)
        for k, mode in enumerate(MODES):
            tags = endpoint['by_mode'][mode]
            x = k/3+.015
            ax.text(x, -.25, MODES[mode], color=MODE_COLORS[mode], transform=ax.transAxes, fontsize=8.3,
                    fontweight='semibold')
            ax.text(x, -.355, f"{tags['final_mean_percent']['test']:.2f}% / {tags['last10_test_mean_percent']:.2f}%",
                    transform=ax.transAxes, fontsize=9)
    fig.suptitle('TEST accuracy by routing method', fontsize=17, fontweight='semibold', y=.965)
    legend = [Line2D([], [], color=MODE_COLORS[k], lw=2, label=v) for k, v in MODES.items()]
    fig.legend(handles=legend, loc='upper center', bbox_to_anchor=(.5,.918), ncol=3, frameon=False)
    fig.text(.075, .083, f'Values below each panel: common endpoint E{last} / mean over E{first}–{last} (blue shading). '
             'Accuracy is averaged equally over seeds 37 and 53.', fontsize=8.5)
    fig.text(.075, .04, 'Dotted E1–8 segments are replay diagnostics. Dots mark the original E8 checkpoints; '
             'solid lines show the continuation. Squares mark the common endpoint.', fontsize=8.5)
    save_figure(fig, out)


def write_html(out, report):
    endpoint = report['decision_history'][-1]
    passed = sum(run['plateau'] for run in endpoint['runs'].values())
    methods = {'fixed': '四区域', 'continuous': '原连续', 'constrained': '加约束连续'}
    rows = []
    panels = []
    for dataset, record in report['datasets'].items():
        title = DATASETS[dataset][0]
        summary = record['summary']
        for mode, label in methods.items():
            end = summary['common_endpoint']['by_mode'][mode]
            splits = end['final_mean_percent']
            locked = summary['mean_percentage_over_two_seeds'][mode]
            values = [splits['train'], splits['val'], splits['test'], end['last10_test_mean_percent'], locked['test']]
            rows.append(f'<tr><th scope="row">{title}</th><td>{label}</td>' +
                        ''.join(f'<td>{value:.2f}%</td>' for value in values) + '</tr>')
        panels.append(f'<section><h2>{title} 的训练、验证和测试曲线</h2>'
                      f'<a href="{dataset}_longrun.svg"><img src="{dataset}_longrun.svg" '
                      f'alt="{title} 三种路由、两个种子的逐轮准确率与训练损失"></a></section>')
    content = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>连续空间路由的等预算比较</title>
<style>body{{font:16px/1.55 system-ui,sans-serif;color:#182920;margin:32px auto;padding:0 20px;max-width:1160px}}
h1,h2{{line-height:1.25}}section{{margin:32px 0}}.scroll{{overflow:auto}}table{{border-collapse:collapse;width:100%;white-space:nowrap}}
th,td{{padding:7px 12px;text-align:right;border-bottom:1px solid #cdd6cf;font-variant-numeric:tabular-nums}}
th:first-child,td:nth-child(2){{text-align:left}}p{{max-width:78ch}}img{{max-width:100%;height:auto}}a{{color:#126957}}</style>
<h1>连续空间路由的等预算比较</h1>
<p>两个数据集，三种路由，配对种子 37 和 53。实验室服务器共同完成第 {endpoint['epoch']} 轮，
{passed}/12 组达到预设验证准确率平台标准。每个数据集内三种路由使用相同的样本顺序、更新次数和学习率。</p>
<div class="scroll"><table><thead><tr><th>数据集</th><th>路由</th><th>末轮 TRAIN</th><th>末轮 VAL</th>
<th>末轮 TEST</th><th>末 10 轮平均 TEST</th><th>最佳 VAL 权重的 TEST</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<p>百分比是两个种子的等权平均。按完整 VAL 正确数选权重，同分时取交叉熵更小、轮次更早者。测试集用于逐轮观察和最终比较。</p>
<section><h2>三种路由的测试轨迹</h2><a href="mean_test_overview.svg"><img src="mean_test_overview.svg" alt="两个数据集上三种路由的平均测试准确率轨迹"></a></section>
{''.join(panels)}
<p>逐轮图中绿色为 TRAIN，蓝色为 VAL，橙色为 TEST，下方为训练损失。前 8 轮的 TRAIN 和 TEST 点线取自单独回放，第 8 轮锚点来自原权重，第 9 轮起为实际续训记录。</p>
<p>两个原测试划分在此前 T17 开发中已有暴露。Blood 的原始划分中有 5 个相同像素哈希跨 TRAIN 与 TEST、1 个跨 VAL 与 TEST。本轮沿用固定划分，两个种子共用相同测试样本。</p>
<p><a href="results.json">逐轮数值、各个种子的结果与停训判据</a> · <a href="model_route_audit.json">权重与落点核验</a> · <a href="route_diagnostics.svg">验证集上的坐标扰动与互换实验</a></p></html>'''
    (out / 'results.html').write_text(content, encoding='utf8')


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
    overview = args.out / 'mean_test_overview'
    draw_test_overview(datasets, overview)
    report = {'protocol': protocol, 'decision_history': decisions, 'datasets': datasets,
              'overview_figure_sha256': {ext: sha(overview.with_suffix('.'+ext)) for ext in ('svg', 'png')},
              'total_optimizer_updates_all_runs': sum(d['summary']['total_optimizer_updates'] for d in datasets.values()),
              'total_continuation_updates_all_runs': sum(d['summary']['total_continuation_updates'] for d in datasets.values()),
              'epoch_evidence': 'VAL and online training loss are from the original trajectory throughout. E1–8 dotted '
                  'TRAIN/TEST curves are companion-replay diagnostics. Direct original checkpoint evaluations are labeled '
                  'per epoch. E8 dots anchor measured E9+ full-split end-of-epoch curves.',
              'selection': 'Best VAL correct count, then minimum VAL cross entropy, then earliest exact tie.',
              'plot_source_sha256': sha(Path(__file__))}
    (args.out / 'results.json').write_bytes((json.dumps(report, indent=2, allow_nan=False)+'\n').encode('utf-8'))
    write_html(args.out, report)


if __name__ == '__main__':
    main()
