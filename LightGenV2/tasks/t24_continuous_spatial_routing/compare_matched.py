"""Verify and summarize a three-arm matched-budget cohort on one dataset."""
import argparse
from itertools import combinations
import json
from pathlib import Path

import numpy as np

from .compare import DATASETS, SPLITS, COPY_SIDE, inspect_run, require


ARMS = ('fixed', 'continuous', 'constrained')
MATCHED_FIELDS = ('initial_state_sha256', 'config', 'config_file_sha256', 'source_commit',
                  'source_sha256', 'parameter_count', 'optimizer', 'selection', 'data')


def paired(first, second):
    require(all(np.array_equal(x, y) for x, y in zip(first[:2], second[:2])),
            'Original image IDs, order, or labels differ')
    first_correct, second_correct = first[2], second[2]
    gains = int((~first_correct & second_correct).sum())
    losses = int((first_correct & ~second_correct).sum())
    return {
        'both_correct': int((first_correct & second_correct).sum()),
        'both_wrong': int((~first_correct & ~second_correct).sum()),
        'only_second_correct': gains, 'only_first_correct': losses,
        'delta_correct': gains - losses,
        'delta_pp': 100 * (gains - losses) / len(first_correct),
    }


def audit_seed(root, dataset, seed, split_reference):
    runs = {arm: inspect_run(root / f'{dataset}_s{seed}_{arm}', seed, arm, dataset) for arm in ARMS}
    reference, history, first_data, _ = runs['fixed']
    order = [row['minibatch_order_sha256'] for row in history]
    steps = [row['train_steps'] for row in history]
    config = reference['config']
    require(config['epochs'] == 8 and config['batch'] == 32, 'Unexpected pilot training budget')
    for split in SPLITS:
        for index, field in enumerate(('ids', 'labels')):
            values = first_data[split][index]
            require(np.array_equal(split_reference.setdefault((split, field), values), values),
                    f'{split} {field} differ across seeds')
    arm_metrics = {}
    for arm, (metadata, rows, data, metrics) in runs.items():
        for key in MATCHED_FIELDS:
            require(reference[key] == metadata[key], f'{seed} {arm} mismatched {key}')
        require(order == [row['minibatch_order_sha256'] for row in rows] and
                steps == [row['train_steps'] for row in rows], f'{seed} {arm} different update budget/order')
        for split in SPLITS:
            positions = data[split][3]
            delta = np.abs(positions[:, 0] - positions[:, 1])
            overlap = np.maximum(0, COPY_SIDE - delta).prod(axis=1) / COPY_SIDE**2
            metrics['splits'][split]['routing']['max_overlap_fraction'] = float(overlap.max())
            if arm == 'constrained':
                require(np.all(overlap <= 0.25 + 1e-6), f'{seed} {arm} {split} exceeds footprint overlap cap')
        arm_metrics[arm] = metrics
    pair_metrics = {
        f'{second}_minus_{first}': {
            split: paired(runs[first][2][split], runs[second][2][split]) for split in SPLITS
        } for first, second in combinations(ARMS, 2)
    }
    return {
        'seed': seed, 'source_commit': reference['source_commit'],
        'common_initial_state_sha256': reference['initial_state_sha256'],
        'common_parameter_count': reference['parameter_count'],
        'minibatch_order_sha256_per_epoch': order,
        'train_steps_per_arm': sum(steps),
        'train_presentations_per_arm': config['expected_counts']['train'] * config['epochs'],
        'budget_validation_passes_per_arm': len(history), 'post_lock_test_passes_per_arm': 1,
        'data': reference['data'], 'arms': arm_metrics, 'paired': pair_metrics,
    }, {k: v for k, v in config.items() if k != 'seed'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=DATASETS, required=True)
    parser.add_argument('--runs', type=Path, required=True, help='Parent of {dataset}_s{seed}_{arm} directories')
    parser.add_argument('--seeds', nargs='+', type=int, default=[37, 53])
    parser.add_argument('--out', type=Path, required=True, help='Directory for results.json')
    args = parser.parse_args()
    require(len(args.seeds) == len(set(args.seeds)), 'Seeds must be distinct')
    reference_ids = {}
    entries, protocols = [], []
    for seed in args.seeds:
        entry, protocol = audit_seed(args.runs, args.dataset, seed, reference_ids)
        entries.append(entry)
        protocols.append(protocol)
    require(all(p == protocols[0] for p in protocols), 'Configuration differs across seeds')
    require(len({r['source_commit'] for r in entries}) == 1, 'Source versions differ across seeds')
    summary = {'dataset': DATASETS[args.dataset][0], 'config': protocols[0], 'runs': entries,
               'mean_over_seeds': {}, 'interpretation': 'Each seed sees the same original test set.'}
    for split in SPLITS:
        mean = {}
        for arm in ARMS:
            values = [r['arms'][arm]['splits'][split] for r in entries]
            mean[arm + '_accuracy'] = float(np.mean([v['accuracy'] for v in values]))
            mean[arm + '_mean_overlap_fraction'] = float(np.mean([
                v['routing']['mean_overlapping_area_fraction_of_one_copy'] for v in values]))
        for name in entries[0]['paired']:
            mean[name + '_pp'] = float(np.mean([r['paired'][name][split]['delta_pp'] for r in entries]))
        summary['mean_over_seeds'][split] = mean
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'results.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary['mean_over_seeds']))


if __name__ == '__main__':
    main()
