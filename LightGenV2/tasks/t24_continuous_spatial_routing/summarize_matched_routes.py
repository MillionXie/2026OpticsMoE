"""Aggregate the locked three-arm classification and VAL routing interventions."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .compare import DATASETS, read_json, require
from .compare_matched import ARMS


TASK = Path(__file__).resolve().parent
SEEDS = (37, 53)


def inspect_audit(trial, mode, directory, dataset, expected_val):
    arm = trial['arms'][mode]
    path = directory / 'result.json'
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    require(read_json(directory / 'manifest.json')['result.json'] == digest, f'{path} hash mismatch')
    require(read_json(directory / 'status.json')['state'] == 'complete', f'{path} incomplete')
    audit = read_json(path)
    require((audit['dataset'], audit['mode'], audit['seed'], audit['split']) ==
            (dataset, mode, trial['seed'], 'val'), f'{path} wrong data/model')
    require(audit['optimization_steps'] == 0 and audit['selected_epoch'] == arm['selected_epoch']
            and audit['best_checkpoint_sha256'] == arm['best_checkpoint_sha256']
            and audit['data_manifest_sha256'] == trial['data']['manifest_sha256']
            and audit['trained_source_commit'] == trial['source_commit'], f'{path} wrong checkpoint')
    require([r['seed'] for r in audit['shot_noise']] == [1069, 1999] and
            [r['permutation_seed'] for r in audit['shuffled']] == [3701, 4423, 5701], 'Unexpected replicates')
    clean = audit['clean']
    require(clean['correct'] == arm['splits']['val']['correct'], 'Clean validation accuracy mismatch')
    for condition in [clean, *audit['shot_noise'], *audit['shuffled'], audit['router_tile_shift'], audit['integer_actuator']]:
        require(condition['n'] == expected_val and np.isclose(condition['correct'] / expected_val, condition['accuracy']),
                'Validation count mismatch')
    jitter = audit['independent_same_image_noise_pair']
    clean_usage = clean['usage']
    test = arm['splits']['test']
    scores = {
        'test_accuracy_pct': 100 * test['accuracy'],
        'test_mean_footprint_overlap_fraction': test['routing']['mean_overlapping_area_fraction_of_one_copy'],
        'test_max_footprint_overlap_fraction': test['routing']['max_overlap_fraction'],
        'clean_val_accuracy_pct': 100 * clean['accuracy'],
        'noise_val_accuracy_pct': [100 * r['accuracy'] for r in audit['shot_noise']],
        'same_image_noise_pair_distance_mean_px': jitter['minimum_assignment_mean_distance_px']['mean'],
        'same_image_noise_pair_distance_p90_px': jitter['minimum_assignment_mean_distance_px']['p90'],
        'same_image_noise_pair_fraction_over_16px': jitter['fraction_over_16px'],
        'different_image_distance_mean_px': float(np.mean([
            r['different_image_distance']['minimum_assignment_mean_distance_px']['mean'] for r in audit['shuffled']])),
        'shuffled_val_accuracy_pct': [100 * r['accuracy'] for r in audit['shuffled']],
        'shuffled_minus_clean_pp': [r['paired_delta_pp'] for r in audit['shuffled']],
        'clean_mean_effective_illuminated_pixels': clean_usage['mean_per_image_effective_illuminated_pixels'],
        'clean_population_effective_illuminated_pixels': clean_usage['population_effective_illuminated_pixels'],
        'clean_mean_power_fraction_in_old_cross_gap': clean_usage['mean_power_fraction_in_original_cross_gap'],
    }
    for name in ('router_tile_shift', 'integer_actuator'):
        entry = audit[name]
        scores[name + '_val_accuracy_pct'] = 100 * entry['accuracy']
        scores[name + '_minus_clean_pp'] = entry['paired_delta_pp']
        scores[name + '_mean_position_change_px'] = entry['vs_clean']['minimum_assignment_mean_distance_px']['mean']
    return {'test_correct': test['correct'], 'test_n': test['n'], 'val_n': expected_val, 'metrics': scores,
            'best_checkpoint_sha256': audit['best_checkpoint_sha256'], 'audit_result_sha256': digest,
            'photons_per_unit_incident_energy': audit['photons_per_unit_incident_energy']}


def summarize(report_root, audit_root):
    summary = {
        'seeds': list(SEEDS), 'noise_seeds': [1069, 1999], 'permutation_seeds': [3701, 4423, 5701],
        'averaging': 'Equal-weight mean over the two training seeds; list entries retain intervention replicate order. '
                     'The p90 reported across seeds is the mean of the two within-seed percentiles.',
        'distance': 'Minimum of the two assignments of the unordered copy pair. Each assignment uses the mean '
                    'of the two Euclidean top-left displacements measured in expert-plane pixels.',
        'jitter': 'Same image: two independent Poisson router CCD acquisitions. Different images: '
                  'clean placements compared with three validation-index derangements.',
        'accuracy': 'Percent; intervention changes in percentage points relative to the same locked clean VAL.',
        'footprint_overlap': 'Fraction of one 224 x 224 footprint area shared by the two placements on TEST.',
        'datasets': {},
    }
    budgets = set()
    for name, (title, counts) in DATASETS.items():
        compared = read_json(report_root / name / 'results.json')
        require(compared['dataset'] == title and sorted(r['seed'] for r in compared['runs']) == list(SEEDS),
                'Unexpected dataset or training seeds')
        dataset = {}
        for mode in ARMS:
            entries = {}
            for trial in compared['runs']:
                seed = trial['seed']
                folder = audit_root / f'{name}_s{seed}_{mode}'
                entries[str(seed)] = inspect_audit(trial, mode, folder, title.lower(), counts[1])
                budgets.add(entries[str(seed)]['photons_per_unit_incident_energy'])
            means = {key: np.mean([r['metrics'][key] for r in entries.values()], axis=0).tolist()
                     for key in entries[str(SEEDS[0])]['metrics']}
            dataset[mode] = {'per_training_seed': entries, 'mean_over_training_seeds': means}
        summary['datasets'][title] = dataset
    require(len(budgets) == 1, 'Poisson dose must match across every model')
    summary['photons_per_unit_incident_energy'] = budgets.pop()
    return summary


def plot(summary, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.fonttype': 'none', 'svg.hashsalt': 't24-matched-routes'})
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 5.8), layout='constrained')
    x = np.arange(len(ARMS))
    labels = ['Four corners', 'Continuous', 'Constrained']
    seed_colors = ['#345690', '#af6128']
    for row, (dataset, variants) in enumerate(summary['datasets'].items()):
        accuracy, jitter, shuffle = axes[row]
        for i, seed in enumerate(summary['seeds']):
            points = [variants[m]['per_training_seed'][str(seed)]['metrics'] for m in ARMS]
            accuracy.plot(x, [p['test_accuracy_pct'] for p in points], 'o-', markersize=4,
                          linewidth=1.4, color=seed_colors[i], label=f'Seed {seed}')
            offset = (i - 0.5) * 0.11
            shuffle.plot(x + offset, [np.mean(p['shuffled_minus_clean_pp']) for p in points],
                         'o', markersize=5, color=seed_colors[i], label=f'Seed {seed}')
        means = [variants[m]['mean_over_training_seeds'] for m in ARMS]
        for key, offset, marker, color, label in (
                ('same_image_noise_pair_distance_mean_px', -.065, 'o', '#00786b', 'Same image, two reads'),
                ('different_image_distance_mean_px', .065, 's', '#635389', 'Different images')):
            jitter.plot(x + offset, [p[key] for p in means], marker, markersize=5, color=color, label=label)
        accuracy.set_ylabel(f'{dataset}\nAccuracy (%)')
        jitter.set_ylabel('Mean distance (px)')
        jitter.set_yscale('symlog', linthresh=1.)
        jitter.set_yticks([0, 1, 10, 100], labels=['0', '1', '10', '100'])
        jitter.set_ylim(bottom=-.15)
        shuffle.axhline(0, color='#555555', linewidth=.8)
        shuffle.set_ylabel('Shuffled − clean (pp)')
        for ax in axes[row]:
            ax.set_xticks(x, labels, rotation=14, ha='right')
            ax.set_xlim(-.32, 2.32)
            ax.grid(axis='y', alpha=.18)
    axes[0, 0].set_title('Classification on TEST')
    axes[0, 1].set_title('Placement displacement on VAL')
    axes[0, 2].set_title('Position shuffling on VAL')
    axes[0, 0].legend(frameon=False, fontsize=8)
    axes[0, 1].legend(frameon=False, fontsize=8)
    fig.savefig(path, metadata={'Date': None})
    plt.close(fig)
    # Matplotlib SVG embeds whitespace on some marker paths.
    lines = path.read_text(encoding='utf-8').splitlines()
    path.write_text('\n'.join(line.rstrip() for line in lines) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reports', type=Path, default=TASK / 'reports/matched_20261128')
    parser.add_argument('--audits', type=Path, default=TASK / 'runs/smoke/matched_20261128')
    args = parser.parse_args()
    summary = summarize(args.reports, args.audits)
    path = args.reports / 'route_diagnostics.json'
    path.write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    plot(summary, args.reports / 'route_diagnostics.svg')


if __name__ == '__main__':
    main()
