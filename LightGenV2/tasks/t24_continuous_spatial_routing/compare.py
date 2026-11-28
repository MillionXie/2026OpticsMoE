"""Compare both matched Kather runs per seed and plot TEST placement densities."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


MODES = ('fixed', 'continuous')
SPLITS = ('train', 'val', 'test')
EXTENT = 254
COPY_SIDE = 224


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def location_metrics(positions):
    locations = positions.reshape(-1, 2)
    edge_dist = np.minimum(locations, EXTENT - locations)
    corner_dist = np.linalg.norm(edge_dist, axis=1)
    separation = np.abs(positions[:, 0] - positions[:, 1])
    overlap = np.maximum(COPY_SIDE - separation, 0).prod(1) / COPY_SIDE**2
    # A pair is unordered. Swapping the first and second peak must not
    # inflate the measured diversity of placements.
    grid = np.rint(positions).astype(int)
    keys = grid[:, :, 0] * (EXTENT + 1) + grid[:, :, 1]
    _, pair_counts = np.unique(np.sort(keys, axis=1), axis=0, return_counts=True)
    return {
        'distinct_positions_at_nearest_pixel': int(np.unique(keys).size),
        'distinct_unordered_pairs_at_nearest_pixel': int(len(pair_counts)),
        'most_common_pair_image_fraction': float(pair_counts.max() / len(positions)),
        'strict_interior_placement_fraction': float(np.mean(edge_dist.min(1) > 1e-4)),
        'mean_distance_from_nearest_fixed_corner_px': float(corner_dist.mean()),
        'placement_fraction_over_16px_from_any_fixed_corner': float((corner_dist > 16).mean()),
        'mean_pair_distance_px': float(np.linalg.norm(separation, axis=1).mean()),
        'any_footprint_overlap_image_fraction': float((overlap > 0).mean()),
        'mean_overlapping_area_fraction_of_one_copy': float(overlap.mean()),
    }


def inspect_run(run, seed, mode):
    manifest = read_json(run / 'manifest.json')
    required = {'metadata.json', 'history.json', 'selection_lock.json', 'result.json',
                'best_checkpoint.pt', 'last_checkpoint.pt', *(f'predictions_{s}.npz' for s in SPLITS)}
    require(required <= manifest.keys(), f'{run} incomplete manifest')
    for filename in required:
        actual = hashlib.sha256((run / filename).read_bytes()).hexdigest()
        require(actual == manifest[filename], f'{run}/{filename} SHA mismatch')
    metadata = read_json(run / 'metadata.json')
    config = metadata['config']
    history = read_json(run / 'history.json')
    lock = read_json(run / 'selection_lock.json')
    result = read_json(run / 'result.json')
    require(read_json(run / 'status.json')['state'] == 'complete', f'{run} incomplete')
    require(metadata['mode'] == lock['mode'] == mode and config['seed'] == lock['seed'] == seed,
            f'{run} experiment identity')
    require(all(result[k] == v for k, v in lock.items()) and lock['TEST_selection_calls'] == 0
            and lock['source_commit'] == metadata['source_commit']
            and lock['best_checkpoint_sha256'] == manifest['best_checkpoint.pt'], f'{run} selection lock')
    counts = config['expected_counts']
    steps = (counts['train'] + config['batch'] - 1) // config['batch']
    require(len(history) == config['epochs'] and
            all(r['epoch'] == i + 1 and r['train_steps'] == steps for i, r in enumerate(history)),
            f'{run} epoch budget')
    best = max(history, key=lambda r: (r['validation']['correct'], -r['validation']['cross_entropy']))
    require(best['epoch'] == lock['selected_epoch'], f'{run} validation selection mismatch')
    data, metrics = {}, {}
    for split in SPLITS:
        with np.load(run / f'predictions_{split}.npz', allow_pickle=False) as z:
            ids, y, pred, scores, pos = (z[k] for k in ('ids', 'labels', 'predictions', 'scores', 'positions'))
        n = len(y)
        require(ids.shape == y.shape == pred.shape == (counts[split],) and np.unique(ids).size == n,
                f'{run} {split} incomplete or duplicate IDs')
        require(scores.shape == (n, config['num_classes']) and np.isfinite(scores).all()
                and np.array_equal(scores.argmax(1), pred), f'{run} {split} score mismatch')
        correct = y == pred
        k = int(correct.sum())
        require(result[split]['correct'] == k and result[split]['n'] == n, f'{run} {split} count mismatch')
        require(pos.shape == (n, 2, 2) and np.isfinite(pos).all() and pos.min() >= -1e-4
                and pos.max() <= EXTENT + 1e-4 and np.all(np.any(pos[:, 0] != pos[:, 1], axis=1)),
                f'{run} {split} invalid coordinates')
        if mode == 'fixed':
            require(np.logical_or(np.isclose(pos, 0), np.isclose(pos, EXTENT)).all(),
                    f'{run} fixed route outside corners')
        data[split] = (ids, y, correct, pos)
        metrics[split] = {'n': n, 'correct': k, 'accuracy': k / n, 'routing': location_metrics(pos)}
    return metadata, history, data, dict(run=run.name, selected_epoch=lock['selected_epoch'],
                                        best_checkpoint_sha256=lock['best_checkpoint_sha256'],
                                        initial_state_sha256=metadata['initial_state_sha256'], splits=metrics)


def pair_outcomes(fixed, continuous):
    require(np.array_equal(fixed[0], continuous[0]) and np.array_equal(fixed[1], continuous[1]),
            'Paired predictions must share IDs, order and labels')
    a, b = fixed[2], continuous[2]
    new_correct = int((~a & b).sum())
    lost_correct = int((a & ~b).sum())
    return dict(both_correct=int((a & b).sum()), both_wrong=int((~a & ~b).sum()),
                only_fixed_correct=lost_correct, only_continuous_correct=new_correct,
                delta_correct=new_correct - lost_correct,
                delta_pp=100 * (new_correct - lost_correct) / len(a))


def plot_positions(samples, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    histogram = {}
    for key, pos in samples.items():
        y, x = pos.reshape(-1, 2).T
        histogram[key] = np.histogram2d(y.clip(0, EXTENT), x.clip(0, EXTENT),
                                        bins=32, range=((0, EXTENT), (0, EXTENT)))[0]
    seeds = sorted({key[0] for key in histogram})
    norm = LogNorm(vmin=1, vmax=max(2, max(h.max() for h in histogram.values())))
    fig, axes = plt.subplots(len(seeds), 2, figsize=(8, 3.4 * len(seeds)), squeeze=False,
                             constrained_layout=True)
    for i, seed in enumerate(seeds):
        for j, mode in enumerate(MODES):
            ax = axes[i, j]
            matrix = histogram[seed, mode]
            im = ax.imshow(np.ma.masked_equal(matrix, 0), extent=(0, EXTENT, EXTENT, 0),
                           origin='upper', interpolation='nearest', cmap='viridis', norm=norm)
            ax.set(title=f'{mode.capitalize()}, seed {seed}', xlabel='Placement x (px)',
                   ylabel='Placement y (px)', xticks=[0, 127, 254], yticks=[0, 127, 254])
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=.65, label='Selected positions per bin (log scale)')
    fig.suptitle('Kather2016 TEST placements (two per image)')
    fig.savefig(out)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, required=True, help='Parent of kather_s{seed}_{mode}')
    parser.add_argument('--out', type=Path, required=True, help='Report directory')
    parser.add_argument('--seeds', type=int, nargs='+', default=[17, 29])
    args = parser.parse_args()
    require(len(set(args.seeds)) == len(args.seeds), 'Seeds must be distinct')
    pairs, samples, reference_ids = [], {}, {}
    for seed in args.seeds:
        runs = [inspect_run(args.runs / f'kather_s{seed}_{mode}', seed, mode) for mode in MODES]
        a, b = runs
        for field in ('initial_state_sha256', 'config', 'config_file_sha256', 'source_commit',
                      'source_sha256', 'parameter_count', 'data', 'selection'):
            require(a[0][field] == b[0][field], f'Seed {seed} mismatched paired {field}')
        require([r['minibatch_order_sha256'] for r in a[1]] == [r['minibatch_order_sha256'] for r in b[1]],
                f'Seed {seed} mismatched shuffle order')
        for split in SPLITS:
            ids, y = a[2][split][:2]
            prev_ids, prev_y = reference_ids.setdefault(split, (ids, y))
            require(np.array_equal(ids, prev_ids) and np.array_equal(y, prev_y), 'Cross-seed split mismatch')
        pairs.append({'seed': seed, 'source_commit': a[0]['source_commit'],
                      'train_steps_per_mode': sum(r['train_steps'] for r in a[1]),
                      'fixed': a[3], 'continuous': b[3],
                      'paired': {split: pair_outcomes(a[2][split], b[2][split]) for split in SPLITS}})
        for mode, run in zip(MODES, runs):
            samples[seed, mode] = run[2]['test'][3]
    summary = {'dataset': 'Kather2016', 'protocol': 'TRAIN fit, VAL checkpoint, original TEST once per run',
               'coordinate_convention': 'top-left y,x in the 478-pixel active plane',
               'paired_difference': 'continuous minus fixed, same original cases within each seed',
               'runs': pairs, 'mean_over_seeds': {}}
    for split in SPLITS:
        row = {mode + '_accuracy': float(np.mean([p[mode]['splits'][split]['accuracy'] for p in pairs]))
               for mode in MODES}
        row['delta_pp'] = float(np.mean([p['paired'][split]['delta_pp'] for p in pairs]))
        summary['mean_over_seeds'][split] = row
    args.out.mkdir(parents=True, exist_ok=True)
    plot_positions(samples, args.out / 'positions.svg')
    (args.out / 'results.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary['mean_over_seeds']))


if __name__ == '__main__':
    main()
