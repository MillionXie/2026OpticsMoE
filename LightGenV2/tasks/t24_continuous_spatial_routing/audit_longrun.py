"""Offline checkpoint, per-epoch matched budget, and realized route audit for T24."""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch

from . import train
from .compare import COPY_SIDE, DATASETS, SPLITS, inspect_run, read_json, require
from .compare_matched import ARMS
from .replay_epoch_curves import metrics


def load(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def verify_source(path, expected):
    # Windows checkout may add CRLF, while the archive records exact server bytes.
    b = path.read_bytes()
    if hashlib.sha256(b).hexdigest() == expected:
        return 'exact_bytes'
    require(hashlib.sha256(b.replace(b'\r\n', b'\n')).hexdigest() == expected,
            f'Archived source content differs even with Linux line endings: {path}')
    return 'matches_after_CRLF_to_LF'


def check_routes(path, mode, data, split_result):
    ids, y, good, positions = data
    separation = np.abs(positions[:, 0] - positions[:, 1])
    overlap = np.maximum(0, COPY_SIDE - separation).prod(1) / COPY_SIDE**2
    with np.load(path, allow_pickle=False) as pred:
        gate, capture = pred['power'], pred['router_capture']
        scores, classes = pred['scores'], pred['predictions']
    n = len(ids)
    require(gate.shape == (n, 2) and np.isfinite(gate).all() and (gate > 0).all()
            and np.allclose(gate.sum(1), 1., rtol=0., atol=1e-5), f'{path}: invalid selected gate powers')
    require(capture.shape == (n,) and np.isfinite(capture).all() and (capture >= 0.).all(),
            f'{path}: invalid captured optical power')
    require((scores >= 0).all() and np.allclose(scores.sum(1), 1., rtol=0., atol=1e-5),
            f'{path}: unnormalized class probabilities')
    nclasses = scores.shape[1]
    require((y >= 0).all() and (y < nclasses).all(), f'{path}: label out of range')
    confusion = np.bincount(y * nclasses + classes, minlength=nclasses**2).reshape(nclasses, nclasses)
    require(confusion.tolist() == split_result['confusion_matrix'] and
            metrics(split_result)['correct'] == int(good.sum()), f'{path}: locked metrics differ')
    if mode == 'fixed':
        require(np.max(overlap) == 0., f'{path}: fixed corner placements overlap')
    if mode == 'constrained':
        require(np.max(overlap) <= 0.25 + 1e-6, f'{path}: footprint cap exceeded')
    return {'n': n, 'correct': int(good.sum()), 'mean_footprint_overlap': float(overlap.mean()),
            'max_footprint_overlap': float(overlap.max()), 'min_selected_gate_power': float(gate.min()),
            'min_top_left_px': float(positions.min()), 'max_top_left_px': float(positions.max())}


def check_pt(best, last, mode, config, history, result, rng):
    e = result['selected_epoch']
    selected = last['selected_best']
    require(best['mode'] == selected['mode'] == last['mode'] == mode and best['epoch'] == selected['epoch'] == e
            and best['validation'] == selected['validation'] == history[e-1]['validation'], 'Invalid embedded best')
    require(last['epoch'] == len(history) and last['history'] == history and last['validation'] == history[-1]['validation'],
            'Latest checkpoint does not match the published full history')
    model = train.build_model(mode, config['num_classes'])
    require(sum(p.numel() for p in model.parameters()) == 513416, 'Unexpected optical model size')
    for pt in (best, last):
        model.load_state_dict(pt['model'], strict=True)
        require(len(pt['model']) == 4 and all(torch.isfinite(t).all() for t in pt['model'].values()),
                'Invalid stored model parameters')
    require(selected['model'].keys() == best['model'].keys()
            and all(torch.equal(t, selected['model'][k]) for k, t in best['model'].items()),
            'Published best differs from the atomically committed validation-best model')
    require(last['shuffle_rng_state'] == rng.bit_generator.state, 'Committed final shuffle state differs')
    optimizer = last['optimizer']
    groups, states = optimizer['param_groups'], optimizer['state']
    updates = sum(r['train_steps'] for r in history)
    require(len(groups) == 2 and [len(g['params']) for g in groups] == [3, 1]
            and [g['lr'] for g in groups] == [config['phase_lr'], config['head_lr']]
            and len(states) == 4, 'Stored phase/head Adam groups differ')
    for s in states.values():
        require(s['step'].item() == updates and torch.isfinite(s['exp_avg']).all()
                and torch.isfinite(s['exp_avg_sq']).all() and (s['exp_avg_sq'] >= 0).all(),
                'Adam step count or moments invalid')
    return updates


def audit_one(run, parent, dataset, mode, seed):
    metadata, history, arrays, observed = inspect_run(run, seed, mode, dataset)
    checksums = read_json(parent / 'manifest.json')
    parent_meta = read_json(parent / 'metadata.json')
    for name, digest in checksums.items():
        require(Path(name).name == name and train.sha256(parent / name) == digest, f'Original artifact differs: {name}')
    config, lineage = metadata['config'], metadata['lineage']
    result = read_json(run / 'result.json')
    earlier = read_json(parent / 'history.json')
    parent_cfg = (train.REPO / parent_meta['command'][parent_meta['command'].index('--config') + 1]).resolve()
    require(parent_cfg.is_relative_to(train.TASK / 'configs'), 'Original config path escapes the task')
    verify_source(parent_cfg, parent_meta['config_file_sha256'])
    require(train.sha256(run / 'config.json') == metadata['config_file_sha256'] and read_json(run / 'config.json') == config,
            'Recorded continuation config file differs')
    require(history[:len(earlier)] == earlier and metadata['parent_config'] == parent_meta['config']
            and config == {**parent_meta['config'], 'epochs': len(history)} and metadata['data'] == parent_meta['data']
            and metadata['initial_state_sha256'] == parent_meta['initial_state_sha256']
            and metadata['optimizer'] == parent_meta['optimizer'], 'Original trajectory or training config changed')
    require(lineage['original_manifest_sha256'] == train.sha256(parent / 'manifest.json')
            and lineage['parent_source_commit'] == parent_meta['source_commit']
            and all(lineage[f'original_{k}_sha256'] == checksums[name] for k, name in
                    [('best', 'best_checkpoint.pt'), ('last', 'last_checkpoint.pt'), ('history', 'history.json')]),
            'Original checkpoint, history, source or manifest differs')
    require(all(metadata['source_sha256'].get(k) == v for k, v in parent_meta['source_sha256'].items()),
            'Archived training dependencies changed between the first eight epochs and continuation')
    source_eol_adjusted = []
    for rel, digest in metadata['source_sha256'].items():
        path = (train.REPO / rel).resolve()
        require(path.is_relative_to(train.REPO / 'LightGenV2'), f'{run.name}: invalid source path: {rel}')
        if verify_source(path, digest) != 'exact_bytes':
            source_eol_adjusted.append(rel)
    rng = np.random.default_rng(seed)
    counts = config['expected_counts']
    for row in history:
        order = rng.permutation(counts['train']).astype(np.int64)
        require(hashlib.sha256(order.tobytes()).hexdigest() == row['minibatch_order_sha256'], 'Batch order differs')
        require(metrics(row['validation'])['n'] == counts['val'] and np.isfinite(row['train_loss']),
                'Invalid recorded validation or optimizer loss')
        if row['epoch'] > len(earlier):
            require(all(metrics(row[s])['n'] == counts[s] for s in ('train', 'test')), 'Incomplete diagnostic split')
    best, last = (load(run / f'{k}_checkpoint.pt') for k in ('best', 'last'))
    require(last['lineage'] == lineage, 'Last checkpoint binds a different parent')
    updates = check_pt(best, last, mode, config, history, result, rng)
    parent_last = load(parent / 'last_checkpoint.pt')['model']
    parameter_changes = {k: float((t - parent_last[k]).abs().max()) for k, t in last['model'].items()}
    require(any(v > 0 for v in parameter_changes.values()), 'Last model still equals archived E8 weights')
    selected = history[result['selected_epoch'] - 1]
    for s in SPLITS:
        if s == 'val' or s in selected:
            require(selected['validation' if s == 'val' else s]['correct'] == result[s]['correct'],
                    f'{s} selected epoch and locked prediction counts differ')
    audited = {}
    for s in SPLITS:
        with np.load(parent / f'predictions_{s}.npz', allow_pickle=False) as ref:
            require(np.array_equal(ref['ids'], arrays[s][0]) and np.array_equal(ref['labels'], arrays[s][1]),
                    f'{s} parent and continued sample IDs or order differ')
        audited[s] = check_routes(run / f'predictions_{s}.npz', mode, arrays[s], result[s])
    order_digest = hashlib.sha256(b''.join(bytes.fromhex(r['minibatch_order_sha256']) for r in history)).hexdigest()
    return {'name': run.name, 'dataset': dataset, 'mode': mode, 'seed': seed, 'epochs': len(history),
            'selected_epoch': result['selected_epoch'], 'parameter_updates': updates,
            'data_manifest_sha256': metadata['data']['manifest_sha256'],
            'data_index_sha256': metadata['data']['index_sha256'],
            'minibatch_order_sequence_sha256': order_digest, 'initial_state_sha256': metadata['initial_state_sha256'],
            'best_checkpoint_sha256': observed['best_checkpoint_sha256'], 'manifest_sha256': train.sha256(run / 'manifest.json'),
            'max_abs_parameter_change_since_parent': parameter_changes,
            'local_source_CRLF_normalized_paths': source_eol_adjusted, 'splits': audited}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parents', type=Path, required=True)
    p.add_argument('--runs', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(2)
    runs = []
    for dataset in DATASETS:
        dataset_start = len(runs)
        for seed in (37, 53):
            matched = []
            for mode in ARMS:
                name = f'{dataset}_s{seed}_{mode}'
                matched.append(audit_one(args.runs / name, args.parents / name, dataset, mode, seed))
            for key in ('minibatch_order_sequence_sha256', 'parameter_updates', 'epochs', 'initial_state_sha256'):
                require(len({run[key] for run in matched}) == 1, f'Different matched {key} in {dataset} seed {seed}')
            runs.extend(matched)
        require(all(len({r[k] for r in runs[dataset_start:]}) == 1 for k in ('data_manifest_sha256', 'data_index_sha256')),
                f'{dataset} data identity differs across seeds or routing modes')
    require(len({r['epochs'] for r in runs}) == 1, 'All twelve runs must have the same total budget')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    train.write(args.out, {'status': 'verified', 'scope': 'Offline checkpoint/state and archived per-image outputs, '
                           'including geometry and powers of both placements. No new inference or training.',
                           'audit_script_sha256': train.sha256(Path(__file__)), 'runs': runs,
                           'total_optimizer_updates': sum(r['parameter_updates'] for r in runs)})


if __name__ == '__main__':
    main()
