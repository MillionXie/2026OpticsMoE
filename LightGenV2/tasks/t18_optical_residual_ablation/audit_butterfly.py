"""CPU-only audit of saved full60 predictions/checkpoints; no optical inference."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
assert os.environ['CUDA_VISIBLE_DEVICES'] == '', 'Audit must not have a GPU visible'
import butterfly_ablation as a


def close(actual, expected, key):
    if isinstance(expected, (list, dict)):
        assert actual == expected, key
    elif isinstance(expected, (float, int)):
        assert abs(actual - expected) <= 1e-8, (key, actual, expected)
    else:
        assert actual == expected, key


def prediction_audit(folder, data, manifest, split, expected):
    with (folder / (split + '_predictions.csv')).open() as stream:
        rows = list(csv.DictReader(stream))
    with a.np.load(data, allow_pickle=False) as z:
        labels = z[split + '_labels'].copy()
        ids = z[split + '_ids'].tolist()
    assert [x['sample_id'] for x in rows] == ids == manifest['split_ids'][split]
    assert len(set(ids)) == len(ids) and len(rows) == len(labels)
    assert [int(x['label_true']) for x in rows] == labels.tolist()
    scores = a.np.array([[float(x[f'score{k}']) for k in range(60)] for x in rows])
    assert a.np.isfinite(scores).all() and a.np.all(scores >= 0)
    assert [int(x['label_pred']) for x in rows] == scores.argmax(1).tolist()
    recomputed = a.metrics(labels, scores)
    for key, value in recomputed.items():
        close(expected[key], value, split + '.' + key)
    return dict(records=len(rows), prediction_sha256=a.r.sha(folder / (split + '_predictions.csv')),
                ids_labels_argmax_support_cm_nll_macro_verified=True)


def audit_arm(folder, suite, data, manifest, final):
    meta = a.r.read(folder / 'metadata.json')
    result = a.r.read(folder / ('result.json' if final else 'fit_result.json'))
    assert result['sources'] == meta['sources']
    history = a.r.read(folder / 'history.json')
    gradients = a.r.read(folder / 'gradients.json')
    best = a.torch.load(folder / 'best_checkpoint.pt', map_location='cpu', weights_only=False)
    last = a.torch.load(folder / 'last_checkpoint.pt', map_location='cpu', weights_only=False)
    assert len(history) == result['epochs_completed'] == last['epoch']
    assert len(history) in (100, 150, 200)
    assert [row['epoch'] for row in history] == list(range(1, len(history) + 1))
    assert all('test' not in str(row.keys()).lower() for row in history)
    selected = min(history, key=lambda row: row['val']['balanced_nll'])
    assert selected['epoch'] == result['selected_epoch'] == best['epoch']
    assert selected['val'] == result['selected_validation'] == best['validation']
    assert best['selected_state_kind'] == result['selected_state_kind'] == 'ema'
    assert best['selection_split'] == result['selection_split'] == 'val'
    for ck in (best, last):
        assert ck['config'] == meta['config'] and ck['sources'] == meta['sources']
        assert ck['data_sha256'] == result['data_sha256'] == manifest['cache_sha256']
        assert ck['depth'] == result['depth']
    assert a.r.sha(folder / 'best_checkpoint.pt') == result['checkpoint_sha256']
    parameters = {name: tensor for name, tensor in best['model'].items() if name.endswith('raw_phase')}
    assert sum(t.numel() for t in parameters.values()) == result['parameters'] == 10000 + 439848 * (result['depth'] // 2)
    assert len(parameters) == 1 + 5 * result['depth']
    assert len(gradients) == len(history) and all(set(g['norms']) == set(parameters) for g in gradients)
    assert all(a.np.isfinite(v) and v > 0 for g in gradients for v in g['norms'].values())
    assert result['orders'] == last['orders'] and result['transforms'] == last['transforms']
    assert result['initial_parameter_sha256'] == last['initial_parameter_sha256']
    assert len(result['orders']) == len(result['transforms']) == len(history)
    for name, digest in meta['sources'].items():
        assert a.r.sha(a.ROOT / name) == digest, ('Runtime source changed', name)
        blob = subprocess.check_output(['git', 'show', meta['git_commit'] + ':' + name], cwd=a.ROOT)
        assert hashlib.sha256(blob).hexdigest() == digest, ('Published source mismatch', name)
    predictions = {split: prediction_audit(folder, data, manifest, split, result['metrics'][split])
                   for split in ('train', 'val')}
    if final:
        lock = a.r.read(folder / 'selection_lock.json')
        global_lock = a.r.read(suite / 'global_selection_lock.json')
        started = a.r.read(folder / 'test_started.json')
        assert result['test_evaluations'] == 1 and result['test_used_for_selection'] is False
        assert lock['test_evaluations_before_lock'] == global_lock['test_evaluations_before_lock'] == 0
        assert len(global_lock['selected_arms']) == 8 and global_lock['selected_arms'][str(folder.resolve())] == lock
        assert lock['checkpoint_sha256'] == result['checkpoint_sha256'] == started['checkpoint_sha256']
        assert lock['epoch'] == best['epoch'] and lock['config'] == meta['config'] and lock['sources'] == meta['sources']
        assert started['selection_lock_sha256'] == a.r.sha(folder / 'selection_lock.json')
        assert started['global_lock_sha256'] == a.r.sha(suite / 'global_selection_lock.json')
        assert (suite / 'global_selection_lock.json').stat().st_mtime <= (folder / 'test_started.json').stat().st_mtime
        predictions['test'] = prediction_audit(folder, data, manifest, 'test', result['metrics']['test'])
    else:
        assert result['test_evaluations'] == 0 and 'test' not in result['metrics']
    compact = []
    for row in history:
        item = dict(epoch=row['epoch'], train_loss=row['train_loss'], lr=row['lr'])
        for split in ('train', 'val'):
            if split in row:
                item[split] = {key: row[split][key] for key in ('accuracy', 'balanced_accuracy', 'macro_f1', 'nll', 'balanced_nll', 'detector_capture')}
        compact.append(item)
    keys = ('depth', 'rho', 'candidate', 'parameters', 'initial_parameter_sha256', 'epochs_completed', 'selected_epoch',
            'selected_state_kind', 'selected_validation', 'metrics', 'checkpoint_sha256', 'data_sha256', 'seconds',
            'selection_split', 'test_evaluations', 'test_used_for_selection', 'fit_diagnostic')
    return dict(schema='butterfly60_validation_v1', run_id=suite.name, result={key: result[key] for key in keys},
        history=compact, predictions=predictions, source_count=len(meta['sources']), source_commit=meta['git_commit'],
        sources_mapping_sha256=hashlib.sha256(json.dumps(meta['sources'], sort_keys=True).encode()).hexdigest(),
        checkpoint_last_sha256=a.r.sha(folder / 'last_checkpoint.pt'), audit_passed=True,
        audit_scope='Saved CPU tensors, labels and predictions only; no GPU or optical inference; no repeated test')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--suite', type=Path, required=True)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--reports', type=Path, required=True)
    p.add_argument('--fit-only', action='store_true')
    args = p.parse_args()
    data = args.data_root / 'butterfly_fixed_split.npz'
    manifest = a.r.read(args.data_root / 'data_manifest.json')
    assert a.r.sha(data) == manifest['cache_sha256'] and manifest['total_records'] == 10013 and len(manifest['classes']) == 60
    assert a.r.sha(args.data_root / 'group_identity.json') == manifest['group_identity_sha256']
    groups = a.r.read(args.data_root / 'group_identity.json')
    by_id = {row['name']: row for row in groups}
    for kind in ('specimen', 'original_hash', 'cache_hash'):
        sets = [{by_id[sid][kind] for sid in manifest['split_ids'][split]} for split in ('train', 'val', 'test')]
        assert not any(sets[i] & sets[j] for i in range(3) for j in range(i))
    args.reports.mkdir(parents=True, exist_ok=True)
    audits = []
    if args.fit_only:
        folders = sorted(path.parent for path in args.suite.glob('*/L*/fit_result.json'))
    else:
        lock = a.r.read(args.suite / 'global_selection_lock.json')
        folders = [Path(path) for path in lock['selected_arms']]
        assert len(folders) == 8
    for folder in folders:
        audit = audit_arm(folder, args.suite, data, manifest, not args.fit_only)
        x = audit['result']
        label = f"L{x['depth']}_rho{x['rho']:.1f}_{x['candidate']}" + ('_fit' if args.fit_only else '')
        a.r.save(args.reports / (label + '_audit.json'), audit)
        audits.append(audit)
    if not args.fit_only:
        for depth in [2, 4, 8, 16]:
            pair = [x['result'] for x in audits if x['result']['depth'] == depth]
            assert len(pair) == 2 and {x['rho'] for x in pair} == {0., .3}
            for key in ('candidate', 'epochs_completed', 'initial_parameter_sha256', 'data_sha256', 'parameters'):
                assert pair[0][key] == pair[1][key]
            paths = [Path(path) for path in a.r.read(args.suite / f'pair_lock_L{depth}.json')['paths']]
            original = [a.r.read(path / 'fit_result.json') for path in paths]
            for key in ('orders', 'transforms', 'sources'):
                assert original[0][key] == original[1][key]
    # Saved public benchmark identity, without copying data arrays or original images.
    a.r.save(args.reports / 'data_identity.json', {key: value for key, value in manifest.items() if key not in ('split_ids', 'license_rows_missing')})
    print(json.dumps(dict(audited_arms=len(audits), mode='fit' if args.fit_only else 'final', inference=False)))


if __name__ == '__main__':
    main()
