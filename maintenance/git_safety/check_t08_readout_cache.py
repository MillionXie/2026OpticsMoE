"""Verify adopted T08 original cache/sample binding; no model, ranking or CCD replay."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import random


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''): h.update(chunk)
    return h.hexdigest()


def selected_indexes(rows, split):
    if split not in ('train', 'test'): raise ValueError('Unknown split')
    if split == 'test':
        if len(rows) != 2400: raise ValueError('Original TEST must have 2400 rows')
        return list(range(2400))
    groups = {i: [] for i in range(100)}
    for index, row in enumerate(rows): groups[int(row['label'])].append(index)
    if any(len(group) != 48 for group in groups.values()):
        raise ValueError('Original TRAIN must have 48 rows per product')
    rng = random.Random(20260926)
    return sorted(index for group in groups.values() for index in rng.sample(group, 8))


def inspect(data_root, train_cache, test_cache, evidence):
    data_root = Path(data_root)
    if set(evidence['dataset_csv_sha256']) != {'train', 'test', 'titles'}:
        raise ValueError('Incomplete original CSV identity manifest')
    if sorted(r['split'] for r in evidence['caches']) != ['test', 'train']:
        raise ValueError('Exactly one TRAIN and TEST cache required')
    paths = {'train': Path(train_cache), 'test': Path(test_cache)}
    rows = {}
    for split, expected in evidence['dataset_csv_sha256'].items():
        path = data_root / (split + '.csv')
        if digest(path) != expected: raise ValueError('Original CSV SHA mismatch: ' + split)
        with path.open(encoding='utf8') as f: rows[split] = list(csv.DictReader(f))
    if [int(r['label']) for r in rows['titles']] != list(range(100)):
        raise ValueError('Original title order mismatch')
    # Validate identities before loading serialized tensors. Only sealed originals accepted.
    for record in evidence['caches']:
        if digest(paths[record['split']]) != record['cache_sha256']:
            raise ValueError('Original cache SHA mismatch: ' + record['split'])
    import torch
    identities = {}; results = []
    for record in evidence['caches']:
        split = record['split']; path = paths[split]
        cache = torch.load(path, map_location='cpu', weights_only=False)
        indexes = selected_indexes(rows[split], split)
        prefix = 'train_' if split == 'train' else 'image_'
        keys = [prefix + f'{i:04d}' for i in indexes]
        if cache['split'] != split or cache['checkpoint_sha256'] != evidence['body_sha256']:
            raise ValueError('Cache body or split mismatch')
        if cache['image_keys'] != keys or cache['title_keys'] != [f'title_{i:03d}' for i in range(100)]:
            raise ValueError('Cache sample order mismatch')
        if split == 'train' and (cache['selection_seed'], cache['train_per_sku']) != (20260926, 8):
            raise ValueError('TRAIN selection contract mismatch')
        if cache['image_labels'] != [int(rows[split][i]['label']) for i in indexes]:
            raise ValueError('Cache labels mismatch')
        if any(rows[split][i]['product_id'] != rows['titles'][int(rows[split][i]['label'])]['product_id'] for i in indexes):
            raise ValueError('Product mapping mismatch')
        identities[split] = [rows[split][i]['sample_id'] for i in indexes]
        if len(set(identities[split])) != len(indexes): raise ValueError('Repeated sample identity')
        for key, count in (('image_features', len(indexes)), ('title_features', 100)):
            tensor = cache[key]
            if list(tensor.shape) != [count, 384] or tensor.dtype != torch.float32 or not torch.isfinite(tensor).all():
                raise ValueError('Invalid original feature tensor: ' + key)
            if hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest() != record['tensor_hashes'][key]:
                raise ValueError('Original feature tensor SHA mismatch: ' + key)
        if digest(path) != record['cache_sha256']: raise RuntimeError('Cache changed during inspection')
        results.append({'split': split, 'images': len(indexes), 'cache_sha256': record['cache_sha256']})
    if set(identities['train']) & set(identities['test']): raise ValueError('TRAIN/TEST sample overlap')
    for record in evidence['caches']:
        if digest(paths[record['split']]) != record['cache_sha256']:
            raise RuntimeError('Cache changed before inspection completed: ' + record['split'])
    for split, expected in evidence['dataset_csv_sha256'].items():
        if digest(data_root / (split + '.csv')) != expected:
            raise RuntimeError('CSV changed during inspection: ' + split)
    return dict(caches=results, read_only=True, model_loaded=False, ranking_computed=False,
                devices_opened=False, ccd_to_features_numerical_reconstruction_proven=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--train-cache', type=Path, required=True)
    p.add_argument('--test-cache', type=Path, required=True)
    args = p.parse_args()
    manifest = Path(__file__).resolve().parents[1] / 'storage/T08_READOUT_CACHE_BINDING_20261006.json'
    print(json.dumps(inspect(args.data_root, args.train_cache, args.test_cache,
                             json.loads(manifest.read_text(encoding='utf8'))), indent=2))


if __name__ == '__main__': main()
