"""CPU-only paired ABO gallery-view-count audit using pinned feature caches.

All120 products/all480 queries remain. One/three/twelve enrollment photos per
product are selected by hash BEFORE scoring. This is a NEW acquisition-budget
condition, not an improvement to the original12-view result or new training.
"""
import argparse
import hashlib
from pathlib import Path
import torch
from torch.nn import functional as F
from .data import _load_contract, GalleryItem, _evaluate, _category_prototypes
from .io import sha256, source_commit, write_json, write_csv


def view_indices(samples, count):
    if count not in (1, 3, 12):
        raise ValueError('Only predeclared 1/3/12 views supported')
    groups = {}
    for i, s in enumerate(samples):
        groups.setdefault(s.product_id, []).append(i)
    selected = []
    for key in sorted(groups):
        indices = groups[key]
        if len(indices) != 12:
            raise ValueError('Require original12 views per TRAIN product')
        order = sorted(indices, key=lambda i: hashlib.sha256(
            ('abo-enrollment42:' + samples[i].sample_id).encode()).hexdigest())
        selected.extend(order[:count])
    return selected


def budget_centroids(samples, vectors, count):
    """Separate new-budget aggregator: NEVER relax original12-view contract."""
    groups = {}
    if len(samples) != len(vectors):
        raise ValueError('Sample/vector count mismatch')
    for sample, vector in zip(samples, vectors):
        groups.setdefault(sample.product_id, []).append((sample, vector))
    centers, metadata = [], []
    for key in sorted(groups):
        values = groups[key]
        first = values[0][0]
        if len(values) != count or any(s.category_id != first.category_id for s, _ in values):
            raise ValueError('Wrong view budget or conflicting category')
        centers.append(F.normalize(torch.stack([v.float() for _, v in values]).mean(0), dim=0))
        metadata.append(GalleryItem(key, first.category_id, first.category_name, count))
    return F.normalize(torch.stack(centers), dim=1), metadata


def instance_rows(samples):
    """Different task: same-SKU heldout product retrieval, all40 products retained."""
    groups = {}
    for i, s in enumerate(samples):
        groups.setdefault(s.product_id, []).append(i)
    rows, indices = [], []
    for key in sorted(groups):
        if len(groups[key]) != 12:
            raise ValueError('Expected12 views per heldout product')
        order = sorted(groups[key], key=lambda i: hashlib.sha256(
            ('abo-instance42:' + samples[i].sample_id).encode()).hexdigest())
        for rank, i in enumerate(order):
            indices.append(i)
            rows.append(dict(sample_id=samples[i].sample_id, product_id=key,
                             split='gallery' if rank < 4 else 'query'))
    return rows, indices


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(4)
    samples, _ = _load_contract(args.data)
    train = [s for s in samples if s.split == 'train']
    test = [s for s in samples if s.split == 'test']
    optical = torch.load(args.optical_cache, map_location='cpu', weights_only=True)
    qwen = torch.load(args.qwen_cache, map_location='cpu', weights_only=True)
    ids = [s.sample_id for s in train + test]
    if optical['train_ids'] + optical['test_ids'] != ids or qwen['identity']['ids'] != ids:
        raise ValueError('Cache image identity/order mismatch')
    manifest_sha = sha256(args.data / 'data/abo_similarity10_manifest.csv')
    if qwen['identity']['manifest_sha256'] != manifest_sha:
        raise ValueError('Qwen cache has another data contract')
    # Native-aspect frozen Qwen64, not a fitted PCA or weaker baseline head.
    vectors = dict(optical=torch.cat([optical['train'], optical['test']]), qwen64=qwen['native'][:, :64])
    for z in vectors.values():
        if z.shape != (1920, 64) or not torch.isfinite(z).all() or (z.norm(dim=1) < 1e-8).any():
            raise ValueError('Invalid64D feature cache')
    args.output.mkdir(parents=True)
    results, enrollment = {}, {}
    for count in (12, 3, 1):
        indices = view_indices(train, count)
        enrollment[str(count)] = [train[i].sample_id for i in indices]
        results[str(count)] = {}
        for name, z in vectors.items():
            z = F.normalize(z.float(), dim=1)
            gallery, metadata = budget_centroids([train[i] for i in indices], z[indices], count)
            metrics, predictions, _ = _evaluate(z[len(train):], test, gallery, metadata, _category_prototypes(gallery, metadata))
            results[str(count)][name] = metrics
            write_csv(args.output / f'{name}_views{count}_predictions.csv', predictions)
        results[str(count)]['gap_pp'] = 100 * (results[str(count)]['qwen64']['hit_at_1'] - results[str(count)]['optical']['hit_at_1'])
    from .retrieval_screen import rank_instances
    rows, indices = instance_rows(test)
    instance = {}
    for name, z in vectors.items():
        instance[name], predictions = rank_instances(z[len(train):][indices], rows)
        write_csv(args.output / f'{name}_same_sku_predictions.csv', predictions)
    instance['gap_pp'] = 100 * (instance['qwen64']['hit_at_1'] - instance['optical']['hit_at_1'])
    write_json(args.output / 'report.json', dict(source_commit=source_commit(), status='complete',
        purpose=__doc__, train_manifest_sha256=manifest_sha, training=False, gpu=False,
        optical_cache_sha256=sha256(args.optical_cache), qwen_cache_sha256=sha256(args.qwen_cache),
        selection='Nested hash-selected1/3/12 gallery images; all120 products/all480 queries; no class filtering',
        caveat='Optical model was trained/selected under original12-view protocol; caches preserve each model original preprocessing. New gallery budget conditions must not replace original metric.',
        enrollment=enrollment, results=results, same_sku_instance=instance, instance_rows=rows,
        instance_note='Separate task:40 heldout products,4 gallery+8 query photos/product fixed by sha256(abo-instance42:<sample_id>),160-gallery/320-query. Same SKU relevant, not same category. All products retained, no score-based selection, no new training. Does not replace original task.'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('data', 'optical-cache', 'qwen-cache', 'output'):
        p.add_argument('--' + key, type=Path, required=True)
    run(p.parse_args())


if __name__ == '__main__':
    main()
