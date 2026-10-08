"""CPU-only fixed-cache error audit; never fits a transform or filters queries."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import torch

from ..standalone.io import sha256, source_commit, write_json
from ..standalone.retrieval_screen import rank_instances


def validate_protocol(rows):
    ids = [r['sample_id'] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate sample identity')
    products = defaultdict(list)
    for row in rows:
        if row['split'] not in ('train', 'query'):
            raise ValueError('Expected enrolled train/query protocol')
        angle = int(row['azimuth'])
        if str(angle) != str(row['azimuth']) or not 0 <= angle < 72:
            raise ValueError('Expected integer spin index in [0, 72)')
        products[row['product_id']].append(row)
    for views in products.values():
        if {r['split'] for r in views} != {'train', 'query'}:
            raise ValueError('Every query SKU needs TRAIN gallery')
        if len({int(r['azimuth']) for r in views}) != len(views):
            raise ValueError('Duplicate view index within SKU')
        if len({r['category_id'] for r in views}) != 1:
            raise ValueError('Inconsistent SKU category')


def rank_cache(rows, cache, manifest_sha):
    if cache['manifest_sha256'] != manifest_sha:
        raise ValueError('Cache manifest SHA mismatch')
    ids = cache['ids']
    lookup = {r['sample_id']: r for r in rows}
    if len(set(ids)) != len(ids) or set(ids) != set(lookup):
        raise ValueError('Cache sample identity mismatch')
    # Protocol order, not arbitrary cache order, determines stable gallery ties.
    index = {key: i for i, key in enumerate(ids)}
    ordered = [r for split in ('train', 'query') for r in rows if r['split'] == split]
    vectors = cache['vectors'][[index[r['sample_id']] for r in ordered]]
    ranking_rows = [dict(r, split='gallery' if r['split'] == 'train' else 'query') for r in ordered]
    metrics, predictions = rank_instances(vectors, ranking_rows)
    return metrics, {r['sample_id']: r for r in predictions}


def summarize(rows, optical, qwen):
    validate_protocol(rows)
    query_ids = {r['sample_id'] for r in rows if r['split'] == 'query'}
    if set(optical) != query_ids or set(qwen) != query_ids:
        raise ValueError('Predictions must cover ALL fixed queries')
    train = defaultdict(list)
    for row in rows:
        if row['split'] == 'train':
            train[row['product_id']].append(int(row['azimuth']))
    groups = defaultdict(list)
    samples = []
    for row in rows:
        if row['split'] != 'query':
            continue
        angle = int(row['azimuth'])
        gap = min(min(abs(angle - b), 72 - abs(angle - b)) for b in train[row['product_id']])
        sid = row['sample_id']
        hits = (int(optical[sid]['hit_at_1']), int(qwen[sid]['hit_at_1']))
        if any(h not in (0, 1) for h in hits):
            raise ValueError('Expected binary hits')
        for dimension, value in [('all', 0), ('azimuth_index', angle),
                                 ('nearest_train_gap_index', gap), ('category_id', row['category_id'])]:
            groups[(dimension, value)].append(hits)
        samples.append(dict(sample_id=sid, product_id=row['product_id'],
                            category_id=row['category_id'], azimuth_index=angle,
                            nearest_train_gap_index=gap, optical=optical[sid], qwen=qwen[sid]))
    summary = []
    for (dimension, value), hits in sorted(groups.items()):
        a, b = sum(x for x, _ in hits), sum(y for _, y in hits)
        summary.append(dict(dimension=dimension, value=value, n=len(hits), optical_hits=a,
                            qwen_hits=b, optical_hit_at_1=a / len(hits), qwen_hit_at_1=b / len(hits),
                            qwen_only=sum(y and not x for x, y in hits),
                            optical_only=sum(x and not y for x, y in hits)))
    return dict(groups=summary, predictions=samples)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'optical-cache', 'qwen-cache', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(2)
    protocol = json.loads(args.manifest.read_text(encoding='utf-8'))
    rows = protocol['rows']
    validate_protocol(rows)
    if Counter(r['split'] for r in rows) != dict(train=1600, query=800):
        raise ValueError('This audit targets the fixed 1600/800 enrolled protocol')
    counts = Counter((r['product_id'], r['split']) for r in rows)
    if len(counts) != 400 or any(n != (8 if split == 'train' else 4) for (_, split), n in counts.items()):
        raise ValueError('Expected 200 SKUs, each 8 TRAIN and 4 QUERY')
    digest = sha256(args.manifest)
    metrics, predictions, identities = {}, {}, {}
    for name, path in [('optical', args.optical_cache), ('qwen', args.qwen_cache)]:
        cache = torch.load(path, map_location='cpu', weights_only=True)
        metrics[name], predictions[name] = rank_cache(rows, cache, digest)
        identities[name] = dict(path=str(path.resolve()), sha256=sha256(path))
    report = summarize(rows, predictions['optical'], predictions['qwen'])
    report.update(source_commit=source_commit(), manifest_sha256=digest, caches=identities,
                  metrics=metrics, fitting=False, query_filtering=False,
                  caveat='Descriptive TEST-informed audit, not causal evidence or independent validation. '
                         'Azimuth is a 72-position spin index, not degrees or an aligned semantic pose across products. '
                         'Existing test-based model selection remains biased.')
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / 'report.json', report)
    print(json.dumps(dict(output=str(args.output), groups=report['groups']), indent=2))


if __name__ == '__main__':
    main()
