"""New closed-catalog SKU protocol; never overwrites the old ABO split."""
import argparse
import csv
import hashlib
from collections import Counter
from pathlib import Path

from .data import _load_contract
from .io import sha256, source_commit, write_json
from .retail_sources import shape_groups

PROTOCOL = 'abo200_enrolled_sku_hash8train4query_v1'


def split_products(samples, root):
    by_product = {}
    for s in samples:
        by_product.setdefault(s.product_id, []).append(s)
    rows = []
    for key in sorted(by_product):
        images = sorted(by_product[key], key=lambda s: hashlib.sha256(
            ('abo-enrolled42:' + s.sample_id).encode()).hexdigest())
        if len(images) != 12:
            raise ValueError('Require12 photos per product')
        for i, s in enumerate(images):
            rows.append(dict(sample_id=s.sample_id, product_id=key,
                category_id=s.category_id, original_split=s.split,
                split='train' if i < 8 else 'query',
                image_path=s.image_path.relative_to(root).as_posix(),
                image_sha256=sha256(s.image_path)))
    return rows


def prepare(root, output):
    if output.exists():
        raise FileExistsError(output)
    root = root.resolve()
    samples, _ = _load_contract(root)
    rows = split_products(samples, root)
    with (root / 'data/abo_similarity10_manifest.csv').open(encoding='utf-8',newline='') as f:
        original = {r['sample_id']: r for r in csv.DictReader(f)}
    for row in rows:
        row['spin_id'] = original[row['sample_id']].get('spin_id')
        row['azimuth'] = original[row['sample_id']].get('azimuth')
    groups = shape_groups(rows)  # Shared enrolled IDs, disjoint image hashes.
    if len(groups['train']) != 1600 or len(groups['query']) != 800:
        raise ValueError('Require ALL200 products including original validation products')
    output.mkdir(parents=True)
    write_json(output / 'protocol.json', dict(schema=1, protocol=PROTOCOL,
        dataset='ABO-200 enrolled SKU', source_commit=source_commit(),
        parent_manifest_sha256=sha256(root / 'data/abo_similarity10_manifest.csv'),
        counts={k: len(v) for k, v in groups.items()},
        original_split_counts=dict(Counter(r['original_split'] for r in rows)),
        relevance='Exact SKU; all1600 TRAIN views are gallery,800 different heldout views query; all200 SKUs seen during fitting',
        selection='8 TRAIN/4 QUERY per SKU ordered by sha256(abo-enrolled42:<sample_id>), before model scores',
        initialization_required='fresh trainable parameters + verified packaged frozen Qwen frontend ONLY; never resume old ABO-trained optical/electronic weights',
        caveat='Closed-catalog turntable-view generalization, NOT unseen-product or independent capture-session generalization. Shared spin sessions/backgrounds; exact duplicates rejected, near-duplicates not ruled out.',
        validation=False, rows=rows))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    prepare(args.data, args.output)


if __name__ == '__main__':
    main()
