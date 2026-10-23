"""Original easy100 data binding for both ResNet-20 retrieval directions.

No downloading, split regeneration, teacher features, training or GPU access.
The historical receipt pins every CSV and all 7,200 image bytes. Historical
cross-split duplicate image content is preserved, not silently deduplicated.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset

from maintenance.storage.check_t08_assets import (
    asset_path, audit_split, file_identity, relative_image_content,
)
from .models import tokenize_text

ROOT = Path(__file__).resolve().parents[3]
RECEIPT = ROOT / 'maintenance/storage/T08_CONTENT_ASSETS_20261004.json'


def validate_rows(manifest, train, test, titles, *, products=100,
                  train_per_product=48, test_per_product=24):
    """Validate exact identities and label-to-SKU mapping, never infer by row index."""
    audit_split(manifest, train, test, titles)
    if len(titles) != products:
        raise ValueError('Wrong number of official title candidates')
    labels = [int(row['label']) for row in titles]
    if set(labels) != set(range(products)) or len(set(labels)) != len(labels):
        raise ValueError('Title labels must be a bijection onto the original class IDs')
    by_label = {int(row['label']): row['product_id'] for row in titles}
    for row in titles:
        if not row['product_id'] or not row['title'].strip():
            raise ValueError('Empty official SKU/title')
    tokenize_text([row['title'] for row in titles])  # Reject, never truncate.
    for rows, per_product in ((train, train_per_product), (test, test_per_product)):
        counts = Counter(row['product_id'] for row in rows)
        if counts != Counter({sku: per_product for sku in by_label.values()}):
            raise ValueError('Original per-SKU split counts differ')
        for row in rows:
            if by_label.get(int(row['label'])) != row['product_id']:
                raise ValueError('Image label-to-SKU mapping differs from official titles')


class Easy100:
    """Loads only after full content verification against the pinned receipt."""

    def __init__(self, root: Path):
        self.root = Path(root).resolve(strict=True)
        receipt = json.loads(RECEIPT.read_text(encoding='utf-8'))
        tables = {}
        for name in ('manifest.csv', 'train.csv', 'test.csv', 'titles.csv'):
            path = self.root / name
            if file_identity(path) != receipt['dataset_csv'][name]:
                raise ValueError(f'Original CSV content identity mismatch: {name}')
            with path.open(encoding='utf-8', newline='') as stream:
                tables[name] = list(csv.DictReader(stream))
        self.train = tables['train.csv']
        self.test = tables['test.csv']
        self.titles = tables['titles.csv']
        validate_rows(tables['manifest.csv'], self.train, self.test, self.titles)
        # Preflight every image path before hashing; reject Windows paths on Linux too.
        for row in tables['manifest.csv']:
            asset_path(self.root, row['image_path'])
        content = relative_image_content(tables['manifest.csv'], self.root)
        if content != receipt['image_content']:
            raise ValueError('Original image content identity mismatch')
        self.title_skus = [row['product_id'] for row in self.titles]
        self.title_tokens = tokenize_text([row['title'] for row in self.titles])
        self.identity = {'receipt_sha256': file_identity(RECEIPT)['sha256'],
                         'dataset_csv': receipt['dataset_csv'], 'image_content': content,
                         'title_order': self.title_skus,
                         'cross_split_content_duplicates': 'historical; preserved',
                         'remote_identity_verified': False}

    def images(self, split: str) -> 'Easy100Images':
        if split not in {'train', 'test'}:
            raise ValueError('Only original train/test splits are allowed')
        return Easy100Images(self.root, self.train if split == 'train' else self.test)


class Easy100Images(Dataset):
    """224 center crop, RGB [0,1], ImageNet channel normalization, no augmentation.

    This explicit CNN preprocessing replaces the Qwen processor; it does not
    claim identical pixel inputs to dynamic-resolution Qwen evaluation.
    Returns metadata separately; sample IDs and SKUs never enter either encoder.
    """
    def __init__(self, root: Path, rows: list[dict]):
        self.root, self.rows = root, rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        path = asset_path(self.root, row['image_path'])
        with Image.open(path) as source:
            image = ImageOps.fit(ImageOps.exif_transpose(source).convert('RGB'),
                                 (224, 224), method=Image.Resampling.BICUBIC)
            values = np.array(image, dtype=np.float32, copy=True) / 255.0
        pixels = torch.from_numpy(values).permute(2, 0, 1)
        pixels = (pixels - torch.tensor([.485, .456, .406])[:, None, None]) / torch.tensor(
            [.229, .224, .225])[:, None, None]
        return {'image': pixels, 'sku': row['product_id'], 'sample_id': row['sample_id']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    args = parser.parse_args()
    dataset = Easy100(args.data_root)
    print(json.dumps({'status': 'local_original_data_verified_no_training',
                      'train': len(dataset.train), 'test': len(dataset.test),
                      'titles': len(dataset.titles), **dataset.identity}, indent=2))


if __name__ == '__main__':
    main()
