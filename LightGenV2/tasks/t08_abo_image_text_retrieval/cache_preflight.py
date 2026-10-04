"""Read-only guards for the fixed reverse entry, before any model is loaded.

Historical teacher caches bind the literal model identifier. Relocating a
model does not authorize editing cache identities or substituting directions.
"""
from __future__ import annotations
import csv
import hashlib
from pathlib import Path


def expected_identity(raw):
    root = Path(raw['dataset']['dataset_root'])
    rows, hashes = {}, {}
    for stem in ('train', 'test', 'titles', 'manifest'):
        path = root / (stem + '.csv')
        hashes[stem + '_csv'] = hashlib.sha256(path.read_bytes()).hexdigest()
        if stem != 'manifest':
            with path.open(encoding='utf-8', newline='') as stream:
                rows[stem] = list(csv.DictReader(stream))
    if tuple(len(rows[key]) for key in ('train', 'test', 'titles')) != (4800, 2400, 100):
        raise ValueError('Adopted reverse contract requires 4800 TRAIN / 2400 TEST / 100 titles')
    return {'schema_version': 1, 'dataset_sha256': hashes,
            'model_id': raw['qwen']['model_id'], 'embedding_dim': 64,
            'retrieval_direction': 'text_to_image',
            'image_instruction': "Represent the user's input.",
            'title_instruction': 'Retrieve product images that match the following product description.',
            'train_ids': [row['sample_id'] for row in rows['train']],
            'test_ids': [row['sample_id'] for row in rows['test']],
            'title_product_ids': [row['product_id'] for row in rows['titles']]}


def validate_teacher_cache(raw, path):
    """Load tensors on CPU only; do not download, encode or write any asset."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError('Fixed evaluation requires an existing teacher cache; '
                                'no model load or cache reconstruction was started: ' + str(path))
    import torch
    # Only tensors and ordinary containers are expected. A legacy pickle with
    # custom code is not accepted merely because it has a .pt extension.
    payload = torch.load(path, map_location='cpu', weights_only=True)
    expected = expected_identity(raw)
    if not isinstance(payload, dict) or payload.get('identity') != expected:
        raise ValueError('Teacher cache identity differs from the fixed direction, prompts, '
                         'dataset/order or literal local model path. Do not rewrite the '
                         'identity to bypass this check; verify a separately bound cache.')
    for name, count in [('train', 4800), ('test', 2400), ('titles', 100)]:
        value = payload.get(name)
        if not torch.is_tensor(value) or value.shape != (count, 64):
            raise ValueError('Teacher cache tensor shape mismatch: ' + name)
        if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
            raise ValueError('Teacher cache requires finite floating embeddings: ' + name)
    return expected
