"""Pinned raw-RGB LGVQ cache; no learned features or teacher predictions."""
from collections import Counter
import csv
import hashlib
from pathlib import Path

import torch
from torch.utils.data import Dataset

MANIFEST_SHA = '607c50d20662a47795c23cd2038081b30368ed108ba9be8a1bb8a9d250f8e7fc'
CACHE_SHA = 'd97ffe7d5e936a32cbf7dcbe6ee08859a66a3c16e627234cec9f74eb7d2f0f4b'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def validate_payload(payload, rows, *, counts=None):
    expected = {'train': 2250, 'test': 558} if counts is None else counts
    if Counter(row['split'] for row in rows) != Counter(expected):
        raise ValueError('Original LGVQ split counts differ')
    ids = [row['sample_id'] for row in rows]
    if len(set(ids)) != len(ids) or payload['sample_ids'] != ids:
        raise ValueError('LGVQ sample identities/order differ')
    if payload['splits'] != [row['split'] for row in rows]:
        raise ValueError('LGVQ split binding differs')
    if payload['video_paths'] != [row['video_path'] for row in rows]:
        raise ValueError('LGVQ video binding differs')
    target = torch.tensor([[float(row['spatial']), float(row['temporal'])] for row in rows])
    if payload['target_names'] != ['spatial', 'temporal'] or not torch.equal(payload['targets'], target):
        raise ValueError('LGVQ MOS binding differs')
    if (payload['feature_contract'] != 'raw_rgb_uint8_four_frames_center65_224_v1'
            or payload['crop_fraction'] != .65 or payload['frame_size'] != 224
            or list(payload['frame_fractions']) != [.10, .37, .63, .90]
            or payload['frames'].dtype != torch.uint8
            or tuple(payload['frames'].shape) != (len(rows), 4, 3, 224, 224)):
        raise ValueError('LGVQ raw frame geometry differs')


class LGVQ:
    def __init__(self, root):
        self.root = Path(root)
        self.manifest = self.root / 'lgvq_train2250_test558.csv'
        self.cache = self.root / 'lgvq_four_frames_224_uint8.pt'
        if sha(self.manifest) != MANIFEST_SHA or sha(self.cache) != CACHE_SHA:
            raise ValueError('Pinned LGVQ manifest/raw frame cache SHA differs')
        with self.manifest.open(newline='', encoding='utf-8') as stream:
            self.rows = list(csv.DictReader(stream))
        self.payload = torch.load(self.cache, map_location='cpu', mmap=True, weights_only=False)
        validate_payload(self.payload, self.rows)
        if self.payload['manifest_sha256'] != MANIFEST_SHA:
            raise ValueError('Cached LGVQ manifest identity differs')
        for row in self.rows:
            if not Path(row['video_path']).is_file():
                raise FileNotFoundError(row['video_path'])
        # Audit deterministic pixels against the baseline decoder, not a model.
        from experiments.lgvq_four_stage_optical_electronic_109_no_attention_vqa.data import decode_four_frames
        checked = sorted({0, 701, 1403, 2105, len(self.rows)-1})
        for index in checked:
            actual = decode_four_frames(Path(self.rows[index]['video_path']), size=224, crop_fraction=.65)
            if not torch.equal(actual, self.payload['frames'][index]):
                raise ValueError(f'Raw frame decode mismatch at sample {index}')
        self.identity = {'manifest_sha256': MANIFEST_SHA, 'raw_rgb_cache_sha256': CACHE_SHA,
                         'cache_bytes': self.cache.stat().st_size, 'decode_checked_indices': checked,
                         'counts': {'train': 2250, 'test': 558}, 'teacher_features_used': False,
                         'frame_fractions': [.10,.37,.63,.90], 'crop_fraction': .65,
                         'CNN_normalization': 'ImageNet_RGB', 'Qwen_spatial448_vs_CNN224': 'disclosed'}

    def dataset(self, split, target):
        return LGVQFrames(self.cache, [i for i,r in enumerate(self.rows) if r['split']==split], target)


class LGVQFrames(Dataset):
    def __init__(self, cache, indices, target):
        self.cache, self.indices, self.target = cache, indices, target
        self._payload = None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        if self._payload is None:
            self._payload = torch.load(self.cache, map_location='cpu', mmap=True, weights_only=False)
        i = self.indices[index]
        pixels = self._payload['frames'][i].float().div(255)
        pixels = (pixels-torch.tensor([.485,.456,.406])[None,:,None,None])/torch.tensor([.229,.224,.225])[None,:,None,None]
        return {'image': pixels, 'score': self._payload['targets'][i, 0 if self.target=='spatial' else 1],
                'sample_id': self._payload['sample_ids'][i]}
