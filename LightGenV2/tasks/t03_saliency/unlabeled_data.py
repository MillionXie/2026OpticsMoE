"""Strict readers for the disjoint image pool and its offline teacher targets.

No SALICON ground truth is manufactured here. This reader is deliberately
separate from TrainTeacherMaps, whose train-only contract remains unchanged.
"""
import io
import json
from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import Dataset

from .prepare_unlabeled_pool import file_sha, ids_sha, image_index

PREPROCESSING = 'RGB_BICUBIC_224_THEN_QWEN_PROCESSOR'


class AuditedUnlabeledImages(Dataset):
    def __init__(self, manifest_path, expected_sha256, salicon_root):
        self.manifest_path = Path(manifest_path).resolve()
        if not expected_sha256 or file_sha(self.manifest_path) != expected_sha256:
            raise ValueError('Unlabeled image manifest SHA mismatch')
        self.manifest_sha256 = expected_sha256
        self.manifest = json.loads(self.manifest_path.read_text(encoding='utf-8'))
        m = self.manifest
        if m.get('schema_version') != 1 or m.get('status') != 'audited_images_only_not_teacher_predictions':
            raise ValueError('Unsupported image-pool contract')
        self.root = Path(m['coco_root']).resolve()
        self.records = m['images']
        ids = [r['image_id'] for r in self.records]
        if (not ids or ids != sorted(set(ids)) or len(ids) != m['selected_count']
                or ids_sha(ids) != m['selected_ids_sha256']):
            raise ValueError('Invalid ordered unlabeled IDs')
        excluded = set()
        for split, folder in [('train', 'train'), ('test', 'val')]:
            index = image_index(Path(salicon_root)/'images'/folder)
            if (len(index) != m[f'salicon_{split}_count']
                    or ids_sha(index) != m[f'salicon_{split}_ids_sha256']):
                raise ValueError('SALICON exclusion split drift')
            excluded.update(index)
        if set(ids) & excluded:
            raise ValueError('Unlabeled pool overlaps a SALICON split')
        self.sample_ids = [f'unlabeled/coco2017/{i:012d}' for i in ids]
        for row in self.records:
            name = row['image_file']
            if Path(name).name != name or (self.root/name).resolve().parent != self.root:
                raise ValueError('Image filename escapes audited COCO root')
            if int(Path(name).stem.rsplit('_', 1)[-1]) != row['image_id']:
                raise ValueError('COCO filename/ID mismatch')

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        import hashlib
        row = self.records[index]
        # Hash the very same bytes decoded by PIL, avoiding separate-read drift.
        contents = (self.root/row['image_file']).read_bytes()
        if len(contents) != row['bytes'] or hashlib.sha256(contents).hexdigest() != row['file_sha256']:
            raise ValueError(f'Image content changed: {self.sample_ids[index]}')
        with Image.open(io.BytesIO(contents)) as image:
            image = image.convert('RGB').resize((224,224), Image.Resampling.BICUBIC)
        return {'image': image, 'sample_id': self.sample_ids[index]}


def collate_unlabeled(rows):
    return {'images': [r['image'] for r in rows], 'sample_ids': [r['sample_id'] for r in rows]}


class UnlabeledTeacherMaps:
    def __init__(self, path, expected_sha256, dataset, teacher_sha256):
        if not expected_sha256 or file_sha(path) != expected_sha256:
            raise ValueError('Unlabeled teacher cache SHA mismatch')
        payload = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
        self.manifest = payload['manifest']
        m = self.manifest
        if (m.get('split') != 'extra_unlabeled_excluding_salicon_train_and_test'
                or m.get('checkpoint_sha256') != teacher_sha256
                or m.get('image_manifest_sha256') != dataset.manifest_sha256
                or m.get('preprocessing') != PREPROCESSING
                or m.get('ground_truth_available') is not False
                or m.get('augmentation') is not False):
            raise ValueError('Unlabeled teacher provenance mismatch')
        ids = payload['sample_ids']
        if ids != dataset.sample_ids or len(set(ids)) != len(ids):
            raise ValueError('Unlabeled teacher ID order mismatch')
        self.values = payload['logits']
        if self.values.dtype != torch.float16 or self.values.shape != (len(ids),1,224,224):
            raise ValueError('Unlabeled teacher shape/dtype mismatch')
        for chunk in self.values.split(64):
            if not torch.isfinite(chunk).all():
                raise ValueError('Nonfinite unlabeled teacher logits')
        self.index = {key:i for i,key in enumerate(ids)}

    def get(self, ids, device):
        if not ids or len(set(ids)) != len(ids):
            raise ValueError('Empty/duplicate unlabeled teacher batch')
        return self.values[[self.index[key] for key in ids]].to(device=device, dtype=torch.float32)
