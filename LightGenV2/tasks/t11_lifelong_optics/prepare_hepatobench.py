"""Prepare a deterministic HepatoBench TUM/NOR binary split from CC-BY-4.0 ZIPs."""
import argparse
import io
import json
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from .data import sha


def image_members(path):
    with zipfile.ZipFile(path) as archive:
        members = sorted(name for name in archive.namelist() if name.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')))
    if len(members) != len(set(members)) or not members:
        raise ValueError(f'Invalid image identities in {path}')
    return members


def read_selected(path, members, prefix, label):
    images = []
    with zipfile.ZipFile(path) as archive:
        for name in members:
            with Image.open(io.BytesIO(archive.read(name))) as image:
                images.append(np.asarray(image.convert('RGB').resize((150, 150), Image.Resampling.LANCZOS), dtype=np.uint8))
    return np.stack(images), np.full(len(members), label, dtype=np.int64), np.asarray([prefix + name for name in members])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tumor-zip', type=Path, required=True)
    parser.add_argument('--normal-zip', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--train-per-class', type=int, default=600)
    parser.add_argument('--val-per-class', type=int, default=80)
    parser.add_argument('--seed', type=int, default=47)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    split = {'train': [], 'val': []}
    for path, prefix, label in ((args.tumor_zip, 'hepato:TUM:', 0), (args.normal_zip, 'hepato:NOR:', 1)):
        members = image_members(path)
        need = args.train_per_class + args.val_per_class
        if len(members) < need:
            raise ValueError(f'{path} has {len(members)} images, need {need}')
        chosen = rng.permutation(len(members))[:need]
        for name, ids in (('train', chosen[:args.train_per_class]), ('val', chosen[args.train_per_class:])):
            split[name].append(read_selected(path, [members[i] for i in ids], prefix, label))
    arrays = {}
    for name in ('train', 'val'):
        arrays[name + '_images'] = np.concatenate([part[0] for part in split[name]])
        arrays[name + '_labels'] = np.concatenate([part[1] for part in split[name]])
        arrays[name + '_ids'] = np.concatenate([part[2] for part in split[name]])
    if set(arrays['train_ids'].tolist()) & set(arrays['val_ids'].tolist()):
        raise RuntimeError('Train/validation identity leakage')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **arrays)
    manifest = {
        'dataset': 'HepatoBench TUM versus NOR', 'license': 'CC BY 4.0',
        'source': 'https://huggingface.co/datasets/xtxx/HepatoBench',
        'doi': '10.57967/hf/8231', 'cache_sha256': sha(args.out),
        'source_sha256': {'01_TUM.zip': sha(args.tumor_zip), '05_NOR.zip': sha(args.normal_zip)},
        'split_protocol': f'Deterministic seed {args.seed} image-level split from the single public split; {args.train_per_class} train and {args.val_per_class} validation images per class.',
        'label_map': {'0': 'tumor', '1': 'normal'},
        'limitations': 'Only TUM and NOR are used. Public files provide no patient/group identifiers, so the split is not claimed patient-independent.'
    }
    args.out.with_name('hepatobench_binary_manifest.json').write_text(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
