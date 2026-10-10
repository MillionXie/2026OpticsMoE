"""Pinned FishNet v1, including HEIC; grouped split without discarding images."""
import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
from PIL import Image, ImageOps
import requests

HERE = Path(__file__).resolve().parent
URL = 'https://data.mendeley.com/public-files/datasets/p3xh4fs7cp/files/f97b1fe0-4a5e-456d-a7a0-9e40f78d9966/file_downloaded'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def grouped_split(rows, seed):
    """Keep original-pixel AND cached-RGB matches in the same split."""
    parent = list(range(len(rows)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    seen = {}
    for i, row in enumerate(rows):
        for kind in ['original_hash', 'cache_hash']:
            key = (kind, row[kind])
            if key in seen:
                j = seen[key]
                assert row['label'] == rows[j]['label'], 'Identical pixels with conflicting labels'
                parent[find(i)] = find(j)
            else:
                seen[key] = i
    groups = {}
    for i in range(len(rows)):
        groups.setdefault(find(i), []).append(i)
    rng = np.random.default_rng(seed)
    result = {k: [] for k in ['train', 'val', 'test']}
    for label in range(8):
        keys = sorted(k for k, members in groups.items() if rows[members[0]]['label'] == label)
        rng.shuffle(keys)
        ntrain, nval = int(len(keys) * .7), int(len(keys) * .15)
        assert min(ntrain, nval, len(keys) - ntrain - nval) > 0
        for split, subset in zip(result, [keys[:ntrain], keys[ntrain:ntrain + nval], keys[ntrain + nval:]]):
            result[split].extend(i for key in subset for i in groups[key])
    result = {k: sorted(v) for k, v in result.items()}
    assert sorted(i for indices in result.values() for i in indices) == list(range(len(rows)))
    for kind in ['original_hash', 'cache_hash']:
        sets = {k: {rows[i][kind] for i in v} for k, v in result.items()}
        assert all(not sets[a].intersection(sets[b]) for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')])
    return result, len(groups)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--profile', type=Path, default=HERE / 'fishnet_profile.json')
    a = parser.parse_args()
    cfg = json.loads(a.profile.read_text())
    a.out.mkdir(parents=True, exist_ok=True)
    assert not (a.out / 'fishnet_fixed_split.npz').exists(), 'Never overwrite a prepared split'
    archive = a.out / 'author_v1.zip'
    if not archive.exists():
        temp = a.out / 'author_v1.zip.partial'
        with requests.get(URL, stream=True, timeout=(30, 120)) as response:
            response.raise_for_status()
            with temp.open('xb') as f:
                for chunk in response.iter_content(8 * 1024 * 1024):
                    f.write(chunk)
        assert sha(temp) == cfg['archive_sha256']
        temp.rename(archive)
    assert sha(archive) == cfg['archive_sha256']
    dependencies = a.out / 'dependencies'
    sys.path.insert(0, str(dependencies))
    import pillow_heif
    assert pillow_heif.__version__ == '0.18.0'
    pillow_heif.register_heif_opener(thumbnails=False)
    rows, sizes = [], Counter()
    with zipfile.ZipFile(archive) as z:
        names = sorted(n for n in z.namelist() if Path(n).suffix.lower() in ['.jpeg', '.jpg', '.png', '.heic'] and not n.startswith('__MACOSX/'))
        assert len(names) == 3013
        counts = Counter(Path(n).parent.name for n in names)
        assert [counts[c] for c in cfg['classes']] == cfg['class_counts'] and len(counts) == 8
        for i, name in enumerate(names):
            with Image.open(io.BytesIO(z.read(name))) as raw:
                image = ImageOps.exif_transpose(raw).convert('RGB')
                pixels = np.asarray(image)
                sizes[image.size] += 1
                original = hashlib.sha256(str(pixels.shape).encode() + pixels.tobytes()).hexdigest()
                cached = np.asarray(image.resize((150, 150), Image.Resampling.BICUBIC)).copy()
            rows.append(dict(name=name, label=cfg['classes'].index(Path(name).parent.name), image=cached,
                             original_hash=original, cache_hash=hashlib.sha256(cached.tobytes()).hexdigest()))
            if (i + 1) % 100 == 0:
                print(json.dumps(dict(decoded=i + 1, total=len(names))), flush=True)
    split, group_count = grouped_split(rows, cfg['split_seed'])
    arrays, supports, ids = {}, {}, {}
    for key, indices in split.items():
        subset = [rows[i] for i in indices]
        arrays[key + '_images'] = np.stack([x['image'] for x in subset])
        arrays[key + '_labels'] = np.array([x['label'] for x in subset], dtype=np.int64)
        arrays[key + '_ids'] = np.array([x['name'] for x in subset])
        supports[key] = np.bincount(arrays[key + '_labels'], minlength=8).tolist()
        ids[key] = arrays[key + '_ids'].tolist()
    cache = a.out / 'fishnet_fixed_split.npz'
    np.savez_compressed(cache, **arrays)
    manifest = dict(dataset=cfg['dataset'], source=cfg['source'], doi=cfg['doi'], license=cfg['license'],
                    classes=cfg['classes'], supports=supports, split_ids=ids, split_seed=cfg['split_seed'],
                    total_records=3013, unique_groups=group_count, original_exact_duplicate_records=3013 - len({x['original_hash'] for x in rows}),
                    split_policy='70/15/15 per-class exact-original-or-cache-pixel groups; all images retained',
                    original_width_height_counts={f'{w}x{h}': n for (w, h), n in sizes.items()},
                    extensions=dict(Counter(Path(x['name']).suffix.lower() for x in rows)),
                    preprocessing='Primary full HEIC/JPEG image; EXIF transpose; RGB8 -> bicubic150; no offline augmentation',
                    limitations='Image-level split; specimen/session identity unavailable, near-duplicate dependence not ruled out',
                    archive_sha256=sha(archive), cache_sha256=sha(cache), profile_sha256=sha(a.profile),
                    preparation_source_sha256=sha(__file__), decoder_version=pillow_heif.__version__)
    save(a.out / 'data_manifest.json', manifest)
    print(json.dumps(dict(state='prepared', supports=supports, unique_groups=group_count, cache_sha256=manifest['cache_sha256'])), flush=True)


if __name__ == '__main__':
    main()
