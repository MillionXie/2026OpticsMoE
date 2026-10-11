"""Prespecified Caltech50 few-shot task; category/sample choices never use scores."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile
import zipfile

import numpy as np
from PIL import Image, ImageOps

AUTHOR_MD5 = '3138e1922a9193bfa496528edbbc45d0'
SEED = 20261011


def split_groups(rows, seed=SEED):
    """20 connected groups train, 5 validation, all remaining groups test per class."""
    parent = list(range(len(rows)))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    seen = {}
    for i, row in enumerate(rows):
        for kind in ('original_hash', 'cache_hash'):
            key = (kind, row[kind])
            if key in seen:
                j = seen[key]
                assert row['label'] == rows[j]['label'], ('Cross-category identical pixels', key)
                parent[root(i)] = root(j)
            else:
                seen[key] = i
    groups = {}
    for i in range(len(rows)):
        groups.setdefault(root(i), []).append(i)
    rng = np.random.default_rng(seed)
    result = {key: [] for key in ('train', 'val', 'test')}
    for label in sorted({r['label'] for r in rows}):
        keys = sorted(k for k, indices in groups.items() if rows[indices[0]]['label'] == label)
        assert len(keys) > 25, ('Too few independent pixel groups', label, len(keys))
        rng.shuffle(keys)
        for split, selected in zip(result, (keys[:20], keys[20:25], keys[25:])):
            result[split].extend(i for k in selected for i in groups[k])
    result = {k: sorted(v) for k, v in result.items()}
    assert sorted(i for indices in result.values() for i in indices) == list(range(len(rows)))
    for kind in ('original_hash', 'cache_hash'):
        sets = [{rows[i][kind] for i in indices} for indices in result.values()]
        assert all(not sets[i] & sets[j] for i in range(3) for j in range(i))
    return result, len(groups)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    archive = a.out/'caltech-101.zip'; cache = a.out/'caltech50_fixed_split.npz'
    assert not cache.exists(), 'Never overwrite an existing task/split'
    content = archive.read_bytes()
    assert hashlib.md5(content).hexdigest() == AUTHOR_MD5, 'Author checksum mismatch'
    z = zipfile.ZipFile(io.BytesIO(content)); assert z.testzip() is None
    names = [n for n in z.namelist() if Path(n).name == '101_ObjectCategories.tar.gz']
    assert len(names) == 1, z.namelist()
    t = tarfile.open(fileobj=io.BytesIO(z.read(names[0])), mode='r:gz')
    members = sorted((m for m in t.getmembers() if m.isfile() and m.name.lower().endswith('.jpg')), key=lambda m:m.name)
    categories = sorted({Path(m.name).parent.name for m in members})
    assert len(categories) == 102 and 'BACKGROUND_Google' in categories and 'Faces_easy' in categories
    # Background is not an object category; Faces_easy is the duplicate face variant.
    pool = [c for c in categories if c not in ('BACKGROUND_Google', 'Faces_easy')]
    classes = sorted(np.random.default_rng(SEED).choice(pool, size=50, replace=False).tolist())
    rows, files = [], []
    for member in members:
        category = Path(member.name).parent.name
        if category not in classes:
            continue
        raw = t.extractfile(member).read()
        with Image.open(io.BytesIO(raw)) as image:
            image = ImageOps.exif_transpose(image).convert('RGB')
            original = np.asarray(image).copy()
            cached = np.asarray(image.resize((150, 150), Image.Resampling.BICUBIC)).copy()
        rows.append(dict(name=member.name, label=classes.index(category), image=cached,
            original_hash=hashlib.sha256(str(original.shape).encode()+original.tobytes()).hexdigest(),
            cache_hash=hashlib.sha256(cached.tobytes()).hexdigest()))
        files.append(dict(name=member.name, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()))
    splits, groups = split_groups(rows)
    arrays, supports, ids = {}, {}, {}
    for split, indices in splits.items():
        selected = [rows[i] for i in indices]
        arrays[split+'_images'] = np.stack([x['image'] for x in selected])
        arrays[split+'_labels'] = np.array([x['label'] for x in selected], dtype=np.int64)
        arrays[split+'_ids'] = np.array([x['name'] for x in selected])
        supports[split] = np.bincount(arrays[split+'_labels'], minlength=50).tolist()
        ids[split] = arrays[split+'_ids'].tolist()
    np.savez_compressed(cache, **arrays)
    manifest = dict(dataset='Caltech101_prespecified50_fewshot',
        source='https://data.caltech.edu/records/mzrjq-6wc02', doi='10.22002/D1.20086',
        license='CC BY 4.0 in primary CaltechDATA API record; original archive retained, not republished',
        classes=classes, category_pool=pool, excluded_categories=['BACKGROUND_Google','Faces_easy'],
        category_selection='50 of 100 category names using numpy default_rng(20261011), without image/model scores',
        class_counts=[sum(x['label']==k for x in rows) for k in range(50)], total_records=len(rows),
        unique_groups=groups, supports=supports, split_ids=ids, split_seed=SEED,
        archive_sha256=hashlib.sha256(content).hexdigest(), author_archive_md5=AUTHOR_MD5,
        archive_bytes=len(content), cache_sha256=hashlib.sha256(cache.read_bytes()).hexdigest(),
        preparation_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        grouping_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        split_policy='Per category exactly20 connected pixel groups train,5 val, all remaining test; duplicates retained together; counts may exceed20/5 images',
        preprocessing='Full author photo; EXIF transpose, RGB, bicubic150; no masks, segmentation, crop or learned features',
        limitations='Prespecified50-category few-shot task, not full Caltech101. Exact pixel groups do not establish subject/photographer independence. Only5 validation groups/class; class support and macro required. No accuracy guarantee.')
    (a.out/'data_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    (a.out/'original_file_manifest.json').write_text(json.dumps(files, indent=2)+'\n')
    (a.out/'group_identity.json').write_text(json.dumps([{k:v for k,v in r.items() if k!='image'} for r in rows], indent=2)+'\n')
    print(json.dumps({k:manifest[k] for k in ('dataset','classes','total_records','unique_groups','supports','archive_sha256','cache_sha256')}))


if __name__ == '__main__': main()
