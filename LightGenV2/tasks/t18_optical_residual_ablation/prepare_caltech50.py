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


def split_groups(rows, seed=SEED, train_groups=20, val_groups=5, parent_roles=None):
    """Fixed group counts; optionally preserve previous train/val roles out of new test."""
    parent_roles = parent_roles or {}
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
        assert len(keys) > train_groups + val_groups, ('Too few independent pixel groups', label, len(keys))
        rng.shuffle(keys)
        pinned_train, pinned_val, available = [], [], []
        for key in keys:
            roles = {parent_roles.get(rows[i].get('name')) for i in groups[key]} & {'train', 'val'}
            assert len(roles) <= 1, 'Conflicting previous train/val pixel group'
            if roles == {'train'}: pinned_train.append(key)
            elif roles == {'val'}: pinned_val.append(key)
            else: available.append(key)
        nt, nv = train_groups-len(pinned_train), val_groups-len(pinned_val)
        assert min(nt, nv) >= 0 and len(available) > nt + nv
        selected_train = pinned_train + available[:nt]
        selected_val = pinned_val + available[nt:nt+nv]
        for split, selected in zip(result, (selected_train, selected_val, available[nt+nv:])):
            result[split].extend(i for k in selected for i in groups[k])
    result = {k: sorted(v) for k, v in result.items()}
    assert sorted(i for indices in result.values() for i in indices) == list(range(len(rows)))
    for kind in ('original_hash', 'cache_hash'):
        sets = [{rows[i][kind] for i in indices} for indices in result.values()]
        assert all(not sets[i] & sets[j] for i in range(3) for j in range(i))
    return result, len(groups)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--plan', type=Path)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    plan = json.loads(a.plan.read_text()) if a.plan else {}
    count = plan.get('classes', 50)
    selection_seed = plan.get('selection_seed', SEED)
    split_seed = plan.get('split_seed', SEED)
    train_groups, val_groups = plan.get('train_groups_per_class', 20), plan.get('val_groups_per_class', 5)
    archive = Path(plan['archive_path']) if 'archive_path' in plan else a.out/'caltech-101.zip'
    cache = a.out/f'caltech{count}_fixed_split.npz'
    assert not cache.exists(), 'Never overwrite an existing task/split'
    content = archive.read_bytes()
    assert hashlib.md5(content).hexdigest() == AUTHOR_MD5, 'Author checksum mismatch'
    z = zipfile.ZipFile(io.BytesIO(content)); assert z.testzip() is None
    names = [n for n in z.namelist() if Path(n).name == '101_ObjectCategories.tar.gz']
    assert len(names) == 1, z.namelist()
    # Decompress once before random member reads; seeking backwards in gzip is quadratic.
    import gzip
    t = tarfile.open(fileobj=io.BytesIO(gzip.decompress(z.read(names[0]))), mode='r:')
    members = sorted((m for m in t.getmembers() if m.isfile() and m.name.lower().endswith('.jpg')), key=lambda m:m.name)
    categories = sorted({Path(m.name).parent.name for m in members})
    assert len(categories) == 102 and 'BACKGROUND_Google' in categories and 'Faces_easy' in categories
    # Background is not an object category; Faces_easy is the duplicate face variant.
    candidate_counts = {c:sum(Path(m.name).parent.name == c for m in members) for c in categories}
    pool = [c for c in categories if c not in ('BACKGROUND_Google', 'Faces_easy')
            and candidate_counts[c] >= plan.get('minimum_source_images', 0)]
    classes = sorted(np.random.default_rng(selection_seed).choice(pool, size=count, replace=False).tolist())
    parent_roles, parent_digest = {}, None
    if 'parent_manifest_path' in plan:
        parent_path = Path(plan['parent_manifest_path'])
        parent_digest = hashlib.sha256(parent_path.read_bytes()).hexdigest()
        assert parent_digest == plan['parent_manifest_sha256']
        previous = json.loads(parent_path.read_text())
        parent_roles = {sid:split for split in ('train','val') for sid in previous['split_ids'][split]}
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
    splits, groups = split_groups(rows, split_seed, train_groups, val_groups, parent_roles)
    arrays, supports, ids = {}, {}, {}
    for split, indices in splits.items():
        selected = [rows[i] for i in indices]
        arrays[split+'_images'] = np.stack([x['image'] for x in selected])
        arrays[split+'_labels'] = np.array([x['label'] for x in selected], dtype=np.int64)
        arrays[split+'_ids'] = np.array([x['name'] for x in selected])
        supports[split] = np.bincount(arrays[split+'_labels'], minlength=count).tolist()
        ids[split] = arrays[split+'_ids'].tolist()
    np.savez_compressed(cache, **arrays)
    assert not set(ids['test']) & set(parent_roles), 'Previous train/val must not enter new test'
    manifest = dict(dataset=plan.get('dataset', 'Caltech101_prespecified50_fewshot'),
        source='https://data.caltech.edu/records/mzrjq-6wc02', doi='10.22002/D1.20086',
        license='CC BY 4.0 in primary CaltechDATA API record; original archive retained, not republished',
        classes=classes, category_pool=pool, excluded_categories=['BACKGROUND_Google','Faces_easy'],
        category_selection=f'{count} of {len(pool)} eligible category names using numpy default_rng({selection_seed}); source image count only, no model scores',
        class_counts=[sum(x['label']==k for x in rows) for k in range(count)], total_records=len(rows),
        unique_groups=groups, supports=supports, split_ids=ids, split_seed=split_seed,
        selection_seed=selection_seed, minimum_source_images=plan.get('minimum_source_images',0),
        preparation_plan_sha256=hashlib.sha256(a.plan.read_bytes()).hexdigest() if a.plan else None,
        parent_manifest_sha256=parent_digest, previous_fit_or_validation_images_in_test=0,
        archive_sha256=hashlib.sha256(content).hexdigest(), author_archive_md5=AUTHOR_MD5,
        archive_bytes=len(content), cache_sha256=hashlib.sha256(cache.read_bytes()).hexdigest(),
        preparation_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        grouping_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        split_policy=f'Per category exactly{train_groups} connected pixel groups train,{val_groups} val, remaining test; duplicates together. Previous train/val pinned to same role, never new test',
        preprocessing='Full author photo; EXIF transpose, RGB, bicubic150; no masks, segmentation, crop or learned features',
        limitations=f'Prespecified{count}-category subset, not full Caltech101. Exact pixel groups do not establish subject/photographer independence. Validation-reused task adaptation; support/macro required. No accuracy guarantee.')
    (a.out/'data_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    (a.out/'original_file_manifest.json').write_text(json.dumps(files, indent=2)+'\n')
    (a.out/'group_identity.json').write_text(json.dumps([{k:v for k,v in r.items() if k!='image'} for r in rows], indent=2)+'\n')
    print(json.dumps({k:manifest[k] for k in ('dataset','classes','total_records','unique_groups','supports','archive_sha256','cache_sha256')}))


if __name__ == '__main__': main()
