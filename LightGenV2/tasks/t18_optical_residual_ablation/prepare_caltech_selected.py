"""One fixed, validation-guided 20-class development task; no optical test scores."""
import argparse
import collections
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile
import zipfile

import numpy as np
from PIL import Image, ImageOps
from prepare_caltech50 import AUTHOR_MD5, split_groups

HERE = Path(__file__).resolve().parent


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path, value): path.write_text(json.dumps(value, indent=2)+'\n')


def features(image):
    # Fixed thumbnail and finite differences, not a learned feature extractor.
    small = np.asarray(Image.fromarray(image).resize((16, 16), Image.Resampling.BOX), dtype=np.float64)/255
    small -= small.mean((0, 1), keepdims=True)
    parts = [small.flatten(), np.diff(small, axis=0).flatten(), np.diff(small, axis=1).flatten()]
    vector = np.concatenate(parts)
    return vector/max(np.linalg.norm(vector), 1e-12)


def choose_classes(train, validation, count=20):
    classes = sorted(train)
    centroids = {c:np.mean(train[c], axis=0) for c in classes}
    steps = []
    while True:
        means = np.stack([centroids[c] for c in classes])
        scores = []
        for k, c in enumerate(classes):
            distances = ((validation[c][:, None]-means[None])**2).sum(2)
            other = np.min(np.delete(distances, k, axis=1), axis=1)
            scores.append(dict(class_name=c, recall=float(np.mean(distances.argmin(1)==k)),
                margin=float(np.mean(other-distances[:, k])), validation_count=len(validation[c])))
        steps.append(dict(classes=classes[:], macro_recall=float(np.mean([r['recall'] for r in scores])), scores=scores))
        if len(classes) == count: break
        worst = min(scores, key=lambda r:(r['recall'], r['margin'], r['class_name']))
        classes.remove(worst['class_name'])
    return classes, steps


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--plan', type=Path, default=HERE/'caltech_selected_plan.json')
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    plan = json.loads(a.plan.read_text())
    archive = Path(plan['archive_path'])
    content = archive.read_bytes()
    assert hashlib.md5(content).hexdigest() == AUTHOR_MD5
    parents, parent_hashes = {}, {}
    for path in plan['parent_manifests']:
        previous = json.loads(Path(path).read_text())
        parent_hashes[path] = sha(path)
        for role in ('train', 'val'):
            for sid in previous['split_ids'][role]:
                assert parents.get(sid, role) == role
                parents[sid] = role
    author = json.loads(Path(plan['author_record_path']).read_text())
    assert any(x['id'] == 'cc-by-4.0' for x in author['metadata']['rights'])
    z = zipfile.ZipFile(io.BytesIO(content))
    assert z.testzip() is None
    member = next(n for n in z.namelist() if Path(n).name == '101_ObjectCategories.tar.gz')
    t = tarfile.open(fileobj=io.BytesIO(gzip.decompress(z.read(member))), mode='r:')
    members = sorted([m for m in t.getmembers() if m.isfile() and m.name.lower().endswith('.jpg')], key=lambda m:m.name)
    counts = collections.Counter(Path(m.name).parent.name for m in members)
    pool = sorted(c for c, n in counts.items() if n >= plan['minimum_source_images'] and c not in ('BACKGROUND_Google', 'Faces_easy'))
    rows_by_class = {c:[] for c in pool}
    files = {}
    for member in members:
        c = Path(member.name).parent.name
        if c not in rows_by_class: continue
        raw = t.extractfile(member).read()
        with Image.open(io.BytesIO(raw)) as im:
            im = ImageOps.exif_transpose(im).convert('RGB')
            original = np.asarray(im).copy()
            cached = np.asarray(im.resize((150, 150), Image.Resampling.BICUBIC)).copy()
        row = dict(name=member.name, label=0, image=cached,
            original_hash=hashlib.sha256(str(original.shape).encode()+original.tobytes()).hexdigest(),
            cache_hash=hashlib.sha256(cached.tobytes()).hexdigest())
        rows_by_class[c].append(row)
        files[member.name] = dict(name=member.name, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    assignments, group_counts, train, val = {}, {}, {}, {}
    for kind in ('original_hash', 'cache_hash'):
        owners = {}
        for c, rows in rows_by_class.items():
            for row in rows:
                assert owners.get(row[kind], c) == c, ('Cross-category duplicate pixels', c)
                owners[row[kind]] = c
    for c, rows in rows_by_class.items():
        # Determine actual connected groups before assigning their roles.
        _, ng = split_groups(rows, train_groups=1, val_groups=1)
        pinned = {role:len({r['original_hash'] for r in rows if parents.get(r['name']) == role}) for role in ('train', 'val')}
        nt = max(pinned['train'], min(plan['maximum_train_groups'], int(.70*ng)))
        nv = max(pinned['val'], min(plan['maximum_validation_groups'], max(10, int(.15*ng))))
        if nt+nv+plan['minimum_test_groups'] > ng:
            nv = max(pinned['val'], ng-nt-plan['minimum_test_groups'])
        assert nt+nv+plan['minimum_test_groups'] <= ng and min(nt, nv) > 0, (c, ng, nt, nv)
        seed = plan['split_seed']+int(hashlib.sha256(c.encode()).hexdigest()[:6], 16)
        assignments[c], group_counts[c] = split_groups(rows, seed, nt, nv, parents)
        train[c] = np.stack([features(rows[i]['image']) for i in assignments[c]['train']])
        val[c] = np.stack([features(rows[i]['image']) for i in assignments[c]['val']])
    classes, steps = choose_classes(train, val, plan['classes'])
    arrays, supports, ids, retained, originals = {}, {}, {}, [], []
    for c in classes:
        retained.extend([{**r, 'label':classes.index(c)} for r in rows_by_class[c]])
        originals.extend(files[r['name']] for r in rows_by_class[c])
    for role in ('train', 'val', 'test'):
        selected = [{**rows_by_class[c][i], 'label':classes.index(c)} for c in classes for i in assignments[c][role]]
        arrays[role+'_images'] = np.stack([r['image'] for r in selected])
        arrays[role+'_labels'] = np.array([r['label'] for r in selected], dtype=np.int64)
        arrays[role+'_ids'] = np.array([r['name'] for r in selected])
        supports[role] = np.bincount(arrays[role+'_labels'], minlength=len(classes)).tolist()
        ids[role] = arrays[role+'_ids'].tolist()
    assert not set(ids['test']) & set(parents)
    for kind in ('original_hash', 'cache_hash'):
        hashes = [{r[kind] for c in classes for i in assignments[c][role] for r in [rows_by_class[c][i]]} for role in ('train', 'val', 'test')]
        assert all(not hashes[i] & hashes[j] for i in range(3) for j in range(i))
    cache = a.out/'caltech20_selected_split.npz'
    np.savez_compressed(cache, **arrays)
    selection = dict(policy=plan['selection_policy'], training_features='Untrained centered RGB16x16 thumbnail and horizontal/vertical finite differences, unit L2 norm',
        pool=pool, selected_classes=classes, steps=steps, uses_optical_or_test_scores=False,
        limitations='Selection adapts to validation. Proxy score is NOT optical-model accuracy or an independent generalization estimate.')
    save(a.out/'category_selection.json', selection)
    manifest = dict(dataset=plan['dataset'], classes=classes, category_pool=pool, class_counts=[counts[c] for c in classes],
        total_records=sum(counts[c] for c in classes), unique_groups=sum(group_counts[c] for c in classes),
        supports=supports, split_ids=ids, cache_sha256=sha(cache), archive_sha256=sha(archive), author_archive_md5=AUTHOR_MD5,
        source='https://data.caltech.edu/records/mzrjq-6wc02', doi='10.22002/D1.20086', license='CC BY 4.0',
        author_record_sha256=sha(plan['author_record_path']), author_license=author['metadata']['rights'],
        preparation_plan_sha256=sha(a.plan), preparation_source_sha256=sha(__file__), grouping_source_sha256=sha(HERE/'prepare_caltech50.py'),
        parent_manifest_sha256=parent_hashes[plan['parent_manifests'][-1]], parent_manifests=parent_hashes,
        category_selection_sha256=sha(a.out/'category_selection.json'), previous_fit_or_validation_images_in_test=0,
        category_selection=plan['selection_policy'], preprocessing='Full RGB photograph; EXIF transpose, bicubic150; unchanged fixed optical amplitude encoding',
        test_previously_observed=True, test_development=True,
        limitations='Validation-screened20 subset. Old Caltech20 test has been observed; future held-out results are exploratory development. Exact duplicates grouped; subject/photographer independence unproven.')
    save(a.out/'data_manifest.json', manifest)
    save(a.out/'original_file_manifest.json', originals)
    save(a.out/'group_identity.json', [{k:v for k,v in r.items() if k != 'image'} for r in retained])
    print(json.dumps({k:manifest[k] for k in ('dataset', 'classes', 'total_records', 'supports', 'cache_sha256')}))


if __name__ == '__main__': main()
