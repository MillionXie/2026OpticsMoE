"""Complete pinned Butterfly-10K; specimen and pixel groups never cross splits."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import threading
import time

import numpy as np
from PIL import Image, ImageOps
import requests

HERE = Path(__file__).resolve().parent
REPO = 'imageomics/VLM4Bio'
REVISION = '354805681278db357ffdff8749a29e2ca145d32e'
PREFIX = 'datasets/Butterfly/'
METADATA = {
    PREFIX + 'metadata/metadata_10k.csv': 'a2f0ebaf0db492ffab05c0a91b6bd6e1ae39abdbbeb4712d84f9d32efb02cbc2',
    PREFIX + 'metadata/imagelist_10k.txt': '31074d776eed2dc8246be2fef4e9346ef9b6c751a25c6e3155254a21f13bb735',
    'license-metadata/butterfly-licenses.csv': '21531ce70c75d69861ff3ebd4db9c926f3d9cd2687f9254519ba7bea25e4b081',
    'license-metadata/butterfly-licenses.json': 'f95b80f9ea1259f586f6396e1fe66a681a26facb73dbbf310bd1ff50b09fb4d9',
}
LOCAL = threading.local()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def fetch(url):
    if not hasattr(LOCAL, 'session'):
        LOCAL.session = requests.Session()
    for attempt in range(5):
        try:
            response = LOCAL.session.get(url, timeout=(20, 90))
            response.raise_for_status()
            return response
        except requests.RequestException:
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)


def specimen(name):
    match = re.fullmatch(r'Butterfly_imbalanced_(?:train|test|val)_[A-Za-z]+_[A-Za-z]+_\d+_(.+)', name, re.I)
    assert match, ('Unknown specimen filename', name)
    token = match[1].upper()
    while re.search(r'\.(?:JPE?G|PNG)$', token):
        token = re.sub(r'\.(?:JPE?G|PNG)$', '', token)
    token = re.sub(r'_WHITESTANDARD$', '', token)
    token = re.sub(r'(?:_?(?:DORSAL|VENTRAL)|_[DV]|-(?:UP|UND)|(?<=\d)[DV]\d*)$', '', token)
    # Numeric-prefix photographs may append taxonomy before their view marker.
    token = re.sub(r'^(\d+)_H_.+$', r'\1', token)
    token = re.sub(r'[^A-Z0-9]', '', token)
    parts = re.findall(r'[A-Z]+|\d+', token)
    assert parts
    # CS1669 and CS001669; CS-CM_4 and CS-CM4 share canonical identities.
    return ''.join(str(int(p)) if p.isdigit() else p for p in parts)


def grouped_split(rows, seed):
    parent = list(range(len(rows)))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    seen = {}
    for i, row in enumerate(rows):
        for kind in ('specimen', 'original_hash', 'cache_hash'):
            key = (kind, row[kind])
            if key in seen:
                j = seen[key]
                assert row['label'] == rows[j]['label'], ('Conflicting grouped labels', key)
                parent[root(i)] = root(j)
            else:
                seen[key] = i
    groups = {}
    for i in range(len(rows)):
        groups.setdefault(root(i), []).append(i)
    result = {key: [] for key in ('train', 'val', 'test')}
    rng = np.random.default_rng(seed)
    for label in sorted({x['label'] for x in rows}):
        keys = sorted(k for k, v in groups.items() if rows[v[0]]['label'] == label)
        rng.shuffle(keys)
        nt, nv = int(len(keys) * .7), int(len(keys) * .15)
        assert min(nt, nv, len(keys) - nt - nv) > 0
        for key, subset in zip(result, (keys[:nt], keys[nt:nt + nv], keys[nt + nv:])):
            result[key].extend(i for g in subset for i in groups[g])
    result = {key: sorted(v) for key, v in result.items()}
    assert sorted(i for v in result.values() for i in v) == list(range(len(rows)))
    for kind in ('specimen', 'original_hash', 'cache_hash'):
        sets = [{rows[i][kind] for i in v} for v in result.values()]
        assert not any(sets[i] & sets[j] for i in range(3) for j in range(i))
    return result, len(groups)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--endpoint', choices=['https://huggingface.co', 'https://hf-mirror.com'], default='https://huggingface.co')
    a = parser.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    assert not (a.out / 'butterfly_fixed_split.npz').exists(), 'Never overwrite a prepared split'
    base = f'{a.endpoint}/datasets/{REPO}/resolve/{REVISION}/'
    for name, expected in METADATA.items():
        path = a.out / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(fetch(base + name).content)
        assert sha(path) == expected, ('Pinned metadata changed', name)
    for name in ('README.md', 'license-metadata/butterfly-data-bib.bib'):
        path = a.out / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(fetch(base + name).content)
    metadata = list(csv.DictReader((a.out / (PREFIX + 'metadata/metadata_10k.csv')).open()))
    names = {x['fileNameAsDelivered'] for x in metadata}
    assert len(metadata) == len(names) == 10013
    assert names == set((a.out / (PREFIX + 'metadata/imagelist_10k.txt')).read_text().splitlines())
    classes = sorted({x['scientificName'] for x in metadata})
    assert len(classes) == 60
    licenses = list(csv.DictReader((a.out / 'license-metadata/butterfly-licenses.csv').open()))
    assert {x['license'] for x in licenses} == {'CC BY 4.0'}
    license_names = {x['image_filename'] for x in licenses}
    assert 'All butterfly images' in (a.out / 'README.md').read_text()
    files = []
    url = f'{a.endpoint}/api/datasets/{REPO}/tree/{REVISION}/datasets/Butterfly?recursive=true&limit=1000'
    while url:
        response = fetch(url)
        files.extend(x for x in response.json() if x['type'] == 'file' and Path(x['path']).name in names)
        url = response.links.get('next', {}).get('url')
        if url:
            url = url.replace('https://huggingface.co/', a.endpoint + '/')
    by_name = {Path(x['path']).name: x for x in files}
    assert len(files) == len(by_name) == len(names)
    save(a.out / 'pinned_download_index.json', files)
    def download(entry):
        path = a.out / entry['path']
        path.parent.mkdir(parents=True, exist_ok=True)
        content = path.read_bytes() if path.exists() else fetch(base + entry['path']).content
        assert len(content) == entry['size']
        assert hashlib.sha1(f'blob {len(content)}\0'.encode() + content).hexdigest() == entry['oid']
        if not path.exists():
            path.write_bytes(content)
        return entry['path'], hashlib.sha256(content).hexdigest()
    hashes = {}
    with ThreadPoolExecutor(max_workers=a.workers) as executor:
        pending = [executor.submit(download, entry) for entry in files]
        for future in as_completed(pending):
            name, digest = future.result()
            hashes[name] = digest
            if len(hashes) % 250 == 0:
                print(json.dumps({'downloaded': len(hashes), 'total': len(names)}), flush=True)
    save(a.out / 'original_file_sha256.json', hashes)
    rows, sizes = [], Counter()
    for i, item in enumerate(sorted(metadata, key=lambda x: x['fileNameAsDelivered'])):
        name = item['fileNameAsDelivered']
        with Image.open(a.out / by_name[name]['path']) as image:
            image = ImageOps.exif_transpose(image).convert('RGB')
            pixels = np.asarray(image)
            sizes[image.size] += 1
            original = hashlib.sha256(str(pixels.shape).encode() + pixels.tobytes()).hexdigest()
            cached = np.asarray(image.resize((150, 150), Image.Resampling.BICUBIC)).copy()
        rows.append(dict(name=name, label=classes.index(item['scientificName']), specimen=specimen(name),
                         image=cached, original_hash=original, cache_hash=hashlib.sha256(cached.tobytes()).hexdigest()))
    indices, ng = grouped_split(rows, 20261011)
    arrays, supports, ids = {}, {}, {}
    for split, subset in indices.items():
        arrays[split + '_images'] = np.stack([rows[i]['image'] for i in subset])
        arrays[split + '_labels'] = np.array([rows[i]['label'] for i in subset], dtype=np.int64)
        arrays[split + '_ids'] = np.array([rows[i]['name'] for i in subset])
        supports[split] = np.bincount(arrays[split + '_labels'], minlength=60).tolist()
        ids[split] = arrays[split + '_ids'].tolist()
    cache = a.out / 'butterfly_fixed_split.npz'
    np.savez_compressed(cache, **arrays)
    save(a.out / 'group_identity.json', [{k: v for k, v in row.items() if k != 'image'} for row in rows])
    manifest = dict(dataset='VLM4Bio_Butterfly10K_full60', source=f'https://huggingface.co/datasets/{REPO}',
        source_revision=REVISION, download_endpoint=a.endpoint, doi='10.57967/hf/3393', license='CC BY 4.0', classes=classes,
        class_counts=[sum(x['label'] == k for x in rows) for k in range(60)], total_records=len(rows),
        supports=supports, split_ids=ids, split_seed=20261011, unique_groups=ng,
        unique_specimens=len({x['specimen'] for x in rows}), metadata_sha256=METADATA,
        individual_license_rows=len(names & license_names), license_rows_missing=sorted(names - license_names),
        missing_license_scope='349 filenames absent from author per-image CSV; covered by explicit all-butterfly CC BY4 author dataset card; retained, not falsely claimed individually mapped',
        split_policy='70/15/15 per-class connected specimen-ID/original-pixel/cache-pixel groups; all 10013 images and 60 species retained; author filename train/test tokens not used as our splits',
        limitations='Known specimen IDs grouped; collection/brood/location dependence and unresolved aliases remain possible; long-tailed species with small evaluation support',
        original_width_height_counts={f'{w}x{h}': n for (w, h), n in sizes.items()},
        original_files_manifest_sha256=sha(a.out / 'original_file_sha256.json'),
        group_identity_sha256=sha(a.out / 'group_identity.json'), cache_sha256=sha(cache),
        preparation_source_sha256=sha(__file__), preprocessing='Full delivered RGB image, EXIF transpose, bicubic150; no segmentation, crop, discarded class or offline augmentation')
    save(a.out / 'data_manifest.json', manifest)
    print(json.dumps({'state': 'prepared', 'supports': supports, 'groups': ng, 'sha256': sha(cache)}), flush=True)


if __name__ == '__main__':
    main()
