"""Official SHAPE preparation and bounded OFF feasibility audit, no GPU.

Dataset archives/metadata are artifacts, not executable source. Never executes
archive members. OFF audit is NOT a retrieval benchmark or a rights clearance.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import urllib.request
from zipfile import ZipFile

from .io import sha256, source_commit, write_json

SHAPE_PROTOCOL = 'shape_hash8_categories_official_train_gallery_v1'
SHAPE_API = 'https://api.figshare.com/v2/articles/24100704'
OFF_URL = ('https://world.openfoodfacts.org/api/v2/search?page_size=100&page=1'
           '&sort_by=unique_scans_n&fields=code,images,categories_tags')
UA = 'LightGenV2-RetrievalResearch/1.0 (https://github.com/MillionXie/2026OpticsMoE)'


def fetch(url, maximum=8_000_000):
    request = urllib.request.Request(url, headers={'User-Agent': UA})
    with urllib.request.urlopen(request, timeout=30) as response:
        data = response.read(maximum + 1)
    if len(data) > maximum:
        raise ValueError('Response exceeds bounded audit size')
    return data


def selected_categories(categories):
    """Eight categories chosen by identifiers, BEFORE looking at model scores."""
    return sorted(set(categories), key=lambda c: hashlib.sha256(
        ('shape-category42:' + c).encode()).hexdigest())[:8]


def image_members(names, split):
    rows = []
    for name in sorted(names):
        p = PurePosixPath(name)
        if p.suffix.lower() not in ('.jpg', '.jpeg', '.png', '.bmp'):
            continue
        if p.is_absolute() or '..' in p.parts or '\\' in name:
            raise ValueError('Unsafe archive image path')
        parts = p.parts
        if len(parts) == 4 and parts[0] in ('training_set', 'test_set'):
            parts = parts[1:]
        if len(parts) != 3 or not parts[0].isdigit() or not parts[1].isdigit():
            raise ValueError(f'Unexpected SHAPE category/SKU/image layout: {name}')
        category, sku, filename = parts
        relative = '/'.join([split, category, sku, filename])
        rows.append(dict(sample_id=relative, image_path=relative, archive_member=name,
            product_id=category + '/' + sku, category_id=category,
            split='train' if split == 'training_set' else 'query'))
    if len({r['sample_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate archive image path')
    return rows


def shape_groups(rows):
    if len({r['sample_id'] for r in rows}) != len(rows) or len({r['image_path'] for r in rows}) != len(rows):
        raise ValueError('Duplicate SHAPE row/path')
    if any(r['split'] not in ('train', 'query') for r in rows):
        raise ValueError('Unexpected SHAPE row role')
    train = [r for r in rows if r['split'] == 'train']
    query = [r for r in rows if r['split'] == 'query']
    if not train or not query or not {r['product_id'] for r in query} <= {r['product_id'] for r in train}:
        raise ValueError('Missing SHAPE reference positives; do not silently drop queries')
    if {r['image_sha256'] for r in train} & {r['image_sha256'] for r in query}:
        raise ValueError('Exact duplicate across official train/test; audit before benchmarking')
    # Shared enrolled SKU protocol, as in the authors' recognition test. Do not
    # store duplicate train/gallery paths in the parent manifest.
    return dict(train=train, query=query,
        gallery=[dict(r, split='gallery', source_split='train') for r in train])


def prepare_shape(root, output):
    if output.exists():
        raise FileExistsError(output)
    root = root.resolve()
    metadata_path = root / 'figshare_metadata.json'
    raw = fetch(SHAPE_API) if not metadata_path.exists() else metadata_path.read_bytes()
    metadata = json.loads(raw)
    if metadata['license']['name'] != 'CC BY 4.0':
        raise ValueError('SHAPE license changed; re-audit')
    files = {f['name']: f for f in metadata['files']}
    archives, all_rows = {}, []
    for split in ('training_set', 'test_set'):
        archive = root / (split + '.zip')
        expected = files[archive.name]
        if archive.stat().st_size != expected['size']:
            raise ValueError('Archive incomplete')
        md5 = hashlib.md5()
        with archive.open('rb') as f:
            for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
                md5.update(chunk)
        if md5.hexdigest() != expected['computed_md5']:
            raise ValueError('Archive differs from author-published checksum')
        archives[split] = dict(sha256=sha256(archive), md5=md5.hexdigest(), size=archive.stat().st_size)
        with ZipFile(archive) as z:
            all_rows += image_members(z.namelist(), split)
    selected = selected_categories(r['category_id'] for r in all_rows)
    rows = [r for r in all_rows if r['category_id'] in selected]
    for split in ('training_set', 'test_set'):
        with ZipFile(root / (split + '.zip')) as z:
            for row in rows:
                if not row['image_path'].startswith(split + '/'):
                    continue
                info = z.getinfo(row['archive_member'])
                if info.file_size > 32_000_000:
                    raise ValueError('Unexpected large image member')
                data = z.read(info)  # CRC validated, no extractall or executable members.
                path = (root / row['image_path']).resolve()
                if not path.is_relative_to(root):
                    raise ValueError('Extraction escaped root')
                if path.exists() and path.read_bytes() != data:
                    raise ValueError('Existing extracted image differs')
                if not path.exists():
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(data)
                row['image_sha256'] = hashlib.sha256(data).hexdigest()
    groups = shape_groups(rows)
    output.mkdir(parents=True)
    if not metadata_path.exists():
        metadata_path.write_bytes(raw)
    write_json(output / 'protocol.json', dict(schema=1, protocol=SHAPE_PROTOCOL, dataset='SHAPE',
        source_commit=source_commit(), source_url=SHAPE_API, archives=archives,
        license='CC BY 4.0; attribution required; not a blanket third-party rights guarantee',
        metadata_sha256=sha256(metadata_path), selected_categories=selected,
        selection='First8 categories by sha256(shape-category42:<category>); no model-dependent selection',
        counts={k: len(v) for k, v in groups.items()}, all_archive_counts=dict(Counter(r['split'] for r in all_rows)),
        relevance='Same anonymized category/SKU pair; ALL selected official training images form gallery; all selected official test images are queries. Enrolled SKU, NOT unseen SKU. Pre-cropped product retrieval, not shelf detection.',
        validation=False, rows=rows))


def off_inventory(products):
    rows = []
    for p in products:
        images = p.get('images', {})
        raw = sorted(k for k in images if str(k).isdigit())
        front = sorted({str(v['imgid']) for k, v in images.items()
                        if k.startswith('front_') and isinstance(v, dict) and 'imgid' in v})
        rows.append(dict(code=str(p.get('code', '')), raw_image_ids=raw,
            independent_selected_front_ids=front, potential_front_pair=len(front) >= 2,
            note='Distinct imgid is necessary, NOT sufficient: inspect duplicated uploads, packaging versions and view semantics'))
    return rows


def audit_off(output):
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    try:
        raw = fetch(OFF_URL)
        (output / 'api_response.json').write_bytes(raw)
        response = json.loads(raw)
        rows = off_inventory(response.get('products', []))
        write_json(output / 'report.json', dict(status='complete', source_commit=source_commit(),
            source_url=OFF_URL, response_sha256=hashlib.sha256(raw).hexdigest(),
            sample='Single popularity-sorted page100; feasibility only, NOT representative and NOT evaluation split',
            products=len(rows), potential_front_pairs=sum(r['potential_front_pair'] for r in rows),
            license='Images CC BY-SA; database ODbL. Keep separate notices. Packaging may involve third-party rights.',
            decision='No training/test protocol established; never pair a raw photo with its resized/cropped version', rows=rows))
    except Exception as exc:
        write_json(output / 'report.json', dict(status='failed', source_url=OFF_URL,
            source_commit=source_commit(), error=repr(exc)))
        raise


def preview_off(root, output):
    """At most10 serial400px photos for HUMAN review, never auto-label a split."""
    if output.exists():
        raise FileExistsError(output)
    raw_path = root / 'api_response.json'
    raw = json.loads(raw_path.read_text(encoding='utf-8'))
    products = {str(p['code']): p for p in raw['products']}
    candidates = [r for r in off_inventory(raw['products']) if r['potential_front_pair']
                  and len(r['code']) == 13 and r['code'].isdigit()]
    candidates.sort(key=lambda r: hashlib.sha256(('off-preview42:' + r['code']).encode()).hexdigest())
    output.mkdir(parents=True)
    rows = []
    for r in candidates[:5]:
        code = r['code']
        folder = '/'.join([code[:3], code[3:6], code[6:9], code[9:]])
        for image_id in r['independent_selected_front_ids'][:2]:
            info = products[code]['images'].get(image_id, {})
            if '400' not in info.get('sizes', {}):
                rows.append(dict(code=code, image_id=image_id, status='missing400')); continue
            url = f'https://images.openfoodfacts.org/images/products/{folder}/{image_id}.400.jpg'
            row = dict(code=code, image_id=image_id, url=url)
            try:
                data = fetch(url, maximum=5_000_000)
                path = output / f'{code}_{image_id}.jpg'
                path.write_bytes(data)
                row.update(status='downloaded', path=path.name, sha256=sha256(path))
            except Exception as exc:
                row.update(status='failed', error=repr(exc))
            rows.append(row)
    write_json(output / 'report.json', dict(status='review_required', source_commit=source_commit(),
        metadata_sha256=sha256(raw_path), selection='First5 eligible13-digit products by sha256(off-preview42:<code>),2 distinct selected-front raw imgids each. No model scores used.',
        purpose='Feasibility review ONLY. Different imgids may still be reuploads, packaging changes, or localization errors. Do not train or count retrieval accuracy without independent-image audit.',
        license='Images CC BY-SA; source URLs retained, metadata database ODbL; not a complete attribution clearance', rows=rows))


def review_off_batch(root, output):
    """Fixed20 products x3 candidate photos via official AWS, not a benchmark."""
    if output.exists():
        raise FileExistsError(output)
    from PIL import Image
    import numpy as np
    raw_path = root / 'api_response.json'
    raw = json.loads(raw_path.read_text(encoding='utf-8'))
    candidates = [r for r in off_inventory(raw['products'])
        if len(r['independent_selected_front_ids']) >= 3 and len(r['code'])==13 and r['code'].isdigit()]
    candidates.sort(key=lambda r: hashlib.sha256(('off-review42:' + r['code']).encode()).hexdigest())
    output.mkdir(parents=True)
    rows, comparisons = [], []
    for candidate in candidates[:20]:
        code = candidate['code']
        folder = '/'.join([code[:3],code[3:6],code[6:9],code[9:]])
        decoded = []
        ids = sorted(candidate['independent_selected_front_ids'], key=lambda x: hashlib.sha256(
            ('off-photo42:' + code + ':' + x).encode()).hexdigest())[:3]
        for image_id in ids:
            url = f'https://openfoodfacts-images.s3.eu-west-3.amazonaws.com/data/{folder}/{image_id}.400.jpg'
            row = dict(code=code,image_id=image_id,url=url,review='pending')
            try:
                data = fetch(url,maximum=5_000_000)
                path = output / f'{code}_{image_id}.jpg'
                path.write_bytes(data)
                with Image.open(path) as im:
                    im.load()
                    pixels = np.asarray(im.convert('RGB').resize((32,32)),dtype=np.float32)/255
                row.update(status='downloaded',path=path.name,sha256=sha256(path))
                for other, previous in decoded:
                    comparisons.append(dict(code=code,image_a=other['image_id'],image_b=image_id,
                        identical_bytes=other['sha256']==row['sha256'],
                        thumbnail_mae=float(np.abs(pixels-previous).mean()),
                        note='Diagnostic only, NOT an automatic independent-photo or exclusion decision'))
                decoded.append((row,pixels))
            except Exception as exc:
                row.update(status='failed',error=repr(exc))
            rows.append(row)
    write_json(output / 'report.json',dict(status='review_required',source_commit=source_commit(),
        metadata_sha256=sha256(raw_path),rows=rows,comparisons=comparisons,
        selection='20 products with >=3 distinct front imgids, fixed by off-review42 hash;3 imgids by off-photo42 hash. No model scores.',
        split='NOT assigned; require independent capture, correct side, consistent packaging-version audit first',
        download_policy='Official AWS only, bounded60 requests, serial, no main-server fallback',
        license='Image CC BY-SA and metadata ODbL notices retained; individual uploader/third-party rights need review'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['prepare-shape', 'audit-off', 'preview-off', 'review-off-batch'])
    p.add_argument('--data', type=Path)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.mode == 'prepare-shape':
        if args.data is None:
            p.error('SHAPE requires --data containing author archives')
        prepare_shape(args.data, args.output)
    elif args.mode in ('preview-off', 'review-off-batch'):
        if args.data is None:
            p.error('OFF preview requires --data containing api_response.json')
        (preview_off if args.mode=='preview-off' else review_off_batch)(args.data, args.output)
    else:
        audit_off(args.output)


if __name__ == '__main__':
    main()
