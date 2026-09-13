"""Bounded external ABO turntable pool; never add views of target products.

Downloads individual official objects, not the 40GB archive. Raw data stays
under data/abo/spins; run manifests/status stay in the supplied task run.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import gzip
import hashlib
import json
import os
import re
from pathlib import Path
import sys
import tempfile
import threading
import urllib.request

import numpy as np
from PIL import Image

from .io import sha256, source_commit, write_csv, write_json
from .prepare_broad_abo import image_signature, near_duplicate, safe_image
from .retrieval_screen import load_screen
from .enrolled_regularization import spin_target_identity

BASE = 'https://amazon-berkeley-objects.s3.us-east-1.amazonaws.com/'
METADATA_SHA = '0da9f44c7f684aee1a43dc0eb05dbe2d3941e376dc9400d2663242dfd67e5132'
TYPES = ('BED', 'CHAIR', 'HOME_MIRROR', 'LIGHT_FIXTURE', 'PILLOW', 'RUG', 'SOFA', 'STOOL_SEATING', 'VASE', 'WALL_ART')


class BudgetExceeded(RuntimeError):
    pass


class DownloadBudget:
    def __init__(self, limit):
        self.limit, self.used, self.lock = limit, 0, threading.Lock()

    def consume(self, size):
        with self.lock:
            if size < 0 or self.used + size > self.limit:
                raise BudgetExceeded('Download byte budget exhausted; no ready manifest will be published')
            self.used += size

    def refund(self, size):
        with self.lock:
            if not 0 <= size <= self.used:
                raise ValueError('Invalid download reservation refund')
            self.used -= size


def fetch_file(url, path, budget, maximum, expected_sha=None):
    if not url.startswith(BASE):
        raise ValueError('Only official ABO object URLs allowed')
    if path.exists():
        if not path.is_file() or path.stat().st_size > maximum or (expected_sha and sha256(path) != expected_sha):
            raise ValueError(f'Existing cached object differs: {path}')
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            declared = response.headers.get('Content-Length')
            reservation = int(declared) if declared is not None else maximum
            if not 0 < reservation <= maximum:
                raise ValueError('Official object exceeds per-file size bound')
            # Reserve before body reads, so concurrent workers cannot each
            # over-read the final remaining chunk of the shared byte budget.
            budget.consume(reservation)
            size = 0
            try:
                with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + '.', suffix='.partial', delete=False) as out:
                    temporary = Path(out.name)
                    while size < reservation:
                        chunk = response.read(min(65536, reservation - size))
                        if not chunk:
                            break
                        size += len(chunk)
                        out.write(chunk)
                if (declared is not None and size != reservation) or (declared is None and size == maximum):
                    raise ValueError('Truncated or unbounded official object')
            finally:
                budget.refund(reservation - size)
        if expected_sha and sha256(temporary) != expected_sha:
            raise ValueError('Official metadata SHA mismatch')
        # Do not silently replace data created by another worker/process.
        if path.exists():
            if sha256(path) != sha256(temporary):
                raise ValueError('Concurrent cached object differs')
        else:
            # Hard-link publishes only if the destination does not already exist.
            os.link(temporary, path)
        return path
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)  # Only this call's owned transient file.


def uniform_views(rows, count):
    by_angle = {}
    for row in rows:
        angle = int(row['azimuth'])
        if angle in by_angle or not 0 <= angle <= 71:
            raise ValueError('Duplicate/invalid azimuth in spin metadata')
        by_angle[angle] = row
    ordered = [by_angle[a] for a in sorted(by_angle)]
    if count < 2 or len(ordered) < count:
        raise ValueError('Insufficient distinct spin views')
    return [ordered[i * len(ordered) // count] for i in range(count)]


def candidates(listings, spins, target_rows, views):
    blocked_skus = {r['product_id'] for r in target_rows}
    blocked_spins = {r['spin_id'] for r in target_rows}
    owners = defaultdict(set)
    records = {}
    for row in listings:
        sid, pid = row.get('spin_id'), row['item_id']
        if sid:
            owners[sid].add(pid)
            records[pid] = row
    groups, excluded = defaultdict(list), Counter()
    for pid, row in sorted(records.items()):
        sid = row['spin_id']
        if pid in blocked_skus or sid in blocked_spins:
            excluded['target_sku_or_spin'] += 1
            continue
        if len(owners[sid]) != 1:
            excluded['shared_spin_sku_alias'] += 1
            continue
        kinds = [x['value'] for x in row.get('product_type', [])]
        if len(kinds) != 1 or kinds[0] not in TYPES or len(spins.get(sid, [])) < views:
            continue
        chosen = uniform_views(spins[sid], views)
        groups[kinds[0]].append((pid, sid, chosen))
    for group in groups.values():
        group.sort(key=lambda item: hashlib.sha256(('abo-spin-pretrain-v1|' + item[0]).encode()).hexdigest())
    return groups, excluded


def prepare(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    if sha256(args.target_manifest) != args.expected_target_sha256:
        raise ValueError('Target protocol SHA mismatch')
    protocol, _ = load_screen(args.target_manifest, args.target_root)
    if protocol['protocol'] != 'abo200_enrolled_sku_hash8train4query_v1' or protocol['counts'] != dict(train=1600, query=800, gallery=1600):
        raise ValueError('Only fixed enrolled-200 protocol supported')
    args.output.mkdir(parents=True)
    budget = DownloadBudget(args.max_download_mib * 1024**2)
    status = dict(status='running', pid=os.getpid(), source_commit=source_commit(), command=sys.argv)
    write_json(args.output / 'status.json', status)
    try:
        metadata = safe_image(args.abo_root, 'spins/metadata/spins.csv.gz')
        fetch_file(BASE + 'spins/metadata/spins.csv.gz', metadata, budget, 10_000_000, METADATA_SHA)
        for name in ('spins/README.md', 'LICENSE-CC-BY-4.0.txt'):
            fetch_file(BASE + name, safe_image(args.abo_root, name), budget, 100_000)
        spins = defaultdict(list)
        with gzip.open(metadata, 'rt', encoding='utf-8') as stream:
            for row in csv.DictReader(stream):
                if not re.fullmatch('[0-9a-f]{8}', row['spin_id']):
                    raise ValueError('Invalid official spin identity')
                expected = f"{row['spin_id'][:2]}/{row['spin_id']}/{row['spin_id']}_{int(row['azimuth']):02d}.jpg"
                if row['path'] != expected:
                    raise ValueError('Unexpected official spin object path')
                spins[row['spin_id']].append(row)
        files = sorted((args.abo_root / 'listings/metadata').glob('*.json.gz'))
        def listing_rows():
            for path in files:
                with gzip.open(path, 'rt', encoding='utf-8') as stream:
                    for line in stream:
                        yield json.loads(line)
        groups, excluded = candidates(listing_rows(), spins, protocol['rows'], args.views)
        protected = np.stack([image_signature(safe_image(args.target_root, r['image_path'])) for r in protocol['rows']])
        blocked_sha = {r['image_sha256'] for r in protocol['rows']}
        seen_sha, seen_ids, rows, counts = set(), set(), [], {}
        def download(row):
            relative = 'spins/original/' + row['path']
            path = safe_image(args.abo_root, relative)
            fetch_file(BASE + relative, path, budget, 5 * 1024**2)
            with Image.open(path) as im:
                if im.size != (int(row['width']), int(row['height'])):
                    raise ValueError('Image dimensions differ from official metadata')
                im.verify()
            return dict(row, image_path=relative, image_sha256=sha256(path), signature=image_signature(path))
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            for cid, kind in enumerate(TYPES):
                count = 0
                for pid, sid, views in groups.get(kind, []):
                    try:
                        images = list(executor.map(download, views))
                    except (OSError, ValueError) as error:
                        excluded['unavailable_or_invalid_product'] += 1
                        print(json.dumps(dict(rejected_product=pid, error=str(error))), flush=True)
                        continue
                    hashes = {r['image_sha256'] for r in images}
                    ids = {r['image_id'] for r in images}
                    if hashes & blocked_sha or any(near_duplicate(r['signature'], protected, args.hamming_threshold) for r in images):
                        excluded['target_exact_or_near_duplicate_product'] += 1
                        continue
                    if len(hashes) != args.views or len(ids) != args.views or hashes & seen_sha or ids & seen_ids:
                        excluded['pool_duplicate_product'] += 1
                        continue
                    for row in images:
                        signature = row.pop('signature')
                        rows.append(dict(row, product_id=pid, category_id=cid, category=kind, split='train',
                                         sample_id=f'{pid}__{sid}__{int(row["azimuth"]):02d}', signature128_hex=signature.tobytes().hex()))
                    seen_sha.update(hashes); seen_ids.update(ids)
                    count += 1
                    status.update(selected_images=len(rows), downloaded_bytes=budget.used, current_type=kind)
                    write_json(args.output / 'status.json', status)
                    if count >= args.products_per_type:
                        break
                counts[kind] = count
                print(json.dumps(dict(product_type=kind, products=count, images=len(rows), downloaded_bytes=budget.used)), flush=True)
        if min(counts.values()) < args.minimum_products:
            raise ValueError(f'Insufficient per-type coverage; no ready manifest: {counts}')
        if sha256(args.target_manifest) != args.expected_target_sha256:
            raise ValueError('Target protocol changed during preparation')
        write_csv(args.output / 'manifest.csv', rows)
        report = dict(status='ready', pool_kind='abo_spin_pretrain_v1', source_commit=source_commit(), command=sys.argv,
            target_manifest_sha256=protocol['parent_manifest_sha256'], enrolled_manifest_sha256=args.expected_target_sha256,
            enrolled_rows_sha256=spin_target_identity(protocol), manifest_sha256=sha256(args.output / 'manifest.csv'),
            spin_metadata_sha256=sha256(metadata), listing_metadata_sha256={p.name: sha256(p) for p in files},
            selected_products=sum(counts.values()), selected_images=len(rows), views_per_product=args.views,
            products_per_type=counts, excluded=dict(excluded), downloaded_bytes=budget.used,
            max_download_bytes=budget.limit, target_product_overlap=0, target_spin_overlap=0, target_sha_overlap=0,
            selection='Stable SKU hash order, up to N per fixed type; uniformly spaced available azimuth indices; all target SKUs/spins excluded',
            duplicate_screen='SHA256 and image_id; target 128-bit dHash, whole product rejected', hamming_threshold=args.hamming_threshold,
            limitation='dHash uses legacy fixed224 center crop only for duplicate screening; training remains contain_white. Heuristic is not semantic-independence proof.',
            license_note='Official homepage/spins README say CC BY4.0; AWS registry says CC BY-NC4.0. Retain actual license; clarify discrepancy before redistribution/publication use.',
            root=str(args.abo_root.resolve()))
        write_json(args.output / 'report.json', report)
        status.update(status='complete', selected_images=len(rows), downloaded_bytes=budget.used)
        write_json(args.output / 'status.json', status)
        print(json.dumps(report), flush=True)
    except BaseException as error:
        status.update(status='failed', error=str(error), downloaded_bytes=budget.used)
        write_json(args.output / 'status.json', status)
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--abo-root', type=Path, required=True)
    p.add_argument('--target-root', type=Path, required=True)
    p.add_argument('--target-manifest', type=Path, required=True)
    p.add_argument('--expected-target-sha256', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--products-per-type', type=int, default=50)
    p.add_argument('--minimum-products', type=int, default=10)
    p.add_argument('--views', type=int, default=12)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--max-download-mib', type=int, default=1024)
    p.add_argument('--hamming-threshold', type=int, default=4)
    a = p.parse_args()
    if not (2 <= a.views <= 24 and 1 <= a.minimum_products <= a.products_per_type <= 100 and 1 <= a.workers <= 4 and 1 <= a.max_download_mib <= 2048 and 0 <= a.hamming_threshold <= 8):
        raise ValueError('Preparation bounds exceeded')
    prepare(a)


if __name__ == '__main__':
    main()
