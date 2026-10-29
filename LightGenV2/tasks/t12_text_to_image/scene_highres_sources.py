"""Fetch original ABO RGBA for the existing TRAIN/DEV cohort, never new SKUs.

Only selected ZIP members are downloaded and CRC checked. Existing 256 JPEGs
are not upscaled and relabelled as high-resolution source data. TEST members
are not opened. This stage creates source assets, not accepted scene targets.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import io
import json
import os
from pathlib import Path
import sys
import traceback
from PIL import Image, ImageOps, ImageDraw
from .prepare_abo_cleanrender import ARCHIVE_URL, _archive, _fetch_remote_member
from .white_reference_scene_probe import digest, write


def source_catalog(rows):
    catalog = {}
    products = {}
    for row in rows:
        if row['split'] not in ['train', 'val']: raise ValueError('TRAIN/DEV only')
        for role in ['source', 'target']:
            item = row[role]
            if item['category'] not in ['lamp', 'pillow']: continue
            key = item['sample_id']
            value = {**item, 'split': 'train' if row['split'] == 'train' else 'dev'}
            if item['license'] != 'CC BY 4.0' or item['source_archive'] != ARCHIVE_URL:
                raise ValueError('Official reviewed ABO provenance required')
            previous = catalog.setdefault(key, value)
            if any(previous[k] != value[k] for k in ['split', 'product_id', 'source_member']):
                raise ValueError('Conflicting source identity')
            owner = products.setdefault(item['product_id'], value['split'])
            if owner != value['split']: raise ValueError('Product leakage')
    result = sorted(catalog.values(), key=lambda r: (r['split'], r['sample_id']))
    if sum(r['split'] == 'train' for r in result) != 76 or sum(r['split'] == 'dev' for r in result) != 16:
        raise ValueError('Existing 76 TRAIN/16 DEV views required')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists() or len(os.environ.get('T12_SOURCE_COMMIT', '')) != 40:
        raise ValueError('New source run and published code required')
    audit = json.loads((a.source_data/'data_audit.json').read_text())
    assert audit['role_disjoint'] and not audit['test_opened']
    rows = source_catalog(json.loads((a.source_data/'annotated_pairs.json').read_text()))
    a.output.mkdir(parents=True)
    (a.output/'rgba').mkdir()
    (a.output/'white').mkdir()
    write(a.output/'execution.json', {'source_commit': os.environ['T12_SOURCE_COMMIT'],
        'source_sha256': digest(Path(__file__)), 'argv': sys.argv,
        'cohort_pair_sha256': digest(a.source_data/'annotated_pairs.json'), 'TEST_opened': False})
    try:
        with _archive(ARCHIVE_URL) as archive:
            infos = {r['source_member']: archive.getinfo(r['source_member']) for r in rows}
        def fetch(row):
            payload = _fetch_remote_member(ARCHIVE_URL, infos[row['source_member']])
            path = a.output/'rgba'/(row['sample_id']+'.png')
            path.write_bytes(payload)
            with Image.open(io.BytesIO(payload)) as im:
                if im.mode != 'RGBA' or min(im.size) < 512:
                    raise ValueError('Genuine native >=512 RGBA required: '+str((im.mode, im.size)))
                rgba = im.convert('RGBA')
                native_size = list(im.size)
                white = Image.new('RGBA', rgba.size, 'white')
                white.alpha_composite(rgba)
                white = ImageOps.pad(white.convert('RGB'), (1024, 1024), color='white', method=Image.Resampling.LANCZOS)
                out = a.output/'white'/(row['sample_id']+'.png')
                white.save(out)
            return {**row, 'original_rgba': str(path.relative_to(a.output)), 'rgba_file_sha256': digest(path),
                'native_RGBA_size': native_size, 'white_reference': str(out.relative_to(a.output)),
                'white_reference_sha256': digest(out), 'reference_resolution': 1024,
                'source_pixels_may_be_resampled': native_size != [1024, 1024]}
        completed = []
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(fetch, row) for row in rows]
            for future in as_completed(futures):
                completed.append(future.result())
                write(a.output/'progress.json', {'stage': 'CRC_checked_original_RGBA', 'done': len(completed), 'total': len(rows)})
        completed.sort(key=lambda r: (r['split'], r['sample_id']))
        write(a.output/'sources.json', completed)
        sheet = Image.new('RGB', (8*160, 12*184), 'white')
        draw = ImageDraw.Draw(sheet)
        for i, row in enumerate(completed):
            x = (i % 8)*160; y = (i//8)*184
            with Image.open(a.output/row['white_reference']) as im:
                sheet.paste(im.resize((160, 160)), (x, y+24))
            draw.text((x+2, y+2), row['split']+' '+row['sample_id'], fill='black')
        sheet.save(a.output/'source_QA.png')
        write(a.output/'report.json', {'complete': True, 'source_commit': os.environ['T12_SOURCE_COMMIT'],
            'n': len(completed), 'TRAIN_views': 76, 'DEV_views': 16, 'TEST_opened': False,
            'sources_sha256': digest(a.output/'sources.json'), 'targets_prepared': 0,
            'all_original_members_CRC_checked': True, 'source_archive': ARCHIVE_URL,
            'license': 'CC BY 4.0', 'upstream_attribution': 'Amazon Berkeley Objects (ABO)',
            'native_size_counts': {str(size): sum(r['native_RGBA_size'] == list(size) for r in completed)
                                   for size in {tuple(r['native_RGBA_size']) for r in completed}}})
    except Exception:
        write(a.output/'failure.json', {'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__': main()
