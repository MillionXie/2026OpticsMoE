"""Full author processed COIL20: group adjacent views and exact duplicates, retain all."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import zipfile
import requests
import numpy as np
from PIL import Image
from prepare_butterfly import grouped_split

URL = 'https://www.cs.columbia.edu/CAVE/databases/SLAM_coil-20_coil-100/coil-20/coil-20-proc.zip'
ARCHIVE_SHA = '517c5594820eb40066ba0ff6842e7f09392bf7fde849bf9cb9c28445b0f29e88'


def main():
    p = argparse.ArgumentParser(); p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    cache = a.out/'coil20_fixed_split.npz'; assert not cache.exists()
    archive = a.out/'coil-20-proc.zip'
    if archive.exists(): content = archive.read_bytes()
    else:
        response = requests.get(URL, timeout=(15, 60)); response.raise_for_status(); content = response.content
        assert hashlib.sha256(content).hexdigest() == ARCHIVE_SHA
        archive.write_bytes(content)
    assert hashlib.sha256(content).hexdigest() == ARCHIVE_SHA
    z = zipfile.ZipFile(io.BytesIO(content)); assert z.testzip() is None
    rows, files, pairs = [], [], []
    for name in sorted(n for n in z.namelist() if not n.endswith('/')):
        match = re.fullmatch(r'coil-20-proc/obj(\d+)__(\d+)\.png', name); assert match
        obj, pose = map(int, match.groups()); assert 1 <= obj <= 20 and 0 <= pose < 72
        raw = z.read(name)
        with Image.open(io.BytesIO(raw)) as image:
            assert image.size == (128, 128) and image.mode == 'L'
            original = np.asarray(image).copy()
            cached = np.asarray(image.convert('RGB').resize((150, 150), Image.Resampling.BICUBIC)).copy()
        rows.append(dict(name=name, label=obj-1, specimen=f'obj{obj}_viewblock{pose//6}',
            original_hash=hashlib.sha256(original.tobytes()).hexdigest(),
            cache_hash=hashlib.sha256(cached.tobytes()).hexdigest(), image=cached))
        pairs.append((obj, pose)); files.append(dict(name=name, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(), view_degrees=pose*5))
    assert len(rows) == 1440 and set(pairs) == {(obj, pose) for obj in range(1, 21) for pose in range(72)}
    split, groups = grouped_split(rows, 20261011)
    arrays, supports, ids = {}, {}, {}
    for key, indices in split.items():
        selected = [rows[i] for i in indices]
        arrays[key+'_images'] = np.stack([x['image'] for x in selected])
        arrays[key+'_labels'] = np.array([x['label'] for x in selected], dtype=np.int64)
        arrays[key+'_ids'] = np.array([x['name'] for x in selected])
        supports[key] = np.bincount(arrays[key+'_labels'], minlength=20).tolist()
        ids[key] = arrays[key+'_ids'].tolist()
    np.savez_compressed(cache, **arrays)
    manifest = dict(dataset='COIL20_author_processed_full', source=URL,
        source_page='https://www1.cs.columbia.edu/CAVE/software/softlib/coil-20.php',
        citation='Nene, Nayar, Murase, Columbia Object Image Library COIL-20, CUCS-005-96, 1996',
        license='Author public research download; no explicit CC license in archive/page; original images are not republished',
        classes=[f'object{k:02d}' for k in range(1, 21)], class_counts=[72]*20,
        total_records=1440, unique_groups=groups, supports=supports, split_ids=ids, split_seed=20261011,
        archive_sha256=ARCHIVE_SHA, archive_bytes=len(content), cache_sha256=hashlib.sha256(cache.read_bytes()).hexdigest(),
        preparation_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        grouping_source_sha256=hashlib.sha256((Path(__file__).parent/'prepare_butterfly.py').read_bytes()).hexdigest(),
        split_policy='Per-object 70/15/15 of connected adjacent 6-pose (30 degree) blocks and exact original/cache duplicates; rounding and merged groups affect image proportions',
        preprocessing='Full author-processed grayscale128; duplicate grayscale channels as RGB, bicubic150, original fixed RGB amplitude encoding; no learned features or new segmentation/crop',
        limitations='Known physical object identity classification on withheld views, not unseen object instances or broad semantic categories. Adjacent block boundary views and acquisition dependence remain; image/view-level exploratory benchmark only. Not CC BY.')
    (a.out/'data_manifest.json').write_text(json.dumps(manifest, indent=2))
    (a.out/'archive_file_manifest.json').write_text(json.dumps(files, indent=2))
    print(json.dumps({k:manifest[k] for k in ['dataset','total_records','unique_groups','supports','cache_sha256','archive_sha256']}, indent=2))


if __name__ == '__main__': main()
