"""CPU-only candidate audit; similarity/time proximity does not prove scene identity."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import io
import json
from pathlib import Path
import subprocess
import zipfile

import numpy as np
from PIL import Image, ImageDraw


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    rows = json.loads((a.data / 'image_manifest.json').read_text())
    save = lambda name, value: (a.out / name).write_text(json.dumps(value, indent=2), encoding='utf-8')
    save('config.json', dict(data=str(a.data), method='64-bit horizontal dHash; cross-split Hamming <=4',
        time_gap_seconds=[2, 10, 30], manifest_sha256=digest(a.data / 'image_manifest.json'),
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        command=__import__('sys').argv, numpy=np.__version__, training_permission_unchanged=True))
    names = [r['sample_id'] for r in rows]
    split = np.array([r['split'] for r in rows])
    hashes = []
    with zipfile.ZipFile(a.data / 'images.zip') as archive:
        members = {Path(n).name: n for n in archive.namelist() if not n.endswith('/')}
        for i, name in enumerate(names):
            im = Image.open(io.BytesIO(archive.read(members[name]))).convert('L').resize((9, 8), Image.Resampling.LANCZOS)
            x = np.asarray(im)
            hashes.append(np.packbits((x[:, 1:] > x[:, :-1]).ravel()))
            if i % 3000 == 0:
                print('hashed', i, flush=True)
        hashes = np.array(hashes)
        lut = np.array([n.bit_count() for n in range(256)], dtype=np.uint8)
        pairs = []
        for start in range(0, len(rows), 128):
            dist = lut[np.bitwise_xor(hashes[start:start+128, None, :], hashes[None, :, :])].sum(-1)
            ix, jx = np.where((dist <= 4) & (split[start:start+128, None] != split[None, :]))
            for i, j in zip(ix + start, jx):
                if i < j:
                    pairs.append(dict(a=names[i], b=names[j], hamming=int(dist[i-start, j]),
                        split_a=rows[i]['split'], split_b=rows[j]['split'],
                        label_a=rows[i]['label'], label_b=rows[j]['label']))
        pairs.sort(key=lambda p: (p['hamming'], p['a'], p['b']))
        save('cross_split_candidates.json', pairs)
        if pairs:
            sheet = Image.new('RGB', (640, 180 * min(12, len(pairs))), 'white')
            draw = ImageDraw.Draw(sheet)
            for i, pair in enumerate(pairs[:12]):
                for col, key in enumerate(['a', 'b']):
                    im = Image.open(io.BytesIO(archive.read(members[pair[key]]))).convert('RGB')
                    im.thumbnail((160, 150))
                    sheet.paste(im, (col*320, i*180))
                    draw.text((col*320, i*180+151), pair[key], fill='black')
                    draw.text((col*320+163, i*180+30), f"{pair['split_'+key]}\nlabel {pair['label_'+key]}\ndHash {pair['hamming']}", fill='black')
            sheet.save(a.out / 'candidate_examples.png')
    cameras = defaultdict(list)
    unparsed = []
    for row in rows:
        try:
            day, tm, cam = Path(row['sample_id']).stem.split('-')
            stamp = datetime.strptime(day+tm, '%Y%m%d%H%M%S').timestamp()
            cameras[(day, cam)].append((stamp, row))
        except ValueError:
            unparsed.append(row['sample_id'])
    temporal = {}
    for threshold in [2, 10, 30]:
        groups = []
        for key, items in cameras.items():
            items = sorted(items, key=lambda x: (x[0], x[1]['sample_id']))
            group = []
            last = None
            for stamp, row in items:
                if last is not None and stamp-last > threshold:
                    groups.append(group)
                    group = []
                group.append(row)
                last = stamp
            if group:
                groups.append(group)
        cross = [g for g in groups if len({r['split'] for r in g}) > 1]
        temporal[str(threshold)] = dict(groups=len(groups), cross_split_groups=len(cross),
            samples_in_cross_split_groups=sum(map(len, cross)), largest_group=max(map(len, groups), default=0),
            examples=[[r['sample_id'] for r in g] for g in cross[:5]])
    save('audit.json', dict(state='complete', samples=len(rows), split_counts=dict(Counter(split.tolist())),
        cross_split_dhash_candidates=len(pairs), temporal=temporal, unparsed_names=unparsed,
        interpretation='Candidates require visual review. Timestamps are not verified plant IDs. Absence of dHash matches does not establish independent scenes.',
        training_ready=False))
    print('audit complete; candidate pairs', len(pairs), flush=True)


if __name__ == '__main__':
    main()
