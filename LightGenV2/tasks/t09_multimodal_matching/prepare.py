"""Official CLEVR images -> a declared, image-disjoint attribute-existence pilot."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import random
import re
import struct
import time
import zipfile
import zlib

import numpy as np
from PIL import Image
import requests

URL = 'https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip'
COLORS = ['gray', 'red', 'blue', 'green', 'brown', 'purple', 'cyan', 'yellow']
SHAPES = ['cube', 'sphere', 'cylinder']
TEMPLATES = [
    'is there a {color} {shape} ?',
    'does the image contain a {color} {shape} ?',
    'can you see a {color} {shape} ?',
    'is a {color} {shape} present in the image ?',
]


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf8')


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


class RemoteZip(io.RawIOBase):
    """Seekable ranged reader, validating every returned byte range."""
    def __init__(self, url):
        self.url = url
        self.position = 0
        response = requests.get(url, headers={'Range': 'bytes=-65536'}, timeout=60)
        response.raise_for_status()
        assert response.status_code == 206, 'Server must support partial content'
        self.size = int(response.headers['Content-Range'].split('/')[-1])
        self.etag = response.headers.get('ETag')
        self.tail_start = self.size - len(response.content)
        self.tail = response.content

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else (self.position if whence == 1 else self.size) + offset
        return self.position

    def block(self, start, stop):
        if start >= self.tail_start and stop <= self.size:
            return self.tail[start-self.tail_start:stop-self.tail_start]
        for attempt in range(5):
            try:
                r = requests.get(self.url, headers={'Range': f'bytes={start}-{stop-1}',
                                                   'If-Match': self.etag}, timeout=90)
                r.raise_for_status()
                assert r.status_code == 206 and len(r.content) == stop-start
                assert r.headers['Content-Range'].startswith(f'bytes {start}-{stop-1}/')
                return r.content
            except (requests.RequestException, AssertionError):
                if attempt == 4:
                    raise
                time.sleep(1 + attempt)

    def read(self, size=-1):
        stop = self.size if size < 0 else min(self.size, self.position+size)
        if stop <= self.position:
            return b''
        result = self.block(self.position, stop)
        self.position = stop
        return result


def member_from_block(info, block, offset):
    pos = info.header_offset-offset
    assert block[pos:pos+4] == b'PK\x03\x04'
    name_len, extra_len = struct.unpack_from('<HH', block, pos+26)
    start = pos+30+name_len+extra_len
    compressed = block[start:start+info.compress_size]
    raw = zlib.decompress(compressed, -15) if info.compress_type == zipfile.ZIP_DEFLATED else compressed
    assert len(raw) == info.file_size
    assert zlib.crc32(raw) & 0xffffffff == info.CRC
    return raw


def tokens(text):
    return re.findall(r'[a-z]+|[?]', text.lower())


def prepare(out, seed=17):
    out.mkdir(parents=True, exist_ok=True)
    if (out/'manifest.json').exists():
        raise FileExistsError('Use a new prepared-data directory')
    archive = RemoteZip(URL)
    with zipfile.ZipFile(archive) as z:
        names = {x.filename: x for x in z.infolist()}
        scene_name = 'CLEVR_v1.0/scenes/CLEVR_train_scenes.json'
        scene_raw = z.read(scene_name)
    scene_all = json.loads(scene_raw)['scenes']
    scenes = {int(x['image_index']): x for x in scene_all}
    chosen = list(range(1500))
    random.Random(seed).shuffle(chosen)
    split_ids = {'train': chosen[:1000], 'val': chosen[1000:1250], 'test_reserved': chosen[1250:]}
    # Reserved test images are not downloaded or loaded by this pilot.
    selected = sorted(split_ids['train'] + split_ids['val'])
    cache = out/'raw_images'
    cache.mkdir(exist_ok=True)
    infos = [names['CLEVR_v1.0/images/train/'+scenes[idx]['image_filename']] for idx in selected]
    infos.sort(key=lambda x: x.header_offset)
    groups = []
    for info in infos:
        if not groups or info.header_offset-groups[-1][0].header_offset > 8*1024*1024:
            groups.append([])
        groups[-1].append(info)

    def fetch(group):
        missing = [i for i in group if not (cache/Path(i.filename).name).exists()]
        if not missing:
            return
        start = missing[0].header_offset
        end = missing[-1].header_offset + missing[-1].compress_size + 4096
        block = archive.block(start, min(end, archive.size))
        for info in missing:
            raw = member_from_block(info, block, start)
            (cache/Path(info.filename).name).write_bytes(raw)

    with ThreadPoolExecutor(6) as pool:
        for number, _ in enumerate(pool.map(fetch, groups), 1):
            if number % 10 == 0 or number == len(groups):
                print(f'downloaded groups {number}/{len(groups)}', flush=True)

    provenance = []
    records = {}
    combinations = [(c, s) for c in COLORS for s in SHAPES]
    for split in ['train', 'val']:
        rows, images = [], []
        for local_index, idx in enumerate(split_ids[split]):
            scene = scenes[idx]
            filename = scene['image_filename']
            raw = (cache/filename).read_bytes()
            info = names['CLEVR_v1.0/images/train/'+filename]
            assert zlib.crc32(raw) & 0xffffffff == info.CRC
            provenance.append(dict(split=split, image_index=idx, file=filename, sha256=digest(raw), crc32=info.CRC))
            image = Image.open(io.BytesIO(raw)).convert('RGB').resize((64, 64), Image.Resampling.LANCZOS)
            images.append(np.asarray(image))
            present = {(obj['color'], obj['shape']) for obj in scene['objects']}
            positive = sorted(present)
            negative = sorted(set(combinations)-present)
            rng = random.Random(seed*100000+idx)
            # Sampling with replacement only if fewer than three distinct positives.
            pos = rng.sample(positive, min(3, len(positive)))
            while len(pos) < 3:
                pos.append(rng.choice(positive))
            neg = rng.sample(negative, 3)
            # Positive and negative of each pair use the same language template.
            for pair, (p, n) in enumerate(zip(pos, neg)):
                template_index = (idx+pair) % len(TEMPLATES)
                for label, (color, shape) in [(1, p), (0, n)]:
                    question = TEMPLATES[template_index].format(color=color, shape=shape)
                    rows.append(dict(image_local=local_index, image_id=filename, question=question,
                                     label=label, color=color, shape=shape, template=template_index))
        records[split] = rows
        np.savez_compressed(out/f'{split}_images.npz', images=np.stack(images))
        save(out/f'{split}_questions.json', rows)

    vocab = {'<pad>': 0, '<unk>': 1}
    for word in sorted({w for row in records['train'] for w in tokens(row['question'])}):
        vocab[word] = len(vocab)
    assert len(vocab) <= 64
    assert all(len(tokens(r['question'])) <= 32 for rows in records.values() for r in rows)
    assert all(w in vocab for r in records['val'] for w in tokens(r['question']))
    save(out/'vocab.json', vocab)
    save(out/'test_reserved_ids.json', split_ids['test_reserved'])
    save(out/'source_files.json', provenance)
    manifest = dict(source=URL, etag=archive.etag, official_scene_sha256=digest(scene_raw),
                    license='CC BY 4.0', task='Derived color-shape existence; custom templates, not official CLEVR score',
                    seed=seed, image_counts={k:len(v) for k,v in split_ids.items()},
                    questions={k:len(v) for k,v in records.items()}, vocab_size=len(vocab),
                    test_accessed=False, image_size=[64,64], image_split_disjoint=True,
                    files={p.name:digest(p.read_bytes()) for p in out.glob('*') if p.is_file()})
    assert not (set(split_ids['train']) & set(split_ids['val']))
    save(out/'manifest.json', manifest)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=17)
    args = parser.parse_args()
    prepare(args.out, args.seed)
