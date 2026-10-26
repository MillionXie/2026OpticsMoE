"""Split-local paired editing protocol, independent of the historical catalogue.

Compositing is DATA PREPARATION only. The generator receives one RGB and text.
This development pilot never opens the historical TEST or creates a final TEST.
"""
from __future__ import annotations

import hashlib
import json
import random
from functools import lru_cache
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(4*1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def seed(*parts) -> int:
    return int.from_bytes(hashlib.sha256(':'.join(map(str, parts)).encode()).digest()[:8], 'big')


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    tmp.replace(path)


def validate_roles(rows: list[dict]) -> dict:
    """Both roles, not just reference products, must be split-disjoint."""
    owners = {}
    for row in rows:
        if row['split'] not in {'train', 'val'}:
            raise ValueError('Pilot only admits TRAIN/VAL; TEST is sealed')
        for role in ('source', 'target'):
            for key in ('product_id', 'family_id', 'rgba_sha256'):
                value = row[role].get(key)
                if not value:
                    raise ValueError(f'Missing {role}.{key}')
                identity = key, value
                previous = owners.setdefault(identity, row['split'])
                if previous != row['split']:
                    raise ValueError(f'Cross-split {key}: {value}')
        if row['mode'] != 'background' and row['source']['product_id'] == row['target']['product_id']:
            raise ValueError('Object editing must change product identity')
    return {'pairs': dict(Counter(x['split'] for x in rows)),
            'modes': dict(Counter(x['mode'] for x in rows)),
            'role_disjoint': True, 'test_opened': False}


def _pixels_hash(row: dict) -> str:
    with Image.open(row['image_path']) as im:
        rgb = im.convert('RGB').tobytes()
    with Image.open(row['mask_path']) as im:
        alpha = im.convert('L').tobytes()
    return hashlib.sha256(rgb + alpha).hexdigest()


def prepare_pairs(data: Path, output: Path, cfg: dict) -> dict:
    """Choose within existing TRAIN/VAL pools; never repurpose old TEST as fresh."""
    if output.exists():
        raise FileExistsError(output)
    selected = {}
    inventories = {}
    manifest_hashes = {}
    for split in ('train', 'val'):
        path = data / f'{split}.jsonl'
        manifest_hashes[split] = sha256(path)
        groups = defaultdict(list)
        for line in path.read_text(encoding='utf-8').splitlines():
            row = json.loads(line)
            if row['category'] not in cfg['categories']:
                continue
            if row.get('license') != 'CC BY 4.0' or not row.get('mask_path'):
                raise ValueError('Require ABO CC BY 4.0 and genuine render alpha')
            for field in ('image_path', 'mask_path'):
                row[field] = str((data / row[field]).resolve())
            row['product_id'] = row['sequence_id']
            # If upstream supplies a parent/family grouping, retain it. Missing
            # family metadata is recorded, not claimed to be a manual audit.
            row['family_id'] = row.get('parent_id') or row.get('family_id') or row['product_id']
            groups[row['category'], row['product_id']].append(row)
        chosen = []
        for category in cfg['categories']:
            ids = sorted(k for k in groups if k[0] == category)
            random.Random(seed(cfg['seed'], split, category)).shuffle(ids)
            n = cfg['products_per_category'][split]
            if len(ids) < n:
                raise ValueError(f'Insufficient {split} {category}: {len(ids)} < {n}')
            for key in ids[:n]:
                views = sorted(groups[key], key=lambda r: r['sample_id'])
                count = cfg['views_per_product'][split]
                indices = np.linspace(0, len(views)-1, count).round().astype(int)
                for i in indices:
                    row = dict(views[int(i)])
                    row['rgba_sha256'] = _pixels_hash(row)
                    chosen.append(row)
        selected[split] = chosen
        inventories[split] = {'products': len({r['product_id'] for r in chosen}), 'renders': len(chosen)}
    # Reuse only the old BACKGROUND RENDERER, never its train-only target pool.
    from .product_scene_replace_data import COMBINATIONS
    pairs = []
    for split, sources in selected.items():
        pools = defaultdict(list)
        for row in sources:
            pools[row['category']].append(row)
        for source in sources:
            possible = [r for r in pools[source['category']] if r['product_id'] != source['product_id']]
            random.Random(seed(cfg['seed'], source['sample_id'], 'targets')).shuffle(possible)
            base = seed(source['sample_id'], cfg['seed']) % len(COMBINATIONS)
            for mode in ('background', 'object', 'joint'):
                for j in range(2):
                    target = source if mode == 'background' else possible[j % len(possible)]
                    source_scene = COMBINATIONS[base]
                    target_scene = source_scene if mode == 'object' else COMBINATIONS[(base + 11 + j*13) % len(COMBINATIONS)]
                    pairs.append({'sample_id': f"{split}:{source['sample_id']}:{mode}:{j}",
                                  'split': split, 'mode': mode, 'category': source['category'],
                                  'source': source, 'target': target,
                                  'source_scene': source_scene, 'target_scene': target_scene,
                                  'noise_seed': seed(cfg['seed'], split, source['sample_id'], mode, j) % (2**31)})
    audit = validate_roles(pairs)
    audit.update({'inventories': inventories, 'raw_manifest_sha256': manifest_hashes,
                  'family_audit': 'upstream parent/family where available; otherwise SKU only; manual near-duplicate review required before final claims',
                  'scope': 'development TRAIN/VAL pilot, no independent TEST claim',
                  'old_task_checkpoint_or_teacher_used': False})
    write_json(output, {'config': cfg, 'audit': audit, 'pairs': pairs})
    return audit


def pilot_subset(pairs: list[dict], per_cell: int = 2) -> list[dict]:
    """Balance category x edit mode; prefer distinct source products per cell."""
    counts, seen, selected = Counter(), defaultdict(set), []
    for row in pairs:
        key = row['split'], row['category'], row['mode']
        if counts[key] < per_cell and row['source']['product_id'] not in seen[key]:
            selected.append(dict(row))
            counts[key] += 1
            seen[key].add(row['source']['product_id'])
    return selected


def render_pair(row: dict, size: int = 256) -> tuple[Image.Image, Image.Image]:
    from .product_scene_replace_data import render_composed_background
    fg, mask = normalized_product(row['source'],size)
    bg = render_composed_background(tuple(row['source_scene']), size, row['sample_id'], 'source')
    reference = Image.composite(fg, bg, mask)
    if row['mode'] != 'background':
        fg, mask = normalized_product(row['target'],size)
    if row['mode'] != 'object':
        bg = render_composed_background(tuple(row['target_scene']), size, row['sample_id'], 'target')
    return reference, Image.composite(fg, bg, mask)


@lru_cache(maxsize=1024)
def _normalized_product(image_path,mask_path,pixels_sha,size):
    from .product_object_replace_data import ProductObjectReplacementDataset
    row={'image_path':image_path,'mask_path':mask_path}
    if _pixels_hash(row)!=pixels_sha:
        raise ValueError('Raw product pixels changed after manifest preparation')
    return ProductObjectReplacementDataset._load_product(row,size)


def normalized_product(row,size):
    return _normalized_product(row['image_path'],row['mask_path'],row['rgba_sha256'],size)


def scene_text(scene) -> str:
    room, tone, brightness, direction = scene
    return f'a {brightness}, {tone}-toned {room.replace("_", " ")} with a window on the {direction} and directional light from the {direction}'


def add_descriptions(rows: list[dict], descriptions: dict[str, str]) -> list[dict]:
    result = []
    for original in rows:
        row = dict(original)
        source = descriptions[row['source']['sample_id']]
        target = descriptions[row['target']['sample_id']]
        scene = scene_text(row['target_scene'])
        if row['mode'] == 'background':
            prompt = f'Keep the appearance and shape of this {row["category"]}; replace only the background with {scene}.'
        elif row['mode'] == 'object':
            prompt = f'Replace the central {row["category"]} with {target}. Preserve the surrounding background and lighting.'
        else:
            prompt = f'Replace the central {row["category"]} with {target}, and replace the background with {scene}.'
        # Frozen CLIP TRAIN-only alignment teacher describes the intended image.
        row['prompt'] = prompt
        if row.get('explicit_category_teacher',False):
            room,tone,brightness,direction=row['target_scene']
            short_scene=f'{room.replace("_"," ")}, {tone}, {brightness}, {direction} window light'
            row['teacher_caption'] = (f'Full view of one {row["category"]}, '
                f'{source if row["mode"] == "background" else target}, centered, entirely visible; {short_scene}.')
        else:
            row['teacher_caption'] = f'A centered product photograph of {source if row["mode"] == "background" else target} in {scene}.'
        result.append(row)
    return result
