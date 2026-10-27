"""Screen real ABO assets for a richer, split-disjoint scene curriculum.

Screening is CPU-only and uses original pixels/alpha, never student outputs.
Held-out products are inventoried for a future frozen-model 576-pair evaluation;
no held-out scenes, feature caches or predictions are produced by screening.
"""
import argparse
import collections
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .qwen_vl_holdout import exposure_inventory, reject_exposure
from .qwen_vl_turbo_data import _pixels_hash, normalized_product, sha256, write_json


def view_quality(mask, category):
    a = np.asarray(mask.convert('L')) > 32
    ys, xs = np.where(a)
    if not len(xs):
        return None
    h, w = a.shape;box = [int(xs.min()), int(ys.min()), int(xs.max()+1), int(ys.max()+1)]
    width, height = box[2]-box[0], box[3]-box[1]
    ratio = width/height; fill = float(a.sum()/(width*height))
    margin = min(box[0], box[1], w-box[2], h-box[3])
    # Geometry screen only: a human/assistant still reviews full contact sheets.
    acceptable = margin >= 2 and (.30 <= ratio <= 1.7 if category == 'lamp' else .65 <= ratio <= 1.6)
    if not acceptable:
        return None
    return {'bbox': box, 'aspect': ratio, 'fill': fill, 'margin': margin,
            'score': fill-.15*abs(ratio-(.6 if category == 'lamp' else 1.))}


def raw_inventory(data, split):
    rows = []
    for line in (data/(split+'.jsonl')).read_text().splitlines():
        row = json.loads(line)
        if row['category'] not in ('lamp', 'pillow'):
            continue
        if row.get('license') != 'CC BY 4.0':
            raise ValueError('Only verified ABO CC BY 4.0 assets')
        row['product_id'] = row['sequence_id']
        row['family_id'] = row.get('parent_id') or row.get('family_id') or row['product_id']
        for key in ('image_path', 'mask_path'):
            row[key] = str((data/row[key]).resolve())
        with Image.open(row['mask_path']) as alpha:
            quality = view_quality(alpha, row['category'])
        if quality is None:
            continue
        row['view_quality'] = quality;row['rgba_sha256'] = _pixels_hash(row)
        rows.append(row)
    return rows


def screen(data, parent, reviewed_data, output, root):
    if output.exists():
        raise FileExistsError('Existing screen is immutable')
    prior, exposure = exposure_inventory(parent, root)
    inherited = json.loads((reviewed_data/'annotated_pairs.json').read_text())
    existing = {split: {r['source']['product_id'] for r in inherited if r['split'] == split} for split in ('train', 'val')}
    output.mkdir(parents=True);audit = {};inventories = {}
    for split in ('train', 'val', 'test'):
        rows = raw_inventory(data, split)
        if split == 'test':
            reject_exposure(rows, prior)
        groups = collections.defaultdict(list)
        for row in rows:
            groups[row['category'], row['product_id']].append(row)
        products = []
        for (category, pid), candidates in sorted(groups.items()):
            ranked = sorted(candidates, key=lambda r: (-r['view_quality']['score'], r['sample_id']))
            if len(ranked) < 2:
                continue
            products.append({'category': category, 'product_id': pid,
                'in_current_cohort': pid in existing.get(split, set()), 'views': ranked[:2],
                'original_available_views': len(candidates), 'review_approved': False})
        inventories[split] = products
        write_json(output/(split+'_candidate_products.json'), products)
        for category in ('lamp', 'pillow'):
            pool = [p for p in products if p['category'] == category]
            for chunk in range(0, len(pool), 12):
                part = pool[chunk:chunk+12]
                sheet = Image.new('RGB', (640, ((len(part)+1)//2)*184), 'white');draw = ImageDraw.Draw(sheet)
                for j, product in enumerate(part):
                    for k, row in enumerate(product['views']):
                        x = ((j % 2)*2+k)*160;y = (j//2)*184
                        fg, alpha = normalized_product(row, 256)
                        im = Image.composite(fg, Image.new('RGB', (256, 256), 'white'), alpha);im.thumbnail((156, 156))
                        sheet.paste(im, (x, y+28));draw.text((x+2, y+2), row['sample_id'], fill='black')
                        draw.text((x+2, y+14), 'CURRENT' if product['in_current_cohort'] else 'NEW candidate', fill='black')
                sheet.save(output/f'{split}_{category}_candidates_{chunk//12:02d}.png')
        audit[split] = {'products': dict(collections.Counter(p['category'] for p in products)),
                        'raw_manifest_sha256': sha256(data/(split+'.jsonl'))}
    write_json(output/'report.json', {'complete': True, 'stage': 'CPU_asset_screen_only',
        'source_commit': os.environ.get('T12_SOURCE_COMMIT'), 'source_sha256': sha256(Path(__file__)),
        'inventories': audit, 'exposure': exposure, 'GPU_used': False, 'model_predictions_used': False,
        'heldout_model_inference': False, 'heldout_scene_or_feature_cache_created': False,
        'visual_review_required': True, 'new_TRAIN_goal_products_per_category': 32,
        'new_TRAIN_goal_pairs': 1920, 'development_pairs': 96,
        'future_heldout_goal': {'products_per_category': 6, 'views_per_product': 2,
                                'conditions_per_view': 8, 'tasks': 3, 'pairs': 576},
        'note': '576 pairs are not 576 independent products. Public-pretraining exposure is unknown. Do not use heldout products or their future targets for training or model choice.'})
    return audit


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('data', 'parent', 'reviewed-data', 'output', 'root'):
        p.add_argument('--'+key, type=Path, required=True)
    a = p.parse_args();print(json.dumps(screen(a.data, a.parent, a.reviewed_data, a.output, a.root), indent=2))


if __name__ == '__main__':
    main()
