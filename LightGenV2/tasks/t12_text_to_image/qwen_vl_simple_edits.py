"""One-change object/background editing, or two-change joint editing.

CPU preparation only. ABO alpha compositing creates supervised TRAIN/DEV pairs;
neither targets nor alpha masks are inference inputs. Original cohorts remain
immutable and the separate held-out cohort is not opened here.
"""
import argparse
import collections
import json
import os
from pathlib import Path
import re
import textwrap

import numpy as np
from PIL import Image, ImageDraw

from .qwen_vl_turbo_data import normalized_product, seed, sha256, validate_roles, write_json


PROTOCOL = 'simple_studio_edit_v1'
PALETTES = {'light_gray': (204, 206, 208), 'warm_beige': (219, 202, 176),
            'pale_blue': (171, 196, 214), 'sage_green': (182, 199, 178),
            'muted_rose': (216, 185, 184)}
LABELS = {'lamp': {'cylindrical': 'a lamp with a cylindrical shade',
                   'tapered': 'a lamp with a tapered shade'},
          'pillow': {'striped': 'a striped cushion', 'geometric': 'a geometric-pattern cushion'}}
CN_LABELS = {'cylindrical': '圆柱灯罩台灯', 'tapered': '锥形灯罩台灯',
             'striped': '条纹靠垫', 'geometric': '几何图案靠垫'}
CN_COLORS = {'light_gray': '浅灰色', 'warm_beige': '暖米色', 'pale_blue': '浅蓝色',
             'sage_green': '灰绿色', 'muted_rose': '淡粉色'}


def product_tag(category, description):
    text = description.lower()
    if category == 'lamp':
        # Match the shade, not a cylindrical base elsewhere in the description.
        if re.search(r'cylindrical(?:\s+(?:fabric|woven))?\s+shade', text):
            return 'cylindrical'
        if re.search(r'tapered(?:\s+fabric)?\s+(?:fabric\s+)?shade', text):
            return 'tapered'
    elif category == 'pillow':
        if 'strip' in text:
            return 'striped'
        if any(word in text for word in ('geometric', 'hexagonal', 'diamond', 'circular')):
            return 'geometric'
    return None  # Still a valid reference; not an unlabelled replacement target.


def studio_background(color, size=256):
    if color not in PALETTES:
        raise ValueError('Unknown simple background')
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)/(size-1)
    # Fixed soft studio illumination, no scene objects, windows, silhouettes,
    # random blocks or independent brightness/direction instructions.
    halo = np.exp(-((xx-.45)**2/.35+(yy-.35)**2/.7))
    floor = 1/(1+np.exp(-(yy-.84)*22))
    shadow = np.exp(-((xx-.5)**2/.038+(yy-.926)**2/.0006))
    gain = .86+.12*halo-.07*floor-.15*shadow
    pixels = np.array(PALETTES[color], dtype=np.float32)[None, None]*gain[:, :, None]
    return Image.fromarray(pixels.clip(0, 255).round().astype(np.uint8))


def simple_prompt(category, mode, tag, color):
    noun = 'lamp' if category == 'lamp' else 'cushion'
    background = color.replace('_', ' ')
    if mode == 'background':
        return f'Change only the background to {background}.'
    target = LABELS[category][tag]
    if mode == 'object':
        return f'Replace only the {noun} with {target}.'
    if mode == 'joint':
        return f'Use {target} on a {background} background.'
    raise ValueError('Unknown edit mode')


def simple_pairs(prior_rows, descriptions, edits_per_mode=None):
    edits_per_mode = edits_per_mode or {'train': 5, 'val': 2}
    selected = collections.defaultdict(dict)
    for row in prior_rows:
        if row['split'] not in ('train', 'val'):
            raise ValueError('Do not open held-out data during TRAIN/DEV preparation')
        for role in ('source', 'target'):
            p = row[role]
            selected[row['split']].setdefault(p['sample_id'], p)
    result = []
    for split, products in selected.items():
        canonical, groups = {}, collections.defaultdict(list)
        for p in sorted(products.values(), key=lambda x: x['sample_id']):
            canonical.setdefault(p['product_id'], p)
        for p in canonical.values():
            tag = product_tag(p['category'], descriptions[p['sample_id']])
            if tag:
                groups[p['category'], tag].append(p)
        for category, labels in LABELS.items():
            for tag in labels:
                if not groups[category, tag]:
                    raise ValueError(f'Missing {split} {category}/{tag} target family')
        target_map = {}
        for category, labels in LABELS.items():
            for tag in labels:
                eligible = [p for p in canonical.values() if p['category'] == category
                            and product_tag(category, descriptions[p['sample_id']]) != tag]
                for k, product in enumerate(eligible):
                    target_map[product['product_id'], tag] = groups[category, tag][k % len(groups[category, tag])]
        for source in sorted(products.values(), key=lambda x: x['sample_id']):
            category = source['category']; labels = list(LABELS[category])
            original_tag = product_tag(category, descriptions[source['sample_id']])
            opposite = [label for label in labels if label != original_tag]
            color_offset = seed(split, source['sample_id'], 'simple-palette') % len(PALETTES)
            for j in range(edits_per_mode[split]):
                source_color = list(PALETTES)[(color_offset+j) % len(PALETTES)]
                target_color = list(PALETTES)[(color_offset+j+1+j % 3) % len(PALETTES)]
                tag = opposite[j % len(opposite)]
                pool = [p for p in groups[category, tag] if p['product_id'] != source['product_id']]
                if not pool:
                    raise ValueError('Replacement target must be a different product')
                # Same reference product + requested family always maps to the
                # same representative; background color must not select a SKU.
                target = target_map[source['product_id'], tag]
                for mode in ('background', 'object', 'joint'):
                    effective_color = source_color if mode == 'object' else target_color
                    intended = source if mode == 'background' else target
                    prompt = simple_prompt(category, mode, tag, effective_color)
                    object_phrase = descriptions[source['sample_id']].replace(', entirely visible', '') if mode == 'background' else LABELS[category][tag]
                    # This full TRAIN-only teacher caption is not the user
                    # instruction and is never passed to deployed Qwen.
                    caption = f'Full view of {object_phrase}, centered on a {effective_color.replace("_", " ")} studio background.'
                    zh = ('背景换成'+CN_COLORS[effective_color] if mode == 'background' else
                          '换成'+CN_LABELS[tag] if mode == 'object' else
                          '换成'+CN_LABELS[tag]+'，背景换成'+CN_COLORS[effective_color])
                    result.append({'sample_id': f'{split}:simple:{source["sample_id"]}:{mode}:{j}',
                        'split': split, 'category': category, 'mode': mode, 'source': source, 'target': intended,
                        'render_protocol': PROTOCOL, 'source_background': source_color, 'target_background': effective_color,
                        'requested_family': tag if mode != 'background' else None, 'source_family': original_tag,
                        'edit_count': 2 if mode == 'joint' else 1, 'prompt': prompt, 'prompt_zh': zh,
                        'teacher_caption': caption, 'noise_seed': 1042,
                        'source_scene': ['studio', 'neutral', 'bright', 'fixed'],
                        'target_scene': ['studio', 'neutral', 'bright', 'fixed']})
    validate_roles(result)
    for row in result:
        if len(row['prompt'].split()) > 14 or re.search(r'\bB0[A-Z0-9]{8}\b|\bSKU\b|\bAmazon\b', row['prompt'], re.I):
            raise ValueError('Instructions must be short and free of product IDs')
        if row['mode'] != 'background' and row['source_family'] == row['requested_family']:
            raise ValueError('Requested family must visibly change')
        if row['mode'] != 'object' and row['source_background'] == row['target_background']:
            raise ValueError('Background instruction must actually change background')
    return result


def render_simple_pair(row, size=256):
    if row['render_protocol'] != PROTOCOL:
        raise ValueError('Not a simple studio row')
    source, alpha = normalized_product(row['source'], size)
    target, target_alpha = normalized_product(row['target'], size)
    before = Image.composite(source, studio_background(row['source_background'], size), alpha)
    after = Image.composite(target, studio_background(row['target_background'], size), target_alpha)
    return before, after


def prepare(source, profile, output):
    if output.exists():
        raise FileExistsError('Immutable dataset: inspect, never overwrite')
    cfg = json.loads(profile.read_text())
    for name in ('annotated_pairs.json', 'product_descriptions.json'):
        if sha256(source/name) != cfg['parent_sha256'][name]:
            raise ValueError('Reviewed source identity differs')
    descriptions = json.loads((source/'product_descriptions.json').read_text())
    rows = simple_pairs(json.loads((source/'annotated_pairs.json').read_text()), descriptions)
    audit = validate_roles(rows)
    if audit['pairs'] != {'train': 960, 'val': 96}:
        raise ValueError('Approved data budget changed')
    output.mkdir(parents=True);artifacts = {};seen_inputs = {}
    for i, row in enumerate(rows):
        for role, im in zip(('reference', 'target'), render_simple_pair(row)):
            path = output/'images'/row['split']/role/(f'{i:05d}.png');path.parent.mkdir(parents=True, exist_ok=True)
            im.save(path);artifacts[str(path.relative_to(output))] = sha256(path)
            if role == 'reference':
                identity = artifacts[str(path.relative_to(output))], row['prompt'], row['noise_seed']
            else:
                digest = artifacts[str(path.relative_to(output))]
                if identity in seen_inputs and seen_inputs[identity] != digest:
                    raise ValueError('Identical RGB/prompt/seed has conflicting target images')
                seen_inputs[identity] = digest
    family_products = collections.defaultdict(set)
    for row in rows:
        if row['mode'] != 'background':
            family_products[row['split'], row['category'], row['requested_family']].add(row['target']['product_id'])
    audit.update(protocol=PROTOCOL, parent_data=str(source.resolve()), parent_sha256=cfg['parent_sha256'],
        target_family_product_counts={'/'.join(k): len(v) for k, v in family_products.items()},
        inventories={s: {'products': len({r['source']['product_id'] for r in rows if r['split'] == s}),
                         'views': len({r['source']['sample_id'] for r in rows if r['split'] == s})} for s in ('train', 'val')},
        words={'maximum': max(len(r['prompt'].split()) for r in rows), 'minimum': min(len(r['prompt'].split()) for r in rows)},
        one_or_two_edits_only=True, old_heldout_240_opened=False, gpu_used=False,
        targets_are_prepared_supervision=True, no_GT_or_mask_in_inference=True,
        model_predictions_used_for_selection=False, native_resolution=256,
        duplicate_input_conflicting_targets=False,
        metrics_scope='960 TRAIN / 96 selection DEV. New protocol; do not compare means to old complex-room task.',
        family_audit_scope='Retain reviewed product/family/RGBA identities from the parent cohort; public pretraining exposure unknown.')
    for name, obj in [('annotated_pairs.json', rows), ('pilot_pairs.json', rows),
                      ('product_descriptions.json', descriptions), ('data_audit.json', audit),
                      ('pairs_manifest.json', {'config': cfg, 'audit': audit, 'pairs': rows}),
                      ('image_sha256.json', artifacts)]:
        write_json(output/name, obj)
    # Representative data examples, not model output. Keep 256px native tiles.
    for split in ('train', 'val'):
        for category in ('lamp', 'pillow'):
            for mode in ('background', 'object', 'joint'):
                pool = [r for r in rows if (r['split'], r['category'], r['mode']) == (split, category, mode)]
                picked = [];seen = set()
                for row in pool:
                    if row['source']['product_id'] not in seen:
                        picked.append(row);seen.add(row['source']['product_id'])
                    if len(picked) == 3:
                        break
                panel = Image.new('RGB', (520, 322*len(picked)), 'white');draw = ImageDraw.Draw(panel)
                for j, row in enumerate(picked):
                    y = j*322;draw.text((4, y+2), split.upper()+' '+category+'/'+mode+' | DATA PAIRS, NOT MODEL OUTPUT', fill='black')
                    draw.multiline_text((4, y+18), '\n'.join(textwrap.wrap(row['prompt'], 82)), fill='black')
                    draw.text((4, y+49), 'REFERENCE', fill='black');draw.text((264, y+49), 'SUPERVISED TARGET', fill='black')
                    for k, im in enumerate(render_simple_pair(row)):
                        panel.paste(im, (4+260*k, y+66))
                panel.save(output/(split+'_'+category+'_'+mode+'.png'))
    write_json(output/'report.json', {'complete': True, 'stage': 'CPU_data_preparation_only', 'audit': audit,
        'pair_sha256': sha256(output/'annotated_pairs.json'), 'profile_sha256': sha256(profile),
        'source_commit': os.environ.get('T12_SOURCE_COMMIT'), 'source_sha256': sha256(Path(__file__)),
        'no_model_training_or_inference': True})
    return audit


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('source', 'profile', 'output'):
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--stage', choices=('prepare', 'cache'), default='prepare')
    p.add_argument('--qwen', type=Path);p.add_argument('--sd', type=Path);p.add_argument('--gpu-uuid')
    a = p.parse_args()
    if a.stage == 'prepare':
        print(json.dumps(prepare(a.source, a.profile, a.output), indent=2));return
    report = json.loads((a.output/'report.json').read_text())
    if not report['complete'] or sha256(a.profile) != report['profile_sha256'] or sha256(a.output/'annotated_pairs.json') != report['pair_sha256']:
        raise ValueError('Prepared simple data identity differs')
    if not a.qwen or not a.sd or os.environ.get('CUDA_VISIBLE_DEVICES') != a.gpu_uuid:
        raise ValueError('Explicit approved GPU and public backbones required')
    import torch,sys,time
    from .qwen_vl_turbo_run import cache
    torch.set_num_threads(4);torch.manual_seed(1042);torch.cuda.set_per_process_memory_fraction(.65, 0)
    write_json(a.output/'cache_execution.json', {'pid': os.getpid(), 'argv': sys.argv, 'started_unix': time.time(),
        'source_commit': os.environ.get('T12_SOURCE_COMMIT'), 'gpu_uuid': a.gpu_uuid,
        'source_sha256': sha256(Path(__file__)), 'original_pair_sha256': report['pair_sha256']})
    rows = json.loads((a.output/'annotated_pairs.json').read_text())
    digest = cache(a, {'image_size': 256, 'use_all_pairs': True, 'uint8_image_cache': True,
                      'clip_cache_batch_size': 16}, rows, torch.device('cuda:0'))
    write_json(a.output/'cache_complete.json', {'complete': True, 'sha256': digest, 'samples': len(rows),
        'source_commit': os.environ.get('T12_SOURCE_COMMIT'), 'reference_RGB_and_short_text_only': True})


if __name__ == '__main__':
    main()
