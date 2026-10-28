"""CPU-only, one/two-edit ABO pairs in a fixed, simple procedural room.

Background edits change only the wall palette; product substitution leaves the
background pixels unchanged. Compositing and masks are data preparation only.
No held-out cohort, generated-model score or teacher photograph is consulted.
"""
import argparse
import collections
import hashlib
import json
import math
import os
from pathlib import Path
import re
import textwrap

import numpy as np
from PIL import Image, ImageDraw

from .qwen_vl_simple_edits import LABELS, product_tag
from .qwen_vl_scene_render import unmatte_white
from .qwen_vl_turbo_data import normalized_product, seed, sha256, validate_roles, write_json


PROTOCOL = 'controlled_room_edit_v1'
PALETTES = {'light_gray': (193, 192, 189), 'warm_beige': (211, 195, 173),
            'pale_blue': (174, 192, 207), 'sage_green': (181, 194, 175),
            'muted_rose': (205, 180, 179)}
DEFAULT_INVENTORY = {'train': {'products': 44, 'views': 76},
                     'val': {'products': 8, 'views': 16}}


def room_background(color, size=256, return_wall_mask=False):
    """Fixed window/table/light; palette affects only the exposed wall pixels."""
    if color not in PALETTES or size < 16:
        raise ValueError('Unknown wall palette or invalid room size')
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)/(size-1)
    fixed_light = .92+.055*(1-xx)-.055*yy
    wall = np.asarray(PALETTES[color], dtype=np.float32)[None, None]*fixed_light[:, :, None]
    array = wall.clip(0, 255).round().astype(np.uint8)
    wall_mask = np.ones((size, size), dtype=bool)
    horizon = int(.70*size)
    table = np.asarray((112, 106, 98), dtype=np.float32)[None, None]*(.95+.09*yy[:, :, None])
    table = np.broadcast_to(table, (size, size, 3)).copy()
    contact = np.exp(-((xx-.50)**2/.035+(yy-.946)**2/.0007))
    table *= 1-.12*contact[:, :, None]
    array[horizon:] = table[horizon:].clip(0, 255).round().astype(np.uint8)
    wall_mask[horizon:] = False
    image = Image.fromarray(array)
    draw = ImageDraw.Draw(image)
    box = (int(.045*size), int(.08*size), int(.235*size), int(.55*size))
    draw.rectangle(box, fill=(218, 218, 207), outline=(99, 102, 102), width=max(1, size//128))
    draw.line(((box[0]+box[2])//2, box[1], (box[0]+box[2])//2, box[3]), fill=(142, 144, 137))
    draw.line((box[0], int(.315*size), box[2], int(.315*size)), fill=(142, 144, 137))
    wall_mask[box[1]:box[3]+1, box[0]:box[2]+1] = False
    draw.line((0, horizon, size-1, horizon), fill=(95, 92, 87), width=1)
    result = (image, wall_mask) if return_wall_mask else image
    return result


def controlled_prompt(category, mode, family, wall_color):
    color = wall_color.replace('_', ' ')
    noun = 'lamp' if category == 'lamp' else 'cushion'
    if mode == 'background':
        return f'Change only the wall to {color}.'
    phrase = LABELS[category][family]
    if mode == 'object':
        return f'Replace only the {noun} with {phrase}.'
    if mode == 'joint':
        return f'Use {phrase} and a {color} wall.'
    raise ValueError('Unknown editing task')


def inventories(prior_rows):
    views = collections.defaultdict(dict)
    for row in prior_rows:
        if row['split'] not in ('train', 'val'):
            raise ValueError('TRAIN/DEV preparation cannot open a held-out cohort')
        for role in ('source', 'target'):
            p = row[role]
            if p['category'] not in LABELS:
                raise ValueError('Only the reviewed lamp/pillow cohort is admitted')
            key = row['split'], p['category']
            previous = views[key].setdefault(p['sample_id'], p)
            if any(previous.get(k) != p.get(k) for k in ('product_id', 'rgba_sha256', 'image_path', 'mask_path')):
                raise ValueError('One approved view has conflicting identities')
    return views


def literal_prompt_statistics(rows):
    groups = collections.defaultdict(collections.Counter)
    for row in rows:
        groups[row['split'], row['category'], row['mode'], row['prompt']][row['target']['product_id']] += 1
    records = []
    for (split, category, mode, prompt), count in sorted(groups.items()):
        total = sum(count.values()); probabilities = [n/total for n in count.values()]
        records.append({'split': split, 'category': category, 'mode': mode, 'prompt': prompt,
            'pairs': total, 'unique_target_products': len(count), 'target_product_counts': dict(sorted(count.items())),
            'maximum_target_share': max(probabilities),
            'conditional_target_entropy_bits': -sum(p*math.log2(p) for p in probabilities)})
    return records


def controlled_pairs(prior_rows, descriptions, cfg):
    views = inventories(prior_rows)
    counts = cfg.get('instruction_conditions_per_category',
        cfg.get('reference_contexts_per_category', {'train': 160, 'val': 16}))
    colors = list(cfg.get('background_colors', PALETTES))
    if len(colors) != 5 or len(set(colors)) != 5 or set(colors) != set(PALETTES):
        raise ValueError('Exactly five reviewed wall palettes required')
    rows = []
    for split in ('train', 'val'):
        for category, labels in LABELS.items():
            available = sorted(views[split, category].values(), key=lambda p: p['sample_id'])
            if not available or counts[split] < len(available):
                raise ValueError('Condition budget cannot retain all reviewed views')
            canonical = {}
            for p in available:
                canonical.setdefault(p['product_id'], p)
            pools = collections.defaultdict(list)
            tags = {}
            for product, p in sorted(canonical.items()):
                tag = product_tag(category, descriptions[p['sample_id']]); tags[product] = tag
                if tag is not None:
                    pools[tag].append(p)
            if any(not pools[label] for label in labels):
                raise ValueError('Missing reviewed target family in '+split+'/'+category)
            if split == 'train' and any(len(pools[label]) < cfg.get('minimum_train_target_products_per_object_prompt', 3) for label in labels):
                raise ValueError('TRAIN target family has fewer than the required distinct products')
            target_map = {}
            for family in labels:
                loads = collections.Counter()
                eligible = sorted(canonical) if split == 'train' else sorted(p for p in canonical if tags[p] != family)
                family_ids = [p['product_id'] for p in pools[family]]
                # Give same-family TRAIN sources a cyclic derangement first;
                # unrestricted sources then balance the remaining target load.
                # This avoids the last self-exclusion forcing one target to
                # receive extra assignments in an otherwise balanced pool.
                restricted = [p for p in eligible if p in family_ids]
                for source_product in restricted:
                    if len(family_ids) < 2:
                        raise ValueError('Same-family replacement needs another real product')
                    target = pools[family][(family_ids.index(source_product)+1) % len(family_ids)]
                    target_map[source_product, family] = target; loads[target['product_id']] += 1
                for source_product in [p for p in eligible if p not in restricted]:
                    # Background, view, instruction variant and noise do not
                    # choose which member of the target family is supervised.
                    candidates = [p for p in pools[family] if p['product_id'] != source_product]
                    if not candidates:
                        raise ValueError('Requested family has no different-product replacement')
                    target = min(candidates, key=lambda p: (loads[p['product_id']], p['product_id']))
                    target_map[source_product, family] = target
                    loads[target['product_id']] += 1
            family_rank = {product: rank for rank, product in enumerate(sorted(canonical))}
            if split == 'train':
                base = (counts[split] // (2*len(available)))*2
                remainder = counts[split]-base*len(available)
                if base < 2 or remainder % 2:
                    raise ValueError('TRAIN conditions must admit paired instructions for every reviewed view')
                per_view = [base+(2 if k < remainder//2 else 0) for k in range(len(available))]
            else:
                base, remainder = divmod(counts[split], len(available))
                per_view = [base+(k < remainder) for k in range(len(available))]
            if any((n//2 if split == 'train' else n) > len(colors) for n in per_view):
                raise ValueError('Distinct reference wall palettes per view exceed the reviewed five-color bank')
            selections = [(source, j) for source, n in zip(available, per_view) for j in range(n)]
            for source, j in selections:
                repeat = j//2 if split == 'train' else j
                branch = j % 2
                original = tags[source['product_id']]
                alternatives = list(labels) if split == 'train' else [family for family in labels if family != original]
                family = alternatives[branch % len(alternatives)] if split == 'train' else alternatives[(family_rank[source['product_id']]+j) % len(alternatives)]
                offset = seed(cfg.get('seed', 1042), split, source['sample_id'], 'wall-palette') % len(colors)
                source_color = colors[(offset+repeat) % len(colors)]
                shift = 1+seed(cfg.get('seed', 1042), split, source['sample_id'], 'wall-edit') % (len(colors)-2)
                target_color = colors[(offset+repeat+shift+branch) % len(colors)]
                target = target_map[source['product_id'], family]
                context_id = f'{split}:room:{source["sample_id"]}:{source_color}'
                noise_seed = seed(cfg.get('seed', 1042), split, source['sample_id'], source_color, 'reference-context') % (2**31)
                for mode in ('background', 'object', 'joint'):
                    effective_color = source_color if mode == 'object' else target_color
                    intended = source if mode == 'background' else target
                    prompt = controlled_prompt(category, mode, family, effective_color)
                    phrase = descriptions[source['sample_id']] if mode == 'background' else LABELS[category][family]
                    rows.append({'sample_id': context_id+':instruction'+str(branch)+':'+mode, 'split': split, 'category': category, 'mode': mode,
                        'source': source, 'target': intended, 'render_protocol': PROTOCOL,
                        'reference_context_id': context_id, 'source_wall_color': source_color, 'target_wall_color': effective_color,
                        'source_background': source_color, 'target_background': effective_color,
                        'source_scene': ['controlled_room', 'neutral', 'bright', 'left'],
                        'target_scene': ['controlled_room', 'neutral', 'bright', 'left'],
                        'requested_family': None if mode == 'background' else family, 'source_family': original,
                        'edit_count': 2 if mode == 'joint' else 1, 'prompt': prompt,
                        'generic_instruction_id': '/'.join([mode, category, '' if mode == 'background' else family,
                            '' if mode == 'object' else effective_color]),
                        'teacher_caption': f'One {phrase.removeprefix("a ")}, in a simple room with a {effective_color.replace("_", " ")} wall, side window and gray tabletop.',
                        'noise_seed': noise_seed})
    validate_roles(rows)
    minimum = cfg.get('minimum_train_target_products_per_object_prompt', 3)
    for record in literal_prompt_statistics(rows):
        if record['split'] == 'train' and record['mode'] == 'object' and record['unique_target_products'] < minimum:
            raise ValueError('One literal TRAIN object instruction has too few target products: '+str(record))
    for row in rows:
        if len(row['prompt'].split()) > 16 or re.search(r'\bB0[A-Z0-9]{8}\b|\bSKU\b|\bAmazon\b', row['prompt'], re.I):
            raise ValueError('Instructions must be at most16 words and free of product IDs')
        if row['mode'] != 'background' and row['target']['product_id'] == row['source']['product_id']:
            raise ValueError('Product substitution must use a different product, including same-family TRAIN edits')
        if row['mode'] != 'object' and row['source_wall_color'] == row['target_wall_color']:
            raise ValueError('Wall edit must change the wall palette')
    return rows


def composite(product, color, size=256):
    rgb, alpha = normalized_product(product, size)
    foreground, a = unmatte_white(rgb, alpha)
    if not np.any(a > .1):
        raise ValueError('Empty genuine ABO foreground')
    background = np.asarray(room_background(color, size), dtype=np.float32)
    result = foreground*a[:, :, None]+background*(1-a[:, :, None])
    return Image.fromarray(result.clip(0, 255).round().astype(np.uint8))


def render_controlled_pair(row, size=256):
    if row.get('render_protocol') != PROTOCOL:
        raise ValueError('Wrong controlled-room protocol')
    return (composite(row['source'], row['source_wall_color'], size),
            composite(row['target'], row['target_wall_color'], size))


def check_reference_target(identity_targets, reference_rgb, prompt, noise_seed, target_rgb):
    # Stronger than the exact-noise check: do not use source-specific random
    # states to hide incompatible supervision for otherwise identical inputs.
    identity = hashlib.sha256(reference_rgb).hexdigest(), prompt
    target_hash = hashlib.sha256(target_rgb).hexdigest()
    previous = identity_targets.setdefault(identity, target_hash)
    if previous != target_hash:
        raise ValueError('Identical reference RGB and instruction have conflicting targets, regardless of noise seed')
    return identity[0], target_hash


def pixel_edit_mask(before, after):
    """Supervised label only: exact per-pixel RGB inequality, threshold zero."""
    a = np.asarray(before.convert('RGB')); b = np.asarray(after.convert('RGB'))
    if a.shape != b.shape:
        raise ValueError('Reference/target image dimensions differ')
    changed = np.any(a != b, axis=-1)
    if not changed.any():
        raise ValueError('No-op supervision is not an editing pair')
    return Image.fromarray(changed.astype(np.uint8)*255)


def _preview_panels(output, rows):
    selected = {}
    for split in ('train', 'val'):
        for category in LABELS:
            for mode in ('background', 'object', 'joint'):
                pool = [r for r in rows if (r['split'], r['category'], r['mode']) == (split, category, mode)]
                picked = []; seen = set()
                for row in pool:
                    if row['source']['product_id'] not in seen:
                        seen.add(row['source']['product_id']); picked.append(row)
                    if len(picked) == 3:break
                panel = Image.new('RGB', (512, 320*len(picked)), 'white'); draw = ImageDraw.Draw(panel)
                for j, row in enumerate(picked):
                    y = j*320
                    draw.text((4, y+2), split.upper()+' '+category+'/'+mode+' | DATA PAIRS, NOT MODEL OUTPUT', fill='black')
                    draw.multiline_text((4, y+18), '\n'.join(textwrap.wrap(row['prompt'], 82)), fill='black')
                    draw.text((4, y+46), 'REFERENCE', fill='black'); draw.text((260, y+46), 'SUPERVISED TARGET', fill='black')
                    for k, image in enumerate(render_controlled_pair(row)):
                        panel.paste(image, (256*k, y+64))
                panel.save(output/(split+'_'+category+'_'+mode+'.png'))
                selected[split, category, mode] = picked[0]
    overview = Image.new('RGB', (1024, 1004), 'white'); draw = ImageDraw.Draw(overview)
    draw.text((4, 4), 'PREPARED TRAIN DATA ONLY - NOT MODEL GENERATIONS. Each pair: reference / supervised target.', fill='black')
    for y, mode in enumerate(('background', 'object', 'joint')):
        for x, category in enumerate(LABELS):
            row = selected['train', category, mode]; top = 44+y*320; left = 512*x
            draw.text((left+4, top+2), category+'/'+mode+' | '+str(row['edit_count'])+' requested edit(s)', fill='black')
            draw.multiline_text((left+4, top+18), '\n'.join(textwrap.wrap(row['prompt'], 80)), fill='black')
            draw.text((left+4, top+46), 'REFERENCE', fill='black'); draw.text((left+260, top+46), 'SUPERVISED TARGET', fill='black')
            for k, image in enumerate(render_controlled_pair(row)):
                overview.paste(image, (left+256*k, top+64))
    overview.save(output/'data_overview.png')


def prepare(source, profile, output):
    if output.exists():
        raise FileExistsError('Preserve existing datasets; require a new run directory')
    cfg = json.loads(profile.read_text(encoding='utf-8'))
    if cfg.get('schema') != PROTOCOL:
        raise ValueError('Explicit controlled-room profile required')
    for name in ('annotated_pairs.json', 'product_descriptions.json'):
        if sha256(source/name) != cfg['parent_sha256'][name]:
            raise ValueError('Reviewed rich parent identity differs: '+name)
    prior = json.loads((source/'annotated_pairs.json').read_text(encoding='utf-8'))
    descriptions = json.loads((source/'product_descriptions.json').read_text(encoding='utf-8'))
    rows = controlled_pairs(prior, descriptions, cfg)
    inventory = {}
    for split in ('train', 'val'):
        subset = [r for r in rows if r['split'] == split]
        inventory[split] = {'products': len({r['source']['product_id'] for r in subset}),
                            'views': len({r['source']['sample_id'] for r in subset})}
    if inventory != cfg.get('expected_source_inventory', DEFAULT_INVENTORY):
        raise ValueError('Reviewed source product/view inventory changed: '+str(inventory))
    audit = validate_roles(rows)
    if audit['pairs'] != {'train': 960, 'val': 96}:
        raise ValueError('The approved960/96 budget changed')
    output.mkdir(parents=True); artifacts = {}; input_targets = {}; contexts = {}; inference = []; reference_hashes = collections.defaultdict(set)
    context_rows = collections.defaultdict(list)
    for i, row in enumerate(rows):
        before, after = render_controlled_pair(row)
        reference_hash, _ = check_reference_target(input_targets, before.tobytes(), row['prompt'], row['noise_seed'], after.tobytes())
        context_key = row['split'], row['reference_context_id']
        context_value = reference_hash, row['noise_seed']
        if contexts.setdefault(context_key, context_value) != context_value:
            raise ValueError('Paired instructions/tasks do not share one source RGB and source-derived noise')
        context_rows[context_key].append(row)
        reference_hashes[row['split'], row['category']].add(reference_hash)
        for role, image in (('reference', before), ('target', after)):
            path = output/'images'/row['split']/role/f'{i:05d}.png'; path.parent.mkdir(parents=True, exist_ok=True); image.save(path)
            artifacts[str(path.relative_to(output))] = sha256(path)
        edit_mask = pixel_edit_mask(before, after)
        mask_path = output/'images'/row['split']/'edit_mask'/f'{i:05d}.png'; mask_path.parent.mkdir(parents=True, exist_ok=True); edit_mask.save(mask_path)
        row['edit_mask_path'] = str(mask_path.resolve()); row['edit_mask_sha256'] = sha256(mask_path)
        artifacts[str(mask_path.relative_to(output))] = row['edit_mask_sha256']
        if np.any(np.asarray(before)[np.asarray(edit_mask) == 0] != np.asarray(after)[np.asarray(edit_mask) == 0]):
            raise ValueError('A target pixel changed outside its supervised edit mask')
        inference.append({'image': str((output/'images'/row['split']/'reference'/f'{i:05d}.png').resolve()),
            'prompt': row['prompt'], 'sample_id': row['sample_id'], 'category': row['category'], 'mode': row['mode']})
    requested_conditions = cfg.get('instruction_conditions_per_category',
        cfg.get('reference_contexts_per_category', {'train': 160, 'val': 16}))
    requested_contexts = {split: count//2 if split == 'train' else count for split, count in requested_conditions.items()}
    if any(len(values) != requested_contexts[split] for (split, category), values in reference_hashes.items()):
        raise ValueError('Distinct prepared RGB contexts are fewer than requested; inspect duplicate views before changing protocol')
    contrasts = []
    for (split, context_id), items in context_rows.items():
        for mode in ('background', 'object', 'joint'):
            alternatives = [r for r in items if r['mode'] == mode]
            if split == 'train':
                if len(alternatives) != 2 or len({r['prompt'] for r in alternatives}) != 2:
                    raise ValueError('Each TRAIN reference/task requires two distinct editing instructions')
                if np.array_equal(np.asarray(render_controlled_pair(alternatives[0])[1]), np.asarray(render_controlled_pair(alternatives[1])[1])):
                    raise ValueError('TRAIN counterfactual instructions have identical supervised targets')
                contrasts.append({'reference_context_id': context_id, 'mode': mode,
                    'sample_ids': [r['sample_id'] for r in alternatives], 'prompts': [r['prompt'] for r in alternatives]})
    stats = literal_prompt_statistics(rows)
    audit.update(protocol=PROTOCOL, expected_counts={'train': 960, 'val': 96}, parent_data=str(source.resolve()),
        profile_sha256=sha256(profile),
        parent_sha256=cfg['parent_sha256'], inventories=inventory, literal_prompt_target_statistics=stats,
        reference_contexts={split: {category: len(reference_hashes[split, category]) for category in LABELS} for split in ('train', 'val')},
        instruction_conditions_per_category=requested_conditions, TRAIN_paired_counterfactual_groups=len(contrasts),
        DEV_paired_counterfactuals=False,
        DEV_counterfactual_limitation='Original DEV contains only one tapered-lamp and one geometric-pillow product; preserve all8 reviewed products, use opposite-family replacement and two distinct reference backgrounds per view. Do not introduce same-SKU no-op targets, duplicate conflicting inputs, new DEV products or held-out products.',
        tasks={split: dict(collections.Counter(r['mode'] for r in rows if r['split'] == split)) for split in ('train', 'val')},
        test_selection_authorized=True, evaluation_label='development_test', independent_TEST540_opened=False,
        native_resolution=256, gpu_used=False, no_model_predictions_used=True, targets_are_prepared_supervision=True,
        no_GT_mask_caption_SKU_in_inference=True, one_or_two_requested_edits_only=True,
        source_context_noise_independent_of_target_and_mode=True, duplicate_input_conflicting_targets=False,
        edit_mask_semantics='Exact RGB changed-pixel support, any unequal channel, threshold0. Offline loss/metric label only; never in inference payload.',
        changed_pixels_outside_supervised_edit_masks=0,
        maximum_instruction_words=max(len(r['prompt'].split()) for r in rows),
        geometry='Fixed left window, opaque tabletop, fixed light field. Only exposed wall palette changes in background mode.',
        compositing='Genuine approved ABO alpha, no erosion; undo white JPEG matte before compositing. No teacher-photo backgrounds.',
        limits='Generic object-family substitution may legitimately alter unmentioned object attributes. One supervised member per given reference/family is not the only valid output. Independent TEST540 stays sealed; DEV96 is selection data.')
    (output/'profile.json').write_bytes(profile.read_bytes())
    for name, value in (('annotated_pairs.json', rows), ('pilot_pairs.json', rows), ('product_descriptions.json', descriptions),
        ('data_audit.json', audit), ('pairs_manifest.json', {'config': cfg, 'audit': audit, 'pairs': rows}),
        ('image_sha256.json', artifacts), ('inference_inputs.json', inference),
        ('TRAIN_counterfactual_groups.json', contrasts),
        ('inference_train_inputs.json', [v for v, r in zip(inference, rows) if r['split'] == 'train']),
        ('inference_dev_inputs.json', [v for v, r in zip(inference, rows) if r['split'] == 'val'])):
        write_json(output/name, value)
    _preview_panels(output, rows)
    write_json(output/'report.json', {'complete': True, 'stage': 'CPU_data_preparation_only', 'audit': audit,
        'pair_sha256': sha256(output/'annotated_pairs.json'), 'profile_sha256': sha256(profile),
        'source_commit': os.environ.get('T12_SOURCE_COMMIT'), 'source_sha256': sha256(Path(__file__)),
        'no_model_training_or_inference': True, 'independent_TEST540_opened': False})
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('source', 'profile', 'output'):
        parser.add_argument('--'+key, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.profile, args.output), indent=2))


if __name__ == '__main__':main()
