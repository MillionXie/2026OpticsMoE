"""Audit completed controlled-room TRAIN pairs; no editor or GPU execution.

Compare two prompts on the identical reference RGB and source-context seed.
Output change alone is not correctness: common retained pixels should remain
unchanged across prompts AND agree with the reference. These are TRAIN fits,
not independent evaluation or evidence that a semantic cause is established.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import textwrap
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont


PROTOCOL = 'controlled_room_edit_v1'
COUNTS = {'train': 960, 'val': 96}
MODES = ('background', 'object', 'joint')
CATEGORIES = ('lamp', 'pillow')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def rgb_array(value):
    value = np.asarray(value)
    if value.shape != (256, 256, 3) or value.dtype != np.uint8:
        raise ValueError('Unmodified native uint8 RGB256 pixels required')
    return value


def error_scores(first, second, support=None):
    first, second = rgb_array(first), rgb_array(second)
    if support is None:
        support = np.ones((256, 256), dtype=bool)
    support = np.asarray(support)
    if support.shape != (256, 256) or support.dtype != bool or not support.any():
        raise ValueError('Nonempty native boolean metric support required')
    difference = (first.astype(np.float64)-second.astype(np.float64))[support]/255
    mse = float(np.square(difference).mean())
    return {'pixels': int(support.sum()), 'L1_RGB_01': float(np.abs(difference).mean()),
            'MSE_RGB_01': mse, 'PSNR_dB': -10*math.log10(max(mse, 1e-12))}


def pair_metrics(reference, targets, predictions, edit_masks):
    reference = rgb_array(reference)
    if len(targets) != 2 or len(predictions) != 2 or len(edit_masks) != 2:
        raise ValueError('Exactly two paired instructions are required')
    masks = []
    for target, prediction, mask in zip(targets, predictions, edit_masks):
        target, prediction, mask = rgb_array(target), rgb_array(prediction), np.asarray(mask)
        if (mask.shape != (256, 256) or mask.dtype != np.uint8 or
                not np.isin(mask, (0, 255)).all()):
            raise ValueError('Exact native binary supervision masks required')
        changed = np.any(reference != target, axis=2)
        if not np.array_equal(changed, mask > 0) or not changed.any():
            raise ValueError('Edit mask must exactly include every changed target pixel')
        masks.append(mask > 0)
    common = ~(masks[0] | masks[1])
    gt_common = error_scores(targets[0], targets[1], common)
    gt_keep = [error_scores(target, reference, ~mask) for target, mask in zip(targets, masks)]
    if gt_common['MSE_RGB_01'] != 0 or any(value['MSE_RGB_01'] != 0 for value in gt_keep):
        raise ValueError('GT retained-region controls must be exactly zero')
    whole = error_scores(predictions[0], predictions[1])
    gt_whole = error_scores(targets[0], targets[1])
    if gt_whole['MSE_RGB_01'] == 0:
        raise ValueError('Paired distinct instructions have identical GT targets')
    return {'common_retained_output_delta': error_scores(predictions[0], predictions[1], common),
        'retained_error_vs_reference': [error_scores(prediction, reference, ~mask)
                                        for prediction, mask in zip(predictions, masks)],
        'whole_output_delta_diagnostic_only': whole,
        'GT_common_retained_delta_control': gt_common,
        'GT_retained_error_controls': gt_keep,
        'GT_whole_target_delta_diagnostic_only': gt_whole}


def paired_row_contract(rows, reference_context_id, mode):
    if (len(rows) != 2 or any(row['split'] != 'train' or row.get('render_protocol') != PROTOCOL
            or row['mode'] != mode or row['reference_context_id'] != reference_context_id for row in rows)
            or len({row['category'] for row in rows}) != 1
            or len({row['source']['sample_id'] for row in rows}) != 1
            or len({row['noise_seed'] for row in rows}) != 1
            or len({row['prompt'] for row in rows}) != 2):
        raise ValueError('Require distinct TRAIN prompts on one reference context and shared source seed')


def expected_probe_ids(rows):
    selected = {category: {} for category in CATEGORIES}
    for row in rows:
        if row['split'] != 'train':
            continue
        sources = selected[row['category']]
        product = row['source']
        if product['product_id'] not in sources and len(sources) < 2:
            sources[product['product_id']] = product['sample_id']
    if any(len(value) != 2 for value in selected.values()):
        raise ValueError('Need the first two distinct original TRAIN source products per category')
    ids = {row['sample_id'] for row in rows if row['split'] == 'train' and
           row['source']['sample_id'] in selected[row['category']].values()}
    if len(ids) != 60:
        raise ValueError('The reviewed fixed TRAIN probe must contain exactly60 rows')
    return ids


def load_native(path, mode):
    with Image.open(path) as image:
        if image.mode != mode or image.size != (256, 256):
            raise ValueError('Native '+mode+'256 PNG required: '+str(path))
        return np.asarray(image).copy()


def load_prepared(data):
    rows_path = data/'annotated_pairs.json'; rows = read_json(rows_path)
    ready = read_json(data/'report.json'); audit = read_json(data/'data_audit.json')
    if (not ready.get('complete') or ready.get('pair_sha256') != digest(rows_path)
            or audit.get('protocol') != PROTOCOL or audit.get('role_disjoint') is not True
            or audit.get('test_opened') is not False or audit.get('expected_counts') != COUNTS
            or dict(Counter(row['split'] for row in rows)) != COUNTS
            or any(row.get('render_protocol') != PROTOCOL or row['category'] not in CATEGORIES
                   or row['mode'] not in MODES for row in rows)
            or len({row['sample_id'] for row in rows}) != sum(COUNTS.values())):
        raise ValueError('Only the completed controlled-room960 TRAIN/96 DEV preparation is allowed')
    wanted = expected_probe_ids(rows); indexed = {row['sample_id']: (i, row) for i, row in enumerate(rows)}
    manifest = read_json(data/'image_sha256.json')
    expected_manifest = {f"images/{row['split']}/{role}/{i:05d}.png"
                         for i, row in enumerate(rows) for role in ('reference', 'target', 'edit_mask')}
    if set(manifest) != expected_manifest:
        raise ValueError('Prepared3168-PNG manifest has missing or additional entries')
    prepared = {}
    for sid in wanted:
        index, row = indexed[sid]; values = {}; hashes = {}
        for role in ('reference', 'target', 'edit_mask'):
            relative = f'images/train/{role}/{index:05d}.png'; path = data/relative
            hashes[role] = digest(path)
            if hashes[role] != manifest[relative]:
                raise ValueError('Prepared PNG SHA differs: '+relative)
            values[role] = load_native(path, 'L' if role == 'edit_mask' else 'RGB')
        if hashes['edit_mask'] != row['edit_mask_sha256']:
            raise ValueError('Prepared mask SHA differs from its row label')
        prepared[sid] = {'row': row, **values, 'sha256': hashes}
    groups = []; seen = set()
    for group in read_json(data/'TRAIN_counterfactual_groups.json'):
        sids = group['sample_ids']
        if not (set(sids) & wanted):
            continue
        key = group['reference_context_id'], group['mode']
        if key in seen or len(sids) != 2 or len(set(sids)) != 2 or not set(sids) <= wanted:
            raise ValueError('Counterfactual manifest does not contain complete unique selected pairs')
        seen.add(key); pair = [prepared[sid] for sid in sids]
        paired_row_contract([item['row'] for item in pair], *key)
        if group['prompts'] != [item['row']['prompt'] for item in pair]:
            raise ValueError('Counterfactual manifest prompts changed')
        if not np.array_equal(pair[0]['reference'], pair[1]['reference']):
            raise ValueError('Same-context instructions do not have identical source RGB')
        groups.append({'reference_context_id': key[0], 'mode': key[1],
                       'category': pair[0]['row']['category'], 'sample_ids': sids})
    if len(groups) != 30 or Counter(sid for group in groups for sid in group['sample_ids']) != Counter(wanted):
        raise ValueError('Exactly30 complete TRAIN pairs covering all60 selected rows required')
    # The first two products have different reviewed view/context allocations
    # between categories. Derive cell budgets from those selected rows, not an
    # assumed half of the total context count.
    selected_counts = Counter((item['row']['category'], item['row']['mode']) for item in prepared.values())
    expected_counts = {}
    for category in CATEGORIES:
        for mode in MODES:
            count = selected_counts[category, mode]
            if not count or count % 2:
                raise ValueError('Selected TRAIN category/task cells need complete paired instructions')
            expected_counts[category, mode] = count//2
    observed_counts = Counter((group['category'], group['mode']) for group in groups)
    if observed_counts != expected_counts:
        raise ValueError('Counterfactual groups differ from the exact selected TRAIN row allocation')
    return prepared, groups


def load_model(label, directory, prepared):
    report_path = directory/'report.json'; report = read_json(report_path)
    records = report.get('metrics', {}).get('images', [])
    if (not report.get('complete') or report.get('cohort') != 'train_probe'
            or report.get('evaluation_label') != 'TRAIN_resubstitution_diagnostic'
            or report.get('independent_test_opened') is not False or len(records) != 60
            or len({record['sample_id'] for record in records}) != 60
            or {record['sample_id'] for record in records} != set(prepared)):
        raise ValueError('Require a completed, matched60-row TRAIN fit review: '+label)
    receipt_path = directory/'review_tree_download_receipt.json'
    receipt = read_json(receipt_path) if receipt_path.is_file() else None
    if receipt is not None and receipt.get('run_report_sha256') != digest(report_path):
        raise ValueError('Verified PNG download receipt belongs to a different review report')
    images = {}; hashes = {}
    for i, record in enumerate(records):
        sid = record['sample_id']; expected = prepared[sid]; row = expected['row']
        if any(record[key] != row[key] for key in ('prompt', 'category', 'mode')):
            raise ValueError('Matched review prompt/category/task differs from prepared TRAIN')
        hashes[sid] = {}
        for role in ('reference', 'target', 'generated'):
            relative = f'images/raw_train/{role}/sample_{i:04d}.png'; path = directory/relative
            actual = digest(path); hashes[sid][role] = actual
            if receipt is not None:
                entry = receipt.get('files', {}).get(relative, {})
                if entry.get('sha256') != actual or entry.get('bytes') != path.stat().st_size:
                    raise ValueError('Review PNG differs from verified download receipt: '+relative)
            pixels = load_native(path, 'RGB')
            if role != 'generated' and not np.array_equal(pixels, expected[role]):
                raise ValueError('Raw review reference/GT pixels differ from the prepared pair')
            if role == 'generated':
                images[sid] = pixels
    provenance = {'label': label, 'review': str(directory.resolve()),
        'review_report_sha256': digest(report_path), 'parent_checkpoint_sha256': report.get('parent_checkpoint_sha256'),
        'source_commit': report.get('source_commit'), 'count': 60, 'native_PNG_sha256': hashes,
        'download_receipt_sha256': digest(receipt_path) if receipt is not None else None,
        'PNG_binding': 'Verified per-file remote download SHA plus completed report' if receipt is not None else
            'Actual local PNG SHA recorded; completed report identity and every reference/GT bound to prepared native pixels. No remote PNG receipt is available.'}
    return {'label': label, 'images': images, 'provenance': provenance}


def summarize(groups):
    def mean(values):
        return sum(values)/len(values)
    return {'groups': len(groups), 'outputs': 2*len(groups),
        'common_retained_delta_L1_RGB_01': mean([group['metrics']['common_retained_output_delta']['L1_RGB_01'] for group in groups]),
        'common_retained_delta_MSE_RGB_01': mean([group['metrics']['common_retained_output_delta']['MSE_RGB_01'] for group in groups]),
        'per_output_retained_L1_RGB_01': mean([value['L1_RGB_01'] for group in groups for value in group['metrics']['retained_error_vs_reference']]),
        'per_output_retained_MSE_RGB_01': mean([value['MSE_RGB_01'] for group in groups for value in group['metrics']['retained_error_vs_reference']]),
        'per_output_retained_PSNR_dB': mean([value['PSNR_dB'] for group in groups for value in group['metrics']['retained_error_vs_reference']]),
        'whole_pair_delta_L1_diagnostic_only': mean([group['metrics']['whole_output_delta_diagnostic_only']['L1_RGB_01'] for group in groups]),
        'GT_common_retained_delta_L1_control': mean([group['metrics']['GT_common_retained_delta_control']['L1_RGB_01'] for group in groups]),
        'GT_per_output_retained_L1_control': mean([value['L1_RGB_01'] for group in groups for value in group['metrics']['GT_retained_error_controls']])}


def font(size):
    for path in ('C:/Windows/Fonts/arial.ttf', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


def pair_panel(group, models, prepared, destination):
    pair = [prepared[sid] for sid in group['sample_ids']]; width = 256*(2+len(models))
    title_font, small = font(16), font(14)
    context = '\n'.join(textwrap.wrap(group['reference_context_id'], max(70, width//8)))
    header = 68+18*len(context.splitlines()); row_height = 352
    panel = Image.new('RGB', (width, header+2*row_height), 'white'); draw = ImageDraw.Draw(panel)
    draw.text((5, 4), 'TRAIN FIT ONLY - no independent evaluation or quality acceptance', fill='black', font=title_font)
    draw.text((5, 26), group['category']+'/'+group['mode']+' | same RGB and source-context seed', fill='black', font=small)
    draw.multiline_text((5, 46), context, fill='black', font=small)
    for j, item in enumerate(pair):
        top = header+j*row_height; branch = 'A' if j == 0 else 'B'
        prompt = '\n'.join(textwrap.wrap(branch+': '+item['row']['prompt'], max(70, width//8)))
        draw.multiline_text((5, top), prompt, fill='black', font=title_font)
        draw.text((5, top+48), 'Shared reference RGB', fill='black', font=small)
        draw.text((261, top+48), 'GT '+branch+' (offline only)', fill='black', font=small)
        panel.paste(Image.fromarray(item['reference']), (0, top+90))
        panel.paste(Image.fromarray(item['target']), (256, top+90))
        for column, model in enumerate(models, 2):
            label = '\n'.join(textwrap.wrap(model['label']+' | prediction '+branch, 31))
            draw.multiline_text((column*256+4, top+48), label, fill='black', font=small)
            panel.paste(Image.fromarray(model['images'][group['sample_ids'][j]]), (column*256, top+90))
    panel.save(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True, help='Prepared controlled-room PNG/JSON directory, not server absolute mask paths')
    parser.add_argument('--model', action='append', required=True, help='LABEL=completed60-row TRAIN raw review; repeat for matched models')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve existing analysis artifacts; require a new output directory')
    prepared, pairs = load_prepared(args.data)
    models = []; labels = set()
    for specification in args.model:
        if '=' not in specification:
            raise ValueError('Each model must be LABEL=completed_review_directory')
        label, directory = specification.split('=', 1)
        if not label.strip() or label in labels:
            raise ValueError('Distinct nonempty model labels required')
        labels.add(label); models.append(load_model(label, Path(directory), prepared))
    results = {}; selections = []
    for model in models:
        groups = []
        for group in pairs:
            sids = group['sample_ids']; data = [prepared[sid] for sid in sids]
            metrics = pair_metrics(data[0]['reference'], [item['target'] for item in data],
                [model['images'][sid] for sid in sids], [item['edit_mask'] for item in data])
            groups.append({**group, 'prompts': [item['row']['prompt'] for item in data],
                'noise_seed': data[0]['row']['noise_seed'],
                'requested_families': [item['row'].get('requested_family') for item in data],
                'target_wall_colors': [item['row']['target_wall_color'] for item in data],
                'prepared_PNG_sha256': [item['sha256'] for item in data], 'metrics': metrics})
        results[model['label']] = {'provenance': model['provenance'], 'groups': groups,
            'summary': {'overall': summarize(groups),
                'by_mode': {mode: summarize([group for group in groups if group['mode'] == mode]) for mode in MODES},
                'by_category_mode': {category: {mode: summarize([group for group in groups if group['category'] == category and group['mode'] == mode]) for mode in MODES} for category in CATEGORIES}}}
    # All sources and predicted PNGs have passed validation before any output.
    args.output.mkdir(parents=True)
    for category in CATEGORIES:
        for mode in MODES:
            group = next(group for group in pairs if group['category'] == category and group['mode'] == mode)
            path = args.output/(category+'_'+mode+'_pair.png')
            pair_panel(group, models, prepared, path)
            selections.append({**group, 'figure': path.name, 'sha256': digest(path)})
    write_json(args.output/'report.json', {'complete': True, 'quality_accepted': False,
        'evaluation_label': 'TRAIN_resubstitution_diagnostic', 'independent_TEST_opened': False,
        'editor_run': False, 'training_run': False, 'GPU_used': False,
        'argv': sys.argv, 'completed_unix': time.time(), 'source_commit': os.environ.get('T12_SOURCE_COMMIT'),
        'source_sha256': digest(Path(__file__)), 'prepared_data_directory': str(args.data.resolve()),
        'prepared_pair_sha256': digest(args.data/'annotated_pairs.json'),
        'prepared_PNG_manifest_sha256': digest(args.data/'image_sha256.json'),
        'counterfactual_manifest_sha256': digest(args.data/'TRAIN_counterfactual_groups.json'),
        'prepared_selected_PNGs_verified': 180, 'fixed_groups': 30, 'fixed_outputs_per_model': 60,
        'selected_group_counts_by_category_mode': {category: {mode: sum(group['category'] == category and group['mode'] == mode for group in pairs) for mode in MODES} for category in CATEGORIES},
        'models': results, 'figures': selections,
        'protocol': {'selection': 'Exact existing train_probe policy: first view of first two distinct TRAIN source products per category, all paired instructions and all three tasks;30 groups from supplied manifest, never quality filtering.',
            'metrics': 'RGB[0,1], channel-mean L1/MSE. Common retained support is complement of union of the exact two changed-pixel masks; per-output retained support uses its own mask. Retained error compares against the shared reference. GT retained controls must be0. Per-image PSNR ceiling120dB, aggregate per-pair/per-output means, not pixel-weighted pooling.',
            'figures': 'First supplied pair per category/task; both full prompts and both GTs/predictions, shared reference pixels. Every tile remains native256 with no resize, retouching or enhancement.',
            'scope': 'Offline CPU numpy/PIL only, no CLIP, approximate alpha, model forward, new training, checkpoint changes or independent TEST access.'},
        'limitations': 'Whole-image or edited-region response does not establish correct instruction following. A common retained delta of0 can mean both outputs have the same incorrect content; inspect retained errors against reference too. These examples were optimized during TRAIN and cannot establish generalization or distinguish irreversible feature compression from loss/optimization failures without controlled interventions.'})


if __name__ == '__main__':
    main()
