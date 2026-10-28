"""CPU-only family, wall-palette and preservation checks of saved DEV PNGs.

The requested product FAMILY is scored, not retrieval of an arbitrary target
SKU. Genuine alpha and changed-pixel masks are offline metric labels only.
This module never runs an editor, prepares new examples or opens a TEST set.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from .qwen_vl_controlled_room_data import PALETTES, PROTOCOL, room_background
from .qwen_vl_simple_edits import LABELS
from .qwen_vl_turbo_data import normalized_product, sha256, validate_roles, write_json


COUNTS = {'train': 960, 'val': 96}
CLIP_REVISION = 'd05afc436d78f1c48dc0dbf8e5980a9d471f35f6'


def family_prototypes(category):
    if category not in LABELS:
        raise ValueError('Unknown reviewed product category')
    families = list(LABELS[category])
    return families, ['A photo of '+LABELS[category][family]+'.' for family in families]


def family_result(similarities, category, requested_family):
    families, _ = family_prototypes(category)
    scores = np.asarray(similarities, dtype=np.float64)
    if (requested_family not in families or scores.shape != (len(families),) or
            not np.isfinite(scores).all()):
        raise ValueError('Finite scores for the two reviewed product families required')
    chosen = families.index(requested_family)
    alternatives = np.delete(scores, chosen)
    maximum = scores.max()
    winners = np.flatnonzero(scores == maximum)
    return {'requested_family': requested_family,
        'predicted_family': families[int(winners[0])] if len(winners) == 1 else None,
        'strict_top1': bool(scores[chosen] > alternatives.max()),
        'requested_cosine': float(scores[chosen]),
        'margin_to_best_alternative': float(scores[chosen]-alternatives.max()),
        'candidate_count': len(families),
        'cosines': {family: float(value) for family, value in zip(families, scores)}}


def _native_rgb(value):
    array = np.asarray(value)
    if array.shape != (256, 256, 3) or array.dtype != np.uint8:
        raise ValueError('Require unmodified native uint8 RGB256 PNG pixels')
    return array


def exposed_wall_roi(source_alpha, target_alpha):
    """Fixed room wall without window/table or either true product support."""
    alphas = [np.asarray(value, dtype=np.float32) for value in (source_alpha, target_alpha)]
    if any(value.shape != (256, 256) or not np.isfinite(value).all() or
           value.min() < 0 or value.max() > 1 for value in alphas):
        raise ValueError('Source and target genuine alpha must be finite native [0,1] masks')
    _, wall = room_background('light_gray', return_wall_mask=True)
    roi = wall & (alphas[0] <= .01) & (alphas[1] <= .01)
    if not roi.any():
        raise ValueError('Empty exposed-wall metric ROI')
    return roi


def wall_palette_result(image, roi, expected_palette):
    image = _native_rgb(image)
    roi = np.asarray(roi)
    if roi.shape != (256, 256) or roi.dtype != bool or not roi.any():
        raise ValueError('Nonempty native boolean wall ROI required')
    if expected_palette not in PALETTES:
        raise ValueError('Unreviewed expected wall palette')
    measured = image[roi].astype(np.float64).mean(0)/255
    prototypes = {}
    for palette in PALETTES:
        # Same fixed light field and exactly the same visible-wall pixels.
        pixels = np.asarray(room_background(palette), dtype=np.float64)
        prototypes[palette] = pixels[roi].mean(0)/255
    distances = {key: float(np.sqrt(np.square(measured-value).mean()))
                 for key, value in prototypes.items()}
    minimum = min(distances.values())
    winners = [key for key, value in distances.items() if value == minimum]
    correct = distances[expected_palette]
    alternative = min(value for key, value in distances.items() if key != expected_palette)
    return {'expected_palette': expected_palette,
        'predicted_palette': winners[0] if len(winners) == 1 else None,
        'strict_top1': bool(correct < alternative), 'wall_pixels': int(roi.sum()),
        'mean_RGB_01': measured.tolist(),
        'expected_distance_RGB_01': correct,
        'margin_to_best_alternative': alternative-correct,
        'distances_RGB_01': distances}


def preservation_result(generated, reference, target, edit_mask):
    generated, reference, target = [_native_rgb(value) for value in (generated, reference, target)]
    mask = np.asarray(edit_mask)
    if mask.shape != (256, 256) or mask.dtype != np.uint8 or not np.isin(mask, (0, 255)).all():
        raise ValueError('Native binary changed-pixel supervision mask required')
    changed = np.any(reference != target, axis=2)
    edit = mask > 0
    if (changed & ~edit).any():
        raise ValueError('Changed target pixels are missing from the supervised edit mask')
    if (edit & ~changed).any() or not edit.any():
        raise ValueError('Supervised edit mask is not the exact changed-pixel support')
    keep = ~edit
    if not keep.any():
        raise ValueError('Empty retained-region metric support')
    error = (generated.astype(np.float64)-reference.astype(np.float64))[keep]/255
    mse = float(np.square(error).mean())
    return {'retained_pixels': int(keep.sum()), 'edit_pixels': int(edit.sum()),
        'L1_RGB_01': float(np.abs(error).mean()), 'MSE_RGB_01': mse,
        'PSNR_dB': -10*math.log10(max(mse, 1e-12))}


def _read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def reviewed_dev_rows(data, review):
    """Require the finalized 96-row raw review of this prepared DEV cohort."""
    audit = _read_json(data/'data_audit.json')
    ready = _read_json(data/'report.json')
    rows = _read_json(data/'annotated_pairs.json')
    if (audit.get('protocol') != PROTOCOL or audit.get('test_opened') is not False or
            audit.get('role_disjoint') is not True or audit.get('expected_counts') != COUNTS or
            not ready.get('complete') or ready.get('pair_sha256') != sha256(data/'annotated_pairs.json') or
            any(row.get('render_protocol') != PROTOCOL or row['split'] not in COUNTS for row in rows) or
            validate_roles(rows)['pairs'] != COUNTS):
        raise ValueError('Only the prepared controlled-room TRAIN/DEV protocol is allowed')
    dev = {row['sample_id']: (index, row) for index, row in enumerate(rows) if row['split'] == 'val'}
    if len(dev) != 96:
        raise ValueError('Exactly 96 unique DEV identities required')
    report = _read_json(review/'report.json')
    records = report.get('metrics', {}).get('images', [])
    if (not report.get('complete') or report.get('cohort') != 'development' or
            report.get('evaluation_label') != 'development_test' or
            report.get('independent_test_opened') is not False or
            len(records) != 96 or len({record['sample_id'] for record in records}) != 96 or
            {record['sample_id'] for record in records} != set(dev)):
        raise ValueError('Require a finalized raw DEV96 review, never independent TEST or TRAIN fit')
    for record in records:
        row = dev[record['sample_id']][1]
        if any(record[key] != row[key] for key in ('prompt', 'category', 'mode')):
            raise ValueError('Raw review identities, prompts or tasks differ from prepared DEV')
        if row['mode'] != 'background' and row.get('requested_family') not in LABELS[row['category']]:
            raise ValueError('Product edit lacks a reviewed requested family')
        box = record.get('object_crop_box')
        if (not isinstance(box, list) or len(box) != 4 or
                any(type(value) is not int for value in box) or
                not 0 <= box[0] < box[2] <= 256 or not 0 <= box[1] < box[3] <= 256):
            raise ValueError('Native recorded offline object crop is missing or invalid')
    return dev, records


def _summary(records):
    result = {}
    for role in ('generated', 'target_control'):
        values = [record[role] for record in records]
        family = [value['family'] for value in values if value['family'] is not None]
        result[role] = {'n': len(values), 'family_count': len(family),
            'family_strict_top1': sum(value['strict_top1'] for value in family)/len(family) if family else None,
            'family_margin': sum(value['margin_to_best_alternative'] for value in family)/len(family) if family else None,
            'wall_palette_strict_top1': sum(value['wall']['strict_top1'] for value in values)/len(values),
            'wall_expected_distance_RGB_01': sum(value['wall']['expected_distance_RGB_01'] for value in values)/len(values),
            'retained_L1_RGB_01': sum(value['retained']['L1_RGB_01'] for value in values)/len(values),
            'retained_PSNR_dB': sum(value['retained']['PSNR_dB'] for value in values)/len(values)}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('data', 'review', 'clip-repo', 'weights', 'output'):
        parser.add_argument('--'+key, type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or args.output.exists():
        raise ValueError('An immutable CPU-only audit directory is required')
    if not args.weights.is_file():
        raise ValueError('Provide an existing local ViT-B/16 weights file; this audit does not download weights')
    dev, records = reviewed_dev_rows(args.data, args.review)
    manifest = _read_json(args.data/'image_sha256.json')
    revision = subprocess.check_output(['git', '-C', str(args.clip_repo), 'rev-parse', 'HEAD']).decode().strip()
    if revision != CLIP_REVISION:
        raise ValueError('Use the reviewed official CLIP implementation')
    sys.path.insert(0, str(args.clip_repo))
    import clip
    if not Path(clip.__file__).resolve().is_relative_to(args.clip_repo.resolve()):
        raise ValueError('Imported CLIP does not belong to the specified official repository')
    torch.set_num_threads(4)
    encoder, preprocess = clip.load(str(args.weights), device='cpu', jit=False)
    encoder = encoder.float().eval().requires_grad_(False)
    if encoder.visual.conv1.kernel_size != (16, 16) or len(encoder.visual.transformer.resblocks) != 12:
        raise ValueError('Use the separate frozen ViT-B/16 audit encoder')
    texts = {}
    with torch.no_grad():
        for category in LABELS:
            _, phrases = family_prototypes(category)
            texts[category] = F.normalize(encoder.encode_text(clip.tokenize(phrases, truncate=False)).float(), dim=-1)
    args.output.mkdir(parents=True)
    write_json(args.output/'execution.json', {'argv': sys.argv, 'pid': os.getpid(), 'device': 'cpu',
        'started_unix': time.time(), 'source_commit': os.environ.get('T12_SOURCE_COMMIT'),
        'source_sha256': sha256(Path(__file__))})
    results = []
    for position, record in enumerate(records):
        data_index, row = dev[record['sample_id']]
        mask_path = Path(row['edit_mask_path'])
        mask_key = f'images/val/edit_mask/{data_index:05d}.png'
        if (manifest.get(mask_key) != row['edit_mask_sha256'] or
                sha256(mask_path) != row['edit_mask_sha256']):
            raise ValueError('Prepared edit-mask bytes changed')
        with Image.open(mask_path) as image:
            if image.mode != 'L' or image.size != (256, 256):
                raise ValueError('Native L256 edit mask required')
            edit_mask = np.asarray(image).copy()
        source_alpha = np.asarray(normalized_product(row['source'], 256)[1], dtype=np.float32)/255
        target_alpha = np.asarray(normalized_product(row['target'], 256)[1], dtype=np.float32)/255
        roi = exposed_wall_roi(source_alpha, target_alpha)
        images = {}; hashes = {}
        for role in ('reference', 'target', 'generated'):
            path = args.review/'images'/'raw_dev'/role/f'sample_{position:04d}.png'
            with Image.open(path) as image:
                if image.mode != 'RGB' or image.size != (256, 256):
                    raise ValueError('Saved raw review must contain native RGB256 PNGs')
                images[role] = np.asarray(image).copy()
            hashes[role] = sha256(path)
            if role != 'generated':
                prepared_path = mask_path.parent.parent/role/mask_path.name
                key = f'images/val/{role}/{data_index:05d}.png'
                if sha256(prepared_path) != manifest.get(key):
                    raise ValueError('Prepared reference/target PNG manifest identity changed')
                with Image.open(prepared_path) as image:
                    if not np.array_equal(images[role], np.asarray(image.convert('RGB'))):
                        raise ValueError('Raw review reference/target pixels differ from prepared data')
        family = {'generated': None, 'target': None}
        if row['mode'] != 'background':
            crops = [preprocess(Image.fromarray(images[role]).crop(tuple(record['object_crop_box'])))
                     for role in ('generated', 'target')]
            with torch.no_grad():
                features = F.normalize(encoder.encode_image(torch.stack(crops)).float(), dim=-1)
                scores = features @ texts[row['category']].T
            for role, values in zip(('generated', 'target'), scores):
                family[role] = family_result(values.cpu().numpy(), row['category'], row['requested_family'])
        result = {'sample_id': row['sample_id'], 'prompt': row['prompt'], 'mode': row['mode'],
            'category': row['category'], 'requested_family': row.get('requested_family'),
            'object_crop_box': record['object_crop_box'], 'edit_mask_sha256': row['edit_mask_sha256'],
            'reference_png_sha256': hashes['reference']}
        for role, label in (('generated', 'generated'), ('target', 'target_control')):
            result[label] = {'png_sha256': hashes[role], 'family': family[role],
                'wall': wall_palette_result(images[role], roi, row['target_wall_color']),
                'retained': preservation_result(images[role], images['reference'], images['target'], edit_mask)}
        results.append(result)
        if (position+1) % 12 == 0:
            write_json(args.output/'progress.json', {'stage': 'CPU_saved_DEV_PNG_audit', 'done': position+1, 'total': 96})
    groups = {(category, mode): [row for row in results if row['category'] == category and row['mode'] == mode]
              for category in LABELS for mode in ('background', 'object', 'joint')}
    if any(len(value) != 16 for value in groups.values()):
        raise ValueError('The balanced DEV96 category/task cohort changed')
    write_json(args.output/'report.json', {'complete': True, 'quality_accepted': False,
        'evaluation_label': 'development_test', 'evaluation_used_for_selection': True,
        'independent_TEST_opened': False, 'editor_run': False, 'device': 'cpu',
        'source_commit': os.environ.get('T12_SOURCE_COMMIT'), 'source_sha256': sha256(Path(__file__)),
        'data_pair_sha256': sha256(args.data/'annotated_pairs.json'),
        'data_audit_sha256': sha256(args.data/'data_audit.json'),
        'prepared_PNG_manifest_sha256': sha256(args.data/'image_sha256.json'),
        'review_report_sha256': sha256(args.review/'report.json'),
        'review_parent_checkpoint_sha256': _read_json(args.review/'report.json').get('parent_checkpoint_sha256'),
        'clip_repo_sha': revision, 'weights_sha256': sha256(args.weights),
        'family_prototypes': {category: dict(zip(*family_prototypes(category))) for category in LABELS},
        'overall': _summary(results),
        'by_mode': {mode: _summary([row for row in results if row['mode'] == mode]) for mode in ('background', 'object', 'joint')},
        'by_category_mode': {category: {mode: _summary(groups[category, mode]) for mode in ('background', 'object', 'joint')} for category in LABELS},
        'images': results,
        'protocol': {'family': 'Frozen official ViT-B/16 preprocessing; two within-category family prototypes. Recorded GT object union box is an offline crop only. Strict ties are errors. Background rows have no requested replacement family.',
            'wall': 'Nearest RGB mean to each of the five exact procedural palette renderings under the same fixed light and identical exposed-wall ROI. Exclude native window/table and source/target genuine alpha>0.01; no generated-image-dependent ROI selection.',
            'preservation': 'L1/MSE in RGB[0,1] against reference, on the complement of the exact prepared changed-pixel mask. Any changed GT pixel omitted from the mask fails. Per-image PSNR averaged; 120dB ceiling for zero error.',
            'inputs': 'Saved generated/reference/target PNGs and offline labels only; no editor inference, GPU, new weights, feature cache, independent TEST, target-derived model input or GT paste.'},
        'limitations': 'Family recognition, nearest wall palette and preservation complement exact-target LPIPS/PSNR; none certifies all attributes or high quality. GT controls expose classifier failures. Family metrics allow alternative SKUs, but do not demonstrate stochastic diversity. A wrong object entering the fixed wall ROI can affect its mean. All96 are development examples, not independent generalization.'})


if __name__ == '__main__':
    main()
