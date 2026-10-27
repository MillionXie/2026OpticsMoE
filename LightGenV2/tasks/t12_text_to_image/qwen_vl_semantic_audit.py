"""CPU-only semantic diagnostics of existing DEV PNGs, never model inference.

Use official CLIP preprocessing and a separately recorded encoder. Caption
cosine and within-category product-description retrieval supplement visual
review; they do not certify that every requested attribute is present.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch
from torch.nn import functional as F
from PIL import Image

from .qwen_vl_turbo_data import sha256, write_json


def retrieval_scores(visual, text, correct):
    """Return cosine, rank and margin; ties count conservatively as errors."""
    scores = F.normalize(visual.float(), dim=-1) @ F.normalize(text.float(), dim=-1).T
    result = []
    for row, index in zip(scores, correct):
        positive = float(row[index])
        alternatives = torch.cat((row[:index], row[index + 1:]))
        maximum = float(alternatives.max()) if len(alternatives) else None
        result.append({'target_cosine': positive,
                       'rank': 1 + int((alternatives >= positive).sum()),
                       'margin_to_best_alternative': positive - maximum if maximum is not None else None,
                       'strict_top1': maximum is not None and positive > maximum,
                       'candidate_count': len(row)})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--review', type=Path, action='append', required=True)
    parser.add_argument('--clip-repo', type=Path, required=True)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or a.output.exists():
        raise ValueError('Use an immutable CPU-only diagnostic run')
    revision = subprocess.check_output(['git', '-C', str(a.clip_repo), 'rev-parse', 'HEAD']).decode().strip()
    if revision != 'd05afc436d78f1c48dc0dbf8e5980a9d471f35f6':
        raise ValueError('Unreviewed CLIP implementation')
    audit = json.loads((a.data / 'data_audit.json').read_text())
    if audit['test_opened'] or audit['pairs'] != {'train': 1824, 'val': 96}:
        raise ValueError('Only the current development cohort is authorized')
    rows = json.loads((a.data / 'annotated_pairs.json').read_text())
    rows = {r['sample_id']: r for r in rows if r['split'] == 'val'}
    sys.path.insert(0, str(a.clip_repo))
    import clip
    torch.set_num_threads(4)
    a.output.mkdir(parents=True)
    write_json(a.output / 'execution.json', {'argv': sys.argv, 'pid': os.getpid(),
        'started_unix': time.time(), 'source_commit': os.environ.get('T12_SOURCE_COMMIT'),
        'source_sha256': sha256(Path(__file__)), 'device': 'cpu',
        'data_audit_sha256': sha256(a.data / 'data_audit.json')})
    encoder, preprocess = clip.load(str(a.weights), device='cpu', jit=False)
    encoder = encoder.float().eval().requires_grad_(False)
    if encoder.visual.conv1.kernel_size != (16, 16) or len(encoder.visual.transformer.resblocks) != 12:
        raise ValueError('This separate audit requires ViT-B/16, not the TRAIN ViT-B/32 encoder')
    def encode_text(texts):
        with torch.no_grad():
            return F.normalize(encoder.encode_text(clip.tokenize(texts, truncate=False)).float(), dim=-1)
    descriptions = {c: sorted({r['target_description'] for r in rows.values() if r['category'] == c})
                    for c in ('lamp', 'pillow')}
    texts = {c: encode_text(v) for c, v in descriptions.items()}
    results = []
    for review in a.review:
        report = json.loads((review / 'report.json').read_text())
        if not report['complete']:
            raise ValueError('Review incomplete')
        if 'metrics' in report:
            metrics, folder = report['metrics'], 'raw_dev'
        else:
            metrics, folder = report['best_val'], 'dev_best'
        records = metrics['images']
        if len(records) != 96 or {r['sample_id'] for r in records} != set(rows):
            raise ValueError('Review cohort differs from annotated DEV')
        scene_text = encode_text([rows[r['sample_id']]['teacher_caption'] for r in records])
        model = {'review': str(review), 'report_sha256': sha256(review / 'report.json'), 'images': []}
        for start in range(0, len(records), 8):
            batch = records[start:start + 8]
            for role in ('generated', 'target'):
                whole, crop, image_sha = [], [], []
                for i, record in enumerate(batch, start):
                    path = review / 'images' / folder / role / f'sample_{i:04d}.png'
                    with Image.open(path) as image:
                        image = image.convert('RGB')
                        if image.size != (256, 256):
                            raise ValueError('Require original native 256 PNG')
                        whole.append(preprocess(image))
                        crop.append(preprocess(image.crop(tuple(record['object_crop_box']))))
                    image_sha.append(sha256(path))
                with torch.no_grad():
                    features = F.normalize(encoder.encode_image(torch.stack(whole)).float(), dim=-1)
                    crops = F.normalize(encoder.encode_image(torch.stack(crop)).float(), dim=-1)
                for j, record in enumerate(batch):
                    row = rows[record['sample_id']]
                    category = row['category']
                    selected = descriptions[category].index(row['target_description'])
                    values = retrieval_scores(crops[j:j + 1], texts[category], [selected])[0]
                    model['images'].append({'sample_id': row['sample_id'], 'mode': row['mode'],
                        'category': category, 'role': role, 'png_sha256': image_sha[j],
                        'scene_caption_cosine': float((features[j] * scene_text[start + j]).sum()),
                        'product_description': row['target_description'], **values})
        model['by_mode'] = {}
        for mode in ('background', 'object', 'joint'):
            model['by_mode'][mode] = {}
            for role in ('generated', 'target'):
                chosen = [r for r in model['images'] if r['mode'] == mode and r['role'] == role]
                model['by_mode'][mode][role] = {'n': len(chosen), **{
                    name: sum(float(r[name]) for r in chosen) / len(chosen)
                    for name in ('scene_caption_cosine', 'target_cosine', 'strict_top1', 'rank')}}
        results.append(model)
    write_json(a.output / 'report.json', {'complete': True, 'source_commit': os.environ.get('T12_SOURCE_COMMIT'),
        'source_sha256': sha256(Path(__file__)), 'clip_repo_sha': revision, 'weights_sha256': sha256(a.weights),
        'encoder_visual_layers': len(encoder.visual.transformer.resblocks),
        'encoder_visual_patch_size': encoder.visual.conv1.kernel_size[0],
        'preprocess': 'Official clip.load PIL resize/center-crop RGB normalization; GT union box is offline crop only',
        'evaluation_label': 'development_test', 'independent_TEST_opened': False,
        'training_encoder': 'ViT-B/32; choose separate ViT-B/16 weights for this audit',
        'candidate_descriptions': descriptions, 'models': results,
        'limitations': 'Cosine/retrieval diagnostics do not certify all attributes. Ambiguous captions and GT retrieval errors must be inspected. GT images are controls only and never enter the editor.'})


if __name__ == '__main__':
    main()
