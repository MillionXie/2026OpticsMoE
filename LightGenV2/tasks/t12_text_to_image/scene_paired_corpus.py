"""Own ABO -> furnished-scene TRAIN pairs, without changing product appearance.

Background generation and true-alpha compositing occur only in data creation.
Inference has no compositing, mask, target, VAE reference or retrieval path.
All public-model backgrounds are retained for visual QA; model prompts do not
prove that the generated room is correct. A review gate precedes composition.
"""
import argparse
from collections import Counter
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import traceback

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from .appearance_scene_protocol import PROFILE, conditions
from .scene_generation_data import request_seed
from .white_reference_scene_probe import GPU, digest, write, checked_tokens, model_manifest


def background_requests():
    requests = []
    for category in ['lamp', 'pillow']:
        for condition in conditions(category):
            attrs = condition['requested_attributes']
            support = attrs['support'].replace('on ', '', 1)
            camera = ('Close-up at tabletop height. The empty tabletop fills the lower third, '
                      'its front edge below the frame. ' if category == 'lamp' else
                      'Close-up at seat height, straight-on, a vacant seat fills the lower third. ')
            caption = (f'Photorealistic interior of a furnished {attrs["scene"]}, {support} in the foreground. '
                       f'{camera}{attrs["background_elements"]} in the background. '
                       'A clear empty center ready for product photography, realistic indoor perspective.')
            for split, variants in [('train', [0, 1]), ('dev', [2])]:
                for variant in variants:
                    key = f'{category}-{split}-scene{condition["condition_id"]}-v{variant}'
                    seed = int(hashlib.sha256(('empty-scene|'+key).encode()).hexdigest()[:8], 16) % (2**31)
                    requests.append({'asset_id': key, 'category': category, 'split': split,
                        'condition_id': condition['condition_id'], 'variant': variant, 'seed': seed,
                        'prompt': caption, 'attributes_requested_not_verified': attrs})
    return requests


def generate(a):
    import torch
    from diffusers import StableDiffusionXLPipeline
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == GPU
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.8, 0)
    write(a.output/'model_assets.json', model_manifest(a.model_path))
    pipe = StableDiffusionXLPipeline.from_pretrained(a.model_path, variant='fp16',
        torch_dtype=torch.float16, local_files_only=True).to('cuda')
    pipe.set_progress_bar_config(disable=True)
    for c in pipe.components.values():
        if isinstance(c, torch.nn.Module): c.eval().requires_grad_(False)
    rows = background_requests()
    for i, row in enumerate(rows):
        token_audit = [checked_tokens(t, row['prompt']) for t in [pipe.tokenizer, pipe.tokenizer_2]]
        negative = 'lamp, lampshade, cushion, pillow, people, person, logo, watermark, letters, text, collage'
        with torch.inference_mode():
            result = pipe(prompt=row['prompt'], negative_prompt=negative, width=1024, height=1024,
                num_inference_steps=30, guidance_scale=7.5,
                generator=torch.Generator(device='cuda').manual_seed(row['seed'])).images[0]
        name = row['asset_id']+'.png'
        result.save(a.output/name)
        row.update(image=name, sha256=digest(a.output/name), token_audit=token_audit,
                   synthetic=True, public_model='SDXL-base-1.0', scene_adherence_verified=False)
        write(a.output/'backgrounds.json', rows[:i+1])
        write(a.output/'progress.json', {'stage': 'empty_background_generation', 'done': i+1, 'total': len(rows)})
    for category in ['lamp', 'pillow']:
        selected = [r for r in rows if r['category'] == category]
        sheet = Image.new('RGB', (3*320, 8*348), 'white'); draw = ImageDraw.Draw(sheet)
        for i, row in enumerate(selected):
            x = (i % 3)*320; y = (i//3)*348
            with Image.open(a.output/row['image']) as im: sheet.paste(im.resize((320, 320)), (x, y+28))
            draw.text((x+4, y+4), row['asset_id'], fill='black')
        sheet.save(a.output/(category+'_all_backgrounds_QA.png'))
    del pipe
    gc.collect(); torch.cuda.empty_cache()
    write(a.output/'report.json', {'complete': True, 'stage': 'background_candidates_not_approved_training_labels',
        'count': len(rows), 'TRAIN': 32, 'DEV': 16, 'TEST': 0, 'resolution': 1024,
        'source_commit': os.environ['T12_SOURCE_COMMIT'], 'backgrounds_sha256': digest(a.output/'backgrounds.json'),
        'original_foreground_or_TEST_loaded': False, 'visual_review_required': True,
        'license_note': 'AI-generated SDXL backdrop assets; separate from ABO CC BY 4.0 foreground attribution.'})


def composite(rgba, background, category, placement=None):
    """Use true alpha, premultiplied resize, and a soft contact shadow."""
    if rgba.mode != 'RGBA' or background.size != (1024, 1024): raise ValueError('Original RGBA/1024 backdrop required')
    box = rgba.getchannel('A').getbbox()
    if box is None: raise ValueError('Empty product')
    # RGBa (premultiplied) prevents bright/black fringes around fine ABO edges.
    product = rgba.crop(box).convert('RGBa')
    placement = placement or {}
    maxw, maxh = ((640, 710) if category == 'lamp' else (590, 510))
    maxw = int(placement.get('max_width', maxw)); maxh = int(placement.get('max_height', maxh))
    center = float(placement.get('center_x', .5)); bottom = int(placement.get('bottom_y', 900))
    if not (200 <= maxw <= 820 and 200 <= maxh <= 820 and .2 <= center <= .8 and 550 <= bottom <= 980):
        raise ValueError('Unreviewed product placement')
    scale = min(maxw/product.width, maxh/product.height)
    product = product.resize((round(product.width*scale), round(product.height*scale)), Image.Resampling.LANCZOS).convert('RGBA')
    x = round(1024*center-product.width/2); y = bottom-product.height
    if x < 0 or y < 0 or x+product.width > 1024: raise ValueError('Product would be cropped')
    alpha = Image.new('L', (1024, 1024)); alpha.paste(product.getchannel('A'), (x, y))
    shadow = Image.new('L', (1024, 1024)); draw = ImageDraw.Draw(shadow)
    draw.ellipse((x+product.width*.08, bottom-16, x+product.width*.92, bottom+18), fill=42)
    shadow = np.asarray(shadow.filter(ImageFilter.GaussianBlur(12)), dtype=np.float32)/255
    rgb = np.asarray(background.convert('RGB'), dtype=np.float32)*(1-shadow[:, :, None])
    result = Image.fromarray(rgb.round().clip(0, 255).astype(np.uint8)).convert('RGBA')
    result.alpha_composite(product, (x, y))
    return result.convert('RGB'), alpha, [x, y, x+product.width, y+product.height]


def compose(a):
    report = json.loads((a.sources/'report.json').read_text())
    backgrounds_report = json.loads((a.backgrounds/'report.json').read_text())
    review = json.loads((a.backgrounds/'visual_review.json').read_text())
    assert report['complete'] and not report['TEST_opened'] and backgrounds_report['complete']
    assert review['backgrounds_sha256'] == digest(a.backgrounds/'backgrounds.json')
    # Per-asset factual recaption is allowed only for TRAIN/DEV data preparation.
    # Missing or rejected assets cannot be silently filled by a different split.
    accepted = {r['asset_id']: r for r in review['assets'] if r['approved']}
    assets = json.loads((a.backgrounds/'backgrounds.json').read_text())
    assert set(accepted) <= {r['asset_id'] for r in assets}
    sources = json.loads((a.sources/'sources.json').read_text())
    (a.output/'references').mkdir(); (a.output/'targets').mkdir()
    pairs = []; seen = set(); previews = {}; source_previews = []
    for source in sources:
        assert source['split'] in ['train', 'dev'] and digest(a.sources/source['original_rgba']) == source['rgba_file_sha256']
        reference = 'references/'+source['sample_id']+'.png'
        assert digest(a.sources/source['white_reference']) == source['white_reference_sha256']
        shutil.copyfile(a.sources/source['white_reference'], a.output/reference)
        with Image.open(a.sources/source['original_rgba']) as im: rgba = im.convert('RGBA').copy()
        for asset in assets:
            if asset['asset_id'] not in accepted or asset['split'] != source['split'] or asset['category'] != source['category']: continue
            assert digest(a.backgrounds/asset['image']) == asset['sha256']
            request = conditions(source['category'])[int(asset['condition_id'])]
            factual = accepted[asset['asset_id']]
            prompt = factual.get('model_request', request['prompt'])
            caption = factual.get('native_caption', request['native_caption'])
            sample_id = source['sample_id']+'-'+asset['asset_id']
            target = 'targets/'+sample_id+'.png'
            with Image.open(a.backgrounds/asset['image']) as bg:
                result, alpha, box = composite(rgba, bg, source['category'], factual.get('placement'))
            result.save(a.output/target)
            seed = request_seed(source['white_reference_sha256'], asset['variant'])
            key = (reference, prompt, seed)
            if key in seen: raise ValueError('Conflicting duplicate model input')
            seen.add(key)
            pairs.append({'sample_id': sample_id, 'split': source['split'], 'category': source['category'],
                'reference_id': source['sample_id'], 'reference': reference,
                'reference_sha256': digest(a.output/reference), 'target': target, 'target_sha256': digest(a.output/target),
                'prompt': prompt, 'native_caption': caption, 'prompt_profile': PROFILE,
                'reward_caption': factual.get('reward_caption', request['reward_caption']),
                'condition_id': asset['condition_id'], 'seed': seed, 'source_product_id': source['product_id'],
                'source_RGBA_sha256': source['rgba_file_sha256'], 'background_id': asset['asset_id'],
                'background_sha256': asset['sha256'], 'target_product_bbox': box,
                'foreground_pixels_recolored': False, 'composition_is_TRAIN_DEV_data_only': True})
            if asset['asset_id'] not in previews: previews[asset['asset_id']] = (sample_id, result.resize((256, 256)))
        own = [r for r in pairs if r['reference_id'] == source['sample_id']]
        if own:
            with Image.open(a.output/own[0]['target']) as im: source_previews.append((source['sample_id'], im.resize((256, 256))))
    owners = {}
    pixel_owners = {}
    for row in pairs:
        if owners.setdefault(row['source_product_id'], row['split']) != row['split']: raise ValueError('SKU leakage')
        if pixel_owners.setdefault(row['source_RGBA_sha256'], row['split']) != row['split']: raise ValueError('Exact foreground leakage')
    write(a.output/'pairs.json', pairs)
    for name, values in [('all_background_placements_QA', list(previews.values())),
                         ('all_source_first_targets_QA', source_previews)]:
        for page, start in enumerate(range(0, len(values), 24)):
            subset = values[start:start+24]
            sheet = Image.new('RGB', (4*256, math.ceil(len(subset)/4)*282), 'white'); draw = ImageDraw.Draw(sheet)
            for i, (label, im) in enumerate(subset):
                x = (i%4)*256; y = (i//4)*282
                draw.text((x+2, y+2), label, fill='black'); sheet.paste(im, (x, y+26))
            sheet.save(a.output/(name+f'_page{page+1}.png'))
    write(a.output/'inference_dev_inputs.json', [{k:r[k] for k in ['sample_id','reference','prompt','seed',
        'reference_id','reference_sha256','category','condition_id','native_caption','reward_caption','prompt_profile']}
        for r in pairs if r['split'] == 'dev'])
    write(a.output/'report.json', {'complete': True, 'source_commit': os.environ['T12_SOURCE_COMMIT'],
        'pairs_sha256': digest(a.output/'pairs.json'), 'counts': dict(Counter(r['split'] for r in pairs)),
        'target_resolution': 1024, 'source_products': dict(Counter(owners.values())),
        'role_disjoint': True, 'TEST_opened': False, 'paired_targets_are_one_valid_training_example_not_unique_test_GT': True,
        'source_report_sha256': digest(a.sources/'report.json'), 'background_review_sha256': digest(a.backgrounds/'visual_review.json'),
        'original_RGBA_native_sizes': report['native_size_counts'], 'compositing_at_inference': False,
        'foreground_attribution': 'Amazon Berkeley Objects (ABO), CC BY 4.0',
        'background_attribution': 'Own public SDXL-generated backdrops; not ABO photographs.',
        'final_scene_visual_review_required': True})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=['backgrounds', 'compose'], required=True)
    for key in ['sources', 'backgrounds', 'model-path']: p.add_argument('--'+key, type=Path)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists() or len(os.environ.get('T12_SOURCE_COMMIT', '')) != 40: raise ValueError('New run and published source required')
    a.output.mkdir(parents=True)
    write(a.output/'execution.json', {'argv': sys.argv, 'pid': os.getpid(),
        'source_sha256': digest(Path(__file__)), 'source_commit': os.environ['T12_SOURCE_COMMIT']})
    try: generate(a) if a.stage == 'backgrounds' else compose(a)
    except Exception:
        write(a.output/'failure.json', {'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__': main()
