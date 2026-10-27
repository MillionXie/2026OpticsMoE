"""Frozen public SD-Turbo backgrounds for TRAIN/development data preparation.

These are synthetic data assets, not editor predictions or benchmark scores.
ABO's CC BY 4.0 provenance applies to ABO renders, not a claim about these assets.
The independent evaluation split is deliberately unsupported by this producer.
"""
import argparse
import itertools
import json
import os
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from .qwen_vl_turbo_data import seed, sha256, write_json
from .qwen_vl_turbo_model import NativeTurboHead
from .qwen_vl_turbo_run import tensor_image


def specifications(base_seed=12491, layout='room', probe=False):
    rows = []
    for split, count in (('train', 2), ('val', 1)):
        for room, tone, brightness, direction in itertools.product(
                ('modern study', 'minimal bedroom', 'boutique lounge'),
                ('warm', 'cool', 'neutral'), ('bright', 'softly lit'), ('left', 'right')):
            for variant in range(count):
                prompt = (f'Interior photograph of a {brightness} {tone} {room}, window on the {direction}, '
                          f'large empty wooden display surface in the foreground, soft {direction} window light, '
                          'eye-level camera, clean uncluttered central surface, no lamp, no cushion, no people.')
                if layout == 'tabletop':
                    context = {'modern study': 'bookshelves', 'minimal bedroom': 'a bed and headboard',
                               'boutique lounge': 'an upholstered sofa and curtains'}[room]
                    prompt = ('An empty wooden tabletop, close-up at table height, tabletop fills the lower half, '
                              f'{context} softly blurred in a {brightness} {tone} {room} behind it, '
                              f'daylight from a single window on the {direction}, '
                              'product photography backdrop, clear center, no products or people.')
                if probe and (split != 'train' or variant != 0 or tone != 'warm' or brightness != 'bright'):
                    continue
                rows.append({'split': split, 'room': room, 'tone': tone, 'brightness': brightness,
                    'direction': direction, 'variant': variant, 'prompt': prompt, 'layout': layout,
                    'seed': seed(base_seed, split, room, tone, brightness, direction, variant) % (2**31)})
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sd', type=Path, required=True);p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpu-uuid', required=True)
    p.add_argument('--layout', choices=('room', 'tabletop'), default='room');p.add_argument('--probe', action='store_true')
    a = p.parse_args()
    if a.output.exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != a.gpu_uuid:
        raise ValueError('Immutable assets and approved GPU only')
    a.output.mkdir(parents=True);torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.6, 0)
    torch.manual_seed(12491);device = torch.device('cuda:0')
    from diffusers import AutoencoderKL, UNet2DConditionModel, EulerDiscreteScheduler
    from transformers import CLIPTextModel, CLIPTokenizer
    unet = UNet2DConditionModel.from_pretrained(a.sd, subfolder='unet', variant='fp16', torch_dtype=torch.bfloat16, local_files_only=True).to(device).eval().requires_grad_(False)
    vae = AutoencoderKL.from_pretrained(a.sd, subfolder='vae', variant='fp16', torch_dtype=torch.bfloat16, local_files_only=True).to(device).eval().requires_grad_(False)
    vae.encoder = None;vae.quant_conv = None
    text = CLIPTextModel.from_pretrained(a.sd, subfolder='text_encoder', variant='fp16', torch_dtype=torch.bfloat16, local_files_only=True).to(device).eval().requires_grad_(False)
    tokenizer = CLIPTokenizer.from_pretrained(a.sd, subfolder='tokenizer', local_files_only=True)
    model = NativeTurboHead(unet, vae, EulerDiscreteScheduler.from_pretrained(a.sd, subfolder='scheduler', local_files_only=True), torch.nn.Identity())
    records = specifications(layout=a.layout, probe=a.probe)
    for i, row in enumerate(records):
        tokens = tokenizer(row['prompt'], padding='max_length', max_length=77, truncation=False, return_tensors='pt').input_ids
        if tokens.shape[1] > 77:
            raise ValueError('Do not silently truncate scene attributes')
        with torch.inference_mode():
            condition = text(tokens.to(device)).last_hidden_state
            noise = torch.randn(1, 4, 64, 64, device=device, generator=torch.Generator(device=device).manual_seed(row['seed']))
            _, image = model.generate_from_conditions(condition, noise)
        original = a.output/row['split']/(f'{i:03d}_512.png');original.parent.mkdir(exist_ok=True)
        rgb = tensor_image(image[0]);rgb.save(original)
        small = original.with_name(f'{i:03d}_256.png');rgb.resize((256, 256), Image.Resampling.LANCZOS).save(small)
        row.update(image=str(small.resolve()), sha256=sha256(small), original_512_sha256=sha256(original))
        write_json(a.output/'progress.json', {'stage': 'TRAIN_DEV_background_asset_generation', 'done': i+1, 'total': len(records)})
    for room in ('modern study', 'minimal bedroom', 'boutique lounge'):
        chosen = [r for r in records if r['split'] == 'train' and r['room'] == room and r['variant'] == 0]
        panel = Image.new('RGB', (640, 3*188), 'white');draw = ImageDraw.Draw(panel)
        for j, row in enumerate(chosen):
            x=j%4*160;y=j//4*188;im=Image.open(row['image']).convert('RGB');im.thumbnail((160, 160))
            draw.text((x+2, y+3), row['tone']+' / '+row['brightness'], fill='black')
            draw.text((x+2, y+15), row['direction']+' window', fill='black');panel.paste(im, (x,y+28))
        panel.save(a.output/(room.replace(' ', '_')+'_candidates.png'))
    write_json(a.output/'background_manifest.json', records)
    write_json(a.output/'report.json', {'complete': True, 'count': len(records),
        'TRAIN': sum(r['split']=='train' for r in records), 'DEV': sum(r['split']=='val' for r in records),
        'layout': a.layout, 'preview_only': a.probe,
        'public_frozen_teacher': 'SD-Turbo', 'dataset_assets_not_editor_outputs': True,
        'source_commit': os.environ.get('T12_SOURCE_COMMIT'), 'source_sha256': sha256(Path(__file__)),
        'manifest_sha256': sha256(a.output/'background_manifest.json'), 'visual_review_required': True,
        'heldout_generated': False, 'not_model_benchmark': True,
        'seed_policy': 'TRAIN and DEV seeds disjoint; no independent TEST path. A future frozen-model holdout needs separately bound backgrounds/seeds.',
        'limitations': 'Scene instructions are teacher generation prompts, not proof that every asset satisfies window direction/clear tabletop. Review before composing ABO or making scene adherence claims.'})


if __name__ == '__main__':
    main()
