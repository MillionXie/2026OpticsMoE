"""TRAIN-only oracle diagnostic of Qwen alignment versus spatial conditioning.

Oracle caption features are privileged TRAIN labels, never a deployable model
result. This module performs no optimizer updates and never opens DEV or TEST.
"""
import argparse
import contextlib
import copy
import gc
import json
import math
import os
import textwrap
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from .qwen_vl_hybrid_run import completed_teacher, load_data
from .qwen_vl_postfree import FocusedHead
from .qwen_vl_postfree_run import build_generator
from .qwen_vl_turbo_data import sha256, write_json
from .qwen_vl_turbo_model import NativeTurboHead, pack_hidden
from .qwen_vl_turbo_run import noise_for, tensor_image
from .qwen_vl_turbo_cache import image_float


def restore_parameters(model, state):
    known = dict(model.named_parameters())
    if not state or set(state)-set(known):
        raise ValueError('Unknown/empty focused model parameters')
    for name, tensor in state.items():
        if known[name].shape != tensor.shape:
            raise ValueError('Focused parameter shape differs: '+name)
        # Training promoted selected parameters to FP32; preserve their dtype.
        known[name].data = tensor.detach().to(known[name].device)


def train_examples(rows):
    result = []
    for category in ('lamp', 'pillow'):
        used = set()
        for i, row in enumerate(rows):
            if row['split'] != 'train' or row['category'] != category or row['mode'] != 'object':
                continue
            target = row['target']['product_id']
            if target in used:
                continue
            used.add(target); result.append(i)
            if len(used) == 2:
                break
        if len(used) != 2:
            raise ValueError('Require two distinct TRAIN target products per category')
    return result


@contextlib.contextmanager
def override_conditions(model, native=None, no_spatial=False):
    adapter, spatial = model.base.adapter, model.base.spatial
    original_adapter, original_spatial = adapter.forward, spatial.forward
    try:
        if native is not None:
            adapter.forward = lambda *args, **kwargs: native
        if no_spatial:
            spatial.forward = lambda *args, **kwargs: original_spatial(*args, **kwargs)*0
        yield
    finally:
        adapter.forward, spatial.forward = original_adapter, original_spatial


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('parent', 'tail', 'data', 'baseline', 'lg', 'sd', 'output'):
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--gpu-uuid', required=True)
    a = p.parse_args()
    if a.output.exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != a.gpu_uuid:
        raise ValueError('Existing output or wrong GPU UUID')
    completed_teacher(a.parent)
    parent = torch.load(a.parent/'best_checkpoint.pt', map_location='cpu', weights_only=False)
    tail = torch.load(a.tail/'best_checkpoint.pt', map_location='cpu', weights_only=False)
    payload = load_data(a.data); ids = train_examples(payload['rows'])
    if any(payload['rows'][i]['split'] != 'train' or i not in payload['train_clip_teacher'] for i in ids):
        raise ValueError('Oracle diagnostic may use only TRAIN labels')
    torch.set_num_threads(4); torch.manual_seed(1042)
    torch.cuda.set_per_process_memory_fraction(.85, 0); device = torch.device('cuda:0')
    a.output.mkdir(parents=True)
    write_json(a.output/'execution.json', {'pid': os.getpid(), 'started_unix': time.time(),
        'source_commit': os.environ.get('T12_SOURCE_COMMIT'), 'source_sha256': sha256(Path(__file__)),
        'gpu_uuid': a.gpu_uuid, 'no_updates': True, 'TRAIN_only': True, 'heldout_240_opened': False,
        'oracle_caption_features_are_privileged_training_labels': True,
        'sample_ids': [payload['rows'][i]['sample_id'] for i in ids]})
    import lpips
    metric = lpips.LPIPS(net='alex').to(device).eval().requires_grad_(False)
    records, pictures, metadata = [], {}, {}
    for label, run in (('baseline', a.baseline), ('LG', a.lg)):
        report = json.loads((run/'report.json').read_text())
        if not report['complete'] or report['best_step'] <= 0 or sha256(run/'best_checkpoint.pt') != report['checkpoint_sha256']:
            raise ValueError('Selected model identity differs')
        selected = torch.load(run/'best_checkpoint.pt', map_location='cpu', weights_only=False)
        if sha256(run/'postfree_features.pt') != selected['feature_sha256']:
            raise ValueError('Frozen actual frontend cache differs')
        features = torch.load(run/'postfree_features.pt', map_location='cpu', weights_only=False, mmap=True)
        if features['sample_ids'] != [r['sample_id'] for r in payload['rows']]:
            raise ValueError('Cache row identity differs')
        base = build_generator(a.sd, parent, tail, device, drop_encoder=True,
                               compressed=selected['config'].get('compressed_tail', True))
        model = FocusedHead(base, parent['visual_layout']['positions'], *selected['instruction_bounds']).to(device)
        restore_parameters(model, selected['trainable_head']); model.eval().requires_grad_(False)
        metadata[label] = {'checkpoint_sha256': report['checkpoint_sha256'],
                           'feature_sha256': selected['feature_sha256']}
        with torch.inference_mode():
            for i in ids:
                row = payload['rows'][i]
                h, valid = pack_hidden([features['original_hidden'][i]], device)
                native = payload['train_clip_teacher'][i][None].float().to(device)
                for variant, use_native, no_spatial in (
                        ('actual_focused', False, False), ('oracle_caption', True, False),
                        ('oracle_caption_no_spatial', True, True), ('actual_no_spatial', False, True)):
                    with override_conditions(model, native if use_native else None, no_spatial):
                        _, image = model(h, valid, noise_for([row], device))
                    image = image.clamp(-1, 1)
                    target = image_float(payload['target'][i:i+1], device)
                    mse = float(((image-target).square()/4).mean())
                    path = a.output/label/(str(i)+'_'+variant+'.png'); path.parent.mkdir(exist_ok=True)
                    picture = tensor_image(image[0]); picture.save(path); pictures[label, i, variant] = picture
                    records.append({'model': label, 'variant': variant, 'sample_id': row['sample_id'],
                        'category': row['category'], 'split': 'train', 'prompt': row['prompt'],
                        'oracle_caption': row['teacher_caption'] if use_native else None,
                        'native_caption_override': use_native, 'spatial_disabled': no_spatial,
                        'TRAIN_diagnostic_psnr': -10*math.log10(max(mse, 1e-12)),
                        'TRAIN_diagnostic_lpips': float(metric(image, target).mean()),
                        'output_sha256': sha256(path), 'not_deployable_performance': True})
        del features, model, base, selected; gc.collect(); torch.cuda.empty_cache()
    # Output-only diagnostics: the source RGB and GT are never sent to the
    # public native caption generator, which has a different task and no editor.
    from diffusers import UNet2DConditionModel, AutoencoderKL, EulerDiscreteScheduler
    public_unet = UNet2DConditionModel.from_pretrained(a.sd, subfolder='unet', variant='fp16',
        torch_dtype=torch.bfloat16, local_files_only=True).to(device).eval().requires_grad_(False)
    vae = AutoencoderKL.from_pretrained(a.sd, subfolder='vae', variant='fp16',
        torch_dtype=torch.bfloat16, local_files_only=True).to(device).eval().requires_grad_(False)
    vae.encoder = None; vae.quant_conv = None
    generator = NativeTurboHead(public_unet, vae, EulerDiscreteScheduler.from_pretrained(a.sd, subfolder='scheduler', local_files_only=True), torch.nn.Identity())
    with torch.inference_mode():
        for i in ids:
            _, image = generator.generate_from_conditions(payload['train_clip_teacher'][i][None].to(device), noise_for([payload['rows'][i]], device))
            picture = tensor_image(image[0]); path = a.output/('public_native_'+str(i)+'.png');picture.save(path)
            pictures['public', i, 'caption'] = picture
    for label in ('baseline', 'LG'):
        for category in ('lamp', 'pillow'):
            selected_ids = [i for i in ids if payload['rows'][i]['category'] == category]
            panel = Image.new('RGB', (1584, 410*len(selected_ids)), 'white'); draw = ImageDraw.Draw(panel)
            for j, i in enumerate(selected_ids):
                row = payload['rows'][i]; y = j*410
                draw.text((8, y+4), 'TRAIN diagnostic ONLY | '+label+' / '+row['sample_id'], fill='black')
                draw.multiline_text((8, y+20), '\n'.join(textwrap.wrap('Edit: '+row['prompt'], 200)), fill='black', spacing=2)
                draw.multiline_text((8, y+55), '\n'.join(textwrap.wrap('Privileged TRAIN caption: '+row['teacher_caption'], 200)), fill='black', spacing=2)
                tiles = [tensor_image(payload['reference'][i]), tensor_image(payload['target'][i]),
                    pictures[label, i, 'actual_focused'], pictures[label, i, 'oracle_caption'],
                    pictures[label, i, 'oracle_caption_no_spatial'], pictures['public', i, 'caption']]
                labels = ['Reference', 'GT (display/metric only)', 'Actual focused model', 'Oracle caption + spatial', 'Oracle caption, no spatial', 'Public text-only oracle']
                for k, (caption, image) in enumerate(zip(labels, tiles)):
                    draw.text((8+k*264, y+116), caption, fill='black'); panel.paste(image, (8+k*264, y+138))
            panel.save(a.output/(label+'_'+category+'_oracle_probe.png'))
    write_json(a.output/'report.json', {'complete': True, 'TRAIN_only': True, 'no_updates': True,
        'heldout_240_opened': False, 'models': metadata, 'records': records,
        'oracle_is_not_a_deployment_or_baseline_comparison': True,
        'GT_image_or_mask_input': False, 'training_label_caption_override_is_explicit': True,
        'public_caption_generator_is_text_only_not_same_editing_task': True})


if __name__ == '__main__':
    main()
