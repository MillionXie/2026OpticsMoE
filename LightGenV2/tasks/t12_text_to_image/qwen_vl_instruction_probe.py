"""Small TRAIN-only, same-RGB/seed instruction sensitivity diagnostic.

Compare full Qwen and its frozen optical replacement without training, GT input,
or opening the 240 held-out cohort. Original panels are saved without enhancement.
"""
import argparse
import gc
import hashlib
import json
import os
import textwrap
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from .qwen_vl_hybrid import apply_hybrid, load_macro_state
from .qwen_vl_hybrid_run import completed_teacher, load_data, load_qwen, raw_inputs
from .qwen_vl_spatial import build_spatial_head
from .qwen_vl_turbo_data import sha256, write_json
from .qwen_vl_turbo_model import pack_hidden
from .qwen_vl_turbo_run import tensor_image


def prompt_variants(category):
    if category == 'lamp':
        long = 'Replace the central lamp with a desk lamp with a red metal shade, a curved dark-bronze arm and a dark circular base, entirely visible. Preserve the surrounding background and lighting.'
        short = 'Change the lamp to a red desk lamp. Keep the background.'
        identity = 'Keep this lamp and background unchanged.'
        background = 'Keep this lamp. Use a bright warm study background.'
    elif category == 'pillow':
        long = 'Replace the central pillow with a square red cushion with a plain woven fabric face, soft rounded corners and narrow edge piping, entirely visible. Preserve the surrounding background and lighting.'
        short = 'Change the cushion to a plain red cushion. Keep the background.'
        identity = 'Keep this cushion and background unchanged.'
        background = 'Keep this cushion. Use a bright warm study background.'
    else:
        raise ValueError('Unapproved probe category')
    return {'long_red': long, 'short_red': short,
            'short_blue': short.replace('red', 'blue'),
            'short_black': short.replace('red', 'black'),
            'identity': identity, 'short_background': background}


def difference(a, b):
    x, y = a.float(), b.float()
    return {'mean_absolute': float((x-y).abs().mean()),
            'relative_rms': float((x-y).square().mean().sqrt()/x.square().mean().sqrt().clamp_min(1e-8))}


def select_train_sources(rows):
    selected = []
    for category in ('lamp', 'pillow'):
        selected.append(next(i for i, row in enumerate(rows)
                             if row['split'] == 'train' and row['category'] == category))
    if any(rows[i]['split'] != 'train' for i in selected):
        raise ValueError('Instruction probe must never use development or held-out rows')
    return selected


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('parent', 'hybrid', 'data', 'qwen', 'sd', 'output'):
        p.add_argument('--'+key, type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError('Preserve existing instruction diagnostic')
    teacher = completed_teacher(a.parent)
    state = torch.load(a.parent/'best_checkpoint.pt', map_location='cpu', weights_only=False)
    hybrid_report = json.loads((a.hybrid/'report.json').read_text())
    if not hybrid_report['complete'] or sha256(a.hybrid/'best_checkpoint.pt') != hybrid_report['checkpoint_sha256']:
        raise ValueError('Need complete selected hybrid checkpoint')
    hybrid = torch.load(a.hybrid/'best_checkpoint.pt', map_location='cpu', weights_only=False)
    gpu = state['config']['gpu_uuid']
    if os.environ.get('CUDA_VISIBLE_DEVICES') != gpu:
        raise ValueError('Only the explicitly approved GPU6 UUID is permitted')
    torch.set_num_threads(4)
    torch.manual_seed(1042)
    torch.cuda.set_per_process_memory_fraction(.85, 0)
    a.output.mkdir(parents=True)
    write_json(a.output/'execution.json', {'pid': os.getpid(), 'started_unix': time.time(),
        'source_commit': os.environ.get('T12_SOURCE_COMMIT'), 'source_sha256': sha256(Path(__file__)),
        'gpu_uuid': gpu, 'teacher_sha256': teacher['checkpoint_sha256'],
        'hybrid_sha256': hybrid_report['checkpoint_sha256'],
        'TRAIN_only': True, 'no_updates': True, 'heldout_240_opened': False})
    payload = load_data(a.data)
    sources = select_train_sources(payload['rows'])
    device = torch.device('cuda:0')
    qwen, processor = load_qwen(a.qwen, device)
    layout = state['visual_layout']
    features, token_audit = {}, []
    for model in ('baseline', 'LG'):
        if model == 'LG':
            macros = apply_hybrid(qwen, hybrid['config']['vision_pairs'],
                                 hybrid['config']['language_pairs'], hybrid['config']['residual_width'])
            load_macro_state(macros, hybrid['macros'])
            qwen.eval().requires_grad_(False)
        for i in sources:
            row = payload['rows'][i]
            for variant, prompt in prompt_variants(row['category']).items():
                inputs = raw_inputs(processor, payload['reference'][i], prompt, layout, device)
                with torch.inference_mode():
                    hidden = qwen.model(**inputs, use_cache=False, return_dict=True).last_hidden_state[0].bfloat16().cpu()
                features[model, row['category'], variant] = hidden
                token_audit.append({'model': model, 'category': row['category'], 'variant': variant,
                    'instruction_tokens': len(processor.tokenizer.encode(prompt, add_special_tokens=False)),
                    'sequence_tokens': len(hidden), 'prompt': prompt})
    del qwen, processor
    gc.collect(); torch.cuda.empty_cache()
    head = build_spatial_head(a.sd, state, state['config'], layout['positions'], device,
                              drop_vae_encoder=True).eval().requires_grad_(False)
    records, outputs, conditions = [], {}, {}
    for model in ('baseline', 'LG'):
        for i in sources:
            row = payload['rows'][i]; category = row['category']
            for variant, prompt in prompt_variants(category).items():
                hidden = features[model, category, variant]
                h, valid = pack_hidden([hidden], device)
                noise = torch.randn(1, 4, 32, 32, device=device,
                                    generator=torch.Generator(device=device).manual_seed(1042))
                with torch.inference_mode():
                    condition = head.semantic_conditions(h, valid).float().cpu()
                    _, rgb = head(h, valid, noise)
                image = tensor_image(rgb[0]); path = a.output/model/(category+'_'+variant+'.png')
                path.parent.mkdir(exist_ok=True); image.save(path)
                outputs[model, category, variant] = np.asarray(image)
                conditions[model, category, variant] = condition
                records.append({'model': model, 'sample_id': row['sample_id'], 'split': row['split'],
                    'category': category, 'variant': variant, 'prompt': prompt, 'seed': 1042,
                    'reference_raw_rgb_sha256': hashlib.sha256(payload['reference'][i].numpy().tobytes()).hexdigest(),
                    'output': str(path), 'png_sha256': sha256(path)})
    distances = []
    for model in ('baseline', 'LG'):
        for i in sources:
            category = payload['rows'][i]['category']
            for other in ('long_red', 'short_blue', 'short_black', 'identity', 'short_background'):
                key = model, category, 'short_red'; compare = model, category, other
                pixels_a, pixels_b = outputs[key].astype(float), outputs[compare].astype(float)
                distances.append({'model': model, 'category': category, 'against_short_red': other,
                    'visual_token_difference': difference(features[key][layout['positions']], features[compare][layout['positions']]),
                    'adapter_condition_difference': difference(conditions[key], conditions[compare]),
                    'generated_mean_absolute_8bit': float(abs(pixels_a-pixels_b).mean()),
                    'note': 'Sensitivity only; large pixel or condition change does not establish semantic correctness.'})
    for i in sources:
        category = payload['rows'][i]['category']
        variants = list(prompt_variants(category).items())
        panel = Image.new('RGB', (1056, 320*len(variants)), 'white')
        draw = ImageDraw.Draw(panel)
        for j, (variant, prompt) in enumerate(variants):
            y = j*320
            draw.text((8, y+3), category+' / '+variant+' | fixed TRAIN reference and seed 1042', fill='black')
            draw.multiline_text((8, y+18), '\n'.join(textwrap.wrap(prompt, 145)), fill='black', spacing=2)
            images = [tensor_image(payload['reference'][i]),
                      Image.fromarray(outputs['baseline', category, variant]),
                      Image.fromarray(outputs['LG', category, variant])]
            for k, (label, image) in enumerate(zip(('REFERENCE', 'BASELINE', 'LG (same full generation tail)'), images)):
                draw.text((k*352+8, y+52), label, fill='black');panel.paste(image, (k*352+8, y+64))
        panel.save(a.output/(category+'_instruction_comparison.png'))
    write_json(a.output/'report.json', {'complete': True, 'TRAIN_only': True, 'no_updates': True,
        'heldout_240_opened': False, 'reference_or_target_vae_input': False,
        'target_or_mask_input': False, 'same_reference_and_seed': True,
        'baseline_checkpoint_sha256': teacher['checkpoint_sha256'],
        'hybrid_checkpoint_sha256': hybrid_report['checkpoint_sha256'],
        'generation_tail': 'Identical uncompressed baseline head for both; isolates Qwen replacement.',
        'records': records, 'token_audit': token_audit, 'sensitivities': distances,
        'not_a_quality_benchmark': True})


if __name__ == '__main__':
    main()
