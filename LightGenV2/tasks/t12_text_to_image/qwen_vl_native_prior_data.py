"""TRAIN-reference text alignment; optional public-teacher latent supervision.

Rendered supervision is explicitly synthetic distillation data, not an ABO
paired target or a DEV/TEST result. No reference image enters the teacher.
"""
import argparse
import gc
import json
import os
from pathlib import Path

import torch

from .qwen_vl_hybrid_run import load_data, load_qwen, raw_inputs
from .qwen_vl_postfree import instruction_bounds
from .qwen_vl_turbo_data import sha256, write_json
from .qwen_vl_turbo_cache import save_with_digest
from .qwen_vl_turbo_model import NativeTurboHead
from .qwen_vl_turbo_run import noise_for, tensor_image


def alignment_specs(rows):
    result = []
    colors = ('red', 'blue', 'green', 'yellow', 'black', 'white', 'gray', 'teal')
    for category in ('lamp', 'pillow'):
        sources = []; used = set()
        for index, row in enumerate(rows):
            if row['split'] == 'train' and row['category'] == category and row['source']['product_id'] not in used:
                used.add(row['source']['product_id']); sources.append(index)
                if len(sources) == 4:
                    break
        if len(sources) != 4:
            raise ValueError('Four TRAIN reference products per category required')
        for index in sources:
            for color in colors:
                descriptions = ((f'a {color} table lamp with a cylindrical fabric shade and a rounded ceramic base',
                                 f'a {color} metal desk lamp with a curved arm and a circular base') if category == 'lamp' else
                                (f'a square {color} fabric cushion', f'a {color} cushion with a white geometric pattern'))
                room, tone, brightness, direction = rows[index]['source_scene']
                for description in descriptions:
                    result.append({'source_index': index, 'source_split': 'train', 'category': category,
                        'prompt': f'Replace the central {category} with {description}. Keep the background.',
                        'caption': f'A full view of {description}, centered and entirely visible, in a {brightness} {tone} {room.replace("_", " ")} with {direction} window light.'})
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('data', 'parent', 'qwen', 'sd', 'output'):
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--gpu-uuid', required=True)
    p.add_argument('--reuse', type=Path)
    p.add_argument('--render-teacher', action='store_true')
    a = p.parse_args()
    if a.output.exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != a.gpu_uuid:
        raise ValueError('Existing output or wrong GPU')
    a.output.mkdir(parents=True);torch.set_num_threads(4);torch.manual_seed(1042)
    torch.cuda.set_per_process_memory_fraction(.85, 0);device = torch.device('cuda:0')
    reused_sha = None
    if a.reuse:
        report = json.loads((a.reuse/'report.json').read_text()); reused_sha = sha256(a.reuse/'alignment_examples.pt')
        if not report['complete'] or reused_sha != report['sha256']:
            raise ValueError('Prior TRAIN text cache SHA differs')
        prior = torch.load(a.reuse/'alignment_examples.pt', map_location='cpu', weights_only=False)
        if not prior['TRAIN_only'] or prior['source_frontend'] != 'full_frozen_qwen':
            raise ValueError('Prior text cache uses a different split/frontend')
        specs, features, targets, boundaries = (prior[k] for k in ('specs', 'hidden', 'native_clip', 'instruction_bounds'))
        if len(specs) != 128 or any(s['source_split'] != 'train' for s in specs):
            raise ValueError('Synthetic distillation requires TRAIN sources')
    else:
        payload = load_data(a.data); specs = alignment_specs(payload['rows'])
        if len(specs) != 128 or any(payload['rows'][s['source_index']]['split'] != 'train' for s in specs):
            raise ValueError('Synthetic text alignment must use TRAIN references only')
        parent = torch.load(a.parent/'best_checkpoint.pt', map_location='cpu', weights_only=False)
        qwen, processor = load_qwen(a.qwen, device); boundaries = None; features = []
        for j, spec in enumerate(specs):
            inputs = raw_inputs(processor, payload['reference'][spec['source_index']], spec['prompt'], parent['visual_layout'], device)
            bound = instruction_bounds(inputs['input_ids'][0], processor.tokenizer.convert_tokens_to_ids('<|vision_end|>'), processor.tokenizer.convert_tokens_to_ids('<|im_end|>'))
            if boundaries is None:
                boundaries = bound
            if bound != boundaries:
                raise ValueError('Synthetic instruction layout differs')
            with torch.inference_mode():
                features.append(qwen.model(**inputs, use_cache=False, return_dict=True).last_hidden_state[0].bfloat16().cpu())
            if (j+1) % 16 == 0:
                write_json(a.output/'progress.json', {'stage': 'TRAIN_text_alignment_qwen', 'done': j+1, 'total': len(specs)})
        del qwen, processor;gc.collect();torch.cuda.empty_cache()
        from transformers import CLIPTextModel, CLIPTokenizer
        teacher = CLIPTextModel.from_pretrained(a.sd, subfolder='text_encoder', variant='fp16', torch_dtype=torch.bfloat16, local_files_only=True).to(device).eval().requires_grad_(False)
        tokenizer = CLIPTokenizer.from_pretrained(a.sd, subfolder='tokenizer', local_files_only=True)
        targets = []
        for start in range(0, len(specs), 16):
            tokens = tokenizer([s['caption'] for s in specs[start:start+16]], padding='max_length', max_length=77, truncation=True, return_tensors='pt').input_ids.to(device)
            with torch.inference_mode():
                targets.extend(teacher(tokens).last_hidden_state.bfloat16().cpu())
        del teacher;gc.collect();torch.cuda.empty_cache()
    cached = {'specs': specs, 'hidden': features, 'native_clip': targets,
        'instruction_bounds': boundaries, 'source_frontend': 'full_frozen_qwen',
        'synthetic_target_images': a.render_teacher, 'TRAIN_only': True}
    if a.render_teacher:
        from diffusers import UNet2DConditionModel, AutoencoderKL, EulerDiscreteScheduler
        unet = UNet2DConditionModel.from_pretrained(a.sd, subfolder='unet', variant='fp16', torch_dtype=torch.bfloat16, local_files_only=True).to(device).eval().requires_grad_(False)
        vae = AutoencoderKL.from_pretrained(a.sd, subfolder='vae', variant='fp16', torch_dtype=torch.bfloat16, local_files_only=True).to(device).eval().requires_grad_(False)
        vae.encoder = None;vae.quant_conv = None
        renderer = NativeTurboHead(unet, vae, EulerDiscreteScheduler.from_pretrained(a.sd, subfolder='scheduler', local_files_only=True), torch.nn.Identity())
        latents, images = [], []
        for start in range(0, len(specs), 4):
            with torch.inference_mode():
                conditions = torch.stack(targets[start:start+4]).to(device)
                z, rgb = renderer.generate_from_conditions(conditions, noise_for([{'noise_seed': 1042}]*len(conditions), device))
            latents.extend(z.bfloat16().cpu());images.extend(rgb.clamp(-1, 1).bfloat16().cpu())
            write_json(a.output/'progress.json', {'stage': 'TRAIN_public_teacher_latents', 'done': start+len(conditions), 'total': len(specs)})
        cached.update(teacher_latents=latents, teacher_rgb=images, noise_seed=1042,
            teacher_reference_input=False, teacher_is_public_native_text_only=True)
        preview = a.output/'teacher_preview';preview.mkdir()
        for i in (0, 2, 4, 6, 64, 66, 68, 70):
            tensor_image(images[i]).save(preview/(str(i)+'.png'))
    digest = save_with_digest(cached, a.output/'alignment_examples.pt')
    write_json(a.output/'report.json', {'complete': True, 'count': len(specs), 'sha256': digest,
        'source_commit': os.environ.get('T12_SOURCE_COMMIT'), 'source_sha256': sha256(Path(__file__)),
        'TRAIN_reference_products': 8, 'extra_ABO_paired_images': 0, 'synthetic_teacher_targets': 128 if a.render_teacher else 0,
        'target_images_generated': a.render_teacher, 'teacher_reference_input': False,
        'new_text_conditions_only': not a.render_teacher, 'no_heldout_forward': True, 'reused_text_sha256': reused_sha,
        'pretraining_exposure_claim': False})


if __name__ == '__main__':
    main()
