"""Frozen native SD-Turbo text-only prior control, not reference-guided editing.

Eight generic captions x two fixed seeds x native256/512 produce32 images.
No dataset, reference image, Qwen, GT, mask, training cache or optical model is
loaded. This different-task/resolution control cannot establish editor quality,
reference preservation, a performance comparison or a twofold speedup.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import textwrap
import time
import traceback

import torch
from PIL import Image, ImageDraw

from .qwen_vl_turbo_data import sha256, write_json
from .qwen_vl_turbo_model import NativeTurboHead
from .qwen_vl_turbo_run import tensor_image


SCHEMA = 'sd_turbo_text_only_prior_control_v1'
GPU_UUID = 'GPU-3f60d773-4dd8-d706-ef48-a4dfc5af6c55'


def control_prompts():
    rows = []
    for category in ('lamp', 'pillow'):
        for color, wall, surface, side in (('red', 'pale blue', 'wooden', 'right'),
                                           ('blue', 'warm beige', 'neutral stone', 'left')):
            noun = f'a table lamp with a {color} lampshade' if category == 'lamp' else f'a pillow with {color} fabric'
            color_key = 'lampshade_color' if category == 'lamp' else 'fabric_color'
            for level in ('single', 'composed'):
                attrs = {color_key: color}
                prompt = noun[0].upper()+noun[1:]
                if level == 'composed':
                    prompt += f', against a {wall} wall, on a {surface} tabletop, lit from the {side}'
                    attrs.update(wall_color=wall, tabletop_material=surface, side_light_direction=side)
                rows.append({'prompt_id': f'{category}_{color}_{level}', 'category': category,
                    'color': color, 'prompt_level': level, 'prompt': prompt+'.', 'requested_attributes': attrs})
    return rows


def control_records():
    return [{**row, 'resolution': resolution, 'noise_seed': seed,
             'image_id': f"{row['prompt_id']}_{resolution}_seed{seed}"}
            for row in control_prompts() for resolution in (256, 512) for seed in (1042, 1043)]


@torch.no_grad()
def native_text_conditions(tokenizer, text, prompt, device):
    if not isinstance(prompt, str) or not prompt.strip() or getattr(tokenizer, 'model_max_length', 77) != 77:
        raise ValueError('A complete nonempty caption and native CLIP77 tokenizer are required')
    full = tokenizer(prompt, padding=False, truncation=False, return_tensors='pt').input_ids
    padded = tokenizer(prompt, padding='max_length', max_length=77, truncation=False, return_tensors='pt').input_ids
    if (full.ndim != 2 or full.shape[0] != 1 or full.shape[1] > 77 or padded.shape != (1, 77) or
            not torch.equal(full, padded[:, :full.shape[1]])):
        raise ValueError('Every native CLIP token must fit77 without truncation or prefix changes')
    conditions = text(padded.to(device)).last_hidden_state
    if not torch.isfinite(conditions).all() or conditions.shape[:2] != (1, 77):
        raise FloatingPointError('Nonfinite or incompatible frozen native text conditions')
    return conditions, {'native_max_length': 77, 'unpadded_token_count': full.shape[1],
        'padded_token_count': 77, 'padding_token_count': 77-full.shape[1], 'all_input_ids': padded[0].tolist(),
        'full_unpadded_input_ids': full[0].tolist(), 'truncation': False, 'complete_prefix_identity': True,
        'prompt_word_count': len(prompt.split())}


def local_component_manifest(sd):
    """Pin actual local fp16 variants; explicit loader mode prevents other fallback."""
    files = {}; modes = {}
    for component in ('unet', 'vae', 'text_encoder'):
        folder = sd/component; base = 'model' if component == 'text_encoder' else 'diffusion_pytorch_model'
        safe = folder/(base+'.fp16.safetensors')
        binary = folder/(('pytorch_model' if component == 'text_encoder' else base)+'.fp16.bin')
        indices = sorted(path for path in folder.glob('*.json')
                         if all(word in path.name for word in ('fp16', 'safetensors', 'index')))
        if safe.is_file():
            chosen = [safe]; modes[component] = True
        elif len(indices) == 1:
            index = indices[0]; metadata = json.loads(index.read_text(encoding='utf-8'))
            shards = sorted(set(metadata['weight_map'].values()))
            chosen = [index]+[folder/name for name in shards]; modes[component] = True
            if not shards or any(path.parent.resolve() != folder.resolve() for path in chosen):
                raise ValueError('Unexpected local fp16 component index paths')
        elif binary.is_file():
            chosen = [binary]; modes[component] = False
        else:
            raise ValueError('Missing or ambiguous official local fp16 component weights: '+component)
        chosen.append(folder/'config.json')
        for path in chosen:
            if not path.is_file():
                raise ValueError('Missing selected local model component: '+str(path))
            files[path.relative_to(sd).as_posix()] = {'sha256': sha256(path), 'bytes': path.stat().st_size}
    tokenizer = sd/'tokenizer'
    for name in ('vocab.json', 'merges.txt', 'tokenizer_config.json', 'special_tokens_map.json',
                 'added_tokens.json', 'tokenizer.json', 'chat_template.jinja'):
        path = tokenizer/name
        if name in ('vocab.json', 'merges.txt', 'tokenizer_config.json') and not path.is_file():
            raise ValueError('Missing native local CLIP tokenizer file: '+name)
        if path.is_file():
            files[path.relative_to(sd).as_posix()] = {'sha256': sha256(path), 'bytes': path.stat().st_size}
    path = sd/'scheduler/scheduler_config.json'
    files[path.relative_to(sd).as_posix()] = {'sha256': sha256(path), 'bytes': path.stat().st_size}
    return {'variant': 'fp16', 'runtime_dtype': 'bfloat16', 'local_files_only': True,
            'tokenizer_scope': 'Native vocabulary/merges/config plus any local optional tokenizer assets potentially consumed by this installed tokenizer version',
            'use_safetensors_by_component': modes, 'files': files}


def panels(output, records):
    result = []
    for category in ('lamp', 'pillow'):
        for level in ('single', 'composed'):
            for resolution in (256, 512):
                width = 2*resolution; row_height = resolution+100
                sheet = Image.new('RGB', (width, 24+2*row_height), 'white'); draw = ImageDraw.Draw(sheet)
                draw.text((4, 3), f'TEXT ONLY CONTROL | {category}/{level} | native{resolution} | manual review pending', fill='black')
                for number, color in enumerate(('red', 'blue')):
                    chosen = [r for r in records if (r['category'], r['prompt_level'], r['resolution'], r['color']) == (category, level, resolution, color)]
                    top = 24+number*row_height
                    draw.multiline_text((4, top), '\n'.join(textwrap.wrap(chosen[0]['prompt'], max(40, (width-8)//6))), fill='black')
                    for col, row in enumerate(chosen):
                        draw.text((col*resolution+4, top+75), 'seed '+str(row['noise_seed']), fill='black')
                        with Image.open(output/row['generated_PNG']) as image:
                            sheet.paste(image, (col*resolution, top+100))
                path = output/'panels'/f'{category}_{level}_{resolution}.png'; path.parent.mkdir(exist_ok=True); sheet.save(path)
                result.append({'path': path.relative_to(output).as_posix(), 'sha256': sha256(path)})
    return result


def run(args, cfg):
    manifest = local_component_manifest(args.sd); write_json(args.output/'model_components.json', manifest)
    device = torch.device('cuda:0'); torch.cuda.set_per_process_memory_fraction(cfg['gpu_memory_fraction'], 0)
    from diffusers import AutoencoderKL, UNet2DConditionModel, EulerDiscreteScheduler
    from transformers import CLIPTextModel, CLIPTokenizer
    modes = manifest['use_safetensors_by_component']
    common = {'variant': 'fp16', 'torch_dtype': torch.bfloat16, 'local_files_only': True}
    unet = UNet2DConditionModel.from_pretrained(args.sd, subfolder='unet', use_safetensors=modes['unet'], **common).to(device).eval().requires_grad_(False)
    vae = AutoencoderKL.from_pretrained(args.sd, subfolder='vae', use_safetensors=modes['vae'], **common).to(device).eval().requires_grad_(False)
    vae.encoder = None; vae.quant_conv = None
    text = CLIPTextModel.from_pretrained(args.sd, subfolder='text_encoder', use_safetensors=modes['text_encoder'], **common).to(device).eval().requires_grad_(False)
    tokenizer = CLIPTokenizer.from_pretrained(args.sd, subfolder='tokenizer', local_files_only=True)
    scheduler = EulerDiscreteScheduler.from_pretrained(args.sd, subfolder='scheduler', local_files_only=True)
    model = NativeTurboHead(unet, vae, scheduler, torch.nn.Identity()).eval().requires_grad_(False)
    counts = {'UNet': sum(p.numel() for p in unet.parameters()), 'VAE_decoder_only': sum(p.numel() for p in vae.parameters()),
              'native_CLIP_text_encoder': sum(p.numel() for p in text.parameters())}
    counts['inference_total'] = sum(counts.values())
    if any(p.requires_grad for module in (model, text) for p in module.parameters()):
        raise ValueError('Frozen inference only; gradients or optimizer are forbidden')
    records = control_records(); conditions = {}; audits = {}
    with torch.inference_mode():
        for number, row in enumerate(records, 1):
            if row['prompt_id'] not in conditions:
                conditions[row['prompt_id']], audits[row['prompt_id']] = native_text_conditions(tokenizer, text, row['prompt'], device)
            resolution = row['resolution']
            noise = torch.randn(1, 4, resolution//8, resolution//8, device=device,
                                generator=torch.Generator(device=device).manual_seed(row['noise_seed']))
            _, image = model.generate_from_conditions(conditions[row['prompt_id']], noise)
            if image.shape != (1, 3, resolution, resolution) or not torch.isfinite(image).all():
                raise FloatingPointError('Native resolution or finite RGB contract failed')
            path = args.output/'images'/(row['image_id']+'.png'); path.parent.mkdir(exist_ok=True); tensor_image(image[0]).save(path)
            row.update(generated_PNG=path.relative_to(args.output).as_posix(), generated_PNG_sha256=sha256(path),
                initial_noise_shape=list(noise.shape), initial_noise_dtype=str(noise.dtype),
                initial_noise_sha256=hashlib.sha256(noise.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest(),
                raw_input_token_audit=audits[row['prompt_id']],
                manual_review={'requested_attribute_match': {key: None for key in row['requested_attributes']},
                    'image_quality': None, 'unwanted_artifacts': None, 'comments': None})
            write_json(args.output/'images.json', records[:number])
            write_json(args.output/'progress.json', {'stage': 'native_text_only_prior_control', 'done': number, 'total': 32})
    previews = panels(args.output, records)
    write_json(args.output/'report.json', {'schema': SCHEMA, 'complete': True, 'quality_accepted': False,
        'evaluation_label': 'native_text_only_prior_diagnostic_not_reference_guided_editor', 'generated_count': len(records),
        'actual_native_UNet_and_VAE_forward_calls': len(records), 'actual_native_text_encoding_calls': len(conditions),
        'source_commit': os.environ['T12_SOURCE_COMMIT'], 'source_sha256': sha256(Path(__file__)), 'config': cfg,
        'parameters': counts, 'model_components': manifest, 'scheduler_class': type(scheduler).__name__,
        'scheduler_prediction_type': scheduler.config.prediction_type, 'inference_steps': 1, 'guidance_scale': 0,
        'classifier_free_guidance_used': False, 'NativeTurboHead_graph_used': True, 'images': records, 'panels': previews,
        'generator_input_whitelist': ['complete native CLIP caption', 'fixed random noise1042/1043'],
        'Qwen_or_reference_image_loaded': False, 'dataset_GT_mask_cache_or_ABO_read': False, 'VAE_encoder_removed': True,
        'training_performed': False, 'TRAIN_assets_created': False, 'independent_TEST540_opened': False,
        'cross_resolution_noise_policy': 'The same numeric seed does not mean the same noise tensor at256 and512. Shapes, dtype and actual tensor hashes are recorded; no pixel-aligned resolution causality is claimed.',
        'resizing_or_sharpening': False, 'timing_benchmark_performed': False, 'manual_review_pending': True,
        'RGB_export': 'Unchanged tensor_image clamp[-1,1], scale127.5 and rounduint8 only; no resizing or other enhancement',
        'limitations': 'Text-only original pretrained prior control on a different task and two native resolutions, not a reference-guided baseline or a same-model quality comparison. Native256 and512 are generated separately, never resized. No identity conditioning, unique-GT similarity metric, training asset, performance result, or2x claim is supplied. All requested-attribute and quality fields await manual review; completion alone is not successful color/composition generation.'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'sd', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--gpu-uuid', required=True)
    args = parser.parse_args(); cfg = json.loads(args.config.read_text(encoding='utf-8')); commit = os.environ.get('T12_SOURCE_COMMIT', '')
    if (cfg.get('schema') != SCHEMA or cfg.get('gpu_uuid') != GPU_UUID or args.gpu_uuid != GPU_UUID or
            os.environ.get('CUDA_VISIBLE_DEVICES') != GPU_UUID or args.output.exists() or cfg.get('seeds') != [1042, 1043] or
            cfg.get('native_resolutions') != [256, 512] or len(commit) != 40 or any(c not in '0123456789abcdef' for c in commit)):
        raise ValueError('Reviewed immutable32-image control, published commit and authorized GPU6 required')
    args.output.mkdir(parents=True)
    write_json(args.output/'execution.json', {'argv': sys.argv, 'pid': os.getpid(), 'started_unix': time.time(),
        'source_commit': commit, 'source_sha256': sha256(Path(__file__)), 'config_sha256': sha256(args.config),
        'config': cfg, 'torch': torch.__version__, 'cuda': torch.version.cuda, 'gpu_uuid': args.gpu_uuid})
    torch.set_num_threads(4)
    try:
        run(args, cfg)
    except Exception:
        write_json(args.output/'failure.json', {'complete': False, 'time': time.time(), 'traceback': traceback.format_exc()})
        raise
    finally:
        gc.collect()
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
