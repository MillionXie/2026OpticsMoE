"""Frozen non-Qwen scene-generation feasibility probes from white ABO inputs.

Prepare reads only four TRAIN source photographs/renders, never paired targets.
Inference receives reference RGB and full text. Generated backgrounds are model
outputs, not retrieved room photographs or automatically accepted training GT.
"""
import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import textwrap
import time
import traceback

from PIL import Image, ImageDraw

GPU = 'GPU-3f60d773-4dd8-d706-ef48-a4dfc5af6c55'
SCHEMA = 'abo_white_reference_scene_feasibility_v1'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def conditions(category, profile='directional_v1'):
    common = {'reference_constraint': 'retain coarse product outline; fine appearance may change',
              'background_source': 'generated entire scene; no external room photograph'}
    if category == 'lamp':
        prompts = [
            ('Generate a realistic modern study around this lamp, retaining its outline. Use a deep blue fabric shade and brass base. Place it on an oak desk beside books and a small plant, with shelves and curtains behind it. Warm window light enters from the right.',
             {'shade_color': 'deep blue', 'shade_material': 'fabric', 'base_finish': 'brass', 'scene': 'modern study', 'support': 'oak desk', 'background_elements': ['books', 'plant', 'shelves', 'curtains'], 'light_side': 'right'}),
            ('Generate a realistic reading corner around this lamp, retaining its outline. Use a terracotta red shade and matte black base. Place it on a pale stone side table beside an armchair, with bookshelves and framed abstract shapes behind it. Soft daylight enters from the left.',
             {'shade_color': 'terracotta red', 'base_finish': 'matte black', 'scene': 'reading corner', 'support': 'pale stone side table', 'background_elements': ['armchair', 'bookshelves', 'abstract shapes'], 'light_side': 'left'})]
    elif category == 'pillow':
        prompts = [
            ('Generate a realistic living room around this cushion, retaining its outline. Give it deep blue woven fabric and thin cream piping. Place it on a beige sofa, with an oak coffee table, curtains and a leafy plant behind it. Soft window light enters from the right.',
             {'fabric_color': 'deep blue', 'fabric_texture': 'woven', 'piping': 'thin cream', 'scene': 'living room', 'support': 'beige sofa', 'background_elements': ['oak coffee table', 'curtains', 'leafy plant'], 'light_side': 'right'}),
            ('Generate a realistic reading nook around this cushion, retaining its outline. Give it terracotta fabric with broad cream stripes. Place it on a grey armchair beside a stone side table, with bookshelves and a leafy plant behind it. Warm daylight enters from the left.',
             {'fabric_color': 'terracotta', 'fabric_pattern': 'broad cream stripes', 'scene': 'reading nook', 'support': 'grey armchair', 'background_elements': ['stone side table', 'bookshelves', 'leafy plant'], 'light_side': 'left'})]
    else:
        raise ValueError('Only reviewed lamp/pillow subjects')
    if profile not in ['directional_v1', 'free_lighting_v2']:
        raise ValueError('Unreviewed scene prompt profile')
    result = [{'condition_id': str(i), 'prompt': prompt, 'requested_attributes': attrs, **common}
              for i, (prompt, attrs) in enumerate(prompts)]
    if profile == 'free_lighting_v2':
        for row in result:
            row['prompt'] = row['prompt'].rsplit('. ', 1)[0]+'.'
            row['requested_attributes'].pop('light_side')
    return result


def choose_sources(rows, indices):
    if len(indices) != 4 or len(set(indices)) != 4:
        raise ValueError('Four distinct prespecified TRAIN source rows required')
    chosen = [rows[i] for i in indices]
    if (sorted(r['category'] for r in chosen) != ['lamp', 'lamp', 'pillow', 'pillow'] or
            any(r['split'] != 'train' or r['source']['category'] != r['category'] for r in chosen) or
            len({r['source']['product_id'] for r in chosen}) != 4):
        raise ValueError('TRAIN-only two lamps/two pillows with distinct product identities')
    return chosen


def native_caption(category, condition, style, profile='directional_v1'):
    """Explicit object-attribute wording, with no target-derived information."""
    if style == 'original': return condition['prompt']
    if style != 'explicit_caption': raise ValueError('Unreviewed prompt translation')
    captions = {
        'lamp': [
            'Generate an interior product photograph of one table lamp with a deep blue fabric lampshade and brass base on an oak desk. Books and a small plant sit beside the lamp. Bookshelves and curtains furnish a modern study behind it. Warm window light enters from the right. Retain the reference lamp outline.',
            'Generate an interior product photograph of one table lamp with a terracotta red lampshade and matte black base on a pale stone side table. An armchair, bookshelves and framed abstract shapes furnish the reading corner behind the lamp. Soft daylight enters from the left. Retain the reference lamp outline.'],
        'pillow': [
            'Generate an interior product photograph of one deep blue woven cushion with thin cream piping on a beige sofa. An oak coffee table, curtains and a leafy plant furnish the living room behind the cushion. Soft window light enters from the right. Retain the reference cushion outline.',
            'Generate an interior product photograph of one terracotta cushion with broad cream stripes on a grey armchair beside a stone side table. Bookshelves and a leafy plant furnish the reading nook behind the cushion. Warm daylight enters from the left. Retain the reference cushion outline.']}
    if condition not in conditions(category, profile): raise ValueError('Canonical requested attributes required')
    caption = captions[category][int(condition['condition_id'])]
    if profile == 'free_lighting_v2':
        # Keep the historical directional_v1 bytes exactly reproducible.
        caption = '. '.join(s for s in caption.split('. ') if not any(
            s.startswith(prefix) for prefix in ['Warm window light', 'Soft daylight', 'Soft window light', 'Warm daylight']))
    return caption


def checked_tokens(tokenizer, prompt):
    ids = tokenizer(prompt, truncation=False, padding=False)['input_ids']
    if len(ids) > tokenizer.model_max_length:
        raise ValueError(f'Full prompt exceeds native text encoder: {len(ids)} tokens; no truncation permitted')
    return {'full_token_count': len(ids), 'maximum': tokenizer.model_max_length,
            'truncation': False, 'input_ids': ids}


def model_manifest(root):
    files = sorted(p for p in root.rglob('*') if p.is_file() and '.cache' not in p.parts
                   and p.suffix in {'.safetensors', '.json', '.txt', '.md'})
    if not files or any(p.suffix == '.incomplete' for p in root.rglob('*')):
        raise ValueError('Complete local model assets required')
    return {p.relative_to(root).as_posix(): {'sha256': digest(p), 'bytes': p.stat().st_size} for p in files}


def prepare(args, cfg):
    rows = json.loads((args.data/'annotated_pairs.json').read_text(encoding='utf-8'))
    audit = json.loads((args.data/'data_audit.json').read_text(encoding='utf-8'))
    if (digest(args.data/'annotated_pairs.json') != cfg['source_pair_sha256'] or
            audit.get('role_disjoint') is not True or audit.get('test_opened') is not False):
        raise ValueError('Original source selection/split audit changed')
    refs = []
    for i, row in enumerate(choose_sources(rows, cfg['source_row_indices'])):
        source = row['source']; original = Path(source['image_path'])
        if source['license'] != 'CC BY 4.0':
            raise ValueError('Reviewed ABO source license required')
        with Image.open(original) as image:
            image = image.convert('RGB')
            if image.size != (256, 256):
                raise ValueError('Existing native256 cleanrender source required for pilot')
            if any(min(image.getpixel(corner)) < 245 for corner in [(0, 0), (255, 0), (0, 255), (255, 255)]):
                raise ValueError('Source corners are not blank white')
            path = args.output/f'reference_{i:02d}.png'; image.save(path)
        refs.append({'reference_id': f'reference_{i:02d}', 'split': 'train', 'category': row['category'],
                     'input_PNG': path.name, 'input_sha256': digest(path), 'source_image_sha256': digest(original),
                     'source_sample_id': source['sample_id'], 'source_product_id': source['product_id'],
                     'source_member': source['source_member'], 'source_original_path': str(original),
                     'attribution': 'Amazon Berkeley Objects (ABO), CC BY 4.0; white-composited/resized render; baseline outputs are AI generated.',
                     'license_url': 'https://creativecommons.org/licenses/by/4.0/',
                     'source_url': source['source_url']})
    sheet = Image.new('RGB', (1024, 290), 'white'); draw = ImageDraw.Draw(sheet)
    for i, row in enumerate(refs):
        draw.text((i*256+3, 3), row['reference_id']+' | TRAIN white ABO input', fill='black')
        with Image.open(args.output/row['input_PNG']) as image: sheet.paste(image, (i*256, 34))
    sheet.save(args.output/'references_preview.png')
    write(args.output/'reference_manifest.json', {'schema': SCHEMA, 'references': refs,
          'source_pair_sha256': cfg['source_pair_sha256'], 'paired_target_or_mask_pixels_read': False,
          'independent_TEST_opened': False, 'conditions': {c: conditions(c) for c in ['lamp', 'pillow']}})
    write(args.output/'report.json', {'complete': True, 'scope': 'CPU white reference selection only; not generated examples',
          'reference_manifest_sha256': digest(args.output/'reference_manifest.json'), 'reference_count': 4})


def render_panels(output, records, inputs):
    paths = []
    for ref in inputs['references']:
        selected = [r for r in records if r['reference_id'] == ref['reference_id']]
        resolution = selected[0]['resolution']; width = 2*resolution+256; stride = resolution+110
        sheet = Image.new('RGB', (width, 28+stride*2), 'white'); draw = ImageDraw.Draw(sheet)
        draw.text((4, 4), f'{selected[0]["model"]} | {ref["reference_id"]} | AI-generated outputs; feasibility pilot, not accepted training GT', fill='black')
        for condition in range(2):
            subset = [r for r in selected if r['condition_id'] == str(condition)]; y = 28+condition*stride
            draw.multiline_text((4, y), '\n'.join(textwrap.wrap(subset[0]['prompt'], max(70, (width-10)//6))), fill='black')
            if subset[0].get('model_prompt', subset[0]['prompt']) != subset[0]['prompt']:
                draw.multiline_text((4, y+34), '\n'.join(textwrap.wrap('MODEL INPUT: '+subset[0]['model_prompt'], max(70, (width-10)//6))), fill='black')
            with Image.open(output/ref['input_PNG']) as im: sheet.paste(im, (0, y+110))
            for j, row in enumerate(subset):
                draw.text((256+j*resolution+4, y+90), 'seed '+str(row['seed']), fill='black')
                with Image.open(output/row['generated_PNG']) as im: sheet.paste(im, (256+j*resolution, y+110))
        p = output/(ref['reference_id']+'_full_prompts.png'); sheet.save(p); paths.append({'path': p.name, 'sha256': digest(p)})
    return paths


def infer(args, cfg):
    import torch
    from diffusers import StableDiffusionInstructPix2PixPipeline, StableDiffusionXLPipeline
    from transformers import CLIPVisionModelWithProjection
    inputs = json.loads((args.inputs/'reference_manifest.json').read_text(encoding='utf-8'))
    if inputs['schema'] != SCHEMA or any(r['split'] != 'train' for r in inputs['references']):
        raise ValueError('Only prepared feasibility TRAIN references; not independent TEST')
    assets = {'base': model_manifest(args.model_path)}
    if args.model == 'sdxl_ipadapter': assets['ip_adapter'] = model_manifest(args.adapter_path)
    write(args.output/'model_assets.json', assets)
    torch.set_num_threads(4); torch.cuda.set_per_process_memory_fraction(.8, 0)
    if args.model == 'ip2p':
        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(args.model_path, torch_dtype=torch.float16, local_files_only=True).to('cuda')
    else:
        encoder = CLIPVisionModelWithProjection.from_pretrained(args.adapter_path/'models/image_encoder', torch_dtype=torch.float16, local_files_only=True)
        pipe = StableDiffusionXLPipeline.from_pretrained(args.model_path, variant='fp16', torch_dtype=torch.float16,
                    image_encoder=encoder, local_files_only=True).to('cuda')
        pipe.load_ip_adapter(str(args.adapter_path), subfolder='sdxl_models', weight_name='ip-adapter-plus_sdxl_vit-h.safetensors',
                             image_encoder_folder=None, local_files_only=True)
        pipe.set_ip_adapter_scale(cfg['sdxl_ipadapter']['image_scale'])
    pipe.set_progress_bar_config(disable=True)
    for component in pipe.components.values():
        if isinstance(component, torch.nn.Module): component.eval().requires_grad_(False)
    count = sum(p.numel() for p in {id(p): p for c in pipe.components.values() if isinstance(c, torch.nn.Module) for p in c.parameters()}.values())
    records = []; preset = cfg[args.model]; resolution = preset['resolution']
    for ref in inputs['references']:
        path = args.inputs/ref['input_PNG']
        if digest(path) != ref['input_sha256']: raise ValueError('Reference RGB changed')
        shutil.copyfile(path, args.output/ref['input_PNG'])
        with Image.open(path) as im: reference = im.convert('RGB').copy()
        profile = cfg.get('prompt_profile', 'directional_v1')
        for condition in conditions(ref['category'], profile):
            prompt = native_caption(ref['category'], condition, preset.get('prompt_style', 'original'), profile)
            token_audit = [checked_tokens(pipe.tokenizer, prompt)]
            if args.model == 'sdxl_ipadapter': token_audit.append(checked_tokens(pipe.tokenizer_2, prompt))
            for seed in cfg['seeds']:
                began = time.time(); kwargs = dict(prompt=prompt, negative_prompt=cfg['negative_prompt'],
                     num_inference_steps=preset['steps'], guidance_scale=preset['text_guidance'],
                     generator=torch.Generator(device='cuda').manual_seed(seed))
                if args.model == 'ip2p':
                    kwargs.update(image=reference.resize((resolution, resolution), Image.Resampling.LANCZOS), image_guidance_scale=preset['image_guidance'])
                else: kwargs.update(ip_adapter_image=reference, width=resolution, height=resolution)
                with torch.inference_mode(): result = pipe(**kwargs).images[0]
                if result.size != (resolution, resolution): raise ValueError('Unexpected native output size')
                name = f'{ref["reference_id"]}_condition{condition["condition_id"]}_seed{seed}.png'; result.save(args.output/name)
                records.append({'model': args.model, 'reference_id': ref['reference_id'], 'condition_id': condition['condition_id'],
                     'category': ref['category'], 'prompt': condition['prompt'], 'model_prompt': prompt,
                     'requested_attributes': condition['requested_attributes'],
                     'seed': seed, 'resolution': resolution, 'preset': preset, 'full_prompt_tokens': token_audit,
                     'input_sha256': ref['input_sha256'], 'generated_PNG': name, 'generated_sha256': digest(args.output/name),
                     'diagnostic_generation_seconds_not_latency_benchmark': time.time()-began,
                     'manual_review': {'coarse_shape': None, 'attribute_matches': {k: None for k in condition['requested_attributes']}, 'quality': None}})
                write(args.output/'images.json', records); write(args.output/'progress.json', {'done': len(records), 'total': 16})
    panels = render_panels(args.output, records, inputs)
    write(args.output/'report.json', {'complete': True, 'schema': SCHEMA, 'model': args.model, 'source_commit': os.environ['T12_SOURCE_COMMIT'],
          'generator_input_whitelist': ['reference RGB', 'complete prompt', 'random noise'], 'no_Qwen_used': True,
          'prompt_profile': cfg.get('prompt_profile', 'directional_v1'),
          'parameters_in_loaded_pipeline': count, 'reference_manifest_sha256': digest(args.inputs/'reference_manifest.json'),
          'model_assets_sha256': digest(args.output/'model_assets.json'), 'generated_count': len(records),
          'input_resolution': 256, 'inference_resolution': resolution, 'input_resizing': 'LANCZOS256to512 for IP2P only; SDXL uses native CLIP image preprocessing',
          'training_performed': False, 'quality_accepted': False, 'manual_review_pending': True, 'panels': panels,
          'independent_TEST_opened': False, 'GT_mask_retrieval_paste_or_external_background_used': False,
          'limitations': 'Four TRAIN references and16 outputs/model are feasibility examples, not independent performance or automatic training labels. Same integer seeds across different pipelines/resolutions do not imply identical latent noise. No claims of copyright clearance, Nature acceptance, attribute success or2x speed. Generated scenes and object details require visual review.'})
    del pipe; gc.collect(); torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('stage', choices=['prepare', 'infer'])
    for n in ['config', 'data', 'inputs', 'model-path', 'adapter-path', 'output']: p.add_argument('--'+n, type=Path, required=n in ['config', 'output'])
    p.add_argument('--model', choices=['ip2p', 'sdxl_ipadapter']); args = p.parse_args()
    cfg = json.loads(args.config.read_text(encoding='utf-8')); commit = os.environ.get('T12_SOURCE_COMMIT', '')
    if cfg['schema'] != SCHEMA or args.output.exists() or len(commit) != 40: raise ValueError('Published commit/new run/profile required')
    if args.stage == 'infer' and (os.environ.get('CUDA_VISIBLE_DEVICES') != GPU or not args.model or not args.inputs or not args.model_path):
        raise ValueError('Authorized GPU6, model and prepared reference inputs required')
    args.output.mkdir(parents=True)
    write(args.output/'execution.json', {'argv': sys.argv, 'pid': os.getpid(), 'source_commit': commit, 'source_sha256': digest(Path(__file__)),
          'config': cfg, 'config_sha256': digest(args.config), 'versions': {n: importlib.metadata.version(n) for n in ['torch', 'diffusers', 'transformers']}})
    try:
        prepare(args, cfg) if args.stage == 'prepare' else infer(args, cfg)
    except Exception:
        write(args.output/'failure.json', {'traceback': traceback.format_exc()}); raise


if __name__ == '__main__': main()
