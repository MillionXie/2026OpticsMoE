"""Zero-shot free-scene pilot: eight references, two prompts, four seeds.

This diagnostic retains the reference subject's overall shape and requests
four scene attributes. Other detail is free: no unique paired target, target
pixels, mask, training cache, CLIP scorer or reference VAE encoder is used.
TRAIN references are previously seen; DEV references were used for model
selection. Neither cohort is an independent generalization evaluation.
"""
import argparse
from collections import Counter
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import textwrap
import time
import traceback

import numpy as np
from PIL import Image, ImageDraw
import torch

from .qwen_vl_hybrid_run import completed_teacher, load_qwen, raw_hidden
from .qwen_vl_joint_adapt import install_joint_lora
from .qwen_vl_joint_feature_cache import input_fingerprint, joint_adapter_digest
from .qwen_vl_ordered_input import ORDER, ordered_inputs
from .qwen_vl_postfree import instruction_bounds
from .qwen_vl_spatial import build_spatial_head
from .qwen_vl_turbo_data import sha256, write_json
from .qwen_vl_turbo_model import pack_hidden
from .qwen_vl_turbo_run import noise_for, tensor_image


SCHEMA = 'qwen_free_scene_zero_shot_pilot_v1'
GPU_UUID = 'GPU-3f60d773-4dd8-d706-ef48-a4dfc5af6c55'
PARENT_SHA = 'f69020e34b34ee23f1072caf435acd0833bf95afcb60daa63f91485fe7fbdb00'
COUNTS = {'train': 960, 'val': 96}


def prompt_conditions(category):
    """Shared category prompts: four explicit edits, no requested shape change."""
    if category not in ('lamp', 'pillow'):
        raise ValueError('Only the reviewed lamp/pillow categories are allowed')
    result = []
    for number, (color, wall, direction, surface) in enumerate((
            ('red', 'pale blue', 'right', 'wooden'),
            ('blue', 'warm beige', 'left', 'neutral stone'))):
        subject = color+' lampshade' if category == 'lamp' else color+' fabric'
        prompt = (f'Generate a new product scene using the reference {category}. '
            f'Retain its overall silhouette, with {"a " if category == "lamp" else ""}{subject}, '
            f'a {wall} wall, light from the {direction}, and a {surface} tabletop. '
            'Other textures, small details, composition, and background objects may vary naturally.')
        result.append({'condition_id': f'condition_{number}', 'prompt': prompt,
            'requested_attributes': {('lampshade_color' if category == 'lamp' else 'fabric_color'): color,
                'wall_color': wall, 'side_light_direction': direction, 'tabletop_material': surface},
            'shape_anchor': 'Retain the reference subject overall silhouette',
            'free_details': ['textures', 'small details', 'composition', 'background objects']})
    return result


def select_references(rows):
    """Metadata-only, deterministic first-view/context selection, no score filter."""
    if (dict(Counter(row['split'] for row in rows)) != COUNTS or
            any(row.get('render_protocol') != 'controlled_room_edit_v1' for row in rows)):
        raise ValueError('Require the original completed controlled-room TRAIN960/DEV96 metadata')
    selected = []
    for split in ('train', 'val'):
        for category in ('lamp', 'pillow'):
            products = set()
            for i, row in enumerate(rows):
                if row['split'] != split or row['category'] != category:
                    continue
                product = row['source']['product_id']
                if product in products:
                    continue
                products.add(product)
                selected.append({'reference_id': f'ref_{len(selected):02d}', 'metadata_row_index': i,
                    'split': split, 'category': category, 'reference_context_id': row['reference_context_id']})
                if len(products) == 2:
                    break
            if len(products) != 2:
                raise ValueError('Need two distinct reference products in each category/split')
    if len(selected) != 8 or len({r['reference_context_id'] for r in selected}) != 8:
        raise ValueError('Eight unique reference contexts required')
    return selected


def reference_seeds(reference_RGB, base_seed=1042, count=4):
    """Common seed0 controls reference use; seeds1–3 depend only on input RGB."""
    if isinstance(reference_RGB, Image.Image):
        value = np.asarray(reference_RGB.convert('RGB'))
    elif isinstance(reference_RGB, torch.Tensor):
        if reference_RGB.dtype != torch.uint8 or tuple(reference_RGB.shape) != (3, 256, 256):
            raise ValueError('Seed derivation requires exact native uint8 RGB')
        value = reference_RGB.cpu().permute(1, 2, 0).contiguous().numpy()
    else:
        value = np.asarray(reference_RGB)
    if value.shape != (256, 256, 3) or value.dtype != np.uint8 or count != 4:
        raise ValueError('Seed derivation requires native RGB256 and exactly four variants')
    seeds = [base_seed]+[(base_seed+int.from_bytes(hashlib.sha256(value.tobytes()+index.to_bytes(4, 'big')).digest()[:8], 'big'))
             % (2**31-1) for index in range(1, count)]
    if len(set(seeds)) != count:
        raise ValueError('Unexpected hash collision in reviewed seed sequence')
    return seeds


def generation_inputs(reference_RGB, prompt):
    """The complete raw-Qwen whitelist; no label/metadata argument is accepted."""
    if (not isinstance(reference_RGB, torch.Tensor) or reference_RGB.dtype != torch.uint8 or
            tuple(reference_RGB.shape) != (3, 256, 256) or not isinstance(prompt, str) or not prompt.strip()):
        raise ValueError('Require native uint8 reference RGB and the complete nonempty prompt')
    return {'reference': reference_RGB[None], 'rows': [{'prompt': prompt}]}


def load_reference_inputs(data):
    """Open metadata and eight prepared REFERENCE PNGs only; never target/mask pixels."""
    report = json.loads((data/'report.json').read_text(encoding='utf-8'))
    audit = json.loads((data/'data_audit.json').read_text(encoding='utf-8'))
    rows = json.loads((data/'annotated_pairs.json').read_text(encoding='utf-8'))
    manifest = json.loads((data/'image_sha256.json').read_text(encoding='utf-8'))
    if (not report.get('complete') or report.get('stage') != 'CPU_data_preparation_only' or
            report['pair_sha256'] != sha256(data/'annotated_pairs.json') or
            audit.get('protocol') != 'controlled_room_edit_v1' or audit.get('expected_counts') != COUNTS or
            audit.get('role_disjoint') is not True or audit.get('test_opened') is not False):
        raise ValueError('Prepared original TRAIN/DEV metadata identity differs')
    selected = select_references(rows); images = []; rgb_hashes = set()
    for record in selected:
        relative = f"images/{record['split']}/reference/{record['metadata_row_index']:05d}.png"
        path = data/relative
        if manifest.get(relative) != sha256(path):
            raise ValueError('Prepared reference PNG bytes changed')
        with Image.open(path) as image:
            if image.mode != 'RGB' or image.size != (256, 256):
                raise ValueError('Native readable RGB256 reference required')
            value = np.asarray(image).copy()
        rgb_sha = hashlib.sha256(value.tobytes()).hexdigest()
        if rgb_sha in rgb_hashes:
            raise ValueError('Do not duplicate identical reference RGB to inflate the pilot')
        rgb_hashes.add(rgb_sha); images.append(torch.from_numpy(value).permute(2, 0, 1).contiguous())
        record.update(prepared_reference_path=relative, prepared_reference_PNG_sha256=manifest[relative],
            reference_RGB_sha256=rgb_sha, seeds=reference_seeds(value),
            cohort_label='TRAIN_seen_reference' if record['split'] == 'train' else 'DEV_used_for_model_selection')
    return selected, images, {'prepared_pair_sha256': sha256(data/'annotated_pairs.json'),
        'prepared_report_sha256': sha256(data/'report.json'), 'prepared_audit_sha256': sha256(data/'data_audit.json'),
        'prepared_image_manifest_sha256': sha256(data/'image_sha256.json'),
        'reference_PNGs_read': 8, 'target_or_mask_PNGs_read': 0,
        'frozen_feature_cache_read': False, 'independent_TEST540_opened': False}


def audit_tokens(processor, image, prompt, layout):
    """Verify canonical helper, complete prompt and no processor truncation."""
    native, permutation, canonical = ordered_inputs(processor, image, prompt)
    canonical_ids = canonical['input_ids'][0]; native_ids = native['input_ids'][0]
    positions = torch.where(canonical_ids == layout['image_token_id'])[0].tolist()
    span = instruction_bounds(canonical_ids, processor.tokenizer.convert_tokens_to_ids('<|vision_end|>'),
                              processor.tokenizer.convert_tokens_to_ids('<|im_end|>'))
    if (positions != layout['positions'] or len(positions) != 64 or list(span) != list(layout['instruction_bounds']) or
            len(canonical_ids) > 224 or not torch.equal(native_ids[permutation], canonical_ids)):
        raise ValueError('New full prompt differs from the reviewed raw/canonical interface')
    # Match untruncated tokenizer output to each actual image-expanded input.
    for order, actual in (('image', canonical_ids), ('text', native_ids)):
        content = [{'type': 'image', 'image': image}, {'type': 'text', 'text': prompt}]
        if order == 'text':
            content.reverse()
        rendered = processor.apply_chat_template([{'role': 'user', 'content': content}], tokenize=False,
                        add_generation_prompt=True, enable_thinking=False)
        expanded = rendered.replace(processor.image_token, processor.image_token*64)
        full = processor.tokenizer(expanded, return_tensors='pt', truncation=False)['input_ids'][0]
        if not torch.equal(full, actual):
            raise ValueError('Actual processor omitted or changed full prompt tokens')
    return {'native_token_count': len(native_ids), 'canonical_token_count': len(canonical_ids),
        'image_token_count': 64, 'instruction_token_count': len(canonical_ids)-span[0]-span[1],
        'prompt_word_count': len(prompt.split()), 'prompt_character_count': len(prompt),
        'complete_untruncated_tokenizer_identity': True, 'native_canonical_token_multiset_equal': True,
        'input_order': ORDER, 'canonical_readout_permutation_sha256': hashlib.sha256(permutation.numpy().tobytes()).hexdigest(),
        'native_input_ids_sha256': hashlib.sha256(native_ids.numpy().tobytes()).hexdigest()}


def save_panel(output, reference, reference_image, records):
    panel = Image.new('RGB', (1280, 772), 'white'); draw = ImageDraw.Draw(panel)
    draw.text((6, 4), reference['reference_id']+' | '+reference['category']+' | '+reference['cohort_label']+
              ' | ZERO-SHOT PILOT, manual review pending; no unique GT', fill='black')
    for condition_index in range(2):
        selected = [row for row in records if row['condition_index'] == condition_index]
        top = 26+condition_index*370
        draw.multiline_text((6, top), '\n'.join(textwrap.wrap(selected[0]['prompt'], 170)), fill='black')
        draw.text((6, top+72), 'REFERENCE: overall shape anchor', fill='black')
        panel.paste(reference_image, (0, top+104))
        for number, row in enumerate(selected):
            draw.text(((number+1)*256+4, top+72), 'seed '+str(row['noise_seed']), fill='black')
            with Image.open(output/row['generated_PNG']) as generated:
                panel.paste(generated, ((number+1)*256, top+104))
    folder = output/'panels'; folder.mkdir(exist_ok=True)
    path = folder/(reference['reference_id']+'.png'); panel.save(path)
    return str(path.relative_to(output)), sha256(path)


def run(args, cfg):
    parent = completed_teacher(args.parent)
    if parent['checkpoint_sha256'] != PARENT_SHA or cfg['expected_parent_checkpoint_sha256'] != PARENT_SHA:
        raise ValueError('Use the completed, unchanged STANDARD checkpoint')
    state = torch.load(args.parent/'best_checkpoint.pt', map_location='cpu', weights_only=False)
    layout = state['visual_layout']
    if (state.get('schema') != 'qwen_spatial_joint_lora_v1' or state['step'] != parent['best_step'] or
            layout.get('native_input_order') != ORDER or layout.get('semantic_prefix', 0) or
            state['config'].get('native_input_order') != ORDER or
            layout['qwen_config_sha256'] != sha256(args.qwen/'config.json')):
        raise ValueError('Complete selected native joint teacher/config/order identity required')
    references, images, provenance = load_reference_inputs(args.data)
    for record in references:
        record['seeds'] = reference_seeds(images[int(record['reference_id'][-2:])], cfg['seed'])
    write_json(args.output/'reference_manifest.json', {'references': references, 'data_provenance': provenance,
        'shared_category_conditions': {category: prompt_conditions(category) for category in ('lamp', 'pillow')}})
    device = torch.device('cuda:0'); torch.cuda.set_per_process_memory_fraction(cfg['gpu_memory_fraction'], 0)
    qwen, processor = load_qwen(args.qwen, device)
    install_joint_lora(qwen, state['config'], state['qwen_joint_lora'], train=False)
    if (len(qwen.model.visual.blocks), len(qwen.model.language_model.layers)) != (24, 28):
        raise ValueError('Full native Qwen24V/28L required')
    head = build_spatial_head(args.sd, state, state['config'], layout['positions'], device, drop_vae_encoder=True)
    head.eval().requires_grad_(False)
    if any(p.requires_grad for module in (qwen, head) for p in module.parameters()) or head.vae.encoder is not None or head.vae.quant_conv is not None:
        raise ValueError('No gradients, VAE encoder or unfrozen inference parameter permitted')
    counts = {'qwen_full_forward_with_joint_adapters': sum(p.numel() for p in qwen.model.parameters()),
        'adapter': sum(p.numel() for p in head.adapter.parameters()), 'unet_including_lora': sum(p.numel() for p in head.unet.parameters()),
        'spatial_readout': sum(p.numel() for p in head.spatial.parameters()), 'vae_decoder_only': sum(p.numel() for p in head.vae.parameters())}
    counts['actual_forward_inference_total'] = sum(counts.values())
    counts['completed_parent_inference_total_unmerged'] = parent['parameters']['inference_total']
    if counts['actual_forward_inference_total'] != counts['completed_parent_inference_total_unmerged']:
        raise ValueError('Frozen decoder-only inference parameter count differs from completed parent')
    records = []; panels = []; calls = 0
    for reference, image in zip(references, images):
        path = args.output/'references'/(reference['reference_id']+'.png'); path.parent.mkdir(exist_ok=True)
        reference_image = tensor_image(image)
        shutil.copyfile(args.data/reference['prepared_reference_path'], path)
        if sha256(path) != reference['prepared_reference_PNG_sha256']:
            raise ValueError('Exported reference PNG bytes differ from prepared reference')
        current = []
        for condition_index, condition in enumerate(prompt_conditions(reference['category'])):
            prompt = condition['prompt']; tokens = audit_tokens(processor, reference_image, prompt, layout)
            for seed_index, seed in enumerate(reference['seeds']):
                # This whitelist is the only payload read by the raw frontend.
                inputs = generation_inputs(image, prompt)
                with torch.inference_mode():
                    hidden = raw_hidden(qwen, processor, inputs, 0, layout, device); calls += 1
                    if not torch.isfinite(hidden.float()).all():
                        raise FloatingPointError('Nonfinite actual full-Qwen RGB/text states')
                    h, valid = pack_hidden([hidden], device)
                    _, rgb = head(h, valid, noise_for([{'noise_seed': seed}], device))
                if not torch.isfinite(rgb).all():
                    raise FloatingPointError('Nonfinite generated RGB')
                identifier = f"{reference['reference_id']}_condition{condition_index}_seed{seed_index}"
                generated = args.output/'generated'/(identifier+'.png'); generated.parent.mkdir(exist_ok=True)
                tensor_image(rgb[0]).save(generated)
                record = {'image_id': identifier, 'reference_id': reference['reference_id'], 'split': reference['split'],
                    'category': reference['category'], 'reference_context_id': reference['reference_context_id'],
                    'reference_PNG': str(path.relative_to(args.output)), 'reference_PNG_sha256': sha256(path),
                    'reference_RGB_sha256': reference['reference_RGB_sha256'], 'condition_index': condition_index,
                    'prompt': prompt, 'noise_seed': seed, 'seed_variant_index': seed_index,
                    'requested_attributes': condition['requested_attributes'], 'shape_anchor': condition['shape_anchor'],
                    'free_details': condition['free_details'], 'raw_input_token_audit': tokens,
                    'input_fingerprint': input_fingerprint(image, prompt),
                    'generator_input_keys': ['reference RGB', 'supplied prompt', 'random noise seed'],
                    'generated_PNG': str(generated.relative_to(args.output)), 'generated_PNG_sha256': sha256(generated),
                    'manual_review': {'overall_shape_retained': None, 'requested_attribute_match':
                        {key: None for key in condition['requested_attributes']}, 'image_quality': None,
                        'unwanted_artifacts': None, 'reference_use': None, 'comments': None}}
                current.append(record); records.append(record)
                write_json(args.output/'images.json', records)
                write_json(args.output/'progress.json', {'stage': 'actual_zero_shot_RGB_text_generation', 'done': calls, 'total': 64})
        panel, digest = save_panel(args.output, reference, reference_image, current)
        panels.append({'path': panel, 'sha256': digest})
        write_json(args.output/'images.json', records)
    if calls != 64 or len(records) != 64:
        raise ValueError('The reviewed eight-reference/two-prompt/four-seed pilot must contain64 actual forwards')
    write_json(args.output/'report.json', {'schema': SCHEMA, 'complete': True,
        'evaluation_label': 'zero_shot_free_scene_TRAIN_and_model_selection_DEV_diagnostic',
        'quality_accepted': False, 'manual_review_pending': True, 'training_performed': False,
        'independent_TEST540_opened': False, 'reference_count': 8, 'generated_count': 64,
        'actual_complete_Qwen_forward_calls': calls, 'vision_blocks': 24, 'language_blocks': 28,
        'model_checkpoint_sha256': parent['checkpoint_sha256'], 'joint_adapter_sha256': joint_adapter_digest(state['qwen_joint_lora']),
        'source_commit': os.environ.get('T12_SOURCE_COMMIT'), 'source_sha256': sha256(Path(__file__)),
        'config': cfg, 'parameters': counts, 'data_provenance': provenance, 'panels': panels, 'images': records,
        'generator_input_whitelist': ['reference RGB', 'full supplied instruction', 'common-fixed or source-RGB-derived random noise'],
        'no_GT_mask_caption_SKU_or_CLIP_inference_input': True, 'VAE_encoder_removed': True,
        'no_frozen_feature_cache_read': True, 'timing_benchmark_performed': False,
        'seed_protocol': 'Variant0 is common seed1042 across all references/prompts. Variants1–3 are seed1042 plus first64bits SHA256(native reference RGB bytes + uint32bigendian seed variant index), modulo2^31-1. Each reference shares its four seeds across both prompts; no target/SKU-derived seed.',
        'reference_use_control': 'Compare same category/prompt at common seed variant0 across the four references. Source-dependent noise variants1–3 cannot isolate reference influence and must not be used alone as evidence of reference use.',
        'limitations': 'Untrained new Generate prompts with four requested attributes are a zero-shot diagnostic of this completed parent, not a new accepted baseline or LG result. Subject overall shape anchors reference use; other detail is free. No unique GT resemblance or pixel-similarity score is used. TRAIN references were seen during training and DEV was used for prior model selection. All attribute/quality/reference-use fields require manual review; variation or image differences alone do not establish successful generation.'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'parent', 'data', 'qwen', 'sd', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--gpu-uuid', required=True)
    args = parser.parse_args(); cfg = json.loads(args.config.read_text(encoding='utf-8'))
    commit = os.environ.get('T12_SOURCE_COMMIT', '')
    if (cfg.get('schema') != SCHEMA or args.gpu_uuid != GPU_UUID or cfg.get('gpu_uuid') != GPU_UUID or
            os.environ.get('CUDA_VISIBLE_DEVICES') != GPU_UUID or args.output.exists() or
            cfg.get('seed') != 1042 or cfg.get('reference_count') != 8 or cfg.get('prompts_per_reference') != 2 or
            cfg.get('seeds_per_prompt') != 4 or len(commit) != 40 or any(c not in '0123456789abcdef' for c in commit)):
        raise ValueError('Reviewed immutable64-image protocol, published source commit and authorized GPU6 required')
    args.output.mkdir(parents=True)
    write_json(args.output/'execution.json', {'argv': sys.argv, 'pid': os.getpid(), 'started_unix': time.time(),
        'source_commit': os.environ.get('T12_SOURCE_COMMIT'), 'source_sha256': sha256(Path(__file__)),
        'config_sha256': sha256(args.config), 'config': cfg, 'torch': torch.__version__, 'cuda': torch.version.cuda,
        'gpu_uuid': args.gpu_uuid, 'model_input': 'Reference RGB, full new generic Generate instruction, source-derived noise only'})
    torch.set_num_threads(4); torch.manual_seed(cfg['seed'])
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
