"""Frozen, fully adapted Qwen fused features for TRAIN readout acceleration.

This is not a deployment shortcut or an LG intermediate-feature protocol.
Every cached state comes from the complete selected native vision+language
LoRA model on the actual reference RGB and supplied instruction.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from PIL import Image
import torch

from .qwen_vl_ordered_input import ORDER
from .qwen_vl_turbo_cache import save_with_digest
from .qwen_vl_turbo_data import sha256, validate_roles, write_json


CACHE_SCHEMA = 'qwen_frozen_joint_fused_features_v1'
TEACHER_SCHEMA = 'qwen_spatial_joint_lora_v1'
DATA_PROTOCOL = 'controlled_room_edit_v1'
COUNTS = {'train': 960, 'val': 96}


def frozen_frontend_checkpoint_sha256(parent_state: dict, immediate_head_sha256: str) -> str:
    """Keep frontend identity stable when only the frozen-Qwen head continues.

    The immediate head checkpoint is separately recorded in every output. Its
    adapted Qwen tensors are still loaded and independently content-digested.
    """
    value = parent_state['config'].get('frozen_joint_teacher_checkpoint_sha256', immediate_head_sha256)
    if not isinstance(value, str) or not value:
        raise ValueError('Frozen frontend checkpoint identity must be nonempty')
    return value


def joint_adapter_digest(state_dict: dict[str, torch.Tensor]) -> str:
    """Bind names, shapes, dtypes and exact bytes, independent of dict order."""
    if not state_dict or any(not isinstance(k, str) or not isinstance(v, torch.Tensor)
                             for k, v in state_dict.items()):
        raise ValueError('A nonempty native joint-adapter tensor mapping is required')
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        value = state_dict[name].detach().cpu().contiguous()
        if not torch.isfinite(value.float()).all():
            raise ValueError('Nonfinite native adapter tensor')
        header = json.dumps([name, str(value.dtype), list(value.shape)],
                            separators=(',', ':')).encode('utf-8')
        raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
        digest.update(len(header).to_bytes(8, 'big')); digest.update(header)
        digest.update(len(raw).to_bytes(8, 'big')); digest.update(raw)
    return digest.hexdigest()


def _rgb_bytes(reference) -> tuple[bytes, list[int]]:
    if isinstance(reference, Image.Image):
        reference = np.asarray(reference.convert('RGB')).copy()
        return reference.tobytes(), list(reference.shape)
    if not isinstance(reference, torch.Tensor) or reference.dtype != torch.uint8:
        raise ValueError('Fingerprint requires exact uint8 reference RGB')
    if reference.ndim != 3 or reference.shape[0] != 3:
        raise ValueError('Fingerprint requires a CHW RGB image')
    value = reference.detach().cpu().permute(1, 2, 0).contiguous()
    return value.numpy().tobytes(), list(value.shape)


def input_fingerprint(reference, prompt: str, order: str = ORDER) -> str:
    """Only forward inputs: never target, masks, caption, SKU or random seed."""
    if order != ORDER or not isinstance(prompt, str) or not prompt.strip():
        raise ValueError('Reviewed native input order and nonempty prompt required')
    raw, shape = _rgb_bytes(reference)
    header = json.dumps({'RGB_shape': shape, 'prompt': prompt,
                         'frontend_input_order': order}, sort_keys=True,
                        ensure_ascii=False, separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(header + b'\0' + raw).hexdigest()


def frontend_input_fingerprint(payload: dict, index: int) -> str:
    return input_fingerprint(payload['reference'][index],
                             payload['rows'][index]['prompt'],
                             payload['frontend_input_order'])


def _target_fingerprint(target: torch.Tensor) -> str:
    raw, shape = _rgb_bytes(target)
    return hashlib.sha256(json.dumps(shape).encode('ascii') + b'\0' + raw).hexdigest()


def validate_frozen_joint_cache(payload: dict, parent_state: dict,
                                parent_checkpoint_sha256: str) -> dict:
    """Validate against canonical frontend SHA, not a later head-only SHA."""
    meta = payload.get('frozen_joint_teacher', {})
    if parent_state.get('schema') != TEACHER_SCHEMA:
        raise ValueError('Frozen joint cache needs the complete joint-adapted teacher')
    if meta.get('schema') != CACHE_SCHEMA:
        raise ValueError('Missing frozen joint fused-feature provenance')
    if meta.get('parent_checkpoint_sha256') != parent_checkpoint_sha256:
        raise ValueError('Frozen frontend parent checkpoint differs')
    actual_adapter = joint_adapter_digest(parent_state['qwen_joint_lora'])
    if meta.get('joint_adapter_sha256') != actual_adapter:
        raise ValueError('Frozen frontend native vision/language adapters differ')
    if (payload.get('frontend_input_order') != ORDER or
            meta.get('frontend_input_order') != ORDER or
            parent_state['config'].get('native_input_order') != ORDER or
            parent_state['visual_layout'].get('native_input_order') != ORDER):
        raise ValueError('Frozen frontend input order differs')
    if (meta.get('all_qwen_frozen') is not True or
            meta.get('feature_scope') != 'final_fused_head_training_only' or
            meta.get('LG_intermediate_cache') is not False or
            meta.get('independent_TEST_opened') is not False):
        raise ValueError('Unreviewed frozen-cache scope')
    rows = payload['rows']
    audit = validate_roles(rows)
    if audit['pairs'] != COUNTS or len({r['sample_id'] for r in rows}) != sum(COUNTS.values()):
        raise ValueError('Controlled-room TRAIN/DEV identities or budget differ')
    if any(r.get('render_protocol') != DATA_PROTOCOL for r in rows):
        raise ValueError('Controlled-room data protocol required')
    reference, target = payload['reference'], payload['target']
    expected_shape = (len(rows), 3, 256, 256)
    if (reference.shape != expected_shape or target.shape != expected_shape or
            reference.dtype != torch.uint8 or target.dtype != torch.uint8):
        raise ValueError('Exact native uint8 reference and supervision images required')
    if meta.get('sample_ids') != [r['sample_id'] for r in rows]:
        raise ValueError('Frozen feature row order differs')
    fingerprints = [frontend_input_fingerprint(payload, i) for i in range(len(rows))]
    if fingerprints != meta.get('input_fingerprints'):
        raise ValueError('Frozen features do not match actual RGB, text and input order')
    if [_target_fingerprint(v) for v in target] != meta.get('target_fingerprints'):
        raise ValueError('Cached training supervision changed independently of prepared data')
    mask_digests = [row.get('edit_mask_sha256') for row in rows]
    if not all(mask_digests) or mask_digests != meta.get('edit_mask_sha256'):
        raise ValueError('Cached edit-mask supervision identity differs')
    for row, expected in zip(rows, mask_digests):
        path = Path(row.get('edit_mask_path', ''))
        if not path.is_file() or sha256(path) != expected:
            raise ValueError('Prepared edit-mask supervision changed')
    hidden = payload['hidden']
    if len(hidden) != len(rows) or any(v.dtype != torch.bfloat16 or v.ndim != 2 or
            v.shape[1] != 2048 or not 64 <= len(v) <= 224 or not torch.isfinite(v.float()).all()
            for v in hidden):
        raise ValueError('Actual complete-Qwen fused BF16 states required')
    train_indices = {i for i, row in enumerate(rows) if row['split'] == 'train'}
    if set(payload['train_clip_teacher']) != train_indices:
        raise ValueError('Native CLIP text targets must cover TRAIN only')
    checks = payload['qwen_checks']
    if (checks.get('vision_blocks') != 24 or checks.get('language_blocks') != 28 or
            checks.get('all_qwen_frozen') is not True or
            meta.get('raw_hidden_recompute_passed') is not True or
            meta.get('raw_hidden_recompute_cells') != 6):
        raise ValueError('Full raw frontend/cache identity audit missing')
    return meta


def _read_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def load_prepared_images(data: Path):
    """Check every prepared PNG, not a renderer approximation or old hidden."""
    report = _read_json(data/'report.json')
    rows = _read_json(data/'annotated_pairs.json')
    audit = _read_json(data/'data_audit.json')
    if (not report.get('complete') or report.get('pair_sha256') != sha256(data/'annotated_pairs.json') or
            audit.get('protocol') != DATA_PROTOCOL or audit.get('role_disjoint') is not True or
            audit.get('test_opened') is not False or audit.get('expected_counts') != COUNTS or
            validate_roles(rows)['pairs'] != COUNTS):
        raise ValueError('Prepared controlled-room TRAIN/DEV audit differs')
    if not (data/'profile.json').is_file() or sha256(data/'profile.json') != report['profile_sha256']:
        raise ValueError('Prepared controlled-room profile bytes differ')
    expected = {f"images/{row['split']}/{role}/{i:05d}.png"
                for i, row in enumerate(rows) for role in ('reference', 'target', 'edit_mask')}
    manifest = _read_json(data/'image_sha256.json')
    if set(manifest) != expected:
        raise ValueError('Prepared PNG manifest has missing or additional images')
    references, targets = [], []
    for i, row in enumerate(rows):
        for role, destination in (('reference', references), ('target', targets)):
            key = f"images/{row['split']}/{role}/{i:05d}.png"
            path = data/key
            if sha256(path) != manifest[key]:
                raise ValueError('Prepared PNG bytes changed: '+key)
            with Image.open(path) as image:
                if image.mode != 'RGB' or image.size != (256, 256):
                    raise ValueError('Prepared images must be native RGB256')
                destination.append(torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).contiguous())
        mask_key = f"images/{row['split']}/edit_mask/{i:05d}.png"
        mask_path = data/mask_key
        if (Path(row.get('edit_mask_path', '')).resolve() != mask_path.resolve() or
                row.get('edit_mask_sha256') != manifest[mask_key] or
                sha256(mask_path) != manifest[mask_key]):
            raise ValueError('Prepared edit-mask path or bytes differ')
        with Image.open(mask_path) as mask:
            if mask.mode != 'L' or mask.size != (256, 256):
                raise ValueError('Prepared supervision masks must be native L256')
    return rows, torch.stack(references), torch.stack(targets), report, audit


def _encode(qwen, processor, reference: torch.Tensor, prompt: str,
            layout: dict, device) -> torch.Tensor:
    from .qwen_vl_ordered_input import ordered_inputs
    image = Image.fromarray(reference.permute(1, 2, 0).numpy())
    inputs, permutation, canonical = ordered_inputs(processor, image, prompt)
    positions = torch.where(canonical['input_ids'][0] == layout['image_token_id'])[0].tolist()
    if (positions != layout['positions'] or len(positions) != 64 or
            inputs['input_ids'].shape[1] > 224 or
            canonical['image_grid_thw'].tolist() != [[1, 16, 16]]):
        raise ValueError('Actual native grid or canonical readout differs')
    with torch.no_grad():
        hidden = qwen.model(**inputs.to(device), use_cache=False, return_dict=True).last_hidden_state[0]
    return hidden[permutation.to(device)].detach().bfloat16().cpu().contiguous()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('data', 'parent', 'qwen', 'sd', 'output'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--gpu-uuid', required=True)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != args.gpu_uuid or args.output.exists():
        raise ValueError('Authorized single GPU and immutable new output required')
    from .qwen_vl_hybrid_run import completed_teacher, load_qwen
    from .qwen_vl_joint_adapt import install_joint_lora
    parent_report = completed_teacher(args.parent)
    workflow = _read_json(args.parent/'workflow_report.json')
    if workflow.get('checkpoint_sha256') != parent_report['checkpoint_sha256']:
        raise ValueError('Completed parent workflow and selected checkpoint identity differ')
    state = torch.load(args.parent/'best_checkpoint.pt', map_location='cpu', weights_only=False)
    if (state.get('schema') != TEACHER_SCHEMA or
            state.get('step') != parent_report['best_step'] or
            state['config'].get('native_input_order') != ORDER or
            state['visual_layout'].get('native_input_order') != ORDER or
            state['visual_layout'].get('semantic_prefix', 0)):
        raise ValueError('Use the completed full joint-adapted text-first teacher')
    if state['visual_layout']['qwen_config_sha256'] != sha256(args.qwen/'config.json'):
        raise ValueError('Public Qwen config differs from the selected teacher')
    frontend_sha = frozen_frontend_checkpoint_sha256(state, parent_report['checkpoint_sha256'])
    identities = {'parent_checkpoint_sha256': frontend_sha,
                  'immediate_head_parent_checkpoint_sha256': parent_report['checkpoint_sha256']}
    rows, reference, target, prepared, audit = load_prepared_images(args.data)
    args.output.mkdir(parents=True)
    for path in args.data.glob('*.json'):
        if path.name == 'report.json':
            shutil.copyfile(path, args.output/'prepared_data_report.json')
        elif path.name == 'execution.json':
            shutil.copyfile(path, args.output/'prepared_data_execution.json')
        elif path.name not in ('cache_complete.json', 'cache_execution.json', 'download_receipt.json'):
            shutil.copyfile(path, args.output/path.name)
    write_json(args.output/'execution.json', {
        'argv': sys.argv, 'pid': os.getpid(), 'source_commit': os.environ.get('T12_SOURCE_COMMIT'),
        'source_sha256': sha256(Path(__file__)), 'started_unix': time.time(), 'gpu_uuid': args.gpu_uuid,
        **identities,
        'prepared_data_directory': str(args.data.resolve()),
        'prepared_pair_sha256': prepared['pair_sha256'], 'torch': torch.__version__,
        'cuda': torch.version.cuda, 'Qwen_forward_inputs': 'reference RGB and supplied prompt only'})
    torch.set_num_threads(4); torch.manual_seed(1042)
    device = torch.device('cuda:0'); torch.cuda.set_per_process_memory_fraction(.75, 0)
    qwen, processor = load_qwen(args.qwen, device)
    base_forward = sum(p.numel() for p in qwen.model.parameters())
    loaded_base = sum(p.numel() for p in qwen.parameters())
    embedding = qwen.model.language_model.embed_tokens.weight.numel()
    adapter_parameters = sum(v.numel() for v in state['qwen_joint_lora'].values())
    install_joint_lora(qwen, state['config'], state['qwen_joint_lora'], train=False)
    if (any(p.requires_grad for p in qwen.parameters()) or len(qwen.model.visual.blocks) != 24 or
            len(qwen.model.language_model.layers) != 28 or
            sum(p.numel() for p in qwen.model.parameters())-base_forward != adapter_parameters):
        raise ValueError('Full native Qwen with only the selected frozen joint adapters required')
    layout = state['visual_layout']; hidden = []
    for i, row in enumerate(rows):
        hidden.append(_encode(qwen, processor, reference[i], row['prompt'], layout, device))
        if (i+1) % 20 == 0 or i+1 == len(rows):
            write_json(args.output/'progress.json', {'stage': 'actual_frozen_joint_Qwen', 'done': i+1, 'total': len(rows)})
            print('frozen_joint_features', i+1, '/', len(rows), flush=True)
    cells = {}; raw_checks = []
    for i, row in enumerate(rows):
        cell = row['category'], row['mode']
        if row['split'] != 'val' or cell in cells:
            continue
        cells[cell] = i
        actual = _encode(qwen, processor, reference[i], row['prompt'], layout, device)
        difference = float((actual.float()-hidden[i].float()).abs().max())
        raw_checks.append({'sample_id': row['sample_id'], 'max_abs_hidden': difference})
        if difference > 1e-3:
            raise ValueError('Actual frozen joint Qwen/cache recomputation differs')
    if set(cells) != {(category, mode) for category in ('lamp', 'pillow') for mode in ('background', 'object', 'joint')}:
        raise ValueError('Need six distinct category/task recomputation cells')
    first = next(i for i, row in enumerate(rows) if row['split'] == 'train')
    blank = _encode(qwen, processor, torch.full_like(reference[first], 127), rows[first]['prompt'], layout, device)
    changed = _encode(qwen, processor, reference[first], 'Keep the product and change only the wall color to violet.', layout, device)
    base = hidden[first].float().mean(0)
    controls = {
        'blank_reference_relative_change': float((blank.float().mean(0)-base).norm()/base.norm().clamp_min(1e-6)),
        'changed_instruction_relative_change': float((changed.float().mean(0)-base).norm()/base.norm().clamp_min(1e-6)),
        'vision_blocks': 24, 'language_blocks': 28, 'all_qwen_frozen': True,
        'qwen_loaded_parameters': loaded_base, 'qwen_forward_parameters': base_forward,
        'qwen_embedding_parameters': embedding, 'joint_lora_parameters': adapter_parameters,
        'qwen_forward_including_joint_lora': base_forward+adapter_parameters, 'no_token_or_layer_pruning': True}
    if not all(np.isfinite(controls[key]) and controls[key] > 1e-5
               for key in ('blank_reference_relative_change', 'changed_instruction_relative_change')):
        raise ValueError('Actual fused Qwen features do not respond to both input modalities')
    write_json(args.output/'qwen_modality_checks.json', controls)
    write_json(args.output/'raw_cache_hidden_identity.json', {'passed': True, 'checks': raw_checks,
        'all_qwen_frozen': True, **identities})
    del qwen, processor; gc.collect(); torch.cuda.empty_cache()
    from transformers import CLIPTextModel, CLIPTokenizer
    clip = CLIPTextModel.from_pretrained(args.sd, subfolder='text_encoder', variant='fp16',
            torch_dtype=torch.bfloat16, local_files_only=True).to(device).eval().requires_grad_(False)
    tokenizer = CLIPTokenizer.from_pretrained(args.sd, subfolder='tokenizer', local_files_only=True)
    indices = [i for i, row in enumerate(rows) if row['split'] == 'train']; teachers = {}; truncated = 0
    with torch.no_grad():
        for start in range(0, len(indices), 16):
            current = indices[start:start+16]; captions = [rows[i]['teacher_caption'] for i in current]
            truncated += sum(len(v) > 77 for v in tokenizer(captions, truncation=False)['input_ids'])
            tokens = tokenizer(captions, padding='max_length', max_length=77, truncation=True,
                               return_tensors='pt').input_ids.to(device)
            encoded = clip(tokens).last_hidden_state.detach().bfloat16().cpu()
            teachers.update({i: v.contiguous() for i, v in zip(current, encoded)})
    del clip; gc.collect(); torch.cuda.empty_cache()
    payload = {'rows': rows, 'reference': reference, 'target': target, 'hidden': hidden,
        'train_clip_teacher': teachers, 'frontend_input_order': ORDER, 'qwen_checks': controls,
        'pilot_sha256': sha256(args.output/'pilot_pairs.json'),
        'clip_scope': 'Frozen native SD-Turbo CLIP text teacher on TRAIN captions only; no image inputs'}
    meta = {'schema': CACHE_SCHEMA, **identities,
        'joint_adapter_sha256': joint_adapter_digest(state['qwen_joint_lora']),
        'frontend_input_order': ORDER, 'all_qwen_frozen': True,
        'feature_scope': 'final_fused_head_training_only', 'LG_intermediate_cache': False,
        'independent_TEST_opened': False, 'sample_ids': [row['sample_id'] for row in rows],
        'input_fingerprints': [frontend_input_fingerprint(payload, i) for i in range(len(rows))],
        'target_fingerprints': [_target_fingerprint(v) for v in target],
        'edit_mask_sha256': [row['edit_mask_sha256'] for row in rows],
        'prepared_pair_sha256': prepared['pair_sha256'],
        'prepared_image_manifest_sha256': sha256(args.data/'image_sha256.json'),
        'raw_hidden_recompute_passed': True, 'raw_hidden_recompute_cells': len(raw_checks),
        'deployment_requires_actual_full_Qwen_and_selected_joint_adapters': True,
        'TRAIN_native_CLIP_captions_truncated_to_77': truncated}
    payload['frozen_joint_teacher'] = meta
    validate_frozen_joint_cache(payload, state, frontend_sha)
    write_json(args.output/'frozen_joint_teacher.json', meta)
    path = args.output/'frozen_features.pt'; digest = save_with_digest(payload, path)
    write_json(args.output/'cache_complete.json', {'complete': True, 'sha256': digest, 'samples': len(rows),
        'source_commit': os.environ.get('T12_SOURCE_COMMIT'), **identities,
        'joint_adapter_sha256': meta['joint_adapter_sha256'], 'all_qwen_frozen': True})
    write_json(args.output/'report.json', {'complete': True, 'stage': 'frozen_selected_joint_Qwen_final_fused_cache',
        'pair_sha256': prepared['pair_sha256'], 'profile_sha256': prepared['profile_sha256'],
        'source_commit': os.environ.get('T12_SOURCE_COMMIT'), 'source_sha256': sha256(Path(__file__)),
        'cache_sha256': digest, **identities,
        'joint_adapter_sha256': meta['joint_adapter_sha256'], 'audit': audit,
        'independent_TEST_opened': False, 'LG_intermediate_cache': False,
        'TRAIN_acceleration_only_deployment_is_full_raw_Qwen': True})


if __name__ == '__main__':
    main()
