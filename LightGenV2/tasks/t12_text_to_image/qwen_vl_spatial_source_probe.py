"""TRAIN-only comparison of two existing Qwen spatial readout sources.

The semantic path always uses the original complete fused language states.
Native merger states replace only the spatial readout's visual-token input.
This is a small fit diagnostic, not a new baseline, LG teacher or deployment
claim. Targets, masks and VAE encodings are TRAIN loss/metric labels only.
"""
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import traceback

from PIL import Image, ImageDraw
import torch
from torch.nn import functional as F

from .qwen_vl_hybrid_run import completed_teacher, load_data, load_qwen, raw_hidden
from .qwen_vl_joint_adapt import install_joint_lora
from .qwen_vl_joint_feature_cache import (
    input_fingerprint, joint_adapter_digest, validate_frozen_joint_cache)
from .qwen_vl_scene_review import review_ids
from .qwen_vl_spatial import build_spatial_head, weighted_edit_regression
from .qwen_vl_spatial_run import masks
from .qwen_vl_turbo_cache import image_float, save_with_digest
from .qwen_vl_turbo_data import sha256, write_json
from .qwen_vl_turbo_losses import crop_regions, region_boxes
from .qwen_vl_turbo_model import pack_hidden
from .qwen_vl_turbo_run import noise_for, tensor_image


VARIANTS = ('lm_visual64', 'native_merger64')
SCHEMA = 'qwen_spatial_source_TRAIN_probe_v1'
GPU_UUID = 'GPU-3f60d773-4dd8-d706-ef48-a4dfc5af6c55'


def native_argument(variant, native):
    if variant not in VARIANTS:
        raise ValueError('Unknown controlled spatial source')
    return native if variant == 'native_merger64' else None


def batch_orders(count, cfg):
    if count < max(cfg['warm_batch_size'], cfg['batch_size']):
        raise ValueError('Probe batch cannot exceed its selected TRAIN cohort')
    rng = random.Random(cfg['seed'])
    warm = [rng.sample(range(count), cfg['warm_batch_size']) for _ in range(cfg['warm_steps'])]
    order = []
    while len(order) < cfg['paired_steps']*cfg['batch_size']:
        indices = list(range(count)); rng.shuffle(indices); order.extend(indices)
    paired = [order[i*cfg['batch_size']:(i+1)*cfg['batch_size']] for i in range(cfg['paired_steps'])]
    return warm, paired


def checked_native(output):
    if (not isinstance(output, torch.Tensor) or output.shape != (64, 2048) or
            output.dtype != torch.bfloat16 or not torch.isfinite(output.float()).all()):
        raise ValueError('Actual main visual.merger must return finite BF16 [64,2048]')
    return output.detach().cpu().contiguous()


def capture_raw(qwen, processor, inputs, index, layout, device):
    captured = []
    handle = qwen.model.visual.merger.register_forward_hook(
        lambda module, args, output: captured.append(checked_native(output)))
    try:
        with torch.no_grad():
            hidden = raw_hidden(qwen, processor, inputs, index, layout, device).detach().cpu().contiguous()
    finally:
        handle.remove()
    if len(captured) != 1 or not torch.isfinite(hidden.float()).all():
        raise ValueError('Require one main-merger hook and finite actual fused states per raw forward')
    return hidden, captured[0], len(captured)


def gradient_sum(parameters):
    values = [value.grad for value in parameters if value.grad is not None]
    if not values or any(not torch.isfinite(value).all() for value in values):
        raise FloatingPointError('Missing or nonfinite real TRAIN gradient')
    total = sum(float(value.detach().abs().sum()) for value in values)
    if total <= 0 or not math.isfinite(total):
        raise FloatingPointError('Expected a nonzero finite real TRAIN gradient')
    return total


def build_head(sd, state, device):
    head = build_spatial_head(sd, state, state['config'], state['visual_layout']['positions'], device)
    head.requires_grad_(False)
    head.spatial.requires_grad_(True); head.unet.conv_in.spatial.requires_grad_(True)
    head.unet.enable_gradient_checkpointing()
    allowed = ('spatial.', 'unet.conv_in.spatial.')
    if any(value.requires_grad and not name.startswith(allowed) for name, value in head.named_parameters()):
        raise ValueError('Unexpected trainable parameter outside existing spatial bridge/injection')
    return head


def initial_digests(head):
    return {'spatial': joint_adapter_digest(head.spatial.state_dict()),
            'spatial_injection': joint_adapter_digest(head.unet.conv_in.spatial.state_dict()),
            'semantic_adapter': joint_adapter_digest(head.adapter.state_dict())}


def first_cells(rows):
    result = {}
    for i, row in enumerate(rows):
        result.setdefault((row['category'], row['mode']), i)
    if set(result) != {(category, mode) for category in ('lamp', 'pillow') for mode in ('background', 'object', 'joint')}:
        raise ValueError('Six TRAIN category/task cells required')
    return list(result.values())


def counterfactual_contract(data, manifest):
    """Bind the exact selected complete pairs, not an ID-string heuristic."""
    indexed = {row['sample_id']: i for i, row in enumerate(data['rows'])}
    covered = []; groups = []; contexts = {}
    for group in manifest:
        sample_ids = group['sample_ids']
        if not set(sample_ids) & set(indexed):
            continue
        if len(sample_ids) != 2 or len(set(sample_ids)) != 2 or not set(sample_ids) <= set(indexed):
            raise ValueError('Selected TRAIN contexts must contain both counterfactual rows')
        a, b = [indexed[sid] for sid in sample_ids]
        rows = [data['rows'][a], data['rows'][b]]
        key = group['reference_context_id'], group['mode']
        if (key in groups or any(row['split'] != 'train' or
                (row['reference_context_id'], row['mode']) != key for row in rows) or
                len({row['category'] for row in rows}) != 1 or
                len({row['noise_seed'] for row in rows}) != 1 or
                len({row['prompt'] for row in rows}) != 2 or
                group['prompts'] != [row['prompt'] for row in rows] or
                not torch.equal(data['reference'][a], data['reference'][b]) or
                torch.equal(data['target'][a], data['target'][b])):
            raise ValueError('TRAIN counterfactual RGB/text/seed/target contract differs')
        context_value = (rows[0]['noise_seed'], data['reference'][a])
        if key[0] in contexts:
            previous = contexts[key[0]]
            if previous[0] != context_value[0] or not torch.equal(previous[1], context_value[1]):
                raise ValueError('A source context must retain identical RGB and noise across tasks')
        contexts[key[0]] = context_value
        groups.append(key); covered.extend(sample_ids)
    if (len(groups) != 30 or len(covered) != len(indexed) or len(set(covered)) != len(indexed)
            or set(covered) != set(indexed)):
        raise ValueError('Require the exact complete 30-pair/60-row TRAIN probe')
    return {'pairs': len(groups), 'rows': len(indexed), 'reference_contexts': len(contexts),
            'same_context_RGB_and_noise': True, 'distinct_paired_prompts_and_targets': True}


def make_inputs(hidden, native, indices, variant, device):
    h, valid = pack_hidden([hidden[i] for i in indices], device)
    field = torch.stack([native[i] for i in indices]).to(device)
    return h, valid, native_argument(variant, field)


@torch.no_grad()
def conditions_for(head, hidden, selected, device):
    head.eval()
    return {i: head.semantic_conditions(*pack_hidden([hidden[i]], device)).detach().cpu() for i in selected}


def check_conditions(head, hidden, expected, device):
    actual = conditions_for(head, hidden, list(expected), device)
    differences = {str(i): float((actual[i].float()-expected[i].float()).abs().max()) for i in expected}
    if max(differences.values()) > 1e-6:
        raise ValueError('The frozen original semantic conditions changed')
    return differences


@torch.no_grad()
def evaluate(head, variant, data, hidden, native, supervision, metric, device, folder, label, export=False):
    head.eval(); records = []; indices = first_cells(data['rows'])
    panel = Image.new('RGB', (768, 310*len(indices)), 'white'); draw = ImageDraw.Draw(panel)
    for i, row in enumerate(data['rows']):
        h, valid, field = make_inputs(hidden, native, [i], variant, device)
        _, prediction = head(h, valid, noise_for([row], device), native_visual=field)
        prediction = prediction.float().clamp(-1, 1)
        reference = image_float(data['reference'][i:i+1], device)
        target = image_float(data['target'][i:i+1], device)
        error = (prediction-target).square()/4
        box = supervision['boxes'][i:i+1].to(device)
        crop = metric(crop_regions(prediction, box), crop_regions(target, box)).mean()
        edit = supervision['edit'][i].to(device); keep = 1-edit
        keep_error = ((prediction-reference).abs()/2*keep).sum()/(3*keep.sum().clamp_min(1))
        record = {'sample_id': row['sample_id'], 'prompt': row['prompt'], 'category': row['category'],
            'mode': row['mode'], 'noise_seed': row['noise_seed'],
            'PSNR': -10*math.log10(max(float(error.mean()), 1e-12)),
            'LPIPS': float(metric(prediction, target).mean()), 'crop_LPIPS': float(crop),
            'retained_L1_RGB_01': float(keep_error), 'object_crop_box_01': box[0].cpu().tolist()}
        records.append(record)
        if export:
            record['PNG_sha256'] = {}
            for role, value in (('reference', data['reference'][i]), ('target', data['target'][i]), ('generated', prediction[0])):
                path = folder/'images'/label/role/f'sample_{i:04d}.png'; path.parent.mkdir(parents=True, exist_ok=True)
                tensor_image(value).save(path)
                record['PNG_sha256'][role] = sha256(path)
        if i in indices:
            top = indices.index(i)*310
            draw.text((4, top+2), 'TRAIN FIT ONLY | '+variant+' | '+row['category']+'/'+row['mode'], fill='black')
            draw.text((4, top+15), row['prompt'], fill='black')
            draw.text((4, top+29), f"P{record['PSNR']:.2f} L{record['LPIPS']:.3f} crop{record['crop_LPIPS']:.3f} | INPUT / GT offline / PREDICTION", fill='black')
            for col, value in enumerate((data['reference'][i], data['target'][i], prediction[0])):
                panel.paste(tensor_image(value), (col*256, top+54))
    result = {'count': len(records), 'images': records,
        **{key: sum(record[key] for record in records)/len(records) for key in ('PSNR', 'LPIPS', 'crop_LPIPS', 'retained_L1_RGB_01')},
        'by_category_mode': {category: {mode: {key: sum(record[key] for record in records if (record['category'], record['mode']) == (category, mode))/sum((record['category'], record['mode']) == (category, mode) for record in records)
            for key in ('PSNR', 'LPIPS', 'crop_LPIPS', 'retained_L1_RGB_01')} for mode in ('background', 'object', 'joint')} for category in ('lamp', 'pillow')}}
    write_json(folder/(label+'.json'), result); panel.save(folder/(label+'.png'))
    return result


def run(args, cfg):
    parent_report = completed_teacher(args.parent)
    if parent_report['checkpoint_sha256'] != cfg['expected_parent_checkpoint_sha256']:
        raise ValueError('Use the exact completed STANDARD parent; preserve all prior candidates')
    state = torch.load(args.parent/'best_checkpoint.pt', map_location='cpu', weights_only=False)
    if (state.get('schema') != 'qwen_spatial_joint_lora_v1' or state['step'] != parent_report['best_step'] or
            state['visual_layout'].get('semantic_prefix', 0) or not state['config'].get('multimodal_semantic_readout')):
        raise ValueError('Expected the unchanged, complete joint-adapted fused semantic parent')
    if state['visual_layout']['qwen_config_sha256'] != sha256(args.qwen/'config.json'):
        raise ValueError('The selected full-Qwen configuration bytes changed')
    payload = load_data(args.data)
    teacher_sha = state['config'].get('frozen_joint_teacher_checkpoint_sha256', parent_report['checkpoint_sha256'])
    validate_frozen_joint_cache(payload, state, teacher_sha)
    audit = json.loads((args.data/'data_audit.json').read_text())
    if audit.get('protocol') != cfg['expected_data_protocol']:
        raise ValueError('Unexpected prepared data protocol')
    ids = review_ids(payload['rows'], 'train_probe')
    if len(ids) != cfg['expected_train_probe_pairs'] or any(payload['rows'][i]['split'] != 'train' for i in ids):
        raise ValueError('Exact existing TRAIN60 probe identities required')
    data = {'rows': [payload['rows'][i] for i in ids], 'reference': payload['reference'][ids].clone(),
            'target': payload['target'][ids].clone()}
    manifest_path = args.data/'TRAIN_counterfactual_groups.json'
    paired_contract = counterfactual_contract(data, json.loads(manifest_path.read_text(encoding='utf-8')))
    paired_contract['manifest_sha256'] = sha256(manifest_path)
    expected_hidden = [payload['hidden'][i] for i in ids]
    cache_sha = sha256(args.data/'frozen_features.pt'); del payload; gc.collect()
    inputs = {'reference': data['reference'], 'rows': [{'prompt': row['prompt']} for row in data['rows']]}
    fingerprints = [input_fingerprint(data['reference'][i], row['prompt']) for i, row in enumerate(data['rows'])]
    write_json(args.output/'TRAIN_pairs.json', data['rows'])
    device = torch.device('cuda:0'); torch.cuda.set_per_process_memory_fraction(cfg['gpu_memory_fraction'], 0)
    qwen, processor = load_qwen(args.qwen, device)
    install_joint_lora(qwen, state['config'], state['qwen_joint_lora'], train=False)
    if any(value.requires_grad for value in qwen.parameters()):
        raise ValueError('The complete selected Qwen frontend must remain frozen')
    frontend_counts = {'vision_blocks': len(qwen.model.visual.blocks), 'language_blocks': len(qwen.model.language_model.layers),
        'actual_full_Qwen_parameters': sum(value.numel() for value in qwen.parameters()),
        'actual_forward_Qwen_parameters': sum(value.numel() for value in qwen.model.parameters()),
        'joint_LoRA_parameters': sum(value.numel() for value in state['qwen_joint_lora'].values())}
    if (frontend_counts['vision_blocks'], frontend_counts['language_blocks']) != (24, 28):
        raise ValueError('Original full native frontend required')
    hidden = []; native = []; checks = []; same_rgb = {}; hook_calls = 0
    for i, row in enumerate(data['rows']):
        fused, field, calls = capture_raw(qwen, processor, inputs, i, state['visual_layout'], device)
        difference = float((fused.float()-expected_hidden[i].float()).abs().max())
        if difference > cfg['max_fused_cache_absolute_difference']:
            raise ValueError('Actual selected complete-Qwen fused states differ from bound cache')
        rgb_sha = hashlib.sha256(tensor_image(data['reference'][i]).tobytes()).hexdigest()
        if rgb_sha in same_rgb and not torch.equal(field, same_rgb[rgb_sha]):
            raise ValueError('Native image-only merger changed across prompts on identical RGB')
        same_rgb.setdefault(rgb_sha, field); hidden.append(fused); native.append(field); hook_calls += calls
        checks.append({'sample_id': row['sample_id'], 'input_fingerprint': fingerprints[i], 'reference_RGB_sha256': rgb_sha,
                       'raw_fused_max_abs_difference': difference, 'main_merger_hook_calls': calls})
        if (i+1) % 12 == 0:
            write_json(args.output/'progress.json', {'stage': 'actual_TRAIN_only_frozen_frontend_cache', 'done': i+1, 'total': len(ids)})
    frontend = {'complete': True, 'TRAIN_only': True, 'DEV_forwards': 0, 'independent_TEST_opened': False,
        'parent_checkpoint_sha256': parent_report['checkpoint_sha256'], 'frozen_frontend_teacher_checkpoint_sha256': teacher_sha,
        'joint_adapter_sha256': joint_adapter_digest(state['qwen_joint_lora']), 'source_fused_cache_sha256': cache_sha,
        'paired_counterfactual_contract': paired_contract,
        'sample_ids': [row['sample_id'] for row in data['rows']], 'input_fingerprints': fingerprints,
        'hook_module': 'qwen.model.visual.merger (main merger only)', 'main_merger_hook_calls': hook_calls,
        'same_RGB_native_field_exactly_equal_across_prompts': True, 'distinct_reference_RGBs': len(same_rgb),
        'qwen_config_sha256': sha256(args.qwen/'config.json'), 'counts': frontend_counts, 'checks': checks}
    feature_sha = save_with_digest({'schema': SCHEMA, 'fused': hidden, 'native_merger': native, 'provenance': frontend}, args.output/'TRAIN_native_features.pt')
    frontend['feature_cache_sha256'] = feature_sha; write_json(args.output/'frontend_cache_report.json', frontend)
    del qwen, processor, expected_hidden, same_rgb; gc.collect(); torch.cuda.empty_cache()
    write_json(args.output/'frontend_training_boundary.json', {'full_Qwen_deleted_before_head_training': True,
        'only_TRAIN_frozen_states_retained': len(ids), 'raw_Qwen_reloaded_only_for_final_selected_head_checks': True})
    import lpips
    metric = lpips.LPIPS(net='alex').to(device).eval().requires_grad_(False)
    warm_batches, paired_batches = batch_orders(len(ids), cfg)
    supervision = None; initial = None; original_conditions = None; results = {}
    for variant in VARIANTS:
        torch.manual_seed(cfg['seed'])
        folder = args.output/variant; folder.mkdir()
        head = build_head(args.sd, state, device)
        current_initial = initial_digests(head)
        if initial is None:
            initial = current_initial
        elif current_initial != initial:
            raise ValueError('The two variants did not start from exactly the same parent head')
        if supervision is None:
            references = []; targets = []; edit = []; boxes = []
            with torch.no_grad():
                for i, row in enumerate(data['rows']):
                    for role, values in (('reference', references), ('target', targets)):
                        image = image_float(data[role][i:i+1], device, head.vae.dtype)
                        values.append((head.vae.encode(image).latent_dist.mode()*head.vae.config.scaling_factor).float().cpu())
                    region = masks(row, device)[0].cpu()
                    actual = (data['reference'][i] != data['target'][i]).any(0)
                    if not torch.equal(region[0, 0] > 0, actual):
                        raise ValueError('TRAIN changed-pixel mask differs from reference/target bytes')
                    edit.append(region); boxes.append(region_boxes(masks(row, device, object_union=True)[0]).cpu())
            supervision = {'reference_latent': torch.cat(references), 'target_latent': torch.cat(targets),
                           'edit': edit, 'boxes': torch.cat(boxes)}
        cell_ids = first_cells(data['rows'])
        start_conditions = conditions_for(head, hidden, cell_ids, device)
        if original_conditions is None:
            original_conditions = start_conditions
        condition_checks_before = check_conditions(head, hidden, original_conditions, device)
        before = evaluate(head, variant, data, hidden, native, supervision, metric, device, folder, 'before_training')
        warm = torch.optim.AdamW(head.spatial.parameters(), lr=cfg['warm_lr'], weight_decay=.01)
        for step, batch in enumerate(warm_batches, 1):
            head.eval(); head.spatial.train(); warm.zero_grad(set_to_none=True)
            h, valid, field = make_inputs(hidden, native, batch, variant, device)
            visual, context = head.spatial_inputs(h, valid, native_visual=field)
            _, auxiliary = head.spatial(visual, with_aux=True, context=context)
            loss = F.mse_loss(auxiliary.float(), supervision['reference_latent'][batch].to(device))
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite TRAIN reference warm loss')
            loss.backward(); norm = torch.nn.utils.clip_grad_norm_(head.spatial.parameters(), cfg['max_grad_norm'])
            if not torch.isfinite(norm):
                raise FloatingPointError('Nonfinite TRAIN reference warm gradients')
            warm.step()
            if step % 50 == 0:
                write_json(args.output/'progress.json', {'stage': 'TRAIN_reference_aux_warm', 'variant': variant, 'step': step, 'steps': cfg['warm_steps'], 'loss': float(loss.detach())})
        write_json(folder/'warm_report.json', {'steps': cfg['warm_steps'], 'TRAIN_only': True, 'final_batch_reference_latent_MSE': float(loss.detach())})
        del warm
        spatial = list(head.spatial.parameters()); injection = list(head.unet.conv_in.spatial.parameters()); parameters = spatial+injection
        optimizer = torch.optim.AdamW([{'params': spatial, 'lr': cfg['spatial_lr']},
                                      {'params': injection, 'lr': cfg['spatial_injection_lr']}], weight_decay=.01)
        history = []; best = float('inf'); best_sha = None; best_step = 0; smoke = []
        for step, batch in enumerate(paired_batches, 1):
            head.train(); head.vae.eval(); head.adapter.eval(); optimizer.zero_grad(set_to_none=True)
            h, valid, field = make_inputs(hidden, native, batch, variant, device)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                z, rgb, auxiliary = head.forward_with_aux(h, valid, noise_for([data['rows'][i] for i in batch], device), native_visual=field)
                target = image_float(data['target'][batch], device)
                edit = torch.cat([supervision['edit'][i] for i in batch]).to(device)
                latent = weighted_edit_regression(z, supervision['target_latent'][batch].to(device), edit,
                    cfg['latent_edit_factor'], cfg['latent_keep_factor'], squared=True)
                pixel = weighted_edit_regression(rgb, target, edit, cfg['rgb_edit_factor'], cfg['rgb_keep_factor'])
                perceptual = metric(rgb.float().clamp(-1, 1), target).mean()
                box = supervision['boxes'][batch].to(device)
                crop = metric(crop_regions(rgb.clamp(-1, 1), box), crop_regions(target, box)).mean()
                reference = F.mse_loss(auxiliary.float(), supervision['reference_latent'][batch].to(device))
                loss = cfg['latent_weight']*latent+cfg['rgb_weight']*pixel+cfg['lpips_weight']*perceptual+cfg['object_crop_weight']*crop+cfg['reference_weight']*reference
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite TRAIN paired loss')
            loss.backward(); norm = torch.nn.utils.clip_grad_norm_(parameters, cfg['max_grad_norm'])
            if not torch.isfinite(norm):
                raise FloatingPointError('Nonfinite TRAIN paired gradients')
            if step <= 4:
                smoke.append({'step': step, 'loss': float(loss.detach()), 'gradient_norm': float(norm),
                    'spatial_projection_gradient_absolute_sum': gradient_sum(head.spatial.project.parameters()),
                    'spatial_injection_gradient_absolute_sum': gradient_sum(injection)})
            if step == 4:
                write_json(folder/'smoke_report.json', {'real_paired_backward_steps': 4, 'finite_loss_gradient': True,
                    'TRAIN_only': True, 'steps': smoke,
                    'Qwen_frozen_and_not_loaded_during_training': True, 'semantic_adapter_frozen': True,
                    'UNet_except_existing_spatial_injection_frozen': True, 'VAE_frozen': True})
            optimizer.step()
            if step % 10 == 0:
                write_json(args.output/'progress.json', {'stage': 'TRAIN_paired_spatial_fit', 'variant': variant, 'step': step, 'steps': cfg['paired_steps'], 'loss': float(loss.detach())})
            if step % cfg['evaluate_every'] == 0 or step == cfg['paired_steps']:
                result = evaluate(head, variant, data, hidden, native, supervision, metric, device, folder, 'train_latest')
                history.append({'step': step, 'loss': float(loss.detach()), **{key: result[key] for key in ('PSNR', 'LPIPS', 'crop_LPIPS', 'retained_L1_RGB_01')}})
                saved = {'schema': SCHEMA, 'variant': variant, 'step': step, 'config': cfg,
                    'parent_checkpoint_sha256': parent_report['checkpoint_sha256'], 'generator_config': state['config'],
                    'visual_layout': state['visual_layout'],
                    'spatial': {name: value.detach().cpu() for name, value in head.spatial.state_dict().items()},
                    'spatial_injection': {name: value.detach().cpu() for name, value in head.unet.conv_in.spatial.state_dict().items()},
                    'TRAIN_metrics': result,
                    'semantic_adapter_sha256': initial['semantic_adapter'], 'frontend_cache_sha256': feature_sha}
                save_with_digest(saved, folder/'last_checkpoint.pt')
                score = result['LPIPS']+.5*result['crop_LPIPS']
                if score < best:
                    best = score; best_step = step
                    best_sha = save_with_digest(saved, folder/'best_checkpoint.pt')
                write_json(folder/'history.json', history)
        if not best_step:
            raise ValueError('A learned positive-step checkpoint is required')
        selected = torch.load(folder/'best_checkpoint.pt', map_location='cpu', weights_only=False)
        head.spatial.load_state_dict(selected['spatial']); head.unet.conv_in.spatial.load_state_dict(selected['spatial_injection'])
        selected_metrics = evaluate(head, variant, data, hidden, native, supervision, metric, device, folder, 'train_best', export=True)
        condition_checks_after = check_conditions(head, hidden, original_conditions, device)
        if joint_adapter_digest(head.adapter.state_dict()) != initial['semantic_adapter']:
            raise ValueError('The original semantic adapter was updated')
        results[variant] = {'complete': True, 'diagnostic_only': True, 'quality_accepted': False,
            'spatial_source': variant, 'semantic_source': 'unchanged complete original fused hidden and original mask',
            'best_step': best_step, 'checkpoint_sha256': best_sha, 'parent_checkpoint_sha256': parent_report['checkpoint_sha256'],
            'selection': 'TRAIN60 mean LPIPS plus .5 object crop LPIPS; not DEV/generalization',
            'before_training': before, 'best': selected_metrics, 'initial_state_sha256': current_initial,
            'semantic_condition_differences_before': condition_checks_before,
            'semantic_condition_differences_after': condition_checks_after,
            'parameters': {'trainable_spatial_including_auxiliary': sum(value.numel() for value in spatial),
                'trainable_existing_spatial_injection': sum(value.numel() for value in injection),
                'trainable_total': sum(value.numel() for value in parameters), 'additional_deployment_parameters': 0,
                'parent_inference_parameters_unmerged': parent_report['parameters']['inference_total']}}
        write_json(folder/'report.json', results[variant])
        # Free the final backward graph as well as modules before loading the
        # next full UNet; deleting only ``head`` leaves the graph's parameters
        # alive through z/rgb/loss and the warm auxiliary references.
        del (head, optimizer, parameters, spatial, injection, selected, h, valid,
             field, visual, context, auxiliary, z, rgb, loss, latent, pixel,
             perceptual, crop, reference, target, edit, box, norm)
        gc.collect(); torch.cuda.empty_cache()
    # No Qwen instance remains while either head is being optimized. Reload it
    # only now to verify selected-head predictions on actual TRAIN RGB+text.
    qwen, processor = load_qwen(args.qwen, device)
    install_joint_lora(qwen, state['config'], state['qwen_joint_lora'], train=False)
    replay = {}
    for variant in VARIANTS:
        folder = args.output/variant; selected = torch.load(folder/'best_checkpoint.pt', map_location='cpu', weights_only=False)
        head = build_head(args.sd, state, device).eval().requires_grad_(False)
        head.spatial.load_state_dict(selected['spatial']); head.unet.conv_in.spatial.load_state_dict(selected['spatial_injection'])
        checks = []
        for i in first_cells(data['rows']):
            fused, field, calls = capture_raw(qwen, processor, inputs, i, state['visual_layout'], device)
            difference = float((fused.float()-hidden[i].float()).abs().max())
            if difference > cfg['max_fused_cache_absolute_difference'] or not torch.equal(field, native[i]):
                raise ValueError('Raw selected frontend replay differs from captured TRAIN states')
            h, valid = pack_hidden([fused], device)
            with torch.no_grad():
                _, rgb = head(h, valid, noise_for([data['rows'][i]], device), native_visual=native_argument(variant, field[None].to(device)))
            actual = tensor_image(rgb[0]); path = folder/'images/train_best/generated'/f'sample_{i:04d}.png'
            with Image.open(path) as expected:
                same = actual.tobytes() == expected.convert('RGB').tobytes()
            checks.append({'sample_id': data['rows'][i]['sample_id'], 'main_merger_hook_calls': calls,
                'raw_fused_max_abs_difference': difference, 'native_exact_same_field': True, 'same_selected_PNG': same})
            if not same:
                raise ValueError('Selected raw RGB differs from its cache-accelerated TRAIN export')
            destination = folder/'raw_replay'/f'sample_{i:04d}.png'; destination.parent.mkdir(exist_ok=True); actual.save(destination)
        replay[variant] = {'passed': True, 'count': len(checks), 'checks': checks}
        write_json(folder/'raw_replay_identity.json', replay[variant])
        del head, selected; gc.collect(); torch.cuda.empty_cache()
    del qwen, processor; gc.collect(); torch.cuda.empty_cache()
    write_json(args.output/'report.json', {'complete': True, 'diagnostic_only': True, 'quality_accepted': False,
        'new_baseline_or_LG_teacher_accepted': False, 'full_raw_deployment_accepted': False,
        'scope': 'TRAIN60 fit only; no DEV or independent TEST forwards, gradients or selection',
        'source_commit': os.environ.get('T12_SOURCE_COMMIT'), 'source_sha256': sha256(Path(__file__)),
        'parent_checkpoint_sha256': parent_report['checkpoint_sha256'], 'parent_quality_accepted': False,
        'config': cfg, 'frontend': frontend, 'variants': results, 'raw_replay': replay,
        'identical_semantic_branch': True, 'matched_initial_heads': True, 'matched_batches_seeds_losses_budget': True,
        'no_new_TF_encoder_or_reference_VAE_inference_branch': True,
        'limitations': 'Changing only the spatial source can identify a readout-dependent fitting difference. TRAIN fit, LPIPS or prompt response cannot establish generalization, unique causation, semantic success, LG quality or speed. Original full fused conditions are unchanged; only the existing spatial bridge reads LMvisual64 versus native merger64.'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'parent', 'data', 'qwen', 'sd', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--gpu-uuid', required=True)
    args = parser.parse_args(); cfg = json.loads(args.config.read_text(encoding='utf-8'))
    if (cfg.get('schema') != SCHEMA or tuple(cfg.get('variants', [])) != VARIANTS or
            args.gpu_uuid != cfg['gpu_uuid'] or args.gpu_uuid != GPU_UUID or
            os.environ.get('CUDA_VISIBLE_DEVICES') != args.gpu_uuid
            or args.output.exists() or cfg['paired_steps'] < 4 or cfg['warm_steps'] < 1):
        raise ValueError('Reviewed paired protocol, authorized single GPU6 and immutable new output required')
    args.output.mkdir(parents=True)
    write_json(args.output/'execution.json', {'argv': sys.argv, 'pid': os.getpid(), 'started_unix': time.time(),
        'source_commit': os.environ.get('T12_SOURCE_COMMIT'), 'source_sha256': sha256(Path(__file__)),
        'config_sha256': sha256(args.config), 'config': cfg, 'torch': torch.__version__, 'cuda': torch.version.cuda,
        'gpu_uuid': args.gpu_uuid, 'forward_input_scope': 'Selected TRAIN reference RGB and supplied prompt only'})
    torch.set_num_threads(4); torch.manual_seed(cfg['seed'])
    try:
        run(args, cfg)
    except Exception:
        write_json(args.output/'failure.json', {'traceback': traceback.format_exc(), 'completed': False, 'time': time.time()})
        raise
    finally:
        gc.collect()
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
