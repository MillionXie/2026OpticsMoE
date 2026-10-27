"""Fast cached-readout pilot after removing all post-optical TF blocks.

Frozen actual postfree Qwen produces both original/concise TRAIN conditions.
Only TRAIN GT supervises latent/RGB/perceptual losses. DEV96 selects best;
the separate 240-pair held-out cohort is never opened by this module.
"""
import gc
import argparse
import json
import math
import os
import random
import sys
import time
import traceback
from pathlib import Path

import torch
from torch.nn import functional as F

from .qwen_vl_hybrid import load_macro_state
from .qwen_vl_hybrid_run import completed_teacher, load_data, load_qwen, raw_inputs, quality_gate
from .qwen_vl_postfree import apply_postfree, instruction_bounds, FocusedHead, short_prompt
from .qwen_vl_spatial import balanced_region_l1, build_spatial_head
from .qwen_vl_spatial_run import evaluate, masks
from .qwen_vl_tail import build_tail_student
from .qwen_vl_turbo_cache import image_float, save_with_digest
from .qwen_vl_turbo_data import pilot_subset, sha256, write_json
from .qwen_vl_turbo_losses import region_boxes, crop_regions
from .qwen_vl_turbo_model import pack_hidden, region_weighted_loss
from .qwen_vl_turbo_run import noise_for, tensor_image


def build_generator(sd, parent, tail, device, drop_encoder=False, compressed=True):
    base = build_spatial_head(sd, parent, parent['config'], parent['visual_layout']['positions'],
                              device, drop_vae_encoder=drop_encoder).eval().requires_grad_(False)
    if compressed:
        smaller, _ = build_tail_student(base.unet, tail['profile']['channels'], tail['profile']['heads'])
        smaller.load_state_dict(tail['unet'], strict=True)
        base.unet = smaller.to(device).bfloat16().eval().requires_grad_(False)
    return base


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('parent', 'hybrid', 'tail', 'data', 'qwen', 'sd', 'config', 'output'):
        p.add_argument('--'+key, type=Path, required=True)
    a = p.parse_args()
    cfg = json.loads(a.config.read_text())
    if os.environ.get('CUDA_VISIBLE_DEVICES') != cfg['gpu_uuid']:
        raise ValueError('Only approved GPU6 UUID')
    if a.output.exists():
        raise FileExistsError('Existing pilot must be inspected, never restarted/overwritten')
    teacher_report = completed_teacher(a.parent)
    hr = json.loads((a.hybrid/'report.json').read_text())
    tr = json.loads((a.tail/'report.json').read_text())
    if not hr['complete'] or not tr['complete'] or hr['best_step'] <= 0 or tr['best_step'] <= 0:
        raise ValueError('Wait for complete VAL-selected parent runs')
    hybrid_path, tail_path = a.hybrid/'best_checkpoint.pt', a.tail/'best_checkpoint.pt'
    if sha256(hybrid_path) != hr['checkpoint_sha256'] or sha256(tail_path) != tr['checkpoint_sha256']:
        raise ValueError('Parent SHA mismatch')
    parent = torch.load(a.parent/'best_checkpoint.pt', map_location='cpu', weights_only=False)
    hybrid = torch.load(hybrid_path, map_location='cpu', weights_only=False)
    tail = torch.load(tail_path, map_location='cpu', weights_only=False)
    if tail['hybrid_checkpoint_sha256'] != hr['checkpoint_sha256']:
        raise ValueError('Tail/macros identity mismatch')
    a.output.mkdir(parents=True)
    torch.set_num_threads(4); torch.manual_seed(cfg['seed'])
    torch.cuda.set_per_process_memory_fraction(.85, 0)
    device = torch.device('cuda:0')
    write_json(a.output/'execution.json', {'pid': os.getpid(), 'started_unix': time.time(),
        'argv': sys.argv, 'source_commit': os.environ.get('T12_SOURCE_COMMIT'),
        'source_sha256': {name: sha256(Path(__file__).with_name(name))
                          for name in ('qwen_vl_postfree.py', 'qwen_vl_postfree_run.py')},
        'parent_sha256': teacher_report['checkpoint_sha256'], 'hybrid_sha256': hr['checkpoint_sha256'],
        'tail_sha256': tr['checkpoint_sha256'], 'config': cfg,
        'evaluation_label': 'development_test', 'heldout_240_opened': False})
    write_json(a.output/'config.json', cfg)
    try:
        payload = load_data(a.data)
        descriptions = json.loads((a.data/'product_descriptions.json').read_text())
        rows = payload['rows']; train = [i for i, r in enumerate(rows) if r['split'] == 'train']
        dev = [i for i, r in enumerate(rows) if r['split'] == 'val']
        qwen, processor = load_qwen(a.qwen, device)
        postfree = cfg.get('frontend', 'postfree') == 'postfree'
        if cfg.get('frontend', 'postfree') not in ('postfree', 'baseline'):
            raise ValueError('Unknown frontend')
        if postfree:
            macros = apply_postfree(qwen, hybrid['config']['vision_pairs'],
                                   hybrid['config']['language_pairs'], hybrid['config']['residual_width'])
            load_macro_state(macros, hybrid['macros'])
        qwen.eval().requires_grad_(False)
        frontend_parameters = sum(p.numel() for p in qwen.model.parameters())
        original_hidden, concise_hidden, concise_prompts, boundary = [], [], [], None
        vision_end = processor.tokenizer.convert_tokens_to_ids('<|vision_end|>')
        message_end = processor.tokenizer.convert_tokens_to_ids('<|im_end|>')
        for i, row in enumerate(rows):
            concise = short_prompt(row, descriptions[row['target']['sample_id']])
            concise_prompts.append(concise)
            for prompt, destination in ((row['prompt'], original_hidden), (concise, concise_hidden)):
                inputs = raw_inputs(processor, payload['reference'][i], prompt, parent['visual_layout'], device)
                bound = instruction_bounds(inputs['input_ids'][0], vision_end, message_end)
                start, suffix = bound
                decoded = processor.tokenizer.decode(inputs['input_ids'][0, start:-suffix]).strip()
                if decoded != prompt.strip():
                    raise ValueError('Native instruction token extraction differs')
                if boundary is None:
                    boundary = bound
                if bound != boundary:
                    raise ValueError('Instruction layout changed; no guessed token masks')
                with torch.inference_mode():
                    h = qwen.model(**inputs, use_cache=False, return_dict=True).last_hidden_state[0].bfloat16().cpu()
                destination.append(h)
            if (i+1) % 24 == 0:
                write_json(a.output/'progress.json', {'stage': 'postfree_frozen_feature_cache', 'done': i+1, 'total': len(rows)})
        feature_sha = save_with_digest({'sample_ids': [r['sample_id'] for r in rows],
            'original_hidden': original_hidden, 'concise_hidden': concise_hidden,
            'concise_prompts': concise_prompts, 'instruction_bounds': boundary,
            'hybrid_sha256': hr['checkpoint_sha256'], 'post_tf_removed': postfree}, a.output/'postfree_features.pt')
        del qwen, processor
        if postfree:
            del macros
        gc.collect(); torch.cuda.empty_cache()
        base = build_generator(a.sd, parent, tail, device, compressed=cfg.get('compressed_tail', True))
        target_latent = {}; edits, unions = {}, {}
        with torch.inference_mode():
            for start in range(0, len(train), 16):
                batch = train[start:start+16]
                z = base.vae.encode(image_float(payload['target'][batch], device, base.vae.dtype)).latent_dist.mode()
                z = (z*base.vae.config.scaling_factor).float().cpu()
                for index, value in zip(batch, z):
                    target_latent[index] = value
        # Encodes ONLY TRAIN targets as loss supervision, never reference input.
        target_sha = save_with_digest(target_latent, a.output/'train_target_latents.pt')
        base.vae.encoder = None; base.vae.quant_conv = None; base.spatial.aux_reference = None
        for i in train:
            edits[i] = masks(rows[i], torch.device('cpu'))
            unions[i] = masks({**rows[i], 'mode': 'object'}, torch.device('cpu'))[0]
        model = FocusedHead(base, parent['visual_layout']['positions'], *boundary,
                            initial_gain=cfg['initial_readout_text_gain']).to(device)
        model.base.requires_grad_(False)
        model.base.adapter.requires_grad_(True); model.base.spatial.requires_grad_(True)
        for name, parameter in model.base.unet.named_parameters():
            parameter.requires_grad_('.attn2.' in name or name.startswith('up_blocks.3.') or name == 'conv_in.spatial.weight')
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.float()
        model.base.unet.enable_gradient_checkpointing()
        groups = [{'params': model.base.adapter.parameters(), 'lr': cfg['readout_lr']},
                  {'params': model.base.spatial.parameters(), 'lr': cfg['readout_lr']},
                  {'params': [p for p in model.base.unet.parameters() if p.requires_grad], 'lr': cfg['unet_lr']},
                  {'params': [model.raw_text_gain], 'lr': cfg['gain_lr']}]
        optimizer = torch.optim.AdamW(groups, weight_decay=.001)
        rates = [g['lr'] for g in groups]
        import lpips
        metric = lpips.LPIPS(net='alex').to(device).eval().requires_grad_(False)
        temporary = dict(payload); temporary['hidden'] = original_hidden
        before = evaluate(model, temporary, dev, device, a.output, 'dev_before', metric)
        names = {r['sample_id'] for r in pilot_subset(rows, 2)}
        diagnostic = [i for i in train if rows[i]['sample_id'] in names]
        counts = {'frontend_including_embedding': frontend_parameters,
                  'generation_head': sum(p.numel() for p in model.parameters()),
                  'trainable': sum(p.numel() for p in model.parameters() if p.requires_grad),
                  'new_readout_parameters': 1, 'retained_vision_tf': 6 if postfree else 24,
                  'retained_language_tf': 4 if postfree else 28,
                  'removed_tf': 42 if postfree else 0, 'alpha_minimum': .6 if postfree else None}
        write_json(a.output/'parameter_counts.json', counts)
        rng = random.Random(cfg['seed']); order = []; steps = math.ceil(len(train)*cfg['epochs']/cfg['batch_size'])
        while len(order) < steps*cfg['batch_size']:
            shuffled = train.copy(); rng.shuffle(shuffled); order.extend(shuffled)
        best, best_step, history = math.inf, 0, []
        def save(step, val, path):
            save_with_digest({'schema': 'qwen_vl_postfree_focused_v1', 'config': cfg, 'step': step,
                'trainable_head': {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad},
                'instruction_bounds': boundary, 'val': val, 'post_tf_removed': postfree,
                'source_commit': os.environ.get('T12_SOURCE_COMMIT'), 'frontend_parameters': frontend_parameters,
                'parent_sha256': teacher_report['checkpoint_sha256'], 'hybrid_sha256': hr['checkpoint_sha256'],
                'tail_sha256': tr['checkpoint_sha256'], 'feature_sha256': feature_sha}, path)
        for step in range(1, steps+1):
            model.train(); model.vae.eval(); optimizer.zero_grad(set_to_none=True)
            factor = min(1., step/max(1, cfg['warmup_steps']))*(.1+.9*.5*(1+math.cos(math.pi*step/steps)))
            for group, rate in zip(optimizer.param_groups, rates):
                group['lr'] = rate*factor
            batch = order[(step-1)*cfg['batch_size']:step*cfg['batch_size']]
            hidden = [concise_hidden[i] if rng.random() < cfg['concise_probability'] else original_hidden[i] for i in batch]
            h, valid = pack_hidden(hidden, device)
            z, image = model(h, valid, noise_for([rows[i] for i in batch], device))
            target = image_float(payload['target'][batch], device)
            edit = torch.cat([edits[i][0] for i in batch]).to(device)
            foreground = torch.cat([edits[i][1] for i in batch]).to(device)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                latent = region_weighted_loss(z, torch.stack([target_latent[i] for i in batch]).to(device), foreground, 3.)
                pixel = balanced_region_l1(image, target, edit)
                perceptual = metric(image.clamp(-1, 1), target).mean()
                boxes = region_boxes(torch.cat([unions[i] for i in batch]).to(device))
                crop = metric(crop_regions(image.clamp(-1, 1), boxes), crop_regions(target, boxes)).mean()
                loss = cfg['latent_weight']*latent+cfg['rgb_weight']*pixel+cfg['lpips_weight']*perceptual+cfg['crop_weight']*crop
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite focused instruction loss')
            loss.backward(); norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
            if not torch.isfinite(norm):
                raise FloatingPointError('Nonfinite focused instruction gradient')
            optimizer.step()
            if step == 4:
                write_json(a.output/'smoke_report.json', {'real_backward_steps': 4, 'finite_loss_gradient': True,
                    'TRAIN_only': True, 'readout_gain_gradient_present': model.raw_text_gain.grad is not None})
            if step % 10 == 0:
                write_json(a.output/'progress.json', {'stage': 'focused_head_training', 'step': step, 'total_steps': steps,
                    'epoch': step*cfg['batch_size']/len(train), 'loss': float(loss.detach()), 'best_step': best_step,
                    'readout_text_gain': float(model.text_gain.detach())})
            if step % cfg['eval_every'] == 0 or step == steps:
                val = evaluate(model, temporary, dev, device, a.output, 'dev_latest', metric)
                train_val = evaluate(model, temporary, diagnostic, device, a.output, 'train_latest', metric)
                score = val['mean_lpips']+.25*val['mean_object_crop_lpips']
                history.append({'step': step, 'epoch': step*cfg['batch_size']/len(train),
                    'train_lpips': train_val['mean_lpips'], 'dev_lpips': val['mean_lpips'],
                    'dev_crop_lpips': val['mean_object_crop_lpips'], 'dev_psnr': val['mean_per_image_psnr_db'],
                    'lpips_by_mode': val['lpips_by_mode'], 'category_mode': val['by_category_mode'],
                    'quality_gate': quality_gate(val, teacher_report['best_val'])})
                save(step, val, a.output/'last_checkpoint.pt')
                if score < best:
                    best, best_step = score, step; save(step, val, a.output/'best_checkpoint.pt')
                write_json(a.output/'history.json', history)
                if step-best_step >= cfg['early_stop_evaluations']*cfg['eval_every']:
                    break
        selected = torch.load(a.output/'best_checkpoint.pt', map_location='cpu', weights_only=False)
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if name in selected['trainable_head']:
                    parameter.copy_(selected['trainable_head'][name].to(device))
        val = evaluate(model, temporary, dev, device, a.output, 'dev_best', metric, export=True)
        # Close the cache/inference gap with real RGB+instruction forwards.
        del optimizer; gc.collect(); torch.cuda.empty_cache()
        qwen, processor = load_qwen(a.qwen, device)
        if postfree:
            macros = apply_postfree(qwen, hybrid['config']['vision_pairs'],
                                   hybrid['config']['language_pairs'], hybrid['config']['residual_width'])
            load_macro_state(macros, hybrid['macros'])
        qwen.eval().requires_grad_(False); model.eval(); raw_checks = []
        for i in [next(i for i in dev if rows[i]['category'] == cat and rows[i]['mode'] == mode)
                  for cat in ('lamp', 'pillow') for mode in ('background', 'object', 'joint')]:
            with torch.inference_mode():
                inputs = raw_inputs(processor, payload['reference'][i], rows[i]['prompt'], parent['visual_layout'], device)
                hidden = qwen.model(**inputs, use_cache=False, return_dict=True).last_hidden_state.bfloat16()
                valid = torch.ones(hidden.shape[:2], device=device, dtype=torch.bool)
                _, image = model(hidden, valid, noise_for([rows[i]], device))
            path = a.output/'raw_input_check'/(str(i)+'.png'); path.parent.mkdir(exist_ok=True)
            tensor_image(image[0]).save(path)
            from PIL import Image
            import numpy as np
            cached = a.output/'images/dev_best/generated'/('sample_'+str(dev.index(i)).zfill(4)+'.png')
            error = abs(np.asarray(Image.open(path)).astype(int)-np.asarray(Image.open(cached)).astype(int))
            raw_checks.append({'sample_id': rows[i]['sample_id'], 'max_abs_8bit': int(error.max()),
                               'mean_abs_8bit': float(error.mean())})
        write_json(a.output/'raw_input_identity.json', {'reference_rgb_and_prompt_only': True,
                   'target_or_mask_input': False, 'checks': raw_checks,
                   'native_pngs_agree_within_one_level': all(r['max_abs_8bit'] <= 1 for r in raw_checks)})
        write_json(a.output/'report.json', {'complete': True, 'best_step': best_step, 'best_val': val,
            'before': before, 'checkpoint_sha256': sha256(a.output/'best_checkpoint.pt'),
            'quality_gate': quality_gate(val, teacher_report['best_val']), 'quality_accepted': False,
            'heldout_240_opened': False, 'TRAIN_only_gradients': True, 'cached_features_training_only': True,
            'deployment_requires_actual_postfree_qwen': True, 'alpha_minimum': .6,
            'train_target_latents_sha256': target_sha, 'parameters': counts,
            'source_commit': os.environ.get('T12_SOURCE_COMMIT'), 'evaluation_label': 'development_test'})
    except Exception:
        write_json(a.output/'failure.json', {'traceback': traceback.format_exc()}); raise


if __name__ == '__main__':
    main()
