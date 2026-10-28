"""Distill a smaller native UNet after the Qwen optical replacement.

Uses frozen actual hybrid-Qwen features during optimization, and recomputes raw
Qwen features for their cache. Deployment still runs the full hybrid frontend.
TRAIN teacher images are loss targets only; the development cohort selects best.
"""
import argparse
import copy
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

from .qwen_vl_hybrid import apply_configured_hybrid, load_macro_state, restore_head_overlay, clone_readout_head
from .qwen_vl_hybrid_run import completed_teacher, load_data, load_qwen, quality_gate, raw_hidden
from .qwen_vl_spatial import balanced_region_l1, build_spatial_head
from .qwen_vl_spatial_run import evaluate, masks
from .qwen_vl_tail import attention_hooks, build_tail_student
from .qwen_vl_turbo_cache import image_float, save_with_digest
from .qwen_vl_turbo_data import pilot_subset, sha256, write_json
from .qwen_vl_turbo_model import pack_hidden
from .qwen_vl_turbo_run import git_identity, noise_for
from .qwen_vl_turbo_losses import crop_regions, region_boxes


def save_student(student, config, profile, hybrid_sha, step, metrics, path):
    return save_with_digest({'schema': 'qwen_vl_compressed_tail_v1', 'config': config,
          'profile': profile, 'unet': {k: v.detach().cpu().clone() for k, v in student.state_dict().items()},
          'hybrid_checkpoint_sha256': hybrid_sha, 'step': step, 'metrics': metrics, 'git': git_identity()}, path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'data', 'parent', 'hybrid', 'qwen', 'sd', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args(); cfg = json.loads(a.config.read_text())
    if os.environ.get('CUDA_VISIBLE_DEVICES') != cfg['gpu_uuid']:
        raise ValueError('Unauthorized GPU')
    if (a.output/'execution.json').exists():
        raise ValueError('Existing compression run; inspect instead of restarting')
    a.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4); torch.manual_seed(cfg['seed'])
    torch.cuda.set_per_process_memory_fraction(cfg['gpu_memory_fraction'], 0)
    device = torch.device('cuda:0')
    teacher_report = completed_teacher(a.parent)
    hybrid_report = json.loads((a.hybrid/'report.json').read_text())
    if not hybrid_report['complete'] or sha256(a.hybrid/'best_checkpoint.pt') != hybrid_report['checkpoint_sha256']:
        raise ValueError('Hybrid recovery checkpoint incomplete or changed')
    state = torch.load(a.parent/'best_checkpoint.pt', map_location='cpu', weights_only=False)
    if state.get('schema')=='qwen_spatial_joint_lora_v1':
        raise ValueError('Wait for validated joint-teacher LG terminal features and recovery before tail compression')
    hybrid = torch.load(a.hybrid/'best_checkpoint.pt', map_location='cpu', weights_only=False)
    payload = load_data(a.data)
    train = [i for i, r in enumerate(payload['rows']) if r['split'] == 'train']
    dev = [i for i, r in enumerate(payload['rows']) if r['split'] == 'val']
    names = {r['sample_id'] for r in pilot_subset(payload['rows'], 2)}
    train_diagnostic = [i for i in train if payload['rows'][i]['sample_id'] in names]
    if cfg.get('smoke'):
        train = train_diagnostic[:6]
        dev = [i for i in dev if payload['rows'][i]['sample_id'] in names][:6]
        train_diagnostic = train
    write_json(a.output/'execution.json', {'argv': sys.argv, 'pid': os.getpid(), 'git': git_identity(),
       'source_sha256': {n: sha256(Path(__file__).with_name(n)) for n in ('qwen_vl_tail.py', 'qwen_vl_tail_run.py')},
       'started_unix': time.time(), 'config': cfg, 'gpu_uuid': cfg['gpu_uuid'],
       'teacher_sha256': teacher_report['checkpoint_sha256'], 'hybrid_sha256': hybrid_report['checkpoint_sha256'],
       'evaluation_label': 'development_test', 'test_gradient_updates': False})
    try:
        qwen, processor = load_qwen(a.qwen, device)
        macros = apply_configured_hybrid(qwen,hybrid['config'])
        load_macro_state(macros, hybrid['macros']); qwen.eval().requires_grad_(False)
        temporary = dict(payload); temporary['hidden'] = list(payload['hidden'])
        with torch.no_grad():
            for j, i in enumerate(train+dev):
                temporary['hidden'][i] = raw_hidden(qwen, processor, payload, i, state['visual_layout'], device).cpu()
                if (j+1) % 24 == 0:
                    write_json(a.output/'progress.json', {'stage': 'actual_hybrid_feature_cache', 'done': j+1, 'total': len(train+dev)})
        cache_sha = save_with_digest({'sample_ids': [r['sample_id'] for r in payload['rows']],
                         'hidden': temporary['hidden'], 'hybrid_sha256': hybrid_report['checkpoint_sha256']}, a.output/'hybrid_features.pt')
        del qwen, processor, macros; __import__('gc').collect(); torch.cuda.empty_cache()
        teacher = build_spatial_head(a.sd, state, state['config'], state['visual_layout']['positions'], device,
                                     drop_vae_encoder=True).eval().requires_grad_(False)
        student_unet, profile = build_tail_student(teacher.unet, cfg['channels'], cfg['heads'])
        student_unet = student_unet.to(device).float().requires_grad_(True)
        student_unet.enable_gradient_checkpointing()
        # Use a new head: a shallow nn.Module copy would alias its registry.
        student = clone_readout_head(teacher,unet=student_unet)
        restore_head_overlay(student,hybrid)
        import lpips
        metric = lpips.LPIPS(net='alex').to(device).eval().requires_grad_(False)
        train_goals, teacher_maps = {}, {}
        handles = attention_hooks(teacher.unet, teacher_maps, detach=True)
        with torch.no_grad():
            for j, i in enumerate(train):
                teacher_maps.clear()
                h, valid = pack_hidden([payload['hidden'][i]], device)
                z, rgb = teacher(h, valid, noise_for([payload['rows'][i]], device))
                train_goals[i] = {'z': z.cpu().half(), 'rgb': rgb.cpu().half(), 'maps': dict(teacher_maps)}
                if (j+1) % 24 == 0:
                    write_json(a.output/'progress.json', {'stage': 'train_teacher_output_cache', 'done': j+1, 'total': len(train)})
        for handle in handles:
            handle.remove()
        del teacher.unet
        __import__('gc').collect(); torch.cuda.empty_cache()
        goal_sha = save_with_digest(train_goals, a.output/'train_teacher_outputs.pt')
        write_json(a.output/'cache_identity.json', {'actual_hybrid_features_sha256': cache_sha,
              'teacher_train_outputs_sha256': goal_sha, 'target_or_mask_input': False,
              'cached_train_teacher_images_are_loss_only': True, 'deployment_requires_raw_qwen': True})
        profile.update(student_total_including_frozen_head=sum(p.numel() for p in student.parameters()))
        write_json(a.output/'parameter_counts.json', profile)
        before = evaluate(student, temporary, dev, device, a.output, 'dev_before', metric)
        optimizer = torch.optim.AdamW(student_unet.parameters(), lr=cfg['lr'], weight_decay=.001)
        rng, history, best, best_step = random.Random(cfg['seed']), [], math.inf, 0
        steps = cfg.get('steps') or math.ceil(len(train)*cfg['epochs']/cfg['batch_size'])
        order = []
        while len(order) < steps*cfg['batch_size']:
            block = train.copy(); rng.shuffle(block); order.extend(block)
        maps = {}; handles = attention_hooks(student_unet, maps)
        for step in range(1, steps+1):
            student_unet.train(); optimizer.zero_grad(set_to_none=True)
            factor = min(1., step/max(1, cfg['warmup_steps']))*(.1+.9*.5*(1+math.cos(math.pi*step/steps)))
            for group in optimizer.param_groups:
                group['lr'] = cfg['lr']*factor
            ids = order[(step-1)*cfg['batch_size']:step*cfg['batch_size']]
            h, valid = pack_hidden([temporary['hidden'][i] for i in ids], device)
            maps.clear()
            z, rgb = student(h, valid, noise_for([payload['rows'][i] for i in ids], device))
            target = image_float(payload['target'][ids], device)
            edit = torch.cat([masks(payload['rows'][i], device)[0] for i in ids])
            with torch.autocast('cuda', dtype=torch.bfloat16):
                latent = F.mse_loss(z.float(), torch.cat([train_goals[i]['z'] for i in ids]).to(device).float())
                image_kd = (rgb.float()-torch.cat([train_goals[i]['rgb'] for i in ids]).to(device).float()).abs().mean()
                rgb_loss = balanced_region_l1(rgb, target, edit)
                perception = metric(rgb.clamp(-1, 1), target).mean()
                union = torch.cat([masks(payload['rows'][i], device, object_union=True)[0] for i in ids])
                boxes = region_boxes(union)
                crop_loss = metric(crop_regions(rgb.clamp(-1, 1), boxes), crop_regions(target, boxes)).mean()
                attention = sum(F.mse_loss(value.float(), torch.cat([train_goals[i]['maps'][k] for i in ids]).to(device)) for k, value in maps.items())/len(maps)
                loss = cfg['latent_kd_weight']*latent+cfg['rgb_kd_weight']*image_kd+cfg['rgb_weight']*rgb_loss+cfg['lpips_weight']*perception+cfg.get('crop_lpips_weight', .2)*crop_loss+cfg['attention_weight']*attention
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite tail loss')
            loss.backward(); norm = torch.nn.utils.clip_grad_norm_(student_unet.parameters(), 1.)
            if not torch.isfinite(norm):
                raise FloatingPointError('Nonfinite tail gradient')
            optimizer.step()
            if step % 10 == 0:
                write_json(a.output/'progress.json', {'stage': 'tail_distillation', 'step': step, 'total_steps': steps,
                    'epoch': step*cfg['batch_size']/len(train), 'train_loss': float(loss.detach()), 'best_step': best_step})
            if step % cfg['eval_every'] == 0 or step == steps:
                val = evaluate(student, temporary, dev, device, a.output, 'dev_latest', metric)
                tr = evaluate(student, temporary, train_diagnostic, device, a.output, 'train_latest', metric)
                score = val['mean_lpips']+.25*val['mean_object_crop_lpips']
                entry = {'step': step, 'epoch': step*cfg['batch_size']/len(train), 'train_lpips': tr['mean_lpips'],
                     'dev_lpips': val['mean_lpips'], 'dev_crop_lpips': val['mean_object_crop_lpips'],
                     'dev_psnr': val['mean_per_image_psnr_db'], 'by_mode': val['lpips_by_mode'],
                     'dev_crop_by_mode': val['object_crop_lpips_by_mode'], 'dev_category_mode': val['by_category_mode'],
                     'quality_gate': quality_gate(val, teacher_report['best_val'])}
                history.append(entry)
                save_student(student_unet, cfg, profile, hybrid_report['checkpoint_sha256'], step, entry, a.output/'last_checkpoint.pt')
                if score < best:
                    best, best_step = score, step
                    save_student(student_unet, cfg, profile, hybrid_report['checkpoint_sha256'], step, entry, a.output/'best_checkpoint.pt')
                write_json(a.output/'history.json', history)
                if cfg.get('early_stop_evaluations') and len(history) >= 5 and step-best_step >= cfg['early_stop_evaluations']*cfg['eval_every']:
                    break
        for handle in handles:
            handle.remove()
        chosen = torch.load(a.output/'best_checkpoint.pt', map_location='cpu', weights_only=False)
        student_unet.load_state_dict(chosen['unet'])
        val = evaluate(student, temporary, dev, device, a.output, 'dev_best', metric, export=True)
        write_json(a.output/'report.json', {'complete': True, 'best_step': best_step, 'best_val': val,
            'before': before, 'profile': profile, 'checkpoint_sha256': sha256(a.output/'best_checkpoint.pt'),
            'teacher_sha256': teacher_report['checkpoint_sha256'], 'hybrid_sha256': hybrid_report['checkpoint_sha256'],
            'quality_gate': quality_gate(val, teacher_report['best_val']), 'alpha_minimum': .60,
            'test_gradient_updates': False, 'evaluation_label': 'development_test', 'evaluation_used_for_selection': True,
            'quality_accepted': False, 'timing_measured': False})
    except Exception:
        write_json(a.output/'failure.json', {'traceback': traceback.format_exc()}); raise


if __name__ == '__main__':
    main()
