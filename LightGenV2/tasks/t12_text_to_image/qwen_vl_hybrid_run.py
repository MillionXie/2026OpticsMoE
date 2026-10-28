"""TRAIN feature distillation, then raw-Qwen hybrid image recovery.

The 96-pair cohort is explicitly development TEST used for selection. Only
TRAIN pairs produce gradients. Native Qwen visual/semantic interfaces remain.
"""
import argparse
import gc
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .qwen_vl_hybrid import OpticalMacro, apply_configured_hybrid, feature_loss, directional_feature_loss, hybrid_config, load_macro_state, optimizer_groups, restore_head_overlay, clone_readout_head
from .qwen_vl_spatial import balanced_region_l1, build_spatial_head
from .qwen_vl_spatial_run import evaluate, masks
from .qwen_vl_turbo_cache import image_float, save_with_digest
from .qwen_vl_turbo_data import pilot_subset, sha256, validate_roles, write_json
from .qwen_vl_turbo_model import pack_hidden
from .qwen_vl_turbo_run import git_identity, noise_for, tensor_image
from .qwen_vl_turbo_losses import crop_regions, region_boxes


def completed_teacher(parent):
    report = json.loads((parent/'report.json').read_text())
    workflow = json.loads((parent/'workflow_report.json').read_text())
    if not report['complete'] or not workflow['complete'] or report['best_step'] <= 0:
        raise ValueError('Wait for the complete electronic baseline, not its current best')
    if sha256(parent/'best_checkpoint.pt') != report['checkpoint_sha256']:
        raise ValueError('Teacher SHA differs from its completed report')
    return report


def load_data(data):
    complete = json.loads((data/'cache_complete.json').read_text())
    audit = json.loads((data/'data_audit.json').read_text())
    if not complete['complete'] or not audit['role_disjoint'] or audit['test_opened']:
        raise ValueError('Reviewed original split audit failed')
    if sha256(data/'frozen_features.pt') != complete['sha256']:
        raise ValueError('Frozen source cache identity changed')
    payload = torch.load(data/'frozen_features.pt', map_location='cpu', weights_only=False, mmap=True)
    validate_roles(payload['rows'])
    expected={'train':960,'val':96}
    if audit.get('protocol')=='rich_product_scene_v1':
        expected={'train':1824,'val':96}
        report=json.loads((data/'report.json').read_text())
        if not report['complete'] or report['pair_sha256']!=sha256(data/'annotated_pairs.json') or audit['expected_counts']!=expected:
            raise ValueError('Rich scene data identity/budget differs')
    if {s: sum(r['split'] == s for r in payload['rows']) for s in ('train', 'val')} != expected:
        raise ValueError('Approved training/development budget differs')
    if audit.get('protocol')!='rich_product_scene_v1':
        payload['rows'] = [{**r, 'noise_seed': 1042} for r in payload['rows']]
    return payload


def load_qwen(path, device):
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    qwen = Qwen3VLForConditionalGeneration.from_pretrained(path, torch_dtype=torch.bfloat16,
                  attn_implementation='sdpa', local_files_only=True).to(device).eval().requires_grad_(False)
    processor = AutoProcessor.from_pretrained(path, local_files_only=True, min_pixels=65536, max_pixels=65536)
    return qwen, processor


def raw_inputs(processor, reference, prompt, layout, device):
    image = tensor_image(reference)
    messages = [{'role': 'user', 'content': [{'type': 'image', 'image': image}, {'type': 'text', 'text': prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = processor(text=[text], images=[image], return_tensors='pt')
    positions = torch.where(inputs['input_ids'][0] == layout['image_token_id'])[0].tolist()
    if positions != layout.get('qwen_image_positions',layout['positions']) or inputs['image_grid_thw'].tolist() != [[1, 16, 16]]:
        raise ValueError('Native Qwen image layout differs')
    if not inputs['attention_mask'].all() or inputs['input_ids'].shape[1] > 224:
        raise ValueError('B1 full prompt without truncation/padding is required')
    return inputs.to(device)


def raw_hidden(qwen, processor, payload, i, layout, device):
    inputs = raw_inputs(processor, payload['reference'][i], payload['rows'][i]['prompt'], layout, device)
    if not layout.get('semantic_prefix',0):
        return qwen.model(**inputs, use_cache=False, return_dict=True).last_hidden_state[0].bfloat16()
    if layout['semantic_prefix']!=64:raise ValueError('Unreviewed native visual prefix')
    native=[]
    handle=qwen.model.visual.register_forward_hook(lambda module,args,out:native.append(out[0]))
    try:
        hidden=qwen.model(**inputs,use_cache=False,return_dict=True).last_hidden_state[0].bfloat16()
    finally:handle.remove()
    if len(native)!=1 or native[0].shape!=(64,2048):raise ValueError('Native visual merger differs')
    return torch.cat([native[0].bfloat16(),hidden],0)


def frontend_input_hash(payload,i):
    image=tensor_image(payload['reference'][i])
    value=hashlib.sha256(image.tobytes()+b'\0'+payload['rows'][i]['prompt'].encode('utf-8'))
    return value.hexdigest()


def stage_cache(a, cfg, payload, parent_state, device):
    if parent_state.get('schema')=='qwen_spatial_language_lora_v1':
        raise ValueError('Adapted language teacher requires an explicitly validated adapted frontend cache; do not reuse the public-Qwen terminal cache')
    if (a.output/'report.json').exists():
        raise ValueError('Existing cache; do not repeat')
    qwen, processor = load_qwen(a.qwen, device)
    captured, handles = {}, []
    def pre(name):
        return lambda module, args: captured.__setitem__(name, args[0].detach().bfloat16().cpu())
    def post(name):
        return lambda module, args, out: captured.__setitem__(name, out.detach().bfloat16().cpu())
    handles.append(qwen.model.visual.blocks[6].register_forward_pre_hook(pre('vision_input')))
    handles.append(qwen.model.language_model.layers[4].register_forward_pre_hook(pre('language_input')))
    for i in ((8,11,14,17,23) if cfg.get('postfree') else (8,11,14,17)):
        handles.append(qwen.model.visual.blocks[i].register_forward_hook(post('vision_'+str(i))))
    for i in ((8,13,18,23,27) if cfg.get('postfree') else (8,13,18,23)):
        handles.append(qwen.model.language_model.layers[i].register_forward_hook(post('language_'+str(i))))
    records = [];vision_pool={};deduplicated=0
    maximum = 0.
    try:
        for i, row in enumerate(payload['rows']):
            captured.clear()
            with torch.no_grad():
                hidden = raw_hidden(qwen, processor, payload, i, parent_state['visual_layout'], device).cpu()
            diff = float((hidden.float()-payload['hidden'][i].float()).abs().max())
            maximum = max(maximum, diff)
            if diff > 1.:
                raise ValueError('Raw Qwen differs from the audited frozen feature cache')
            record = dict(captured)
            for key in list(record):
                if key.startswith('vision_'):
                    record[key] = record[key].unsqueeze(0)
            image_key=hashlib.sha256(tensor_image(payload['reference'][i]).tobytes()).hexdigest()
            if image_key in vision_pool:
                for key,value in vision_pool[image_key].items():
                    if torch.equal(record[key],value):record[key]=value;deduplicated+=1
            else:vision_pool[image_key]={k:v for k,v in record.items() if k.startswith('vision_')}
            record.update(final=hidden, sample_id=row['sample_id'], split=row['split'],input_sha256=frontend_input_hash(payload,i))
            records.append(record)
            if (i+1) % 24 == 0:
                write_json(a.output/'progress.json', {'stage': 'teacher_features', 'done': i+1, 'total': len(payload['rows'])})
    finally:
        for handle in handles:
            handle.remove()
    source_cache_sha=json.loads((a.data/'cache_complete.json').read_text())['sha256']
    digest = save_with_digest({'records': records, 'source_cache_sha256': source_cache_sha}, a.output/'teacher_features.pt')
    write_json(a.output/'report.json', {'complete': True, 'count': len(records), 'sha256': digest,
        'raw_qwen_vs_parent_cache_max_abs': maximum, 'target_or_mask_input': False,
        'input_contract': 'Reference RGB and full edit text only; Qwen original vision24/language28',
        'input_fingerprints_exclude_targets_and_noise':True,'exactly_equal_vision_tensors_shared':deduplicated,
        'qwen_path':str(a.qwen.resolve()),'qwen_config_sha256':sha256(a.qwen/'config.json'),
        'feature_targets_are_teacher_activations': True})


def checked_features(path, payload):
    report = json.loads((path/'report.json').read_text())
    if not report['complete'] or sha256(path/'teacher_features.pt') != report['sha256']:
        raise ValueError('Teacher feature cache incomplete or changed')
    features = torch.load(path/'teacher_features.pt', map_location='cpu', weights_only=False, mmap=True)['records']
    if [f['sample_id'] for f in features] != [r['sample_id'] for r in payload['rows']]:
        raise ValueError('Feature/source sample identity mismatch')
    for i,record in enumerate(features):
        if 'input_sha256' in record:
            if record['input_sha256']!=frontend_input_hash(payload,i):raise ValueError('Teacher RGB/text fingerprint differs')
        elif 'target_only_rebind' in payload:
            raise ValueError('Rebound targets require an input-fingerprinted teacher cache')
    return features


def macro_goals(record, modality, pairs, postfree=False):
    indexes = ((11, 17) if pairs == 1 else (8, 11, 14, 17)) if modality == 'vision' else ((13, 23) if pairs == 1 else (8, 13, 18, 23))
    if postfree:
        indexes=((11,23) if pairs==1 else (8,11,17,23)) if modality=='vision' else ((13,27) if pairs==1 else (8,13,23,27))
    return [record[modality+'_'+str(i)] for i in indexes]


def save_macros(macros, cfg, step, metrics, output, best=False, extra=None):
    state = {'schema': 'qwen_vl_native_hybrid_v1', 'step': step, 'config': cfg,
             'hybrid_config': hybrid_config(cfg['vision_pairs'], cfg['language_pairs']),
             'macros': macros.state_dict(), 'metrics': metrics, 'git': git_identity(), **(extra or {})}
    if cfg.get('postfree'):
        state['hybrid_config'].update(postfree=True,vision_replaced=list(range(6,24)),language_replaced=list(range(4,28)))
    save_with_digest(state, output/('best_checkpoint.pt' if best else 'last_checkpoint.pt'))


def stage_distill(a, cfg, payload, state, device):
    features = checked_features(a.features, payload)
    macros = nn.ModuleDict({m: OpticalMacro(m, w, cfg[m+'_pairs'], cfg['residual_width'])
                          for m, w in (('vision', 1024), ('language', 2048))}).to(device)
    if a.initial:
        initial=torch.load(a.initial/'best_checkpoint.pt',map_location='cpu',weights_only=False)
        load_macro_state(macros,initial['macros'])
    train = [i for i, r in enumerate(payload['rows']) if r['split'] == 'train']
    dev = [i for i, r in enumerate(payload['rows']) if r['split'] == 'val']
    if cfg.get('smoke'):
        names = {r['sample_id'] for r in pilot_subset(payload['rows'], 1)}
        train = [i for i in train if payload['rows'][i]['sample_id'] in names]
        dev = [i for i in dev if payload['rows'][i]['sample_id'] in names]
    optimizer = torch.optim.AdamW(macros.parameters(), lr=cfg['feature_lr'], weight_decay=.001)
    best, best_step, history = math.inf, 0, []
    rng = random.Random(cfg['seed'])
    def loss_for(i):
        values = []
        for modality in ('vision', 'language'):
            macro = macros[modality]
            macro(features[i][modality+'_input'].to(device))
            goals = macro_goals(features[i], modality, macro.pairs,cfg.get('postfree',False))
            distance=directional_feature_loss if cfg.get('directional_feature_loss') else feature_loss
            values.append(sum(distance(x, y.to(device)) for x, y in zip(macro.last_stages, goals))/len(goals))
        return sum(values)+cfg['balance_weight']*sum(m.balance_loss() for m in macros.values())
    for step in range(1, cfg['feature_steps']+1):
        macros.train(); optimizer.zero_grad(set_to_none=True)
        loss = loss_for(rng.choice(train))
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite feature loss')
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(macros.parameters(),1.)
        if not torch.isfinite(norm):raise FloatingPointError('Nonfinite feature-distillation gradient')
        if step==4:
            phase=sum(float(p.grad.detach().abs().sum()) for n,p in macros.named_parameters() if 'raw_phase' in n and p.grad is not None)
            router=sum(float(p.grad.detach().abs().sum()) for n,p in macros.named_parameters() if 'raw_router_phase' in n and p.grad is not None)
            write_json(a.output/'smoke_report.json',{'real_backward_steps':4,'finite_loss_gradient':True,'TRAIN_only':True,
                'phase_gradient_absolute_sum':phase,'router_phase_gradient_absolute_sum':router,'generation_teacher_images_used':False})
        optimizer.step()
        if step % 10 == 0:
            write_json(a.output/'progress.json', {'stage': 'feature_distillation', 'step': step, 'total_steps': cfg['feature_steps'], 'train_loss': float(loss.detach()), 'best_step': best_step})
        if step % cfg['feature_eval_every'] == 0 or step == cfg['feature_steps']:
            macros.eval()
            with torch.no_grad():
                metric = sum(float(loss_for(i)) for i in dev)/len(dev)
            entry = {'step': step, 'train_sample_loss': float(loss.detach()), 'development_feature_loss': metric}
            history.append(entry)
            save_macros(macros, cfg, step, entry, a.output)
            if metric < best:
                best, best_step = metric, step
                save_macros(macros, cfg, step, entry, a.output, best=True)
            write_json(a.output/'history.json', history)
            # Raw intermediate amplitudes need not remain stable under detached
            # fusion scales. Do not spend the rest of the budget on divergence.
            patience = cfg.get('feature_early_stop_evaluations', 0)
            if patience and step-best_step >= patience*cfg['feature_eval_every'] and metric > best*1.2:
                write_json(a.output/'feature_early_stop.json', {'step': step, 'best_step': best_step,
                    'reason': 'Sustained train/development feature-loss deterioration; retain best, do not call it convergence'})
                break
    write_json(a.output/'report.json', {'complete': True, 'best_step': best_step, 'best_development_feature_loss': best,
       'evaluation_label': 'development_test', 'evaluation_used_for_selection': True,
       'test_gradient_updates': False, 'alpha_minimum': .60, 'checkpoint_sha256': sha256(a.output/'best_checkpoint.pt')})


@torch.no_grad()
def evaluate_hybrid(qwen, processor, head, payload, ids, layout, device, output, label, metric, export=False):
    qwen.eval(); head.eval()
    temporary = dict(payload)
    temporary['hidden'] = list(payload['hidden'])
    routes = []
    for i in ids:
        temporary['hidden'][i] = raw_hidden(qwen, processor, payload, i, layout, device).cpu()
        routes.append({'sample_id': payload['rows'][i]['sample_id'], **{m: v.diagnostics() for m, v in qwen.model.hybrid.items()}})
    result = evaluate(head, temporary, ids, device, output, label, metric, export=export)
    result.update(evaluation_label='development_test' if all(payload['rows'][i]['split'] == 'val' for i in ids) else 'train_diagnostic',
                  evaluation_used_for_selection=True, independent_test_claim=False)
    write_json(output/(label+'.json'), result)
    write_json(output/(label+'_routing.json'), routes)
    return result


def quality_gate(actual, baseline):
    if {r['sample_id'] for r in actual['images']} != {r['sample_id'] for r in baseline['images']}:
        return {'evaluated': False, 'reason': 'Cohorts differ (smoke is not a formal performance comparison)',
                'passed_perceptual': False}
    cells = {}
    for mode in ('background', 'object', 'joint'):
        lpips_ratio = actual['lpips_by_mode'][mode]/baseline['lpips_by_mode'][mode]
        crop_ratio = actual['object_crop_lpips_by_mode'][mode]/baseline['object_crop_lpips_by_mode'][mode]
        cells[mode] = {'lpips_ratio': lpips_ratio, 'object_crop_lpips_ratio': crop_ratio,
                       'passed_perceptual': max(lpips_ratio, crop_ratio) <= 1.05}
    return {'evaluated': True, 'by_mode': cells, 'passed_perceptual': all(x['passed_perceptual'] for x in cells.values()),
            'speed_gate_evaluated': False, 'semantic_clip_gate_evaluated': False,
            'scope': 'Development TEST-selected; no independent generalization claim'}


def router_batch_balance(macros, features, ids, device, mi_weight=.25):
    """Same-parameter TRAIN batch, not an EMA across changing model versions.

    Only the existing optical router/field encoder is used. This supplemental
    TRAIN loss never adds a deployment router or forces per-example uniformity.
    """
    if not ids or any(features[i]['split'] != 'train' for i in ids):
        raise ValueError('Router calibration gradients are TRAIN-only')
    losses = []
    for modality, macro in macros.items():
        rows = [features[i][modality+'_input'][0] for i in ids]
        fields=[]
        # Encode each actual token length before batching physical fields. A
        # padded token sent through a biased projection need not remain zero.
        for row in rows:
            packed=macro.codecs[0].pack(row[None].to(device))
            padding=torch.zeros(packed.shape[:2],dtype=torch.bool,device=device)
            field,_=macro.paths[0]._encode_input_fields(packed,padding)
            fields.append(field)
        fields=torch.cat(fields,0)
        routing = macro.paths[0].core.router(fields)
        p = routing['probabilities'].float().clamp_min(1e-8)
        mean = p.mean(0)
        load = routing['selected_mask'].float().mean(0).detach()/2
        importance = 4*mean.square().sum()-1
        hard_load = 4*(mean*load).sum()-1
        information = (p*(p.log()-mean.clamp_min(1e-8).log())).sum(-1).mean()
        losses.append(importance+hard_load-mi_weight*information)
    return sum(losses)


def stage_joint(a, cfg, payload, state, device):
    if state.get('schema')=='qwen_spatial_language_lora_v1':
        raise ValueError('Adapted teacher terminal-feature protocol must be implemented and validated before LG recovery')
    features = checked_features(a.features, payload)
    qwen, processor = load_qwen(a.qwen, device)
    macros = apply_configured_hybrid(qwen,cfg)
    init = torch.load(a.initial/'best_checkpoint.pt', map_location='cpu', weights_only=False)
    compatibility = load_macro_state(macros, init['macros'])
    for macro in macros.values():
        macro.hard_load_balance_weight = cfg.get('hard_load_balance_weight', 0.)
    layout = state['visual_layout']
    if state['config'].get('semantic_prefix', 0):
        raise ValueError('This native fused-only protocol does not silently add a visual prefix')
    head = build_spatial_head(a.sd, state, state['config'], layout['positions'], device,
                              drop_vae_encoder=True).eval().requires_grad_(False)
    teacher_head=head
    if cfg.get('adapt_head',False):
        head=clone_readout_head(teacher_head,trainable=True)
        restore_head_overlay(head,init);head.eval()
    import lpips
    metric = lpips.LPIPS(net='alex').to(device).eval().requires_grad_(False)
    train = [i for i, r in enumerate(payload['rows']) if r['split'] == 'train']
    dev = [i for i, r in enumerate(payload['rows']) if r['split'] == 'val']
    names = {r['sample_id'] for r in pilot_subset(payload['rows'], 2)}
    train_diagnostic = [i for i in train if payload['rows'][i]['sample_id'] in names]
    if cfg.get('smoke'):
        train, dev = train_diagnostic[:6], [i for i in dev if payload['rows'][i]['sample_id'] in names][:6]
    before = evaluate_hybrid(qwen, processor, head, payload, dev, layout, device, a.output, 'dev_before', metric)
    teacher_report = completed_teacher(a.parent)
    groups=optimizer_groups(macros,cfg)
    if cfg.get('adapt_head',False):
        groups.append({'params':list(head.adapter.parameters())+list(head.spatial.parameters()),
            'lr':cfg['head_lr'],'base_lr':cfg['head_lr'],'name':'existing_head_readouts'})
    optimizer = torch.optim.AdamW(groups, weight_decay=.001)
    trainable=[p for group in groups for p in group['params']]
    def extra_state():
        extra={'teacher_checkpoint_sha256':sha256(a.parent/'best_checkpoint.pt')}
        if cfg.get('adapt_head',False):
            extra['head_overlay']={name:{k:v.detach().cpu().clone() for k,v in getattr(head,name).state_dict().items()}
                                   for name in ('adapter','spatial')}
        return extra
    best, best_step, history, rng = math.inf, 0, [], random.Random(cfg['seed'])
    steps = cfg.get('joint_steps') or math.ceil(len(train)*cfg['joint_epochs']/cfg['gradient_accumulation'])
    order = []
    while len(order) < steps*cfg['gradient_accumulation']:
        block = train.copy(); rng.shuffle(block); order.extend(block)
    for step in range(1, steps+1):
        macros.train(); optimizer.zero_grad(set_to_none=True); total = 0.
        warmup = cfg.get('joint_warmup_steps', 0)
        factor = (.1+.9*.5*(1+math.cos(math.pi*step/steps)))*min(1., max(.1, step/max(1, warmup))) if warmup else .1+.9*.5*(1+math.cos(math.pi*step/steps))
        for group in optimizer.param_groups:
            group['lr'] = group['base_lr']*factor
        for micro in range(cfg['gradient_accumulation']):
            i = order[(step-1)*cfg['gradient_accumulation']+micro]
            row = payload['rows'][i]
            # Electronic teacher output is only a TRAIN loss target, never a
            # student input or an inference image retrieved from a catalogue.
            with torch.no_grad():
                teacher_hidden, teacher_valid = pack_hidden([features[i]['final']], device)
                teacher_z, teacher_rgb = teacher_head(teacher_hidden, teacher_valid, noise_for([row], device))
            hidden = raw_hidden(qwen, processor, payload, i, layout, device)
            valid = torch.ones(1, len(hidden), dtype=torch.bool, device=device)
            z, rgb = head(hidden[None], valid, noise_for([row], device))
            target = image_float(payload['target'][i:i+1], device)
            edit, foreground = masks(row, device)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                distance=directional_feature_loss if cfg.get('directional_feature_loss') else feature_loss
                h_loss = distance(hidden, features[i]['final'].to(device))
                z_loss = F.mse_loss(z.float(), teacher_z.float())
                rgb_loss = balanced_region_l1(rgb, target, edit)
                p_loss = metric(rgb.clamp(-1, 1), target).mean()
                union = masks({**row, 'mode': 'object'}, device)[0]
                boxes = region_boxes(union)
                crop_loss = metric(crop_regions(rgb.clamp(-1, 1), boxes), crop_regions(target, boxes)).mean()
                image_kd = (rgb.float()-teacher_rgb.float()).abs().mean()
                balance = sum(m.balance_loss() for m in macros.values())
                loss = (cfg['hidden_weight']*h_loss+cfg['latent_kd_weight']*z_loss+
                        cfg['rgb_weight']*rgb_loss+cfg['lpips_weight']*p_loss+
                        cfg['rgb_kd_weight']*image_kd+cfg.get('crop_lpips_weight', .2)*crop_loss+cfg['balance_weight']*balance)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite joint loss')
            (loss/cfg['gradient_accumulation']).backward()
            total += float(loss.detach())/cfg['gradient_accumulation']
        calibration = None
        if cfg.get('router_calibration_every') and step % cfg['router_calibration_every'] == 0:
            selected_train = rng.sample(train, min(len(train), cfg.get('router_calibration_batch', 18)))
            calibration = router_batch_balance(macros, features, selected_train, device,
                                               cfg.get('router_mutual_information_weight', .25))
            (cfg.get('router_calibration_weight', .05)*calibration).backward()
        norm = torch.nn.utils.clip_grad_norm_(trainable, 1.)
        if not torch.isfinite(norm):
            raise FloatingPointError('Nonfinite hybrid gradient')
        optimizer.step()
        if step % 10 == 0:
            write_json(a.output/'progress.json', {'stage': 'raw_hybrid_joint', 'step': step, 'total_steps': steps,
                'epoch': step*cfg['gradient_accumulation']/len(train), 'train_loss': total, 'best_step': best_step,
                'learning_rates': {g['name']: g['lr'] for g in optimizer.param_groups},
                'train_router_calibration_loss': float(calibration.detach()) if calibration is not None else None,
                'alpha': {m: [float(f.alpha.detach()) for f in v.fusions] for m, v in macros.items()}})
        if step % cfg['joint_eval_every'] == 0 or step == steps:
            val = evaluate_hybrid(qwen, processor, head, payload, dev, layout, device, a.output, 'dev_latest', metric)
            tr = evaluate_hybrid(qwen, processor, head, payload, train_diagnostic, layout, device, a.output, 'train_latest', metric)
            score = val['mean_lpips']+cfg['crop_selection_weight']*val['mean_object_crop_lpips']
            entry = {'step': step, 'epoch': step*cfg['gradient_accumulation']/len(train), 'train_sample_loss': total,
                'train_lpips': tr['mean_lpips'], 'dev_lpips': val['mean_lpips'], 'dev_crop_lpips': val['mean_object_crop_lpips'],
                'dev_psnr': val['mean_per_image_psnr_db'], 'dev_by_mode': val['lpips_by_mode'],
                'dev_crop_by_mode': val['object_crop_lpips_by_mode'], 'dev_category_mode': val['by_category_mode'],
                'quality_gate': quality_gate(val, teacher_report['best_val'])}
            history.append(entry)
            save_macros(macros, cfg, step, entry, a.output, extra=extra_state())
            if score < best:
                best, best_step = score, step
                save_macros(macros, cfg, step, entry, a.output, best=True,
                            extra=extra_state())
            write_json(a.output/'history.json', history)
            # Stop sustained late deterioration; retain best/last, not the last
            # epoch as a purported optimum. Only after at least five evaluations.
            if cfg.get('early_stop_evaluations') and len(history) >= 5 and step-best_step >= cfg['early_stop_evaluations']*cfg['joint_eval_every']:
                break
    selected = torch.load(a.output/'best_checkpoint.pt', map_location='cpu', weights_only=False)
    macros.load_state_dict(selected['macros'])
    restore_head_overlay(head,selected)
    val = evaluate_hybrid(qwen, processor, head, payload, dev, layout, device, a.output, 'dev_best', metric, export=True)
    write_json(a.output/'report.json', {'complete': True, 'best_step': best_step, 'best_val': val,
        'before': before, 'teacher_checkpoint_sha256': sha256(a.parent/'best_checkpoint.pt'),
        'checkpoint_sha256': sha256(a.output/'best_checkpoint.pt'), 'alpha_minimum': .60,
        'quality_gate': quality_gate(val, teacher_report['best_val']), 'evaluation_label': 'development_test',
        'initial_checkpoint_sha256': sha256(a.initial/'best_checkpoint.pt'), 'checkpoint_compatibility': compatibility,
        'evaluation_used_for_selection': True, 'test_gradient_updates': False,
        'inference_parameters_including_embedding': sum(p.numel() for p in qwen.model.parameters())+sum(p.numel() for p in head.parameters()),
        'macro_parameters': sum(p.numel() for p in macros.parameters()),
        'quality_accepted': False, 'note': 'Requires native prompt-labelled visual review, semantic metrics and matched timing before any success claim.'})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=('cache', 'distill', 'joint'), required=True)
    for name in ('config', 'data', 'parent', 'qwen', 'sd', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--features', type=Path)
    p.add_argument('--initial', type=Path)
    a = p.parse_args(); cfg = json.loads(a.config.read_text())
    if os.environ.get('CUDA_VISIBLE_DEVICES') != cfg['gpu_uuid']:
        raise ValueError('Unauthorized GPU UUID')
    if a.output.exists() and (a.output/'execution.json').exists():
        raise ValueError('Existing run; inspect instead of duplicate launch')
    a.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4); torch.manual_seed(cfg['seed']); random.seed(cfg['seed'])
    torch.cuda.set_per_process_memory_fraction(cfg['gpu_memory_fraction'], 0)
    completed_teacher(a.parent)
    payload = load_data(a.data)
    state = torch.load(a.parent/'best_checkpoint.pt', map_location='cpu', weights_only=False)
    write_json(a.output/'execution.json', {'argv': sys.argv, 'pid': os.getpid(), 'git': git_identity(),
       'started_unix': time.time(), 'config': cfg, 'gpu_uuid': cfg['gpu_uuid'],
       'source_sha256': {n: sha256(Path(__file__).with_name(n)) for n in ('qwen_vl_hybrid.py', 'qwen_vl_hybrid_run.py')},
       'evaluation_label': 'development_test', 'evaluation_used_for_selection': True,
       'teacher_checkpoint_sha256': sha256(a.parent/'best_checkpoint.pt')})
    write_json(a.output/'config.json', cfg)
    try:
        {'cache': stage_cache, 'distill': stage_distill, 'joint': stage_joint}[a.stage](a, cfg, payload, state, torch.device('cuda:0'))
    except Exception:
        write_json(a.output/'failure.json', {'stage': a.stage, 'traceback': traceback.format_exc()}); raise


if __name__ == '__main__':
    main()
