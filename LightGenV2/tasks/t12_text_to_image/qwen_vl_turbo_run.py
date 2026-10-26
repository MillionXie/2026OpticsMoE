"""Reproducible electronic feasibility pilot, TRAIN/VAL only, one shared GPU.

Run with --stage prepare|cache|train|all. Only public pretrained backbones are
loaded. Run-local Qwen features are a frozen-compute cache, not a lookup used by
the final architecture. Every cached model input is reference RGB + instruction.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
import traceback

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.nn import functional as F

from .qwen_vl_turbo_data import (add_descriptions, pilot_subset, prepare_pairs,
                                  render_pair, sha256, validate_roles, write_json, normalized_product)
from .qwen_vl_turbo_model import (NativeTurboHead, QueryResampler, encode_reference,
                                   image_tensor, pack_hidden, region_weighted_loss)
from .qwen_vl_turbo_training import configure_unet_training
from .qwen_vl_turbo_cache import image_float, save_with_digest


def git_identity() -> dict:
    root = Path(__file__).resolve().parents[3]
    def git(*args):
        return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()
    files = [Path(__file__), Path(__file__).with_name('qwen_vl_turbo_model.py'),
             Path(__file__).with_name('qwen_vl_turbo_data.py'),
             Path(__file__).with_name('qwen_vl_turbo_training.py'),
             Path(__file__).with_name('qwen_vl_turbo_cache.py'),
             Path(__file__).with_name('qwen_vl_turbo_losses.py'),
             Path(__file__).with_name('product_scene_replace_data.py'),
             Path(__file__).with_name('product_object_replace_data.py'),
             Path(__file__).with_name('product_scene_data.py')]
    return {'head': git('rev-parse', 'HEAD'), 'branch': git('branch', '--show-current'),
            'source_sha256': {str(p.relative_to(root)): sha256(p) for p in files},
            'published_source_commit': os.environ.get('T12_SOURCE_COMMIT', git('rev-parse', 'HEAD'))}


def pretrained_manifest(path: Path) -> dict:
    files = sorted(p for p in path.rglob('*') if p.is_file() and p.suffix in {'.json', '.safetensors'})
    return {str(p.relative_to(path)): {'bytes': p.stat().st_size, 'sha256': sha256(p)} for p in files}


def flush_gpu():
    gc.collect()
    torch.cuda.empty_cache()


def noise_for(rows, device, offset=0):
    return torch.cat([torch.randn(1, 4, 32, 32, device=device,
                       generator=torch.Generator(device=device).manual_seed(r['noise_seed'] + offset))
                      for r in rows])


def prepare(args, cfg):
    manifest = args.output / 'pairs_manifest.json'
    if manifest.exists():
        payload = json.loads(manifest.read_text())
        if payload['config'] != cfg:
            raise ValueError('Existing pair manifest has a different config; choose a new run')
        for split, digest in payload['audit']['raw_manifest_sha256'].items():
            if sha256(args.data / f'{split}.jsonl') != digest:
                raise ValueError('Raw manifest changed')
        validate_roles(payload['pairs'])
    else:
        prepare_pairs(args.data, manifest, cfg)
    payload = json.loads(manifest.read_text())
    selected = (list(payload['pairs']) if cfg.get('use_all_pairs',False) else
                pilot_subset(payload['pairs'], cfg['pilot_pairs_per_category_mode']))
    if cfg.get('explicit_category_teacher',False):
        selected=[{**r,'explicit_category_teacher':True} for r in selected]
    validate_roles(selected)
    write_json(args.output / 'pilot_pairs.json', selected)
    print(json.dumps({'stage': 'prepared', 'audit': payload['audit'],
                      'pilot_pairs': len(selected)}), flush=True)
    return selected


def cache(args, cfg, rows, device):
    path = args.output / 'frozen_features.pt'
    if path.exists():
        payload = torch.load(path, map_location='cpu', weights_only=False)
        if payload['pilot_sha256'] != sha256(args.output / 'pilot_pairs.json'):
            raise ValueError('Feature cache does not match selected pairs')
        return sha256(path)
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    # Public Qwen weights, all vision and all 28 language blocks preserved.
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.qwen, torch_dtype=torch.bfloat16, local_files_only=True,
        attn_implementation='sdpa').to(device).eval().requires_grad_(False)
    processor = AutoProcessor.from_pretrained(args.qwen, local_files_only=True,
                                              min_pixels=65536, max_pixels=65536)
    qwen_parameters = sum(p.numel() for p in model.parameters())
    qwen_model_parameters = sum(p.numel() for p in model.model.parameters())
    embedding_parameters = model.model.language_model.embed_tokens.weight.numel()
    desc_path = args.output / 'product_descriptions.json'
    descriptions = json.loads(desc_path.read_text()) if desc_path.exists() else {}
    products = {r[role]['sample_id']: r[role] for r in rows for role in ('source', 'target')}
    for sid, product in products.items():
        if sid in descriptions:
            continue
        with Image.open(product['image_path']) as im:
            im = im.convert('RGB')
            messages = [{'role': 'user', 'content': [{'type': 'image', 'image': im},
                        {'type': 'text', 'text': 'Describe only the central product in one short noun phrase, at most 25 words. State its category, visible shape, main color, material and viewpoint. No background, speculation or brand.'}]}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                 enable_thinking=False)
            inputs = processor(text=[text], images=[im], return_tensors='pt').to(device)
            with torch.inference_mode():
                tokens = model.generate(**inputs, max_new_tokens=48, do_sample=False)
            description = processor.batch_decode(tokens[:, inputs['input_ids'].shape[1]:],
                                                   skip_special_tokens=True)[0].strip().strip('"').rstrip('.')
        descriptions[sid] = description
        write_json(desc_path, descriptions)
        print('description', sid, description, flush=True)
        write_json(args.output/'progress.json',{'stage':'product_annotation','done':len(descriptions),
                   'needed':len(products)})
    annotated = add_descriptions(rows, descriptions)
    write_json(args.output / 'annotated_pairs.json', annotated)
    tensors, hidden = [], []
    preview_ids={r['sample_id'] for r in pilot_subset(annotated,2)} if cfg.get('use_all_pairs') else {r['sample_id'] for r in annotated}
    for i, row in enumerate(annotated):
        reference, target = render_pair(row, cfg['image_size'])
        if cfg.get('uint8_image_cache',False):
            tensors.append(tuple(torch.from_numpy(np.asarray(im).copy()).permute(2,0,1) for im in (reference,target)))
        else:
            tensors.append((image_tensor(reference),image_tensor(target)))
        hidden.append(encode_reference(model, processor, reference, row['prompt'], device))
        (args.output / 'data_preview').mkdir(exist_ok=True)
        if row['sample_id'] in preview_ids:
            reference.save(args.output / 'data_preview' / f'{i:03d}_input.png')
            target.save(args.output / 'data_preview' / f'{i:03d}_target.png')
        print('multimodal_features', i+1, '/', len(rows), list(hidden[-1].shape), flush=True)
        if (i+1)%20==0:
            write_json(args.output/'progress.json',{'stage':'multimodal_features','done':i+1,'total':len(rows)})
    # Demonstrate BOTH modalities reach Qwen's fused hidden representation.
    reference, _ = render_pair(annotated[0])
    blank = encode_reference(model, processor, Image.new('RGB', (256, 256), (127,127,127)),
                             annotated[0]['prompt'], device)
    changed = encode_reference(model, processor, reference,
                               'Replace only the background with a bright warm room.', device)
    base = hidden[0].float().mean(0)
    controls = {'blank_reference_relative_change': float((blank.float().mean(0)-base).norm()/base.norm()),
                'changed_instruction_relative_change': float((changed.float().mean(0)-base).norm()/base.norm()),
                'vision_blocks': len(model.model.visual.blocks),
                'language_blocks': len(model.model.language_model.layers),
                'qwen_loaded_parameters': qwen_parameters,
                'qwen_forward_parameters': qwen_model_parameters,
                'qwen_embedding_parameters': embedding_parameters,
                'no_token_or_layer_pruning': True}
    if min(controls['blank_reference_relative_change'], controls['changed_instruction_relative_change']) <= 1e-5:
        raise ValueError('A required conditioning modality appears disconnected')
    write_json(args.output / 'qwen_modality_checks.json', controls)
    del model, processor
    flush_gpu()
    from transformers import CLIPTextModel, CLIPTokenizer
    clip = CLIPTextModel.from_pretrained(args.sd, subfolder='text_encoder', variant='fp16',
                                        torch_dtype=torch.bfloat16, local_files_only=True).to(device).eval()
    tokenizer = CLIPTokenizer.from_pretrained(args.sd, subfolder='tokenizer', local_files_only=True)
    train_indices = [i for i,r in enumerate(annotated) if r['split'] == 'train']
    train_captions = [annotated[i]['teacher_caption'] for i in train_indices]
    teacher = {}
    with torch.inference_mode():
        batch=cfg.get('clip_cache_batch_size',1)
        for start in range(0,len(train_indices),batch):
            current=train_indices[start:start+batch]
            captions=train_captions[start:start+batch]
            ids=tokenizer(captions,padding='max_length',max_length=77,truncation=True,return_tensors='pt').input_ids.to(device)
            encoded=clip(ids).last_hidden_state.bfloat16().cpu()
            for i,value in zip(current,encoded):teacher[i]=value.contiguous()
    del clip
    flush_gpu()
    digest=save_with_digest({'pilot_sha256': sha256(args.output / 'pilot_pairs.json'), 'rows': annotated,
                'hidden': hidden, 'reference': torch.stack([t[0] for t in tensors]),
                'target': torch.stack([t[1] for t in tensors]), 'train_clip_teacher': teacher,
                'qwen_checks': controls,
                'clip_scope': 'frozen TRAIN-only native text-condition alignment teacher'}, path)
    write_json(args.output/'cache_identity.json',{'sha256':digest,'bytes':path.stat().st_size,
               'images_uint8':cfg.get('uint8_image_cache',False),'stream_hash':True})
    return digest


def tensor_image(tensor):
    if tensor.dtype==torch.uint8:
        return Image.fromarray(tensor.cpu().permute(1,2,0).numpy())
    return Image.fromarray(((tensor.detach().cpu().clamp(-1,1)+1)*127.5).round().byte().permute(1,2,0).numpy())


@torch.no_grad()
def evaluate(model, payload, indices, device, output, label, ablation=False, native_teacher=False, export_png=False, metric_network=None):
    model.eval()
    errors, latent_errors = [], []
    cell = 256
    sheet = Image.new('RGB', (cell*3, (cell+42)*len(indices)), 'white')
    draw = ImageDraw.Draw(sheet)
    for pos, i in enumerate(indices):
        hidden, valid = pack_hidden([payload['hidden'][i]], device)
        if ablation:
            other = indices[(pos+1) % len(indices)]
            hidden, valid = pack_hidden([payload['hidden'][other]], device)
        row = payload['rows'][i]
        if native_teacher:
            if row['split'] != 'train':
                raise ValueError('Native caption-teacher probe is TRAIN diagnostic only')
            latent, prediction = model.generate_from_conditions(
                payload['train_clip_teacher'][i][None].to(device), noise_for([row], device))
        else:
            latent, prediction = model(hidden, valid, noise_for([row], device))
        prediction = prediction.clamp(-1,1)
        target = image_float(payload['target'][i:i+1],device)
        mse = (prediction-target).square().mean().item()/4
        psnr = -10*np.log10(max(mse,1e-12))
        errors.append({'sample_id': row['sample_id'], 'mode': row['mode'], 'category': row['category'],
                       'psnr': psnr, 'mse_01': mse})
        if metric_network is not None:
            errors[-1]['lpips']=float(metric_network(prediction.float(),target.float()).mean())
        if export_png:
            for role,value in (('reference',payload['reference'][i]),('target',payload['target'][i]),
                               ('generated',prediction[0])):
                directory=output/'images'/label/role
                directory.mkdir(parents=True,exist_ok=True)
                tensor_image(value).save(directory/f'sample_{pos:04d}.png')
        for col, image in enumerate((payload['reference'][i], payload['target'][i], prediction[0])):
            sheet.paste(tensor_image(image), (col*cell, pos*(cell+42)+42))
        draw.text((6,pos*(cell+42)+4), f'{label} {row["split"]} {row["category"]}/{row["mode"]} PSNR {psnr:.2f} dB', fill='black')
        draw.text((6,pos*(cell+42)+20), 'REFERENCE                           TARGET                              GENERATED', fill='black')
    sheet.save(output / f'{label}.jpg', quality=94)
    mean = float(np.mean([r['psnr'] for r in errors]))
    report = {'mean_per_image_psnr_db': mean, 'count': len(errors), 'images': errors,
              'by_mode': {m: float(np.mean([r['psnr'] for r in errors if r['mode']==m]))
                          for m in ('background','object','joint')},
              'scope': ('TRAIN native target-caption-teacher diagnostic, NOT a Qwen inference result'
                        if native_teacher else 'development pilot TRAIN or VAL, never independent TEST')}
    if metric_network is not None:
        report['mean_lpips']=float(np.mean([r['lpips'] for r in errors]))
        report['lpips_by_mode']={m:float(np.mean([r['lpips'] for r in errors if r['mode']==m]))
                                 for m in ('background','object','joint')}
    write_json(output / f'{label}.json', report)
    return report


def train(args, cfg, device, cache_digest=None):
    from diffusers import AutoencoderKL, EulerDiscreteScheduler, UNet2DConditionModel
    from peft import LoraConfig
    import lpips
    payload = torch.load(args.output / 'frozen_features.pt', map_location='cpu', weights_only=False)
    if payload['pilot_sha256']!=sha256(args.output/'pilot_pairs.json'):
        raise ValueError('Training cache inputs do not match this run selection')
    cache_digest=cache_digest or sha256(args.output/'frozen_features.pt')
    validate_roles(payload['rows'])
    if 'shared_noise_seed' in cfg:
        # Same noise across all pairs forces differences to come from Qwen's
        # RGB/text condition. This pilot does not claim seed diversity.
        for row in payload['rows']: row['noise_seed']=int(cfg['shared_noise_seed'])
    train_ids = [i for i,r in enumerate(payload['rows']) if r['split']=='train']
    val_ids = [i for i,r in enumerate(payload['rows']) if r['split']=='val']
    train_eval_ids=train_ids
    if cfg.get('use_all_pairs',False):
        evaluation_names={r['sample_id'] for r in pilot_subset(payload['rows'],2) if r['split']=='train'}
        train_eval_ids=[i for i in train_ids if payload['rows'][i]['sample_id'] in evaluation_names]
    adapter = QueryResampler(hidden=cfg['adapter_hidden'], layers=cfg['adapter_layers']).to(device)
    resumed = None
    if args.resume:
        resumed = torch.load(args.resume, map_location='cpu', weights_only=False)
        if resumed['cache_sha256'] != cache_digest:
            if not args.allow_expanded_train:
                raise ValueError('Resume weights refer to a different data/conditioning cache')
            parent_cache=args.resume.parent/'frozen_features.pt'
            if sha256(parent_cache)!=resumed['cache_sha256']:
                raise ValueError('Parent conditioning cache fingerprint mismatch')
            parent=torch.load(parent_cache,map_location='cpu',weights_only=False)
            validate_roles(parent['rows']+payload['rows'])
            child={r['sample_id']:r for r in payload['rows']}
            for previous in parent['rows']:
                current=child.get(previous['sample_id'])
                if current is None:raise ValueError('Expanded run omitted a parent TRAIN/VAL pair')
                for field in ('split','prompt','source_scene','target_scene'):
                    if previous[field]!=current[field]:raise ValueError('Parent model input changed: '+field)
                for role in ('source','target'):
                    if previous[role]['rgba_sha256']!=current[role]['rgba_sha256']:
                        raise ValueError('Parent source/target pixels changed')
            write_json(args.output/'expanded_resume_audit.json',{'parent_pairs':len(parent['rows']),
                       'child_pairs':len(payload['rows']),'parent_inputs_and_roles_preserved':True,
                       'train_val_roles_disjoint':True,'test_used':False})
            del parent
        adapter.load_state_dict(resumed['adapter'])
        write_json(args.output/'resume_identity.json',{'checkpoint':str(args.resume),
                   'sha256':sha256(args.resume),'source_step':resumed['step'],
                   'source_git':resumed['git'],'optimizer_restarted':True})
    opt = torch.optim.AdamW(adapter.parameters(), lr=cfg['alignment_lr'])
    # Match native UNet condition statistics using TRAIN only, never fit VAL.
    with torch.no_grad():
        mean = torch.stack(list(payload['train_clip_teacher'].values())).float().mean((0,1)).to(device)
        if resumed is None: adapter.output.bias.copy_(mean)
    rng = random.Random(cfg['seed'])
    do_align=resumed is None or cfg.get('realign_expanded_train',False)
    for step in range(cfg['alignment_steps'] if do_align else 0):
        ids = rng.sample(train_ids, min(cfg.get('alignment_batch_size',6),len(train_ids)))
        h, valid = pack_hidden([payload['hidden'][i] for i in ids], device)
        target = torch.stack([payload['train_clip_teacher'][i] for i in ids]).float().to(device)
        loss = F.mse_loss(adapter(h,valid), target)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(),1)
        opt.step()
        if (step+1)%40==0:
            print('alignment',step+1,'loss',float(loss.detach()),flush=True)
            write_json(args.output / 'progress.json', {'stage':'alignment','step':step+1,'loss':float(loss.detach())})
    with torch.no_grad():
        alignment_sum=0.;alignment_count=0
        for begin in range(0,len(train_ids),16):
            ids=train_ids[begin:begin+16]
            h,valid=pack_hidden([payload['hidden'][i] for i in ids],device)
            target=torch.stack([payload['train_clip_teacher'][i] for i in ids]).float().to(device)
            alignment_sum+=float(F.mse_loss(adapter(h,valid),target))*len(ids)
            alignment_count+=len(ids)
        alignment_mse=alignment_sum/alignment_count
    write_json(args.output/'alignment_report.json',{'train_condition_mse':alignment_mse,
               'steps':cfg['alignment_steps'] if do_align else 0,
               'resumed_adapter':resumed is not None,'no_val_fit':True})
    del opt
    unet = UNet2DConditionModel.from_pretrained(args.sd, subfolder='unet', variant='fp16',
                    torch_dtype=torch.bfloat16, local_files_only=True).to(device).requires_grad_(False)
    unet.add_adapter(LoraConfig(r=cfg['unet_lora_rank'],lora_alpha=cfg['unet_lora_rank'],
                               target_modules=['to_q','to_k','to_v','to_out.0'],init_lora_weights=True))
    configure_unet_training(unet,cfg.get('unet_training','lora'))
    for p in unet.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    if resumed is not None:
        names={n for n,p in unet.named_parameters() if p.requires_grad}
        if not set(resumed['unet_trainable']).issubset(names):
            raise ValueError('Resume LoRA parameter names changed')
        with torch.no_grad():
            for name,p in unet.named_parameters():
                if name in resumed['unet_trainable']: p.copy_(resumed['unet_trainable'][name].to(device))
    unet.enable_gradient_checkpointing()
    vae = AutoencoderKL.from_pretrained(args.sd, subfolder='vae', variant='fp16',
                    torch_dtype=torch.bfloat16, local_files_only=True).to(device).eval().requires_grad_(False)
    scheduler = EulerDiscreteScheduler.from_pretrained(args.sd,subfolder='scheduler', local_files_only=True)
    model = NativeTurboHead(unet,vae,scheduler,adapter)
    # Encoder is used solely for supervised targets and VAE diagnostic, not reference conditioning.
    targets_latent, target_regions, vae_mse = {}, {}, []
    gpu_regions={}
    target_boxes={}
    has_clip=any(cfg.get(k,0)>0 for k in ('foreground_clip_text_weight','foreground_clip_image_weight','global_clip_text_weight'))
    has_gan=cfg.get('gan_weight',0)>0
    if has_clip or has_gan:
        from .qwen_vl_turbo_losses import (FrozenCLIPSemantic,PatchDiscriminator,region_boxes,
                  crop_regions,discriminator_views,hinge_discriminator_loss)
    diagnostic_ids=set(train_eval_ids+val_ids)
    if cfg.get('use_all_pairs',False):
        names={r['sample_id'] for r in pilot_subset(payload['rows'],2)}
        diagnostic_ids={i for i,r in enumerate(payload['rows']) if r['sample_id'] in names}
    with torch.no_grad():
        for i in train_ids + val_ids:
            t = image_float(payload['target'][i:i+1],device,vae.dtype)
            z = vae.encode(t).latent_dist.mode()
            if i in diagnostic_ids:
                rec = vae.decode(z).sample.float().clamp(-1,1)
                vae_mse.append({'sample_id':payload['rows'][i]['sample_id'], 'split':payload['rows'][i]['split'],
                              'psnr':float(-10*torch.log10((rec-t.float()).square().mean()/4).clamp(min=-120))})
            if i in train_ids:
                targets_latent[i] = (z*vae.config.scaling_factor).float().detach()
                product=payload['rows'][i]['target'];sid=product['sample_id']
                if sid not in gpu_regions:
                    _,mask=normalized_product(product,cfg['image_size'])
                    gpu_regions[sid]=torch.from_numpy(np.asarray(mask).copy()).float()[None,None].to(device)/255
                target_regions[i]=gpu_regions[sid]
                if has_clip or has_gan:
                    target_boxes[i]=region_boxes(target_regions[i])
            if i == train_ids[0]:
                tensor_image(rec[0]).save(args.output/'vae_reconstruction.png')
    write_json(args.output/'vae_diagnostic.json', {'targets_only':True,'records':vae_mse})
    # Allocation may change while sharing the GPU; do not occupy the whole card.
    lp = lpips.LPIPS(net='alex').to(device).eval().requires_grad_(False)
    metric_network=lp if cfg.get('report_lpips',False) else None
    semantic=None;discriminator=None;optimizer_disc=None
    foreground_text={};global_text={};foreground_visual={}
    if has_clip:
        assets=args.sd.parent.parent
        vendor=assets/'vendor/CLIP'
        sys.path.insert(0,str(vendor));sys.path.insert(1,str(assets/'pix2pix_dependencies'))
        import clip
        weights=Path(cfg.get('clip_weights',str(Path.home()/'.cache/clip/ViT-B-32.pt')))
        clip_model,_=clip.load(str(weights),device=device,jit=False)
        semantic=FrozenCLIPSemantic(clip_model).to(device).eval()
        descriptions=json.loads((args.output/'product_descriptions.json').read_text())
        with torch.no_grad():
            for start in range(0,len(train_ids),16):
                ids=train_ids[start:start+16]
                captions=[f"One {payload['rows'][i]['category']}: {descriptions[payload['rows'][i]['target']['sample_id']]}" for i in ids]
                fg_tokens=clip.tokenize(captions,truncate=True).to(device)
                whole_tokens=clip.tokenize([payload['rows'][i]['teacher_caption'] for i in ids],truncate=True).to(device)
                fg_features=F.normalize(semantic.clip.encode_text(fg_tokens).float(),dim=-1)
                whole_features=F.normalize(semantic.clip.encode_text(whole_tokens).float(),dim=-1)
                images=image_float(payload['target'][ids],device)
                boxes=torch.cat([target_boxes[i] for i in ids])
                visual=semantic.image_features(crop_regions(images,boxes,224))
                for i,fg,whole,image_feat in zip(ids,fg_features,whole_features,visual):
                    foreground_text[i]=fg.detach();global_text[i]=whole.detach();foreground_visual[i]=image_feat.detach()
        vendor_commit=subprocess.check_output(['git','-C',str(vendor),'rev-parse','HEAD'],text=True).strip()
        write_json(args.output/'semantic_teacher_identity.json',{'weight_path':str(weights),'sha256':sha256(weights),
              'vendor_git':vendor_commit,'clip_module_path':clip.__file__,
              'optimized_on':'TRAIN only','inference_required':False,
              'foreground_pairwise_loss':'positive image/text pairs; no off-diagonal batch average'})
    if has_gan:
        discriminator=PatchDiscriminator(cfg.get('discriminator_width',32)).to(device)
        if resumed is not None and 'discriminator' in resumed.get('training_aux',{}):
            discriminator.load_state_dict(resumed['training_aux']['discriminator'])
        optimizer_disc=torch.optim.AdamW(discriminator.parameters(),lr=cfg['discriminator_lr'],betas=(.5,.999))
    params = [p for p in model.parameters() if p.requires_grad]
    adapter_params=list(adapter.parameters())
    unet_params=[p for p in unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([{'params':adapter_params,'lr':cfg.get('adapter_lr',cfg['train_lr'])},
                                  {'params':unet_params,'lr':cfg['train_lr']}])
    counts = {'qwen_full_forward': payload['qwen_checks']['qwen_forward_parameters'],
              'qwen_loaded_including_lm_head':payload['qwen_checks']['qwen_loaded_parameters'],
              'qwen_embedding':payload['qwen_checks']['qwen_embedding_parameters'],
              'adapter':sum(p.numel() for p in adapter.parameters()),
              'unet_including_lora':sum(p.numel() for p in unet.parameters()),
              'vae_encoder':sum(p.numel() for p in vae.encoder.parameters())+sum(p.numel() for p in vae.quant_conv.parameters()),
              'vae_decoder':sum(p.numel() for p in vae.decoder.parameters())+sum(p.numel() for p in vae.post_quant_conv.parameters()),
              'trainable':sum(p.numel() for p in params)}
    counts['unet_training_scope']=cfg.get('unet_training','lora')
    counts['training_only_discriminator']=sum(p.numel() for p in discriminator.parameters()) if discriminator is not None else 0
    counts['training_only_semantic_clip']=sum(p.numel() for p in semantic.parameters()) if semantic is not None else 0
    counts['inference_total'] = counts['qwen_full_forward']+counts['adapter']+counts['unet_including_lora']+counts['vae_decoder']
    write_json(args.output/'parameter_counts.json',counts)
    native = (evaluate(model,payload,train_eval_ids,device,args.output,'train_native_caption_teacher',native_teacher=True)
              if resumed is None else {'not_run':'Resumed UNet is not the untouched public prior'})
    baseline = evaluate(model,payload,val_ids,device,args.output,'val_before_paired_training',metric_network=metric_network)
    best, best_step, history = float('-inf'),0,[]
    batch_size=cfg.get('batch_size',1)
    sample_order=[]
    order_rng=random.Random(cfg['seed']+8181)
    needed=cfg['train_steps']*cfg['gradient_accumulation']*batch_size
    while len(sample_order)<needed:
        order=list(train_ids)
        if cfg.get('shuffle_training',False):order_rng.shuffle(order)
        sample_order.extend(order)
    for step in range(1,cfg['train_steps']+1):
        model.train(); vae.eval(); optimizer.zero_grad(set_to_none=True)
        total=0.0
        fake_views=[];real_views=[];last_losses={}
        if discriminator is not None:discriminator.eval().requires_grad_(False)
        for micro in range(cfg['gradient_accumulation']):
            begin=((step-1)*cfg['gradient_accumulation']+micro)*batch_size
            ids=sample_order[begin:begin+batch_size]
            h,valid=pack_hidden([payload['hidden'][i] for i in ids],device)
            regions=torch.cat([target_regions[i] for i in ids])
            with torch.autocast('cuda',dtype=torch.bfloat16):
                z,pred=model(h,valid,noise_for([payload['rows'][i] for i in ids],device))
                target=image_float(payload['target'][ids],device)
                latent_loss=region_weighted_loss(z,torch.cat([targets_latent[i] for i in ids]),regions,
                                                cfg.get('latent_foreground_weight',0))
                rgb_loss=region_weighted_loss(pred,target,regions,
                                             cfg.get('rgb_foreground_weight',0),squared=False)
                perceptual=lp(pred.clamp(-1,1),target).mean()
                cond=adapter(h,valid)
                align=F.mse_loss(cond,torch.stack([payload['train_clip_teacher'][i] for i in ids]).float().to(device))
                loss=(cfg['latent_mse_weight']*latent_loss+cfg['rgb_l1_weight']*rgb_loss+
                      cfg['lpips_weight']*perceptual+cfg['condition_alignment_weight']*align)
                boxes=torch.cat([target_boxes[i] for i in ids]) if (has_clip or has_gan) else None
                if semantic is not None:
                    fg=crop_regions(pred,boxes,224)
                    whole=F.interpolate(pred.float(),size=(224,224),mode='bilinear',align_corners=False)
                    features=semantic.image_features(torch.cat([fg,whole]))
                    n=len(ids);fg_features=features[:n];whole_features=features[n:]
                    text_goal=torch.stack([foreground_text[i] for i in ids])
                    image_goal=torch.stack([foreground_visual[i] for i in ids])
                    whole_goal=torch.stack([global_text[i] for i in ids])
                    clip_text=(1-(fg_features*text_goal).sum(-1)).mean()
                    clip_image=(1-(fg_features*image_goal).sum(-1)).mean()
                    clip_global=(1-(whole_features*whole_goal).sum(-1)).mean()
                    loss=loss+cfg['foreground_clip_text_weight']*clip_text+cfg['foreground_clip_image_weight']*clip_image+cfg['global_clip_text_weight']*clip_global
                    last_losses.update(clip_fg_text=float(clip_text.detach()),clip_fg_image=float(clip_image.detach()))
                if discriminator is not None:
                    fake=discriminator_views(pred,boxes)
                    real=discriminator_views(target,boxes)
                    adversarial=-discriminator(fake).mean()
                    scale=cfg['gan_weight']*min(1.,step/cfg['gan_ramp_steps'])
                    loss=loss+scale*adversarial
                    fake_views.append(fake.detach());real_views.append(real.detach())
                    last_losses['gan_generator']=float(adversarial.detach())
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Non-finite loss at {step}')
            (loss/cfg['gradient_accumulation']).backward()
            total+=float(loss.detach())/cfg['gradient_accumulation']
        norm=torch.nn.utils.clip_grad_norm_(params,1.)
        optimizer.step()
        if discriminator is not None:
            discriminator.train().requires_grad_(True)
            optimizer_disc.zero_grad(set_to_none=True)
            loss_disc=hinge_discriminator_loss(discriminator(torch.cat(real_views)),discriminator(torch.cat(fake_views)))
            if not torch.isfinite(loss_disc):raise FloatingPointError('Non-finite discriminator loss')
            loss_disc.backward()
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(),1.)
            optimizer_disc.step()
            last_losses['discriminator']=float(loss_disc.detach())
        if step%10==0:
            print('paired',step,'/',cfg['train_steps'],'loss',total,'grad_norm',float(norm),flush=True)
            write_json(args.output/'progress.json',{'stage':'paired_training','step':step,'total_steps':cfg['train_steps'],
                       'loss':total,'best_val_psnr':None if best_step==0 else best,'best_step':best_step,
                       'auxiliary_losses':last_losses})
        if step%cfg['eval_every']==0 or step==cfg['train_steps']:
            val=evaluate(model,payload,val_ids,device,args.output,'val_latest',metric_network=metric_network)
            tr=evaluate(model,payload,train_eval_ids,device,args.output,'train_latest',metric_network=metric_network)
            history.append({'step':step,'train_loss':total,'val_psnr':val['mean_per_image_psnr_db'],
                            'train_psnr':tr['mean_per_image_psnr_db'],
                            'val_lpips':val.get('mean_lpips'),'train_lpips':tr.get('mean_lpips')})
            state={'step':step,'adapter':adapter.state_dict(),
                   'unet_trainable':{n:p.detach().cpu().clone() for n,p in unet.named_parameters() if p.requires_grad},
                   'config':cfg,'git':git_identity(),'val':val,
                   'cache_sha256':cache_digest}
            if discriminator is not None:
                state['training_aux']={'discriminator':{n:p.detach().cpu().clone() for n,p in discriminator.state_dict().items()}}
            torch.save(state,args.output/'last_checkpoint.pt')
            if val['mean_per_image_psnr_db']>best:
                best,best_step=val['mean_per_image_psnr_db'],step
                torch.save(state,args.output/'best_checkpoint.pt')
            write_json(args.output/'history.json',history)
            print('validation',step,val['mean_per_image_psnr_db'],'train',tr['mean_per_image_psnr_db'],flush=True)
    selected=torch.load(args.output/'best_checkpoint.pt',map_location='cpu',weights_only=False)
    adapter.load_state_dict(selected['adapter'])
    with torch.no_grad():
        for name,p in unet.named_parameters():
            if name in selected['unet_trainable']:
                p.copy_(selected['unet_trainable'][name].to(device))
    val=evaluate(model,payload,val_ids,device,args.output,'val_best',export_png=cfg.get('export_png',False),metric_network=metric_network)
    tr=evaluate(model,payload,train_eval_ids,device,args.output,'train_best',export_png=cfg.get('export_png',False),metric_network=metric_network)
    wrong=evaluate(model,payload,val_ids,device,args.output,'val_wrong_condition',ablation=True,metric_network=metric_network)
    write_json(args.output/'report.json',{'complete':True,'protocol':cfg['protocol'],'best_step':best_step,
               'best_val':val,'train':tr,'val_before_paired_training':baseline,'val_wrong_condition':wrong,
               'parameters':counts,'native_caption_teacher_train_diagnostic':native,
               'train_optimization_pairs':len(train_ids),'train_evaluation_pairs':len(train_eval_ids),
               'alignment_train_mse':alignment_mse,'test_evaluated':False,
               'overfit_pilot_only':not cfg.get('use_all_pairs',False),
               'shared_noise_seed':cfg.get('shared_noise_seed'),
               'checkpoint_sha256':sha256(args.output/'best_checkpoint.pt'),
               'interpretation':'TRAIN fit with development VAL selection. No independent TEST or final generalization claim.'})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--qwen',type=Path,required=True)
    p.add_argument('--sd',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--stage',choices=['prepare','cache','train','all'],default='all')
    p.add_argument('--resume',type=Path,help='Same-data pilot best/last only; never an old T12 task weight')
    p.add_argument('--allow-expanded-train',action='store_true',help='Audit a parent TRAIN/VAL subset before reusing its weights')
    args=p.parse_args()
    cfg=json.loads(args.config.read_text())
    if cfg['test_enabled']:
        raise ValueError('This pilot has no TEST execution path')
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'report.json').exists() and json.loads((args.output/'report.json').read_text()).get('complete'):
        print('Run already complete; refusing repeated training',flush=True);return
    random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
    torch.set_num_threads(4)
    write_json(args.output/'config.json',cfg)
    write_json(args.output/'execution.json',{'argv':sys.argv,'git':git_identity(),'python':sys.version,
               'torch':torch.__version__,'platform':platform.platform(),'pid':os.getpid(),
               'config_sha256':sha256(args.config),
               'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES'),'started_unix':time.time()})
    rows=prepare(args,cfg)
    if args.stage=='prepare': return
    if os.environ.get('CUDA_VISIBLE_DEVICES') != cfg['gpu_uuid']:
        raise ValueError('Select only the user-authorized GPU UUID explicitly')
    torch.cuda.set_per_process_memory_fraction(cfg['gpu_memory_fraction'],0)
    device=torch.device('cuda:0')
    if not (args.output/'pretrained_manifest.json').exists():
        write_json(args.output/'pretrained_manifest.json',{'qwen':pretrained_manifest(args.qwen),
                                                          'sd_turbo':pretrained_manifest(args.sd)})
    try:
        cache_digest=None
        if args.stage in ('cache','all'): cache_digest=cache(args,cfg,rows,device)
        if args.stage in ('train','all'): train(args,cfg,device,cache_digest)
    except Exception:
        write_json(args.output/'failure.json',{'time':time.time(),'traceback':traceback.format_exc()})
        raise
    finally:
        flush_gpu()


if __name__=='__main__':
    main()
