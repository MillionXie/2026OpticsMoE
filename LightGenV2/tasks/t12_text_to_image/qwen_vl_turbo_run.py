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
                                  render_pair, sha256, validate_roles, write_json)
from .qwen_vl_turbo_model import (NativeTurboHead, QueryResampler, encode_reference,
                                   image_tensor, pack_hidden, region_weighted_loss)


def git_identity() -> dict:
    root = Path(__file__).resolve().parents[3]
    def git(*args):
        return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()
    files = [Path(__file__), Path(__file__).with_name('qwen_vl_turbo_model.py'),
             Path(__file__).with_name('qwen_vl_turbo_data.py'),
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
    selected = pilot_subset(payload['pairs'], cfg['pilot_pairs_per_category_mode'])
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
        return
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
    annotated = add_descriptions(rows, descriptions)
    write_json(args.output / 'annotated_pairs.json', annotated)
    tensors, hidden = [], []
    for i, row in enumerate(annotated):
        reference, target = render_pair(row, cfg['image_size'])
        tensors.append((image_tensor(reference), image_tensor(target)))
        hidden.append(encode_reference(model, processor, reference, row['prompt'], device))
        (args.output / 'data_preview').mkdir(exist_ok=True)
        reference.save(args.output / 'data_preview' / f'{i:03d}_input.png')
        target.save(args.output / 'data_preview' / f'{i:03d}_target.png')
        print('multimodal_features', i+1, '/', len(rows), list(hidden[-1].shape), flush=True)
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
        for i, caption in zip(train_indices, train_captions):
            ids = tokenizer(caption, padding='max_length', max_length=77, truncation=True,
                            return_tensors='pt').input_ids.to(device)
            teacher[i] = clip(ids).last_hidden_state[0].bfloat16().cpu()
    torch.save({'pilot_sha256': sha256(args.output / 'pilot_pairs.json'), 'rows': annotated,
                'hidden': hidden, 'reference': torch.stack([t[0] for t in tensors]),
                'target': torch.stack([t[1] for t in tensors]), 'train_clip_teacher': teacher,
                'qwen_checks': controls,
                'clip_scope': 'frozen TRAIN-only native text-condition alignment teacher'}, path)
    del clip
    flush_gpu()


def tensor_image(tensor):
    return Image.fromarray(((tensor.detach().cpu().clamp(-1,1)+1)*127.5).round().byte().permute(1,2,0).numpy())


@torch.no_grad()
def evaluate(model, payload, indices, device, output, label, ablation=False, native_teacher=False):
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
        target = payload['target'][i:i+1].to(device)
        mse = (prediction-target).square().mean().item()/4
        psnr = -10*np.log10(max(mse,1e-12))
        errors.append({'sample_id': row['sample_id'], 'mode': row['mode'], 'category': row['category'],
                       'psnr': psnr, 'mse_01': mse})
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
    write_json(output / f'{label}.json', report)
    return report


def train(args, cfg, device):
    from diffusers import AutoencoderKL, EulerDiscreteScheduler, UNet2DConditionModel
    from peft import LoraConfig
    import lpips
    payload = torch.load(args.output / 'frozen_features.pt', map_location='cpu', weights_only=False)
    validate_roles(payload['rows'])
    train_ids = [i for i,r in enumerate(payload['rows']) if r['split']=='train']
    val_ids = [i for i,r in enumerate(payload['rows']) if r['split']=='val']
    adapter = QueryResampler(hidden=cfg['adapter_hidden'], layers=cfg['adapter_layers']).to(device)
    resumed = None
    if args.resume:
        resumed = torch.load(args.resume, map_location='cpu', weights_only=False)
        if resumed['cache_sha256'] != sha256(args.output/'frozen_features.pt'):
            raise ValueError('Resume weights refer to a different data/conditioning cache')
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
    for step in range(0 if resumed is not None else cfg['alignment_steps']):
        ids = rng.sample(train_ids, min(6,len(train_ids)))
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
        h,valid=pack_hidden([payload['hidden'][i] for i in train_ids],device)
        target=torch.stack([payload['train_clip_teacher'][i] for i in train_ids]).float().to(device)
        alignment_mse=float(F.mse_loss(adapter(h,valid),target))
    write_json(args.output/'alignment_report.json',{'train_condition_mse':alignment_mse,
               'steps':0 if resumed is not None else cfg['alignment_steps'],
               'resumed_adapter':resumed is not None,'no_val_fit':True})
    del opt
    unet = UNet2DConditionModel.from_pretrained(args.sd, subfolder='unet', variant='fp16',
                    torch_dtype=torch.bfloat16, local_files_only=True).to(device).requires_grad_(False)
    unet.add_adapter(LoraConfig(r=cfg['unet_lora_rank'],lora_alpha=cfg['unet_lora_rank'],
                               target_modules=['to_q','to_k','to_v','to_out.0'],init_lora_weights=True))
    for p in unet.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    if resumed is not None:
        names={n for n,p in unet.named_parameters() if p.requires_grad}
        if names != set(resumed['unet_trainable']):
            raise ValueError('Resume LoRA parameter names changed')
        with torch.no_grad():
            for name,p in unet.named_parameters():
                if name in names: p.copy_(resumed['unet_trainable'][name].to(device))
    unet.enable_gradient_checkpointing()
    vae = AutoencoderKL.from_pretrained(args.sd, subfolder='vae', variant='fp16',
                    torch_dtype=torch.bfloat16, local_files_only=True).to(device).eval().requires_grad_(False)
    scheduler = EulerDiscreteScheduler.from_pretrained(args.sd,subfolder='scheduler', local_files_only=True)
    model = NativeTurboHead(unet,vae,scheduler,adapter)
    # Encoder is used solely for supervised targets and VAE diagnostic, not reference conditioning.
    targets_latent, target_regions, vae_mse = {}, {}, []
    with torch.no_grad():
        for i in train_ids + val_ids:
            t = payload['target'][i:i+1].to(device, dtype=vae.dtype)
            z = vae.encode(t).latent_dist.mode()
            rec = vae.decode(z).sample.float().clamp(-1,1)
            vae_mse.append({'sample_id':payload['rows'][i]['sample_id'], 'split':payload['rows'][i]['split'],
                            'psnr':float(-10*torch.log10((rec-t.float()).square().mean()/4).clamp(min=-120))})
            if i in train_ids:
                targets_latent[i] = (z*vae.config.scaling_factor).float().detach()
                from .product_object_replace_data import ProductObjectReplacementDataset
                _,mask=ProductObjectReplacementDataset._load_product(payload['rows'][i]['target'],cfg['image_size'])
                target_regions[i]=torch.from_numpy(np.asarray(mask).copy()).float()[None,None].to(device)/255
            if i == train_ids[0]:
                tensor_image(rec[0]).save(args.output/'vae_reconstruction.png')
    write_json(args.output/'vae_diagnostic.json', {'targets_only':True,'records':vae_mse})
    # Allocation may change while sharing the GPU; do not occupy the whole card.
    lp = lpips.LPIPS(net='alex').to(device).eval().requires_grad_(False)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg['train_lr'])
    counts = {'qwen_full_forward': payload['qwen_checks']['qwen_forward_parameters'],
              'qwen_loaded_including_lm_head':payload['qwen_checks']['qwen_loaded_parameters'],
              'qwen_embedding':payload['qwen_checks']['qwen_embedding_parameters'],
              'adapter':sum(p.numel() for p in adapter.parameters()),
              'unet_including_lora':sum(p.numel() for p in unet.parameters()),
              'vae_encoder':sum(p.numel() for p in vae.encoder.parameters())+sum(p.numel() for p in vae.quant_conv.parameters()),
              'vae_decoder':sum(p.numel() for p in vae.decoder.parameters())+sum(p.numel() for p in vae.post_quant_conv.parameters()),
              'trainable':sum(p.numel() for p in params)}
    counts['inference_total'] = counts['qwen_full_forward']+counts['adapter']+counts['unet_including_lora']+counts['vae_decoder']
    write_json(args.output/'parameter_counts.json',counts)
    native = (evaluate(model,payload,train_ids,device,args.output,'train_native_caption_teacher',native_teacher=True)
              if resumed is None else {'not_run':'Resumed UNet is not the untouched public prior'})
    baseline = evaluate(model,payload,val_ids,device,args.output,'val_before_paired_training')
    best, best_step, history = float('-inf'),0,[]
    for step in range(1,cfg['train_steps']+1):
        model.train(); vae.eval(); optimizer.zero_grad(set_to_none=True)
        total=0.0
        for micro in range(cfg['gradient_accumulation']):
            i=train_ids[((step-1)*cfg['gradient_accumulation']+micro)%len(train_ids)]
            h,valid=pack_hidden([payload['hidden'][i]],device)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                z,pred=model(h,valid,noise_for([payload['rows'][i]],device))
                target=payload['target'][i:i+1].to(device)
                latent_loss=region_weighted_loss(z,targets_latent[i],target_regions[i],
                                                cfg.get('latent_foreground_weight',0))
                rgb_loss=region_weighted_loss(pred,target,target_regions[i],
                                             cfg.get('rgb_foreground_weight',0),squared=False)
                perceptual=lp(pred.clamp(-1,1),target).mean()
                cond=adapter(h,valid)
                align=F.mse_loss(cond,payload['train_clip_teacher'][i][None].float().to(device))
                loss=(cfg['latent_mse_weight']*latent_loss+cfg['rgb_l1_weight']*rgb_loss+
                      cfg['lpips_weight']*perceptual+cfg['condition_alignment_weight']*align)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Non-finite loss at {step}')
            (loss/cfg['gradient_accumulation']).backward()
            total+=float(loss.detach())/cfg['gradient_accumulation']
        norm=torch.nn.utils.clip_grad_norm_(params,1.)
        optimizer.step()
        if step%10==0:
            print('paired',step,'/',cfg['train_steps'],'loss',total,'grad_norm',float(norm),flush=True)
            write_json(args.output/'progress.json',{'stage':'paired_training','step':step,'total_steps':cfg['train_steps'],
                       'loss':total,'best_val_psnr':None if best_step==0 else best,'best_step':best_step})
        if step%cfg['eval_every']==0 or step==cfg['train_steps']:
            val=evaluate(model,payload,val_ids,device,args.output,'val_latest')
            tr=evaluate(model,payload,train_ids,device,args.output,'train_latest')
            history.append({'step':step,'train_loss':total,'val_psnr':val['mean_per_image_psnr_db'],
                            'train_psnr':tr['mean_per_image_psnr_db']})
            state={'step':step,'adapter':adapter.state_dict(),
                   'unet_trainable':{n:p.detach().cpu().clone() for n,p in unet.named_parameters() if p.requires_grad},
                   'config':cfg,'git':git_identity(),'val':val,
                   'cache_sha256':sha256(args.output/'frozen_features.pt')}
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
    val=evaluate(model,payload,val_ids,device,args.output,'val_best')
    tr=evaluate(model,payload,train_ids,device,args.output,'train_best')
    wrong=evaluate(model,payload,val_ids,device,args.output,'val_wrong_condition',ablation=True)
    write_json(args.output/'report.json',{'complete':True,'protocol':cfg['protocol'],'best_step':best_step,
               'best_val':val,'train':tr,'val_before_paired_training':baseline,'val_wrong_condition':wrong,
               'parameters':counts,'native_caption_teacher_train_diagnostic':native,
               'alignment_train_mse':alignment_mse,'test_evaluated':False,'overfit_pilot_only':True,
               'checkpoint_sha256':sha256(args.output/'best_checkpoint.pt'),
               'interpretation':'Feasibility/overfit pilot; VAL is development only. Not final independent generalization evidence.'})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--qwen',type=Path,required=True)
    p.add_argument('--sd',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--stage',choices=['prepare','cache','train','all'],default='all')
    p.add_argument('--resume',type=Path,help='Same-data pilot best/last only; never an old T12 task weight')
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
        if args.stage in ('cache','all'): cache(args,cfg,rows,device)
        if args.stage in ('train','all'): train(args,cfg,device)
    except Exception:
        write_json(args.output/'failure.json',{'time':time.time(),'traceback':traceback.format_exc()})
        raise
    finally:
        flush_gpu()


if __name__=='__main__':
    main()
