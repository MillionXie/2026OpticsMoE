"""Standard paired SDXL epsilon training, Qwen-only inputs or native control.

Frozen complete Qwen is cached as a training optimization, with RGB/prompt
fingerprints; final rendering recomputes it from raw inputs. TRAIN target VAE
latents only supervise diffusion noise. DEV noise loss is a diagnostic selection
criterion, never advertised as free-generation quality. TEST is not loaded.
"""
import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time
import traceback

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from .qwen3_scene_conditioning import QwenSDXLConditions, fresh_qwen, qwen_asset_manifest
from .white_reference_scene_probe import GPU, digest, write, checked_tokens, model_manifest


def raw_scene_features(model, processor, image, prompt, image_side,keep_on_device=False):
    content = [{'type': 'image', 'image': image}, {'type': 'text', 'text': prompt}]
    text = processor.apply_chat_template([{'role': 'user', 'content': content}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False)
    inputs = processor(text=[text], images=[image], return_tensors='pt').to('cuda')
    assert inputs['image_grid_thw'].tolist() == [[1, image_side//16, image_side//16]] and inputs['attention_mask'].all()
    native = []
    hook = model.model.visual.register_forward_hook(lambda m, a, out: native.append(out[0]))
    try:
        with torch.inference_mode(): hidden = model.model(**inputs, use_cache=False, return_dict=True).last_hidden_state
    finally: hook.remove()
    assert len(native) == 1 and native[0].shape == ((image_side//32)**2, 2048)
    if keep_on_device:return hidden[0],native[0],inputs['input_ids']
    return hidden[0].cpu(), native[0].cpu(), inputs['input_ids'].cpu()


def new_qwen(a):
    from transformers import AutoProcessor
    model, _ = fresh_qwen(a.qwen)
    processor = AutoProcessor.from_pretrained(a.qwen, local_files_only=True,
        min_pixels=a.qwen_side**2, max_pixels=a.qwen_side**2)
    return model, processor


def checked_pairs(a):
    report = json.loads((a.data/'report.json').read_text())
    review = json.loads((a.data/'visual_review.json').read_text())
    assert report['complete'] and report['role_disjoint'] and not report['TEST_opened']
    assert not report.get('preview_only',False) and report.get('eligible_for_training',True)
    assert review['approved_for_training'] and review['pairs_sha256'] == digest(a.data/'pairs.json')
    rows = json.loads((a.data/'pairs.json').read_text())
    assert all(r['split'] in ['train', 'dev'] for r in rows)
    owners = {}
    for row in rows:
        for field in ['source_product_id', 'reference_sha256', 'source_RGBA_sha256']:
            key = field, row[field]
            if owners.setdefault(key, row['split']) != row['split']: raise ValueError('Cross-split '+field)
        assert digest(a.data/row['reference']) == row['reference_sha256']
        assert digest(a.data/row['target']) == row['target_sha256']
    return rows


def pipeline(a):
    from diffusers import StableDiffusionXLPipeline
    from .scene_precision import dtype
    pipe = StableDiffusionXLPipeline.from_pretrained(a.model_path, variant='fp16', torch_dtype=dtype(a),
        text_encoder=None, text_encoder_2=None, tokenizer=None, tokenizer_2=None, image_encoder=None,
        local_files_only=True).to('cuda')
    # Load the optional UNet directly from the original FP16 asset. Converting
    # an already BF16-rounded UNet back to FP16 cannot restore its precision.
    pipe.vae.to(dtype=torch.bfloat16)
    pipe.load_ip_adapter(str(a.adapter_path), subfolder='sdxl_models',
        weight_name='ip-adapter-plus_sdxl_vit-h.safetensors', image_encoder_folder=None, local_files_only=True)
    pipe.set_ip_adapter_scale(.25); pipe.set_progress_bar_config(disable=True)
    for component in pipe.components.values():
        if isinstance(component, torch.nn.Module): component.eval().requires_grad_(False)
    return pipe


def native_negative_conditions(tokenizers,encoders,caption,force_zeros,device='cuda'):
    # SDXL defaults to ZERO conditions when negative_prompt=None. An explicit
    # negative string (including "") instead follows the native text encoders.
    if caption is None and force_zeros:
        return torch.zeros((1,77,2048),dtype=torch.bfloat16),torch.zeros((1,1280),dtype=torch.bfloat16)
    conditions=[];pooled=None
    for tokenizer,encoder in zip(tokenizers,encoders):
        checked_tokens(tokenizer,caption or '')
        ids=tokenizer(caption or '',padding='max_length',max_length=77,truncation=False,return_tensors='pt').input_ids.to(device)
        with torch.inference_mode():out=encoder(ids,output_hidden_states=True)
        conditions.append(out.hidden_states[-2].cpu())
        if hasattr(out,'text_embeds'):pooled=out.text_embeds.cpu()
    text=torch.cat(conditions,dim=-1)
    assert text.shape==(1,77,2048) and pooled.shape==(1,1280)
    assert torch.isfinite(text).all() and torch.isfinite(pooled).all()
    return text,pooled


def optical_cache(a,rows):
    from .scene_paired_render import install_optical_frontend,inference_inputs
    if a.cache is None:raise ValueError('Reviewed electronic cache required for LG refresh')
    report=json.loads((a.cache/'report.json').read_text());assert report['complete']
    parent_sha=digest(a.cache/'paired_cache.pt');assert parent_sha==report['cache_sha256']
    payload=torch.load(a.cache/'paired_cache.pt',map_location='cpu',weights_only=False)
    assert payload['pair_sha256']==digest(a.data/'pairs.json')
    assert [r['sample_id'] for r in rows]==[r['sample_id'] for r in payload['records']]
    qwen,processor=new_qwen(a)
    optical=install_optical_frontend(qwen,a.lg_frontend,a.lg_features,a.cache)
    features={};dev_keys=set();all_rows=rows+[dict(r,split='dev') for r in inference_inputs(a.data)]
    for i,row in enumerate(all_rows):
        src=a.data/row['reference'];assert digest(src)==row['reference_sha256']
        key=hashlib.sha256(src.read_bytes()+b'\0'+row['prompt'].encode()+f'\0Qwen{a.qwen_side}'.encode()).hexdigest()
        if row['split']=='dev':dev_keys.add(key)
        if key not in features:
            with Image.open(src) as im:h,v,ids=raw_scene_features(qwen,processor,im.convert('RGB'),row['prompt'],a.qwen_side)
            assert torch.equal(ids,payload['features'][key]['input_ids'])
            features[key]={'hidden':h,'visual':v,'input_ids':ids}
        if i%16==0:write(a.output/'progress.json',{'stage':'actual_frozen_postfree_RGB_text_features','done':i+1,'total':len(all_rows),'unique':len(features)})
    assert set(features)==set(payload['features'])
    payload['features']=features;payload['optical_frontend']=optical
    torch.save(payload,a.output/'paired_cache.pt')
    torch.save({'features':{k:v for k,v in features.items() if k in dev_keys},'Qwen_input_side':a.qwen_side,
        'contains_targets_or_masks':False,'optical_frontend':optical},a.output/'DEV_frontend_identity_only.pt')
    shutil.copyfile(a.cache/'model_assets.json',a.output/'model_assets.json')
    write(a.output/'report.json',{'complete':True,'source_commit':os.environ['T12_SOURCE_COMMIT'],
        'cache_sha256':digest(a.output/'paired_cache.pt'),'parent_electronic_cache_sha256':parent_sha,
        'DEV_input_identity_sha256':digest(a.output/'DEV_frontend_identity_only.pt'),
        'optical_frontend':optical,'n':len(rows),'complete_DEV_frontend_contexts':len(dev_keys),
        'Qwen_input_side':a.qwen_side,'target_resolution':a.resolution,'visual_tokens':64,
        'TRAIN_DEV_targets_and_native_condition_supervision_reused_unchanged':True,
        'actual_LG_RGB_text_forward_not_teacher_features_at_deployment':True,'TEST_opened':False})


def cache(a, rows):
    if a.lg_frontend:return optical_cache(a,rows)
    from transformers import CLIPTokenizer, CLIPTextModel, CLIPTextModelWithProjection, CLIPVisionModelWithProjection, CLIPImageProcessor
    write(a.output/'model_assets.json', {'Qwen':qwen_asset_manifest(a.qwen),
        'SDXL':model_manifest(a.model_path),'IPAdapter':model_manifest(a.adapter_path)})
    qwen, processor = new_qwen(a)
    text_encoders = [cls.from_pretrained(a.model_path/('text_encoder'+suffix), variant='fp16',
        torch_dtype=torch.bfloat16, local_files_only=True).cuda().eval().requires_grad_(False)
        for cls, suffix in [(CLIPTextModel, ''), (CLIPTextModelWithProjection, '_2')]]
    tokenizers = [CLIPTokenizer.from_pretrained(a.model_path/('tokenizer'+s), local_files_only=True) for s in ['', '_2']]
    vision = CLIPVisionModelWithProjection.from_pretrained(a.adapter_path/'models/image_encoder',
        torch_dtype=torch.bfloat16, local_files_only=True).cuda().eval().requires_grad_(False)
    image_processor = CLIPImageProcessor()
    negative_caption=a.config_profile.get('negative_prompt')
    model_config=json.loads((a.model_path/'model_index.json').read_text())
    force_zeros=model_config.get('force_zeros_for_empty_prompt',True)
    negative_text,negative_pooled=native_negative_conditions(tokenizers,text_encoders,negative_caption,force_zeros)
    features = {}; captions = {}; images = {}; records = []
    for i, row in enumerate(rows):
        path = a.data/row['reference']
        key = hashlib.sha256(path.read_bytes()+b'\0'+row['prompt'].encode()+f'\0Qwen{a.qwen_side}'.encode()).hexdigest()
        if key not in features:
            with Image.open(path) as im: h, v, ids = raw_scene_features(qwen, processor, im.convert('RGB'), row['prompt'],a.qwen_side)
            features[key] = {'hidden': h, 'visual': v, 'input_ids': ids}
        if row['native_caption'] not in captions:
            conditions = []; lengths = []
            for tokenizer, encoder in zip(tokenizers, text_encoders):
                lengths.append(checked_tokens(tokenizer, row['native_caption'])['full_token_count'])
                ids = tokenizer(row['native_caption'], padding='max_length', max_length=77,
                    truncation=False, return_tensors='pt').input_ids.cuda()
                with torch.inference_mode(): out = encoder(ids, output_hidden_states=True)
                conditions.append(out.hidden_states[-2].cpu())
            captions[row['native_caption']] = {'text': torch.cat(conditions, dim=-1)[0],
                'pooled': out.text_embeds[0].cpu(), 'length': max(lengths)}
        if row['reference_sha256'] not in images:
            with Image.open(path) as im: pixels = image_processor(images=im.convert('RGB'), return_tensors='pt').pixel_values.cuda().bfloat16()
            with torch.inference_mode(): visual = vision(pixels, output_hidden_states=True).hidden_states[-2]
            assert visual.shape == (1, 257, 1280)
            images[row['reference_sha256']] = visual[0].cpu()
        records.append({**row, 'input_key': key})
        if i % 16 == 0: write(a.output/'progress.json', {'stage': 'full_Qwen_and_native_condition_cache', 'Qwen_input_side':a.qwen_side,'done': i+1, 'total': len(rows)})
    # The final raw renderer must evaluate the fixed complete DEV input set,
    # including requests whose teacher candidate was rejected as a noise-loss GT.
    dev_inputs=json.loads((a.data/'inference_dev_inputs.json').read_text())
    dev_keys=set()
    for row in dev_inputs:
        path=a.data/row['reference'];assert digest(path)==row['reference_sha256']
        key=hashlib.sha256(path.read_bytes()+b'\0'+row['prompt'].encode()+f'\0Qwen{a.qwen_side}'.encode()).hexdigest()
        dev_keys.add(key)
        if key not in features:
            with Image.open(path) as im:h,v,ids=raw_scene_features(qwen,processor,im.convert('RGB'),row['prompt'],a.qwen_side)
            features[key]={'hidden':h,'visual':v,'input_ids':ids}
    with torch.inference_mode(): negative = vision(torch.zeros_like(pixels), output_hidden_states=True).hidden_states[-2].cpu()
    frontend_count = sum(p.numel() for p in qwen.parameters())
    del qwen, processor, text_encoders, vision; gc.collect(); torch.cuda.empty_cache()
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(a.model_path/'vae', variant='fp16', torch_dtype=torch.float32, local_files_only=True).cuda().eval().requires_grad_(False)
    latents = {}
    for i, row in enumerate(records):
        with Image.open(a.data/row['target']) as im:
            assert im.size == (a.resolution, a.resolution)
            rgb = np.asarray(im.convert('RGB')).copy()
        image = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).cuda().float()/127.5-1
        with torch.inference_mode(): latent = vae.encode(image).latent_dist.mode()*vae.config.scaling_factor
        assert torch.isfinite(latent).all()
        latents[row['target_sha256']] = latent[0].cpu().bfloat16()
        if i % 16 == 0: write(a.output/'progress.json', {'stage': 'TRAIN_DEV_target_latents_for_noise_loss_only', 'done': i+1, 'total': len(records)})
    del vae; gc.collect(); torch.cuda.empty_cache()
    payload = {'records': records, 'features': features, 'captions': captions, 'images': images,
        'latents': latents, 'negative_image': negative, 'pair_sha256': digest(a.data/'pairs.json'),
        'negative_text':negative_text,'negative_pooled':negative_pooled,'negative_caption':negative_caption}
    torch.save(payload, a.output/'paired_cache.pt')
    torch.save({'features': {k:v for k,v in features.items() if k in dev_keys}, 'Qwen_input_side':a.qwen_side,
        'contains_targets_or_masks':False}, a.output/'DEV_frontend_identity_only.pt')
    write(a.output/'report.json', {'complete': True, 'source_commit': os.environ['T12_SOURCE_COMMIT'],
        'cache_sha256': digest(a.output/'paired_cache.pt'), 'DEV_input_identity_sha256':digest(a.output/'DEV_frontend_identity_only.pt'), 'n': len(rows), 'Qwen_input_side': a.qwen_side,
        'target_resolution':a.resolution,'complete_DEV_frontend_contexts':len(dev_keys),
        'visual_tokens': (a.qwen_side//32)**2, 'complete_Qwen_parameters':frontend_count, 'vision_layers':24, 'language_layers':28,
        'TEST_opened': False, 'target_images_enter_Qwen': False,
        'native_CFG_negative_caption':negative_caption,
        'native_CFG_mode':'default_zero' if negative_caption is None and force_zeros else 'encoded_explicit_text',
        'cache_inputs_only_RGB_and_prompt': True, 'cache_is_training_optimization_not_deployment': True})


def conditions_for(head, payload, row):
    feature = payload['features'][row['input_key']]
    if head is not None:
        return head(feature['hidden'].unsqueeze(0).cuda().float(), feature['visual'].unsqueeze(0).cuda().float())
    target = payload['captions'][row['native_caption']]
    return {'text': target['text'].unsqueeze(0).cuda(), 'pooled': target['pooled'].unsqueeze(0).cuda(),
            'image': payload['images'][row['reference_sha256']].unsqueeze(0).cuda()}


def anchor_loss(pred, payload, row):
    target = payload['captions'][row['native_caption']]
    length = target['length']
    losses = []
    for name, wanted in [('text', target['text']), ('pooled', target['pooled']),
                         ('image', payload['images'][row['reference_sha256']])]:
        wanted = wanted.unsqueeze(0).cuda().float()
        value = pred[name].float()
        scale = wanted.square().mean().clamp_min(1e-6)
        if name == 'text':
            active = (value[:, :length]-wanted[:, :length]).square().mean()
            padding = (value[:, length:]-wanted[:, length:]).square().mean() if length < 77 else active*0
            losses.append((active+.15*padding)/scale)
        else: losses.append((value-wanted).square().mean()/scale)
    return sum(losses)


def electronic_condition_loss(pred,payload,row):
    if row['split']!='train':raise ValueError('Condition KD gradients are TRAIN-only')
    wanted=payload['TRAIN_electronic_conditions'][row['input_key']]
    return sum(weight*(pred[name].float()-wanted[name].unsqueeze(0).to(pred[name].device).float()).square().mean()/
        wanted[name].float().square().mean().clamp_min(1e-6).to(pred[name].device)
        for name,weight in [('image',2.),('text',1.),('pooled',.25)])


class TRAINReferenceContrast:
    """Distinguish reference instances using frozen TRAIN teacher conditions.

    Product IDs only exclude same-product negatives; none enters the model.
    Common white-background components are centered before cosine comparison.
    """
    def __init__(self,payload,device='cuda'):
        keys={r['input_key'] for r in payload['records'] if r['split']=='train'}
        if set(payload['TRAIN_electronic_conditions'])!=keys:raise ValueError('Condition bank must contain exactly TRAIN keys')
        rows={r['reference_sha256']:r for r in payload['records'] if r['split']=='train'}
        self.rows=list(rows.values());self.rng=random.Random(1742)
        self.vectors=torch.stack([payload['TRAIN_electronic_conditions'][r['input_key']]['image'].float().reshape(-1) for r in self.rows]).to(device)
        self.mean=self.vectors.mean(0,keepdim=True)
        self.vectors=F.normalize(self.vectors-self.mean,dim=1)
        self.target=payload['TRAIN_electronic_conditions']
    def loss(self,pred,row):
        if row['split']!='train':raise ValueError('Reference contrast uses TRAIN only')
        available=[i for i,r in enumerate(self.rows) if r['source_product_id']!=row['source_product_id']]
        if not available:raise ValueError('Different TRAIN products required')
        selected=self.rng.sample(available,min(16,len(available)))
        positive=self.target[row['input_key']]['image'].float().reshape(1,-1).to(pred['image'].device)
        wanted=F.normalize(positive-self.mean,dim=1)
        choices=torch.cat([wanted,self.vectors[selected]],dim=0).detach()
        actual=F.normalize(pred['image'].float().reshape(1,-1)-self.mean,dim=1)
        logits=actual@choices.T/.1
        return F.cross_entropy(logits,torch.zeros(1,device=logits.device,dtype=torch.long))


def fit(a, payload):
    from diffusers import DDPMScheduler
    from peft import LoraConfig
    torch.manual_seed(20261029)
    pipe = pipeline(a); pipe.vae.cpu(); unet = pipe.unet
    unet.add_adapter(LoraConfig(r=4, lora_alpha=4, init_lora_weights='gaussian',
        target_modules=['to_q', 'to_k', 'to_v', 'to_out.0']), adapter_name='scene_pair')
    unet.enable_gradient_checkpointing(); unet.train()
    lora = [p for p in unet.parameters() if p.requires_grad]
    for p in lora: p.data = p.data.float()
    assert all('lora_' in n for n, p in unet.named_parameters() if p.requires_grad)
    head = None
    parent=None
    alignment_identity=None
    parent_data_transfer=None
    original_alignment_sha=a.config_profile.get('fresh_original_alignment_sha256')
    if original_alignment_sha and a.initial_paired is not None:raise ValueError('Fresh balanced baseline cannot inherit paired UNet LoRA')
    if payload.get('optical_frontend') and (a.variant!='qwen' or a.initial_paired is None):
        raise ValueError('LG recovery requires the matched paired Qwen parent')
    if a.initial_paired:
        ready=json.loads((a.initial_paired/'report.json').read_text())
        assert ready['complete'] and digest(a.initial_paired/'best_checkpoint.pt')==ready['best_sha256']
        parent=torch.load(a.initial_paired/'best_checkpoint.pt',map_location='cpu',weights_only=False)
        assert parent['variant']==a.variant and parent['resolution']==a.resolution and parent['step']>0
        if parent['pair_sha256']!=payload['pair_sha256']:
            if getattr(a,'reviewed_parent_data',None) is None:
                raise ValueError('Different paired data requires explicit reviewed recaption parent provenance')
            from .scene_instruction_data import checked_parent_transfer
            parent_data_transfer=checked_parent_transfer(a.data,a.reviewed_parent_data,parent['pair_sha256'],payload['pair_sha256'])
        assert parent['Qwen_input_side']==a.qwen_side
        named=dict(unet.named_parameters());assert set(parent['unet_lora'])=={n for n in named if 'lora_' in n}
        with torch.no_grad():
            for n,value in parent['unet_lora'].items():named[n].copy_(value)
    if payload.get('optical_frontend'):
        payload['optical_frontend']=dict(payload['optical_frontend'],feature_distillation_only_generator_recovery_not_yet_trained=False)
    if a.variant == 'qwen':
        head = QwenSDXLConditions(joint_image_context=a.config_profile.get('joint_image_context',False)).cuda().train()
        if parent is not None:head.load_state_dict(parent['head'],strict=True)
        else:
            state = torch.load(a.alignment/'best_checkpoint.pt', map_location='cpu', weights_only=False)
            from .scene_initial_alignment import checked_alignment
            alignment_identity=checked_alignment(a.alignment,state,original_alignment_sha)
            head.load_state_dict(state['state_dict'], strict=True)
    train = [r for r in payload['records'] if r['split'] == 'train']
    dev = [r for r in payload['records'] if r['split'] == 'dev' and r['condition_id'] in ['0', '1']]
    assert train and dev and not ({r['source_product_id'] for r in train} & {r['source_product_id'] for r in dev})
    cache_digest = digest(a.cache/'paired_cache.pt')
    history = []; rng = random.Random(20261029)
    if head is not None:
        warm = torch.optim.AdamW(head.parameters(), lr=1e-4, weight_decay=.01)
        for step in range(a.warmup):
            chosen = rng.sample(train, 4)
            loss = sum(anchor_loss(conditions_for(head, payload, row), payload, row) for row in chosen)/4
            warm.zero_grad(); loss.backward()
            assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in head.parameters() if p.grad is not None)
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.); warm.step()
            if step % 20 == 0: write(a.output/'progress.json', {'stage': 'TRAIN_only_full_cohort_condition_warmup', 'step': step+1, 'total': a.warmup, 'loss': loss.item()})
        del warm
    trainable = lora + ([] if head is None else list(head.parameters()))
    contrast_weight=a.config_profile.get('TRAIN_reference_contrast_weight',0.)
    reference_contrast=TRAINReferenceContrast(payload) if contrast_weight else None
    scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    assert scheduler.config.prediction_type == 'epsilon'
    conditioning_side=int(a.config_profile.get('SDXL_conditioning_size',a.resolution))
    time_ids = torch.tensor([[conditioning_side, conditioning_side, 0, 0, conditioning_side, conditioning_side]], device='cuda', dtype=torch.bfloat16)
    def loss_for(row, seed, auxiliary=True):
        cond = conditions_for(head, payload, row)
        latent = payload['latents'][row['target_sha256']].unsqueeze(0).cuda()
        generator = torch.Generator(device='cuda').manual_seed(seed)
        noise = torch.randn(latent.shape, generator=generator, device='cuda', dtype=torch.bfloat16)
        t = torch.randint(0, scheduler.config.num_train_timesteps, (1,), generator=generator, device='cuda')
        noisy = scheduler.add_noise(latent, noise, t)
        prediction = unet(noisy, t, encoder_hidden_states=cond['text'].bfloat16(),
            added_cond_kwargs={'text_embeds': cond['pooled'].bfloat16(), 'time_ids': time_ids,
                               'image_embeds': [cond['image'].bfloat16().unsqueeze(1)]}).sample
        noise_mse = F.mse_loss(prediction.float(), noise.float())
        # Native-encoder anchoring uses TRAIN inputs, not a second image target.
        anchor = anchor_loss(cond, payload, row) if head is not None and auxiliary else noise_mse*0
        kd_weight=a.config_profile.get('teacher_condition_kd_weight',0.)
        kd=electronic_condition_loss(cond,payload,row) if auxiliary and kd_weight else noise_mse*0
        contrast=reference_contrast.loss(cond,row) if auxiliary and reference_contrast is not None else noise_mse*0
        return noise_mse+.02*anchor+kd_weight*kd+contrast_weight*contrast, noise_mse, anchor
    def optimizer():
        groups = [{'params': lora, 'lr': a.lr}]
        if head is not None: groups.append({'params': head.parameters(), 'lr': a.lr})
        return torch.optim.AdamW(groups, weight_decay=.01)
    # Real 1024 backward smoke, then restore every trainable tensor before fitting.
    initial = [p.detach().clone() for p in trainable]; opt = optimizer(); smoke = []
    for step in range(4):
        began = time.monotonic(); loss, mse, anchor = loss_for(train[step], 1042+step)
        direct = {}
        if head is not None:
            probes = [head.text.out.weight, head.image.out.weight]
            gradients = torch.autograd.grad(mse, probes, retain_graph=True)
            direct = {name: grad.float().abs().sum().item() for name,grad in zip(['text','image'],gradients)}
            assert all(value > 0 and math.isfinite(value) for value in direct.values())
        opt.zero_grad(); loss.backward()
        lg = sum(p.grad.float().abs().sum().item() for p in lora if p.grad is not None)
        hg = 0 if head is None else sum(p.grad.float().abs().sum().item() for p in head.parameters() if p.grad is not None)
        assert torch.isfinite(loss) and lg > 0 and (head is None or hg > 0)
        assert all(torch.isfinite(p.grad).all() for p in trainable if p.grad is not None)
        torch.nn.utils.clip_grad_norm_(trainable, 1.); opt.step(); torch.cuda.synchronize()
        smoke.append({'step': step+1, 'loss': loss.item(), 'LoRA_gradient': lg, 'head_gradient': hg,
                      'epsilon_only_head_gradient':direct,
                      'seconds_including_backward': time.monotonic()-began})
    write(a.output/'smoke_report.json', {'complete': True, 'resolution': a.resolution, 'steps': smoke,
        'Qwen_and_VAE_and_UNet_base_frozen': True, 'reset_after_smoke': True})
    with torch.no_grad():
        for p, before in zip(trainable, initial): p.copy_(before)
    del opt, initial; opt = optimizer(); best = math.inf; stale = 0; updates = 0
    rng = random.Random(20261029); schedule = hashlib.sha256()
    total = len(train)*a.epochs
    for epoch in range(1, a.epochs+1):
        rng.shuffle(train); running = []; began = time.monotonic()
        for i, row in enumerate(train):
            assert row['split'] == 'train'
            noise_seed = rng.randrange(2**31)
            schedule.update((row['sample_id']+'|'+str(noise_seed)+'\n').encode())
            loss, mse, anchor = loss_for(row, noise_seed)
            opt.zero_grad(); loss.backward()
            if not torch.isfinite(loss): raise ValueError('Nonfinite TRAIN loss')
            torch.nn.utils.clip_grad_norm_(trainable, 1., error_if_nonfinite=True); opt.step()
            updates += 1; running.append(mse.item())
            lr = a.lr*min(1., updates/100)*(.1+.9*.5*(1+math.cos(math.pi*updates/total)))
            for group in opt.param_groups: group['lr'] = lr
            if updates % 20 == 0:
                write(a.output/'progress.json', {'stage': 'standard_paired_epsilon_training', 'variant': a.variant,
                    'epoch': epoch, 'update': updates, 'total': total, 'TRAIN_epsilon_MSE_recent': sum(running[-20:])/len(running[-20:]),
                    'epoch_elapsed_seconds': time.monotonic()-began})
        unet.eval()
        if head is not None: head.eval()
        validation = []
        with torch.no_grad():
            for row in dev:
                _, mse, _ = loss_for(row, row['seed'], auxiliary=False)
                validation.append(mse.item())
        score = sum(validation)/len(validation)
        history.append({'epoch': epoch, 'step': updates, 'TRAIN_epsilon_MSE': sum(running)/len(running),
                        'DEV_epsilon_MSE': score, 'DEV_n': len(dev), 'epoch_seconds': time.monotonic()-began})
        write(a.output/'history.json', history)
        state = {'schema': 'qwen3_scene_paired_v1', 'variant': a.variant, 'step': updates, 'epoch': epoch,
            'head': None if head is None else {n:p.detach().cpu() for n,p in head.state_dict().items()},
            'unet_lora': {n:p.detach().cpu() for n,p in unet.named_parameters() if 'lora_' in n},
            'negative_image': payload['negative_image'], 'DEV_epsilon_MSE': score,
            'pair_sha256': payload['pair_sha256'], 'cache_sha256': cache_digest,
            'reviewed_parent_data_transfer':parent_data_transfer,
            'initial_alignment_identity':alignment_identity,
            'source_commit': os.environ['T12_SOURCE_COMMIT'], 'Qwen_input_side': a.qwen_side,'resolution':a.resolution,
            'SDXL_conditioning_size':conditioning_side,'negative_text':payload['negative_text'],
            'negative_pooled':payload['negative_pooled'],'negative_caption':payload['negative_caption'],
            'optical_frontend':payload.get('optical_frontend'),
            'frozen_optical_frontend_head_UNet_recovery':bool(payload.get('optical_frontend'))}
        state['joint_image_context']=False if head is None else head.joint_image_context
        torch.save(state, a.output/'last_checkpoint.pt')
        if score < best:
            best = score; stale = 0; torch.save(state, a.output/'best_checkpoint.pt')
        else: stale += 1
        if stale >= 2: break
        unet.train()
        if head is not None: head.train()
    write(a.output/'report.json', {'complete': True, 'source_commit': os.environ['T12_SOURCE_COMMIT'],
        'variant': a.variant, 'epochs': len(history), 'actual_paired_updates': updates,
        'TRAIN_pairs': len(train), 'DEV_noise_loss_pairs': len(dev), 'best_DEV_epsilon_MSE': best,
        'best_sha256': digest(a.output/'best_checkpoint.pt'), 'head_parameters': 0 if head is None else sum(p.numel() for p in head.parameters()),
        'trainable_UNet_LoRA_parameters': sum(p.numel() for p in lora), 'VAE_and_Qwen_and_UNet_base_frozen': True,
        'TEST_opened': False, 'training_resolution': a.resolution, 'training_precision': 'BF16 base, FP32 trainable parameters',
        'best_selection': 'Fixed DEV diffusion noise loss; not free-generation quality/comparability proof',
        'training_input_order_noise_schedule_sha256':schedule.hexdigest(),
        'optical_frontend':payload.get('optical_frontend'),
        'initial_paired_checkpoint_sha256':None if a.initial_paired is None else digest(a.initial_paired/'best_checkpoint.pt'),
        'initial_alignment_identity':alignment_identity,
        'optical_parameters_updated':False,
        'teacher_condition_kd_weight':a.config_profile.get('teacher_condition_kd_weight',0.),
        'condition_teacher_sha256':payload.get('condition_teacher_sha256'),
        'joint_image_context':a.config_profile.get('joint_image_context',False),
        'TRAIN_reference_contrast_weight':contrast_weight,
        'image_feedback': False, 'rotation': False, 'quality_or_speed_accepted': False})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=['cache', 'train'], required=True)
    p.add_argument('--variant', choices=['qwen', 'native'], default='qwen')
    p.add_argument('--config', type=Path, default=Path(__file__).parent/'configs/scene_paired_training512_20261029.json')
    for key in ['data', 'qwen', 'model-path', 'adapter-path', 'alignment', 'cache', 'output']:
        p.add_argument('--'+key, type=Path, required=key in ['data', 'model-path', 'adapter-path', 'output'])
    p.add_argument('--epochs', type=int)
    p.add_argument('--warmup', type=int)
    p.add_argument('--lr', type=float)
    p.add_argument('--initial-paired',type=Path)
    p.add_argument('--reviewed-parent-data',type=Path,help='Explicit old paired corpus for a SHA-bound visible-facts recaption transfer; default same-data guard unchanged')
    p.add_argument('--lg-frontend',type=Path);p.add_argument('--lg-features',type=Path)
    a = p.parse_args()
    config = json.loads(a.config.read_text())
    a.config_profile=config
    assert config['training_resolution'] in [512,1024] and config['Qwen_image_side'] in [256,512]
    a.resolution=config['training_resolution'];a.qwen_side=config['Qwen_image_side']
    assert config['UNet_LoRA_rank'] == 4 and config['condition_anchor_weight'] == .02 and config['batch_size'] == 1
    for key in ['epochs','warmup','lr']:
        if getattr(a,key) is None: setattr(a,key,config[key])
    if a.epochs < 1 or a.warmup < 0 or a.lr <= 0: raise ValueError('Positive training budget required')
    if a.output.exists() or len(os.environ.get('T12_SOURCE_COMMIT', '')) != 40 or os.environ.get('CUDA_VISIBLE_DEVICES') != GPU:
        raise ValueError('Fresh run, published source and GPU6 required')
    a.output.mkdir(parents=True); torch.set_num_threads(4); torch.cuda.set_per_process_memory_fraction(.8, 0)
    write(a.output/'configuration.json', {'profile':config,'resolved_arguments':
        {k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},'config_sha256':digest(a.config)})
    write(a.output/'execution.json', {'argv': sys.argv, 'pid': os.getpid(), 'source_commit': os.environ['T12_SOURCE_COMMIT'],
        'source_sha256': digest(Path(__file__)), 'versions': {n:importlib.metadata.version(n) for n in ['torch','transformers','diffusers','peft']}})
    try:
        rows = checked_pairs(a)
        if a.stage == 'cache': cache(a, rows)
        else:
            report = json.loads((a.cache/'report.json').read_text()); assert report['complete']
            assert digest(a.cache/'paired_cache.pt') == report['cache_sha256']
            payload = torch.load(a.cache/'paired_cache.pt', map_location='cpu', weights_only=False)
            assert payload['pair_sha256'] == digest(a.data/'pairs.json')
            fit(a, payload)
    except Exception:
        write(a.output/'failure.json', {'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__': main()
