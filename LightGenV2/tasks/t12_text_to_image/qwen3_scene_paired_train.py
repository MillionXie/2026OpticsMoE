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


def raw_features_512(model, processor, image, prompt):
    content = [{'type': 'image', 'image': image}, {'type': 'text', 'text': prompt}]
    text = processor.apply_chat_template([{'role': 'user', 'content': content}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False)
    inputs = processor(text=[text], images=[image], return_tensors='pt').to('cuda')
    assert inputs['image_grid_thw'].tolist() == [[1, 32, 32]] and inputs['attention_mask'].all()
    native = []
    hook = model.model.visual.register_forward_hook(lambda m, a, out: native.append(out[0]))
    try:
        with torch.inference_mode(): hidden = model.model(**inputs, use_cache=False, return_dict=True).last_hidden_state
    finally: hook.remove()
    assert len(native) == 1 and native[0].shape == (256, 2048)
    return hidden[0].cpu(), native[0].cpu(), inputs['input_ids'].cpu()


def new_qwen(a):
    from transformers import AutoProcessor
    model, _ = fresh_qwen(a.qwen)
    processor = AutoProcessor.from_pretrained(a.qwen, local_files_only=True,
        min_pixels=512*512, max_pixels=512*512)
    return model, processor


def checked_pairs(a):
    report = json.loads((a.data/'report.json').read_text())
    review = json.loads((a.data/'visual_review.json').read_text())
    assert report['complete'] and report['role_disjoint'] and not report['TEST_opened']
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
    pipe = StableDiffusionXLPipeline.from_pretrained(a.model_path, variant='fp16', torch_dtype=torch.bfloat16,
        text_encoder=None, text_encoder_2=None, tokenizer=None, tokenizer_2=None, image_encoder=None,
        local_files_only=True).to('cuda')
    pipe.load_ip_adapter(str(a.adapter_path), subfolder='sdxl_models',
        weight_name='ip-adapter-plus_sdxl_vit-h.safetensors', image_encoder_folder=None, local_files_only=True)
    pipe.set_ip_adapter_scale(.25); pipe.set_progress_bar_config(disable=True)
    for component in pipe.components.values():
        if isinstance(component, torch.nn.Module): component.eval().requires_grad_(False)
    return pipe


def cache(a, rows):
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
    features = {}; captions = {}; images = {}; records = []
    for i, row in enumerate(rows):
        path = a.data/row['reference']
        key = hashlib.sha256(path.read_bytes()+b'\0'+row['prompt'].encode()+b'\0Qwen512').hexdigest()
        if key not in features:
            with Image.open(path) as im: h, v, ids = raw_features_512(qwen, processor, im.convert('RGB'), row['prompt'])
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
        if i % 16 == 0: write(a.output/'progress.json', {'stage': 'full_Qwen_512_and_native_condition_cache', 'done': i+1, 'total': len(rows)})
    with torch.inference_mode(): negative = vision(torch.zeros_like(pixels), output_hidden_states=True).hidden_states[-2].cpu()
    frontend_count = sum(p.numel() for p in qwen.parameters())
    del qwen, processor, text_encoders, vision; gc.collect(); torch.cuda.empty_cache()
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(a.model_path/'vae', variant='fp16', torch_dtype=torch.float32, local_files_only=True).cuda().eval().requires_grad_(False)
    latents = {}
    for i, row in enumerate(records):
        with Image.open(a.data/row['target']) as im:
            assert im.size == (1024, 1024)
            rgb = np.asarray(im.convert('RGB')).copy()
        image = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).cuda().float()/127.5-1
        with torch.inference_mode(): latent = vae.encode(image).latent_dist.mode()*vae.config.scaling_factor
        assert torch.isfinite(latent).all()
        latents[row['target_sha256']] = latent[0].cpu().bfloat16()
        if i % 16 == 0: write(a.output/'progress.json', {'stage': 'TRAIN_DEV_target_latents_for_noise_loss_only', 'done': i+1, 'total': len(records)})
    del vae; gc.collect(); torch.cuda.empty_cache()
    payload = {'records': records, 'features': features, 'captions': captions, 'images': images,
        'latents': latents, 'negative_image': negative, 'pair_sha256': digest(a.data/'pairs.json')}
    torch.save(payload, a.output/'paired_cache.pt')
    dev_keys = {r['input_key'] for r in records if r['split'] == 'dev'}
    torch.save({'features': {k:v for k,v in features.items() if k in dev_keys}, 'Qwen_input_side':512,
        'contains_targets_or_masks':False}, a.output/'DEV_frontend_identity_only.pt')
    write(a.output/'report.json', {'complete': True, 'source_commit': os.environ['T12_SOURCE_COMMIT'],
        'cache_sha256': digest(a.output/'paired_cache.pt'), 'n': len(rows), 'Qwen_input_side': 512,
        'visual_tokens': 256, 'complete_Qwen_parameters':frontend_count, 'vision_layers':24, 'language_layers':28,
        'TEST_opened': False, 'target_images_enter_Qwen': False,
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
    if a.variant == 'qwen':
        head = QwenSDXLConditions().cuda().train()
        state = torch.load(a.alignment/'best_checkpoint.pt', map_location='cpu', weights_only=False)
        identity = json.loads((a.alignment/'adaptation_report.json').read_text())
        assert digest(a.alignment/'best_checkpoint.pt') == identity['best_sha256']
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
    scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    assert scheduler.config.prediction_type == 'epsilon'
    time_ids = torch.tensor([[1024, 1024, 0, 0, 1024, 1024]], device='cuda', dtype=torch.bfloat16)
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
        return noise_mse+.02*anchor, noise_mse, anchor
    def optimizer():
        groups = [{'params': lora, 'lr': a.lr}]
        if head is not None: groups.append({'params': head.parameters(), 'lr': a.lr})
        return torch.optim.AdamW(groups, weight_decay=.01)
    # Real 1024 backward smoke, then restore every trainable tensor before fitting.
    initial = [p.detach().clone() for p in trainable]; opt = optimizer(); smoke = []
    for step in range(4):
        began = time.monotonic(); loss, mse, anchor = loss_for(train[step], 1042+step)
        opt.zero_grad(); loss.backward()
        lg = sum(p.grad.float().abs().sum().item() for p in lora if p.grad is not None)
        hg = 0 if head is None else sum(p.grad.float().abs().sum().item() for p in head.parameters() if p.grad is not None)
        assert torch.isfinite(loss) and lg > 0 and (head is None or hg > 0)
        assert all(torch.isfinite(p.grad).all() for p in trainable if p.grad is not None)
        torch.nn.utils.clip_grad_norm_(trainable, 1.); opt.step(); torch.cuda.synchronize()
        smoke.append({'step': step+1, 'loss': loss.item(), 'LoRA_gradient': lg, 'head_gradient': hg,
                      'seconds_including_backward': time.monotonic()-began})
    write(a.output/'smoke_report.json', {'complete': True, 'resolution': 1024, 'steps': smoke,
        'Qwen_and_VAE_and_UNet_base_frozen': True, 'reset_after_smoke': True})
    with torch.no_grad():
        for p, before in zip(trainable, initial): p.copy_(before)
    del opt, initial; opt = optimizer(); best = math.inf; stale = 0; updates = 0
    total = len(train)*a.epochs
    for epoch in range(1, a.epochs+1):
        rng.shuffle(train); running = []; began = time.monotonic()
        for i, row in enumerate(train):
            assert row['split'] == 'train'
            loss, mse, anchor = loss_for(row, rng.randrange(2**31))
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
            'source_commit': os.environ['T12_SOURCE_COMMIT'], 'Qwen_input_side': 512}
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
        'TEST_opened': False, 'training_resolution': 1024, 'training_precision': 'BF16 base, FP32 trainable parameters',
        'best_selection': 'Fixed DEV diffusion noise loss; not free-generation quality/comparability proof',
        'image_feedback': False, 'rotation': False, 'quality_or_speed_accepted': False})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=['cache', 'train'], required=True)
    p.add_argument('--variant', choices=['qwen', 'native'], default='qwen')
    p.add_argument('--config', type=Path, default=Path(__file__).parent/'configs/scene_paired_training_20261029.json')
    for key in ['data', 'qwen', 'model-path', 'adapter-path', 'alignment', 'cache', 'output']:
        p.add_argument('--'+key, type=Path, required=key in ['data', 'model-path', 'adapter-path', 'output'])
    p.add_argument('--epochs', type=int)
    p.add_argument('--warmup', type=int)
    p.add_argument('--lr', type=float)
    a = p.parse_args()
    config = json.loads(a.config.read_text())
    assert config['training_resolution'] == 1024 and config['Qwen_image_side'] == 512
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
