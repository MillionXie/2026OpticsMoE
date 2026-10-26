"""Uncached RGB + instruction inference. No dataset, GT, mask or CLIP needed."""
from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path

import torch
from PIL import Image

from .qwen_vl_turbo_data import sha256, write_json
from .qwen_vl_turbo_model import NativeTurboHead, QueryResampler, encode_reference, pack_hidden
from .qwen_vl_turbo_training import configure_unet_training


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image',type=Path,required=True)
    p.add_argument('--prompt',required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--qwen',type=Path,required=True)
    p.add_argument('--sd',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,default=1042)
    args=p.parse_args()
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    cfg=state['config']
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=cfg['gpu_uuid']:
        raise ValueError('Explicitly select the authorized GPU UUID before inference')
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(cfg['gpu_memory_fraction'],0)
    device=torch.device('cuda:0')
    from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
    model=Qwen3VLForConditionalGeneration.from_pretrained(
        args.qwen,torch_dtype=torch.bfloat16,attn_implementation='sdpa',
        local_files_only=True).to(device).eval().requires_grad_(False)
    processor=AutoProcessor.from_pretrained(args.qwen,local_files_only=True,
                                           min_pixels=65536,max_pixels=65536)
    with Image.open(args.image) as im:
        hidden=encode_reference(model,processor,im.convert('RGB'),args.prompt,device)
    del model,processor
    gc.collect();torch.cuda.empty_cache()
    from diffusers import AutoencoderKL,EulerDiscreteScheduler,UNet2DConditionModel
    from peft import LoraConfig
    adapter=QueryResampler(hidden=cfg['adapter_hidden'],layers=cfg['adapter_layers']).to(device)
    adapter.load_state_dict(state['adapter'])
    unet=UNet2DConditionModel.from_pretrained(args.sd,subfolder='unet',variant='fp16',
             torch_dtype=torch.bfloat16,local_files_only=True).to(device)
    unet.requires_grad_(False)
    unet.add_adapter(LoraConfig(r=cfg['unet_lora_rank'],lora_alpha=cfg['unet_lora_rank'],
                     target_modules=['to_q','to_k','to_v','to_out.0'],init_lora_weights=True))
    configure_unet_training(unet,cfg.get('unet_training','lora'))
    names={n for n,p in unet.named_parameters() if p.requires_grad}
    if names!=set(state['unet_trainable']):
        raise ValueError('Checkpoint LoRA parameter names differ')
    with torch.no_grad():
        for name,param in unet.named_parameters():
            if name in names:
                param.data=state['unet_trainable'][name].to(device)
    vae=AutoencoderKL.from_pretrained(args.sd,subfolder='vae',variant='fp16',
             torch_dtype=torch.bfloat16,local_files_only=True).to(device)
    # Only decode is used. Physically discard the unused encoder after loading
    # the native checkpoint; post_quant_conv and decoder remain unchanged.
    vae.encoder=None
    vae.quant_conv=None
    scheduler=EulerDiscreteScheduler.from_pretrained(args.sd,subfolder='scheduler',local_files_only=True)
    head=NativeTurboHead(unet,vae,scheduler,adapter).eval().requires_grad_(False)
    h,valid=pack_hidden([hidden],device)
    noise=torch.randn(1,4,cfg['image_size']//8,cfg['image_size']//8,device=device,
                      generator=torch.Generator(device=device).manual_seed(args.seed))
    with torch.inference_mode():
        _,rgb=head(h,valid,noise)
    array=((rgb[0].cpu().clamp(-1,1)+1)*127.5).round().byte().permute(1,2,0).numpy()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    Image.fromarray(array).save(args.output)
    write_json(args.output.with_suffix('.json'),{'input_image':str(args.image),'input_sha256':sha256(args.image),
          'prompt':args.prompt,'seed':args.seed,'checkpoint_sha256':sha256(args.checkpoint),
          'training_source_commit':state['git']['published_source_commit'],
          'training_shared_noise_seed':cfg.get('shared_noise_seed'),'seed_diversity_validated':False,
          'qwen':str(args.qwen),'sd_turbo':str(args.sd),'output_sha256':sha256(args.output),
          'model_input':'RGB + text + Gaussian noise; no GT, masks, dataset lookup or reference VAE',
          'weight_hash_verification':'Use the training run pretrained_manifest.json; full public weight hashes are not recomputed here'})


if __name__=='__main__': main()
