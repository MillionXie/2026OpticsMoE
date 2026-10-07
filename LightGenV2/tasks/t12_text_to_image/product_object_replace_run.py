"""CLI for the compact three-layer-Qwen object replacement task."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import torch
from .product_object_replace_data import build_object_instruction_cache
from .product_object_replace_training import cache_object_latents,load_scene_config,train_object_model

def main()->int:
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest="command",required=True)
    c=sub.add_parser("cache-instructions");c.add_argument("--output",type=Path,required=True);c.add_argument("--data-dir",type=Path,required=True);c.add_argument("--qwen-checkpoint",type=Path,required=True);c.add_argument("--keep-layers",type=int,default=3);c.add_argument("--device",default="cuda");c.add_argument("--force",action="store_true")
    l=sub.add_parser("cache-latents");l.add_argument("--data-dir",type=Path,required=True);l.add_argument("--instruction-cache",type=Path,required=True);l.add_argument("--vae-checkpoint",type=Path,required=True);l.add_argument("--output-dir",type=Path,required=True);l.add_argument("--image-size",type=int,default=256);l.add_argument("--batch-size",type=int,default=16);l.add_argument("--num-workers",type=int,default=4);l.add_argument("--device",default="cuda")
    t=sub.add_parser("train")
    for name in ("initial-unet","turbo-checkpoint","latent-cache-dir","data-dir","instruction-cache","adapter-checkpoint","output-dir","config"):t.add_argument(f"--{name}",type=Path,required=True)
    t.add_argument("--warm-start-checkpoint",type=Path);t.add_argument("--seed",type=int,default=42);t.add_argument("--device",default="cuda")
    a=p.parse_args()
    if a.command=="cache-instructions":r=build_object_instruction_cache(a.output.resolve(),a.data_dir.resolve(),a.qwen_checkpoint.resolve(),torch.device(a.device),keep_layers=a.keep_layers,force=a.force)
    elif a.command=="cache-latents":r=cache_object_latents(data_dir=a.data_dir.resolve(),instruction_cache=a.instruction_cache.resolve(),vae_checkpoint=a.vae_checkpoint.resolve(),output_dir=a.output_dir.resolve(),image_size=a.image_size,device=torch.device(a.device),batch_size=a.batch_size,num_workers=a.num_workers)
    else:
        mc,tc=load_scene_config(a.config.resolve());r=train_object_model(initial_unet=a.initial_unet.resolve(),turbo_checkpoint=a.turbo_checkpoint.resolve(),latent_cache_dir=a.latent_cache_dir.resolve(),data_dir=a.data_dir.resolve(),instruction_cache=a.instruction_cache.resolve(),adapter_checkpoint=a.adapter_checkpoint.resolve(),output_dir=a.output_dir.resolve(),model_config=mc,training_config=tc,device=torch.device(a.device),warm_start_checkpoint=a.warm_start_checkpoint.resolve() if a.warm_start_checkpoint else None,seed=a.seed)
    print(json.dumps(r,indent=2));return 0
if __name__=="__main__":raise SystemExit(main())
