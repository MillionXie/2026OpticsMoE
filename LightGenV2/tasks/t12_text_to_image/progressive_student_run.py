"""CLI for cached-teacher progressive sub-300M editor distillation."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import torch
from .progressive_student_training import build_teacher_cache,load_scene_config,train_progressive_student

def main()->int:
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest="command",required=True)
    cache=sub.add_parser("cache-teacher")
    for name in ("teacher-checkpoint","adapter-checkpoint","initial-unet","teacher-latent-dir","output-dir","turbo-checkpoint"):cache.add_argument(f"--{name}",type=Path,required=True)
    cache.add_argument("--batch-size",type=int,default=4);cache.add_argument("--num-workers",type=int,default=4);cache.add_argument("--device",default="cuda")
    train=sub.add_parser("train")
    for name in ("initial-unet","turbo-checkpoint","student-latent-dir","teacher-cache-dir","data-dir","instruction-cache","adapter-checkpoint","teacher-checkpoint","output-dir","config"):train.add_argument(f"--{name}",type=Path,required=True)
    train.add_argument("--stage-epochs",default="1,1,2");train.add_argument("--seed",type=int,default=42);train.add_argument("--device",default="cuda")
    args=parser.parse_args();device=torch.device(args.device)
    if args.command=="cache-teacher":
        result=build_teacher_cache(teacher_checkpoint=args.teacher_checkpoint.resolve(),adapter_checkpoint=args.adapter_checkpoint.resolve(),initial_unet=args.initial_unet.resolve(),teacher_latent_dir=args.teacher_latent_dir.resolve(),output_dir=args.output_dir.resolve(),turbo_checkpoint=args.turbo_checkpoint.resolve(),device=device,batch_size=args.batch_size,num_workers=args.num_workers)
    else:
        model,training=load_scene_config(args.config.resolve());epochs=tuple(int(x) for x in args.stage_epochs.split(","))
        result=train_progressive_student(initial_unet=args.initial_unet.resolve(),turbo_checkpoint=args.turbo_checkpoint.resolve(),student_latent_dir=args.student_latent_dir.resolve(),teacher_cache_dir=args.teacher_cache_dir.resolve(),data_dir=args.data_dir.resolve(),instruction_cache=args.instruction_cache.resolve(),adapter_checkpoint=args.adapter_checkpoint.resolve(),teacher_checkpoint=args.teacher_checkpoint.resolve(),output_dir=args.output_dir.resolve(),model_config=model,training_config=training,device=device,stage_epochs=epochs,seed=args.seed)
    print(json.dumps(result,indent=2));return 0
if __name__=="__main__":raise SystemExit(main())
