from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .qwen_mini_small import build_qwen_embedding_cache, train_qwen_mini_editor


def main() -> int:
    parser=argparse.ArgumentParser(description="Train the sub-50M two-block Qwen-mini optical editor")
    sub=parser.add_subparsers(dest="command",required=True)
    cache=sub.add_parser("cache");cache.add_argument("--instruction-cache",type=Path,required=True);cache.add_argument("--qwen-checkpoint",type=Path,required=True);cache.add_argument("--output",type=Path,required=True);cache.add_argument("--device",default="cuda");cache.add_argument("--force",action="store_true")
    train=sub.add_parser("train");train.add_argument("--task",choices=("background","redesign","premium","unified","unified_expanded"),required=True);train.add_argument("--data-dir",type=Path,required=True);train.add_argument("--instruction-cache",type=Path,required=True);train.add_argument("--embedding-cache",type=Path,required=True);train.add_argument("--output-dir",type=Path,required=True);train.add_argument("--epochs",type=int,default=20);train.add_argument("--batch-size",type=int,default=16);train.add_argument("--learning-rate",type=float,default=2e-4);train.add_argument("--seed",type=int,default=42);train.add_argument("--device",default="cuda");train.add_argument("--text-width",type=int,default=768);train.add_argument("--intermediate-width",type=int,default=2048);train.add_argument("--text-heads",type=int,default=12);train.add_argument("--warm-start-checkpoint",type=Path);train.add_argument("--residual-limit",type=float,default=2.0);train.add_argument("--change-loss-weight",type=float,default=0.0)
    args=parser.parse_args();device=torch.device(args.device)
    if args.command=="cache":result=build_qwen_embedding_cache(instruction_cache=args.instruction_cache.resolve(),qwen_checkpoint=args.qwen_checkpoint.resolve(),output=args.output.resolve(),device=device,force=args.force)
    else:result=train_qwen_mini_editor(task=args.task,data_dir=args.data_dir.resolve(),instruction_cache=args.instruction_cache.resolve(),embedding_cache=args.embedding_cache.resolve(),output_dir=args.output_dir.resolve(),device=device,epochs=args.epochs,batch_size=args.batch_size,learning_rate=args.learning_rate,seed=args.seed,text_width=args.text_width,intermediate_width=args.intermediate_width,text_heads=args.text_heads,warm_start_checkpoint=args.warm_start_checkpoint,residual_limit=args.residual_limit,change_loss_weight=args.change_loss_weight)
    print(json.dumps(result,indent=2));return 0


if __name__=="__main__":raise SystemExit(main())
