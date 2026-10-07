from __future__ import annotations
import argparse,json
from pathlib import Path
import torch
from .small_fullframe import train_small_editor

def main()->int:
    parser=argparse.ArgumentParser();parser.add_argument("--task",choices=("background","redesign","premium"),required=True);parser.add_argument("--data-dir",type=Path,required=True);parser.add_argument("--instruction-cache",type=Path,required=True);parser.add_argument("--output-dir",type=Path,required=True);parser.add_argument("--epochs",type=int,default=20);parser.add_argument("--batch-size",type=int,default=16);parser.add_argument("--learning-rate",type=float,default=2e-4);parser.add_argument("--seed",type=int,default=42);parser.add_argument("--device",default="cuda");parser.add_argument("--structured-control",action="store_true")
    args=parser.parse_args();result=train_small_editor(task=args.task,data_dir=args.data_dir.resolve(),instruction_cache=args.instruction_cache.resolve(),output_dir=args.output_dir.resolve(),device=torch.device(args.device),epochs=args.epochs,batch_size=args.batch_size,learning_rate=args.learning_rate,seed=args.seed,structured_control=args.structured_control);print(json.dumps(result,indent=2));return 0
if __name__=="__main__":raise SystemExit(main())
