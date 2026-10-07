"""Prepare Qwen instruction features and train a decoder-optical task."""

from __future__ import annotations

import argparse,json
from pathlib import Path

import torch

from .decoder_optical_generation import load_decoder_optical_config
from .decoder_optical_training import train_decoder_optical
from .product_instruction_data import build_instruction_cache


def main()->int:
    p=argparse.ArgumentParser();p.add_argument("--config",type=Path,required=True);p.add_argument("--data-dir",type=Path,required=True);p.add_argument("--run-dir",type=Path,required=True);p.add_argument("--qwen-checkpoint",type=Path,required=True);p.add_argument("--initialize",type=Path);p.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu");p.add_argument("--seed",type=int,default=113);p.add_argument("--force-cache",action="store_true");args=p.parse_args()
    config=load_decoder_optical_config(args.config);data=args.data_dir.resolve();cache=data/f"qwen_{config.task}_instruction_cache.pt";device=torch.device(args.device)
    build_instruction_cache(config.task,cache,args.qwen_checkpoint.resolve(),device,force=args.force_cache)
    result=train_decoder_optical(data,cache,args.run_dir.resolve(),config,device,seed=args.seed,initialize=None if args.initialize is None else args.initialize.resolve())
    print(json.dumps(result,indent=2),flush=True);return 0


if __name__=="__main__":raise SystemExit(main())
