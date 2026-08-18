from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .datasets import load_datasets
from .fixed_feedback_training import METHODS, compare, prepare_common_checkpoint, run_method
from .formal_settings import load_formal_settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the four fixed-feedback formal groups")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--phase", choices=("head_warmup", "run", "compare"), required=True)
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = load_formal_settings(args.config)
    if args.phase == "compare":
        print(json.dumps(compare(settings), indent=2), flush=True)
        return
    datasets = load_datasets(settings.base, download=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    if args.phase == "head_warmup":
        prepare_common_checkpoint(settings, datasets, device, force=args.force)
        return
    if args.method is None:
        raise ValueError("--method is required for --phase run")
    seeds = (args.seed,) if args.seed is not None else settings.formal.finetune_seeds
    for seed in seeds:
        run_method(settings, datasets, device, method=args.method, seed=int(seed), force=args.force)


if __name__ == "__main__":
    main()
