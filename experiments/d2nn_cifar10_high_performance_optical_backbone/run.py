from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .datasets import load_datasets, prepare_data
from .settings import load_settings
from .training import aggregate, evaluate_checkpoint, train_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize a high-performance CIFAR optical backbone")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--phase", choices=("prepare_data", "train", "evaluate", "aggregate", "all"), required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--checkpoint", default="best.pt")
    parser.add_argument("--force", action="store_true", help="Ignore an existing latest checkpoint")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = load_settings(args.config)
    if args.phase in {"prepare_data", "all"}:
        print(json.dumps(prepare_data(settings), indent=2), flush=True)
        if args.phase == "prepare_data":
            return
    datasets = load_datasets(settings, download=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} config_digest={settings.digest()}", flush=True)
    seeds = (args.seed,) if args.seed is not None else settings.training.seeds
    if args.phase in {"train", "all"}:
        for seed in seeds:
            train_seed(settings, datasets, device, int(seed), force=args.force)
    elif args.phase == "evaluate":
        for seed in seeds:
            evaluate_checkpoint(settings, datasets, device, int(seed), checkpoint=args.checkpoint)
    if args.phase in {"aggregate", "all"} or (args.phase == "train" and args.seed is None):
        print(json.dumps(aggregate(settings), indent=2), flush=True)


if __name__ == "__main__":
    main()
