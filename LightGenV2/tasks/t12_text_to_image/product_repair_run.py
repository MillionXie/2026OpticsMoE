"""CLI for preparing, caching, and training ABO product restoration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .product_repair_data import build_instruction_cache
from .product_repair_training import (
    cache_repair_latents,
    load_repair_config,
    train_repair_model,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train text-selected ABO product restoration")
    subparsers = parser.add_subparsers(dest="command", required=True)

    instruction = subparsers.add_parser("cache-instructions")
    instruction.add_argument("--output", type=Path, required=True)
    instruction.add_argument("--qwen-checkpoint", type=Path, required=True)
    instruction.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    instruction.add_argument("--force", action="store_true")

    latent = subparsers.add_parser("cache-latents")
    latent.add_argument("--data-dir", type=Path, required=True)
    latent.add_argument("--instruction-cache", type=Path, required=True)
    latent.add_argument("--vae-checkpoint", type=Path, required=True)
    latent.add_argument("--output-dir", type=Path, required=True)
    latent.add_argument("--image-size", type=int, default=256)
    latent.add_argument("--batch-size", type=int, default=8)
    latent.add_argument("--num-workers", type=int, default=4)
    latent.add_argument("--seed", type=int, default=42)
    latent.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    train = subparsers.add_parser("train")
    train.add_argument("--initial-unet", type=Path, required=True)
    train.add_argument("--turbo-checkpoint", type=Path, required=True)
    train.add_argument("--latent-cache-dir", type=Path, required=True)
    train.add_argument("--data-dir", type=Path, required=True)
    train.add_argument("--instruction-cache", type=Path, required=True)
    train.add_argument("--adapter-checkpoint", type=Path, required=True)
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.command == "cache-instructions":
        report = build_instruction_cache(
            args.output.resolve(), args.qwen_checkpoint.resolve(), torch.device(args.device),
            force=args.force,
        )
    elif args.command == "cache-latents":
        report = cache_repair_latents(
            data_dir=args.data_dir.resolve(), instruction_cache=args.instruction_cache.resolve(),
            vae_checkpoint=args.vae_checkpoint.resolve(), output_dir=args.output_dir.resolve(),
            image_size=args.image_size, device=torch.device(args.device), seed=args.seed,
            batch_size=args.batch_size, num_workers=args.num_workers,
        )
    else:
        model_config, training_config = load_repair_config(args.config.resolve())
        report = train_repair_model(
            initial_unet=args.initial_unet.resolve(),
            turbo_checkpoint=args.turbo_checkpoint.resolve(),
            latent_cache_dir=args.latent_cache_dir.resolve(), data_dir=args.data_dir.resolve(),
            instruction_cache=args.instruction_cache.resolve(),
            adapter_checkpoint=args.adapter_checkpoint.resolve(), output_dir=args.output_dir.resolve(),
            model_config=model_config, training_config=training_config,
            device=torch.device(args.device), seed=args.seed,
        )
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
