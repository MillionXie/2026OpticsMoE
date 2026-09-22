"""CLI for compositional background replacement with half-depth Qwen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .product_scene_replace_data import build_half_qwen_instruction_cache
from .product_scene_replace_training import (
    cache_replacement_latents,
    load_scene_config,
    train_replacement_model,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train compositional ABO background replacement")
    commands = parser.add_subparsers(dest="command", required=True)
    instruction = commands.add_parser("cache-instructions")
    instruction.add_argument("--output", type=Path, required=True)
    instruction.add_argument("--qwen-checkpoint", type=Path, required=True)
    instruction.add_argument("--keep-layers", type=int, default=14)
    instruction.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    instruction.add_argument("--force", action="store_true")
    latent = commands.add_parser("cache-latents")
    latent.add_argument("--data-dir", type=Path, required=True)
    latent.add_argument("--instruction-cache", type=Path, required=True)
    latent.add_argument("--vae-checkpoint", type=Path, required=True)
    latent.add_argument("--output-dir", type=Path, required=True)
    latent.add_argument("--image-size", type=int, default=256)
    latent.add_argument("--batch-size", type=int, default=8)
    latent.add_argument("--num-workers", type=int, default=4)
    latent.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    train = commands.add_parser("train")
    train.add_argument("--initial-unet", type=Path, required=True)
    train.add_argument("--turbo-checkpoint", type=Path, required=True)
    train.add_argument("--latent-cache-dir", type=Path, required=True)
    train.add_argument("--data-dir", type=Path, required=True)
    train.add_argument("--instruction-cache", type=Path, required=True)
    train.add_argument("--adapter-checkpoint", type=Path, required=True)
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--warm-start-checkpoint", type=Path)
    train.add_argument("--compact-optical-mid", action="store_true")
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.command == "cache-instructions":
        report = build_half_qwen_instruction_cache(
            args.output.resolve(), args.qwen_checkpoint.resolve(), torch.device(args.device),
            keep_layers=args.keep_layers, force=args.force,
        )
    elif args.command == "cache-latents":
        report = cache_replacement_latents(
            data_dir=args.data_dir.resolve(), instruction_cache=args.instruction_cache.resolve(),
            vae_checkpoint=args.vae_checkpoint.resolve(), output_dir=args.output_dir.resolve(),
            image_size=args.image_size, batch_size=args.batch_size,
            num_workers=args.num_workers, device=torch.device(args.device),
        )
    else:
        model_config, training_config = load_scene_config(args.config.resolve())
        report = train_replacement_model(
            initial_unet=args.initial_unet.resolve(), turbo_checkpoint=args.turbo_checkpoint.resolve(),
            latent_cache_dir=args.latent_cache_dir.resolve(), data_dir=args.data_dir.resolve(),
            instruction_cache=args.instruction_cache.resolve(),
            adapter_checkpoint=args.adapter_checkpoint.resolve(), output_dir=args.output_dir.resolve(),
            model_config=model_config, training_config=training_config,
            device=torch.device(args.device), seed=args.seed,
            warm_start_checkpoint=args.warm_start_checkpoint.resolve() if args.warm_start_checkpoint else None,
            compact_optical_mid=args.compact_optical_mid,
        )
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
