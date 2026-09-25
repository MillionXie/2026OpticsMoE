"""Prepare and train one checkpoint for background, object, and joint edits."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .product_global_redesign_training import cache_redesign_latents, load_scene_config, train_redesign_model
from .product_unified_edit_data import UnifiedProductEditDataset, build_unified_instruction_cache


def main() -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    instructions = commands.add_parser("cache-instructions")
    instructions.add_argument("--output", type=Path, required=True)
    instructions.add_argument("--data-dir", type=Path, required=True)
    instructions.add_argument("--qwen-checkpoint", type=Path, required=True)
    instructions.add_argument("--device", default="cuda")
    latents = commands.add_parser("cache-latents")
    for name in ("data-dir", "instruction-cache", "vae-checkpoint", "output-dir"):
        latents.add_argument(f"--{name}", type=Path, required=True)
    latents.add_argument("--image-size", type=int, default=256)
    latents.add_argument("--batch-size", type=int, default=12)
    latents.add_argument("--num-workers", type=int, default=4)
    latents.add_argument("--device", default="cuda")
    train = commands.add_parser("train")
    for name in ("initial-unet", "turbo-checkpoint", "latent-cache-dir", "data-dir",
                 "instruction-cache", "adapter-checkpoint", "warm-start-checkpoint",
                 "output-dir", "config"):
        train.add_argument(f"--{name}", type=Path, required=True)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--device", default="cuda")
    train.add_argument("--student-widths", nargs=3, type=int)
    args = parser.parse_args()
    device = torch.device(args.device)
    if args.command == "cache-instructions":
        result = build_unified_instruction_cache(args.output.resolve(), args.data_dir.resolve(),
                                                 args.qwen_checkpoint.resolve(), device)
    elif args.command == "cache-latents":
        result = cache_redesign_latents(
            data_dir=args.data_dir.resolve(), instruction_cache=args.instruction_cache.resolve(),
            vae_checkpoint=args.vae_checkpoint.resolve(), output_dir=args.output_dir.resolve(),
            image_size=args.image_size, device=device, batch_size=args.batch_size,
            num_workers=args.num_workers, dataset_class=UnifiedProductEditDataset,
            task_name="independently controlled product and background editing")
    else:
        model_config, training_config = load_scene_config(args.config.resolve())
        result = train_redesign_model(
            initial_unet=args.initial_unet.resolve(), turbo_checkpoint=args.turbo_checkpoint.resolve(),
            latent_cache_dir=args.latent_cache_dir.resolve(), data_dir=args.data_dir.resolve(),
            instruction_cache=args.instruction_cache.resolve(), adapter_checkpoint=args.adapter_checkpoint.resolve(),
            warm_start_checkpoint=args.warm_start_checkpoint.resolve(), output_dir=args.output_dir.resolve(),
            model_config=model_config, training_config=training_config, device=device, seed=args.seed,
            dataset_class=UnifiedProductEditDataset,
            task_name="independently controlled product and background editing",
            student_widths=tuple(args.student_widths) if args.student_widths else None)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
