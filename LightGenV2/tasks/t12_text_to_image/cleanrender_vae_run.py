"""Build Qwen text features and train the chair-only compact VAE-GAN."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .cleanrender_cache import build_cleanrender_text_cache
from .cleanrender_vae import load_cleanrender_vae_config
from .cleanrender_vae_training import train_cleanrender_vae_gan
from .settings import TASK_DIR


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the sub-10M chair conditional VAE-GAN")
    parser.add_argument("--config", type=Path, default=TASK_DIR / "configs/qwen_cleanrender_chair_vae_gan.yaml")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--qwen-checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--force-cache", action="store_true")
    parser.add_argument("--skip-cache", action="store_true")
    parser.add_argument("--seed", type=int, default=73)
    args = parser.parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    device = torch.device(args.device)
    if not args.skip_cache:
        build_cleanrender_text_cache(
            data_dir, args.qwen_checkpoint.expanduser().resolve(), device,
            batch_size=32, force=args.force_cache,
        )
    result = train_cleanrender_vae_gan(
        data_dir, args.run_dir.expanduser().resolve(),
        load_cleanrender_vae_config(args.config.expanduser().resolve()), device, seed=args.seed,
    )
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
