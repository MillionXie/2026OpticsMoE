"""CLI for the prior-only electronic Qwen + VAE baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .electronic_baseline import load_electronic_gan_config
from .electronic_baseline_training import train_electronic_baseline
from .settings import TASK_DIR, load_settings


def main() -> int:
    parser = argparse.ArgumentParser(description="Prior-only electronic Qwen + VAE GAN baseline")
    parser.add_argument("--config", type=Path, default=TASK_DIR / "configs/qwen_vae_prior_gan.yaml")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--vae-checkpoint", type=Path, default=None)
    parser.add_argument("--qwen-checkpoint", type=Path, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    settings = load_settings(TASK_DIR / "configs/qwen_vae_baseline.yaml")
    settings.data_dir = args.data_dir.expanduser().resolve()
    settings.output_dir = args.run_dir.expanduser().resolve()
    if args.vae_checkpoint:
        settings.vae_checkpoint = args.vae_checkpoint.expanduser().resolve()
    if args.qwen_checkpoint:
        settings.qwen_checkpoint = args.qwen_checkpoint.expanduser().resolve()
    config = load_electronic_gan_config(args.config.expanduser().resolve())
    result = train_electronic_baseline(settings, config, settings.output_dir, torch.device(args.device))
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
