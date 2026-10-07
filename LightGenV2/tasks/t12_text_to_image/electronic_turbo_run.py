"""CLI for the Qwen-conditioned frozen one-step SD-Turbo baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .electronic_turbo import load_turbo_adapter_config
from .electronic_turbo_training import train_turbo_adapter
from .settings import TASK_DIR, load_settings


def main() -> int:
    parser = argparse.ArgumentParser(description="Qwen + one-step SD-Turbo + VAE electronic baseline")
    parser.add_argument("--config", type=Path, default=TASK_DIR / "configs/qwen_sd_turbo_one_step.yaml")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--turbo-checkpoint", type=Path, required=True)
    parser.add_argument("--qwen-checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    settings = load_settings(TASK_DIR / "configs/qwen_vae_baseline.yaml")
    settings.data_dir = args.data_dir.expanduser().resolve()
    settings.output_dir = args.run_dir.expanduser().resolve()
    config = load_turbo_adapter_config(args.config.expanduser().resolve())
    report = train_turbo_adapter(
        settings, config, args.turbo_checkpoint.expanduser().resolve(),
        args.qwen_checkpoint.expanduser().resolve(),
        settings.output_dir, torch.device(args.device),
    )
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
