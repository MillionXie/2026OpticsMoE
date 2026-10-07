"""CLI for structured-pruning recovery of the compact Turbo UNet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .compact_turbo import load_compact_turbo_config
from .pruned_turbo_training import train_pruned_turbo


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a structurally pruned Turbo UNet")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--initial-unet", type=Path, required=True)
    parser.add_argument("--turbo-checkpoint", type=Path, required=True)
    parser.add_argument("--latent-cache-dir", type=Path, required=True)
    parser.add_argument("--adapter-checkpoint", type=Path, required=True)
    parser.add_argument("--feature-cache-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    paths = {
        name: getattr(args, name).expanduser().resolve()
        for name in (
            "initial_unet", "turbo_checkpoint", "latent_cache_dir", "adapter_checkpoint",
            "feature_cache_dir", "data_dir", "output_dir",
        )
    }
    report = train_pruned_turbo(
        **paths, config=load_compact_turbo_config(args.config),
        device=torch.device(args.device), seed=args.seed,
    )
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
