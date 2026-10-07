"""CLI for the V0 parallel optical mid-block experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .optical_turbo import load_optical_turbo_config
from .optical_turbo_training import train_optical_turbo


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the parallel optical Turbo mid block")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--compact-unet", type=Path, required=True)
    parser.add_argument("--turbo-checkpoint", type=Path, required=True)
    parser.add_argument("--latent-cache-dir", type=Path, required=True)
    parser.add_argument("--condition-cache", type=Path)
    parser.add_argument("--adapter-checkpoint", type=Path)
    parser.add_argument("--feature-cache-dir", type=Path)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    report = train_optical_turbo(
        compact_unet=args.compact_unet.expanduser().resolve(),
        turbo_checkpoint=args.turbo_checkpoint.expanduser().resolve(),
        latent_cache_dir=args.latent_cache_dir.expanduser().resolve(),
        condition_cache=(
            args.condition_cache.expanduser().resolve() if args.condition_cache else None
        ),
        data_dir=args.data_dir.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
        config=load_optical_turbo_config(args.config),
        device=torch.device(args.device), seed=args.seed,
        adapter_checkpoint=(
            args.adapter_checkpoint.expanduser().resolve() if args.adapter_checkpoint else None
        ),
        feature_cache_dir=(
            args.feature_cache_dir.expanduser().resolve() if args.feature_cache_dir else None
        ),
    )
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
