"""CLI for compact one-step latent-student training."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch

from .distilled_decoder import load_distilled_decoder_config
from .distilled_decoder_training import train_distilled_decoder


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a compact SD-Turbo latent student")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--turbo-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    config = load_distilled_decoder_config(args.config)
    if args.epochs is not None:
        config = replace(config, epochs=args.epochs)
    if args.batch_size is not None:
        config = replace(config, batch_size=args.batch_size)
    report = train_distilled_decoder(
        args.cache_dir.expanduser().resolve(),
        config,
        args.turbo_checkpoint.expanduser().resolve(),
        args.output_dir.expanduser().resolve(),
        torch.device(args.device),
        args.seed,
    )
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
