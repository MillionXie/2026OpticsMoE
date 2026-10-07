"""Build style prompt features and train the chair style-transfer model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .chair_style_cache import build_style_text_cache
from .chair_style_training import train_chair_style_transfer
from .chair_style_transfer import load_chair_style_config
from .settings import TASK_DIR


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=TASK_DIR / "configs/chair_style_electronic.yaml")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--qwen-checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--initialize-checkpoint", type=Path)
    parser.add_argument("--force-style-cache", action="store_true")
    parser.add_argument("--seed", type=int, default=91)
    args = parser.parse_args(); device = torch.device(args.device)
    data_dir = args.data_dir.expanduser().resolve(); cache = data_dir / "qwen_style_text_cache.pt"
    build_style_text_cache(cache, args.qwen_checkpoint.expanduser().resolve(), device, force=args.force_style_cache)
    result = train_chair_style_transfer(
        data_dir, cache, args.run_dir.expanduser().resolve(), load_chair_style_config(args.config), device,
        seed=args.seed, initialize_checkpoint=None if args.initialize_checkpoint is None else args.initialize_checkpoint.expanduser().resolve(),
    )
    print(json.dumps(result, indent=2), flush=True); return 0


if __name__ == "__main__":
    raise SystemExit(main())
