"""Render reference/target/prediction grids from a trained style-transfer checkpoint."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import torch

from .chair_style_training import ChairStyleDataset, save_sample
from .chair_style_transfer import ChairStyleConfig, ChairStyleTransfer


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--rows", type=int, default=12)
    args = parser.parse_args(); device = torch.device(args.device)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    raw = copy.deepcopy(payload["config"]); raw["widths"] = tuple(raw["widths"])
    config = ChairStyleConfig(**raw)
    model = ChairStyleTransfer(config).to(device).eval(); model.load_state_dict(payload["generator_ema"])
    dataset = ChairStyleDataset(args.data_dir, args.split, config.image_size, args.data_dir / "qwen_style_text_cache.pt")
    loader = torch.utils.data.DataLoader(dataset, batch_size=max(args.rows, config.batch_size), shuffle=False)
    save_sample(model, loader, args.output, device, rows=args.rows); return 0


if __name__ == "__main__":
    raise SystemExit(main())
