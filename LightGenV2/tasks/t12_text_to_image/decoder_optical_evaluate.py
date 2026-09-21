"""Evaluate a trained instruction generator on held-out object identities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .dataset import sha256
from .decoder_optical_generation import DecoderOpticalGenerator, architecture_report, load_decoder_optical_config
from .decoder_optical_training import evaluate, save_grid
from .product_instruction_data import BackpackStyleDataset, TurntableViewDataset


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    config = load_decoder_optical_config(args.config)
    dataset_type = BackpackStyleDataset if config.task == "style" else TurntableViewDataset
    dataset = dataset_type(args.data_dir.resolve(), "test", config.image_size, args.cache.resolve())
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    device = torch.device(args.device)
    model = DecoderOpticalGenerator(config).to(device)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["generator_ema"], strict=True)
    model.eval()
    metrics = evaluate(model, loader, device)
    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "checkpoint_epoch": int(payload["epoch"]),
        "test_samples": len(dataset),
        "metrics": metrics,
        "architecture": architecture_report(model),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "test_evaluation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    save_grid(model, dataset, args.output_dir / "test_grid.png", device)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
