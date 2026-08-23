from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors import safe_open
from torch.nn import functional as F
from transformers import AutoProcessor

from .extract_stem import PATCH_BIAS_KEY, PATCH_WEIGHT_KEY, POSITION_KEY
from .stem import StaticQwenPatchStem, merge_order


def validate(qwen_directory: Path, stem_checkpoint: Path) -> dict[str, object]:
    processor = AutoProcessor.from_pretrained(qwen_directory, local_files_only=True)
    generator = np.random.default_rng(7)
    array = generator.integers(0, 256, size=(224, 224, 3), dtype=np.uint8)
    processed = processor.image_processor(
        images=[Image.fromarray(array)],
        return_tensors="pt",
        min_pixels=224 * 224,
        max_pixels=224 * 224,
    )
    grid = processed["image_grid_thw"]
    if grid.tolist() != [[1, 14, 14]]:
        raise RuntimeError(f"Expected static Qwen grid [1,14,14], got {grid.tolist()}")
    with safe_open(qwen_directory / "model.safetensors", framework="pt", device="cpu") as handle:
        conv3d = handle.get_tensor(PATCH_WEIGHT_KEY).float()
        bias = handle.get_tensor(PATCH_BIAS_KEY).float()
        position_table = handle.get_tensor(POSITION_KEY).float()
    official_patch = F.linear(
        processed["pixel_values"].float(), conv3d.reshape(1024, -1), bias
    )
    position_grid = F.interpolate(
        position_table.view(48, 48, 1024).permute(2, 0, 1).unsqueeze(0),
        size=(14, 14),
        mode="bilinear",
        align_corners=True,
    )
    official = official_patch + merge_order(position_grid, 2).squeeze(0)
    stem = StaticQwenPatchStem(stem_checkpoint)
    image_tensor = torch.from_numpy(array.copy()).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    extracted = stem(image_tensor).squeeze(0)
    error = (official - extracted).abs()
    report = {
        "status": "passed" if torch.allclose(official, extracted, atol=1.0e-4, rtol=1.0e-4) else "failed",
        "qwen_grid_thw": grid.tolist(),
        "output_shape": list(extracted.shape),
        "max_absolute_error": float(error.max()),
        "mean_absolute_error": float(error.mean()),
        "rms_error": float(error.square().mean().sqrt()),
        "official_output_rms": float(official.square().mean().sqrt()),
        "atol": 1.0e-4,
        "rtol": 1.0e-4,
        "full_qwen_model_loaded": False,
    }
    if report["status"] != "passed":
        raise RuntimeError(json.dumps(report, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate extracted static stem against Qwen tensors")
    parser.add_argument("--qwen-directory", type=Path, required=True)
    parser.add_argument("--stem-checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = validate(args.qwen_directory, args.stem_checkpoint)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
