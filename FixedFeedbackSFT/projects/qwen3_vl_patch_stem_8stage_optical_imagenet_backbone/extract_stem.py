from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from .stem import STEM_FORMAT, merge_order, sha256_file


PATCH_WEIGHT_KEY = "model.visual.patch_embed.proj.weight"
PATCH_BIAS_KEY = "model.visual.patch_embed.proj.bias"
POSITION_KEY = "model.visual.pos_embed.weight"


def extract(checkpoint: Path, output: Path, *, image_size: int = 224) -> dict[str, object]:
    from safetensors import safe_open

    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    patch_size = 16
    merge_size = 2
    grid_size = image_size // patch_size
    if image_size % patch_size or grid_size % merge_size:
        raise ValueError("Image size must produce a patch grid divisible by two")
    with safe_open(checkpoint, framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        missing = {PATCH_WEIGHT_KEY, PATCH_BIAS_KEY, POSITION_KEY} - keys
        if missing:
            raise RuntimeError(f"Qwen checkpoint is missing stem tensors: {sorted(missing)}")
        conv3d = handle.get_tensor(PATCH_WEIGHT_KEY).float()
        bias = handle.get_tensor(PATCH_BIAS_KEY).float()
        position_table = handle.get_tensor(POSITION_KEY).float()

    if tuple(conv3d.shape) != (1024, 3, 2, 16, 16):
        raise RuntimeError(f"Unexpected Qwen patch tensor shape: {tuple(conv3d.shape)}")
    if tuple(position_table.shape) != (2304, 1024):
        raise RuntimeError(f"Unexpected Qwen position table: {tuple(position_table.shape)}")

    # Qwen repeats a still image over temporal_patch_size=2. Summing the two
    # temporal kernels therefore gives the exact static-image Conv2D operator.
    conv2d = conv3d.sum(dim=2)
    source_grid = int(position_table.shape[0] ** 0.5)
    position_grid = position_table.view(source_grid, source_grid, 1024)
    position_grid = F.interpolate(
        position_grid.permute(2, 0, 1).unsqueeze(0),
        size=(grid_size, grid_size),
        mode="bilinear",
        align_corners=True,
    ).squeeze(0)
    position = merge_order(position_grid.unsqueeze(0), merge_size).squeeze(0)
    metadata = {
        "source_checkpoint": str(checkpoint.resolve()),
        "source_checkpoint_sha256": sha256_file(checkpoint),
        "source_keys": [PATCH_WEIGHT_KEY, PATCH_BIAS_KEY, POSITION_KEY],
        "static_temporal_reduction": "sum_two_repeated_frame_kernels",
        "image_size": int(image_size),
        "patch_size": patch_size,
        "temporal_patch_size": 2,
        "spatial_merge_size": merge_size,
        "hidden_size": 1024,
        "token_count": grid_size * grid_size,
        "image_mean": [0.5, 0.5, 0.5],
        "image_std": [0.5, 0.5, 0.5],
        "position_interpolation": "bilinear_align_corners_true",
        "position_order": "qwen_2x2_merge_order",
        "full_qwen_loaded": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(
        {
            "format": STEM_FORMAT,
            "conv2d_weight": conv2d.contiguous(),
            "conv2d_bias": bias.contiguous(),
            "position_embedding": position.contiguous(),
            "metadata": metadata,
        },
        temporary,
    )
    temporary.replace(output)
    report = {
        **metadata,
        "output": str(output.resolve()),
        "output_sha256": sha256_file(output),
        "frozen_parameter_count": int(conv2d.numel() + bias.numel() + position.numel()),
        "output_bytes": output.stat().st_size,
    }
    report_path = output.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract the static-image Qwen3-VL patch/position stem")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=224)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(extract(args.checkpoint, args.output, image_size=args.image_size), indent=2))


if __name__ == "__main__":
    main()
