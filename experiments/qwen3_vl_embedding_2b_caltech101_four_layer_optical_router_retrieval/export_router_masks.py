from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from experiments.hardware_sdk.workflows.reconstruct_slm import (
    encode_active_phase,
    reconstruct_directory,
    save_active_png,
)

from .modeling import architecture_label
from .hardware_contract import require_empty_directory
from .settings import load_settings


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def export_masks(config: Path, checkpoint: Path, output_dir: Path) -> dict[str, object]:
    settings = load_settings(config)
    if settings.router_backend != "optical":
        raise ValueError("Router phase masks exist only for an optical-router config")
    checkpoint = checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Optical-router checkpoint is missing: {checkpoint}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    expected = architecture_label(settings)
    actual = str(payload.get("metadata", {}).get("optical_architecture", ""))
    if actual != expected:
        raise RuntimeError(
            f"Checkpoint architecture mismatch: expected={expected!r}, actual={actual!r}"
        )
    output_dir = require_empty_directory(output_dir, label="Router phase export")
    compact = output_dir / "compact_phase"
    compact.mkdir(parents=True, exist_ok=True)
    margin = (settings.active_size - settings.expert_size) // 2
    rows: dict[str, object] = {}
    for name in ("vision", "language"):
        state = payload[f"{name}_optical"]
        key = "core.optical_branch.core.router.raw_router_phase"
        if key not in state:
            raise RuntimeError(f"Checkpoint has no {name} router tensor {key!r}")
        raw = state[key].float()
        if tuple(raw.shape) != (settings.expert_size, settings.expert_size):
            raise RuntimeError(f"Unexpected {name} router phase shape {tuple(raw.shape)}")
        phase = 2.0 * math.pi * torch.sigmoid(raw)
        active = torch.zeros(settings.active_size, settings.active_size)
        active[
            margin : margin + settings.expert_size,
            margin : margin + settings.expert_size,
        ] = phase
        if settings.hardware_phase_flip_vertical:
            active = torch.flip(active, (-2,))
        if settings.hardware_phase_flip_horizontal:
            active = torch.flip(active, (-1,))
        path = compact / f"{name}_router.png"
        save_active_png(encode_active_phase(active.numpy()), path)
        rows[name] = {
            "raw_tensor_key": key,
            "trainable_phase_size": [settings.expert_size, settings.expert_size],
            "logical_export_size": [settings.active_size, settings.active_size],
            "logical_trainable_bounds_xyxy": [
                margin,
                margin,
                margin + settings.expert_size,
                margin + settings.expert_size,
            ],
            "compact_png": str(path),
            "compact_sha256": _sha256(path),
            "phase_mean_rad": float(phase.mean()),
            "phase_std_rad": float(phase.std(unbiased=False)),
        }

    reconstruction = reconstruct_directory(
        compact,
        output_dir / "phase_to_play",
        slm_size_wh=(
            settings.hardware_phase_slm_width,
            settings.hardware_phase_slm_height,
        ),
        scale_factor=None,
        center_xy=(
            settings.hardware_phase_slm_center_x,
            settings.hardware_phase_slm_center_y,
        ),
        logical_pixel_pitch_um=settings.language_optical_pixel_pitch_um,
        slm_pixel_pitch_um=settings.hardware_phase_slm_pixel_pitch_um,
    )
    preview = Image.new("RGB", (settings.active_size, settings.active_size), "black")
    draw = ImageDraw.Draw(preview)
    colors = ("#e41a1c", "#377eb8", "#4daf4a", "#984ea3")
    intervals = settings.optical_router_detector_intervals
    index = 0
    for top, bottom in intervals:
        for left, right in intervals:
            draw.rectangle(
                (left, top, right - 1, bottom - 1), outline=colors[index], width=3
            )
            draw.text((left + 3, top + 3), str(index), fill=colors[index])
            index += 1
    preview_path = output_dir / "router_detector_regions_478.png"
    preview.save(preview_path)
    report = {
        "schema_version": 1,
        "config": str(Path(config).expanduser().resolve()),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "checkpoint_architecture": actual,
        "router_masks": rows,
        "detector_intervals_half_open": [list(value) for value in intervals],
        "detector_preview": str(preview_path),
        "phase_flip_vertical_applied": settings.hardware_phase_flip_vertical,
        "phase_flip_horizontal_applied": settings.hardware_phase_flip_horizontal,
        "reconstruction": reconstruction,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "router_mask_export_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Export Vision/Language router phase BMPs")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    report = export_masks(args.config, args.checkpoint, args.output_dir)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
