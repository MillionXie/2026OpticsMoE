"""Render the six learned 9-video x 4-frame phase-mask layouts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from .multivideo_settings import load_settings


MASKS = (
    ("frame_router", "parallel_router.raw_router_phase"),
    ("frame_experts", "parallel_optics.raw_expert_phase"),
    ("frame_global", "parallel_optics.raw_global_phase"),
    ("video_router", "serial_router.raw_router_phase"),
    ("video_experts", "serial_optics.raw_expert_phase"),
    ("video_global", "serial_optics.raw_global_phase"),
)


def _phase(value: torch.Tensor) -> np.ndarray:
    return (2.0 * math.pi * torch.sigmoid(value.float())).cpu().numpy()


def _empty(size: int) -> np.ndarray:
    return np.full((size, size), np.nan, dtype=np.float32)


def _compose(name: str, phase: np.ndarray, geometry) -> np.ndarray:
    canvas = _empty(geometry.active_size)
    if name in ("frame_router", "frame_experts"):
        index = 0
        for video_top, video_left in geometry.video_origins:
            for frame_top, frame_left in geometry.frame_origins_local:
                origins = (
                    (((geometry.frame_lane_size - geometry.frame_expert_size) // 2,) * 2,)
                    if name == "frame_router"
                    else geometry.frame_expert_origins_local
                )
                for expert_top, expert_left in origins:
                    size = geometry.frame_expert_size
                    top = video_top + frame_top + expert_top
                    left = video_left + frame_left + expert_left
                    canvas[top : top + size, left : left + size] = phase[index]
                    index += 1
    elif name in ("frame_global", "video_global"):
        offset = geometry.video_phase_offset
        size = geometry.video_phase_tile_size
        for index, (top, left) in enumerate(geometry.video_origins):
            canvas[top + offset : top + offset + size, left + offset : left + offset + size] = phase[index]
        index = len(geometry.video_origins)
    elif name == "video_router":
        offset = geometry.video_field_offset
        size = geometry.video_field_size
        for index, (top, left) in enumerate(geometry.video_origins):
            canvas[top + offset : top + offset + size, left + offset : left + offset + size] = phase[index]
        index = len(geometry.video_origins)
    elif name == "video_experts":
        index = 0
        size = geometry.video_field_size
        for video_top, video_left in geometry.video_origins:
            for expert_top, expert_left in geometry.video_expert_origins_local:
                top = video_top + expert_top
                left = video_left + expert_left
                canvas[top : top + size, left : left + size] = phase[index]
                index += 1
    else:  # pragma: no cover - guarded by MASKS
        raise KeyError(name)
    if index != len(phase):
        raise RuntimeError(f"{name}: placed {index} masks but checkpoint has {len(phase)}")
    return canvas


def _plot_row(items, path: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 7,
            "axes.titlesize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    cmap = plt.colormaps["twilight"].copy()
    cmap.set_bad("black")
    figure, axes = plt.subplots(1, 3, figsize=(18 / 2.54, 5 / 2.54), constrained_layout=True)
    image = None
    for axis, (title, canvas) in zip(axes, items):
        image = axis.imshow(canvas, cmap=cmap, vmin=0, vmax=2 * math.pi, interpolation="nearest")
        axis.set_title(title.replace("_", " "))
        axis.set_xlabel("active-field x (px)")
        axis.set_ylabel("active-field y (px)")
    colorbar = figure.colorbar(image, ax=axes, fraction=0.018, pad=0.015)
    colorbar.set_label("phase (rad)")
    colorbar.set_ticks((0, math.pi, 2 * math.pi), labels=("0", "π", "2π"))
    figure.savefig(path, dpi=300)
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    settings = load_settings(args.config)
    checkpoint = Path(args.checkpoint).resolve()
    output = Path(args.output_dir).resolve() if args.output_dir else checkpoint.parent / "mask_visualization"
    output.mkdir(parents=True, exist_ok=True)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = saved["state_dict"]
    rendered = []
    statistics = {}
    for name, key in MASKS:
        phase = _phase(state[key])
        canvas = _compose(name, phase, settings.geometry)
        rendered.append((name, canvas))
        finite = np.isfinite(canvas)
        statistics[name] = {
            "parameter_key": key,
            "individual_mask_count": int(len(phase)),
            "individual_mask_size_hw": list(phase.shape[-2:]),
            "occupied_active_pixels": int(finite.sum()),
            "active_field_coverage_fraction": float(finite.mean()),
            "phase_mean_rad": float(phase.mean()),
            "phase_std_rad": float(phase.std()),
        }
    _plot_row(rendered[:3], output / "phase_masks_passes_1_to_3.png")
    _plot_row(rendered[3:], output / "phase_masks_passes_4_to_6.png")
    report = {
        "schema_version": 1,
        "checkpoint": str(checkpoint),
        "checkpoint_epoch": saved.get("epoch"),
        "architecture": saved.get("architecture"),
        "active_field_size": settings.geometry.active_size,
        "physical_semantics": "nine unrelated videos x four frames; six whole-field coherent passes",
        "masks": statistics,
    }
    (output / "phase_mask_statistics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
