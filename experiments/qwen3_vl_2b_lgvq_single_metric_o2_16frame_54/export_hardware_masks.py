"""Export six trained optical phase planes and audited dual-SLM layout files.

The trainable phase tensors live on several small logical tiles.  This module
places those tiles at exactly the same coordinates used by ``modeling.py``,
keeps an unflipped canonical 478x478 record, and only then applies the phase
SLM orientation and the 17 um -> 8 um physical-pitch rasterization.

The 1024x1024 amplitude BMPs produced here are *layout templates*.  White
pixels show where a stage puts its input field; they are not sample-dependent
network amplitudes and must not be used as formal video inputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image

from experiments.hardware_sdk.workflows.reconstruct_slm import (
    encode_active_phase,
    place_at_center,
    reconstruct_directory,
    save_active_png,
)

from .modeling import LGVQSingleMetricOEO16, build_model
from .settings import ExperimentSettings, load_settings


STAGES = (
    "vision_router",
    "vision_expert",
    "vision_global",
    "language_router",
    "language_expert",
    "language_global",
)


@dataclass(frozen=True)
class StagePlane:
    """One canonical active-plane phase plus its learned and input supports."""

    name: str
    phase_rad: np.ndarray
    learned_support: np.ndarray
    learned_boxes_xyxy: tuple[tuple[int, int, int, int], ...]
    input_boxes_xyxy: tuple[tuple[int, int, int, int], ...]
    tile_phase_rad: tuple[np.ndarray, ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _phase(raw: torch.Tensor) -> np.ndarray:
    return (
        torch.sigmoid(raw.detach().float()).mul(2.0 * math.pi).cpu().numpy()
    ).astype(np.float32, copy=False)


def _boxes_from_origins(
    origins_yx: Iterable[tuple[int, int]], size: int
) -> tuple[tuple[int, int, int, int], ...]:
    return tuple(
        (int(left), int(top), int(left + size), int(top + size))
        for top, left in origins_yx
    )


def _assemble_tiles(
    name: str,
    tiles: np.ndarray,
    origins_yx: tuple[tuple[int, int], ...],
    *,
    active_size: int,
    input_origins_yx: tuple[tuple[int, int], ...] | None = None,
    input_size: int | None = None,
) -> StagePlane:
    if tiles.ndim != 3 or tiles.shape[0] != len(origins_yx):
        raise ValueError(
            f"{name}: tile tensor {tiles.shape} does not match {len(origins_yx)} origins"
        )
    tile_size = int(tiles.shape[-1])
    if tuple(tiles.shape[-2:]) != (tile_size, tile_size):
        raise ValueError(f"{name}: phase tiles must be square")
    phase = np.zeros((active_size, active_size), dtype=np.float32)
    support = np.zeros((active_size, active_size), dtype=bool)
    for tile, (top, left) in zip(tiles, origins_yx):
        bottom, right = top + tile_size, left + tile_size
        if top < 0 or left < 0 or bottom > active_size or right > active_size:
            raise ValueError(f"{name}: phase tile exceeds the 478-pixel active field")
        if bool(support[top:bottom, left:right].any()):
            raise ValueError(f"{name}: learned phase tiles overlap")
        phase[top:bottom, left:right] = tile
        support[top:bottom, left:right] = True
    input_origins = origins_yx if input_origins_yx is None else input_origins_yx
    input_tile_size = tile_size if input_size is None else int(input_size)
    return StagePlane(
        name=name,
        phase_rad=phase,
        learned_support=support,
        learned_boxes_xyxy=_boxes_from_origins(origins_yx, tile_size),
        input_boxes_xyxy=_boxes_from_origins(input_origins, input_tile_size),
        tile_phase_rad=tuple(np.asarray(tile, dtype=np.float32) for tile in tiles),
    )


def _assemble_global(
    name: str,
    phase: np.ndarray,
    *,
    active_size: int,
    input_origins_yx: tuple[tuple[int, int], ...],
    input_size: int,
) -> StagePlane:
    phase = np.asarray(phase, dtype=np.float32)
    if phase.shape != (active_size, active_size):
        raise ValueError(
            f"{name}: global phase must be {(active_size, active_size)}, got {phase.shape}"
        )
    return StagePlane(
        name=name,
        phase_rad=phase,
        learned_support=np.ones_like(phase, dtype=bool),
        learned_boxes_xyxy=((0, 0, active_size, active_size),),
        input_boxes_xyxy=_boxes_from_origins(input_origins_yx, input_size),
        tile_phase_rad=(phase,),
    )


def build_stage_planes(
    model: LGVQSingleMetricOEO16, settings: ExperimentSettings
) -> dict[str, StagePlane]:
    """Map all learned phase tensors to canonical, unflipped 478x478 planes."""

    geometry = settings.geometry
    geometry.validate(formal=True)
    active = geometry.active_size

    lane_center_offset = (geometry.lane_size - geometry.parallel_expert_size) // 2
    parallel_center_origins = tuple(
        (top + lane_center_offset, left + lane_center_offset)
        for top, left in geometry.lane_origins
    )
    parallel_expert_origins = tuple(
        (lane_top + local_top, lane_left + local_left)
        for lane_top, lane_left in geometry.lane_origins
        for local_top, local_left in geometry.parallel_expert_origins
    )

    # The serial field is centered on the 518 simulation canvas.  Subtracting
    # the 20-pixel active margin gives its exact origin on the 478 plane.
    serial_center = (
        (geometry.canvas_size - geometry.serial_expert_size) // 2
        - geometry.active_margin
    )
    serial_center_origins = ((serial_center, serial_center),)

    values = {
        "vision_router": _assemble_tiles(
            "vision_router",
            _phase(model.parallel_router.raw_router_phase),
            parallel_center_origins,
            active_size=active,
        ),
        "vision_expert": _assemble_tiles(
            "vision_expert",
            _phase(model.parallel_optics.raw_expert_phase),
            parallel_expert_origins,
            active_size=active,
        ),
        "vision_global": _assemble_global(
            "vision_global",
            _phase(model.parallel_optics.raw_global_phase),
            active_size=active,
            input_origins_yx=parallel_center_origins,
            input_size=geometry.parallel_expert_size,
        ),
        "language_router": _assemble_tiles(
            "language_router",
            _phase(model.serial_router.raw_router_phase).reshape(
                1, geometry.serial_expert_size, geometry.serial_expert_size
            ),
            serial_center_origins,
            active_size=active,
        ),
        "language_expert": _assemble_tiles(
            "language_expert",
            _phase(model.serial_optics.raw_expert_phase),
            geometry.serial_expert_origins,
            active_size=active,
        ),
        "language_global": _assemble_global(
            "language_global",
            _phase(model.serial_optics.raw_global_phase),
            active_size=active,
            input_origins_yx=serial_center_origins,
            input_size=geometry.serial_expert_size,
        ),
    }
    if tuple(values) != STAGES:
        raise AssertionError("The six-stage export order changed unexpectedly")
    return values


def _load_checkpoint_model(
    settings: ExperimentSettings, checkpoint: Path
) -> tuple[LGVQSingleMetricOEO16, dict[str, Any]]:
    checkpoint = checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Single-metric checkpoint is missing: {checkpoint}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("state_dict"), dict):
        raise ValueError("Checkpoint must contain the training state_dict")
    expected_architecture = settings.architecture_label
    if payload.get("architecture") != expected_architecture:
        raise RuntimeError(
            "Checkpoint architecture mismatch: "
            f"expected {expected_architecture!r}, got {payload.get('architecture')!r}"
        )
    if payload.get("target_name") != settings.target_name:
        raise RuntimeError(
            "Spatial and Temporal checkpoints/masks must remain separate: "
            f"config={settings.target_name!r}, checkpoint={payload.get('target_name')!r}"
        )
    if payload.get("prompt") != settings.prompt:
        raise RuntimeError("Checkpoint prompt does not match the target-specific config")
    model = build_model(settings)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return model, payload


def _phase_stats(value: np.ndarray) -> dict[str, float]:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    phasor = np.exp(1j * array).mean()
    circular_mean = float(np.mod(np.angle(phasor), 2.0 * math.pi))
    residual = np.angle(np.exp(1j * (array - circular_mean)))
    resultant = max(float(abs(phasor)), np.finfo(np.float64).tiny)
    return {
        "phase_rad_min": float(array.min()),
        "phase_rad_max": float(array.max()),
        "phase_rad_mean_linear": float(array.mean()),
        "phase_rad_std_linear": float(array.std()),
        "phase_rad_circular_mean": circular_mean,
        "phase_rad_residual_std": float(residual.std()),
        "phase_circular_resultant_length": float(abs(phasor)),
        "phase_rad_circular_std": float(math.sqrt(max(0.0, -2.0 * math.log(resultant)))),
    }


def _save_amplitude_layout(
    plane: StagePlane,
    destination: Path,
    *,
    center_xy: tuple[float, float],
) -> dict[str, Any]:
    active = np.zeros((plane.phase_rad.shape[0], plane.phase_rad.shape[1]), dtype=np.uint8)
    for left, top, right, bottom in plane.input_boxes_xyxy:
        active[top:bottom, left:right] = 255
    canvas, bounds, actual_center = place_at_center(
        Image.fromarray(active, mode="L"),
        slm_size_wh=(1024, 1024),
        center_xy=center_xy,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, format="BMP")
    return {
        "path": str(destination.resolve()),
        "sha256": _sha256(destination),
        "size_wh": [1024, 1024],
        "mode": "L",
        "active_bounds_xyxy": list(bounds),
        "active_center_xy": list(actual_center),
        "white_support_pixels_in_active478": int(np.count_nonzero(active)),
        "purpose": "layout-only; replace white regions with sample-dependent amplitudes",
    }


def _save_previews(
    planes: dict[str, StagePlane], output_dir: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    preview_dir = output_dir / "preview"
    preview_dir.mkdir(parents=True, exist_ok=True)
    all_residual = []
    for plane in planes.values():
        values = plane.phase_rad[plane.learned_support]
        mean = _phase_stats(values)["phase_rad_circular_mean"]
        all_residual.append(np.angle(np.exp(1j * (values - mean))))
    residual_limit = max(
        0.05,
        min(math.pi, float(np.percentile(np.abs(np.concatenate(all_residual)), 99.0))),
    )

    combined, axes = plt.subplots(3, 4, figsize=(12.0, 9.0), constrained_layout=True)
    for index, (name, plane) in enumerate(planes.items()):
        stats = _phase_stats(plane.phase_rad[plane.learned_support])
        circular_mean = stats["phase_rad_circular_mean"]
        residual = np.angle(np.exp(1j * (plane.phase_rad - circular_mean)))
        residual = np.where(plane.learned_support, residual, np.nan)

        figure, local_axes = plt.subplots(1, 2, figsize=(8.0, 3.8), constrained_layout=True)
        for axis, image, cmap, vmin, vmax, title in (
            (local_axes[0], plane.phase_rad, "twilight", 0.0, 2.0 * math.pi, "absolute phase"),
            (local_axes[1], residual, "RdBu_r", -residual_limit, residual_limit, "circular-mean residual"),
        ):
            shown = axis.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
            for left, top, right, bottom in plane.learned_boxes_xyxy:
                axis.add_patch(
                    Rectangle(
                        (left - 0.5, top - 0.5),
                        right - left,
                        bottom - top,
                        fill=False,
                        edgecolor="white",
                        linewidth=0.25,
                        alpha=0.35,
                    )
                )
            axis.set_title(title, fontsize=9)
            axis.set_axis_off()
            figure.colorbar(shown, ax=axis, fraction=0.046, pad=0.02)
        figure.suptitle(
            f"{name} | tiles={len(plane.tile_phase_rad)} | residual std="
            f"{stats['phase_rad_residual_std']:.4f} rad",
            fontsize=10,
        )
        figure.savefig(preview_dir / f"{name}.png", dpi=180, facecolor="white")
        plt.close(figure)

        row, pair = divmod(index, 2)
        absolute_axis = axes[row, pair * 2]
        residual_axis = axes[row, pair * 2 + 1]
        absolute_axis.imshow(
            plane.phase_rad, cmap="twilight", vmin=0.0, vmax=2.0 * math.pi,
            interpolation="nearest",
        )
        residual_axis.imshow(
            residual, cmap="RdBu_r", vmin=-residual_limit, vmax=residual_limit,
            interpolation="nearest",
        )
        absolute_axis.set_title(f"{name}\nabsolute", fontsize=8)
        residual_axis.set_title(
            f"residual; std={stats['phase_rad_residual_std']:.3f} rad", fontsize=8
        )
        absolute_axis.set_axis_off()
        residual_axis.set_axis_off()
    combined.suptitle(
        "Six trained phase planes (canonical model orientation; hardware flip not applied)",
        fontsize=11,
    )
    combined.savefig(output_dir / "phase_preview.png", dpi=180, facecolor="white")
    plt.close(combined)

    layout, layout_axes = plt.subplots(2, 3, figsize=(9.0, 6.0), constrained_layout=True)
    for axis, (name, plane) in zip(layout_axes.flat, planes.items()):
        active = np.zeros_like(plane.phase_rad, dtype=np.uint8)
        for left, top, right, bottom in plane.input_boxes_xyxy:
            active[top:bottom, left:right] = 255
        axis.imshow(active, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        axis.set_title(f"{name} input support", fontsize=8)
        axis.set_axis_off()
    layout.suptitle("Amplitude-SLM stage layout templates (logical 478 x 478)", fontsize=10)
    layout.savefig(output_dir / "amplitude_layout_preview.png", dpi=180, facecolor="white")
    plt.close(layout)


def export_hardware_masks(
    settings: ExperimentSettings,
    checkpoint: Path,
    output_dir: Path,
    *,
    phase_center_xy: tuple[float, float] = (980.0, 590.0),
    amplitude_center_xy: tuple[float, float] = (512.0, 512.0),
    phase_flip_vertical: bool = True,
    phase_flip_horizontal: bool = False,
) -> dict[str, Any]:
    """Export one Spatial or Temporal checkpoint without changing its optics."""

    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model, checkpoint_payload = _load_checkpoint_model(settings, checkpoint)
    planes = build_stage_planes(model, settings)

    canonical_dir = output_dir / "logical_phase_478_canonical"
    payload_dir = output_dir / "phase_payload_478_hardware_orientation"
    native_dir = output_dir / "phase_slm_1920x1200"
    amplitude_dir = output_dir / "amplitude_layout_1024x1024"
    canonical_dir.mkdir(parents=True, exist_ok=True)
    payload_dir.mkdir(parents=True, exist_ok=True)
    amplitude_dir.mkdir(parents=True, exist_ok=True)

    amplitude_reports: dict[str, Any] = {}
    stage_reports: dict[str, Any] = {}
    tile_rows: list[dict[str, Any]] = []
    for plane in planes.values():
        canonical = encode_active_phase(plane.phase_rad)
        save_active_png(canonical, canonical_dir / f"{plane.name}.png")
        hardware_oriented = canonical
        if phase_flip_vertical:
            hardware_oriented = np.flip(hardware_oriented, axis=0)
        if phase_flip_horizontal:
            hardware_oriented = np.flip(hardware_oriented, axis=1)
        hardware_oriented = np.ascontiguousarray(hardware_oriented)
        save_active_png(hardware_oriented, payload_dir / f"{plane.name}.png")

        amplitude_reports[plane.name] = _save_amplitude_layout(
            plane,
            amplitude_dir / f"{plane.name}_layout_1024x1024.bmp",
            center_xy=amplitude_center_xy,
        )
        aggregate = _phase_stats(plane.phase_rad[plane.learned_support])
        stage_reports[plane.name] = {
            "logical_phase_png_canonical": str(
                (canonical_dir / f"{plane.name}.png").resolve()
            ),
            "logical_phase_png_canonical_sha256": _sha256(
                canonical_dir / f"{plane.name}.png"
            ),
            "phase_payload_png_hardware_orientation": str(
                (payload_dir / f"{plane.name}.png").resolve()
            ),
            "phase_payload_png_hardware_orientation_sha256": _sha256(
                payload_dir / f"{plane.name}.png"
            ),
            "learned_tile_count": len(plane.tile_phase_rad),
            "learned_pixel_count": int(plane.learned_support.sum()),
            "learned_fraction_of_active478": float(plane.learned_support.mean()),
            "learned_boxes_xyxy_canonical": [list(box) for box in plane.learned_boxes_xyxy],
            "input_support_boxes_xyxy_canonical": [list(box) for box in plane.input_boxes_xyxy],
            "phase_statistics_learned_pixels": aggregate,
        }
        for index, tile in enumerate(plane.tile_phase_rad):
            tile_rows.append(
                {
                    "stage": plane.name,
                    "tile_index": index,
                    "box_xyxy": ",".join(map(str, plane.learned_boxes_xyxy[index])),
                    **_phase_stats(tile),
                }
            )

    reconstruction = reconstruct_directory(
        payload_dir,
        native_dir,
        slm_size_wh=(1920, 1200),
        scale_factor=None,
        center_xy=phase_center_xy,
        logical_pixel_pitch_um=17.0,
        slm_pixel_pitch_um=8.0,
    )
    for name in STAGES:
        bmp = native_dir / f"{name}.bmp"
        with Image.open(bmp) as image:
            if image.mode != "L" or image.size != (1920, 1200):
                raise RuntimeError(f"Invalid native phase BMP contract: {bmp}")
        stage_reports[name]["phase_slm_bmp"] = str(bmp.resolve())
        stage_reports[name]["phase_slm_bmp_sha256"] = _sha256(bmp)

    # A generic all-white active aperture helps verify the amplitude placement
    # independently from the six sparse stage supports.
    aperture = Image.fromarray(np.full((478, 478), 255, dtype=np.uint8), mode="L")
    aperture_canvas, aperture_bounds, aperture_center = place_at_center(
        aperture,
        slm_size_wh=(1024, 1024),
        center_xy=amplitude_center_xy,
    )
    aperture_path = amplitude_dir / "active478_white_aperture_1024x1024.bmp"
    aperture_canvas.save(aperture_path, format="BMP")

    _save_previews(planes, output_dir)
    statistics_dir = output_dir / "statistics"
    statistics_dir.mkdir(parents=True, exist_ok=True)
    with (statistics_dir / "phase_tile_statistics.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(tile_rows[0]))
        writer.writeheader()
        writer.writerows(tile_rows)

    checkpoint = checkpoint.expanduser().resolve()
    physical_native_size = int(round(478 * 17.0 / 8.0))
    report = {
        "schema_version": 1,
        "target_name": settings.target_name,
        "prompt": settings.prompt,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "checkpoint_epoch": int(checkpoint_payload.get("epoch", -1)),
        "checkpoint_architecture": checkpoint_payload.get("architecture"),
        "stages_in_physical_order": list(STAGES),
        "optical_contract_unchanged": {
            "wavelength_nm": settings.wavelength_nm,
            "propagation_distance_m": settings.distance_m,
            "simulation_canvas": [settings.geometry.canvas_size] * 2,
            "logical_active_field": [478, 478],
            "logical_pixel_pitch_um": 17.0,
            "logical_active_extent_mm": 478 * 17.0 / 1000.0,
            "parallel_frame_layout": (
                f"{settings.geometry.lane_grid}x{settings.geometry.lane_grid}"
            ),
            "parallel_expert_layout_per_frame": "2x2",
            "parallel_expert_size": [
                settings.geometry.parallel_expert_size,
                settings.geometry.parallel_expert_size,
            ],
            "parallel_expert_count": settings.frame_count * 4,
            "serial_expert_size": [109, 109],
            "serial_expert_count": 4,
            "router": "optical Top-2",
        },
        "phase_slm": {
            "size_wh": [1920, 1200],
            "pixel_pitch_um": 8.0,
            "center_xy": list(phase_center_xy),
            "flip_vertical_before_physical_raster": bool(phase_flip_vertical),
            "flip_horizontal_before_physical_raster": bool(phase_flip_horizontal),
            "mapping": "17um logical pixel centers -> 8um native pixels by nearest physical coordinate",
            "native_active_size_wh": [physical_native_size, physical_native_size],
            "native_active_extent_mm": physical_native_size * 8.0 / 1000.0,
            "extent_error_um_vs_logical": physical_native_size * 8.0 - 478 * 17.0,
            "reconstruction": reconstruction,
        },
        "amplitude_slm": {
            "size_wh": [1024, 1024],
            "pixel_pitch_um": 17.0,
            "center_xy": list(amplitude_center_xy),
            "mapping": "native 1:1; no resize",
            "active_bounds_xyxy": list(aperture_bounds),
            "active_center_xy": list(aperture_center),
            "white_aperture": {
                "path": str(aperture_path.resolve()),
                "sha256": _sha256(aperture_path),
            },
            "stage_layout_templates": amplitude_reports,
            "warning": "layout-only templates are not sample-dependent network amplitudes",
        },
        "stages": stage_reports,
        "previews": {
            "combined_phase": str((output_dir / "phase_preview.png").resolve()),
            "amplitude_layout": str(
                (output_dir / "amplitude_layout_preview.png").resolve()
            ),
            "orientation": "canonical model orientation; hardware phase flip is shown only in payload/BMP files",
        },
        "statistics_csv": str(
            (statistics_dir / "phase_tile_statistics.csv").resolve()
        ),
    }
    _write_json(output_dir / "hardware_mask_export_report.json", report)
    _write_json(statistics_dir / "phase_statistics.json", stage_reports)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--phase-center-x", type=float, default=980.0)
    parser.add_argument("--phase-center-y", type=float, default=590.0)
    parser.add_argument("--amplitude-center-x", type=float, default=512.0)
    parser.add_argument("--amplitude-center-y", type=float, default=512.0)
    parser.add_argument(
        "--phase-flip-vertical",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep enabled to match the established 17um/8um phase export contract.",
    )
    parser.add_argument(
        "--phase-flip-horizontal",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    args = parser.parse_args()
    settings = load_settings(args.config)
    report = export_hardware_masks(
        settings,
        Path(args.checkpoint),
        Path(args.output_dir),
        phase_center_xy=(args.phase_center_x, args.phase_center_y),
        amplitude_center_xy=(args.amplitude_center_x, args.amplitude_center_y),
        phase_flip_vertical=args.phase_flip_vertical,
        phase_flip_horizontal=args.phase_flip_horizontal,
    )
    print(
        json.dumps(
            {
                "status": "exported",
                "target_name": report["target_name"],
                "stages": report["stages_in_physical_order"],
                "phase_bmp_dir": str(
                    Path(args.output_dir).expanduser().resolve()
                    / "phase_slm_1920x1200"
                ),
                "amplitude_layout_dir": str(
                    Path(args.output_dir).expanduser().resolve()
                    / "amplitude_layout_1024x1024"
                ),
                "report": str(
                    Path(args.output_dir).expanduser().resolve()
                    / "hardware_mask_export_report.json"
                ),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["STAGES", "StagePlane", "build_stage_planes", "export_hardware_masks", "main"]
