"""Export, capture-adapt, and evaluate the LGVQ model on the laboratory optics.

This module never opens hardware directly.  It creates allowlisted stage
folders consumed by ``experiments.hardware_sdk.workflows.acquire_folder`` and
uses the resulting canonical 478x478 CCD PNG files for sequential fine-tuning.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
import yaml

from experiments.hardware_sdk.workflows.reconstruct_slm import (
    encode_active_amplitude_with_metadata,
    encode_active_phase,
    reconstruct_directory,
    save_active_png,
)

from .data import load_frame_cache
from .hardware_contract import (
    FUSION_STAGES,
    OPTICAL_PASSES,
    PASS_DIRECTORIES,
    forward_hardware,
    phase_canvases,
)
from .metrics import regression_metrics
from .modeling import LGVQFourStageOEO
from .settings import ExperimentSettings, load_settings, resolved_dict
from .training import (
    batch_correlation_loss,
    pairwise_ranking_loss,
    weighted_smooth_l1_loss,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return value


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


SESSION_MANIFEST_FIELDS = (
    "order",
    "cache_index",
    "key",
    "sample_id",
    "split",
    "spatial_target",
    "temporal_target",
    "video_path",
)


def _normalize_session_rows(
    rows: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    seen_indices: set[int] = set()
    for position, row in enumerate(rows):
        missing = [field for field in SESSION_MANIFEST_FIELDS if field not in row]
        if missing:
            raise RuntimeError(
                f"Session manifest row {position} is missing fields: {missing}"
            )
        try:
            value = {
                "order": int(row["order"]),
                "cache_index": int(row["cache_index"]),
                "key": str(row["key"]),
                "sample_id": str(row["sample_id"]),
                "split": str(row["split"]),
                "spatial_target": float(row["spatial_target"]),
                "temporal_target": float(row["temporal_target"]),
                "video_path": str(row["video_path"]),
            }
        except (TypeError, ValueError) as error:
            raise RuntimeError(
                f"Session manifest row {position} contains an invalid value"
            ) from error
        if value["order"] != position:
            raise RuntimeError(
                "Session manifest order is not contiguous: "
                f"row={position}, declared={value['order']}"
            )
        if value["split"] not in {"train", "test"}:
            raise RuntimeError(
                f"Session manifest has unsupported split {value['split']!r}"
            )
        if not value["key"] or Path(value["key"]).name != value["key"]:
            raise RuntimeError(f"Unsafe/empty session key: {value['key']!r}")
        if value["key"] in seen_keys:
            raise RuntimeError(f"Duplicate session key: {value['key']}")
        if value["cache_index"] in seen_indices:
            raise RuntimeError(
                f"Duplicate session cache_index: {value['cache_index']}"
            )
        seen_keys.add(value["key"])
        seen_indices.add(value["cache_index"])
        normalized.append(value)
    if not normalized:
        raise RuntimeError("Session manifest is empty")
    if {row["split"] for row in normalized} != {"train", "test"}:
        raise RuntimeError("Session manifest must contain train and test samples")
    return normalized


def _same_session_rows(
    observed: Iterable[Mapping[str, Any]],
    expected: Iterable[Mapping[str, Any]],
) -> bool:
    return _normalize_session_rows(observed) == _normalize_session_rows(expected)


def _safe_key(index: int, split: str, sample_id: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", sample_id).strip("_")[:48] or "sample"
    digest = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()[:10]
    return f"{split}__{index:05d}__{slug}__{digest}"


def _load_model(
    settings: ExperimentSettings, checkpoint: Path, device: torch.device
) -> tuple[LGVQFourStageOEO, dict[str, Any]]:
    checkpoint = checkpoint.expanduser().resolve()
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(saved, dict) or "state_dict" not in saved:
        raise RuntimeError(f"Unsupported checkpoint payload: {checkpoint}")
    if saved.get("architecture") != settings.architecture_label:
        raise RuntimeError(
            "Checkpoint architecture mismatch: "
            f"{saved.get('architecture')!r} != {settings.architecture_label!r}"
        )
    model = LGVQFourStageOEO(settings)
    model.load_state_dict(saved["state_dict"], strict=True)
    model.to(device)
    return model, saved


def _stage_dir(session_dir: Path, optical_pass: str) -> Path:
    return session_dir / PASS_DIRECTORIES[optical_pass]


def _phase_export_array(
    phase: torch.Tensor, *, flip_vertical: bool, flip_horizontal: bool
) -> np.ndarray:
    value = phase.detach().float().cpu().numpy()
    if flip_vertical:
        value = np.flip(value, axis=0)
    if flip_horizontal:
        value = np.flip(value, axis=1)
    return np.ascontiguousarray(value)


def _phase_statistics(value: torch.Tensor) -> dict[str, float]:
    phase = value.detach().float()
    learned = phase[phase != 0.0]
    if not learned.numel():
        learned = phase.reshape(-1)
    phasor = torch.exp(1j * learned.to(torch.complex64)).mean()
    circular_std = math.sqrt(max(0.0, -2.0 * math.log(max(1.0e-8, abs(complex(phasor))))))
    return {
        "learned_support_pixels": int(learned.numel()),
        "minimum_rad": float(learned.min()),
        "maximum_rad": float(learned.max()),
        "mean_rad": float(learned.mean()),
        "linear_std_rad": float(learned.std(unbiased=False)),
        "circular_std_rad": float(circular_std),
    }


def _draw_phase_gallery(
    canvases: Mapping[str, torch.Tensor],
    output_dir: Path,
    *,
    flip_vertical: bool,
    flip_horizontal: bool,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 7,
            "axes.titlesize": 7,
            "axes.labelsize": 7,
        }
    )
    names = list(OPTICAL_PASSES)
    figure, axes = plt.subplots(2, 3, figsize=(7.1, 4.45), constrained_layout=True)
    image = None
    for axis, name in zip(axes.flat, names):
        image = axis.imshow(
            canvases[name].detach().cpu().numpy(),
            cmap="twilight",
            vmin=0.0,
            vmax=2.0 * math.pi,
            interpolation="nearest",
        )
        axis.set_title(name.replace("_", " "))
        axis.set_xticks((0, 239, 477))
        axis.set_yticks((0, 239, 477))
        axis.set_xlabel("logical x (17 μm)")
        axis.set_ylabel("logical y (17 μm)")
    assert image is not None
    colorbar = figure.colorbar(image, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
    colorbar.set_label("wrapped phase (rad)")
    colorbar.set_ticks((0.0, math.pi, 2.0 * math.pi), labels=("0", "π", "2π"))
    figure.savefig(output_dir / "phase_masks_logical_overview.png", dpi=300)
    figure.savefig(output_dir / "phase_masks_logical_overview.pdf")
    plt.close(figure)
    figure, axes = plt.subplots(2, 3, figsize=(7.1, 4.45), constrained_layout=True)
    residual_image = None
    for axis, name in zip(axes.flat, names):
        value = canvases[name].detach().float().cpu()
        support = value != 0.0
        mean_angle = torch.angle(torch.exp(1j * value[support].to(torch.complex64)).mean())
        residual = torch.atan2(torch.sin(value - mean_angle), torch.cos(value - mean_angle))
        array = residual.numpy()
        array[~support.numpy()] = np.nan
        residual_image = axis.imshow(
            array,
            cmap="coolwarm",
            vmin=-math.pi,
            vmax=math.pi,
            interpolation="nearest",
        )
        axis.set_title(name.replace("_", " "))
        axis.set_xticks((0, 239, 477))
        axis.set_yticks((0, 239, 477))
    assert residual_image is not None
    colorbar = figure.colorbar(
        residual_image, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02
    )
    colorbar.set_label("phase relative to circular mean (rad)")
    colorbar.set_ticks((-math.pi, 0.0, math.pi), labels=("−π", "0", "π"))
    figure.savefig(output_dir / "phase_masks_relative_overview.png", dpi=300)
    figure.savefig(output_dir / "phase_masks_relative_overview.pdf")
    plt.close(figure)

    figure, axes = plt.subplots(2, 3, figsize=(7.1, 4.45), constrained_layout=True)
    for axis, name in zip(axes.flat, names):
        exported = _phase_export_array(
            canvases[name],
            flip_vertical=flip_vertical,
            flip_horizontal=flip_horizontal,
        )
        axis.imshow(encode_active_phase(exported), cmap="gray", vmin=0, vmax=255)
        axis.set_title(name.replace("_", " "))
        axis.set_xticks((0, 239, 477))
        axis.set_yticks((0, 239, 477))
    figure.savefig(output_dir / "phase_masks_export_uint8_overview.png", dpi=300)
    plt.close(figure)

    from matplotlib.patches import Rectangle

    figure, axes = plt.subplots(1, 2, figsize=(7.1, 3.25), constrained_layout=True)
    parallel, serial = axes
    parallel.set_title("Parallel router: 4 frames × 4 detector ROIs")
    for lane, (top, left) in enumerate(((0, 0), (0, 246), (246, 0), (246, 246))):
        parallel.add_patch(Rectangle((left, top), 232, 232, fill=False, edgecolor="0.5", linewidth=0.8))
        parallel.text(left + 4, top + 12, f"frame {lane + 1}", color="0.25")
        expert = 0
        for y0, y1 in ((79, 108), (124, 153)):
            for x0, x1 in ((79, 108), (124, 153)):
                parallel.add_patch(
                    Rectangle(
                        (left + x0, top + y0),
                        x1 - x0,
                        y1 - y0,
                        fill=False,
                        edgecolor=f"C{expert}",
                        linewidth=1.0,
                    )
                )
                parallel.text(
                    left + x0 + 2,
                    top + y0 + 10,
                    f"E{expert + 1}",
                    color=f"C{expert}",
                )
                expert += 1
    serial.set_title("Serial router: 1 sequence × 4 detector ROIs")
    expert = 0
    for y0, y1 in ((164, 223), (255, 314)):
        for x0, x1 in ((164, 223), (255, 314)):
            serial.add_patch(
                Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=f"C{expert}", linewidth=1.2)
            )
            serial.text(x0 + 3, y0 + 15, f"E{expert + 1}", color=f"C{expert}")
            expert += 1
    for axis in axes:
        axis.set_xlim(0, 478)
        axis.set_ylim(478, 0)
        axis.set_aspect("equal")
        axis.set_xlabel("canonical CCD x")
        axis.set_ylabel("canonical CCD y")
        axis.set_xticks((0, 239, 477))
        axis.set_yticks((0, 239, 477))
        axis.grid(False)
    figure.savefig(output_dir / "router_detector_rois.png", dpi=300)
    figure.savefig(output_dir / "router_detector_rois.pdf")
    plt.close(figure)

    # Individual learned 109x109 tiles, shown without the large inactive canvas.
    lane_origins = ((0, 0), (0, 246), (246, 0), (246, 246))
    local_center = (232 - 109) // 2
    stage1_expert_tiles = []
    for lane_top, lane_left in lane_origins:
        for local_top in (0, 123):
            for local_left in (0, 123):
                stage1_expert_tiles.append(
                    canvases["stage1_expert"][
                        lane_top + local_top : lane_top + local_top + 109,
                        lane_left + local_left : lane_left + local_left + 109,
                    ]
                )
    figure, axes = plt.subplots(4, 4, figsize=(6.8, 6.1), constrained_layout=True)
    tile_image = None
    for expert, (axis, tile) in enumerate(zip(axes.flat, stage1_expert_tiles)):
        tile_image = axis.imshow(
            tile.detach().cpu().numpy(), cmap="twilight", vmin=0.0, vmax=2.0 * math.pi,
            interpolation="nearest",
        )
        axis.set_title(f"frame {expert // 4 + 1}, expert {expert % 4 + 1}")
        axis.set_xticks(())
        axis.set_yticks(())
    assert tile_image is not None
    colorbar = figure.colorbar(tile_image, ax=axes.ravel().tolist(), fraction=0.018, pad=0.015)
    colorbar.set_label("wrapped phase (rad)")
    colorbar.set_ticks((0.0, math.pi, 2.0 * math.pi), labels=("0", "π", "2π"))
    figure.savefig(output_dir / "phase_expert_tiles_stage1.png", dpi=300)
    figure.savefig(output_dir / "phase_expert_tiles_stage1.pdf")
    plt.close(figure)

    detail_tiles: list[tuple[str, torch.Tensor | None]] = []
    for lane, (top, left) in enumerate(lane_origins):
        detail_tiles.append(
            (
                f"stage 1 router, frame {lane + 1}",
                canvases["stage1_router"][
                    top + local_center : top + local_center + 109,
                    left + local_center : left + local_center + 109,
                ],
            )
        )
    serial_router_offset = (478 - 109) // 2
    detail_tiles.extend(
        [
            (
                "stage 3 router",
                canvases["stage3_router"][
                    serial_router_offset : serial_router_offset + 109,
                    serial_router_offset : serial_router_offset + 109,
                ],
            ),
            ("", None),
            ("", None),
            ("", None),
        ]
    )
    for expert, (top, left) in enumerate(
        ((123, 123), (123, 246), (246, 123), (246, 246))
    ):
        detail_tiles.append(
            (
                f"stage 3 expert {expert + 1}",
                canvases["stage3_expert"][top : top + 109, left : left + 109],
            )
        )
    figure, axes = plt.subplots(3, 4, figsize=(6.8, 5.25), constrained_layout=True)
    detail_image = None
    for axis, (title, tile) in zip(axes.flat, detail_tiles):
        if tile is None:
            axis.axis("off")
            continue
        detail_image = axis.imshow(
            tile.detach().cpu().numpy(), cmap="twilight", vmin=0.0, vmax=2.0 * math.pi,
            interpolation="nearest",
        )
        axis.set_title(title)
        axis.set_xticks(())
        axis.set_yticks(())
    assert detail_image is not None
    colorbar = figure.colorbar(detail_image, ax=axes.ravel().tolist(), fraction=0.018, pad=0.015)
    colorbar.set_label("wrapped phase (rad)")
    colorbar.set_ticks((0.0, math.pi, 2.0 * math.pi), labels=("0", "π", "2π"))
    figure.savefig(output_dir / "phase_router_and_stage3_tiles.png", dpi=300)
    figure.savefig(output_dir / "phase_router_and_stage3_tiles.pdf")
    plt.close(figure)

    figure, axes = plt.subplots(2, 3, figsize=(7.1, 3.6), constrained_layout=True)
    for axis, name in zip(axes.flat, names):
        value = canvases[name].detach().float().cpu().reshape(-1)
        learned = value[value != 0.0]
        axis.hist(learned.numpy(), bins=48, range=(0.0, 2.0 * math.pi), color="#2673b8", linewidth=0)
        axis.set_title(name.replace("_", " "))
        axis.set_xlim(0.0, 2.0 * math.pi)
        axis.set_xticks((0.0, math.pi, 2.0 * math.pi), labels=("0", "π", "2π"))
        axis.set_ylabel("pixels")
    figure.savefig(output_dir / "phase_value_histograms.png", dpi=300)
    figure.savefig(output_dir / "phase_value_histograms.pdf")
    plt.close(figure)


def _draw_physical_phase_gallery(
    output_dir: Path,
    *,
    phase_center_xy: tuple[float, float],
) -> None:
    """Show the actual 1920x1200 BMP canvases and the intended 1016px aperture."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 7,
            "axes.titlesize": 7,
            "axes.labelsize": 7,
        }
    )
    active_span = int(round(478 * 17.0 / 8.0))
    left = phase_center_xy[0] - active_span / 2.0
    top = phase_center_xy[1] - active_span / 2.0
    figure, axes = plt.subplots(2, 3, figsize=(7.1, 3.65), constrained_layout=True)
    for axis, name in zip(axes.flat, OPTICAL_PASSES):
        path = output_dir / PASS_DIRECTORIES[name] / "phase_to_play" / f"{name}.bmp"
        with Image.open(path) as image:
            value = np.asarray(image.convert("L"))
        axis.imshow(value, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        axis.add_patch(
            Rectangle(
                (left, top),
                active_span,
                active_span,
                fill=False,
                edgecolor="#e64b35",
                linewidth=0.7,
            )
        )
        axis.plot(*phase_center_xy, marker="+", color="#00a087", markersize=5, markeredgewidth=0.8)
        axis.set_title(name.replace("_", " "))
        axis.set_xlim(0, 1919)
        axis.set_ylim(1199, 0)
        axis.set_xticks((0, int(phase_center_xy[0]), 1919))
        axis.set_yticks((0, int(phase_center_xy[1]), 1199))
        axis.set_xlabel("phase SLM x (8 μm)")
        axis.set_ylabel("phase SLM y (8 μm)")
    figure.suptitle(
        f"Physical phase BMPs: red = {active_span}×{active_span}, green = center "
        f"({phase_center_xy[0]:g},{phase_center_xy[1]:g})",
        fontsize=7,
    )
    figure.savefig(output_dir / "visualization" / "phase_masks_physical_1920x1200.png", dpi=300)
    figure.savefig(output_dir / "visualization" / "phase_masks_physical_1920x1200.pdf")
    plt.close(figure)


def export_phase_masks(
    settings: ExperimentSettings,
    checkpoint: Path,
    output_dir: Path,
    *,
    phase_center_xy: tuple[float, float],
    flip_vertical: bool,
    flip_horizontal: bool,
    device: torch.device,
) -> dict[str, Any]:
    """Export all six final learned phase planes for review and deployment."""

    model, saved = _load_model(settings, checkpoint, device)
    model.eval()
    canvases = phase_canvases(model)
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    gallery_dir = output_dir / "visualization"
    gallery_dir.mkdir(exist_ok=True)
    _draw_phase_gallery(
        canvases,
        gallery_dir,
        flip_vertical=flip_vertical,
        flip_horizontal=flip_horizontal,
    )
    records: list[dict[str, Any]] = []
    for name, phase in canvases.items():
        destination = _stage_dir(output_dir, name)
        compact = destination / "compact_phase"
        full = destination / "phase_to_play"
        compact.mkdir(parents=True, exist_ok=True)
        full.mkdir(parents=True, exist_ok=True)
        exported = _phase_export_array(
            phase,
            flip_vertical=flip_vertical,
            flip_horizontal=flip_horizontal,
        )
        encoded = encode_active_phase(exported)
        compact_path = compact / f"{name}.png"
        save_active_png(encoded, compact_path)
        reconstruction = reconstruct_directory(
            compact,
            full,
            slm_size_wh=(1920, 1200),
            scale_factor=None,
            center_xy=phase_center_xy,
            logical_pixel_pitch_um=settings.pixel_pitch_um,
            slm_pixel_pitch_um=8.0,
        )
        bmp_path = full / f"{name}.bmp"
        records.append(
            {
                "optical_pass": name,
                "checkpoint_epoch": int(saved.get("epoch", -1)),
                "logical_phase_png": str(compact_path.relative_to(output_dir)),
                "logical_phase_sha256": _sha256(compact_path),
                "physical_phase_bmp": str(bmp_path.relative_to(output_dir)),
                "physical_phase_sha256": _sha256(bmp_path),
                "logical_size_wh": [478, 478],
                "physical_size_wh": [1920, 1200],
                "active_physical_center_xy": reconstruction["active_center_xy"],
                "flip_vertical_before_raster": bool(flip_vertical),
                "flip_horizontal_before_raster": bool(flip_horizontal),
                **_phase_statistics(phase),
            }
        )
    _draw_physical_phase_gallery(output_dir, phase_center_xy=phase_center_xy)
    report = {
        "schema_version": 1,
        "checkpoint": str(checkpoint.expanduser().resolve()),
        "checkpoint_sha256": _sha256(checkpoint.expanduser().resolve()),
        "checkpoint_epoch": int(saved.get("epoch", -1)),
        "architecture": saved.get("architecture"),
        "physical_pass_order": list(OPTICAL_PASSES),
        "four_frame_parallel_contract": (
            "stage1_router, stage1_expert, and stage2_global carry four video frames "
            "concurrently, one frame in each 232x232 lane"
        ),
        "phase_encoding": "floor(mod(rad,2pi)/(2pi)*256), uint8",
        "logical_pixel_pitch_um": settings.pixel_pitch_um,
        "phase_slm_pixel_pitch_um": 8.0,
        "phase_slm_size_wh": [1920, 1200],
        "phase_slm_center_xy": list(phase_center_xy),
        "flip_vertical_before_raster": bool(flip_vertical),
        "flip_horizontal_before_raster": bool(flip_horizontal),
        "records": records,
    }
    _json(output_dir / "phase_export_report.json", report)
    _write_csv(output_dir / "phase_export_manifest.csv", records)
    return report


def _select_manifest(
    payload: Mapping[str, Any], *, max_train: int | None, max_test: int | None
) -> list[dict[str, Any]]:
    counters = defaultdict(int)
    limits = {"train": max_train, "test": max_test}
    rows: list[dict[str, Any]] = []
    for cache_index, (sample_id, split, target) in enumerate(
        zip(payload["sample_ids"], payload["splits"], payload["targets"])
    ):
        split = str(split)
        limit = limits[split]
        if limit is not None and counters[split] >= limit:
            continue
        key = _safe_key(cache_index, split, str(sample_id))
        rows.append(
            {
                "order": len(rows),
                "cache_index": cache_index,
                "key": key,
                "sample_id": str(sample_id),
                "split": split,
                "spatial_target": float(target[0]),
                "temporal_target": float(target[1]),
                "video_path": str(payload["video_paths"][cache_index]),
            }
        )
        counters[split] += 1
    if not rows or not counters["train"] or not counters["test"]:
        raise RuntimeError("Hardware manifest must contain both train and test samples")
    return rows


class CcdFolderLoader:
    def __init__(self, session_dir: Path) -> None:
        self.session_dir = session_dir
        self.cache: dict[tuple[str, str], torch.Tensor] = {}

    def __call__(
        self, optical_pass: str, keys: list[str], device: torch.device
    ) -> torch.Tensor:
        directory = _stage_dir(self.session_dir, optical_pass) / "ccd_captured"
        values: list[torch.Tensor] = []
        for key in keys:
            cache_key = (optical_pass, key)
            if cache_key in self.cache:
                values.append(self.cache[cache_key])
                continue
            candidates = [directory / f"{key}{suffix}" for suffix in (".png", ".tif", ".tiff", ".npy")]
            source = next((path for path in candidates if path.is_file()), None)
            if source is None:
                raise FileNotFoundError(
                    f"CCD frame missing for pass={optical_pass}, key={key}: {directory}"
                )
            if source.suffix.lower() == ".npy":
                array = np.load(source, allow_pickle=False)
            else:
                with Image.open(source) as image:
                    array = np.asarray(image.convert("L"), dtype=np.float32)
            if array.shape != (478, 478):
                raise ValueError(f"CCD must be canonical 478x478, got {array.shape}: {source}")
            if not np.isfinite(array).all():
                raise ValueError(f"CCD contains NaN/Inf: {source}")
            tensor = torch.from_numpy(np.array(array, dtype=np.float32, copy=True))
            if float(tensor.min()) >= 0.0 and float(tensor.max()) <= 255.0:
                tensor = tensor.round().to(torch.uint8)
            self.cache[cache_key] = tensor
            values.append(tensor)
        return torch.stack(values).to(device)


def _ensure_session_manifest(
    session_dir: Path,
    payload: Mapping[str, Any],
    *,
    frame_cache: Path,
    max_train: int | None,
    max_test: int | None,
) -> list[dict[str, str]]:
    """Create one immutable sample allowlist, or verify it byte-for-byte.

    A later optical pass must never silently expand, shrink, reorder, or retarget
    the sample set selected by the first pass.  The seal also binds the exact
    frame-cache bytes, so copying an identically named but different cache into
    a laboratory bundle fails before any amplitude export or fine-tuning.
    """

    session_dir = session_dir.expanduser().resolve()
    cache_path = frame_cache.expanduser().resolve()
    if not cache_path.is_file():
        raise FileNotFoundError(f"Frame cache is missing: {cache_path}")
    path = session_dir / "session_manifest.csv"
    identity_path = session_dir / "session_identity.json"
    expected = _select_manifest(
        payload, max_train=max_train, max_test=max_test
    )
    if path.exists() != identity_path.exists():
        raise RuntimeError(
            "Incomplete session identity: session_manifest.csv and "
            "session_identity.json must either both exist or both be absent"
        )
    if path.is_file():
        identity = _read_json(identity_path)
        sealed_limits = identity.get("selection_limits")
        requested_limits = {"max_train": max_train, "max_test": max_test}
        if sealed_limits != requested_limits:
            raise RuntimeError(
                "Requested sample limits differ from the sealed session: "
                f"sealed={sealed_limits}, requested={requested_limits}"
            )
        return _load_sealed_session_manifest(
            session_dir, payload, frame_cache=cache_path
        )

    normalized = _normalize_session_rows(expected)
    _write_csv(path, expected)
    observed = _read_csv(path)
    if not _same_session_rows(observed, normalized):
        raise RuntimeError("Newly written session manifest failed round-trip validation")
    split_counts = {
        split: sum(row["split"] == split for row in normalized)
        for split in ("train", "test")
    }
    identity = {
        "schema_version": 1,
        "frame_cache_sha256": _sha256(cache_path),
        "frame_cache_bytes": cache_path.stat().st_size,
        "frame_cache_sample_count": len(payload["sample_ids"]),
        "session_manifest_sha256": _sha256(path),
        "session_rows_canonical_sha256": _canonical_sha256(normalized),
        "sample_count": len(normalized),
        "split_counts": split_counts,
        "selection_limits": {"max_train": max_train, "max_test": max_test},
    }
    _json(identity_path, identity)
    return _load_sealed_session_manifest(
        session_dir, payload, frame_cache=cache_path
    )


def _load_sealed_session_manifest(
    session_dir: Path,
    payload: Mapping[str, Any],
    *,
    frame_cache: Path,
) -> list[dict[str, str]]:
    session_dir = session_dir.expanduser().resolve()
    cache_path = frame_cache.expanduser().resolve()
    path = session_dir / "session_manifest.csv"
    identity_path = session_dir / "session_identity.json"
    if not path.is_file() or not identity_path.is_file():
        raise FileNotFoundError(
            "Sealed session identity is missing; export the first optical pass "
            "before capture/fine-tuning"
        )
    if not cache_path.is_file():
        raise FileNotFoundError(f"Frame cache is missing: {cache_path}")
    identity = _read_json(identity_path)
    if int(identity.get("schema_version", -1)) != 1:
        raise RuntimeError("Unsupported session identity schema")
    if identity.get("frame_cache_sha256") != _sha256(cache_path):
        raise RuntimeError("Frame cache SHA-256 differs from the sealed session")
    if int(identity.get("frame_cache_bytes", -1)) != cache_path.stat().st_size:
        raise RuntimeError("Frame cache byte size differs from the sealed session")
    if int(identity.get("frame_cache_sample_count", -1)) != len(
        payload["sample_ids"]
    ):
        raise RuntimeError("Frame cache sample count differs from the sealed session")
    if identity.get("session_manifest_sha256") != _sha256(path):
        raise RuntimeError("session_manifest.csv was modified after it was sealed")

    rows = _read_csv(path)
    normalized = _normalize_session_rows(rows)
    if identity.get("session_rows_canonical_sha256") != _canonical_sha256(normalized):
        raise RuntimeError("Session sample identity differs from its canonical seal")
    limits = identity.get("selection_limits")
    if not isinstance(limits, dict) or set(limits) != {"max_train", "max_test"}:
        raise RuntimeError("Session identity has invalid selection_limits")
    expected = _select_manifest(
        payload,
        max_train=limits["max_train"],
        max_test=limits["max_test"],
    )
    if not _same_session_rows(rows, expected):
        raise RuntimeError(
            "Sealed session manifest does not match the selected frame-cache rows"
        )
    if int(identity.get("sample_count", -1)) != len(rows):
        raise RuntimeError("Session sample count differs from its identity seal")
    split_counts = {
        split: sum(row["split"] == split for row in normalized)
        for split in ("train", "test")
    }
    if identity.get("split_counts") != split_counts:
        raise RuntimeError("Session split counts differ from their identity seal")
    return rows


def _router_rows(
    routing: Mapping[str, Mapping[str, Any]], keys: list[str]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for stage, result in routing.items():
        probabilities = result["probabilities"].detach().cpu()
        selected = result["selected_mask"].detach().cpu()
        if probabilities.shape[0] != len(keys):
            continue
        lanes = probabilities.shape[1] if probabilities.ndim == 3 else 1
        if probabilities.ndim == 2:
            probabilities = probabilities[:, None, :]
            selected = selected[:, None, :]
        for sample, key in enumerate(keys):
            for lane in range(lanes):
                row: dict[str, Any] = {"key": key, "router_stage": stage, "frame_lane": lane}
                for expert in range(4):
                    row[f"probability_{expert}"] = float(probabilities[sample, lane, expert])
                    row[f"selected_{expert}"] = int(selected[sample, lane, expert])
                rows.append(row)
    return rows


def _plain_filename(value: Any, *, label: str, suffix: str | None = None) -> str:
    name = str(value).strip()
    path = Path(name)
    if not name or path.name != name or name in {".", ".."}:
        raise RuntimeError(f"{label} must be a plain filename, got {name!r}")
    if suffix is not None and path.suffix.lower() != suffix.lower():
        raise RuntimeError(f"{label} must end in {suffix}, got {name!r}")
    return name


def _unique_rows(
    rows: Iterable[Mapping[str, Any]], *, column: str, label: str
) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for position, row in enumerate(rows):
        if column not in row:
            raise RuntimeError(f"{label} has no {column!r} column")
        key = str(row[column]).strip()
        if not key:
            raise RuntimeError(f"{label} row {position} has an empty {column}")
        if key in result:
            raise RuntimeError(f"{label} contains duplicate {column}={key!r}")
        result[key] = row
    return result


def _csv_bool(value: Any, *, label: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise RuntimeError(f"{label} is not a boolean: {value!r}")


def _sha_field(row: Mapping[str, Any], field: str, *, label: str) -> str:
    value = str(row.get(field, "")).strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        raise RuntimeError(f"{label} has no valid {field}")
    return value


def _validate_uint8_image(
    path: Path, *, size_wh: tuple[int, int], label: str
) -> None:
    try:
        with Image.open(path) as image:
            if image.mode != "L" or tuple(image.size) != tuple(size_wh):
                raise RuntimeError(
                    f"{label} must be uint8 grayscale {size_wh}, got "
                    f"mode={image.mode}, size={image.size}: {path}"
                )
    except OSError as error:
        raise RuntimeError(f"Could not read {label}: {path}") from error


def _validate_ccd_file(path: Path, *, label: str) -> None:
    if path.suffix.lower() == ".npy":
        array = np.load(path, allow_pickle=False)
        if array.shape != (478, 478) or array.dtype != np.uint8:
            raise RuntimeError(
                f"{label} must be uint8 [478,478], got {array.dtype} {array.shape}"
            )
        return
    _validate_uint8_image(path, size_wh=(478, 478), label=label)


def _capture_passes_for_stage(stage: str) -> tuple[str, ...]:
    passes = tuple(FUSION_STAGES[stage])
    stage_number = int(stage.removeprefix("stage"))
    if stage_number == 1:
        return passes
    previous = tuple(FUSION_STAGES[f"stage{stage_number - 1}"])
    return passes[len(previous) :]


def validate_capture_pass(
    session_dir: Path,
    optical_pass: str,
    session_rows: Iterable[Mapping[str, Any]],
    *,
    expected_checkpoint: Path | None = None,
) -> dict[str, Any]:
    """Validate and seal one exact amplitude/phase/CCD acquisition.

    This is deliberately stricter than loading same-named PNG files.  It binds
    every captured CCD to the reconstructed amplitude BMP, manually loaded
    phase BMP, detector homography, and the immutable session sample keys.
    """

    if optical_pass not in OPTICAL_PASSES:
        raise ValueError(f"Unknown optical pass {optical_pass!r}")
    session_dir = session_dir.expanduser().resolve()
    destination = _stage_dir(session_dir, optical_pass)
    expected = _normalize_session_rows(session_rows)
    expected_keys = [str(row["key"]) for row in expected]
    expected_key_set = set(expected_keys)
    expected_order = {str(row["key"]): int(row["order"]) for row in expected}

    export_report_path = destination / "export_report.json"
    if not export_report_path.is_file():
        raise FileNotFoundError(f"Stage export report is missing: {export_report_path}")
    export_report = _read_json(export_report_path)
    if export_report.get("optical_pass") != optical_pass:
        raise RuntimeError("Stage export report names the wrong optical pass")
    if int(export_report.get("sample_count", -1)) != len(expected_keys):
        raise RuntimeError("Stage export sample count differs from the sealed session")
    if expected_checkpoint is not None:
        checkpoint = expected_checkpoint.expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Expected stage checkpoint is missing: {checkpoint}")
        if export_report.get("checkpoint_sha256") != _sha256(checkpoint):
            raise RuntimeError(
                f"{optical_pass} was exported from a different checkpoint"
            )

    compact_manifest_path = destination / "amplitude_manifest.csv"
    if not compact_manifest_path.is_file():
        raise FileNotFoundError(
            f"Compact amplitude manifest is missing: {compact_manifest_path}"
        )
    compact_rows = _unique_rows(
        _read_csv(compact_manifest_path),
        column="key",
        label="compact amplitude manifest",
    )
    if set(compact_rows) != expected_key_set:
        raise RuntimeError(
            f"{optical_pass} compact amplitude keys differ from the sealed session"
        )

    compact_dir = destination / "compact_amplitude"
    reconstruction_path = destination / "amplitude_to_play" / "reconstruction_manifest.csv"
    if not reconstruction_path.is_file():
        raise FileNotFoundError(
            f"Amplitude reconstruction manifest is missing: {reconstruction_path}"
        )
    reconstruction_by_key: dict[str, Mapping[str, Any]] = {}
    for row in _read_csv(reconstruction_path):
        output_name = _plain_filename(
            row.get("output_bmp"), label="reconstructed amplitude", suffix=".bmp"
        )
        key = Path(output_name).stem
        if key in reconstruction_by_key:
            raise RuntimeError(f"Duplicate reconstructed amplitude key: {key}")
        reconstruction_by_key[key] = row
    if set(reconstruction_by_key) != expected_key_set:
        raise RuntimeError(
            f"{optical_pass} reconstructed amplitude keys differ from the sealed session"
        )

    amplitude_records: dict[str, dict[str, Any]] = {}
    for key in expected_keys:
        compact_row = compact_rows[key]
        compact_name = _plain_filename(
            compact_row.get("amplitude_file"),
            label=f"compact amplitude for {key}",
            suffix=".png",
        )
        if Path(compact_name).stem != key:
            raise RuntimeError(f"Compact amplitude filename/key mismatch for {key}")
        compact_path = compact_dir / compact_name
        if not compact_path.is_file():
            raise FileNotFoundError(f"Compact amplitude is missing: {compact_path}")
        _validate_uint8_image(
            compact_path,
            size_wh=(478, 478),
            label=f"compact amplitude for {key}",
        )
        compact_sha = _sha256(compact_path)
        if _sha_field(
            compact_row,
            "amplitude_sha256",
            label=f"compact amplitude row {key}",
        ) != compact_sha:
            raise RuntimeError(f"Compact amplitude SHA-256 mismatch for {key}")

        reconstruction = reconstruction_by_key[key]
        source_name = _plain_filename(
            reconstruction.get("source_png"),
            label=f"amplitude reconstruction source for {key}",
            suffix=".png",
        )
        output_name = _plain_filename(
            reconstruction.get("output_bmp"),
            label=f"amplitude reconstruction output for {key}",
            suffix=".bmp",
        )
        if source_name != compact_name or output_name != f"{key}.bmp":
            raise RuntimeError(f"Amplitude reconstruction filename mismatch for {key}")
        if _sha_field(
            reconstruction, "source_sha256", label=f"amplitude reconstruction {key}"
        ) != compact_sha:
            raise RuntimeError(f"Amplitude reconstruction source SHA mismatch for {key}")
        amplitude_path = destination / "amplitude_to_play" / output_name
        if not amplitude_path.is_file():
            raise FileNotFoundError(f"Reconstructed amplitude is missing: {amplitude_path}")
        _validate_uint8_image(
            amplitude_path,
            size_wh=(1024, 1024),
            label=f"reconstructed amplitude for {key}",
        )
        amplitude_sha = _sha256(amplitude_path)
        if _sha_field(
            reconstruction, "output_sha256", label=f"amplitude reconstruction {key}"
        ) != amplitude_sha:
            raise RuntimeError(f"Reconstructed amplitude SHA mismatch for {key}")
        amplitude_records[key] = {
            "path": amplitude_path,
            "sha256": amplitude_sha,
        }

    phase_dir = destination / "phase_to_play"
    phase_path = phase_dir / f"{optical_pass}.bmp"
    phase_manifest_path = phase_dir / "reconstruction_manifest.csv"
    if not phase_path.is_file() or not phase_manifest_path.is_file():
        raise FileNotFoundError(
            f"Phase BMP/reconstruction manifest is incomplete for {optical_pass}"
        )
    phase_manifest_rows = _read_csv(phase_manifest_path)
    if len(phase_manifest_rows) != 1:
        raise RuntimeError(
            f"{optical_pass} phase reconstruction manifest must contain one row"
        )
    phase_reconstruction = phase_manifest_rows[0]
    if _plain_filename(
        phase_reconstruction.get("output_bmp"),
        label="phase reconstruction output",
        suffix=".bmp",
    ) != phase_path.name:
        raise RuntimeError(f"Phase reconstruction filename mismatch for {optical_pass}")
    compact_phase_name = _plain_filename(
        phase_reconstruction.get("source_png"),
        label="phase reconstruction source",
        suffix=".png",
    )
    compact_phase_path = destination / "compact_phase" / compact_phase_name
    if not compact_phase_path.is_file():
        raise FileNotFoundError(f"Compact phase is missing: {compact_phase_path}")
    _validate_uint8_image(
        compact_phase_path,
        size_wh=(478, 478),
        label=f"compact phase for {optical_pass}",
    )
    compact_phase_sha = _sha256(compact_phase_path)
    if _sha_field(
        phase_reconstruction, "source_sha256", label="phase reconstruction"
    ) != compact_phase_sha:
        raise RuntimeError(f"Compact phase SHA mismatch for {optical_pass}")
    phase_sha = _sha256(phase_path)
    _validate_uint8_image(
        phase_path,
        size_wh=(1920, 1200),
        label=f"physical phase for {optical_pass}",
    )
    if _sha_field(
        phase_reconstruction, "output_sha256", label="phase reconstruction"
    ) != phase_sha:
        raise RuntimeError(f"Physical phase SHA mismatch for {optical_pass}")
    phase_manifest_sha = _sha256(phase_manifest_path)

    phase_export_path = session_dir / "phase_export_manifest.csv"
    if not phase_export_path.is_file():
        raise FileNotFoundError(f"Phase export manifest is missing: {phase_export_path}")
    phase_exports = _unique_rows(
        _read_csv(phase_export_path),
        column="optical_pass",
        label="phase export manifest",
    )
    if optical_pass not in phase_exports:
        raise RuntimeError(f"Phase export manifest has no {optical_pass} row")
    phase_export = phase_exports[optical_pass]
    if _sha_field(
        phase_export, "logical_phase_sha256", label="phase export manifest"
    ) != compact_phase_sha:
        raise RuntimeError(f"Phase compact/export SHA mismatch for {optical_pass}")
    if _sha_field(
        phase_export, "physical_phase_sha256", label="phase export manifest"
    ) != phase_sha:
        raise RuntimeError(f"Phase physical/export SHA mismatch for {optical_pass}")

    acquisition_path = destination / "acquisition_logs" / "capture_manifest.csv"
    if not acquisition_path.is_file():
        raise FileNotFoundError(
            f"Formal acquisition manifest is missing: {acquisition_path}"
        )
    acquisition_rows = _read_csv(acquisition_path)
    acquisition_by_key: dict[str, Mapping[str, Any]] = {}
    for row in acquisition_rows:
        amplitude_name = _plain_filename(
            row.get("amplitude_bmp"), label="acquisition amplitude", suffix=".bmp"
        )
        key = Path(amplitude_name).stem
        if key in acquisition_by_key:
            raise RuntimeError(f"Duplicate acquisition key: {key}")
        acquisition_by_key[key] = row
    if set(acquisition_by_key) != expected_key_set:
        raise RuntimeError(
            f"{optical_pass} acquisition keys differ from the sealed session"
        )

    capture_dir = destination / "ccd_captured"
    capture_files = [
        path
        for path in capture_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".png", ".tif", ".tiff", ".npy"}
    ] if capture_dir.is_dir() else []
    capture_stems = [path.stem for path in capture_files]
    if len(capture_stems) != len(set(capture_stems)):
        raise RuntimeError(f"Duplicate CCD basenames exist for {optical_pass}")
    if set(capture_stems) != expected_key_set:
        raise RuntimeError(
            f"{optical_pass} CCD file set differs from the sealed session"
        )

    geometry_file_hashes: set[str] = set()
    geometry_payload_hashes: set[str] = set()
    sealed_rows: list[dict[str, Any]] = []
    for key in expected_keys:
        logged = acquisition_by_key[key]
        amplitude = amplitude_records[key]
        if _plain_filename(
            logged.get("amplitude_bmp"),
            label=f"logged amplitude for {key}",
            suffix=".bmp",
        ) != f"{key}.bmp":
            raise RuntimeError(f"Logged amplitude filename mismatch for {key}")
        if _sha_field(
            logged, "amplitude_bmp_sha256", label=f"acquisition row {key}"
        ) != amplitude["sha256"]:
            raise RuntimeError(f"Played amplitude SHA mismatch for {key}")
        ccd_name = _plain_filename(
            logged.get("ccd_capture"), label=f"logged CCD for {key}"
        )
        if Path(ccd_name).stem != key or Path(ccd_name).suffix.lower() not in {
            ".png", ".tif", ".tiff", ".npy"
        }:
            raise RuntimeError(f"Logged CCD filename/key mismatch for {key}")
        ccd_path = capture_dir / ccd_name
        if not ccd_path.is_file():
            raise FileNotFoundError(f"Logged CCD is missing: {ccd_path}")
        _validate_ccd_file(ccd_path, label=f"CCD for {key}")
        ccd_sha = _sha256(ccd_path)
        if _sha_field(logged, "output_sha256", label=f"acquisition row {key}") != ccd_sha:
            raise RuntimeError(f"CCD SHA mismatch for {key}")
        if _plain_filename(
            logged.get("phase_mask"), label=f"logged phase for {key}", suffix=".bmp"
        ) != phase_path.name:
            raise RuntimeError(f"Wrong phase filename was recorded for {key}")
        if _sha_field(
            logged, "phase_mask_sha256", label=f"acquisition row {key}"
        ) != phase_sha:
            raise RuntimeError(f"Wrong phase SHA was recorded for {key}")
        if not _csv_bool(
            logged.get("phase_manifest_verified"),
            label=f"phase_manifest_verified for {key}",
        ):
            raise RuntimeError(f"Phase reconstruction was not verified for {key}")
        if _sha_field(
            logged, "phase_manifest_sha256", label=f"acquisition row {key}"
        ) != phase_manifest_sha:
            raise RuntimeError(f"Phase reconstruction manifest SHA mismatch for {key}")
        if not _csv_bool(
            logged.get("orientation_canonicalized"),
            label=f"orientation_canonicalized for {key}",
        ):
            raise RuntimeError(f"CCD {key} was not canonicalized by homography")
        if str(logged.get("saved_frame_orientation", "")) != "canonical_model_xy":
            raise RuntimeError(f"CCD {key} is not in canonical model orientation")
        if _csv_bool(
            logged.get("downstream_loader_flip_required"),
            label=f"downstream_loader_flip_required for {key}",
        ):
            raise RuntimeError(f"CCD {key} still requests a downstream flip")
        if _csv_bool(
            logged.get("background_subtraction"),
            label=f"background_subtraction for {key}",
        ):
            raise RuntimeError(f"CCD {key} unexpectedly used background subtraction")
        if _csv_bool(
            logged.get("per_frame_minmax_normalization"),
            label=f"per_frame_minmax_normalization for {key}",
        ):
            raise RuntimeError(f"CCD {key} unexpectedly used per-frame min/max")
        try:
            saved_size = json.loads(str(logged.get("saved_frame_size_wh", "")))
        except json.JSONDecodeError as error:
            raise RuntimeError(f"CCD {key} has an invalid saved size") from error
        if saved_size != [478, 478]:
            raise RuntimeError(f"CCD {key} is not saved as 478x478")
        geometry_file_hashes.add(
            _sha_field(
                logged,
                "detector_geometry_file_sha256",
                label=f"acquisition row {key}",
            )
        )
        geometry_payload_hashes.add(
            _sha_field(
                logged,
                "detector_geometry_payload_sha256",
                label=f"acquisition row {key}",
            )
        )
        sealed_rows.append(
            {
                "order": expected_order[key],
                "key": key,
                "amplitude_bmp": f"{key}.bmp",
                "amplitude_bmp_sha256": amplitude["sha256"],
                "phase_bmp": phase_path.name,
                "phase_bmp_sha256": phase_sha,
                "ccd_capture": ccd_name,
                "ccd_capture_sha256": ccd_sha,
            }
        )
    if len(geometry_file_hashes) != 1 or len(geometry_payload_hashes) != 1:
        raise RuntimeError("Detector homography identity changed within one acquisition")

    sealed_manifest = destination / "ccd_capture_manifest.csv"
    if sealed_manifest.is_file():
        existing = _read_csv(sealed_manifest)
        if _canonical_sha256(existing) != _canonical_sha256(
            [{key: str(value) for key, value in row.items()} for row in sealed_rows]
        ):
            raise RuntimeError(
                f"Sealed CCD capture manifest changed for {optical_pass}"
            )
    else:
        _write_csv(sealed_manifest, sealed_rows)
    report = {
        "schema_version": 1,
        "optical_pass": optical_pass,
        "sample_count": len(expected_keys),
        "session_manifest_sha256": _sha256(session_dir / "session_manifest.csv"),
        "export_report_sha256": _sha256(export_report_path),
        "compact_amplitude_manifest_sha256": _sha256(compact_manifest_path),
        "amplitude_reconstruction_manifest_sha256": _sha256(reconstruction_path),
        "phase_reconstruction_manifest_sha256": phase_manifest_sha,
        "phase_bmp_sha256": phase_sha,
        "acquisition_manifest_sha256": _sha256(acquisition_path),
        "ccd_capture_manifest": str(sealed_manifest),
        "ccd_capture_manifest_sha256": _sha256(sealed_manifest),
        "detector_geometry_file_sha256": next(iter(geometry_file_hashes)),
        "detector_geometry_payload_sha256": next(iter(geometry_payload_hashes)),
        "orientation": "canonical_model_xy",
        "background_subtraction": False,
        "per_frame_minmax_normalization": False,
    }
    _json(destination / "capture_validation_report.json", report)
    return report


@torch.no_grad()
def export_pass_amplitudes(
    settings: ExperimentSettings,
    checkpoint: Path,
    session_dir: Path,
    optical_pass: str,
    *,
    frame_cache: Path | None,
    max_train: int | None,
    max_test: int | None,
    batch_size: int,
    simulate_upstream: bool,
    reconstruct_amplitude: bool,
    theoretical_count: int,
    phase_center_xy: tuple[float, float],
    flip_vertical: bool,
    flip_horizontal: bool,
    device: torch.device,
) -> dict[str, Any]:
    if optical_pass not in OPTICAL_PASSES:
        raise ValueError(f"Unknown optical pass {optical_pass!r}")
    model, saved = _load_model(settings, checkpoint, device)
    model.eval()
    cache_path = frame_cache or settings.frame_cache_path
    if cache_path is None:
        raise ValueError("A frame cache is required")
    payload = load_frame_cache(cache_path)
    session_dir = session_dir.expanduser().resolve()
    session_dir.mkdir(parents=True, exist_ok=True)
    rows = _ensure_session_manifest(
        session_dir,
        payload,
        frame_cache=cache_path,
        max_train=max_train,
        max_test=max_test,
    )
    pass_index = OPTICAL_PASSES.index(optical_pass)
    measured_prefix = () if simulate_upstream else OPTICAL_PASSES[:pass_index]
    loader = CcdFolderLoader(session_dir)
    for measured_pass in measured_prefix:
        validate_capture_pass(session_dir, measured_pass, rows)
    destination = _stage_dir(session_dir, optical_pass)
    compact_amplitude = destination / "compact_amplitude"
    compact_amplitude.mkdir(parents=True, exist_ok=True)
    # Re-export the exact phase mask from the checkpoint used for this pass.
    export_phase_masks(
        settings,
        checkpoint,
        session_dir,
        phase_center_xy=phase_center_xy,
        flip_vertical=flip_vertical,
        flip_horizontal=flip_horizontal,
        device=device,
    )
    phase = phase_canvases(model)[optical_pass]
    phase_quantized = torch.from_numpy(encode_active_phase(phase.cpu().numpy())).to(device).float()
    phase_quantized = phase_quantized * (2.0 * math.pi / 256.0)
    amplitude_rows: list[dict[str, Any]] = []
    router_rows: list[dict[str, Any]] = []
    theoretical_dir = destination / "theoretical_ccd"
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        indices = [int(row["cache_index"]) for row in batch_rows]
        keys = [row["key"] for row in batch_rows]
        frames = payload["frames"][indices].to(device)
        result = forward_hardware(
            model,
            frames,
            keys,
            measured_passes=measured_prefix,
            measurement_loader=loader if measured_prefix else None,
            stop_before=optical_pass,
        )
        amplitude = result.amplitudes[optical_pass]
        router_rows.extend(_router_rows(result.routing, keys))
        for offset, (row, value) in enumerate(zip(batch_rows, amplitude)):
            encoded, metadata = encode_active_amplitude_with_metadata(value.cpu().numpy())
            output = compact_amplitude / f"{row['key']}.png"
            save_active_png(encoded, output)
            amplitude_rows.append(
                {
                    "order": int(row["order"]),
                    "key": row["key"],
                    "sample_id": row["sample_id"],
                    "split": row["split"],
                    "amplitude_file": output.name,
                    "amplitude_sha256": _sha256(output),
                    "encoding_percentile": metadata["percentile"],
                    "encoding_scale": metadata["scale"],
                }
            )
            global_index = start + offset
            if global_index < theoretical_count:
                logical_amplitude = torch.from_numpy(encoded).to(device).float() / 255.0
                margin = settings.geometry.active_margin
                padded_amplitude = torch.nn.functional.pad(
                    logical_amplitude,
                    (margin, margin, margin, margin),
                )
                padded_phase = torch.nn.functional.pad(
                    phase_quantized,
                    (margin, margin, margin, margin),
                )
                # The propagation contract is batched: [B, 518, 518].  Keep the
                # exported sample dimension explicit even though this preview is
                # generated one file at a time.
                incident = (
                    padded_amplitude.to(torch.complex64)
                    * torch.exp(1j * padded_phase.to(torch.complex64))
                ).unsqueeze(0)
                detector = model.parallel_optics.propagation(incident).abs().square().float()
                active = detector[
                    0,
                    margin : margin + settings.geometry.active_size,
                    margin : margin + settings.geometry.active_size,
                ].cpu().numpy()
                theoretical_dir.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    theoretical_dir / f"{row['key']}.npz",
                    intensity=active.astype(np.float32),
                    optical_pass=optical_pass,
                    key=row["key"],
                )
                preview = np.log1p(active / max(float(active.mean()), 1.0e-8))
                preview = np.rint(255.0 * preview / max(float(preview.max()), 1.0e-8)).astype(np.uint8)
                Image.fromarray(preview, mode="L").save(
                    theoretical_dir / f"{row['key']}_log_preview.png"
                )
        print(f"[export {optical_pass}] {min(start + len(batch_rows), len(rows))}/{len(rows)}", flush=True)
    _write_csv(destination / "amplitude_manifest.csv", amplitude_rows)
    if router_rows:
        _write_csv(destination / "router_decisions_upstream.csv", router_rows)
    reconstruction = None
    if reconstruct_amplitude:
        reconstruction = reconstruct_directory(
            compact_amplitude,
            destination / "amplitude_to_play",
            slm_size_wh=(1024, 1024),
            scale_factor=1,
            center_xy=(512.0, 512.0),
        )
    report = {
        "schema_version": 1,
        "optical_pass": optical_pass,
        "physical_pass_index": pass_index + 1,
        "checkpoint": str(checkpoint.expanduser().resolve()),
        "checkpoint_sha256": _sha256(checkpoint.expanduser().resolve()),
        "checkpoint_epoch": int(saved.get("epoch", -1)),
        "sample_count": len(rows),
        "measured_upstream_passes": list(measured_prefix),
        "simulated_upstream": bool(simulate_upstream),
        "four_frames_parallel": optical_pass in {"stage1_router", "stage1_expert", "stage2_global"},
        "logical_amplitude_size_wh": [478, 478],
        "amplitude_slm_size_wh": [1024, 1024],
        "amplitude_slm_pixel_pitch_um": 17.0,
        "phase_slm_size_wh": [1920, 1200],
        "phase_slm_pixel_pitch_um": 8.0,
        "phase_slm_center_xy": list(phase_center_xy),
        "reconstruction": reconstruction,
        "theoretical_ccd_count": min(theoretical_count, len(rows)),
    }
    _json(destination / "export_report.json", report)
    return report


class _SessionDataset(Dataset):
    def __init__(self, payload: Mapping[str, Any], rows: list[dict[str, str]], split: str) -> None:
        self.payload = payload
        self.rows = [row for row in rows if row["split"] == split]
        if not self.rows:
            raise RuntimeError(f"No hardware-session samples for split={split}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        source = int(row["cache_index"])
        return {
            "frames": self.payload["frames"][source],
            "target": self.payload["targets"][source].float(),
            "key": row["key"],
            "sample_id": row["sample_id"],
        }


def _set_trainable_after_capture(model: LGVQFourStageOEO, stage: str) -> list[str]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if stage == "stage1":
        modules = [
            model.frame_stem,
            model.parallel_optics.expert_readout,
            model.fusions[0],
            model.electronic_stage2,
            model.parallel_optics.width_to_field,
            model.parallel_optics.tokens_to_field,
            model.parallel_optics.global_readout,
            model.parallel_optics.raw_global_phase,
            model.fusions[1],
            model.electronic_stage3,
            model.serial_optics,
            model.serial_router,
            model.fusions[2],
            model.electronic_stage4,
            model.fusions[3],
            model.readout,
        ]
    elif stage == "stage2":
        modules = [
            model.electronic_stage2,
            model.parallel_optics.global_readout,
            model.fusions[1],
            model.electronic_stage3,
            model.serial_optics,
            model.serial_router,
            model.fusions[2],
            model.electronic_stage4,
            model.fusions[3],
            model.readout,
        ]
    elif stage == "stage3":
        modules = [
            model.electronic_stage3,
            model.serial_optics.expert_readout,
            model.fusions[2],
            model.electronic_stage4,
            model.serial_optics.global_readout,
            model.serial_optics.raw_global_phase,
            model.fusions[3],
            model.readout,
        ]
    elif stage == "stage4":
        modules = [
            model.electronic_stage4,
            model.serial_optics.global_readout,
            model.fusions[3],
            model.readout,
        ]
    else:
        raise ValueError(f"Unknown fusion stage {stage!r}")
    for module in modules:
        if isinstance(module, torch.nn.Parameter):
            module.requires_grad_(True)
        else:
            for parameter in module.parameters():
                parameter.requires_grad_(True)
    return [name for name, parameter in model.named_parameters() if parameter.requires_grad]


def _hardware_optimizer(model: LGVQFourStageOEO, settings: ExperimentSettings) -> torch.optim.Optimizer:
    phase, router, electronic = [], [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if "router" in name and "phase" in name:
            router.append(parameter)
        elif "phase" in name:
            phase.append(parameter)
        else:
            electronic.append(parameter)
    groups = []
    if electronic:
        groups.append({"params": electronic, "lr": settings.learning_rate, "weight_decay": settings.weight_decay})
    if phase:
        groups.append({"params": phase, "lr": settings.phase_learning_rate, "weight_decay": 0.0})
    if router:
        groups.append({"params": router, "lr": settings.router_phase_learning_rate, "weight_decay": 0.0})
    if not groups:
        raise RuntimeError("No downstream parameters are trainable")
    return torch.optim.AdamW(groups)


@torch.no_grad()
def _evaluate_hardware(
    model: LGVQFourStageOEO,
    loader: DataLoader,
    ccd_loader: CcdFolderLoader,
    measured_passes: tuple[str, ...],
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    predictions, targets = [], []
    rows: list[dict[str, Any]] = []
    for batch in loader:
        result = forward_hardware(
            model,
            batch["frames"].to(device),
            batch["key"],
            measured_passes=measured_passes,
            measurement_loader=ccd_loader,
        )
        assert result.prediction is not None
        prediction = result.prediction.detach().cpu()
        target = batch["target"].cpu()
        predictions.append(prediction)
        targets.append(target)
        for index, key in enumerate(batch["key"]):
            rows.append(
                {
                    "key": key,
                    "sample_id": batch["sample_id"][index],
                    "spatial_target": float(target[index, 0]),
                    "spatial_prediction": float(prediction[index, 0]),
                    "temporal_target": float(target[index, 1]),
                    "temporal_prediction": float(prediction[index, 1]),
                }
            )
    metrics = regression_metrics(torch.cat(predictions), torch.cat(targets))
    metrics["measured_passes"] = list(measured_passes)
    metrics["selection_mean_srcc"] = 0.5 * (
        metrics["spatial"]["srcc"] + metrics["temporal"]["srcc"]
    )
    return metrics, rows


def finetune_hardware_stage(
    settings: ExperimentSettings,
    checkpoint: Path,
    session_dir: Path,
    stage: str,
    *,
    frame_cache: Path | None,
    epochs: int,
    batch_size: int,
    test_interval: int,
    device: torch.device,
) -> dict[str, Any]:
    if stage not in FUSION_STAGES:
        raise ValueError(f"Unknown fusion stage {stage!r}")
    model, source = _load_model(settings, checkpoint, device)
    cache_path = frame_cache or settings.frame_cache_path
    if cache_path is None:
        raise ValueError("A frame cache is required")
    cache_path = cache_path.expanduser().resolve()
    payload = load_frame_cache(cache_path)
    session_dir = session_dir.expanduser().resolve()
    rows = _load_sealed_session_manifest(
        session_dir, payload, frame_cache=cache_path
    )
    measured_passes = tuple(FUSION_STAGES[stage])
    new_capture_passes = set(_capture_passes_for_stage(stage))
    checkpoint_path = checkpoint.expanduser().resolve()
    for optical_pass in measured_passes:
        validate_capture_pass(
            session_dir,
            optical_pass,
            rows,
            expected_checkpoint=(
                checkpoint_path if optical_pass in new_capture_passes else None
            ),
        )
    ccd_loader = CcdFolderLoader(session_dir)
    # Fail before training if any exact allowlisted capture is absent.
    keys = [row["key"] for row in rows]
    for optical_pass in measured_passes:
        for start in range(0, len(keys), 32):
            ccd_loader(optical_pass, keys[start : start + 32], torch.device("cpu"))
    train_loader = DataLoader(_SessionDataset(payload, rows, "train"), batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(_SessionDataset(payload, rows, "test"), batch_size=batch_size, shuffle=False, num_workers=0)
    trainable = _set_trainable_after_capture(model, stage)
    optimizer = _hardware_optimizer(model, settings)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_score = float("-inf")
    best_epoch = 0
    output_dir = session_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / f"after_{stage}_best_test.pt"
    history: list[dict[str, Any]] = []
    weights = (settings.spatial_target_weight, settings.temporal_target_weight)
    for epoch in range(1, epochs + 1):
        model.train()
        totals = defaultdict(float)
        batches = 0
        for batch in train_loader:
            target = batch["target"].to(device)
            normalized_target = (target - model.target_mean) / model.target_std
            optimizer.zero_grad(set_to_none=True)
            result = forward_hardware(
                model,
                batch["frames"].to(device),
                batch["key"],
                measured_passes=measured_passes,
                measurement_loader=ccd_loader,
            )
            assert result.normalized_prediction is not None
            regression = weighted_smooth_l1_loss(result.normalized_prediction, normalized_target, weights)
            ranking = pairwise_ranking_loss(result.normalized_prediction, normalized_target, target_weights=weights)
            correlation = batch_correlation_loss(result.normalized_prediction, normalized_target, target_weights=weights)
            loss = regression + settings.ranking_weight * ranking + settings.correlation_weight * correlation
            if result.optical_alignment_loss is not None:
                loss = loss + settings.optical_alignment_weight * result.optical_alignment_loss
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("Non-finite hardware fine-tuning loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            totals["loss"] += float(loss.detach())
            totals["regression"] += float(regression.detach())
            totals["ranking"] += float(ranking.detach())
            totals["correlation"] += float(correlation.detach())
            batches += 1
        scheduler.step()
        row: dict[str, Any] = {
            "epoch": epoch,
            **{name: value / max(1, batches) for name, value in totals.items()},
            "test_evaluated": False,
        }
        if epoch == 1 or epoch % test_interval == 0 or epoch == epochs:
            metrics, prediction_rows = _evaluate_hardware(
                model, test_loader, ccd_loader, measured_passes, device
            )
            score = float(metrics["selection_mean_srcc"])
            row["test_evaluated"] = True
            row["test"] = metrics
            if score > best_score:
                best_score, best_epoch = score, epoch
                checkpoint_payload = {
                    "schema_version": 1,
                    "architecture": settings.architecture_label,
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "metrics_optical_hardware": metrics,
                    "settings": resolved_dict(settings),
                    "hardware_finetune": {
                        "stage": stage,
                        "measured_passes": list(measured_passes),
                        "source_checkpoint": str(checkpoint.expanduser().resolve()),
                        "source_checkpoint_sha256": _sha256(checkpoint.expanduser().resolve()),
                        "trainable_parameter_names": trainable,
                        "test_used_for_selection": True,
                        "separately_trained_electronic_baseline": False,
                    },
                }
                temporary = best_path.with_suffix(".pt.tmp")
                torch.save(checkpoint_payload, temporary)
                temporary.replace(best_path)
                _json(output_dir / f"after_{stage}_best_metrics.json", metrics)
                _write_csv(output_dir / f"after_{stage}_best_predictions.csv", prediction_rows)
        history.append(row)
        _json(output_dir / f"after_{stage}_history.json", history)
        print(
            f"[finetune {stage}] epoch={epoch:03d}/{epochs:03d} "
            f"loss={row['loss']:.6f} best_mean_SRCC={best_score:.4f}",
            flush=True,
        )
    report = {
        "stage": stage,
        "best_epoch": best_epoch,
        "best_test_mean_srcc": best_score,
        "checkpoint": str(best_path),
        "checkpoint_sha256": _sha256(best_path),
        "measured_passes": list(measured_passes),
        "trainable_parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "frozen_parameter_count": sum(p.numel() for p in model.parameters() if not p.requires_grad),
        "test_used_for_selection": True,
    }
    _json(output_dir / f"after_{stage}_report.json", report)
    return report


def evaluate_hardware_checkpoint(
    settings: ExperimentSettings,
    checkpoint: Path,
    session_dir: Path,
    stage: str,
    *,
    frame_cache: Path | None,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    model, _ = _load_model(settings, checkpoint, device)
    cache_path = frame_cache or settings.frame_cache_path
    if cache_path is None:
        raise ValueError("A frame cache is required")
    cache_path = cache_path.expanduser().resolve()
    payload = load_frame_cache(cache_path)
    session_dir = session_dir.expanduser().resolve()
    rows = _load_sealed_session_manifest(
        session_dir, payload, frame_cache=cache_path
    )
    loader = DataLoader(_SessionDataset(payload, rows, "test"), batch_size=batch_size, shuffle=False, num_workers=0)
    measured_passes = tuple(FUSION_STAGES[stage])
    for optical_pass in measured_passes:
        validate_capture_pass(session_dir, optical_pass, rows)
    metrics, predictions = _evaluate_hardware(
        model, loader, CcdFolderLoader(session_dir), measured_passes, device
    )
    output = session_dir / "final_evaluation"
    _json(output / f"through_{stage}_metrics.json", metrics)
    _write_csv(output / f"through_{stage}_predictions.csv", predictions)
    return metrics


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _phase_export_geometry(args: argparse.Namespace) -> tuple[tuple[float, float], bool, bool]:
    """Resolve the one operator-owned phase geometry, with explicit CLI overrides."""

    center = (980.0, 590.0)
    flip_vertical = True
    flip_horizontal = False
    lab_config = Path(args.lab_config).expanduser()
    if lab_config.is_file():
        raw = yaml.safe_load(lab_config.read_text(encoding="utf-8")) or {}
        phase = raw.get("phase_slm", {})
        if not isinstance(phase, Mapping):
            raise ValueError("LAB_CONFIG.yaml: phase_slm must be a mapping")
        configured_center = phase.get("center_xy", center)
        if not isinstance(configured_center, (list, tuple)) or len(configured_center) != 2:
            raise ValueError("LAB_CONFIG.yaml: phase_slm.center_xy must be [x,y]")
        center = (float(configured_center[0]), float(configured_center[1]))
        flip_vertical = bool(phase.get("flip_vertical_before_raster", flip_vertical))
        flip_horizontal = bool(phase.get("flip_horizontal_before_raster", flip_horizontal))
    if args.phase_center_x is not None:
        center = (float(args.phase_center_x), center[1])
    if args.phase_center_y is not None:
        center = (center[0], float(args.phase_center_y))
    if args.phase_flip_vertical is not None:
        flip_vertical = bool(args.phase_flip_vertical)
    if args.phase_flip_horizontal is not None:
        flip_horizontal = bool(args.phase_flip_horizontal)
    if not all(math.isfinite(value) for value in center):
        raise ValueError("Phase SLM center must be finite")
    return center, flip_vertical, flip_horizontal


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "export-masks",
            "export-pass",
            "validate-capture",
            "finetune",
            "evaluate",
        ),
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--session-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--lab-config", default="experiments/lab_lgvq/LAB_CONFIG.yaml")
    parser.add_argument("--phase-center-x", type=float)
    parser.add_argument("--phase-center-y", type=float)
    parser.add_argument("--phase-flip-vertical", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--phase-flip-horizontal", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--optical-pass", choices=OPTICAL_PASSES)
    parser.add_argument("--stage", choices=tuple(FUSION_STAGES))
    parser.add_argument("--frame-cache")
    parser.add_argument("--max-train", type=int, default=64)
    parser.add_argument("--max-test", type=int, default=32)
    parser.add_argument("--all-data", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--test-interval", type=int, default=5)
    parser.add_argument("--simulate-upstream", action="store_true")
    parser.add_argument("--reconstruct-amplitude", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--theoretical-count", type=int, default=8)
    args = parser.parse_args()
    settings = load_settings(args.config)
    checkpoint = Path(args.checkpoint)
    session = Path(args.session_dir)
    device = _device(args.device)
    frame_cache = None if args.frame_cache is None else Path(args.frame_cache)
    center, phase_flip_vertical, phase_flip_horizontal = _phase_export_geometry(args)
    if args.command == "export-masks":
        report = export_phase_masks(
            settings,
            checkpoint,
            session,
            phase_center_xy=center,
            flip_vertical=phase_flip_vertical,
            flip_horizontal=phase_flip_horizontal,
            device=device,
        )
    elif args.command == "export-pass":
        if args.optical_pass is None:
            parser.error("export-pass requires --optical-pass")
        report = export_pass_amplitudes(
            settings,
            checkpoint,
            session,
            args.optical_pass,
            frame_cache=frame_cache,
            max_train=None if args.all_data else args.max_train,
            max_test=None if args.all_data else args.max_test,
            batch_size=args.batch_size,
            simulate_upstream=args.simulate_upstream,
            reconstruct_amplitude=args.reconstruct_amplitude,
            theoretical_count=args.theoretical_count,
            phase_center_xy=center,
            flip_vertical=phase_flip_vertical,
            flip_horizontal=phase_flip_horizontal,
            device=device,
        )
    elif args.command == "validate-capture":
        if args.optical_pass is None:
            parser.error("validate-capture requires --optical-pass")
        cache_path = frame_cache or settings.frame_cache_path
        if cache_path is None:
            parser.error("validate-capture requires --frame-cache or config data.frame_cache")
        cache_path = cache_path.expanduser().resolve()
        payload = load_frame_cache(cache_path)
        rows = _load_sealed_session_manifest(
            session.expanduser().resolve(), payload, frame_cache=cache_path
        )
        report = validate_capture_pass(
            session,
            args.optical_pass,
            rows,
            expected_checkpoint=checkpoint,
        )
    elif args.command == "finetune":
        if args.stage is None:
            parser.error("finetune requires --stage")
        report = finetune_hardware_stage(
            settings,
            checkpoint,
            session,
            args.stage,
            frame_cache=frame_cache,
            epochs=args.epochs,
            batch_size=args.batch_size,
            test_interval=args.test_interval,
            device=device,
        )
    else:
        if args.stage is None:
            parser.error("evaluate requires --stage")
        report = evaluate_hardware_checkpoint(
            settings,
            checkpoint,
            session,
            args.stage,
            frame_cache=frame_cache,
            batch_size=args.batch_size,
            device=device,
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
