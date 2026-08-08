from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from PIL import Image, ImageOps
from torch.nn import functional as F

from .features import move_inputs, preprocess_images, student_embeddings, validate_token_budgets
from .io_utils import seed_everything, write_csv, write_json
from .modeling import build_optical_student, load_backbone
from .optical_artifacts import export_centered_bmp, phase_tensors, tensor_stats
from .prepare_grocery_retrieval_subset import GroceryRetrievalBundle, GrocerySample, prepare_grocery_subset
from .retrieval_metrics import evaluate_embeddings
from .settings import Settings, load_settings
from .train_optical_retrieval import load_checkpoint


STAGES = {
    "vision_expert": "01_vision_expert",
    "vision_global": "02_vision_global",
    "language_expert": "03_language_expert",
    "language_global": "04_language_global",
}
PHASES = {
    "prepare",
    "process_vision_expert",
    "process_vision_global",
    "process_language_expert",
    "process_language_global",
    "all_simulation",
}


@dataclass(frozen=True)
class HardwareConfig:
    config_path: Path
    model_config: Path
    checkpoint: Path
    output_dir: Path
    selection_mode: str
    queries_per_sku: int
    prefer_correct_queries: bool
    selection_results_csv: Path | None
    amplitude_slm_size: tuple[int, int]
    phase_slm_size: tuple[int, int]
    slm_pixel_pitch_um: float
    amplitude_encoding_mode: str
    amplitude_encoding_percentile: float
    amplitude_encoding_gamma: float
    phase_flip_vertical: bool
    capture_roi_xywh: tuple[int, int, int, int] | None
    capture_flip_vertical: bool
    capture_flip_horizontal: bool
    capture_dark_level: float
    capture_binning_factor: int
    capture_binning_reduction: str
    capture_registration_mode: str
    simulation_tensor_dtype: torch.dtype
    simulation_save_complex_fields: bool
    minimal_artifacts: bool
    clean_output_before_prepare: bool
    copy_checkpoint_to_output: bool
    sample_limit: int | None = None


@dataclass
class Runtime:
    hardware: HardwareConfig
    settings: Settings
    bundle: GroceryRetrievalBundle
    loaded: Any
    replacement: Any
    readout: Any


def _resolve(value: Any, base: Path) -> Path:
    path = Path(str(value)).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def load_hardware_config(path: str | Path) -> HardwareConfig:
    config_path = Path(path).expanduser().resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Hardware deployment config must be a YAML mapping")
    base = config_path.parent
    roi = raw.get("capture", {}).get("roi_xywh")
    dtype_name = str(raw.get("simulation", {}).get("tensor_dtype", "float16"))
    dtypes = {"float16": torch.float16, "float32": torch.float32}
    if dtype_name not in dtypes:
        raise ValueError("simulation.tensor_dtype must be float16 or float32")
    amplitude_encoding = raw.get("slm", {}).get("amplitude_encoding", {})
    result = HardwareConfig(
        config_path=config_path,
        model_config=_resolve(raw["model_config"], base),
        checkpoint=_resolve(raw["checkpoint"], base),
        output_dir=_resolve(raw["output_dir"], base),
        selection_mode=str(
            raw.get("selection", {}).get("mode", "selected_test")
        ),
        queries_per_sku=int(raw.get("selection", {}).get("queries_per_sku", 10)),
        prefer_correct_queries=bool(raw.get("selection", {}).get("prefer_correct_queries", True)),
        selection_results_csv=(
            None
            if raw.get("selection", {}).get("results_csv") is None
            else _resolve(raw["selection"]["results_csv"], base)
        ),
        amplitude_slm_size=tuple(int(v) for v in raw.get("slm", {}).get("amplitude_size_wh", [1920, 1080])),
        phase_slm_size=tuple(int(v) for v in raw.get("slm", {}).get("phase_size_wh", [1920, 1200])),
        slm_pixel_pitch_um=float(raw.get("slm", {}).get("pixel_pitch_um", 8.0)),
        amplitude_encoding_mode=str(
            amplitude_encoding.get("mode", "per_plane_max")
        ),
        amplitude_encoding_percentile=float(
            amplitude_encoding.get("percentile", 99.5)
        ),
        amplitude_encoding_gamma=float(amplitude_encoding.get("gamma", 1.0)),
        phase_flip_vertical=bool(
            raw.get("slm", {}).get("phase_transform", {}).get(
                "flip_vertical", True
            )
        ),
        capture_roi_xywh=None if roi is None else tuple(int(v) for v in roi),
        capture_flip_vertical=bool(
            raw.get("capture", {}).get("flip_vertical", False)
        ),
        capture_flip_horizontal=bool(
            raw.get("capture", {}).get("flip_horizontal", False)
        ),
        capture_dark_level=float(raw.get("capture", {}).get("dark_level", 0.0)),
        capture_binning_factor=int(raw.get("capture", {}).get("binning_factor", 1)),
        capture_binning_reduction=str(raw.get("capture", {}).get("binning_reduction", "mean")),
        capture_registration_mode=str(
            raw.get("capture", {}).get("registration_mode", "strict")
        ),
        simulation_tensor_dtype=dtypes[dtype_name],
        simulation_save_complex_fields=bool(raw.get("simulation", {}).get("save_complex_fields", True)),
        minimal_artifacts=(
            str(raw.get("artifacts", {}).get("profile", "full")) == "minimal"
        ),
        clean_output_before_prepare=bool(
            raw.get("artifacts", {}).get("clean_output_before_prepare", False)
        ),
        copy_checkpoint_to_output=bool(
            raw.get("artifacts", {}).get("copy_checkpoint", False)
        ),
    )
    if result.selection_mode not in {
        "selected_test",
        "test_only",
        "full_dataset",
    }:
        raise ValueError(
            "selection.mode must be selected_test, test_only or full_dataset"
        )
    if result.queries_per_sku <= 0:
        raise ValueError("selection.queries_per_sku must be positive")
    if len(result.amplitude_slm_size) != 2 or len(result.phase_slm_size) != 2:
        raise ValueError("SLM sizes must be [width,height]")
    if result.amplitude_encoding_mode not in {
        "per_plane_max",
        "positive_percentile",
    }:
        raise ValueError(
            "slm.amplitude_encoding.mode must be per_plane_max or positive_percentile"
        )
    if not 0.0 < result.amplitude_encoding_percentile <= 100.0:
        raise ValueError("slm.amplitude_encoding.percentile must be in (0,100]")
    if result.amplitude_encoding_gamma <= 0.0:
        raise ValueError("slm.amplitude_encoding.gamma must be positive")
    if result.capture_roi_xywh is not None and len(result.capture_roi_xywh) != 4:
        raise ValueError("capture.roi_xywh must be null or [x,y,width,height]")
    if result.capture_binning_factor <= 0:
        raise ValueError("capture.binning_factor must be positive")
    if result.capture_binning_reduction not in {"mean", "sum"}:
        raise ValueError("capture.binning_reduction must be mean or sum")
    if result.capture_registration_mode not in {"strict", "nearest_resize"}:
        raise ValueError(
            "capture.registration_mode must be strict or nearest_resize"
        )
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in "-_" else "_" for char in value)
    return cleaned[:64].strip("_") or "sample"


def _sample_key(index: int, sample: GrocerySample, role: str) -> str:
    digest = hashlib.sha1(sample.sample_id.encode("utf-8")).hexdigest()[:8]
    return f"{index:05d}__{role}__{_safe(sample.sku_name)}__{digest}"


def _read_previous_results(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle)]
    return {
        row["sample_id"]: row
        for row in rows
        if row.get("system") == "optical_student_query_vs_optical_student_gallery"
    }


def select_samples(bundle: GroceryRetrievalBundle, settings: Settings, hardware: HardwareConfig) -> list[tuple[str, GrocerySample]]:
    selected: list[tuple[str, GrocerySample]] = [
        ("gallery", sample)
        for sample in sorted(bundle.gallery_samples, key=lambda value: value.sample_id)
    ]
    if hardware.selection_mode in {"test_only", "full_dataset"}:
        if hardware.selection_mode == "full_dataset":
            selected.extend(
                ("train", sample)
                for sample in sorted(
                    bundle.train_samples, key=lambda value: value.sample_id
                )
            )
        selected.extend(
            ("query", sample)
            for sample in sorted(bundle.test_samples, key=lambda value: value.sample_id)
        )
        sample_ids = [sample.sample_id for _, sample in selected]
        image_paths = [str(sample.image_path.resolve()) for _, sample in selected]
        if len(sample_ids) != len(set(sample_ids)):
            raise RuntimeError("Hardware manifest contains duplicate sample_id values")
        if len(image_paths) != len(set(image_paths)):
            raise RuntimeError("Hardware manifest contains duplicate image paths")
        return (
            selected
            if hardware.sample_limit is None
            else selected[: hardware.sample_limit]
        )
    results_path = hardware.selection_results_csv or settings.output_dir / "retrieval_results.csv"
    previous = _read_previous_results(results_path)
    by_sku: dict[int, list[GrocerySample]] = {}
    for sample in bundle.test_samples:
        by_sku.setdefault(sample.sku_index, []).append(sample)
    for sku_index in range(len(bundle.class_names)):
        choices = by_sku.get(sku_index, [])
        choices.sort(
            key=lambda sample: (
                0 if previous.get(sample.sample_id, {}).get("top1_correct", "").lower() in {"true", "1"} else 1,
                -float(previous.get(sample.sample_id, {}).get("similarity_margin") or "-inf"),
                sample.sample_id,
            )
            if hardware.prefer_correct_queries
            else (sample.sample_id,)
        )
        selected.extend(("query", sample) for sample in choices[: hardware.queries_per_sku])
    return selected if hardware.sample_limit is None else selected[: hardware.sample_limit]


def _manifest_rows(runtime: Runtime) -> list[dict[str, Any]]:
    path = runtime.hardware.output_dir / "00_manifest" / "play_order.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Hardware play manifest is missing: {path}; run --phase prepare first")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _samples_by_id(bundle: GroceryRetrievalBundle) -> dict[str, GrocerySample]:
    return {sample.sample_id: sample for sample in bundle.all_samples()}


def _active_crop(core: Any, value: torch.Tensor) -> torch.Tensor:
    aperture = core.geometry.active_aperture
    return value[..., aperture.y0 : aperture.y1, aperture.x0 : aperture.x1]


def _active_to_canvas(core: Any, amplitude: torch.Tensor) -> torch.Tensor:
    expected = (core.geometry.active_size, core.geometry.active_size)
    if tuple(amplitude.shape) != expected:
        raise ValueError(f"Active amplitude must be {expected}, got {tuple(amplitude.shape)}")
    output = amplitude.new_zeros((core.geometry.canvas_size, core.geometry.canvas_size))
    aperture = core.geometry.active_aperture
    output[aperture.y0 : aperture.y1, aperture.x0 : aperture.x1] = amplitude
    return torch.complex(output, torch.zeros_like(output))


def _intensity_active_to_canvas(core: Any, intensity: torch.Tensor) -> torch.Tensor:
    expected = (core.geometry.active_size, core.geometry.active_size)
    if tuple(intensity.shape) != expected:
        raise ValueError(f"Active expert CCD intensity must be {expected}, got {tuple(intensity.shape)}")
    output = intensity.new_zeros((core.geometry.canvas_size, core.geometry.canvas_size))
    aperture = core.geometry.active_aperture
    output[aperture.y0 : aperture.y1, aperture.x0 : aperture.x1] = intensity
    return output


def _preview(value: torch.Tensor, path: Path, title: str, kind: str) -> None:
    tensor = value.detach().cpu().float()
    while tensor.ndim > 2:
        tensor = tensor[0]
    upper = max(float(torch.quantile(tensor.flatten(), 0.995)), 1.0e-8)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8.2, 7.2), constrained_layout=True)
    shown = axis.imshow(tensor.numpy(), cmap="viridis", vmin=0.0, vmax=upper)
    figure.colorbar(shown, ax=axis, label=kind)
    axis.set_xlabel("x pixel")
    axis.set_ylabel("y pixel")
    stats = tensor_stats(tensor)
    axis.set_title(
        f"{title}\nshape={tuple(tensor.shape)} min={stats['min']:.4g} "
        f"max={stats['max']:.4g} mean={stats['mean']:.4g}"
    )
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _save_tensor(path: Path, value: torch.Tensor, dtype: torch.dtype | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tensor = value.detach().cpu()
    if dtype is not None and tensor.is_floating_point():
        tensor = tensor.to(dtype)
    torch.save(tensor, path)


def _save_amplitude(runtime: Runtime, stage: str, key: str, amplitude: torch.Tensor, *, simulation: bool) -> dict[str, Any]:
    if runtime.hardware.minimal_artifacts:
        bmp_path = (
            runtime.hardware.output_dir
            / "amplitude_bmp"
            / STAGES[stage]
            / f"{key}.bmp"
        )
        ratio = runtime.settings.pixel_pitch_um / runtime.hardware.slm_pixel_pitch_um
        factor = int(round(ratio))
        if factor <= 0 or not math.isclose(ratio, factor, abs_tol=1.0e-9):
            raise RuntimeError("Simulation/SLM pixel pitches require a positive integer scale")
        return export_centered_bmp(
            amplitude,
            bmp_path,
            value_type="amplitude",
            scale_factor=factor,
            slm_width=runtime.hardware.amplitude_slm_size[0],
            slm_height=runtime.hardware.amplitude_slm_size[1],
            amplitude_encoding_mode=runtime.hardware.amplitude_encoding_mode,
            amplitude_percentile=runtime.hardware.amplitude_encoding_percentile,
            amplitude_gamma=runtime.hardware.amplitude_encoding_gamma,
        )
    root = runtime.hardware.output_dir / STAGES[stage]
    prefix = root / "simulation_reference" if simulation else root
    raw_path = prefix / "amplitude_raw" / f"{key}.pt"
    bmp_path = prefix / "amplitude_to_play" / f"{key}.bmp"
    preview_path = prefix / "amplitude_preview" / f"{key}.png"
    _save_tensor(raw_path, amplitude, runtime.hardware.simulation_tensor_dtype if simulation else torch.float32)
    ratio = runtime.settings.pixel_pitch_um / runtime.hardware.slm_pixel_pitch_um
    factor = int(round(ratio))
    if factor <= 0 or not math.isclose(ratio, factor, abs_tol=1.0e-9):
        raise RuntimeError("Simulation/SLM pixel pitches require a positive integer scale")
    report = export_centered_bmp(
        amplitude,
        bmp_path,
        value_type="amplitude",
        scale_factor=factor,
        slm_width=runtime.hardware.amplitude_slm_size[0],
        slm_height=runtime.hardware.amplitude_slm_size[1],
        amplitude_encoding_mode=runtime.hardware.amplitude_encoding_mode,
        amplitude_percentile=runtime.hardware.amplitude_encoding_percentile,
        amplitude_gamma=runtime.hardware.amplitude_encoding_gamma,
    )
    _preview(amplitude, preview_path, f"{stage} amplitude {key}", "amplitude")
    write_json(prefix / "amplitude_metadata" / f"{key}.json", {**report, "raw_tensor": str(raw_path), "stats": tensor_stats(amplitude)})
    return report


def _save_simulated_ccd(runtime: Runtime, stage: str, key: str, intensity: torch.Tensor) -> None:
    if runtime.hardware.minimal_artifacts:
        _preview(
            intensity,
            runtime.hardware.output_dir
            / "theoretical_ccd"
            / STAGES[stage]
            / f"{key}.png",
            f"{stage} theoretical CCD {key}",
            "intensity",
        )
        return
    root = runtime.hardware.output_dir / STAGES[stage] / "simulation_reference"
    _save_tensor(root / "ccd_intensity" / f"{key}.pt", intensity, runtime.hardware.simulation_tensor_dtype)
    _preview(intensity, root / "ccd_preview" / f"{key}.png", f"{stage} simulated CCD {key}", "intensity")
    write_json(root / "ccd_stats" / f"{key}.json", tensor_stats(intensity))


def _save_simulated_complex_field(runtime: Runtime, stage: str, key: str, field: torch.Tensor) -> None:
    if runtime.hardware.minimal_artifacts or not runtime.hardware.simulation_save_complex_fields:
        return
    root = runtime.hardware.output_dir / STAGES[stage] / "simulation_reference"
    # PyTorch has no portable complex-float16 tensor. Preserve real/imaginary
    # parts explicitly as [2,H,W] FP16/FP32; this is the pre-square-law field.
    packed = torch.stack([field.real.float(), field.imag.float()], dim=0)
    _save_tensor(root / "complex_field_real_imag" / f"{key}.pt", packed, runtime.hardware.simulation_tensor_dtype)
    amplitude = field.abs().float()
    phase = torch.angle(field).float()
    _preview(amplitude, root / "complex_amplitude_preview" / f"{key}.png", f"{stage} simulated detector-plane amplitude {key}", "amplitude")
    path = root / "complex_phase_preview" / f"{key}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8.2, 7.2), constrained_layout=True)
    shown = axis.imshow(phase.detach().cpu().numpy(), cmap="twilight", vmin=-math.pi, vmax=math.pi)
    figure.colorbar(shown, ax=axis, label="phase (rad)")
    axis.set_xlabel("x pixel"); axis.set_ylabel("y pixel")
    axis.set_title(f"{stage} simulated detector-plane phase {key}")
    figure.savefig(path, dpi=150); plt.close(figure)


def _load_numeric_image(path: Path) -> torch.Tensor:
    with Image.open(path) as source:
        array = np.array(source)
    if array.ndim == 3:
        if array.shape[2] not in {3, 4}:
            raise RuntimeError(f"Unsupported CCD image shape {array.shape} in {path}")
        rgb = array[..., :3]
        if not np.array_equal(rgb[..., 0], rgb[..., 1]) or not np.array_equal(rgb[..., 0], rgb[..., 2]):
            raise RuntimeError(f"CCD image {path} is RGB with unequal channels; export a raw monochrome image")
        array = rgb[..., 0]
    return torch.from_numpy(np.asarray(array).copy()).float()


def bin_ccd_superpixels(
    intensity: torch.Tensor,
    factor: int,
    reduction: str = "mean",
) -> torch.Tensor:
    """Combine physical CCD pixels into logical model pixels without interpolation."""
    if intensity.ndim != 2:
        raise ValueError(f"CCD superpixel binning expects 2-D intensity, got {tuple(intensity.shape)}")
    factor = int(factor)
    if factor <= 0:
        raise ValueError("CCD binning factor must be positive")
    height, width = intensity.shape
    if height % factor or width % factor:
        raise ValueError(
            f"CCD shape {tuple(intensity.shape)} is not divisible by binning factor {factor}"
        )
    blocks = intensity.reshape(height // factor, factor, width // factor, factor)
    if reduction == "mean":
        return blocks.mean(dim=(1, 3))
    if reduction == "sum":
        return blocks.sum(dim=(1, 3))
    raise ValueError("CCD binning reduction must be mean or sum")


def load_captured_intensity(runtime: Runtime, stage: str, key: str, *, use_simulation: bool) -> torch.Tensor:
    root = runtime.hardware.output_dir / STAGES[stage]
    if use_simulation:
        path = root / "simulation_reference" / "ccd_intensity" / f"{key}.pt"
        if not path.is_file():
            raise FileNotFoundError(f"Simulated CCD reference is missing: {path}")
        value = torch.load(path, map_location="cpu", weights_only=True).float()
    else:
        capture = root / "ccd_captured"
        matches = [candidate for suffix in (".pt", ".npy", ".tif", ".tiff", ".png") for candidate in [capture / f"{key}{suffix}"] if candidate.is_file()]
        if len(matches) != 1:
            raise FileNotFoundError(
                f"Expected exactly one captured CCD file for {key} in {capture}; "
                "allowed extensions are .pt/.npy/.tif/.tiff/.png"
            )
        path = matches[0]
        if path.suffix == ".pt":
            value = torch.load(path, map_location="cpu", weights_only=True).float()
        elif path.suffix == ".npy":
            value = torch.from_numpy(np.load(path)).float()
        else:
            value = _load_numeric_image(path)
    value = value.squeeze()
    source_shape = tuple(int(v) for v in value.shape)
    roi = runtime.hardware.capture_roi_xywh
    if roi is not None:
        x, y, width, height = roi
        if value.ndim != 2 or x < 0 or y < 0 or y + height > value.shape[0] or x + width > value.shape[1]:
            raise RuntimeError(f"Configured CCD ROI {roi} is outside captured shape {tuple(value.shape)}")
        value = value[y : y + height, x : x + width]
    if value.ndim != 2:
        raise RuntimeError(f"Registered CCD intensity must be 2-D; got {tuple(value.shape)}")
    cropped_shape = tuple(int(v) for v in value.shape)
    value = (value - runtime.hardware.capture_dark_level).clamp_min(0.0)
    if not use_simulation and runtime.hardware.capture_flip_vertical:
        value = torch.flip(value, dims=(-2,))
    if not use_simulation and runtime.hardware.capture_flip_horizontal:
        value = torch.flip(value, dims=(-1,))
    expected = (runtime.settings.active_size, runtime.settings.active_size)
    factor = 1 if use_simulation else runtime.hardware.capture_binning_factor
    physical_expected = (expected[0] * factor, expected[1] * factor)
    captured_shape = tuple(int(v) for v in value.shape)
    if captured_shape != physical_expected:
        if runtime.hardware.capture_registration_mode == "strict":
            raise RuntimeError(
                f"Registered CCD intensity must be exactly {physical_expected} before "
                f"{factor}x binning; got {captured_shape}. Set capture.roi_xywh "
                "explicitly, or set capture.registration_mode=nearest_resize."
            )
        value = F.interpolate(
            value[None, None], size=physical_expected, mode="nearest"
        )[0, 0]
    if factor > 1:
        value = bin_ccd_superpixels(
            value,
            factor,
            runtime.hardware.capture_binning_reduction,
        )
    if tuple(value.shape) != expected:
        raise RuntimeError(f"Binned CCD intensity must be {expected}, got {tuple(value.shape)}")
    if not torch.isfinite(value).all() or torch.any(value < 0):
        raise RuntimeError(f"CCD intensity {path} contains invalid values")
    if not use_simulation:
        registration_root = root / "registered_ccd"
        registration_root.mkdir(parents=True, exist_ok=True)
        write_json(
            registration_root / f"{key}.json",
            {
                "source": str(path),
                "source_shape": list(source_shape),
                "roi_xywh_in_source_coordinates": runtime.hardware.capture_roi_xywh,
                "shape_after_roi": list(cropped_shape),
                "flip_vertical_after_roi": runtime.hardware.capture_flip_vertical,
                "flip_horizontal_after_roi": runtime.hardware.capture_flip_horizontal,
                "target_physical_shape": list(physical_expected),
                "registration_interpolation": (
                    "none" if captured_shape == physical_expected else "nearest"
                ),
                "binning_factor": factor,
                "binning_reduction": runtime.hardware.capture_binning_reduction,
                "final_logical_shape": list(expected),
                "operation_order": [
                    "load_raw_intensity",
                    "crop_roi_in_camera_coordinates",
                    "subtract_dark_level_and_clamp_nonnegative",
                    "flip_vertical" if runtime.hardware.capture_flip_vertical else "no_flip",
                    "flip_horizontal" if runtime.hardware.capture_flip_horizontal else "no_flip",
                    "nearest_resize_if_needed",
                    "exact_superpixel_binning",
                ],
                "warning": (
                    "Nearest-neighbor registration corrects array size only. "
                    "Set capture.roi_xywh from an optical calibration target to "
                    "avoid geometric misalignment or aspect-ratio distortion."
                ),
            },
        )
    return value


def has_captured_intensity(runtime: Runtime, stage: str, key: str) -> bool:
    """Return whether exactly one supported physical CCD file is available."""
    capture = runtime.hardware.output_dir / STAGES[stage] / "ccd_captured"
    matches = [
        capture / f"{key}{suffix}"
        for suffix in (".pt", ".npy", ".tif", ".tiff", ".png")
        if (capture / f"{key}{suffix}").is_file()
    ]
    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple captured CCD files share sample key {key}: "
            f"{[str(path) for path in matches]}"
        )
    return len(matches) == 1


def _processed_amplitude(runtime: Runtime, stage: str, key: str) -> torch.Tensor:
    path = runtime.hardware.output_dir / STAGES[stage] / "amplitude_raw" / f"{key}.pt"
    if not path.is_file():
        raise FileNotFoundError(f"Processed amplitude is missing: {path}; complete the preceding electronic bridge")
    return torch.load(path, map_location="cpu", weights_only=True).float()


def _input_for_sample(runtime: Runtime, sample: GrocerySample) -> tuple[dict[str, torch.Tensor], Image.Image, Image.Image]:
    with Image.open(sample.image_path) as source:
        original = source.convert("RGB")
    processed = ImageOps.fit(
        original,
        (runtime.settings.image_size, runtime.settings.image_size),
        method=Image.Resampling.BICUBIC,
        centering=(0.5, 0.5),
    )
    inputs = preprocess_images(runtime.loaded.processor, [processed], runtime.settings.instruction)
    validate_token_budgets(inputs, runtime.settings)
    return move_inputs(inputs, runtime.loaded.device), original, processed


def _clear_replay(runtime: Runtime) -> None:
    runtime.replacement.vision_surrogate.core.clear_hardware_replay()
    runtime.replacement.language_surrogate.core.clear_hardware_replay()


@torch.no_grad()
def _forward(runtime: Runtime, sample: GrocerySample) -> tuple[torch.Tensor, dict[str, torch.Tensor], Image.Image, Image.Image]:
    inputs, original, processed = _input_for_sample(runtime, sample)
    embedding, _ = student_embeddings(runtime.loaded.model, runtime.replacement, runtime.readout, inputs)
    return embedding[0].detach().cpu(), inputs, original, processed


def _capture_values(core: Any) -> dict[str, torch.Tensor]:
    conversion = core.interlayer_conversions[0]
    required = {
        "token_field": core.last_input_fields,
        "routed_amplitude": core.last_amplitude_slm_canvas,
        "expert_ccd": conversion.last_input_intensity,
        "reload": conversion.last_output_amplitude,
        "final_ccd": core.last_detector_intensity,
        "expert_complex": conversion.last_input_complex_field,
        "final_complex": core.last_detector_complex_field,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise RuntimeError(f"Optical capture is missing tensors: {missing}")
    return {
        "token_field": required["token_field"][0],
        "routed_amplitude": _active_crop(core, required["routed_amplitude"][0]),
        "expert_ccd": _active_crop(core, required["expert_ccd"][0]),
        "reload": _active_crop(core, required["reload"][0].real),
        "final_ccd": required["final_ccd"][0],
        "expert_complex": _active_crop(core, required["expert_complex"][0]),
        "final_complex": required["final_complex"][0],
    }


def _save_phase_mask(runtime: Runtime, stack: str, plane: str, value: torch.Tensor, stage: str) -> dict[str, Any]:
    if runtime.hardware.minimal_artifacts:
        bmp = (
            runtime.hardware.output_dir
            / "phase_bmp"
            / f"{stack}_{plane}_phase_1920x1200.bmp"
        )
        ratio = runtime.settings.pixel_pitch_um / runtime.hardware.slm_pixel_pitch_um
        return export_centered_bmp(
            value,
            bmp,
            value_type="phase",
            scale_factor=int(round(ratio)),
            slm_width=runtime.hardware.phase_slm_size[0],
            slm_height=runtime.hardware.phase_slm_size[1],
            flip_vertical=runtime.hardware.phase_flip_vertical,
        )
    root = runtime.hardware.output_dir / "00_masks" / STAGES[stage]
    raw = root / f"{stack}_{plane}_phase_rad.pt"
    bmp = root / f"{stack}_{plane}_phase_1920x1200.bmp"
    _save_tensor(raw, value, torch.float32)
    ratio = runtime.settings.pixel_pitch_um / runtime.hardware.slm_pixel_pitch_um
    report = export_centered_bmp(
        value,
        bmp,
        value_type="phase",
        scale_factor=int(round(ratio)),
        slm_width=runtime.hardware.phase_slm_size[0],
        slm_height=runtime.hardware.phase_slm_size[1],
        flip_vertical=runtime.hardware.phase_flip_vertical,
    )
    write_json(root / "mask_manifest.json", {**report, "raw_tensor": str(raw), "stats": tensor_stats(value)})
    return report


def _capture_readme(runtime: Runtime, stage: str) -> None:
    directory = runtime.hardware.output_dir / STAGES[stage] / "ccd_captured"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "README.md").write_text(
        "# CCD capture input\n\n"
        "Play `../amplitude_to_play/*.bmp` in the exact order given by "
        "`../../00_manifest/play_order.csv`, while the shared mask in "
        f"`../../00_masks/{STAGES[stage]}/` is loaded. Save one file with the exact "
        "same basename as each amplitude BMP. Accepted lossless formats: `.pt`, `.npy`, "
        "`.tif`, `.tiff`, `.png`. Data must be square-law detector intensity, not "
        "amplitude. The registered physical ROI must be "
        f"{runtime.settings.active_size * runtime.hardware.capture_binning_factor}x"
        f"{runtime.settings.active_size * runtime.hardware.capture_binning_factor}; "
        "use `capture.roi_xywh` for a larger sensor image. If "
        f"`capture.flip_vertical={str(runtime.hardware.capture_flip_vertical).lower()}`, "
        "the cropped physical CCD is vertically flipped into simulation coordinates. "
        f"With `capture.flip_horizontal={str(runtime.hardware.capture_flip_horizontal).lower()}`, "
        "it is then horizontally flipped. "
        "`capture.registration_mode=nearest_resize`, a mismatched cropped frame is "
        "resized to that physical ROI using nearest-neighbor interpolation. The pipeline then performs "
        f"non-interpolating {runtime.hardware.capture_binning_factor}x"
        f" {runtime.hardware.capture_binning_reduction} binning. JPEG is forbidden.\n",
        encoding="utf-8",
    )


def prepare(runtime: Runtime) -> dict[str, Any]:
    output = runtime.hardware.output_dir
    if runtime.hardware.clean_output_before_prepare and output.exists():
        shutil.rmtree(output)
    elif output.exists():
        checkpoint_metadata = (
            output / "00_manifest" / "checkpoint_metadata.json"
        ).resolve()
        existing = [
            path
            for path in output.rglob("*")
            if path.is_file() and path.resolve() != checkpoint_metadata
        ]
        if existing:
            raise FileExistsError(
                f"Hardware session already contains {len(existing)} files under {output}. "
                "Use a new --output-dir; do not overwrite a session containing CCD captures."
            )
    output.mkdir(parents=True, exist_ok=True)
    selected = select_samples(runtime.bundle, runtime.settings, runtime.hardware)
    rows: list[dict[str, Any]] = []
    for index, (role, sample) in enumerate(selected):
        rows.append({
            "play_index": index,
            "sample_key": _sample_key(index, sample, role),
            "role": role,
            "dataset_split": sample.split,
            "source_split": sample.source_split,
            "is_gallery": bool(sample.is_gallery),
            "sample_id": sample.sample_id,
            "sku_index": sample.sku_index,
            "sku_name": sample.sku_name,
            "source_image_path": str(sample.image_path),
        })
    write_csv(output / "00_manifest" / "play_order.csv", rows, list(rows[0]))
    checkpoint_copy = None
    if not runtime.hardware.minimal_artifacts:
        shutil.copy2(runtime.hardware.config_path, output / "00_manifest" / "hardware_config.yaml")
        shutil.copy2(runtime.settings.config_path, output / "00_manifest" / "model_config.yaml")
        if runtime.hardware.copy_checkpoint_to_output:
            shutil.copy2(runtime.hardware.checkpoint, output / "00_manifest" / "student_checkpoint.pt")
            checkpoint_copy = output / "00_manifest" / "student_checkpoint.pt"
    phases = {
        "vision": phase_tensors(runtime.replacement.vision_surrogate.core),
        "language": phase_tensors(runtime.replacement.language_surrogate.core),
    }
    phase_reports = {
        "vision_expert": _save_phase_mask(runtime, "vision", "expert", phases["vision"]["physical_expert_mosaic_rad"], "vision_expert"),
        "vision_global": _save_phase_mask(runtime, "vision", "global", phases["vision"]["physical_global_phase_rad"], "vision_global"),
        "language_expert": _save_phase_mask(runtime, "language", "expert", phases["language"]["physical_expert_mosaic_rad"], "language_expert"),
        "language_global": _save_phase_mask(runtime, "language", "global", phases["language"]["physical_global_phase_rad"], "language_global"),
    }
    if not runtime.hardware.minimal_artifacts:
        for stage in STAGES:
            _capture_readme(runtime, stage)
    runtime.replacement.set_intermediate_field_capture(True, sample_count=1)
    runtime.loaded.model.eval(); runtime.replacement.vision_surrogate.eval(); runtime.replacement.language_surrogate.eval(); runtime.readout.eval()
    by_id = _samples_by_id(runtime.bundle)
    try:
        for row in rows:
            key = row["sample_key"]
            sample = by_id[row["sample_id"]]
            _clear_replay(runtime)
            embedding, inputs, original, processed = _forward(runtime, sample)
            if not runtime.hardware.minimal_artifacts:
                original_path = output / "00_input_images" / "original" / f"{key}.png"
                processor_path = output / "00_input_images" / "processor_224" / f"{key}.png"
                original_path.parent.mkdir(parents=True, exist_ok=True)
                processor_path.parent.mkdir(parents=True, exist_ok=True)
                original.save(original_path)
                processed.save(processor_path)
            vision = _capture_values(runtime.replacement.vision_surrogate.core)
            language = _capture_values(runtime.replacement.language_surrogate.core)
            if not runtime.hardware.minimal_artifacts:
                _save_tensor(output / STAGES["vision_expert"] / "token_field_224" / f"{key}.pt", vision["token_field"], torch.float32)
            _save_amplitude(runtime, "vision_expert", key, vision["routed_amplitude"], simulation=False)
            _save_simulated_ccd(runtime, "vision_expert", key, vision["expert_ccd"])
            _save_simulated_complex_field(runtime, "vision_expert", key, vision["expert_complex"])
            _save_amplitude(runtime, "vision_global", key, vision["reload"], simulation=True)
            _save_simulated_ccd(runtime, "vision_global", key, vision["final_ccd"])
            _save_simulated_complex_field(runtime, "vision_global", key, vision["final_complex"])
            if not runtime.hardware.minimal_artifacts:
                _save_tensor(output / STAGES["language_expert"] / "simulation_reference" / "token_field_224" / f"{key}.pt", language["token_field"], runtime.hardware.simulation_tensor_dtype)
            _save_amplitude(runtime, "language_expert", key, language["routed_amplitude"], simulation=True)
            _save_simulated_ccd(runtime, "language_expert", key, language["expert_ccd"])
            _save_simulated_complex_field(runtime, "language_expert", key, language["expert_complex"])
            _save_amplitude(runtime, "language_global", key, language["reload"], simulation=True)
            _save_simulated_ccd(runtime, "language_global", key, language["final_ccd"])
            _save_simulated_complex_field(runtime, "language_global", key, language["final_complex"])
            if not runtime.hardware.minimal_artifacts:
                _save_tensor(output / "05_retrieval" / "simulation_embeddings" / f"{key}.pt", embedding, torch.float32)
                write_json(output / "00_manifest" / "sample_metadata" / f"{key}.json", {
                **row,
                "image_grid_thw": inputs["image_grid_thw"].detach().cpu().tolist(),
                "visual_token_count": int(inputs["image_grid_thw"].long().prod(dim=-1).max()),
                "language_sequence_length": int(inputs["attention_mask"].sum()),
                "vision_routing_weights": runtime.replacement.vision_surrogate.core.last_routing["weights"][0].detach().cpu().tolist(),
                "language_routing_weights": runtime.replacement.language_surrogate.core.last_routing["weights"][0].detach().cpu().tolist(),
                })
    finally:
        runtime.replacement.set_intermediate_field_capture(False, sample_count=1)
        _clear_replay(runtime)
    report = {
        "schema_version": 1,
        "model_contains_language_optical_stack": True,
        "physical_exposure_count_per_sample": 4,
        "sample_count": len(rows),
        "gallery_count": sum(row["role"] == "gallery" for row in rows),
        "train_count": sum(row["role"] == "train" for row in rows),
        "query_count": sum(row["role"] == "query" for row in rows),
        "selection_mode": runtime.hardware.selection_mode,
        "checkpoint": str(runtime.hardware.checkpoint),
        "checkpoint_copy": str(checkpoint_copy) if checkpoint_copy else None,
        "checkpoint_sha256": _sha256(runtime.hardware.checkpoint),
        "model_config": str(runtime.settings.config_path),
        "amplitude_encoding": {
            "mode": runtime.hardware.amplitude_encoding_mode,
            "percentile": runtime.hardware.amplitude_encoding_percentile,
            "percentile_population": "strictly_positive_pixels",
            "gamma": runtime.hardware.amplitude_encoding_gamma,
            "warning": (
                "Percentile clipping raises average SLM transmission but changes "
                "the clipped tail. Every per-file divisor is recorded in "
                "amplitude_metadata."
            ),
        },
        "capture": {
            "roi_xywh": runtime.hardware.capture_roi_xywh,
            "flip_vertical_after_roi": runtime.hardware.capture_flip_vertical,
            "flip_horizontal_after_roi": runtime.hardware.capture_flip_horizontal,
            "dark_level": runtime.hardware.capture_dark_level,
            "binning_factor": runtime.hardware.capture_binning_factor,
            "binning_reduction": runtime.hardware.capture_binning_reduction,
            "registration_mode": runtime.hardware.capture_registration_mode,
            "logical_active_shape": [runtime.settings.active_size, runtime.settings.active_size],
            "required_physical_roi_shape": [
                runtime.settings.active_size * runtime.hardware.capture_binning_factor,
                runtime.settings.active_size * runtime.hardware.capture_binning_factor,
            ],
        },
        "dataset_manifest_sha256": runtime.bundle.manifest_digest,
        "phase_masks": phase_reports,
        "phase_flip_vertical_before_export": runtime.hardware.phase_flip_vertical,
        "artifact_profile": "minimal" if runtime.hardware.minimal_artifacts else "full",
        "sequence": [
            "vision expert -> CCD -> expert LayerNorm/ReLU/same routing weights/hard zero",
            "vision global -> CCD -> pooling/LN/ReLU/output adapter/residual/frozen Qwen bridge",
            "language expert -> CCD -> expert LayerNorm/ReLU/same routing weights/hard zero",
            "language global -> CCD -> pooling/LN/ReLU/output adapter/residual/final RMSNorm/retrieval readout",
        ],
        "warning": "CCD files are already intensity and are never squared a second time.",
    }
    write_json(output / "00_manifest" / "deployment.json", report)
    return report


def _write_bridge_report(runtime: Runtime, stage: str, key: str, values: dict[str, Any]) -> None:
    write_json(runtime.hardware.output_dir / STAGES[stage] / "electronic_output" / f"{key}.json", values)


def _replay_vision(runtime: Runtime, key: str, use_simulation: bool) -> None:
    core = runtime.replacement.vision_surrogate.core
    final_ccd = load_captured_intensity(runtime, "vision_global", key, use_simulation=use_simulation).to(runtime.loaded.device)
    reload_path = (
        runtime.hardware.output_dir
        / STAGES["vision_global"]
        / "amplitude_raw"
        / f"{key}.pt"
    )
    stage_reload_fields = None
    if reload_path.is_file():
        reload_amplitude = torch.load(
            reload_path, map_location="cpu", weights_only=True
        ).float().to(runtime.loaded.device)
        stage_reload_fields = {
            0: _active_to_canvas(core, reload_amplitude).unsqueeze(0)
        }
    core.set_hardware_replay(
        stage_reload_fields=stage_reload_fields,
        final_detector_intensity=final_ccd.unsqueeze(0),
    )


def process_vision_expert(runtime: Runtime, *, use_simulation: bool) -> None:
    by_id = _samples_by_id(runtime.bundle)
    core = runtime.replacement.vision_surrogate.core
    for row in _manifest_rows(runtime):
        key = row["sample_key"]
        _clear_replay(runtime)
        _forward(runtime, by_id[row["sample_id"]])
        measured = load_captured_intensity(runtime, "vision_expert", key, use_simulation=use_simulation)
        full = _intensity_active_to_canvas(core, measured).unsqueeze(0).to(runtime.loaded.device)
        reload_field = core.interlayer_conversions[0].forward_intensity(
            full,
            selected_experts=core.last_routing["selected_mask"],
            routing_weights=core.last_routing["weights"],
        )
        amplitude = _active_crop(core, reload_field[0].real).detach().cpu()
        _save_amplitude(runtime, "vision_global", key, amplitude, simulation=False)
        reference = _processed_or_reference(runtime, "vision_global", key, reference=True)
        _write_bridge_report(runtime, "vision_expert", key, {
            "input": "measured square-law intensity; no second squaring",
            "operation": "per-expert LayerNorm -> ReLU -> same routing weight -> hard zero unselected",
            "next_stage": STAGES["vision_global"],
            "comparison_to_simulation": _comparison(amplitude, reference),
        })


def _processed_or_reference(runtime: Runtime, stage: str, key: str, *, reference: bool) -> torch.Tensor:
    root = runtime.hardware.output_dir / STAGES[stage]
    if reference:
        root = root / "simulation_reference"
    path = root / "amplitude_raw" / f"{key}.pt"
    return torch.load(path, map_location="cpu", weights_only=True).float()


def _comparison(actual: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    actual = actual.float(); reference = reference.float()
    difference = actual - reference
    cosine = torch.nn.functional.cosine_similarity(actual.flatten()[None], reference.flatten()[None]).item()
    return {
        "mse": float(difference.square().mean()),
        "mae": float(difference.abs().mean()),
        "relative_l2": float(difference.norm() / reference.norm().clamp_min(1.0e-12)),
        "cosine": float(cosine),
    }


def process_vision_global(runtime: Runtime, *, use_simulation: bool) -> None:
    by_id = _samples_by_id(runtime.bundle)
    language = runtime.replacement.language_surrogate.core
    runtime.replacement.set_intermediate_field_capture(True, sample_count=1)
    try:
        for row in _manifest_rows(runtime):
            key = row["sample_key"]
            _clear_replay(runtime); _replay_vision(runtime, key, use_simulation)
            _forward(runtime, by_id[row["sample_id"]])
            if language.last_input_fields is None or language.last_amplitude_slm_canvas is None:
                raise RuntimeError("Vision-to-language bridge did not produce language optical input")
            token = language.last_input_fields[0].detach().cpu()
            amplitude = _active_crop(language, language.last_amplitude_slm_canvas[0]).detach().cpu()
            _save_tensor(runtime.hardware.output_dir / STAGES["language_expert"] / "token_field_224" / f"{key}.pt", token, torch.float32)
            _save_amplitude(runtime, "language_expert", key, amplitude, simulation=False)
            reference = _processed_or_reference(runtime, "language_expert", key, reference=True)
            _write_bridge_report(runtime, "vision_global", key, {
                "input": "measured vision-global CCD intensity",
                "operation": "CCD pooling/LN/ReLU -> vision output adapter/residual -> frozen Qwen merger/DeepStack/token injection -> language input adapter/router",
                "next_stage": STAGES["language_expert"],
                "comparison_to_simulation": _comparison(amplitude, reference),
            })
    finally:
        runtime.replacement.set_intermediate_field_capture(False, sample_count=1)
        _clear_replay(runtime)


def process_language_expert(runtime: Runtime, *, use_simulation: bool) -> None:
    by_id = _samples_by_id(runtime.bundle)
    core = runtime.replacement.language_surrogate.core
    for row in _manifest_rows(runtime):
        key = row["sample_key"]
        _clear_replay(runtime)
        if use_simulation or has_captured_intensity(runtime, "vision_global", key):
            _replay_vision(runtime, key, use_simulation)
        _forward(runtime, by_id[row["sample_id"]])
        measured = load_captured_intensity(runtime, "language_expert", key, use_simulation=use_simulation)
        full = _intensity_active_to_canvas(core, measured).unsqueeze(0).to(runtime.loaded.device)
        reload_field = core.interlayer_conversions[0].forward_intensity(
            full,
            selected_experts=core.last_routing["selected_mask"],
            routing_weights=core.last_routing["weights"],
        )
        amplitude = _active_crop(core, reload_field[0].real).detach().cpu()
        _save_amplitude(runtime, "language_global", key, amplitude, simulation=False)
        reference = _processed_or_reference(runtime, "language_global", key, reference=True)
        _write_bridge_report(runtime, "language_expert", key, {
            "input": "measured square-law intensity; no second squaring",
            "operation": "per-expert LayerNorm -> ReLU -> same routing weight -> hard zero unselected",
            "next_stage": STAGES["language_global"],
            "comparison_to_simulation": _comparison(amplitude, reference),
        })
        _clear_replay(runtime)


def process_language_global(runtime: Runtime, *, use_simulation: bool) -> dict[str, Any]:
    by_id = _samples_by_id(runtime.bundle)
    embeddings: list[torch.Tensor] = []
    rows = _manifest_rows(runtime)
    for row in rows:
        key = row["sample_key"]
        _clear_replay(runtime)
        # A final-plane-only calibration has no earlier physical CCD files.
        # In that case the upstream Vision/Language context remains simulated,
        # while the Language-global detector is still replaced by the measured
        # final CCD below.  If a Vision-global capture exists, prefer it.
        if use_simulation or has_captured_intensity(runtime, "vision_global", key):
            _replay_vision(runtime, key, use_simulation)
        language = runtime.replacement.language_surrogate.core
        final_ccd = load_captured_intensity(runtime, "language_global", key, use_simulation=use_simulation).to(runtime.loaded.device)
        reload_path = (
            runtime.hardware.output_dir
            / STAGES["language_global"]
            / "amplitude_raw"
            / f"{key}.pt"
        )
        stage_reload_fields = None
        if reload_path.is_file():
            reload_amplitude = torch.load(
                reload_path, map_location="cpu", weights_only=True
            ).float().to(runtime.loaded.device)
            stage_reload_fields = {
                0: _active_to_canvas(language, reload_amplitude).unsqueeze(0)
            }
        language.set_hardware_replay(
            stage_reload_fields=stage_reload_fields,
            final_detector_intensity=final_ccd.unsqueeze(0),
        )
        embedding, _, _, _ = _forward(runtime, by_id[row["sample_id"]])
        embeddings.append(embedding)
        _save_tensor(runtime.hardware.output_dir / "05_retrieval" / "measured_embeddings" / f"{key}.pt", embedding, torch.float32)
        reference_path = runtime.hardware.output_dir / "05_retrieval" / "simulation_embeddings" / f"{key}.pt"
        reference = torch.load(reference_path, map_location="cpu", weights_only=True).float()
        _write_bridge_report(runtime, "language_global", key, {
            "input": "measured language-global CCD intensity",
            "operation": "CCD pooling/LN/ReLU -> language output adapter/residual -> frozen final RMSNorm -> retrieval LN/Linear64/L2 normalize",
            "embedding_norm": float(embedding.norm()),
            "comparison_to_simulation": _comparison(embedding, reference),
        })
        _clear_replay(runtime)
    matrix = torch.stack(embeddings)
    samples = [by_id[row["sample_id"]] for row in rows]
    gallery_indexes = [index for index, row in enumerate(rows) if row["role"] == "gallery"]
    query_indexes = [index for index, row in enumerate(rows) if row["role"] == "query"]
    evaluation = evaluate_embeddings(
        matrix[query_indexes],
        [samples[index] for index in query_indexes],
        matrix[gallery_indexes],
        [samples[index] for index in gallery_indexes],
        runtime.bundle.class_names,
        runtime.settings.gallery_aggregation,
        system_name="hardware_replay_query_vs_hardware_replay_gallery",
    )
    metrics = {
        **evaluation.metrics,
        "capture_source": "simulation_reference" if use_simulation else "physical_ccd",
        "checkpoint": str(runtime.hardware.checkpoint),
        "hardware_adaptation": "none_in_hardware_pipeline",
        "test_captures_used_for_hardware_adaptation": False,
        "independent_test_evaluation": True,
    }
    write_json(runtime.hardware.output_dir / "05_retrieval" / "metrics.json", metrics)
    write_csv(runtime.hardware.output_dir / "05_retrieval" / "retrieval_results.csv", evaluation.rows, list(evaluation.rows[0]))
    write_csv(
        runtime.hardware.output_dir / "05_retrieval" / "confusion_matrix.csv",
        [{"true_sku": name, **{runtime.bundle.class_names[col]: int(evaluation.confusion[row, col]) for col in range(len(runtime.bundle.class_names))}} for row, name in enumerate(runtime.bundle.class_names)],
        ["true_sku", *runtime.bundle.class_names],
    )
    return metrics


def build_runtime(hardware: HardwareConfig) -> Runtime:
    settings = load_settings(hardware.model_config)
    bundle = prepare_grocery_subset(settings, persist=True)
    device = torch.device(settings.device if settings.device != "cuda" or torch.cuda.is_available() else "cpu")
    loaded = load_backbone(settings, device)
    settings.resolve_architecture(loaded.model)
    replacement, readout = build_optical_student(loaded, settings)
    checkpoint = load_checkpoint(hardware.checkpoint, replacement, readout)
    if len(replacement.vision_surrogate.core.expert_layers) != 1 or len(replacement.language_surrogate.core.expert_layers) != 1:
        raise RuntimeError("Hardware pipeline requires the trained one-expert-plane + one-global-plane baseline")
    replacement.set_phase_dropout_active(False)
    loaded.model.eval(); replacement.vision_surrogate.eval(); replacement.language_surrogate.eval(); readout.eval()
    write_json(hardware.output_dir / "00_manifest" / "checkpoint_metadata.json", checkpoint.get("metadata", {}))
    return Runtime(hardware, settings, bundle, loaded, replacement, readout)


def close_runtime(runtime: Runtime) -> None:
    _clear_replay(runtime)
    runtime.replacement.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage-first SLM/CCD deployment pipeline for the best Grocery10 optical Student")
    parser.add_argument("--config", required=True)
    parser.add_argument("--phase", required=True, choices=sorted(PHASES))
    parser.add_argument("--use-simulation", action="store_true", help="Use saved simulated CCD intensity instead of physical capture files")
    parser.add_argument("--output-dir", default=None, help="Optional deployment-output override, useful for a disposable replay smoke test")
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint override; its architecture must match model_config")
    parser.add_argument("--queries-per-sku", type=int, default=None, help="Optional query-count override; all gallery images remain included")
    parser.add_argument(
        "--selection-mode",
        choices=("selected_test", "test_only", "full_dataset"),
        default=None,
        help=(
            "Prepare manifest override: test_only exports gallery+all test; "
            "full_dataset exports gallery+all train+all test"
        ),
    )
    parser.add_argument("--sample-limit", type=int, default=None, help="Prepare-only equipment smoke limit; do not use for final retrieval evaluation")
    parser.add_argument(
        "--artifact-profile",
        choices=("minimal", "full"),
        default=None,
        help=(
            "prepare defaults to minimal (BMPs + theoretical CCD + play manifest). "
            "Use full only when manually constructing a complete bridge session; "
            "hardware_automation selects the full profile from its YAML."
        ),
    )
    args = parser.parse_args()
    hardware = load_hardware_config(args.config)
    # `hardware_pipeline --phase prepare` is the human-facing export command.
    # Keep it clean by default even though the same YAML uses `full` when it is
    # consumed by hardware_automation for iterative CCD/electronic bridges.
    requested_profile = args.artifact_profile
    if requested_profile is None and args.phase == "prepare":
        requested_profile = "minimal"
    if requested_profile is not None:
        hardware = replace(
            hardware,
            minimal_artifacts=requested_profile == "minimal",
        )
    if hardware.minimal_artifacts and args.phase != "prepare":
        raise RuntimeError(
            "artifacts.profile=minimal is an export-only package; use --phase prepare. "
            "Use the full hardware profile for physical CCD bridge processing."
        )
    if args.output_dir is not None:
        hardware = replace(hardware, output_dir=Path(args.output_dir).expanduser().resolve())
    if args.checkpoint is not None:
        hardware = replace(hardware, checkpoint=Path(args.checkpoint).expanduser().resolve())
    if args.queries_per_sku is not None:
        if args.queries_per_sku <= 0:
            raise ValueError("--queries-per-sku must be positive")
        hardware = replace(hardware, queries_per_sku=args.queries_per_sku)
    if args.selection_mode is not None:
        hardware = replace(hardware, selection_mode=args.selection_mode)
    if args.sample_limit is not None:
        if args.sample_limit <= 0:
            raise ValueError("--sample-limit must be positive")
        hardware = replace(hardware, sample_limit=args.sample_limit)
    seed_everything(42)
    runtime = build_runtime(hardware)
    try:
        if args.phase == "prepare":
            report = prepare(runtime)
            print(
                f"Prepared {report['sample_count']} samples "
                f"(gallery={report['gallery_count']}, train={report['train_count']}, "
                f"test={report['query_count']}) and four shared masks "
                f"under {hardware.output_dir} (artifact_profile={report['artifact_profile']})"
            )
        elif args.phase == "process_vision_expert":
            process_vision_expert(runtime, use_simulation=args.use_simulation)
        elif args.phase == "process_vision_global":
            process_vision_global(runtime, use_simulation=args.use_simulation)
        elif args.phase == "process_language_expert":
            process_language_expert(runtime, use_simulation=args.use_simulation)
        elif args.phase == "process_language_global":
            metrics = process_language_global(runtime, use_simulation=args.use_simulation)
            print(f"Hardware replay Top-1={metrics['top1_retrieval_accuracy']:.4f} Top-3={metrics['top3_retrieval_accuracy']:.4f} MRR={metrics['mrr']:.4f}")
        else:
            prepare(runtime)
            process_vision_expert(runtime, use_simulation=True)
            process_vision_global(runtime, use_simulation=True)
            process_language_expert(runtime, use_simulation=True)
            metrics = process_language_global(runtime, use_simulation=True)
            print(f"Simulation replay complete: Top-1={metrics['top1_retrieval_accuracy']:.4f}")
    finally:
        close_runtime(runtime)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
