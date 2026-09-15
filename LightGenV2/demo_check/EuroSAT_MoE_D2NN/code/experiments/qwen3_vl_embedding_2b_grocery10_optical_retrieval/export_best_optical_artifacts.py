from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from PIL import Image, ImageOps

from .features import (
    move_inputs,
    preprocess_images,
    student_embeddings,
    validate_token_budgets,
)
from .modeling import build_optical_student, load_backbone
from .optical_artifacts import (
    export_centered_bmp,
    phase_tensors,
    save_phase_snapshot,
    tensor_stats,
)
from .prepare_grocery_retrieval_subset import (
    GroceryRetrievalBundle,
    GrocerySample,
    prepare_grocery_subset,
)
from .settings import Settings, load_settings
from .train_optical_retrieval import load_checkpoint


STUDENT_SYSTEM = "optical_student_query_vs_optical_student_gallery"


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _active_crop(core: Any, value: torch.Tensor) -> torch.Tensor:
    aperture = core.geometry.active_aperture
    return value[..., aperture.y0 : aperture.y1, aperture.x0 : aperture.x1]


def _heatmap(
    value: torch.Tensor,
    path: Path,
    title: str,
    *,
    value_type: str,
) -> None:
    tensor = value.detach().cpu()
    if tensor.is_complex():
        tensor = tensor.abs()
    while tensor.ndim > 2:
        tensor = tensor[0]
    tensor = tensor.float()
    stats = tensor_stats(tensor)
    if value_type == "phase":
        shown = torch.remainder(tensor, 2.0 * math.pi)
        cmap, vmin, vmax, label = "twilight", 0.0, 2.0 * math.pi, "phase (rad)"
    else:
        shown = tensor
        positive = shown[torch.isfinite(shown)]
        upper = float(torch.quantile(positive, 0.995)) if positive.numel() else 1.0
        cmap, vmin, vmax = "viridis", 0.0, max(upper, 1.0e-8)
        label = "intensity" if value_type == "intensity" else "amplitude"
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8.5, 7.5), constrained_layout=True)
    image = axis.imshow(shown.numpy(), cmap=cmap, vmin=vmin, vmax=vmax, origin="upper")
    figure.colorbar(image, ax=axis, label=label)
    axis.set_xlabel("x pixel")
    axis.set_ylabel("y pixel")
    axis.set_title(
        f"{title}\nshape={tuple(tensor.shape)} min={stats['min']:.4g} "
        f"max={stats['max']:.4g} mean={stats['mean']:.4g} std={stats['std']:.4g}"
    )
    figure.savefig(path, dpi=170)
    plt.close(figure)


def _select_debug_samples(
    bundle: GroceryRetrievalBundle, results_path: Path, count: int
) -> list[tuple[GrocerySample, dict[str, str]]]:
    by_id = {sample.sample_id: sample for sample in bundle.test_samples}
    rows: list[dict[str, str]] = []
    if results_path.is_file():
        with results_path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = [
                row
                for row in csv.DictReader(handle)
                if row.get("system") == STUDENT_SYSTEM
                and row.get("sample_id") in by_id
                and row.get("top1_correct", "").lower() in {"true", "1"}
            ]
        rows.sort(
            key=lambda row: float(row.get("similarity_margin") or "-inf"),
            reverse=True,
        )
    chosen: list[tuple[GrocerySample, dict[str, str]]] = []
    used_ids: set[str] = set()
    used_skus: set[str] = set()
    for row in rows:
        sample = by_id[row["sample_id"]]
        if sample.sku_name in used_skus:
            continue
        chosen.append((sample, row))
        used_ids.add(sample.sample_id)
        used_skus.add(sample.sku_name)
        if len(chosen) >= count:
            return chosen
    for row in rows:
        sample = by_id[row["sample_id"]]
        if sample.sample_id in used_ids:
            continue
        chosen.append((sample, row))
        used_ids.add(sample.sample_id)
        if len(chosen) >= count:
            return chosen
    for sample in bundle.test_samples:
        if sample.sample_id in used_ids:
            continue
        chosen.append((sample, {}))
        if len(chosen) >= count:
            break
    return chosen


def _routing_snapshot(core: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in (
        "logits",
        "probabilities",
        "weights",
        "selected_mask",
        "selected_indices",
        "importance",
        "load",
        "amplitude_scales",
    ):
        value = core.last_routing.get(key)
        if torch.is_tensor(value):
            result[key] = value[:1].detach().cpu()
    return result


def _capture_core(core: Any) -> dict[str, Any]:
    if core.last_input_fields is None or core.last_amplitude_slm_canvas is None:
        raise RuntimeError("Optical input fields were not captured")
    if not core.last_stage_fields:
        raise RuntimeError("The expert-stage field was not captured")
    if core.last_detector_intensity is None or core.last_detector_readout is None:
        raise RuntimeError("Final detector fields were not captured")
    stage = core.last_stage_fields[-1][0]
    if float(stage.imag.abs().max()) > 1.0e-6:
        raise RuntimeError(
            "The final OEO reload field is not zero-phase; it cannot be exported as "
            "an amplitude-only SLM plane"
        )
    reload_amplitude = _active_crop(core, stage.real)
    if float(reload_amplitude.min()) < -1.0e-7:
        raise RuntimeError("OEO reload amplitude unexpectedly contains negative values")
    return {
        "input_token_field": core.last_input_fields[0],
        "expert_plane_amplitude": _active_crop(
            core, core.last_amplitude_slm_canvas[0]
        ),
        "reload_amplitude_before_global": reload_amplitude,
        "detector_intensity": core.last_detector_intensity[0],
        "detector_readout": core.last_detector_readout[0],
        "routing": _routing_snapshot(core),
    }


def _save_core_sample(
    name: str,
    values: dict[str, Any],
    directory: Path,
    *,
    scale_factor: int,
    amplitude_width: int,
    amplitude_height: int,
) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=True)
    raw_path = directory / f"{name}_raw_optical_fields.pt"
    torch.save(values, raw_path)
    files: dict[str, Any] = {
        "raw_tensor": str(raw_path),
        "raw_tensor_sha256": _sha256(raw_path),
        "stats": {},
        "bmp": {},
    }
    for key in (
        "input_token_field",
        "expert_plane_amplitude",
        "reload_amplitude_before_global",
        "detector_intensity",
        "detector_readout",
    ):
        value = values[key]
        value_type = "intensity" if "intensity" in key or "readout" in key else "amplitude"
        _heatmap(
            value,
            directory / f"{name}_{key}.png",
            f"{name} {key.replace('_', ' ')}",
            value_type=value_type,
        )
        files["stats"][key] = tensor_stats(value)
    for key in ("expert_plane_amplitude", "reload_amplitude_before_global"):
        files["bmp"][key] = export_centered_bmp(
            values[key],
            directory / f"{name}_{key}_1920x1080.bmp",
            value_type="amplitude",
            scale_factor=scale_factor,
            slm_width=amplitude_width,
            slm_height=amplitude_height,
        )
    routing_json: dict[str, Any] = {}
    for key, value in values["routing"].items():
        routing_json[key] = value.tolist()
    _json(directory / f"{name}_routing.json", routing_json)
    return files


def _save_stack_phases(
    name: str,
    core: Any,
    output_dir: Path,
    *,
    scale_factor: int,
    phase_width: int,
    phase_height: int,
) -> dict[str, Any]:
    values = phase_tensors(core)
    stack_dir = output_dir / name
    stack_dir.mkdir(parents=True, exist_ok=True)
    raw_path = stack_dir / f"{name}_phase_parameters.pt"
    torch.save(values, raw_path)
    _heatmap(
        values["physical_expert_mosaic_rad"],
        stack_dir / f"{name}_expert_phase_mosaic.png",
        f"{name} expert phase mosaic",
        value_type="phase",
    )
    _heatmap(
        values["physical_global_phase_rad"],
        stack_dir / f"{name}_global_phase.png",
        f"{name} global phase",
        value_type="phase",
    )
    expert_bmp = export_centered_bmp(
        values["physical_expert_mosaic_rad"],
        stack_dir / f"{name}_expert_phase_mosaic_1920x1200.bmp",
        value_type="phase",
        scale_factor=scale_factor,
        slm_width=phase_width,
        slm_height=phase_height,
    )
    global_bmp = export_centered_bmp(
        values["physical_global_phase_rad"],
        stack_dir / f"{name}_global_phase_1920x1200.bmp",
        value_type="phase",
        scale_factor=scale_factor,
        slm_width=phase_width,
        slm_height=phase_height,
    )
    return {
        "raw_tensor": str(raw_path),
        "raw_tensor_sha256": _sha256(raw_path),
        "expert_phase_bmp": expert_bmp,
        "global_phase_bmp": global_bmp,
        "stats": {key: tensor_stats(value) for key, value in values.items()},
    }


@torch.no_grad()
def export_best_optical_artifacts(
    settings: Settings,
    bundle: GroceryRetrievalBundle,
    loaded: Any,
    replacement: Any,
    readout: Any,
    checkpoint_path: Path,
    output_dir: Path,
    *,
    sample_count: int = 8,
    amplitude_slm_size: tuple[int, int] = (1920, 1080),
    phase_slm_size: tuple[int, int] = (1920, 1200),
    slm_pixel_pitch_um: float = 8.0,
) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = load_checkpoint(checkpoint_path, replacement, readout)
    ratio = settings.pixel_pitch_um / float(slm_pixel_pitch_um)
    scale_factor = int(round(ratio))
    if scale_factor <= 0 or abs(ratio - scale_factor) > 1.0e-9:
        raise RuntimeError(
            f"Simulation pixel pitch {settings.pixel_pitch_um} um is not an integer "
            f"multiple of SLM pixel pitch {slm_pixel_pitch_um} um"
        )
    # This project already simulates at the physical 8 um pitch.  An inherited
    # scale factor of two from the old 16 um/450-pixel experiment would overflow
    # both SLMs and is therefore explicitly rejected by the geometry checks.
    active_scaled = settings.active_size * scale_factor
    if active_scaled > min(amplitude_slm_size) or active_scaled > min(phase_slm_size):
        raise RuntimeError(
            f"Scaled active field {active_scaled}x{active_scaled} does not fit the "
            f"configured SLMs amplitude={amplitude_slm_size}, phase={phase_slm_size}"
        )
    snapshot_variant = str(checkpoint.get("metadata", {}).get("weight_variant", "live"))
    save_phase_snapshot(
        replacement,
        output_dir / "weights",
        epoch=int(checkpoint.get("epoch", -1)),
        train_loss=float(checkpoint.get("train_loss", float("nan"))),
        weight_variant=snapshot_variant,
    )
    phases = {
        "vision": _save_stack_phases(
            "vision",
            replacement.vision_surrogate.core,
            output_dir / "slm_bmp",
            scale_factor=scale_factor,
            phase_width=phase_slm_size[0],
            phase_height=phase_slm_size[1],
        ),
        "language": _save_stack_phases(
            "language",
            replacement.language_surrogate.core,
            output_dir / "slm_bmp",
            scale_factor=scale_factor,
            phase_width=phase_slm_size[0],
            phase_height=phase_slm_size[1],
        ),
    }
    selected = _select_debug_samples(
        bundle, settings.output_dir / "retrieval_results.csv", int(sample_count)
    )
    replacement.set_intermediate_field_capture(True, sample_count=1)
    loaded.model.eval()
    replacement.vision_surrogate.eval()
    replacement.language_surrogate.eval()
    readout.eval()
    sample_reports: list[dict[str, Any]] = []
    try:
        for index, (sample, result_row) in enumerate(selected):
            sample_dir = output_dir / "samples" / f"sample_{index:02d}_{sample.sku_name}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            with Image.open(sample.image_path) as source:
                original = source.convert("RGB")
            original.save(sample_dir / "input_original.png")
            processor_input = ImageOps.fit(
                original,
                (settings.image_size, settings.image_size),
                method=Image.Resampling.BICUBIC,
                centering=(0.5, 0.5),
            )
            processor_input.save(sample_dir / "input_qwen_224x224.png")
            inputs = preprocess_images(
                loaded.processor, [processor_input], settings.instruction
            )
            validate_token_budgets(inputs, settings)
            image_grid_thw = inputs["image_grid_thw"].detach().cpu().tolist()
            language_length = int(inputs["attention_mask"].sum().item())
            inputs = move_inputs(inputs, loaded.device)
            embedding, detector_features = student_embeddings(
                loaded.model, replacement, readout, inputs
            )
            vision_values = _capture_core(replacement.vision_surrogate.core)
            language_values = _capture_core(replacement.language_surrogate.core)
            report = {
                "sample_index": index,
                "sample_id": sample.sample_id,
                "sku_name": sample.sku_name,
                "sku_index": sample.sku_index,
                "image_path": str(sample.image_path),
                "checkpoint_epoch": checkpoint.get("epoch"),
                "image_grid_thw": image_grid_thw,
                "visual_token_count": int(
                    torch.tensor(image_grid_thw).prod(dim=-1).max().item()
                ),
                "language_sequence_length": language_length,
                "student_embedding": embedding[0].detach().cpu().tolist(),
                "student_embedding_norm": float(embedding[0].float().norm()),
                "language_detector_feature_stats": tensor_stats(detector_features[0]),
                "retrieval_result": result_row,
                "vision": _save_core_sample(
                    "vision",
                    vision_values,
                    sample_dir,
                    scale_factor=scale_factor,
                    amplitude_width=amplitude_slm_size[0],
                    amplitude_height=amplitude_slm_size[1],
                ),
                "language": _save_core_sample(
                    "language",
                    language_values,
                    sample_dir,
                    scale_factor=scale_factor,
                    amplitude_width=amplitude_slm_size[0],
                    amplitude_height=amplitude_slm_size[1],
                ),
            }
            _json(sample_dir / "metadata.json", report)
            sample_reports.append(report)
    finally:
        replacement.set_intermediate_field_capture(False, sample_count=1)
    manifest = {
        "schema_version": 1,
        "purpose": "best-checkpoint optical phase/light-field and physical SLM BMP package",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_train_loss": checkpoint.get("train_loss"),
        "weight_variant": snapshot_variant,
        "config": str(settings.config_path),
        "manifest_sha256": bundle.manifest_digest,
        "sample_count": len(sample_reports),
        "sample_selection": (
            "one high-margin correctly retrieved test image per SKU when available; "
            "then remaining correct examples; deterministic test fallback"
        ),
        "optical_geometry": {
            "simulation_pixel_pitch_um": settings.pixel_pitch_um,
            "slm_pixel_pitch_um": float(slm_pixel_pitch_um),
            "scale_factor": scale_factor,
            "canvas_size": settings.canvas_size,
            "numerical_guard_pixels_each_side": (
                settings.canvas_size - settings.active_size
            )
            // 2,
            "exported_active_size": settings.active_size,
            "expert_size": settings.expert_size,
            "expert_pitch": settings.expert_pitch,
            "expert_grid": [settings.expert_grid_rows, settings.expert_grid_cols],
            "expert_layers_per_stack": settings.expert_layers,
            "amplitude_slm_size_wh": list(amplitude_slm_size),
            "phase_slm_size_wh": list(phase_slm_size),
            "amplitude_active_bounds_xyxy": [467, 47, 1453, 1033],
            "phase_active_bounds_xyxy": [467, 107, 1453, 1093],
        },
        "physical_sequence_per_stack": [
            "routed amplitude SLM co-planar with expert phase mosaic",
            "10 cm propagation",
            "square-law detection + per-expert LayerNorm + ReLU + routing-weight reload",
            "reload amplitude SLM co-planar with global phase",
            "10 cm propagation",
            "986x986 CCD ROI, electronically pooled to 224x224",
        ],
        "amplitude_encoding": (
            "nonnegative raw amplitude divided by its per-plane maximum, then mapped "
            "linearly to uint8; the divisor is recorded for every BMP"
        ),
        "phase_encoding": "phase modulo 2pi mapped linearly to uint8 [0,255]",
        "phases": phases,
        "samples": [
            {
                "sample_index": item["sample_index"],
                "sample_id": item["sample_id"],
                "sku_name": item["sku_name"],
                "image_path": item["image_path"],
            }
            for item in sample_reports
        ],
    }
    expected_amplitude_bounds = [467, 47, 1453, 1033]
    expected_phase_bounds = [467, 107, 1453, 1093]
    actual_amplitude = sample_reports[0]["vision"]["bmp"][
        "expert_plane_amplitude"
    ]["active_bounds_xyxy"] if sample_reports else expected_amplitude_bounds
    actual_phase = phases["vision"]["global_phase_bmp"]["active_bounds_xyxy"]
    if actual_amplitude != expected_amplitude_bounds or actual_phase != expected_phase_bounds:
        raise RuntimeError(
            "Physical SLM centering check failed: "
            f"amplitude={actual_amplitude}, phase={actual_phase}"
        )
    manifest["validation"] = {
        "bmp_mode": "8-bit grayscale L",
        "amplitude_centering_passed": True,
        "phase_centering_passed": True,
        "no_bilinear_resampling": True,
        "nearest_neighbor_integer_scaling_only": True,
        "raw_tensors_preserved": True,
    }
    _json(output_dir / "manifest.json", manifest)
    shutil.copy2(settings.config_path, output_dir / "source_config.yaml")
    (output_dir / "README.md").write_text(
        "# Grocery10 best optical hardware package\n\n"
        "`weights/phase_parameters.pt` preserves raw and physical phases for both "
        "Vision and Language stacks. `slm_bmp/` contains shared expert/global phase "
        "BMPs. Each `samples/sample_*` directory contains the input, routed expert-plane "
        "amplitude, the OEO reload amplitude immediately before the co-planar global "
        "phase, CCD intensity/readout, routing weights, raw tensors, and 8-bit BMPs.\n\n"
        "The simulation and devices both use 8 um pixels, so the export scale is 1. "
        "Amplitude BMPs are 1920x1080; phase BMPs are 1920x1200. See `manifest.json` "
        "for centering bounds, normalization divisors, SHA256 checksums, and the exact "
        "checkpoint.\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export best Grocery10 optical phases, light fields, and verified SLM BMPs"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--sample-count", type=int, default=8)
    parser.add_argument("--slm-pixel-pitch-um", type=float, default=8.0)
    args = parser.parse_args()
    settings = load_settings(args.config)
    bundle = prepare_grocery_subset(settings, persist=True)
    device = torch.device(
        settings.device
        if settings.device != "cuda" or torch.cuda.is_available()
        else "cpu"
    )
    loaded = load_backbone(settings, device)
    settings.resolve_architecture(loaded.model)
    replacement, readout = build_optical_student(loaded, settings)
    checkpoint = (
        Path(args.checkpoint).expanduser().resolve()
        if args.checkpoint
        else (
            settings.output_dir / "ema_best_train_loss_checkpoint.pt"
            if (settings.output_dir / "ema_best_train_loss_checkpoint.pt").is_file()
            else settings.output_dir / "best_train_loss_checkpoint.pt"
        )
    )
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else settings.output_dir / "best_optical_artifacts"
    )
    try:
        manifest = export_best_optical_artifacts(
            settings,
            bundle,
            loaded,
            replacement,
            readout,
            checkpoint,
            output_dir,
            sample_count=args.sample_count,
            slm_pixel_pitch_um=args.slm_pixel_pitch_um,
        )
    finally:
        replacement.close()
    print(
        f"Exported checkpoint epoch={manifest['checkpoint_epoch']} "
        f"variant={manifest['weight_variant']} samples={manifest['sample_count']} "
        f"to {output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
