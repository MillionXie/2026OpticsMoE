from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import torch
from PIL import Image
from torch.utils.data import DataLoader

from ..datasets import CIFAR10_CLASSES, IndexedDataset, indexed_collate, load_cifar10
from ..features import move_inputs, pool_token_groups, preprocess_images, run_visual
from ..io_utils import resolve_device, resolve_dtype, set_seed, write_json
from ..modeling import build_head, load_backbone
from ..optics import VisionHomogeneousMoESurrogate, VisionStackReplacement
from ..settings import load_settings
from ..training import load_student_parts
from .bmp import encode_amplitude_uint8, encode_phase_uint8, export_plane_bmp
from .config import HardwareSettings, load_hardware_settings, to_jsonable


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export final-OEO amplitude and co-planar global phase BMPs for Qwen vision MoE hardware validation"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device")
    parser.add_argument("--output-dir", type=Path)
    return parser


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _save_unit_preview(value: torch.Tensor, path: Path) -> None:
    image = value.detach().cpu().float()
    while image.ndim > 2:
        image = image[0]
    peak = float(image.max())
    if peak > 0:
        image = image / peak
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(torch.round(image.clamp(0, 1) * 255).to(torch.uint8).numpy(), mode="L").save(path)


def _save_input_image(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(path)


def _load_student(settings: HardwareSettings):
    source = load_settings(settings.source_config)
    if settings.cache_dir is not None:
        source.cache_dir = settings.cache_dir
    if settings.local_files_only is not None:
        source.local_files_only = settings.local_files_only
    model_id = source.model_id
    if source.local_files_only and not Path(model_id).is_dir():
        # Some transformers tokenizer versions call model_info() even with
        # local_files_only=True when given a repository id.  Resolve the
        # already-cached snapshot first so every downstream loader sees a
        # genuine local path and cannot make a network request.
        from huggingface_hub import snapshot_download

        model_id = snapshot_download(
            repo_id=model_id,
            cache_dir=str(source.cache_dir) if source.cache_dir else None,
            local_files_only=True,
        )
    device = resolve_device(settings.device)
    loaded = load_backbone(
        model_id, source.cache_dir, source.local_files_only, resolve_dtype(source.dtype), device,
        source.attn_implementation, source.processor_min_pixels, source.processor_max_pixels,
    )
    source.resolve_architecture(loaded.model)
    surrogate = VisionHomogeneousMoESurrogate(source.vision_hidden_size, source).to(device)
    replacement = VisionStackReplacement(loaded.model, surrogate)
    head = build_head(source, source.vision_hidden_size, len(CIFAR10_CLASSES)).to(device)
    load_student_parts(settings.source_run_dir, replacement, head, settings.checkpoint_tag)
    replacement.use_student()
    replacement.surrogate.requires_grad_(False).eval()
    replacement.surrogate.set_phase_dropout_active(False)
    head.requires_grad_(False).eval()
    return source, loaded, replacement, head, device


@torch.inference_mode()
def _forward_student(model: torch.nn.Module, processor: Any, replacement: VisionStackReplacement,
                     head: torch.nn.Module, images: Sequence[Image.Image], device: torch.device) -> torch.Tensor:
    run_visual(model, move_inputs(preprocess_images(processor, images), device))
    groups = list(replacement.surrogate.last_output.split(replacement.surrogate.last_token_counts, dim=0))
    return head(pool_token_groups(groups))


@torch.inference_mode()
def _rank_test_samples(model: torch.nn.Module, processor: Any, replacement: VisionStackReplacement,
                       head: torch.nn.Module, dataset: Any, settings: HardwareSettings,
                       device: torch.device) -> tuple[dict[int, list[dict[str, Any]]], list[int]]:
    loader = DataLoader(
        IndexedDataset(dataset), batch_size=settings.selection_batch_size, shuffle=False,
        num_workers=settings.num_workers, collate_fn=indexed_collate,
        pin_memory=device.type == "cuda", persistent_workers=settings.num_workers > 0,
    )
    correct: dict[int, list[dict[str, Any]]] = defaultdict(list)
    all_indices: list[int] = []
    for batch_index, (images, labels, indices) in enumerate(loader, start=1):
        logits = _forward_student(model, processor, replacement, head, images, device).detach().float().cpu()
        predictions = logits.argmax(1)
        for label, index, prediction, values in zip(labels.tolist(), indices.tolist(), predictions.tolist(), logits):
            all_indices.append(int(index))
            if prediction != label:
                continue
            alternatives = torch.cat((values[:label], values[label + 1:]))
            correct[label].append({
                "sample_index": int(index), "margin": float(values[label] - alternatives.max()),
                "confidence": float(values.softmax(0)[label]), "float_logits": values.tolist(),
            })
        if batch_index % 250 == 0 or batch_index == len(loader):
            print(f"[selection] batch={batch_index}/{len(loader)} samples={len(all_indices)}/{len(dataset)}", flush=True)
    for label in correct:
        correct[label].sort(key=lambda row: (row["margin"], row["confidence"]), reverse=True)
    return correct, all_indices


def _active_amplitude(replacement: VisionStackReplacement) -> tuple[torch.Tensor, float]:
    stages = replacement.surrogate.last_stage_fields
    if len(stages) != replacement.surrogate.expert_layers.__len__():
        raise RuntimeError("Debug capture did not retain every expert stage")
    # This is the fifth-stage square-detect -> LN -> activation -> routed hard-gate
    # -> zero-phase reload.  global_phase(field) is the very next model operation.
    field = stages[-1]["after_oeo"][0].to(torch.complex64)
    aperture = replacement.surrogate.geometry.active_aperture
    amplitude = field.abs()[aperture.y0:aperture.y1, aperture.x0:aperture.x1].float()
    peak = float(amplitude.max())
    if not math.isfinite(peak) or peak <= 0:
        raise RuntimeError(f"Invalid final-OEO amplitude peak: {peak}")
    return amplitude / peak, peak


@torch.inference_mode()
def _quantized_replay(replacement: VisionStackReplacement, head: torch.nn.Module,
                      amplitude: torch.Tensor, phase: torch.Tensor, token_count: int,
                      device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    amplitude_q = encode_amplitude_uint8(amplitude).float().to(device) / 255.0
    phase_q = encode_phase_uint8(phase).float().to(device) * ((2.0 * math.pi) / 255.0)
    canvas = torch.zeros(1, 480, 480, dtype=torch.complex64, device=device)
    aperture = replacement.surrogate.geometry.active_aperture
    canvas[:, aperture.y0:aperture.y1, aperture.x0:aperture.x1] = (
        amplitude_q * torch.exp(1j * phase_q)
    ).to(torch.complex64)
    detector_field = replacement.surrogate.to_detector(canvas)
    readout, intensity = replacement.surrogate.detector_readout(detector_field)
    restored = replacement.surrogate.output_adapter(readout[0, :token_count, :])
    logits = head(restored.float().mean(0, keepdim=True))[0]
    return logits.detach().cpu().float(), intensity[0].detach().cpu().float()


def _copy_student_package(settings: HardwareSettings) -> list[dict[str, Any]]:
    destination = settings.output_dir / "student_package"
    destination.mkdir(parents=True, exist_ok=True)
    names = [
        f"vision_homogeneous_moe_{settings.checkpoint_tag}.pt",
        f"student_head_{settings.checkpoint_tag}.pt",
    ]
    copied: list[dict[str, Any]] = []
    if settings.copy_student_checkpoints:
        for name in names:
            source = settings.source_run_dir / "checkpoints" / name
            if not source.is_file():
                raise FileNotFoundError(f"Student checkpoint is missing: {source}")
            target = destination / "checkpoints" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            copied.append({"path": str(target.relative_to(settings.output_dir)), "sha256": _sha256(target)})
    shutil.copy2(settings.source_config, destination / "source_config_resolved.json")
    for name in ("best_test.json", "student_inference.json", "model.json"):
        source = settings.source_run_dir / "metrics" / name
        if not source.is_file():
            source = settings.source_run_dir / name
        if source.is_file():
            target = destination / name
            shutil.copy2(source, target)
    write_json(destination / "hardware_export_config_resolved.json", to_jsonable(settings))
    return copied


@torch.inference_mode()
def _export_sample(group: str, sequence: int, sample_index: int, image: Image.Image, label: int,
                   source_logits: list[float] | None, source_margin: float | None,
                   loaded: Any, replacement: VisionStackReplacement, head: torch.nn.Module,
                   phase: torch.Tensor, settings: HardwareSettings, device: torch.device) -> dict[str, Any]:
    replacement.surrogate.set_debug_capture(True)
    try:
        logits = _forward_student(loaded.model, loaded.processor, replacement, head, [image], device)[0].detach().float().cpu()
        token_count = replacement.surrogate.last_token_counts[0]
        amplitude, amplitude_peak = _active_amplitude(replacement)
        replay_logits, replay_intensity = _quantized_replay(replacement, head, amplitude, phase, token_count, device)
        routing = replacement.surrogate.last_routing
        selected = routing["selected_indices"][0].detach().cpu().tolist()
        weights = routing["weights"][0].detach().cpu().tolist()
        detector_intensity = replacement.surrogate.last_detector_intensity[0].detach().cpu().float()
    finally:
        replacement.surrogate.set_debug_capture(False)

    class_name = CIFAR10_CLASSES[label]
    stem = f"{sequence:04d}_{group}_idx{sample_index:05d}_true-{label:02d}-{class_name}"
    directory = settings.output_dir / "samples" / group / f"class_{label:02d}_{class_name}" / stem
    directory.mkdir(parents=True, exist_ok=True)
    amplitude_path = directory / f"{stem}_amplitude_final_oeo_coplanar_global_1920x1080.bmp"
    amplitude_export = export_plane_bmp(
        amplitude, amplitude_path, "amplitude", 2, settings.amplitude_slm_width, settings.amplitude_slm_height,
    )
    _save_input_image(image, directory / f"{stem}_input_rgb.png")
    _save_unit_preview(amplitude, directory / f"{stem}_amplitude_active450_preview.png")
    _save_unit_preview(detector_intensity, directory / f"{stem}_simulated_detector_intensity480.png")
    _save_unit_preview(replay_intensity, directory / f"{stem}_quantized_bmp_replay_detector480.png")
    if settings.save_raw_tensors:
        torch.save(amplitude, directory / f"{stem}_amplitude_active450_unit_peak.pt")
        torch.save({"peak": amplitude_peak, "amplitude": amplitude * amplitude_peak},
                   directory / f"{stem}_amplitude_active450_raw.pt")
        torch.save(detector_intensity, directory / f"{stem}_simulated_detector_intensity480.pt")
    prediction, replay_prediction = int(logits.argmax()), int(replay_logits.argmax())
    row: dict[str, Any] = {
        "sequence": sequence, "selection_group": group, "sample_index": sample_index,
        "true_label": label, "true_name": class_name,
        "student_pred_label": prediction, "student_pred_name": CIFAR10_CLASSES[prediction],
        "student_correct": prediction == label,
        "quantized_replay_pred_label": replay_prediction,
        "quantized_replay_pred_name": CIFAR10_CLASSES[replay_prediction],
        "quantized_replay_correct": replay_prediction == label,
        "visual_token_count": token_count, "amplitude_peak_before_normalization": amplitude_peak,
        "source_selection_margin": source_margin,
        "routing_selected_experts": json.dumps(selected), "routing_weights": json.dumps(weights),
        "student_logits": json.dumps(logits.tolist()), "quantized_replay_logits": json.dumps(replay_logits.tolist()),
        "amplitude_bmp": str(amplitude_path.relative_to(settings.output_dir)),
        "input_image": str((directory / f"{stem}_input_rgb.png").relative_to(settings.output_dir)),
        "sample_dir": str(directory.relative_to(settings.output_dir)),
    }
    write_json(directory / "metadata.json", {
        **row, "source_rank_logits": source_logits, "amplitude_export": amplitude_export,
        "plane_contract": "final expert stage OEO reload amplitude; immediately multiplied by global phase; no free-space gap",
    })
    return row


def export_package(settings: HardwareSettings) -> dict[str, Any]:
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(settings.seed)
    source, loaded, replacement, head, device = _load_student(settings)
    data = load_cifar10(source)
    copied = _copy_student_package(settings)
    phase = replacement.surrogate.global_phase.phase.phase().detach().cpu().float()
    if tuple(phase.shape) != (450, 450):
        raise RuntimeError(f"Expected active global phase [450,450], got {tuple(phase.shape)}")
    phase_path = settings.output_dir / "00_shared_global_phase_active450_scale2_1920x1200.bmp"
    phase_export = export_plane_bmp(phase, phase_path, "phase", 2, settings.phase_slm_width, settings.phase_slm_height)
    if settings.save_raw_tensors:
        torch.save(phase, settings.output_dir / "00_shared_global_phase_active450_raw.pt")

    try:
        ranked, all_indices = _rank_test_samples(
            loaded.model, loaded.processor, replacement, head, data.test, settings, device,
        )
        rows: list[dict[str, Any]] = []
        selected_correct: set[int] = set()
        sequence = 0
        required_candidates = settings.correct_samples_per_class * settings.correct_candidate_multiplier
        for label in range(len(CIFAR10_CLASSES)):
            accepted = 0
            for candidate in ranked.get(label, [])[:required_candidates]:
                image, truth = data.test[candidate["sample_index"]]
                row = _export_sample(
                    "correct_high_confidence", sequence, candidate["sample_index"], image, int(truth),
                    candidate["float_logits"], candidate["margin"], loaded, replacement, head, phase, settings, device,
                )
                if not row["quantized_replay_correct"]:
                    shutil.rmtree(settings.output_dir / row["sample_dir"])
                    continue
                rows.append(row)
                selected_correct.add(candidate["sample_index"])
                sequence += 1
                accepted += 1
                if accepted >= settings.correct_samples_per_class:
                    break
            if accepted < settings.correct_samples_per_class:
                print(f"WARNING: class {label} exported {accepted}/{settings.correct_samples_per_class} correct samples", flush=True)

        random_pool = [index for index in all_indices if not settings.random_exclude_selected_correct or index not in selected_correct]
        random.Random(settings.seed + 991).shuffle(random_pool)
        for sample_index in random_pool[:settings.random_test_samples]:
            image, truth = data.test[sample_index]
            rows.append(_export_sample(
                "random_test", sequence, sample_index, image, int(truth), None, None,
                loaded, replacement, head, phase, settings, device,
            ))
            sequence += 1
        _write_csv(settings.output_dir / "manifest.csv", rows)
        ccd_rows = [{
            "sample_index": row["sample_index"], "true_label": row["true_label"], "true_name": row["true_name"],
            "visual_token_count": row["visual_token_count"], "selection_group": row["selection_group"],
            "split": "test", "ccd_path": "", "exposure_ms": "",
        } for row in rows]
        _write_csv(settings.output_dir / "ccd_capture_manifest_template.csv", ccd_rows)
        counts = {
            group: sum(row["selection_group"] == group for row in rows)
            for group in ("correct_high_confidence", "random_test")
        }
        manifest = {
            "experiment": "Qwen3-VL-2B CIFAR-10 vision homogeneous MoE final-stage hardware validation",
            "source_run_dir": str(settings.source_run_dir), "source_config": str(settings.source_config),
            "checkpoint_tag": settings.checkpoint_tag, "copied_checkpoints": copied,
            "source_best_test_top1": _read_metric(settings.source_run_dir / "metrics" / "best_test.json", "test_top1_accuracy"),
            "source_best_test_epoch": _read_metric(settings.source_run_dir / "metrics" / "best_test.json", "epoch"),
            "plane_sequence": [
                "expert layer 5 phase modulation", "20 cm propagation to global plane",
                "square-law detection", "per-selected-expert non-affine LayerNorm", source.interlayer_nonlinearity,
                "routing weight reapplication and unselected-expert hard zero",
                "zero-phase amplitude reload", "co-planar global phase modulation",
                "20 cm propagation", "480x480 square-law CCD detector",
            ],
            "coplanar_verified": True,
            "coplanar_evidence": "VisionHomogeneousMoESurrogate.forward calls to_detector(global_phase(field)) immediately after final OEO field assignment.",
            "last_expert_to_global_distance_m": source.last_expert_to_global_distance_m,
            "global_phase_to_detector_distance_m": source.global_to_detector_distance_m,
            "source_simulation": {"active_shape_hw": [450, 450], "pixel_pitch_um": source.pixel_pitch_um, "wavelength_nm": source.wavelength_nm},
            "amplitude_device": {"size_wh": [settings.amplitude_slm_width, settings.amplitude_slm_height], "pixel_pitch_um": settings.hardware_pixel_pitch_um},
            "phase_device": {"size_wh": [settings.phase_slm_width, settings.phase_slm_height], "pixel_pitch_um": settings.hardware_pixel_pitch_um},
            "global_phase_export": phase_export, "sample_counts": counts, "exported_total": len(rows),
            "amplitude_encoding": "Per-sample peak normalization, uint8 round(255*A), nearest-neighbor 2x from 16um to 8um, centered zero padding.",
            "ccd_note": "CCD frames are detector intensity, not amplitude. Use ccd_readout.py to run the identical AvgPool/LN/readout adapter/head electronic tail.",
        }
        write_json(settings.output_dir / "manifest.json", manifest)
        print(f"exported={len(rows)} groups={counts} output={settings.output_dir}", flush=True)
        return manifest
    finally:
        replacement.close()


def _read_metric(path: Path, key: str) -> Any:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8")).get(key)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = load_hardware_settings(args.config)
    if args.device:
        settings.device = args.device
    if args.output_dir:
        settings.output_dir = args.output_dir.resolve()
    settings.validate()
    export_package(settings)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
