from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from .datasets import DatasetBundle, make_loader
from .features import move_inputs, preprocess_image_text, run_multimodal_forward
from .io_utils import write_json
from .optics.replacement import TeacherTapCapture, locate_visual
from .pca import (
    FixedPCAProjection,
    StreamingPCAFitter,
    load_projection,
    projection_paths,
    save_projection,
)


@torch.inference_mode()
def fit_shared_pca(
    model: torch.nn.Module,
    processor: Any,
    data: DatasetBundle,
    settings: Any,
    device: torch.device,
) -> dict[str, Any]:
    capture = TeacherTapCapture(model, tuple(settings.language_tap_indexes))
    vision_fitter = StreamingPCAFitter(
        int(settings.vision_hidden_size),
        settings.latent_dim,
        settings.pca_vision_calibration_tokens,
        device,
    )
    language_fitter = StreamingPCAFitter(
        int(settings.text_hidden_size),
        settings.latent_dim,
        settings.pca_language_calibration_tokens,
        device,
    )
    loader = make_loader(
        data.calibration,
        settings.feature_batch_size,
        settings.num_workers,
        True,
        settings.seed,
    )
    try:
        for batch_index, (images, captions, _indices) in enumerate(loader, start=1):
            cpu_inputs = preprocess_image_text(
                processor, images, captions, settings.caption_prompt_template
            )
            inputs = move_inputs(cpu_inputs, device)
            capture.clear()
            run_multimodal_forward(model, inputs)
            capture.validate_complete()
            vision_lengths = capture.vision_lengths()
            language_lengths = cpu_inputs["attention_mask"].sum(1).long().tolist()
            _check_limits(vision_lengths, language_lengths, settings)
            assert capture.vision_input is not None and capture.language_input is not None
            vision_values = torch.cat([
                capture.vision_input,
                *[capture.vision_taps[index] for index in capture.vision_tap_indexes],
            ])
            valid = inputs["attention_mask"].bool()
            language_values = torch.cat([
                capture.language_input[valid],
                *[
                    capture.language_taps[index][valid]
                    for index in capture.language_tap_indexes
                ],
            ])
            vision_fitter.update(vision_values)
            language_fitter.update(language_values)
            if batch_index % settings.log_interval_batches == 0 or (
                vision_fitter.count >= vision_fitter.max_tokens
                and language_fitter.count >= language_fitter.max_tokens
            ):
                print(
                    f"[fit_pca] batch={batch_index}/{len(loader)} "
                    f"vision_tokens={vision_fitter.count}/{vision_fitter.max_tokens} "
                    f"language_tokens={language_fitter.count}/{language_fitter.max_tokens}",
                    flush=True,
                )
            if (
                vision_fitter.count >= vision_fitter.max_tokens
                and language_fitter.count >= language_fitter.max_tokens
            ):
                break
        vision_pca, vision_report = vision_fitter.finalize()
        language_pca, language_report = language_fitter.finalize()
        vision_path, language_path = projection_paths(settings)
        vision_metadata = save_projection(
            vision_path,
            vision_pca,
            vision_report,
            stack="vision",
            seed=settings.seed,
            source_taps=["stack_input", *capture.vision_tap_indexes],
        )
        language_metadata = save_projection(
            language_path,
            language_pca,
            language_report,
            stack="language",
            seed=settings.seed,
            source_taps=["stack_input", *capture.language_tap_indexes],
        )
        report = {
            "vision": vision_metadata,
            "language": language_metadata,
            "shared_coordinate_system_per_stack": True,
            "trainable_pca_parameters": 0,
        }
        write_json(settings.output_dir / "pca" / "fit_report.json", report)
        return report
    finally:
        capture.close()


@torch.inference_mode()
def run_pca_oracle_check(
    model: torch.nn.Module,
    processor: Any,
    data: DatasetBundle,
    settings: Any,
    device: torch.device,
) -> dict[str, Any]:
    vision_path, language_path = projection_paths(settings)
    vision_pca = load_projection(vision_path, settings.vision_hidden_size).to(device)
    language_pca = load_projection(language_path, settings.text_hidden_size).to(device)
    capture = TeacherTapCapture(model, tuple(settings.language_tap_indexes))
    visual = locate_visual(model)
    loader = make_loader(
        data.calibration,
        settings.feature_batch_size,
        0,
        False,
        settings.seed,
    )
    vision_accumulator: dict[str, list[float]] = {}
    language_accumulator: dict[str, list[float]] = {}
    seen = 0
    try:
        for images, captions, _indices in loader:
            cpu_inputs = preprocess_image_text(
                processor, images, captions, settings.caption_prompt_template
            )
            inputs = move_inputs(cpu_inputs, device)
            capture.clear()
            run_multimodal_forward(model, inputs)
            capture.validate_complete()
            valid = inputs["attention_mask"].bool()
            vision_values = [
                ("stack_input", capture.vision_input),
                *[
                    (f"tap_{index}", capture.vision_taps[index])
                    for index in capture.vision_tap_indexes
                ],
            ]
            for slot, (name, hidden) in enumerate(vision_values):
                assert hidden is not None
                reconstructed = vision_pca.decode(vision_pca.encode(hidden))
                metrics = _reconstruction_metrics(hidden, reconstructed)
                _append_metrics(vision_accumulator, name, metrics)
                if slot > 0:
                    merger = (
                        visual.deepstack_merger_list[slot - 1]
                        if slot <= len(capture.deepstack_indexes)
                        else visual.merger
                    )
                    source_out = merger(hidden)
                    reconstructed_out = merger(reconstructed.to(hidden.dtype))
                    _append_metrics(
                        vision_accumulator,
                        f"{name}_frozen_merger",
                        _pair_metrics(source_out, reconstructed_out),
                    )
            language_values = [
                ("stack_input", capture.language_input),
                *[
                    (f"tap_{index}", capture.language_taps[index])
                    for index in capture.language_tap_indexes
                ],
            ]
            for name, hidden in language_values:
                assert hidden is not None
                hidden_valid = hidden[valid]
                reconstructed = language_pca.decode(language_pca.encode(hidden_valid))
                _append_metrics(
                    language_accumulator,
                    name,
                    _reconstruction_metrics(hidden_valid, reconstructed),
                )
                source_norm = capture.language.norm(hidden_valid)
                reconstructed_norm = capture.language.norm(reconstructed.to(hidden.dtype))
                _append_metrics(
                    language_accumulator,
                    f"{name}_frozen_final_rmsnorm",
                    _pair_metrics(source_norm, reconstructed_norm),
                )
            seen += len(images)
            if seen >= settings.pca_oracle_samples:
                break
        report = {
            "samples": seen,
            "vision": _mean_metrics(vision_accumulator),
            "language": _mean_metrics(language_accumulator),
            "warning_threshold_hidden_cosine": settings.pca_oracle_warn_cosine_below,
            "adapter_fallback_added": False,
        }
        minimum_cosine = min(
            [
                values.get("normalized_hidden_cosine", 1.0)
                for stack in (report["vision"], report["language"])
                for values in stack.values()
            ],
            default=1.0,
        )
        report["minimum_normalized_hidden_cosine"] = minimum_cosine
        if minimum_cosine < settings.pca_oracle_warn_cosine_below:
            message = (
                f"PCA oracle cosine {minimum_cosine:.4f} is below configured warning "
                f"threshold {settings.pca_oracle_warn_cosine_below:.4f}. No trainable "
                "adapter will be added; inspect the oracle report before training."
            )
            warnings.warn(message, RuntimeWarning)
            report["warning"] = message
        write_json(settings.output_dir / "pca" / "oracle_report.json", report)
        return report
    finally:
        capture.close()


def _check_limits(vision_lengths: list[int], language_lengths: list[int], settings: Any) -> None:
    if max(vision_lengths) > settings.max_visual_tokens:
        raise RuntimeError(
            f"visual token count {max(vision_lengths)} exceeds max_visual_tokens="
            f"{settings.max_visual_tokens}. Lower processor_max_pixels; silent truncation is forbidden."
        )
    if max(language_lengths) > settings.max_language_tokens:
        raise RuntimeError(
            f"language sequence length {max(language_lengths)} exceeds max_language_tokens="
            f"{settings.max_language_tokens}. Shorten captions/prompt; silent truncation is forbidden."
        )


def _reconstruction_metrics(source: torch.Tensor, reconstructed: torch.Tensor) -> dict[str, float]:
    normalized = F.layer_norm(source.float(), (source.shape[-1],))
    return {
        "raw_hidden_cosine": float(
            F.cosine_similarity(source.float(), reconstructed.float(), dim=-1).mean()
        ),
        "normalized_hidden_cosine": float(
            F.cosine_similarity(normalized, reconstructed.float(), dim=-1).mean()
        ),
        "normalized_mse": float(F.mse_loss(reconstructed.float(), normalized)),
        "relative_reconstruction_error": float(
            torch.linalg.vector_norm(reconstructed.float() - normalized)
            / torch.linalg.vector_norm(normalized).clamp_min(1e-12)
        ),
    }


def _pair_metrics(source: torch.Tensor, reconstructed: torch.Tensor) -> dict[str, float]:
    source = source.float()
    reconstructed = reconstructed.float()
    return {
        "cosine": float(F.cosine_similarity(source, reconstructed, dim=-1).mean()),
        "mse": float(F.mse_loss(reconstructed, source)),
    }


def _append_metrics(
    accumulator: dict[str, list[float]],
    prefix: str,
    metrics: dict[str, float],
) -> None:
    for key, value in metrics.items():
        accumulator.setdefault(f"{prefix}.{key}", []).append(value)


def _mean_metrics(accumulator: dict[str, list[float]]) -> dict[str, dict[str, float]]:
    nested: dict[str, dict[str, float]] = {}
    for combined, values in accumulator.items():
        prefix, key = combined.split(".", 1)
        nested.setdefault(prefix, {})[key] = sum(values) / len(values)
    return nested
