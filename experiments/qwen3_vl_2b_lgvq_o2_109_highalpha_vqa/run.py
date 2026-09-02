from __future__ import annotations

import argparse
import json
import random
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from .data import load_canonical_cache
from .modeling import LGVQSpatiotemporalModel, build_model
from .preflight import run_preflight
from .settings import ExperimentSettings, OpticalGeometry, load_settings, resolved_dict
from .training import _optimizer, evaluate_checkpoint, pairwise_ranking_loss, train


PHASES = {"preflight", "train", "evaluate", "smoke"}


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _seed(value: int) -> None:
    random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def _device(settings: ExperimentSettings) -> torch.device:
    requested = settings.device
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def _synthetic_payload(settings: ExperimentSettings, samples: int = 10) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(settings.random_seed)
    features = torch.randn(
        samples,
        settings.frame_count,
        settings.token_count,
        settings.input_width,
        generator=generator,
    )
    language_length = min(settings.language_token_count, 5)
    language = torch.randn(
        samples,
        language_length,
        settings.language_input_width,
        generator=generator,
    )
    targets = torch.stack(
        (
            50.0 + 4.0 * features.mean((1, 2, 3)),
            50.0 + 4.0 * features[:, 1:].sub(features[:, :-1]).abs().mean((1, 2, 3)),
        ),
        -1,
    )
    splits = ["train"] * (samples - 2) + ["test"] * 2
    return {
        "schema_version": 1,
        "frame_tokens": features,
        "language_tokens": language,
        "language_mask": torch.ones(samples, language_length, dtype=torch.bool),
        "targets": targets,
        "target_names": ["spatial", "temporal"],
        "sample_ids": [f"synthetic_{index:03d}" for index in range(samples)],
        "video_paths": [f"synthetic_{index:03d}.mp4" for index in range(samples)],
        "splits": splits,
        "alignment_target_present": False,
        "qwen_prompt": settings.prompt,
    }


def synthetic_smoke(settings: ExperimentSettings) -> dict[str, Any]:
    """Small CPU test of the only supported formal route: optical Top-2."""

    settings = replace(
        settings,
        geometry=OpticalGeometry(
            canvas_size=96,
            active_size=88,
            quadrant_size=40,
            expert_size=16,
            expert_pitch=24,
        ),
        token_count=16,
        token_grid_height=4,
        token_grid_width=4,
        input_width=32,
        model_width=16,
        detector_projection_size=16,
        language_input_width=64,
        language_token_count=8,
        router_pool_size=4,
        router_detector_intervals=((6, 12), (28, 34)),
        language_router_detector_intervals=((20, 30), (58, 68)),
        router_input_shift_pixels=2,
        router_phase_shift_pixels=2,
        router_ccd_shift_pixels=2,
        router_phase_dropout_block_size=4,
        batch_size=2,
        num_workers=0,
        k_space_enabled=False,
        synthetic=True,
    )
    reports: dict[str, Any] = {}
    payload = _synthetic_payload(settings)
    batch_features = payload["frame_tokens"][:2]
    batch_language = payload["language_tokens"][:2]
    batch_mask = payload["language_mask"][:2]
    target = payload["targets"][:2]
    variant = replace(
        settings,
        router_backend="optical",
        top_k=2,
        initialization_checkpoint=None,
        optional_sister_checkpoint=None,
        output_dir=settings.output_dir / "smoke_optical",
        synthetic=True,
        device="cpu",
    )
    variant.validate()
    model, initialization = build_model(variant)
    optimizer = _optimizer(model, variant)
    model.set_target_statistics(
        payload["targets"][:-2].mean(0),
        payload["targets"][:-2].std(0, unbiased=False).clamp_min(1.0e-6),
    )
    model.train()
    result = model(batch_features, batch_language, batch_mask)
    normalized_target = (target - model.target_mean) / model.target_std
    loss = torch.nn.functional.smooth_l1_loss(
        result["normalized_prediction"], normalized_target
    ) + 0.1 * pairwise_ranking_loss(
        result["normalized_prediction"], normalized_target
    )
    loss.backward()
    bad = [
        name
        for name, parameter in model.named_parameters()
        if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())
    ]
    if bad or not bool(torch.isfinite(loss)):
        raise RuntimeError(f"Synthetic optical smoke has non-finite values: {bad}")
    vision_weights = result["routing"]["vision"]["weights"]
    language_weights = result["routing"]["language"]["weights"]
    torch.testing.assert_close(
        vision_weights.square().sum(-1), torch.ones_like(vision_weights[..., 0])
    )
    torch.testing.assert_close(
        language_weights.square().sum(-1), torch.ones_like(language_weights[..., 0])
    )
    if not bool((result["routing"]["vision"]["selected_mask"].sum(-1) == 2).all()):
        raise RuntimeError("Vision router did not select exactly two experts")
    if not bool((result["routing"]["language"]["selected_mask"].sum(-1) == 2).all()):
        raise RuntimeError("Language router did not select exactly two experts")
    if result["prediction"].shape != (2, 2):
        raise RuntimeError("Model must output exactly spatial and temporal")
    reports["optical"] = {
        "loss": float(loss.detach()),
        "prediction_shape": list(result["prediction"].shape),
        "vision_weight_l2": vision_weights.square().sum(-1).sqrt().tolist(),
        "language_weight_l2": language_weights.square().sum(-1).sqrt().tolist(),
        "vision_router_windows": [list(value) for value in variant.router_detector_intervals],
        "language_router_windows": [
            list(value) for value in variant.language_router_detector_intervals
        ],
        "fusion": model.fusion_diagnostics(),
        "parameters": model.parameter_breakdown(),
        "initialization": initialization,
        "optimizer_groups": {
            str(group["name"]): {
                "lr": float(group["lr"]),
                "parameter_tensors": len(group["params"]),
                "parameters": sum(parameter.numel() for parameter in group["params"]),
            }
            for group in optimizer.param_groups
        },
    }
    return {
        "status": "passed",
        "alignment_output": False,
        "targets": ["spatial", "temporal"],
        "variants": reports,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="LGVQ Qwen Vision+Language balanced optical-router VQA"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--phase", required=True, choices=sorted(PHASES))
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    synthetic = args.phase == "smoke"
    settings = load_settings(args.config, synthetic=synthetic)
    _seed(settings.random_seed)
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    _json(settings.output_dir / "resolved_config.json", resolved_dict(settings))

    if args.phase == "preflight":
        report = run_preflight(settings, require_cache=False)
        _json(settings.output_dir / "preflight.json", report)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0 if report["status"] == "ready" else 2
    if args.phase == "smoke":
        report = synthetic_smoke(settings)
        _json(settings.output_dir / "synthetic_smoke.json", report)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0
    preflight = run_preflight(settings, require_cache=True)
    _json(settings.output_dir / "preflight.json", preflight)
    if preflight["status"] != "ready":
        raise RuntimeError(f"Preflight is blocked: {preflight}")
    assert settings.cache_path is not None
    payload = load_canonical_cache(settings.cache_path)
    model, initialization = build_model(settings)
    _json(settings.output_dir / "initialization_report.json", initialization)
    _json(settings.output_dir / "parameter_breakdown.json", model.parameter_breakdown())
    device = _device(settings)
    if args.phase == "train":
        report = train(model, payload, settings, device, initialization)
    else:
        if args.checkpoint is None:
            parser.error("evaluate requires --checkpoint")
        report = evaluate_checkpoint(
            model,
            payload,
            settings,
            device,
            Path(args.checkpoint).expanduser().resolve(),
        )
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=True))
    return 0


__all__ = ["main", "synthetic_smoke"]


