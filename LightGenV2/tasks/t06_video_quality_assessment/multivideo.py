"""CLI for the LightGenV2 T06 true full-field 9-video x 4-frame graph."""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import torch

from experiments.qwen3_vl_2b_lgvq_single_metric_o2_16frame_54.data import load_single_metric_cache

from .models.multivideo9x4 import build_model
from .multivideo_settings import MultiVideoGeometry, load_settings, resolved_dict
from .multivideo_training import compatible_warm_start, evaluate, train


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=True) + "\n", encoding="utf-8")


def _seed(value: int) -> None:
    random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def _git_commit() -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], text=True, capture_output=True
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def synthetic_smoke(settings):
    geometry = MultiVideoGeometry(
        canvas_size=96,
        active_size=88,
        video_grid=3,
        video_count=9,
        video_tile_size=28,
        video_tile_pitch=29,
        video_tile_offset=1,
        frame_grid=2,
        frames_per_video=4,
        frame_lane_size=14,
        frame_lane_pitch=14,
        frame_expert_size=6,
        frame_expert_pitch=8,
        video_field_size=12,
        video_expert_pitch=16,
        video_phase_tile_size=26,
    )
    small = replace(
        settings,
        geometry=geometry,
        vision_input_width=32,
        quality_input_width=6,
        language_input_width=40,
        model_width=24,
        detector_projection_size=8,
        head_width=32,
        maximum_language_tokens=12,
        frame_router_intervals=((2, 6), (8, 12)),
        video_router_intervals=((4, 10), (18, 24)),
        input_shift_pixels=1,
        phase_shift_pixels=1,
        ccd_shift_pixels=1,
        phase_dropout_cell_size=2,
        k_space_enabled=False,
        initialization_checkpoint=None,
        batch_size=1,
        num_workers=0,
        synthetic=True,
        device="cpu",
    )
    small.validate()
    model = build_model(small).train()
    vision = torch.randn(1, 9, 4, 49, 32)
    quality = torch.randn(1, 9, 4, 49, 6)
    language = torch.randn(1, 8, 40)
    mask = torch.ones(1, 8, dtype=torch.bool)
    result = model(vision, quality, language, mask, optical_enabled=True)
    loss = result["normalized_prediction"].square().mean() + result["router_balance_loss"]
    loss.backward()
    phase_gradients = {
        name: parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        for name, parameter in model.named_parameters()
        if "phase" in name
    }
    if result["prediction"].shape != (1, 9) or not all(phase_gradients.values()):
        raise RuntimeError("9x4 smoke contract failed")
    return {
        "status": "passed",
        "prediction_shape": list(result["prediction"].shape),
        "frame_router_shape": list(result["routing"]["frame"]["weights"].shape),
        "video_router_shape": list(result["routing"]["video"]["weights"].shape),
        "all_phase_gradients_finite": True,
        "parameters": model.parameter_breakdown(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--phase", required=True, choices=("smoke", "preflight", "train", "evaluate"))
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()
    settings = load_settings(args.config, synthetic=args.phase == "smoke")
    _seed(settings.random_seed)
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    _json(settings.output_dir / "resolved_config.json", resolved_dict(settings))
    _json(
        settings.output_dir / "run_manifest.json",
        {
            "schema_version": 1,
            "architecture": settings.architecture_label,
            "phase": args.phase,
            "git_commit": _git_commit(),
            "command": sys.argv,
            "python": sys.version,
            "torch": torch.__version__,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cuda_available": torch.cuda.is_available(),
            "frame_semantics": "nine unrelated videos x four frames",
            "output_contract": "[physical_batch,9], one Temporal MOS per video",
        },
    )
    (settings.output_dir / "command.txt").write_text(
        subprocess.list2cmdline([sys.executable, "-m", __package__ + ".multivideo", *sys.argv[1:]]) + "\n",
        encoding="utf-8",
    )
    if args.phase == "smoke":
        report = synthetic_smoke(settings)
        _json(settings.output_dir / "synthetic_smoke.json", report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    required = [settings.manifest_path, settings.vision_cache_path, settings.language_cache_path]
    missing = [str(path) for path in required if not path.is_file()]
    report = {"status": "ready" if not missing else "blocked", "missing": missing, "settings": resolved_dict(settings)}
    _json(settings.output_dir / "preflight.json", report)
    if args.phase == "preflight":
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if not missing else 2
    if missing:
        raise FileNotFoundError(f"Preflight missing inputs: {missing}")
    payload = load_single_metric_cache(settings)
    model = build_model(settings)
    initialization = (
        compatible_warm_start(model, settings)
        if args.phase == "train"
        else {
            "used": False,
            "reason": "evaluation loads the requested checkpoint strictly; no training initialization is required",
        }
    )
    _json(settings.output_dir / "initialization_report.json", initialization)
    _json(settings.output_dir / "parameter_breakdown.json", model.parameter_breakdown())
    device = torch.device(settings.device)
    if args.phase == "train":
        result = train(model, payload, settings, device)
    else:
        if args.checkpoint is None:
            parser.error("--phase evaluate requires --checkpoint")
        saved = torch.load(Path(args.checkpoint), map_location="cpu", weights_only=False)
        model.load_state_dict(saved["state_dict"], strict=True)
        model.set_target_statistics(model.target_mean, model.target_std)
        model.to(device)
        result = evaluate(model, payload, settings, device, optical_enabled=True)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
