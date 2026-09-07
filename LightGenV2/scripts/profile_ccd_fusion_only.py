"""Benchmark only the neural CCD-readout-to-residual-fusion boundary.

This deliberately excludes router post-processing, ROI/tile arrangement,
next-SLM canvas construction, the parallel electronic residual branch,
inter-modal bridges, task heads, file I/O, and optical propagation.  For tiled
video layouts, already-extracted CCD patches are supplied to the timed region.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import platform
import sys
from pathlib import Path
from typing import Any, Callable

import torch
from torch.nn import functional as F

from LightGenV2.common.baseline_measurement import validate_cuda_device
from LightGenV2.scripts.profile_optical_electronics_5090d import (
    SPECS,
    ScaleMatchedFusion,
    StandardCCDReadout,
    T06FrameReadout,
    T06SpatialLanguageReadout,
    T06SpatialVisionReadout,
    T06VideoReadout,
    _normalize_patch,
    benchmark,
)


STANDARD_TASKS = ("t01", "t02", "t03", "t04", "t08")


def _measure(
    name: str,
    function: Callable[[], torch.Tensor],
    *,
    warmup: int,
    repeats: int,
    logical_samples: int,
    shape: str,
) -> dict[str, Any]:
    return benchmark(
        name,
        function,
        warmup=warmup,
        repeats=repeats,
        logical_samples_per_call=logical_samples,
        physical_fields_per_call=1,
        shape_contract=shape,
    )


def _standard(
    task: str, device: torch.device, warmup: int, repeats: int
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    spec = SPECS[task]
    detector = torch.rand(1, 478, 478, device=device)
    fusion = ScaleMatchedFusion().to(device).eval()
    rows: list[dict[str, Any]] = []
    occurrences: dict[str, int] = {}

    def add(label: str, tokens: int) -> None:
        electronic = torch.randn(1, tokens, 192, device=device)
        mask = torch.ones(1, tokens, dtype=torch.bool, device=device)
        readout = StandardCCDReadout(tokens).to(device).eval()
        rows.append(
            _measure(
                f"{label}_neural_ccd_to_fusion",
                lambda: fusion(electronic, readout(detector), mask),
                warmup=warmup,
                repeats=repeats,
                logical_samples=1,
                shape=(
                    f"CCD [1,478,478] -> normalize/pool/LN/Linear/ReLU -> "
                    f"[1,{tokens},192] -> scale-matched residual fusion"
                ),
            )
        )
        occurrences[rows[-1]["component"]] = 2

    add("vision", spec.vision_tokens)
    if spec.language:
        add("language", spec.language_tokens)
    return rows, occurrences


def _t06(
    device: torch.device, warmup: int, repeats: int
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    # Cropping the 64/16 tiles out of the CCD canvas is intentionally performed
    # before timing.  The timed functions begin with already arranged patches.
    frame_patches = torch.rand(64, 56, 56, device=device)
    video_patches = torch.rand(16, 115, 115, device=device)
    frame_electronic = torch.randn(16, 4, 49, 192, device=device)
    frame_mask = torch.ones(16, 4, 49, dtype=torch.bool, device=device)
    video_electronic = torch.randn(16, 42, 192, device=device)
    video_mask = torch.ones(16, 42, dtype=torch.bool, device=device)
    frame = T06FrameReadout().to(device).eval()
    video = T06VideoReadout().to(device).eval()
    fusion = ScaleMatchedFusion().to(device).eval()

    def frame_neural() -> torch.Tensor:
        normalized = _normalize_patch(frame_patches)
        pooled = frame.pool(normalized.unsqueeze(1)).squeeze(1)
        optical = frame.output(F.softplus(frame.norm(pooled))).reshape(
            16, 4, 49, 192
        )
        return fusion(frame_electronic, optical, frame_mask)

    def video_neural() -> torch.Tensor:
        normalized = _normalize_patch(video_patches)
        pooled = F.adaptive_avg_pool2d(
            normalized.unsqueeze(1), (42, 96)
        ).squeeze(1)
        optical = video.output(F.softplus(video.norm(pooled))).reshape(16, 42, 192)
        return fusion(video_electronic, optical, video_mask)

    rows = [
        _measure(
            "frame_neural_ccd_to_fusion",
            frame_neural,
            warmup=warmup,
            repeats=repeats,
            logical_samples=16,
            shape=(
                "pre-extracted CCD patches [64,56,56] -> normalize/pool/LN/Linear/"
                "softplus -> [16,4,49,192] -> scale-matched residual fusion"
            ),
        ),
        _measure(
            "video_neural_ccd_to_fusion",
            video_neural,
            warmup=warmup,
            repeats=repeats,
            logical_samples=16,
            shape=(
                "pre-extracted CCD patches [16,115,115] -> normalize/pool/LN/Linear/"
                "softplus -> [16,42,192] -> scale-matched residual fusion"
            ),
        ),
    ]
    return rows, {row["component"]: 2 for row in rows}


def _t06_spatial(
    device: torch.device, warmup: int, repeats: int
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    vision_patches = torch.rand(4, 232, 232, device=device)
    language_active = torch.rand(1, 478, 478, device=device)
    vision_electronic = torch.randn(1, 4, 196, 192, device=device)
    vision_mask = torch.ones(1, 4, 196, dtype=torch.bool, device=device)
    language_electronic = torch.randn(1, 42, 192, device=device)
    language_mask = torch.ones(1, 42, dtype=torch.bool, device=device)
    vision = T06SpatialVisionReadout().to(device).eval()
    language = T06SpatialLanguageReadout().to(device).eval()
    fusion = ScaleMatchedFusion().to(device).eval()

    def vision_neural() -> torch.Tensor:
        normalized = _normalize_patch(vision_patches)
        pooled = vision.pool(normalized.unsqueeze(1)).squeeze(1)
        optical = vision.output(F.softplus(vision.norm(pooled))).reshape(
            1, 4, 196, 192
        )
        return fusion(vision_electronic, optical, vision_mask)

    def language_neural() -> torch.Tensor:
        normalized = _normalize_patch(language_active)
        pooled = F.adaptive_avg_pool2d(
            normalized.unsqueeze(1), (42, 96)
        ).squeeze(1)
        optical = language.output(F.softplus(language.norm(pooled)))
        return fusion(language_electronic, optical, language_mask)

    rows = [
        _measure(
            "spatial_vision_neural_ccd_to_fusion",
            vision_neural,
            warmup=warmup,
            repeats=repeats,
            logical_samples=1,
            shape=(
                "pre-extracted CCD lanes [4,232,232] -> normalize/pool/LN/Linear/"
                "softplus -> [1,4,196,192] -> scale-matched residual fusion"
            ),
        ),
        _measure(
            "spatial_language_neural_ccd_to_fusion",
            language_neural,
            warmup=warmup,
            repeats=repeats,
            logical_samples=1,
            shape=(
                "active CCD [1,478,478] -> normalize/pool/LN/Linear/softplus -> "
                "[1,42,192] -> scale-matched residual fusion"
            ),
        ),
    ]
    return rows, {row["component"]: 2 for row in rows}


def _summary(
    rows: list[dict[str, Any]], occurrences: dict[str, int], logical_samples: int
) -> dict[str, Any]:
    by_name = {row["component"]: row for row in rows}
    cuda_total = sum(
        by_name[name]["cuda_event_ms"]["median"] * count
        for name, count in occurrences.items()
    )
    wall_total = sum(
        by_name[name]["synchronized_wall_ms"]["median"] * count
        for name, count in occurrences.items()
    )
    return {
        "component_occurrences_per_call": occurrences,
        "feature_blocks_per_call": sum(occurrences.values()),
        "cuda_event_median_sum_ms_per_call": cuda_total,
        "synchronized_wall_median_sum_ms_per_call": wall_total,
        "cuda_event_median_average_ms_per_feature_block": cuda_total
        / sum(occurrences.values()),
        "synchronized_wall_median_average_ms_per_feature_block": wall_total
        / sum(occurrences.values()),
        "logical_samples_per_call": logical_samples,
        "cuda_event_median_sum_ms_per_logical_sample": cuda_total / logical_samples,
        "synchronized_wall_median_sum_ms_per_logical_sample": wall_total
        / logical_samples,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", choices=sorted(SPECS), default=sorted(SPECS))
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=1000)
    parser.add_argument("--expected-gpu", default="A100")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.warmup < 0 or args.repeats < 20:
        raise ValueError("warmup must be nonnegative and repeats must be at least 20")
    validate_cuda_device(args.expected_gpu)
    device = torch.device("cuda:0")
    torch.manual_seed(20260908)
    torch.cuda.manual_seed_all(20260908)
    tasks: dict[str, Any] = {}
    for task in args.tasks:
        print(f"[fusion-only] {task}: {SPECS[task].label}", flush=True)
        if task in STANDARD_TASKS:
            rows, occurrences = _standard(task, device, args.warmup, args.repeats)
        elif task == "t06":
            rows, occurrences = _t06(device, args.warmup, args.repeats)
        else:
            rows, occurrences = _t06_spatial(device, args.warmup, args.repeats)
        tasks[task] = {
            "label": SPECS[task].label,
            "components": rows,
            "fusion_only_summary": _summary(
                rows, occurrences, SPECS[task].logical_samples_per_call
            ),
        }

    report = {
        "schema_version": 1,
        "timing_boundary": (
            "already acquired CCD intensity through required normalization, pooling, "
            "learned electronic readout/nonlinearity and scale-matched fusion with an "
            "already-computed parallel electronic residual"
        ),
        "excluded": [
            "CCD/SLM driver and exposure",
            "file I/O and image decode",
            "ROI/tile cropping, stacking and canvas layout",
            "router post-processing and expert fan-out",
            "next-SLM field reconstruction or packing",
            "parallel electronic residual computation",
            "inter-modal/frame-to-sequence bridges",
            "final task head or decoder",
            "optical propagation",
        ],
        "measurement_policy": (
            "CUDA-event median is the lower pure-GPU neural estimate; synchronized wall "
            "median is the auditable host-observed value. Minimum samples are not used."
        ),
        "environment": {
            "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "device": torch.cuda.get_device_name(device),
            "precision": "float32",
            "warmup_calls_per_component": args.warmup,
            "measured_calls_per_component": args.repeats,
        },
        "tasks": tasks,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "ccd_fusion_only_a100.json"
    csv_path = args.output_dir / "ccd_fusion_only_a100.csv"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    flat_rows: list[dict[str, Any]] = []
    for task, value in tasks.items():
        summary = value["fusion_only_summary"]
        flat_rows.append({"task": task, "label": value["label"], **summary})
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)
    print(json.dumps({"status": "complete", "json": str(json_path), "csv": str(csv_path)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
