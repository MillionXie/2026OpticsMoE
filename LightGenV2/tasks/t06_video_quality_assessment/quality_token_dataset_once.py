from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from LightGenV2.common.baseline_measurement import (
    NvidiaSmiPowerSampler,
    gpu_power_limit_w,
    power_report,
    save_power_samples,
    validate_cuda_device,
)
from . import quality_token_common as core


SCHEME1 = "scheme1_scalar_linear"
SCHEME2 = "scheme2_five_quality_tokens"


class ScalarHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2048, 1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.linear(value).squeeze(-1)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "std": float(array.std()),
        "p05": float(np.percentile(array, 5)),
        "p95": float(np.percentile(array, 95)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


@torch.inference_mode()
def forward_once(
    *,
    inputs: dict[str, torch.Tensor],
    model: nn.Module,
    timer: core.BoundaryTimer,
    scheme: str,
    scalar_head: ScalarHead | None,
    target_mean: torch.Tensor | None,
    target_std: torch.Tensor | None,
    quality_head: core.FiveNativeTokenRows | None,
    quality_scores: torch.Tensor | None,
) -> tuple[float, float, dict[str, Any]]:
    torch.cuda.synchronize()
    timer.reset(expected_calls=1)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        hidden = model.model(**inputs, return_dict=True, use_cache=False).last_hidden_state
    mask = inputs["attention_mask"].bool()
    indices = torch.arange(mask.shape[1], device=hidden.device).expand_as(mask)
    last = indices.masked_fill(~mask, -1).amax(1)
    pooled = hidden[torch.arange(hidden.shape[0], device=hidden.device), last].float()
    if scheme == SCHEME1:
        assert scalar_head is not None and target_mean is not None and target_std is not None
        score = scalar_head(pooled) * target_std + target_mean
    elif scheme == SCHEME2:
        assert quality_head is not None and quality_scores is not None
        # Scheme 2 retains the complete native vocabulary projection.  The five
        # trainable quality-token rows provide the selected quality logits.
        _native_full_logits = model.lm_head(pooled.to(model.lm_head.weight.dtype))
        score = quality_head(pooled).softmax(-1) @ quality_scores
    else:
        raise ValueError(scheme)
    timing = timer.finish()
    return float(timing["vision_first_block_to_score_cuda_ms"]), float(score.item()), timing


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, required=True, choices=(4, 9, 16))
    parser.add_argument("--target", choices=sorted(core.PROMPTS), default="temporal")
    parser.add_argument("--scheme", required=True, choices=(SCHEME1, SCHEME2))
    parser.add_argument("--image-size", type=int, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-gpu", default="NVIDIA GeForce RTX 5090 D")
    args = parser.parse_args()
    gpu_name = validate_cuda_device(args.expected_gpu)
    rated_power_w = gpu_power_limit_w()
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    device = torch.device("cuda:0")
    if args.image_size < 224 or args.image_size % 32:
        raise ValueError("Qwen3-VL image-size must be at least 224 and divisible by 32")
    model_path = args.model.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    rows = core.read_manifest(manifest_path)
    test_rows = [row for row in rows if row["split"] == "test"]
    if len(test_rows) != 558:
        raise RuntimeError(f"Expected 558 test videos, got {len(test_rows)}")

    processor_started = time.perf_counter()
    processor = AutoProcessor.from_pretrained(
        str(model_path),
        min_pixels=args.image_size * args.image_size,
        max_pixels=args.image_size * args.image_size,
        local_files_only=True,
        trust_remote_code=True,
    )
    processor_load_seconds = time.perf_counter() - processor_started
    model_started = time.perf_counter()
    model = (
        Qwen3VLForConditionalGeneration.from_pretrained(
            str(model_path),
            local_files_only=True,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        )
        .to(device)
        .eval()
        .requires_grad_(False)
    )
    torch.cuda.synchronize()
    model_load_seconds = time.perf_counter() - model_started

    scalar_head: ScalarHead | None = None
    target_mean: torch.Tensor | None = None
    target_std: torch.Tensor | None = None
    quality_head: core.FiveNativeTokenRows | None = None
    quality_scores: torch.Tensor | None = None
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if args.scheme == SCHEME1:
        scalar_head = ScalarHead().to(device).eval().requires_grad_(False)
        scalar_head.load_state_dict(payload["state_dict"], strict=True)
        target_mean = payload["target_mean"][1].to(device)
        target_std = payload["target_std"][1].to(device)
    else:
        if payload.get("architecture") != "frozen_qwen3vl_plus_five_untied_native_lm_output_rows":
            raise RuntimeError("Checkpoint is not the five-quality-token baseline")
        feature_identity = payload.get("feature_identity", {})
        if feature_identity.get("image_size_wh") != [args.image_size, args.image_size]:
            raise RuntimeError(
                "Checkpoint resolution does not match benchmark resolution: "
                f"{feature_identity.get('image_size_wh')} vs {[args.image_size, args.image_size]}"
            )
        quality_head = core.FiveNativeTokenRows(payload["state_dict"]["weight"]).to(device).eval()
        quality_scores = payload["level_scores"].float().to(device)

    prompt = core.render_prompt(processor, args.target)
    timer = core.BoundaryTimer(model)
    sampler = NvidiaSmiPowerSampler(gpu_index=0, interval_ms=50)
    sampler.start()
    sampler.set_phase("idle")
    time.sleep(2.0)
    sampler.set_phase(None)
    fractions = core.FRAME_FRACTIONS[args.frames]
    records: list[dict[str, Any]] = []
    loop_started = time.perf_counter()
    for index, row in enumerate(test_rows):
        preprocessing_started = time.perf_counter()
        inputs, positions = core.prepare_inputs(
            row, fractions, processor, prompt, device, args.image_size
        )
        torch.cuda.synchronize()
        preprocessing_ms = 1000.0 * (time.perf_counter() - preprocessing_started)
        sampler.set_phase(f"active:{index}")
        try:
            latency, prediction, timing = forward_once(
                inputs=inputs,
                model=model,
                timer=timer,
                scheme=args.scheme,
                scalar_head=scalar_head,
                target_mean=target_mean,
                target_std=target_std,
                quality_head=quality_head,
                quality_scores=quality_scores,
            )
        finally:
            sampler.set_phase(None)
        records.append(
            {
                "sample_index": index,
                "sample_id": row["sample_id"],
                "target_mos": float(row[args.target]),
                "prediction": prediction,
                "model_internal_cuda_ms": latency,
                "preprocessing_ms": preprocessing_ms,
                "sequence_length": int(inputs["input_ids"].shape[1]),
                "selected_frame_positions": positions,
                "vision_block0_input_shape": timing["vision_first_block_input_shape"],
                "language_block0_input_shape": timing["language_first_block_input_shape"],
            }
        )
        if index == 0 or (index + 1) % 25 == 0 or index + 1 == len(test_rows):
            print(
                f"[{args.scheme} {args.frames}f] {index + 1}/{len(test_rows)} "
                f"model={latency:.3f}ms preprocess={preprocessing_ms:.1f}ms",
                flush=True,
            )
    loop_wall_seconds = time.perf_counter() - loop_started
    timer.close()
    power_samples = sampler.stop()

    targets = np.asarray([record["target_mos"] for record in records], dtype=np.float64)
    predictions = np.asarray([record["prediction"] for record in records], dtype=np.float64)
    latencies = [record["model_internal_cuda_ms"] for record in records]
    report = {
        "schema_version": 1,
        "status": "complete",
        "protocol": "one_process_one_model_load_full_test_no_explicit_warmup",
        "scheme": args.scheme,
        "frame_count": args.frames,
        "target": args.target,
        "prompt": core.PROMPTS[args.target],
        "image_size_wh": [args.image_size, args.image_size],
        "test_videos": len(records),
        "explicit_warmup_forwards": 0,
        "pretest_inference_forwards": 0,
        "first_test_video_included_in_aggregate": True,
        "model_loaded_once": True,
        "complete_qwen_forwards": len(records),
        "timing_boundary": (
            "pre-hook at first Vision Transformer block through all Vision/Language blocks "
            "and the scheme-specific readout until the score is ready on GPU"
        ),
        "main_latency_excludes": [
            "processor/model loading",
            "MP4 open/random seek/decode",
            "center crop and resize",
            "processor/tokenizer",
            "CPU-to-GPU transfer",
            "vision patch embedding before Vision block 0",
        ],
        "model_internal_cuda_ms_all_558_first_included": summarize(latencies),
        "first_measured_video_cuda_ms": latencies[0],
        "preprocessing_ms": summarize([record["preprocessing_ms"] for record in records]),
        "performance": core.metrics(targets, predictions),
        f"{args.target}_performance": core.metrics(targets, predictions),
        "power": power_report(power_samples, latencies, power_limit_w=rated_power_w),
        "processor_load_seconds": processor_load_seconds,
        "model_load_seconds": model_load_seconds,
        "test_loop_wall_seconds": loop_wall_seconds,
        "model": str(model_path),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "gpu": gpu_name,
        "hardware_contract": {
            "expected_gpu_name_substring": args.expected_gpu,
            "actual_gpu_name": gpu_name,
            "rated_power_limit_w": rated_power_w,
        },
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
    }
    output = (
        args.output.expanduser().resolve()
        / f"resolution{args.image_size}"
        / f"frames{args.frames}"
        / args.scheme
    )
    output.mkdir(parents=True, exist_ok=True)
    save_power_samples(output / "power_samples.csv", power_samples)
    core.write_json(output / "report.json", report)
    with (output / "per_video_predictions_and_timing.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        for record in records:
            row = dict(record)
            for key in (
                "selected_frame_positions",
                "vision_block0_input_shape",
                "language_block0_input_shape",
            ):
                row[key] = json.dumps(row[key])
            writer.writerow(row)
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
