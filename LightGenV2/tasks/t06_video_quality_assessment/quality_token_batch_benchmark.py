"""A100 batch-scaling and formal LGVQ temporal-quality benchmark.

Each batch contains distinct videos. Video decoding, processor work and H2D
transfer are measured separately. The primary model-only CUDA boundary starts
after all input tensors are resident on the GPU, immediately before the full
Qwen model forward, and ends when the scalar quality score is ready. The older
first-native-Vision-block boundary is retained as a secondary diagnostic.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import torch
from torch import nn

from LightGenV2.common.baseline_measurement import (
    gpu_power_limit_w,
    nvidia_smi_gpu_id,
    validate_cuda_device,
)
from . import quality_token_common as core


@dataclass(frozen=True)
class TelemetrySample:
    phase: str
    host_time_unix: float
    watts: float
    utilization_percent: float
    memory_used_mib: float


class TelemetrySampler:
    """Sample power, utilization, and device memory from the physical CUDA GPU."""

    def __init__(self, interval_ms: int = 50) -> None:
        self.interval_ms = interval_ms
        self.phase: str | None = None
        self.lock = threading.Lock()
        self.samples: list[TelemetrySample] = []
        self.process: subprocess.Popen[str] | None = None
        self.thread: threading.Thread | None = None
        self.error: BaseException | None = None

    def start(self) -> None:
        command = [
            "nvidia-smi",
            f"--id={nvidia_smi_gpu_id()}",
            "--query-gpu=power.draw,utilization.gpu,memory.used",
            "--format=csv,noheader,nounits",
            f"--loop-ms={self.interval_ms}",
        ]
        self.process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1
        )
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self) -> None:
        assert self.process is not None and self.process.stdout is not None
        try:
            for line in self.process.stdout:
                fields = [item.strip() for item in line.split(",")]
                if len(fields) != 3:
                    continue
                try:
                    watts, utilization, memory = map(float, fields)
                except ValueError:
                    continue
                with self.lock:
                    phase = self.phase
                if phase is not None:
                    self.samples.append(
                        TelemetrySample(
                            phase, time.time(), watts, utilization, memory
                        )
                    )
        except BaseException as error:  # surfaced by stop
            self.error = error

    def set_phase(self, phase: str | None) -> None:
        with self.lock:
            self.phase = phase

    def stop(self) -> list[TelemetrySample]:
        if self.process is None:
            return list(self.samples)
        self.set_phase(None)
        self.process.terminate()
        try:
            self.process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=3)
        if self.thread is not None:
            self.thread.join(timeout=3)
        stderr = self.process.stderr.read().strip() if self.process.stderr else ""
        returncode = self.process.returncode
        self.process = None
        if self.error is not None:
            raise RuntimeError("GPU telemetry sampler failed") from self.error
        if returncode not in (0, -15, 1) and stderr:
            raise RuntimeError(f"GPU telemetry sampler failed: {stderr}")
        return list(self.samples)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(*args: str) -> str:
    return subprocess.run(
        ["git", *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def summary(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if not array.size:
        raise ValueError("Cannot summarize an empty sequence")
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "std": float(array.std()),
        "p05": float(np.percentile(array, 5)),
        "p95": float(np.percentile(array, 95)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def telemetry_summary(
    samples: Sequence[TelemetrySample],
    latencies_ms: Sequence[float],
    videos_per_call: int,
    *,
    active_phase_suffix: str | None = None,
) -> dict[str, Any]:
    idle = [sample for sample in samples if sample.phase == "idle"]
    active = [
        sample
        for sample in samples
        if sample.phase.startswith("active:")
        and (active_phase_suffix is None or sample.phase.endswith(active_phase_suffix))
    ]
    if not idle or not active:
        raise RuntimeError(
            f"Insufficient telemetry samples: idle={len(idle)}, active={len(active)}"
        )
    latency = summary(latencies_ms)
    idle_w = float(np.mean([sample.watts for sample in idle]))
    active_w = float(np.mean([sample.watts for sample in active]))
    rated_w = gpu_power_limit_w()
    batch_energy = active_w * latency["mean"] / 1000.0
    return {
        "sampling_interval_ms": 50,
        "idle_samples": len(idle),
        "active_samples": len(active),
        "idle_mean_w": idle_w,
        "active_mean_w": active_w,
        "active_peak_w": float(max(sample.watts for sample in active)),
        "active_mean_gpu_utilization_percent": float(
            np.mean([sample.utilization_percent for sample in active])
        ),
        "active_peak_gpu_utilization_percent": float(
            max(sample.utilization_percent for sample in active)
        ),
        "active_peak_memory_used_mib": float(max(sample.memory_used_mib for sample in active)),
        "rated_power_limit_w": rated_w,
        "active_mean_fraction_of_power_limit": active_w / rated_w,
        "active_peak_fraction_of_power_limit": float(
            max(sample.watts for sample in active)
        )
        / rated_w,
        "measured_active_energy_j_per_batch": batch_energy,
        "measured_active_energy_j_per_video": batch_energy / videos_per_call,
        "idle_subtracted_energy_j_per_batch": max(0.0, active_w - idle_w)
        * latency["mean"]
        / 1000.0,
        "rated_upper_bound_energy_j_per_batch": rated_w * latency["mean"] / 1000.0,
    }


def save_telemetry(path: Path, samples: Sequence[TelemetrySample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["phase", "host_time_unix", "watts", "utilization_percent", "memory_used_mib"]
        )
        for sample in samples:
            writer.writerow(
                [
                    sample.phase,
                    sample.host_time_unix,
                    sample.watts,
                    sample.utilization_percent,
                    sample.memory_used_mib,
                ]
            )


@torch.inference_mode()
def prepare_batch(
    rows: Sequence[dict[str, Any]],
    processor: Any,
    prompt: str,
    device: torch.device,
    image_size: int,
) -> tuple[dict[str, torch.Tensor], list[list[int]], list[dict[str, Any]]]:
    videos: list[Any] = []
    metadata: list[Any] = []
    positions: list[list[int]] = []
    traces: list[dict[str, Any]] = []
    for row in rows:
        video_path = Path(row["video_path"])
        capture = cv2.VideoCapture(str(video_path))
        source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        source_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        source_fps = float(capture.get(cv2.CAP_PROP_FPS))
        capture.release()
        nominal_crop_side = max(2, round(min(source_height, source_width) * 0.65))
        frames, video_metadata, frame_positions = core.decode_random_seek(
            video_path, core.FRAME_FRACTIONS[4], image_size
        )
        videos.append(frames)
        metadata.append(video_metadata)
        positions.append(frame_positions)
        traces.append(
            {
                "sample_id": row["sample_id"],
                "video_path": str(video_path),
                "source_file_bytes": video_path.stat().st_size,
                "source_video_size_wh": [source_width, source_height],
                "source_frame_count": source_frames,
                "source_fps": source_fps,
                "selected_frame_positions": frame_positions,
                "center_crop_fraction_of_short_side": 0.65,
                "nominal_center_crop_size_wh": [nominal_crop_side, nominal_crop_side],
                "resize_output_size_wh": [image_size, image_size],
                "resize_interpolation": "OpenCV INTER_AREA",
                "processor_pixel_constraint": {
                    "min_pixels": image_size**2,
                    "max_pixels": image_size**2,
                },
            }
        )
    inputs = processor(
        text=[prompt] * len(rows),
        videos=videos,
        video_metadata=metadata,
        padding=True,
        return_tensors="pt",
        do_sample_frames=False,
    )
    return {key: value.to(device) for key, value in inputs.items()}, positions, traces


@torch.inference_mode()
def forward_batch(
    inputs: dict[str, torch.Tensor],
    model: nn.Module,
    head: core.FiveNativeTokenRows,
    scores: torch.Tensor,
    timer: core.BoundaryTimer,
) -> tuple[float, list[float], dict[str, Any]]:
    torch.cuda.synchronize()
    full_host_started = time.perf_counter()
    full_start_event = torch.cuda.Event(enable_timing=True)
    full_end_event = torch.cuda.Event(enable_timing=True)
    full_start_event.record()
    timer.reset(expected_calls=1)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        hidden = model.model(**inputs, return_dict=True, use_cache=False).last_hidden_state
    mask = inputs["attention_mask"].bool()
    indices = torch.arange(mask.shape[1], device=hidden.device).expand_as(mask)
    last = indices.masked_fill(~mask, -1).amax(1)
    pooled = hidden[torch.arange(hidden.shape[0], device=hidden.device), last].float()
    # Retain the same complete native vocabulary projection used by the formal
    # scheme-2 baseline before selecting the five trainable quality rows.
    _native_full_logits = model.lm_head(pooled.to(model.lm_head.weight.dtype))
    prediction = head(pooled).softmax(-1) @ scores
    full_end_event.record()
    timing = timer.finish()
    full_host_ended = time.perf_counter()
    timing["full_gpu_input_to_score_cuda_ms"] = float(
        full_start_event.elapsed_time(full_end_event)
    )
    timing["full_gpu_input_to_score_synchronized_wall_ms"] = 1000.0 * (
        full_host_ended - full_host_started
    )
    return (
        float(timing["full_gpu_input_to_score_cuda_ms"]),
        [float(value) for value in prediction.detach().cpu().tolist()],
        timing,
    )


def load_runtime(args: argparse.Namespace) -> tuple[Any, nn.Module, Any, Any, list[dict[str, Any]], str]:
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    model_path = args.model.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    processor = AutoProcessor.from_pretrained(
        str(model_path),
        min_pixels=args.image_size**2,
        max_pixels=args.image_size**2,
        local_files_only=True,
        trust_remote_code=True,
    )
    model = (
        Qwen3VLForConditionalGeneration.from_pretrained(
            str(model_path),
            local_files_only=True,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        )
        .to("cuda:0")
        .eval()
        .requires_grad_(False)
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("architecture") != "frozen_qwen3vl_plus_five_untied_native_lm_output_rows":
        raise RuntimeError("Checkpoint is not the formal five-quality-token baseline")
    head = core.FiveNativeTokenRows(payload["state_dict"]["weight"]).to("cuda:0").eval()
    scores = payload["level_scores"].float().to("cuda:0")
    rows = [
        row
        for row in core.read_manifest(args.manifest.expanduser().resolve())
        if row["split"] == "test"
    ]
    if len(rows) != 558:
        raise RuntimeError(f"Expected 558 fixed test videos, got {len(rows)}")
    return processor, model, head, scores, rows, core.render_prompt(processor, "temporal")


def provenance(args: argparse.Namespace, gpu_name: str) -> dict[str, Any]:
    return {
        "git_commit": git_value("rev-parse", "HEAD"),
        "git_worktree_clean": git_value("status", "--porcelain") == "",
        "script_sha256": sha256(Path(__file__)),
        "manifest": str(args.manifest.expanduser().resolve()),
        "manifest_sha256": sha256(args.manifest.expanduser().resolve()),
        "checkpoint": str(args.checkpoint.expanduser().resolve()),
        "checkpoint_sha256": sha256(args.checkpoint.expanduser().resolve()),
        "model": str(args.model.expanduser().resolve()),
        "gpu": gpu_name,
        "physical_gpu_id": nvidia_smi_gpu_id(),
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
    }


def run_sweep(args: argparse.Namespace, runtime: tuple[Any, ...], gpu_name: str) -> dict[str, Any]:
    processor, model, head, scores, rows, prompt = runtime
    timer = core.BoundaryTimer(model)
    results: list[dict[str, Any]] = []
    for batch_size in args.batch_sizes:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        try:
            inputs, _, _ = prepare_batch(
                rows[:batch_size], processor, prompt, torch.device("cuda:0"), args.image_size
            )
            preprocessing_ms = 1000.0 * (time.perf_counter() - started)
            for _ in range(args.sweep_warmup):
                forward_batch(inputs, model, head, scores, timer)
            sampler = TelemetrySampler()
            sampler.start()
            sampler.set_phase("idle")
            time.sleep(2.0)
            sampler.set_phase(None)
            latencies: list[float] = []
            wall_latencies: list[float] = []
            legacy_latencies: list[float] = []
            for trial in range(args.sweep_trials):
                sampler.set_phase(f"active:b{batch_size}:trial{trial}")
                try:
                    latency, _, timing = forward_batch(inputs, model, head, scores, timer)
                finally:
                    sampler.set_phase(None)
                latencies.append(latency)
                wall_latencies.append(
                    float(timing["full_gpu_input_to_score_synchronized_wall_ms"])
                )
                legacy_latencies.append(
                    float(timing["vision_first_block_to_score_cuda_ms"])
                )
            samples = sampler.stop()
            latency_report = summary(latencies)
            telemetry = telemetry_summary(samples, latencies, batch_size)
            result = {
                "batch_size_videos": batch_size,
                "frames_per_video": 4,
                "distinct_videos_per_batch": True,
                "sweep_warmup_forwards": args.sweep_warmup,
                "timed_forwards": args.sweep_trials,
                "preprocessing_ms_once_not_in_primary_latency": preprocessing_ms,
                "full_gpu_input_to_score_batch_cuda_ms": latency_report,
                "full_gpu_input_to_score_batch_synchronized_wall_ms": summary(
                    wall_latencies
                ),
                "legacy_vision_block0_to_score_batch_cuda_ms": summary(
                    legacy_latencies
                ),
                "throughput_videos_per_second": 1000.0 * batch_size / latency_report["mean"],
                "torch_peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                "torch_peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
                "telemetry": telemetry,
                "status": "complete",
            }
            batch_dir = args.output / "sweep" / f"batch_{batch_size:02d}"
            save_telemetry(batch_dir / "telemetry.csv", samples)
            core.write_json(batch_dir / "result.json", result)
            results.append(result)
            print(json.dumps(result, indent=2), flush=True)
            del inputs
        except torch.OutOfMemoryError as error:
            torch.cuda.empty_cache()
            result = {
                "batch_size_videos": batch_size,
                "status": "oom",
                "error": str(error),
            }
            results.append(result)
            print(json.dumps(result, indent=2), flush=True)
            break
    timer.close()
    report = {
        "schema_version": 1,
        "status": "complete",
        "mode": "batch_power_sweep",
        "workload": "LGVQ temporal quality, four distinct 448x448 frames per video",
        "timing_boundary": (
            "all input tensors resident on GPU, immediately before the full Qwen forward, "
            "through the complete native vocabulary projection and five-quality-token score"
        ),
        "timing_excludes": [
            "model/processor load",
            "MP4 open and random-seek decode",
            "center crop/resize",
            "processor/tokenizer",
            "host-to-device copy",
        ],
        "secondary_timing_boundary": (
            "first native Vision Transformer block input to scalar score ready on GPU"
        ),
        "selection_rule": (
            "Select the largest feasible batch at or before the throughput/power plateau; "
            "report measured power fraction instead of claiming the 250 W limit was reached."
        ),
        "results": results,
        **provenance(args, gpu_name),
    }
    core.write_json(args.output / "sweep" / "report.json", report)
    return report


def run_formal(args: argparse.Namespace, runtime: tuple[Any, ...], gpu_name: str) -> dict[str, Any]:
    processor, model, head, scores, rows, prompt = runtime
    batch_size = args.formal_batch
    timer = core.BoundaryTimer(model)
    prepared_batches: list[dict[str, Any]] = []
    preprocessing_phase_started = time.perf_counter()
    for batch_index, start in enumerate(range(0, len(rows), batch_size)):
        selected = rows[start : start + batch_size]
        preprocessing_started = time.perf_counter()
        cpu_inputs, positions, preprocessing_traces = prepare_batch(
            selected, processor, prompt, torch.device("cpu"), args.image_size
        )
        preprocessing_ms = 1000.0 * (time.perf_counter() - preprocessing_started)
        prepared_batches.append(
            {
                "batch_index": batch_index,
                "start": start,
                "selected": selected,
                "inputs": cpu_inputs,
                "positions": positions,
                "preprocessing_traces": preprocessing_traces,
                "preprocessing_ms": preprocessing_ms,
            }
        )
        if batch_index == 0 or (batch_index + 1) % 10 == 0:
            print(
                f"[preprocess batch={batch_size}] {min(start + batch_size, len(rows))}/{len(rows)} "
                f"cpu={preprocessing_ms:.1f}ms",
                flush=True,
            )
    preprocessing_phase_wall_seconds = time.perf_counter() - preprocessing_phase_started
    sampler = TelemetrySampler()
    sampler.start()
    sampler.set_phase("idle")
    time.sleep(2.0)
    sampler.set_phase(None)
    records: list[dict[str, Any]] = []
    batch_records: list[dict[str, Any]] = []
    inference_phase_started = time.perf_counter()
    for prepared in prepared_batches:
        batch_index = int(prepared["batch_index"])
        start = int(prepared["start"])
        selected = prepared["selected"]
        positions = prepared["positions"]
        preprocessing_traces = prepared["preprocessing_traces"]
        transfer_started = time.perf_counter()
        inputs = {key: value.to("cuda:0") for key, value in prepared["inputs"].items()}
        prepared["inputs"] = {}
        torch.cuda.synchronize()
        transfer_ms = 1000.0 * (time.perf_counter() - transfer_started)
        preprocessing_ms = float(prepared["preprocessing_ms"])
        sampler.set_phase(f"active:batch{batch_index}:size{len(selected)}")
        try:
            latency, predictions, timing = forward_batch(inputs, model, head, scores, timer)
        finally:
            sampler.set_phase(None)
        batch_records.append(
            {
                "batch_index": batch_index,
                "batch_size_videos": len(selected),
                "full_gpu_input_to_score_cuda_ms": latency,
                "full_gpu_input_to_score_synchronized_wall_ms": float(
                    timing["full_gpu_input_to_score_synchronized_wall_ms"]
                ),
                "legacy_vision_block0_to_score_cuda_ms": float(
                    timing["vision_first_block_to_score_cuda_ms"]
                ),
                "legacy_vision_block0_to_score_synchronized_wall_ms": float(
                    timing["vision_first_block_to_score_host_ms"]
                ),
                "preprocessing_ms": preprocessing_ms,
                "host_to_device_ms": transfer_ms,
                "component_sum_decode_processor_transfer_model_ms": (
                    preprocessing_ms + transfer_ms + latency
                ),
                "input_tensor_contract": json.dumps(
                    {
                        key: {
                            "shape": list(value.shape),
                            "dtype": str(value.dtype),
                            "device": str(value.device),
                        }
                        for key, value in inputs.items()
                    },
                    sort_keys=True,
                ),
                "vision_block0_input_shape": timing["vision_first_block_input_shape"],
                "language_block0_input_shape": timing["language_first_block_input_shape"],
            }
        )
        for offset, (row, prediction) in enumerate(zip(selected, predictions)):
            records.append(
                {
                    "sample_index": start + offset,
                    "sample_id": row["sample_id"],
                    "target_mos": float(row["temporal"]),
                    "prediction": prediction,
                    "batch_index": batch_index,
                    "batch_size_videos": len(selected),
                    "selected_frame_positions": positions[offset],
                    "preprocessing_trace": preprocessing_traces[offset],
                }
            )
        del inputs
        if batch_index == 0 or len(records) % 100 < batch_size or len(records) == len(rows):
            print(
                f"[formal batch={batch_size}] {len(records)}/{len(rows)} "
                f"model={latency:.3f}ms preprocess={preprocessing_ms:.1f}ms",
                flush=True,
            )
    inference_phase_wall_seconds = time.perf_counter() - inference_phase_started
    timer.close()
    samples = sampler.stop()
    full_batches = [row for row in batch_records if row["batch_size_videos"] == batch_size]
    full_latencies = [
        float(row["full_gpu_input_to_score_cuda_ms"]) for row in full_batches
    ]
    all_latencies = [
        float(row["full_gpu_input_to_score_cuda_ms"]) for row in batch_records
    ]
    full_wall_latencies = [
        float(row["full_gpu_input_to_score_synchronized_wall_ms"])
        for row in full_batches
    ]
    full_legacy_latencies = [
        float(row["legacy_vision_block0_to_score_cuda_ms"])
        for row in full_batches
    ]
    targets = np.asarray([row["target_mos"] for row in records], dtype=np.float64)
    predictions = np.asarray([row["prediction"] for row in records], dtype=np.float64)
    telemetry = telemetry_summary(
        samples, full_latencies, batch_size, active_phase_suffix=f":size{batch_size}"
    )
    mean_batch_ms = summary(full_latencies)["mean"]
    batches_for_16 = math.ceil(16 / batch_size)
    comparable_16_ms = mean_batch_ms * batches_for_16
    report = {
        "schema_version": 1,
        "status": "complete",
        "mode": "formal_full_test_batched",
        "protocol": "one_process_one_model_load_full_558_test_first_batch_included",
        "target": "temporal",
        "prompt": core.PROMPTS["temporal"],
        "image_size_wh": [args.image_size, args.image_size],
        "frames_per_video": 4,
        "batch_size_videos": batch_size,
        "test_videos": len(records),
        "full_batches": len(full_batches),
        "final_partial_batch_videos": len(rows) % batch_size,
        "explicit_warmup_forwards": 0,
        "first_test_batch_included_in_aggregate": True,
        "performance": core.metrics(targets, predictions),
        "timing_boundary": (
            "all input tensors resident on GPU, immediately before the full Qwen forward, "
            "through the complete native vocabulary projection and five-quality-token score"
        ),
        "timing_excludes": [
            "model/processor load",
            "MP4 open and random-seek decode",
            "center crop/resize",
            "processor/tokenizer",
            "host-to-device copy",
        ],
        "secondary_timing_boundary": (
            "first native Vision Transformer block input to scalar score ready on GPU"
        ),
        "full_gpu_input_to_score_full_batch_cuda_ms": summary(full_latencies),
        "full_gpu_input_to_score_full_batch_synchronized_wall_ms": summary(
            full_wall_latencies
        ),
        "legacy_vision_block0_to_score_full_batch_cuda_ms": summary(
            full_legacy_latencies
        ),
        "full_gpu_input_to_score_all_batches_cuda_ms": summary(all_latencies),
        "preprocessing_full_batch_ms": summary(
            [float(row["preprocessing_ms"]) for row in full_batches]
        ),
        "host_to_device_full_batch_ms": summary(
            [float(row["host_to_device_ms"]) for row in full_batches]
        ),
        "component_sum_decode_processor_transfer_model_full_batch_ms": summary(
            [
                float(row["component_sum_decode_processor_transfer_model_ms"])
                for row in full_batches
            ]
        ),
        "model_throughput_videos_per_second": 1000.0 * batch_size / mean_batch_ms,
        "same_workload_16_video_model_ms": comparable_16_ms,
        "same_workload_16_video_batch_calls": batches_for_16,
        "same_workload_16_video_measured_active_energy_j": (
            telemetry["measured_active_energy_j_per_batch"] * batches_for_16
        ),
        "same_workload_16_video_idle_subtracted_energy_j": (
            telemetry["idle_subtracted_energy_j_per_batch"] * batches_for_16
        ),
        "same_workload_16_video_rated_upper_bound_energy_j": (
            telemetry["rated_upper_bound_energy_j_per_batch"] * batches_for_16
        ),
        "telemetry": telemetry,
        "preprocessing_phase_wall_seconds": preprocessing_phase_wall_seconds,
        "continuous_inference_phase_wall_seconds": inference_phase_wall_seconds,
        "formal_execution_order": (
            "all CPU decode/processor batches first, then all GPU batches consecutively; "
            "this prevents excluded decode gaps from down-clocking the model-active power sample"
        ),
        "preprocessing_contract": {
            "source_sampling": "four deterministic positions at FRAME_FRACTIONS[4]",
            "frame_fractions": list(core.FRAME_FRACTIONS[4]),
            "per_decoded_frame": (
                "center crop to round(0.65 * min(source width, source height)), "
                "then OpenCV INTER_AREA resize to 448x448 RGB"
            ),
            "processor": (
                "Qwen3-VL AutoProcessor, min_pixels=max_pixels=448^2, padding enabled, "
                "do_sample_frames=False"
            ),
            "prompt_template": prompt,
            "raw_per_sample_trace": "sample_preprocessing.jsonl",
            "raw_per_batch_tensor_shapes": "batch_timing.csv:input_tensor_contract",
        },
        **provenance(args, gpu_name),
    }
    output = args.output / "formal" / f"batch_{batch_size:02d}"
    save_telemetry(output / "telemetry.csv", samples)
    core.write_json(output / "report.json", report)
    with (output / "batch_timing.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(batch_records[0]))
        writer.writeheader()
        for row in batch_records:
            value = dict(row)
            value["vision_block0_input_shape"] = json.dumps(value["vision_block0_input_shape"])
            value["language_block0_input_shape"] = json.dumps(value["language_block0_input_shape"])
            writer.writerow(value)
    with (output / "predictions.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        for row in records:
            value = dict(row)
            value["selected_frame_positions"] = json.dumps(value["selected_frame_positions"])
            value["preprocessing_trace"] = json.dumps(
                value["preprocessing_trace"], sort_keys=True
            )
            writer.writerow(value)
    with (output / "sample_preprocessing.jsonl").open(
        "w", encoding="utf-8", newline="\n"
    ) as stream:
        for row in records:
            stream.write(
                json.dumps(row["preprocessing_trace"], sort_keys=True) + "\n"
            )
    print(json.dumps(report, indent=2), flush=True)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("sweep", "formal"))
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--expected-gpu", default="NVIDIA A100-PCIE-40GB")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16, 24, 32])
    parser.add_argument("--sweep-warmup", type=int, default=3)
    parser.add_argument("--sweep-trials", type=int, default=30)
    parser.add_argument("--formal-batch", type=int, default=16)
    args = parser.parse_args()
    args.output = args.output.expanduser().resolve()
    if args.image_size != 448:
        raise ValueError("This formal contract is fixed to 448x448")
    if any(value < 1 for value in args.batch_sizes) or args.formal_batch < 1:
        raise ValueError("Batch sizes must be positive")
    gpu_name = validate_cuda_device(args.expected_gpu)
    runtime = load_runtime(args)
    if args.mode == "sweep":
        run_sweep(args, runtime, gpu_name)
    else:
        run_formal(args, runtime, gpu_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
