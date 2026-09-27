"""Narrow, GPU-only A100 timing for the electronic neural path of optical MoE.

The measured scope is intentionally narrower than the deployment audit:

* inputs are already resident on the GPU and already cropped/stacked;
* router timing begins from the four measured expert energies;
* layer timing contains CCD intensity nonlinearity and learned readout only;
* RMS/alpha fusion, SLM field construction, layout/scatter, I/O and transfers are excluded;
* the learned bridge and task head are reported separately;
* the electronic residual route is timed separately because it runs in parallel with optics.

Every CUDA-event and synchronized-wall observation is retained.  The paper value is
the pooled CUDA-event median, never the minimum observation.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

import torch
from torch import nn
from torch.nn import functional as F

from LightGenV2.scripts import profile_latest_optical_electronics_a100 as full


RAW: list[dict[str, Any]] = []
POWER_SAMPLER: "GpuTelemetrySampler | None" = None
POWER_DWELL_SECONDS = 0.0


class GpuTelemetrySampler:
    """Phase-tagged raw A100 power/utilization/memory/clock telemetry."""

    def __init__(self, gpu_index: int, interval_ms: int = 10) -> None:
        self.gpu_index = gpu_index
        self.interval_ms = interval_ms
        self.phase: str | None = None
        self.lock = threading.Lock()
        self.samples: list[dict[str, Any]] = []
        self.process: subprocess.Popen[str] | None = None
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        command = [
            "nvidia-smi", f"--id={self.gpu_index}",
            "--query-gpu=power.draw,utilization.gpu,memory.used,clocks.sm",
            "--format=csv,noheader,nounits", f"--loop-ms={self.interval_ms}",
        ]
        self.process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self) -> None:
        assert self.process is not None and self.process.stdout is not None
        for line in self.process.stdout:
            pieces = [piece.strip() for piece in line.split(",")]
            if len(pieces) != 4:
                continue
            try:
                power, utilization, memory, clock = map(float, pieces)
            except ValueError:
                continue
            with self.lock:
                phase = self.phase
            if phase is not None:
                self.samples.append({
                    "host_time_unix": time.time(),
                    "host_monotonic_s": time.monotonic(),
                    "phase": phase,
                    "power_w": power,
                    "gpu_utilization_percent": utilization,
                    "memory_used_mib": memory,
                    "sm_clock_mhz": clock,
                })

    def set_phase(self, phase: str | None) -> None:
        with self.lock:
            self.phase = phase

    def stop(self) -> list[dict[str, Any]]:
        self.set_phase(None)
        if self.process is not None:
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=3)
        if self.thread is not None:
            self.thread.join(timeout=3)
        return list(self.samples)


def compute_processes(gpu_index: int) -> list[dict[str, Any]]:
    result = subprocess.run([
        "nvidia-smi", f"--id={gpu_index}",
        "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ], text=True, capture_output=True, check=True)
    rows: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        pieces = [piece.strip() for piece in line.split(",", 2)]
        if len(pieces) == 3 and pieces[0].isdigit():
            rows.append({"pid": int(pieces[0]), "process_name": pieces[1], "used_memory_mib": float(pieces[2])})
    return rows


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def statistics_for(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "std": statistics.pstdev(values),
        "p05": percentile(values, 0.05),
        "p95": percentile(values, 0.95),
        "minimum": min(values),
        "maximum": max(values),
    }


@torch.inference_mode()
def benchmark(
    task: str,
    component: str,
    function: Callable[[], Any],
    *,
    trial: int,
    warmup: int,
    repeats: int,
    shape_contract: str,
) -> None:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    for iteration in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        wall_start = time.perf_counter_ns()
        start.record()
        function()
        end.record()
        end.synchronize()
        wall_ms = (time.perf_counter_ns() - wall_start) / 1.0e6
        RAW.append({
            "task": task,
            "component": component,
            "trial": trial,
            "iteration": iteration,
            "cuda_event_ms": float(start.elapsed_time(end)),
            "synchronized_wall_ms": wall_ms,
            "shape_contract": shape_contract,
        })
    if POWER_SAMPLER is not None:
        POWER_SAMPLER.set_phase(f"active:{task}:{component}")
        deadline = time.monotonic() + POWER_DWELL_SECONDS
        try:
            while time.monotonic() < deadline:
                for _ in range(64):
                    function()
                torch.cuda.synchronize()
        finally:
            POWER_SAMPLER.set_phase(None)


def standard_router_core(energy: torch.Tensor) -> torch.Tensor:
    centered = energy - energy.mean(-1, keepdim=True)
    logits = centered / centered.square().mean(-1, keepdim=True).add(1.0e-8).sqrt()
    return full.legacy._sparse_top2(torch.softmax(logits / 2.0, -1))


def temporal_router_core(energy: torch.Tensor) -> torch.Tensor:
    centered = energy - energy.mean(-1, keepdim=True)
    logits = centered / centered.square().mean(-1, keepdim=True).add(1.0e-8).sqrt()
    return full.legacy._sparse_top2(torch.softmax(logits / 1.1, -1))


def add_standard(
    task: str,
    legacy_key: str,
    device: torch.device,
    *,
    trial: int,
    warmup: int,
    repeats: int,
) -> dict[str, int]:
    spec = full.legacy.SPECS[legacy_key]
    detector = torch.rand(1, 478, 478, device=device)
    energy = torch.rand(1, 4, device=device)

    benchmark(task, "router_core_from_4_energies", lambda: standard_router_core(energy),
        trial=trial, warmup=warmup, repeats=repeats,
        shape_contract="GPU-resident four expert energies [1,4] -> normalize/softmax/Top2")

    modalities: list[tuple[str, int, bool, int]] = [
        ("vision", spec.vision_tokens, False, 3)
    ]
    if spec.language:
        modalities.append(("language", spec.language_tokens, True, 5))

    for modality, tokens, causal, kernel in modalities:
        readout = full.legacy.StandardCCDReadout(tokens).to(device).eval()
        hidden = torch.randn(1, tokens, 192, device=device)
        mask = torch.ones(1, tokens, dtype=torch.bool, device=device)
        if task == "abo_image_to_image":
            residual = full.AboI2IResidual(modality == "vision", kernel_size=kernel, mlp_width=384).to(device).eval()
            residual_call = lambda residual=residual, hidden=hidden: residual(hidden)
        else:
            residual = full.legacy.StandardResidualBlock(causal=causal, kernel_size=kernel).to(device).eval()
            residual_call = lambda residual=residual, hidden=hidden, mask=mask: residual(hidden, mask)
        benchmark(task, f"{modality}_ccd_nonlinearity_readout_nn", lambda readout=readout: readout(detector),
            trial=trial, warmup=warmup, repeats=repeats,
            shape_contract=f"GPU-resident CCD [1,478,478] -> log nonlinearity/pool/LN/Linear -> [1,{tokens},192]; no fusion")
        benchmark(task, f"{modality}_parallel_residual_nn", residual_call,
            trial=trial, warmup=warmup, repeats=repeats,
            shape_contract=f"GPU-resident [{1},{tokens},192] -> learned residual network only")

    if task == "abo_image_to_text":
        head = full.legacy.RetrievalHead().to(device).eval()
        value = torch.randn(1, 224, device=device)
        head_call = lambda: head(value)
        head_shape = "GPU-resident [1,224] -> retrieval LN/Linear64/L2"
    elif task == "abo_image_to_image":
        head = full.AboI2IHead("linear64").to(device).eval()
        value = torch.randn(1, 77, 192, device=device)
        head_call = lambda: head(value)
        head_shape = "GPU-resident [1,77,192] -> mean/max/LN/Linear64/L2"
    elif task == "lsp":
        head = full.PoseHeatmapDecoder().to(device).eval()
        spatial = torch.randn(1, 192, 14, 14, device=device)
        head_call = lambda: head(spatial)
        head_shape = "GPU-resident [1,192,14,14] -> current pose decoder"
    elif task == "salicon":
        head = full.SaliencyDensityDecoder().to(device).eval()
        spatial = torch.randn(1, 192, 14, 14, device=device)
        head_call = lambda: head(spatial)
        head_shape = "GPU-resident [1,192,14,14] -> current saliency decoder"
    elif task == "openmoji":
        bridge = full.LatestOpenMojiBridge().to(device).eval()
        language = torch.randn(1, 64, 192, device=device)
        condition = bridge.shared.summarize([row for row in language])

        def bridge_nn() -> tuple[torch.Tensor, torch.Tensor]:
            summary = bridge.shared.summarize([row for row in language])
            return summary, torch.tanh(bridge.prompt_to_vision(summary))

        benchmark(task, "bridge_nn_only", bridge_nn,
            trial=trial, warmup=warmup, repeats=repeats,
            shape_contract="GPU-resident language [1,64,192] -> learned position summary and prompt projection; no broadcast/add")
        head = full.LatestOpenMojiHead().to(device).eval()
        spatial = torch.randn(1, 192, 14, 14, device=device)
        head_call = lambda: head(spatial, condition)
        head_shape = "GPU-resident spatial+condition -> current shared OpenMoji neural decoder"
    else:
        raise KeyError(task)

    benchmark(task, "task_head_nn", head_call, trial=trial, warmup=warmup, repeats=repeats,
        shape_contract=head_shape)
    return {
        "router_core_from_4_energies": spec.router_passes,
        "vision_ccd_nonlinearity_readout_nn": 2,
        "vision_parallel_residual_nn": 2,
        **({
            "language_ccd_nonlinearity_readout_nn": 2,
            "language_parallel_residual_nn": 2,
        } if spec.language else {}),
        **({"bridge_nn_only": 1} if task == "openmoji" else {}),
        "task_head_nn": 1,
    }


def add_temporal(
    device: torch.device,
    *,
    trial: int,
    warmup: int,
    repeats: int,
) -> dict[str, int]:
    task = "lgvq_temporal"
    frame_readout = full.legacy.T06FrameReadout().to(device).eval()
    video_readout = full.legacy.T06VideoReadout().to(device).eval()
    frame_patches = torch.rand(64, 56, 56, device=device)
    video_patches = torch.rand(16, 115, 115, device=device)
    frame_hidden = torch.randn(16, 4, 49, 192, device=device)
    video_hidden = torch.randn(16, 42, 192, device=device)
    video_mask = torch.ones(16, 42, dtype=torch.bool, device=device)

    def frame_readout_nn() -> torch.Tensor:
        normalized = full.legacy._normalize_patch(frame_patches)
        pooled = frame_readout.pool(normalized.unsqueeze(1)).squeeze(1)
        return frame_readout.output(F.softplus(frame_readout.norm(pooled)))

    def video_readout_nn() -> torch.Tensor:
        normalized = full.legacy._normalize_patch(video_patches)
        pooled = F.adaptive_avg_pool2d(normalized.unsqueeze(1), (42, 96)).squeeze(1)
        return video_readout.output(F.softplus(video_readout.norm(pooled)))

    benchmark(task, "frame_ccd_nonlinearity_readout_nn", frame_readout_nn,
        trial=trial, warmup=warmup, repeats=repeats,
        shape_contract="64 pre-cropped GPU CCD patches [64,56,56] -> log/pool/LN/Linear; no crop/stack/fusion")
    benchmark(task, "video_ccd_nonlinearity_readout_nn", video_readout_nn,
        trial=trial, warmup=warmup, repeats=repeats,
        shape_contract="16 pre-cropped GPU CCD patches [16,115,115] -> log/pool/LN/Linear; no crop/stack/fusion")

    frame_residual = full.legacy.T06VisionResidual().to(device).eval()
    video_residual = full.legacy.T06LanguageResidual().to(device).eval()
    benchmark(task, "frame_parallel_residual_nn", lambda: frame_residual(frame_hidden),
        trial=trial, warmup=warmup, repeats=repeats,
        shape_contract="GPU-resident 16x4x49x192 -> learned frame residual only")
    benchmark(task, "video_parallel_residual_nn", lambda: video_residual(video_hidden, video_mask),
        trial=trial, warmup=warmup, repeats=repeats,
        shape_contract="GPU-resident 16x42x192 -> learned video residual only")

    frame_energy = torch.rand(1, 16, 4, 4, device=device)
    video_energy = torch.rand(1, 16, 4, device=device)
    benchmark(task, "frame_router_core_from_4_energies", lambda: temporal_router_core(frame_energy),
        trial=trial, warmup=warmup, repeats=repeats,
        shape_contract="GPU-resident 16x4 frame four-expert energies -> normalize/softmax/Top2")
    benchmark(task, "video_router_core_from_4_energies", lambda: temporal_router_core(video_energy),
        trial=trial, warmup=warmup, repeats=repeats,
        shape_contract="GPU-resident 16-video four-expert energies -> normalize/softmax/Top2")

    bridge = full.legacy.T06Bridge().to(device).eval()
    frame_summary = torch.randn(1, 16, 4, 384, device=device)
    serial_sequence = torch.randn(1, 16, 42, 192, device=device)

    def bridge_nn() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image = bridge.frame_merger(frame_summary)
        serial = F.softplus(bridge.serial_field(serial_sequence))
        router_encoded = F.softplus(bridge.router_width(image))
        router = F.softplus(bridge.router_frames(router_encoded.transpose(-2, -1))).transpose(-2, -1)
        return image, serial, router

    benchmark(task, "bridge_nn_only", bridge_nn,
        trial=trial, warmup=warmup, repeats=repeats,
        shape_contract="pre-shaped GPU frame summary/sequence -> learned frame merger and field projections; no concat/layout")

    head = full.legacy.T06Head().to(device).eval()
    benchmark(task, "task_head_nn", lambda: head(frame_hidden, video_hidden, video_mask),
        trial=trial, warmup=warmup, repeats=repeats,
        shape_contract="GPU-resident 16-video frame/video tokens -> current temporal neural readout -> 16 MOS")
    return {
        "frame_router_core_from_4_energies": 1,
        "video_router_core_from_4_energies": 1,
        "frame_ccd_nonlinearity_readout_nn": 2,
        "video_ccd_nonlinearity_readout_nn": 2,
        "frame_parallel_residual_nn": 2,
        "video_parallel_residual_nn": 2,
        "bridge_nn_only": 1,
        "task_head_nn": 1,
    }


def add_spatial(
    device: torch.device,
    *,
    trial: int,
    warmup: int,
    repeats: int,
) -> dict[str, int]:
    task = "lgvq_spatial"
    settings, model = full.load_spatial_model(device)
    lane_size = settings.geometry.lane_size
    lanes = torch.rand(settings.frame_count, lane_size, lane_size, device=device)
    active = torch.rand(1, settings.geometry.active_size, settings.geometry.active_size, device=device)

    def vision_readout_nn() -> torch.Tensor:
        normalized = model.parallel_optics._normalize(lanes)
        return model.parallel_optics.expert_readout(normalized)

    def language_readout_nn() -> torch.Tensor:
        normalized = active.float().clamp_min(0.0)
        mean = normalized.mean((-2, -1), keepdim=True).clamp_min(1.0e-6)
        normalized = torch.log1p(settings.ccd_log_compression * (normalized / mean).clamp_max(settings.ccd_relative_clip))
        readout = model.serial_optics.expert_readout
        pooled = F.adaptive_avg_pool2d(normalized.unsqueeze(1), (settings.maximum_language_tokens, settings.detector_projection_size)).squeeze(1)
        return readout.output(F.softplus(readout.norm(pooled)))

    benchmark(task, "vision_ccd_nonlinearity_readout_nn", vision_readout_nn,
        trial=trial, warmup=warmup, repeats=repeats,
        shape_contract="four pre-cropped GPU lanes -> current log nonlinearity/CcdReadout; no crop/stack/fusion")
    benchmark(task, "language_ccd_nonlinearity_readout_nn", language_readout_nn,
        trial=trial, warmup=warmup, repeats=repeats,
        shape_contract="pre-cropped GPU active CCD -> current log nonlinearity/pool/LN/Linear; no fusion")

    vision = torch.randn(1, settings.frame_count, settings.token_count, settings.model_width, device=device)
    language = torch.randn(1, settings.maximum_language_tokens, settings.model_width, device=device)
    mask = torch.ones(1, settings.maximum_language_tokens, dtype=torch.bool, device=device)
    benchmark(task, "vision_parallel_residual_nn", lambda: model.vision_routes[0](vision),
        trial=trial, warmup=warmup, repeats=repeats,
        shape_contract="current learned vision residual only")
    benchmark(task, "language_parallel_residual_nn", lambda: model.language_routes[0](language, mask),
        trial=trial, warmup=warmup, repeats=repeats,
        shape_contract="current learned language residual only")

    energy = torch.rand(1, 4, device=device)
    benchmark(task, "router_core_from_4_energies", lambda: standard_router_core(energy),
        trial=trial, warmup=warmup, repeats=repeats,
        shape_contract="GPU-resident four expert energies -> normalize/softmax/Top2")

    summary = torch.cat((vision.mean(2), vision.amax(2)), -1)
    benchmark(task, "bridge_nn_only", lambda: model.frame_merger(summary),
        trial=trial, warmup=warmup, repeats=repeats,
        shape_contract="pre-aggregated GPU [1,4,384] -> current LN/Linear/GELU frame merger")
    benchmark(task, "task_head_nn", lambda: model.readout(vision, language, mask),
        trial=trial, warmup=warmup, repeats=repeats,
        shape_contract=f"GPU-resident tokens -> current {type(model.readout).__name__} neural readout")
    return {
        "router_core_from_4_energies": 2,
        "vision_ccd_nonlinearity_readout_nn": 2,
        "language_ccd_nonlinearity_readout_nn": 2,
        "vision_parallel_residual_nn": 2,
        "language_parallel_residual_nn": 2,
        "bridge_nn_only": 1,
        "task_head_nn": 1,
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize_power(samples: list[dict[str, Any]]) -> dict[str, Any]:
    idle = [float(sample["power_w"]) for sample in samples if sample["phase"] == "idle"]
    active = [float(sample["power_w"]) for sample in samples if sample["phase"].startswith("active:")]
    if not idle or not active:
        raise RuntimeError("Power run did not capture both idle and active samples")
    components: dict[str, dict[str, float | int]] = {}
    phases = sorted({str(sample["phase"]) for sample in samples if str(sample["phase"]).startswith("active:")})
    for phase in phases:
        selected = [sample for sample in samples if sample["phase"] == phase]
        values = [float(sample["power_w"]) for sample in selected]
        utilization = [float(sample["gpu_utilization_percent"]) for sample in selected]
        memory = [float(sample["memory_used_mib"]) for sample in selected]
        clocks = [float(sample["sm_clock_mhz"]) for sample in selected]
        components[phase.removeprefix("active:")] = {
            "samples": len(values),
            "mean_w": statistics.fmean(values),
            "median_w": statistics.median(values),
            "p05_w": percentile(values, 0.05),
            "p95_w": percentile(values, 0.95),
            "peak_w": max(values),
            "gpu_utilization_mean_percent": statistics.fmean(utilization),
            "gpu_utilization_median_percent": statistics.median(utilization),
            "gpu_utilization_p95_percent": percentile(utilization, 0.95),
            "memory_used_mean_mib": statistics.fmean(memory),
            "memory_used_peak_mib": max(memory),
            "sm_clock_mean_mhz": statistics.fmean(clocks),
        }
    idle_rows = [sample for sample in samples if sample["phase"] == "idle"]
    active_rows = [sample for sample in samples if str(sample["phase"]).startswith("active:")]
    return {
        "sampler": "nvidia-smi board power.draw",
        "interval_ms": 10,
        "idle_samples": len(idle),
        "idle_mean_w": statistics.fmean(idle),
        "idle_median_w": statistics.median(idle),
        "idle_p95_w": percentile(idle, 0.95),
        "active_samples": len(active),
        "active_mean_w": statistics.fmean(active),
        "active_median_w": statistics.median(active),
        "active_p95_w": percentile(active, 0.95),
        "active_peak_w": max(active),
        "active_gpu_utilization_mean_percent": statistics.fmean(float(row["gpu_utilization_percent"]) for row in active_rows),
        "active_gpu_utilization_median_percent": statistics.median(float(row["gpu_utilization_percent"]) for row in active_rows),
        "active_gpu_utilization_p95_percent": percentile([float(row["gpu_utilization_percent"]) for row in active_rows], 0.95),
        "active_memory_used_mean_mib": statistics.fmean(float(row["memory_used_mib"]) for row in active_rows),
        "active_memory_used_peak_mib": max(float(row["memory_used_mib"]) for row in active_rows),
        "idle_gpu_utilization_mean_percent": statistics.fmean(float(row["gpu_utilization_percent"]) for row in idle_rows),
        "idle_memory_used_mean_mib": statistics.fmean(float(row["memory_used_mib"]) for row in idle_rows),
        "rated_power_limit_w": 250.0,
        "per_component": components,
    }


def main() -> int:
    global POWER_SAMPLER, POWER_DWELL_SECONDS
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=2000)
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--device-warmup-seconds", type=float, default=5.0)
    parser.add_argument("--measure-power", action="store_true")
    parser.add_argument("--nvidia-smi-index", type=int, default=6)
    parser.add_argument("--idle-seconds", type=float, default=10.0)
    parser.add_argument("--power-dwell-seconds", type=float, default=3.0)
    parser.add_argument(
        "--allow-200-no-warmup",
        action="store_true",
        help="Allow the dataset-run audit protocol: one trial, 200 calls, zero explicit warm-up.",
    )
    parser.add_argument(
        "--formal-statistic", choices=("median", "mean"), default="median",
        help="Statistic used to compose formal per-task electronic latency.",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda:0")
    if "A100" not in torch.cuda.get_device_name(device):
        raise RuntimeError(f"A100 required, got {torch.cuda.get_device_name(device)}")
    if (args.repeats < 1000 or args.trials < 2) and not (
        args.allow_200_no_warmup
        and args.repeats == 200
        and args.trials == 1
        and args.warmup == 0
        and args.device_warmup_seconds == 0
    ):
        raise ValueError("Paper run requires >=1000 calls and >=2 trials")
    torch.manual_seed(20260914)
    torch.cuda.manual_seed_all(20260914)
    full.condition_device_clock(device, args.device_warmup_seconds)

    samples: list[dict[str, Any]] = []
    process_audit: dict[str, Any] = {"before": compute_processes(args.nvidia_smi_index)}
    foreign_before = [row for row in process_audit["before"] if row["pid"] != os.getpid()]
    if foreign_before:
        raise RuntimeError(f"A100 has foreign compute processes before measurement: {foreign_before}")
    sampler = GpuTelemetrySampler(args.nvidia_smi_index, interval_ms=10) if args.measure_power else None
    if sampler is not None:
        sampler.start()
        sampler.set_phase("idle")
        time.sleep(args.idle_seconds)
        sampler.set_phase(None)
        POWER_SAMPLER = sampler
        POWER_DWELL_SECONDS = args.power_dwell_seconds
    occurrence_contracts: dict[str, dict[str, int]] = {}
    try:
        for trial in range(1, args.trials + 1):
            print(f"[trial {trial}/{args.trials}]", flush=True)
            occurrence_contracts["lgvq_temporal"] = add_temporal(device, trial=trial, warmup=args.warmup, repeats=args.repeats)
            occurrence_contracts["lgvq_spatial"] = add_spatial(device, trial=trial, warmup=args.warmup, repeats=args.repeats)
            for task, legacy_key in (
                ("abo_image_to_text", "t08"),
                ("abo_image_to_image", "t08"),
                ("lsp", "t02"),
                ("salicon", "t03"),
                ("openmoji", "t04"),
            ):
                print(f"  {task}", flush=True)
                occurrence_contracts[task] = add_standard(task, legacy_key, device,
                    trial=trial, warmup=args.warmup, repeats=args.repeats)
    finally:
        POWER_SAMPLER = None
        if sampler is not None:
            samples = sampler.stop()
    process_audit["after"] = compute_processes(args.nvidia_smi_index)
    process_audit["foreign_after"] = [row for row in process_audit["after"] if row["pid"] != os.getpid()]

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    raw_path = output / "all_per_call_timings.csv"
    with raw_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(RAW[0]))
        writer.writeheader()
        writer.writerows(RAW)

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in RAW:
        grouped.setdefault((row["task"], row["component"]), []).append(row)
    summaries: list[dict[str, Any]] = []
    for (task, component), rows in sorted(grouped.items()):
        cuda = statistics_for([float(row["cuda_event_ms"]) for row in rows])
        wall = statistics_for([float(row["synchronized_wall_ms"]) for row in rows])
        summaries.append({
            "task": task, "component": component,
            "trials": args.trials, "calls": len(rows),
            **{f"cuda_{key}_ms": value for key, value in cuda.items()},
            **{f"wall_{key}_ms": value for key, value in wall.items()},
            "shape_contract": rows[0]["shape_contract"],
        })
    summary_path = output / "component_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    selected = {
        (row["task"], row["component"]): float(row[f"cuda_{args.formal_statistic}_ms"])
        for row in summaries
    }
    formal: list[dict[str, Any]] = []
    for task, occurrences in occurrence_contracts.items():
        router = sum(selected[task, name] * count for name, count in occurrences.items() if "router_core" in name)
        layer = sum(selected[task, name] * count for name, count in occurrences.items() if "ccd_nonlinearity_readout_nn" in name)
        residual = sum(selected[task, name] * count for name, count in occurrences.items() if "parallel_residual_nn" in name)
        bridge = sum(selected[task, name] * count for name, count in occurrences.items() if name == "bridge_nn_only")
        head = sum(selected[task, name] * count for name, count in occurrences.items() if name == "task_head_nn")
        formal.append({
            "task": task,
            "router_core_cuda_ms": router,
            "layer_nonlinearity_readout_nn_cuda_ms": layer,
            "task_head_nn_cuda_ms": head,
            "main_electronic_excluding_bridge_cuda_ms": router + layer + head,
            "bridge_nn_cuda_ms_separate": bridge,
            "main_electronic_including_bridge_cuda_ms": router + layer + head + bridge,
            "parallel_residual_nn_cuda_ms_separate": residual,
            "fusion_ms_in_formal_scope": 0.0,
        })
    formal_path = output / "paper_narrow_summary.csv"
    with formal_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(formal[0]))
        writer.writeheader()
        writer.writerows(formal)

    power = summarize_power(samples) if samples else None
    if samples:
        with (output / "power_samples.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(samples[0]))
            writer.writeheader()
            writer.writerows(samples)
        (output / "power_summary.json").write_text(
            json.dumps(power, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    environment = {
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": torch.cuda.get_device_name(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "precision": "float32",
        "execution": "eager torch.inference_mode; CUDA event; synchronize every call; no torch.compile",
        "warmup_per_component_per_trial": args.warmup,
        "measured_calls_per_component_per_trial": args.repeats,
        "trials": args.trials,
        "formal_statistic": args.formal_statistic,
        "scope": {
            "included": ["router arithmetic from four GPU-resident energies", "CCD nonlinearity and learned readout", "learned bridge", "actual task head"],
            "separate": ["parallel electronic residual network", "bridge"],
            "excluded": ["host/device transfer", "CCD crop/stack", "RMS/alpha fusion", "SLM encoding/layout/scatter", "file I/O"],
        },
        "raw_rows": len(RAW),
        "power_measurement": {
            "enabled": bool(samples),
            "sampling_interval_ms": 10 if samples else None,
            "idle_seconds_requested": args.idle_seconds if samples else None,
            "dwell_seconds_per_component_per_trial": args.power_dwell_seconds if samples else None,
        },
        "compute_process_audit": process_audit,
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], text=True, capture_output=True).stdout.strip(),
    }
    (output / "report.json").write_text(json.dumps({
        "schema_version": 1,
        "environment": environment,
        "occurrence_contracts": occurrence_contracts,
        "formal_narrow_cuda_event": formal,
        "power": power,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "README.md").write_text(
        "# A100 narrow optical-MoE electronics timing\n\n"
        f"Formal values use the pooled CUDA-event {args.formal_statistic} from all measured calls. Inputs are preloaded "
        "and pre-shaped on GPU. Fusion, transfers, layout, SLM encoding and I/O are excluded. "
        "Bridge and parallel residual are retained as separate columns.\n",
        encoding="utf-8",
    )
    files = [raw_path, summary_path, formal_path, output / "report.json", output / "README.md"]
    if samples:
        files.extend([output / "power_samples.csv", output / "power_summary.json"])
    (output / "SHA256SUMS.txt").write_text("".join(f"{sha256(path)}  {path.name}\n" for path in files), encoding="utf-8")
    print(json.dumps({"output": str(output), "raw_rows": len(RAW), "formal": formal}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
