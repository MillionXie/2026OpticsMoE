"""Paper-grade A100 timing audit for the latest LightGenV2 optical students.

This program measures only the electronic operations surrounding physical
SLM/CCD calls.  It deliberately does not execute FFT propagation.  Every
measured invocation is retained in CSV with both a CUDA-event duration and a
fully synchronized host-wall duration.  The formal optical-student latency is
composed from CUDA-event medians; synchronized wall values are retained as an
audit, not mixed into the formal estimate.

The low-level 478/518 field packing routines are imported from the established
5090-D profiler.  Task heads that changed after that profiler was written are
imported from their current task modules below.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import importlib.util
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

import torch
from torch import nn
from torch.nn import functional as F

from LightGenV2.common.baseline_measurement import (
    NvidiaSmiPowerSampler,
    PowerSample,
    save_power_samples,
)
from LightGenV2.scripts import profile_optical_electronics_5090d as legacy
from LightGenV2.tasks.t04_semantic_interaction.embedding_model import PositionReadout
from experiments.qwen3_vl_2b_openmoji_instruction_four_stage_optical_editing.modeling import SemanticGridDecoder
from experiments.qwen3_vl_2b_synthetic_instruction_four_stage_optical_editing.modeling import ConditionedResidual2D
try:
    from LightGenV2.tasks.t07_abo_image_retrieval.standalone.model import (
        Residual as AboI2IResidual,
        RetrievalHead as AboI2IHead,
    )
    T07_CLASS_SOURCE = "imported current standalone.model"
except ModuleNotFoundError:
    # Some lab-server branches contain the completed T07 artifact but predate
    # the standalone source directory.  Keep the benchmark runnable without
    # modifying that dirty branch by embedding the two bit-for-bit small
    # modules whose source SHA is still recorded from the measurement script.
    T07_CLASS_SOURCE = "embedded exact source fallback from current standalone.model"

    class AboI2IResidual(nn.Module):
        def __init__(self, vision: bool, kernel_size: int, mlp_width: int = 384) -> None:
            super().__init__()
            self.vision, self.kernel_size = vision, kernel_size
            self.token_norm = nn.LayerNorm(192)
            self.token_depthwise = (
                nn.Conv2d(192, 192, kernel_size, groups=192, bias=False)
                if vision else nn.Conv1d(192, 192, kernel_size, groups=192, bias=False)
            )
            self.token_pointwise = nn.Linear(192, 192)
            self.token_dropout = nn.Dropout(0.1)
            self.token_residual_logit = nn.Parameter(torch.zeros(()))
            self.residual_logit = nn.Parameter(torch.zeros(()))
            self.norm = nn.LayerNorm(192)
            self.mlp = nn.Sequential(
                nn.Linear(192, mlp_width), nn.GELU(), nn.Dropout(0.1),
                nn.Linear(mlp_width, 192), nn.Dropout(0.1),
            )

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            normalized = self.token_norm(value)
            if self.vision:
                outputs = []
                for row in normalized:
                    grid = row.view(1, 7, 7, 2, 2, 192).permute(0, 5, 1, 3, 2, 4).reshape(1, 192, 14, 14)
                    pad = self.kernel_size // 2
                    grid = self.token_depthwise(F.pad(grid, (pad, pad, pad, pad)))
                    outputs.append(grid.view(1, 192, 7, 2, 7, 2).permute(0, 2, 4, 3, 5, 1).reshape(196, 192))
                update = torch.stack(outputs)
            else:
                update = self.token_depthwise(F.pad(normalized.transpose(1, 2), (self.kernel_size - 1, 0))).transpose(1, 2)
            update = self.token_dropout(self.token_pointwise(F.gelu(update)))
            value = value + self.token_residual_logit.sigmoid() * update
            return value + self.residual_logit.sigmoid() * self.mlp(self.norm(value))

    class AboI2IHead(nn.Module):
        def __init__(self, kind: str = "linear64") -> None:
            super().__init__()
            if kind != "linear64":
                raise ValueError("Formal T07 timing requires linear64")
            self.norm = nn.LayerNorm(384)
            self.projection = nn.Linear(384, 64)

        def forward(self, latent: torch.Tensor) -> torch.Tensor:
            pooled = torch.stack([torch.cat((row.mean(0), row.amax(0))) for row in latent])
            return F.normalize(self.projection(self.norm(pooled.float())), p=2, dim=-1)
from experiments.vision2_hybrid_dense.modeling import (
    PoseHeatmapDecoder,
    SaliencyDensityDecoder,
)


SLM_REFRESH_MS = 0.714
CCD_EXPOSURE_MS = 0.300
CAMERA_READOUT_MS = 0.0307
PHYSICAL_PASS_MS = SLM_REFRESH_MS + CCD_EXPOSURE_MS + CAMERA_READOUT_MS
OPTICAL_RIG_POWER_W = 80.388
A100_RATED_POWER_W = 250.0

RAW_ROWS: list[dict[str, Any]] = []
CURRENT_TASK = ""
POWER_SAMPLER: NvidiaSmiPowerSampler | None = None
POWER_DWELL_SECONDS = 0.0


METRIC_IDENTITIES: dict[str, dict[str, Any]] = {
    "lgvq_temporal": {
        "metric": "SRCC", "value": 0.804372636109386,
        "display_value": 0.8044,
        "artifact": "LightGenV2/tasks/t06_video_quality_assessment/reports/paper_results/temporal_multivideo16x4/result.json",
        "checkpoint_run": "LightGenV2/tasks/t06_video_quality_assessment/runs/simulation/multivideo16x4_rank_s163",
        "status": "exact_artifact_bound",
    },
    "lgvq_spatial": {
        "metric": "SRCC", "value": 0.6710079009479295,
        "display_value": 0.6710,
        "artifact": "LightGenV2/tasks/t06_video_quality_assessment/configs/lightgen/spatial_single_video4_custom_conv_readout_1m.yaml",
        "checkpoint": "LightGenV2/tasks/t06_video_quality_assessment/runs/simulation/spatial_readout_1m_srcc067/best_checkpoint.pt",
        "expected_checkpoint_sha256": "95e12397ccf8c960fa30ba9dfb400b69d2c2ebd02288ac6a4879870ab592828b",
        "status": "exact_artifact_bound",
    },
    "abo_image_to_text": {
        "metric": "R@1", "value": 1934 / 2400,
        "display_value": 0.8058,
        "artifact": "ABO_Lab_8um/README.md",
        "expected_checkpoint_sha256": "a2aa9a93028410dfb780df7d320a9be65d07b6e7ec87ede08b53c4bddcbaf7af",
        "status": "historical_fixed_epoch25_ema; current-environment replay differs by 3/2400 and must not be relabelled",
    },
    "abo_image_to_image": {
        "metric": "R@1", "value": 0.8125, "display_value": 0.8125,
        "artifact": "LightGenV2/tasks/t07_abo_image_retrieval/runs/simulation/abo200_optical_pretrain_20260914/final_report.json",
        "checkpoint_run": "LightGenV2/tasks/t07_abo_image_retrieval/runs/simulation/abo200_optical_pretrain_20260914",
        "expected_source_checkpoint_sha256": "d11f3428efa67c7c5084eb9056d991c4692d36a3d177cd82b4601238357d444a",
        "status": "exact_artifact_bound",
    },
    "lsp": {
        "metric": "PCK@0.2", "value": 0.7347857142857143,
        "display_value": 0.7353,
        "artifact": "LightGenV2/tasks/t02_keypoint_detection/runs/simulation/refinement_20260909/staged_heatmap/final_report.json",
        "expected_checkpoint_sha256": "495b9c2c4e3df15d3715f1ce8f2faea7cb9156275b31103ec684f4e96a3228518",
        "status": "exact_artifact_bound; screenshot is rounded/possibly copied as 0.7353",
    },
    "salicon": {
        "metric": "CC", "value": 0.8624925081777596,
        "display_value": 0.8625,
        "artifact": "LightGenV2/tasks/t03_saliency/runs/simulation/moe_alpha40_sam_batch8_crosssample_20260913_seed42/candidate_selected_evaluation",
        "status": "exact_artifact_bound",
    },
    "openmoji": {
        "metric": "changed_cell_accuracy", "value": 0.8715,
        "display_value": 0.8715,
        "checkpoint": "LightGenV2/tasks/t04_semantic_interaction/runs/simulation/routerfill_shared_s73/best_checkpoint.pt",
        "status": "exact_artifact_bound",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    point = (len(ordered) - 1) * q
    lo, hi = math.floor(point), math.ceil(point)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - point) + ordered[hi] * (point - lo)


def summary(values: list[float]) -> dict[str, float]:
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
def measured_benchmark(
    name: str,
    function: Callable[[], Any],
    *,
    warmup: int,
    repeats: int,
    logical_samples_per_call: int,
    physical_fields_per_call: int,
    shape_contract: str,
) -> dict[str, Any]:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    events: list[float] = []
    walls: list[float] = []
    for index in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        wall_start = time.perf_counter_ns()
        start.record()
        function()
        end.record()
        end.synchronize()
        wall = (time.perf_counter_ns() - wall_start) / 1.0e6
        event = float(start.elapsed_time(end))
        events.append(event)
        walls.append(wall)
        RAW_ROWS.append({
            "task": CURRENT_TASK,
            "component": name,
            "iteration": index,
            "cuda_event_ms": event,
            "synchronized_wall_ms": wall,
            "logical_samples_per_call": logical_samples_per_call,
            "physical_fields_per_call": physical_fields_per_call,
            "shape_contract": shape_contract,
        })
    if POWER_SAMPLER is not None:
        POWER_SAMPLER.set_phase(f"active:{CURRENT_TASK}:{name}")
        deadline = time.monotonic() + POWER_DWELL_SECONDS
        try:
            while time.monotonic() < deadline:
                for _ in range(128):
                    function()
                torch.cuda.synchronize()
        finally:
            POWER_SAMPLER.set_phase(None)
    return {
        "component": name,
        "shape_contract": shape_contract,
        "warmup_calls": warmup,
        "measured_calls": repeats,
        "logical_samples_per_call": logical_samples_per_call,
        "physical_fields_per_call": physical_fields_per_call,
        "logical_samples_measured": repeats * logical_samples_per_call,
        "physical_fields_measured": repeats * physical_fields_per_call,
        "cuda_event_ms": summary(events),
        "synchronized_wall_ms": summary(walls),
    }


class CurrentSharedGridReadout(nn.Module):
    """Current two-condition-group OpenMoji head (routerfill_shared_s73)."""

    def __init__(self, width: int = 192, max_tokens: int = 64) -> None:
        super().__init__()
        self.position_readout = PositionReadout(max_tokens)
        self.language_pool = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.GELU())
        self.post_film = nn.Linear(width, width * 2)
        self.coordinate_projection = nn.Conv2d(2, width, 1)
        self.editor = nn.ModuleList([ConditionedResidual2D(width, d) for d in (1, 2)])
        self.decoder = SemanticGridDecoder(width, 6, 16)
        self.task_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 4))

    def summarize(self, language_groups: list[torch.Tensor]) -> torch.Tensor:
        return self.language_pool(self.position_readout(language_groups))

    def forward(self, spatial: torch.Tensor, condition: torch.Tensor) -> dict[str, torch.Tensor]:
        gamma, beta = self.post_film(condition).chunk(2, -1)
        spatial = spatial * (1 + 0.1 * torch.tanh(gamma)[:, :, None, None]) + 0.1 * torch.tanh(beta)[:, :, None, None]
        axis = torch.linspace(-1.0, 1.0, 14, device=spatial.device, dtype=spatial.dtype)
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        spatial = spatial + self.coordinate_projection(torch.stack((xx, yy))[None])
        for layer in self.editor:
            spatial = layer(spatial, condition)
        category, edit = self.decoder(spatial)
        return {"category_logits": category, "edit_logits": edit, "task_logits": self.task_head(condition)}


class LatestOpenMojiBridge(nn.Module):
    """Actual routerfill_shared_s73 language summary and prompt-to-vision bridge."""

    def __init__(self) -> None:
        super().__init__()
        self.shared = CurrentSharedGridReadout(192, 64)
        self.prompt_to_vision = nn.Sequential(nn.LayerNorm(192), nn.Linear(192, 1024))
        self.gate = nn.Parameter(torch.logit(torch.tensor(0.055)))

    def forward(self, language: torch.Tensor, vision: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        condition = self.shared.summarize([row for row in language])
        bias = torch.tanh(self.prompt_to_vision(condition))
        return condition, vision + torch.sigmoid(self.gate) * bias[:, None]


class LatestOpenMojiHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.shared = CurrentSharedGridReadout(192, 64)

    def forward(self, spatial: torch.Tensor, condition: torch.Tensor) -> tuple[torch.Tensor, ...]:
        result = self.shared(spatial, condition)
        return result["category_logits"], result["edit_logits"], result["task_logits"]


def task_component(task: dict[str, Any], name: str, basis: str) -> float:
    return next(
        float(row[basis]["median"])
        for row in task["components"]
        if row["component"] == name
    )


def compose_standard(task: dict[str, Any], *, basis: str) -> dict[str, Any]:
    spec = task["specification"]
    router = task_component(task, "router_ccd_to_expert_slm", basis)
    head = task_component(task, "task_head", basis)
    total = spec["router_passes"] * (PHYSICAL_PASS_MS + router)
    electronic_serial = spec["router_passes"] * router + head
    residual_total = 0.0
    for modality in ("vision", "language"):
        if modality == "language" and not spec["language"]:
            continue
        fused = task_component(task, f"{modality}_ccd_to_fusion", basis)
        nxt = task_component(task, f"{modality}_ccd_to_next_slm", basis)
        residual = task_component(task, f"{modality}_parallel_residual", basis)
        total += max(PHYSICAL_PASS_MS, residual) + nxt
        total += max(PHYSICAL_PASS_MS, residual) + fused
        electronic_serial += nxt + fused
        residual_total += 2.0 * residual
    bridge_rows = [row for row in task["components"] if row["component"] == "language_to_vision_bridge"]
    bridge = float(bridge_rows[0][basis]["median"]) if bridge_rows else 0.0
    total += bridge + head
    electronic_serial += bridge
    return {
        "composed_critical_ms_per_call": total,
        "physical_only_ms_per_call": (spec["feature_passes"] + spec["router_passes"]) * PHYSICAL_PASS_MS,
        "serialized_electronic_ms_per_call": electronic_serial,
        "parallel_residual_sum_ms_per_call": residual_total,
        "bridge_ms": bridge,
        "task_head_ms": head,
        "logical_samples_per_call": spec["logical_samples_per_call"],
        "composed_critical_ms_per_logical_sample": total / spec["logical_samples_per_call"],
    }


def compose_standard_neural_only(task: dict[str, Any], *, basis: str) -> dict[str, Any]:
    """Formal paper scope: no SLM-canvas packing and no parallel residual sum."""
    spec = task["specification"]
    router = task_component(task, "router_ccd_to_weights", basis)
    head = task_component(task, "task_head", basis)
    serial = spec["router_passes"] * router + head
    residual_sum = 0.0
    for modality in ("vision", "language"):
        if modality == "language" and not spec["language"]:
            continue
        serial += 2.0 * task_component(task, f"{modality}_ccd_to_fusion", basis)
        residual_sum += 2.0 * task_component(task, f"{modality}_parallel_residual", basis)
    bridge_rows = [row for row in task["components"] if row["component"] == "language_to_vision_bridge"]
    bridge = float(bridge_rows[0][basis]["median"]) if bridge_rows else 0.0
    serial += bridge
    physical = (spec["feature_passes"] + spec["router_passes"]) * PHYSICAL_PASS_MS
    return {
        "electronic_serial_ms_per_call": serial,
        "parallel_residual_sum_ms_per_call": residual_sum,
        "each_residual_is_covered_by_one_physical_pass": all(
            task_component(task, f"{modality}_parallel_residual", basis) <= PHYSICAL_PASS_MS
            for modality in ("vision", "language")
            if modality == "vision" or spec["language"]
        ),
        "bridge_ms": bridge, "task_head_ms": head,
        "physical_only_ms_per_call": physical,
        "hybrid_physical_plus_serial_electronic_ms_per_call": physical + serial,
        "logical_samples_per_call": spec["logical_samples_per_call"],
        "hybrid_ms_per_logical_sample": (physical + serial) / spec["logical_samples_per_call"],
        "scope": "CCD readout/nonlinearity/fusion twice per modality + optical-router weights + bridge + task head; excludes SLM canvas packing/reload and parallel residual",
    }


def compose_temporal(task: dict[str, Any], *, basis: str) -> dict[str, Any]:
    t = lambda name: task_component(task, name, basis)
    total = PHYSICAL_PASS_MS + t("frame_router_ccd_to_expert_slm")
    total += max(PHYSICAL_PASS_MS, t("frame_parallel_residual")) + t("frame_ccd_to_next_slm")
    total += max(PHYSICAL_PASS_MS, t("frame_parallel_residual")) + t("frame_ccd_to_fusion")
    total += t("frame_to_video_bridge")
    total += PHYSICAL_PASS_MS + t("video_router_ccd_to_expert_slm")
    total += max(PHYSICAL_PASS_MS, t("video_parallel_residual")) + t("video_ccd_to_next_slm")
    total += max(PHYSICAL_PASS_MS, t("video_parallel_residual")) + t("video_ccd_to_fusion")
    total += t("task_head")
    serial = sum(t(name) for name in (
        "frame_router_ccd_to_expert_slm", "frame_ccd_to_next_slm", "frame_ccd_to_fusion",
        "frame_to_video_bridge", "video_router_ccd_to_expert_slm", "video_ccd_to_next_slm",
        "video_ccd_to_fusion", "task_head",
    ))
    residual = 2 * t("frame_parallel_residual") + 2 * t("video_parallel_residual")
    return {
        "composed_critical_ms_per_call": total,
        "physical_only_ms_per_call": 6 * PHYSICAL_PASS_MS,
        "serialized_electronic_ms_per_call": serial,
        "parallel_residual_sum_ms_per_call": residual,
        "bridge_ms": t("frame_to_video_bridge"),
        "task_head_ms": t("task_head"),
        "logical_samples_per_call": 16,
        "composed_critical_ms_per_logical_sample": total / 16,
    }


def compose_temporal_neural_only(task: dict[str, Any], *, basis: str) -> dict[str, Any]:
    t = lambda name: task_component(task, name, basis)
    serial = (
        t("frame_router_ccd_to_weights") + 2 * t("frame_ccd_to_fusion")
        + t("frame_to_video_bridge") + t("video_router_ccd_to_weights")
        + 2 * t("video_ccd_to_fusion") + t("task_head")
    )
    residual = 2 * t("frame_parallel_residual") + 2 * t("video_parallel_residual")
    physical = 6 * PHYSICAL_PASS_MS
    return {
        "electronic_serial_ms_per_call": serial,
        "parallel_residual_sum_ms_per_call": residual,
        "each_residual_is_covered_by_one_physical_pass": (
            t("frame_parallel_residual") <= PHYSICAL_PASS_MS
            and t("video_parallel_residual") <= PHYSICAL_PASS_MS
        ),
        "bridge_ms": t("frame_to_video_bridge"), "task_head_ms": t("task_head"),
        "physical_only_ms_per_call": physical,
        "hybrid_physical_plus_serial_electronic_ms_per_call": physical + serial,
        "logical_samples_per_call": 16,
        "hybrid_ms_per_logical_sample": (physical + serial) / 16,
        "scope": "16-video shared-field CCD readout/fusion + router weights + frame/video bridge + task head; excludes SLM canvas packing/reload and parallel residual",
    }


def temporal_router_weights(device: torch.device, *, frame: bool) -> tuple[nn.Module, torch.Tensor, Callable[[], torch.Tensor], str]:
    detector = torch.rand(1, 518, 518, device=device)
    if frame:
        router = legacy.T06FrameRouterPost().to(device).eval()

        def function() -> torch.Tensor:
            rows = []
            for top, left in router.origins:
                for dy, dx in router.frames:
                    y, x = 20 + top + dy, 20 + left + dx
                    tile = detector[:, y:y + 56, x:x + 56]
                    rows.append(torch.stack([
                        tile[:, y0:y1, x0:x1].sum((-2, -1))
                        for y0, y1 in router.intervals for x0, x1 in router.intervals
                    ], -1))
            energy = torch.stack(rows, 1).reshape(1, 16, 4, 4)
            centered = energy - energy.mean(-1, keepdim=True)
            probability = torch.softmax(centered / centered.square().mean(-1, keepdim=True).add(1e-8).sqrt() / 1.1, -1)
            return legacy._sparse_top2(probability)

        shape = "actual 64 frame lanes: CCD region sums -> normalization -> softmax -> Top2 weights; no fanout"
    else:
        router = legacy.T06VideoRouterPost().to(device).eval()

        def function() -> torch.Tensor:
            rows = []
            for top, left in router.origins:
                y, x = 20 + top, 20 + left
                tile = detector[:, y:y + 115, x:x + 115]
                rows.append(torch.stack([
                    tile[:, y0:y1, x0:x1].sum((-2, -1))
                    for y0, y1 in router.intervals for x0, x1 in router.intervals
                ], -1))
            energy = torch.stack(rows, 1)
            centered = energy - energy.mean(-1, keepdim=True)
            probability = torch.softmax(centered / centered.square().mean(-1, keepdim=True).add(1e-8).sqrt() / 1.1, -1)
            return legacy._sparse_top2(probability)

        shape = "actual 16 video lanes: CCD region sums -> normalization -> softmax -> Top2 weights; no fanout"
    return router, detector, function, shape


def replace_task_head(
    task: dict[str, Any],
    function: Callable[[], Any],
    shape: str,
    warmup: int,
    repeats: int,
    logical_samples: int = 1,
) -> None:
    task["components"] = [row for row in task["components"] if row["component"] != "task_head"]
    task["components"].append(measured_benchmark(
        "task_head", function, warmup=warmup, repeats=repeats,
        logical_samples_per_call=logical_samples, physical_fields_per_call=1,
        shape_contract=shape,
    ))


def run_standard(task_name: str, legacy_key: str, device: torch.device, warmup: int, repeats: int) -> dict[str, Any]:
    global CURRENT_TASK
    CURRENT_TASK = task_name
    task = legacy._standard_task(legacy_key, device, warmup, repeats)
    detector = torch.rand(1, 478, 478, device=device)
    router_weights = legacy.StandardRouterPost().to(device).eval()
    task["components"].append(measured_benchmark(
        "router_ccd_to_weights", lambda: router_weights(detector),
        warmup=warmup, repeats=repeats, logical_samples_per_call=1,
        physical_fields_per_call=1,
        shape_contract="CCD [1,478,478] -> four ROI energies -> normalize/softmax -> Top2 weights; no SLM fanout",
    ))
    if task_name == "abo_image_to_image":
        # Replace both residual timings and the legacy row-readout with the exact
        # six-capture standalone architecture used by the 0.8125 checkpoint.
        task["components"] = [row for row in task["components"] if row["component"] not in {
            "vision_parallel_residual", "language_parallel_residual", "task_head"
        }]
        vision = torch.randn(1, 196, 192, device=device)
        language = torch.randn(1, 77, 192, device=device)
        vr = AboI2IResidual(True, kernel_size=3, mlp_width=384).to(device).eval()
        lr = AboI2IResidual(False, kernel_size=5, mlp_width=384).to(device).eval()
        head = AboI2IHead("linear64").to(device).eval()
        task["components"].append(measured_benchmark(
            "vision_parallel_residual", lambda: vr(vision), warmup=warmup, repeats=repeats,
            logical_samples_per_call=1, physical_fields_per_call=1,
            shape_contract="actual T07 Residual vision [1,196,192], k3, MLP384",
        ))
        task["components"].append(measured_benchmark(
            "language_parallel_residual", lambda: lr(language), warmup=warmup, repeats=repeats,
            logical_samples_per_call=1, physical_fields_per_call=1,
            shape_contract="actual T07 Residual language [1,77,192], causal k5, MLP384",
        ))
        task["components"].append(measured_benchmark(
            "task_head", lambda: head(language), warmup=warmup, repeats=repeats,
            logical_samples_per_call=1, physical_fields_per_call=1,
            shape_contract="actual T07 all-token mean/max [1,77,192] -> LN384 -> Linear64 -> L2",
        ))
        task["specification"].update(task=task_name, label="ABO image-to-image", language_tokens=77)
    task["formal_cuda_event"] = compose_standard(task, basis="cuda_event_ms")
    task["synchronized_wall_audit"] = compose_standard(task, basis="synchronized_wall_ms")
    task["formal_neural_only_cuda_event"] = compose_standard_neural_only(task, basis="cuda_event_ms")
    task["neural_only_wall_audit"] = compose_standard_neural_only(task, basis="synchronized_wall_ms")
    return task


def load_spatial_model(device: torch.device) -> tuple[Any, nn.Module]:
    snapshot = os.environ.get("LIGHTGEN_LATEST_LGVQ_SOURCE")
    if snapshot:
        source = Path(snapshot).resolve()
        package = "_lightgen_latest_lgvq_snapshot"
        package_module = type(sys)(package)
        package_module.__path__ = [str(source)]
        sys.modules[package] = package_module
        settings_spec = importlib.util.spec_from_file_location(f"{package}.settings", source / "settings.py")
        if settings_spec is None or settings_spec.loader is None:
            raise RuntimeError("Cannot load latest LGVQ settings snapshot")
        settings_module = importlib.util.module_from_spec(settings_spec)
        sys.modules[settings_spec.name] = settings_module
        settings_spec.loader.exec_module(settings_module)
        modeling_spec = importlib.util.spec_from_file_location(f"{package}.modeling", source / "modeling.py")
        if modeling_spec is None or modeling_spec.loader is None:
            raise RuntimeError("Cannot load latest LGVQ modeling snapshot")
        modeling_module = importlib.util.module_from_spec(modeling_spec)
        sys.modules[modeling_spec.name] = modeling_module
        modeling_spec.loader.exec_module(modeling_module)
        load_settings = settings_module.load_settings
        model_class = modeling_module.LGVQSingleMetricOEO16
    else:
        from experiments.qwen3_vl_2b_lgvq_single_metric_o2_16frame_54.settings import load_settings
        from experiments.qwen3_vl_2b_lgvq_single_metric_o2_16frame_54.modeling import LGVQSingleMetricOEO16
        model_class = LGVQSingleMetricOEO16
    path = Path("experiments/qwen3_vl_2b_lgvq_single_metric_o2_16frame_54/configs/release/spatial_readout_1m_srcc067.yaml")
    settings = load_settings(path, synthetic=True)
    return settings, model_class(settings).to(device).eval()


def run_spatial(device: torch.device, warmup: int, repeats: int) -> dict[str, Any]:
    """Measure the current four-frame LGVQ modules rather than legacy clones."""
    global CURRENT_TASK
    CURRENT_TASK = "lgvq_spatial"
    settings, model = load_spatial_model(device)
    detector = torch.rand(1, settings.geometry.canvas_size, settings.geometry.canvas_size, device=device)
    vision = torch.randn(1, settings.frame_count, settings.token_count, settings.model_width, device=device)
    language = torch.randn(1, settings.maximum_language_tokens, settings.model_width, device=device)
    mask = torch.ones(1, settings.maximum_language_tokens, dtype=torch.bool, device=device)
    weights = torch.softmax(torch.rand(1, settings.frame_count, 4, device=device), -1)
    serial_weights = torch.softmax(torch.rand(1, 4, device=device), -1)
    results: list[dict[str, Any]] = []

    def parallel_fused() -> torch.Tensor:
        optical = model.parallel_optics._read(detector, "vision_expert")
        return model.fusions[0](vision, optical)

    def parallel_next() -> torch.Tensor:
        return model.parallel_optics.fields(parallel_fused())

    def serial_fused() -> torch.Tensor:
        optical = model.serial_optics._read(detector, "language_expert", language.shape[1])
        return model.fusions[2](language, optical, mask)

    def serial_next() -> torch.Tensor:
        return model.serial_optics.fields(serial_fused())

    # Post-CCD router code is represented by the same measured four-energy
    # Top-2/fanout contract used by the deployed hardware workflow.
    legacy_router = legacy.StandardRouterPost().to(device).eval()
    legacy_fanout = legacy.StandardFanout().to(device).eval()
    router_field = torch.rand(1, 224, 224, device=device)
    calls: list[tuple[str, Callable[[], Any], str]] = [
        ("vision_ccd_to_fusion", parallel_fused, "actual LGVQ 4-lane CCD [1,518,518] -> [1,4,196,192] + RMS fusion"),
        ("vision_ccd_to_next_slm", parallel_next, "actual LGVQ CCD/fusion -> four 109x109 next-stage fields"),
        ("vision_parallel_residual", lambda: model.vision_routes[0](vision), "actual current vision route [1,4,196,192]"),
        ("language_ccd_to_fusion", serial_fused, f"actual LGVQ serial CCD -> [1,{settings.maximum_language_tokens},192] + RMS fusion"),
        ("language_ccd_to_next_slm", serial_next, "actual LGVQ serial CCD/fusion -> 109x109 field"),
        ("language_parallel_residual", lambda: model.language_routes[0](language, mask), "actual current causal language route"),
        ("router_ccd_to_expert_slm", lambda: legacy_fanout(router_field, legacy_router(detector[:, 20:-20, 20:-20])), "hardware-equivalent CCD ROI energies -> Top2 -> fanout"),
        ("router_ccd_to_weights", lambda: legacy_router(detector[:, 20:-20, 20:-20]), "hardware-equivalent CCD ROI energies -> Top2 weights; no fanout"),
        ("language_to_vision_bridge", lambda: model.frame_merger(torch.cat((vision.mean(2), vision.amax(2)), -1)), "actual frame merger [1,4,196,192] -> four sequence image tokens"),
        ("task_head", lambda: model.readout(vision, language, mask), "actual 967,458-param SpatialLowRankPrunedGridCompactResidualReadout"),
    ]
    for name, function, shape in calls:
        results.append(measured_benchmark(name, function, warmup=warmup, repeats=repeats,
            logical_samples_per_call=1, physical_fields_per_call=1, shape_contract=shape))
    task = {
        "specification": {
            "task": "lgvq_spatial", "label": "LGVQ spatial single-video 4-frame",
            "language": True, "physical_fields_per_call": 1, "logical_samples_per_call": 1,
            "feature_passes": 4, "router_passes": 2,
            "vision_tokens": settings.token_count, "language_tokens": settings.maximum_language_tokens,
            "frame_count": settings.frame_count,
            "actual_readout_class": type(model.readout).__name__,
            "actual_readout_parameters": sum(p.numel() for p in model.readout.parameters()),
        },
        "components": results,
    }
    task["formal_cuda_event"] = compose_standard(task, basis="cuda_event_ms")
    task["synchronized_wall_audit"] = compose_standard(task, basis="synchronized_wall_ms")
    task["formal_neural_only_cuda_event"] = compose_standard_neural_only(task, basis="cuda_event_ms")
    task["neural_only_wall_audit"] = compose_standard_neural_only(task, basis="synchronized_wall_ms")
    return task


def power_summary(samples: list[PowerSample]) -> dict[str, Any]:
    idle = [s.watts for s in samples if s.phase == "idle"]
    active = [s.watts for s in samples if s.phase.startswith("active:")]
    phases: dict[str, Any] = {}
    for phase in sorted({s.phase for s in samples if s.phase.startswith("active:")}):
        values = [s.watts for s in samples if s.phase == phase]
        phases[phase.removeprefix("active:")] = {
            "samples": len(values), "mean_w": statistics.fmean(values), "peak_w": max(values)
        }
    return {
        "sampler": "nvidia-smi board power.draw", "interval_ms": 10,
        "idle_samples": len(idle), "active_samples": len(active),
        "idle_mean_w": statistics.fmean(idle), "active_mean_w": statistics.fmean(active),
        "active_peak_w": max(active), "rated_upper_w": A100_RATED_POWER_W,
        "per_component": phases,
    }


def artifact_manifest(root: Path) -> list[dict[str, Any]]:
    rows = []
    for identity in METRIC_IDENTITIES.values():
        for key in ("artifact", "checkpoint"):
            value = identity.get(key)
            if not value:
                continue
            path = root / value
            rows.append({
                "role": key, "path": value, "exists": path.is_file(),
                "bytes": path.stat().st_size if path.is_file() else None,
                "sha256": sha256(path) if path.is_file() else None,
            })
    source_files = [
        Path(__file__), Path(legacy.__file__),
        root / "LightGenV2/tasks/t04_semantic_interaction/shared_readout.py",
        root / "LightGenV2/tasks/t07_abo_image_retrieval/standalone/model.py",
        root / "experiments/vision2_hybrid_dense/modeling.py",
        root / "experiments/qwen3_vl_2b_lgvq_single_metric_o2_16frame_54/modeling.py",
        root / "experiments/qwen3_vl_2b_lgvq_single_metric_o2_16frame_54/settings.py",
    ]
    for path in source_files:
        resolved = path.resolve()
        exists = resolved.is_file()
        try:
            display_path = str(resolved.relative_to(root))
        except ValueError:
            display_path = str(resolved)
        rows.append({
            "role": "source", "path": display_path,
            "exists": exists, "bytes": resolved.stat().st_size if exists else None,
            "sha256": sha256(resolved) if exists else None,
        })
    snapshot = os.environ.get("LIGHTGEN_LATEST_LGVQ_SOURCE")
    if snapshot:
        for name in ("settings.py", "modeling.py"):
            path = (Path(snapshot) / name).resolve()
            rows.append({
                "role": "external_latest_source_snapshot", "path": str(path),
                "exists": path.is_file(), "bytes": path.stat().st_size if path.is_file() else None,
                "sha256": sha256(path) if path.is_file() else None,
            })
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@torch.inference_mode()
def condition_device_clock(device: torch.device, seconds: float) -> dict[str, Any]:
    """Make dynamic GPU clock state reproducible before microbenchmarks."""
    if seconds <= 0:
        return {"seconds_requested": seconds, "matrix_multiplications": 0}
    left = torch.randn(4096, 4096, device=device)
    right = torch.randn(4096, 4096, device=device)
    deadline = time.monotonic() + seconds
    count = 0
    while time.monotonic() < deadline:
        left = left @ right
        left = left / left.abs().amax().clamp_min(1.0)
        count += 1
    torch.cuda.synchronize()
    del left, right
    return {"seconds_requested": seconds, "matrix_multiplications": count}


def main() -> int:
    global CURRENT_TASK, POWER_DWELL_SECONDS, POWER_SAMPLER
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=1000)
    parser.add_argument("--nvidia-smi-index", type=int, default=6)
    parser.add_argument("--measure-power", action="store_true")
    parser.add_argument("--idle-seconds", type=float, default=5.0)
    parser.add_argument("--power-dwell-seconds", type=float, default=2.0)
    parser.add_argument("--device-warmup-seconds", type=float, default=0.0)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda:0")
    device_name = torch.cuda.get_device_name(device)
    if "A100" not in device_name:
        raise RuntimeError(f"Formal run requires A100; selected device is {device_name}")
    if args.repeats < 100:
        raise ValueError("Formal run requires at least 100 measured calls per component")
    root = Path.cwd().resolve()
    torch.manual_seed(20260914)
    torch.cuda.manual_seed_all(20260914)
    device_conditioning = condition_device_clock(device, args.device_warmup_seconds)

    # Make the legacy exact-shape building blocks emit our paper-grade raw rows.
    legacy.benchmark = measured_benchmark
    legacy.PHYSICAL_PASS_MS = PHYSICAL_PASS_MS
    legacy.GPU_RATED_POWER_W = A100_RATED_POWER_W
    legacy.PoseHead = PoseHeatmapDecoder
    legacy.SaliencyHead = SaliencyDensityDecoder
    legacy.OpenMojiBridge = LatestOpenMojiBridge
    legacy.OpenMojiHead = LatestOpenMojiHead

    samples: list[PowerSample] = []
    sampler = NvidiaSmiPowerSampler(args.nvidia_smi_index, interval_ms=10) if args.measure_power else None
    if sampler is not None:
        POWER_DWELL_SECONDS = args.power_dwell_seconds
        sampler.start()
        sampler.set_phase("idle")
        time.sleep(args.idle_seconds)
        sampler.set_phase(None)
        POWER_SAMPLER = sampler
    try:
        tasks: dict[str, Any] = {}
        plan = [
            ("lgvq_temporal", "t06"), ("lgvq_spatial", "spatial"),
            ("abo_image_to_text", "t08"), ("abo_image_to_image", "t08"),
            ("lsp", "t02"), ("salicon", "t03"), ("openmoji", "t04"),
        ]
        for task_name, legacy_key in plan:
            print(f"[A100] {task_name}", flush=True)
            CURRENT_TASK = task_name
            if legacy_key == "t06":
                task = legacy._t06_task(device, args.warmup, args.repeats)
                _, _, frame_weights, frame_shape = temporal_router_weights(device, frame=True)
                _, _, video_weights, video_shape = temporal_router_weights(device, frame=False)
                task["components"].append(measured_benchmark(
                    "frame_router_ccd_to_weights", frame_weights, warmup=args.warmup, repeats=args.repeats,
                    logical_samples_per_call=16, physical_fields_per_call=1, shape_contract=frame_shape,
                ))
                task["components"].append(measured_benchmark(
                    "video_router_ccd_to_weights", video_weights, warmup=args.warmup, repeats=args.repeats,
                    logical_samples_per_call=16, physical_fields_per_call=1, shape_contract=video_shape,
                ))
                task["formal_cuda_event"] = compose_temporal(task, basis="cuda_event_ms")
                task["synchronized_wall_audit"] = compose_temporal(task, basis="synchronized_wall_ms")
                task["formal_neural_only_cuda_event"] = compose_temporal_neural_only(task, basis="cuda_event_ms")
                task["neural_only_wall_audit"] = compose_temporal_neural_only(task, basis="synchronized_wall_ms")
            elif legacy_key == "spatial":
                task = run_spatial(device, args.warmup, args.repeats)
            else:
                task = run_standard(task_name, legacy_key, device, args.warmup, args.repeats)
            tasks[task_name] = task
    finally:
        POWER_SAMPLER = None
        if sampler is not None:
            samples = sampler.stop()

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    component_rows: list[dict[str, Any]] = []
    for task_name, task in tasks.items():
        for component in task["components"]:
            component_rows.append({
                "task": task_name, "component": component["component"],
                "shape_contract": component["shape_contract"],
                "warmup_calls": component["warmup_calls"], "measured_calls": component["measured_calls"],
                "logical_samples_per_call": component["logical_samples_per_call"],
                "cuda_event_mean_ms": component["cuda_event_ms"]["mean"],
                "cuda_event_median_ms": component["cuda_event_ms"]["median"],
                "cuda_event_p95_ms": component["cuda_event_ms"]["p95"],
                "wall_mean_ms": component["synchronized_wall_ms"]["mean"],
                "wall_median_ms": component["synchronized_wall_ms"]["median"],
                "wall_p95_ms": component["synchronized_wall_ms"]["p95"],
            })
    manifest_rows = artifact_manifest(root)
    report = {
        "schema_version": 2,
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "scope": "post-CCD readout/nonlinearity/fusion/reload, parallel electronic residuals, bridges and task heads; no FFT propagation",
        "formal_latency_basis": "per-component CUDA Event median; synchronized host wall retained only as audit",
        "constants": {
            "slm_refresh_ms": SLM_REFRESH_MS, "ccd_exposure_ms": CCD_EXPOSURE_MS,
            "camera_readout_ms": CAMERA_READOUT_MS, "physical_pass_ms": PHYSICAL_PASS_MS,
            "optical_rig_power_w": OPTICAL_RIG_POWER_W, "a100_rated_power_w": A100_RATED_POWER_W,
        },
        "environment": {
            "python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda, "device": device_name,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "device_total_memory_bytes": torch.cuda.get_device_properties(device).total_memory,
            "precision": "float32", "execution": "eager torch.inference_mode; no torch.compile; synchronize each call",
            "t07_class_source": T07_CLASS_SOURCE,
            "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
        },
        "measurement_counts": {
            "warmup_per_component": args.warmup, "repeats_per_component": args.repeats,
            "raw_timing_rows": len(RAW_ROWS), "device_clock_conditioning": device_conditioning,
        },
        "metric_identities": METRIC_IDENTITIES,
        "artifact_and_source_manifest": manifest_rows,
        "power_measurement": power_summary(samples) if samples else None,
        "tasks": tasks,
        "caveats": [
            "Weights are identity-bound and hashed where present; weight values do not change the executed CUDA kernels.",
            "ABO image-to-text 0.8058 is the documented historical 1934/2400 checkpoint result; the current A100 replay discrepancy of 3 images must remain disclosed.",
            "Physical time is inserted from the user-confirmed 0.714+0.300+0.0307 ms contract and is not GPU-measured.",
        ],
    }
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(output / "per_call_timings.csv", RAW_ROWS)
    write_csv(output / "component_summary.csv", component_rows)
    write_csv(output / "source_and_artifact_sha256.csv", manifest_rows)
    if samples:
        save_power_samples(output / "power_samples.csv", samples)
    (output / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    (output / "environment.txt").write_text(
        subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout
        + "\n\n" + subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout,
        encoding="utf-8",
    )
    checks = []
    for path in sorted(output.iterdir()):
        if path.is_file() and path.name != "SHA256SUMS.txt":
            checks.append(f"{sha256(path)}  {path.name}")
    (output / "SHA256SUMS.txt").write_text("\n".join(checks) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "output_dir": str(output), "tasks": list(tasks)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
