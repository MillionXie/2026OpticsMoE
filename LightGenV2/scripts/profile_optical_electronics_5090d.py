"""Profile the electronic boundaries of the runnable LightGenV2 optical MoE tasks.

The benchmark deliberately excludes angular-spectrum propagation.  It measures
the eager PyTorch operations that remain around a physical SLM/CCD pass:

* optical-router CCD integration, Top-2 selection and expert-field fan-out;
* feature CCD normalization/readout, scale-matched fusion and optional reload;
* the electronic residual route that can run in parallel with the optical pass;
* task-specific readout/decoder heads.

Inputs are synthetic but have the exact formal deployment shapes.  Learned
weight values do not affect latency, so checkpoints are neither required nor
silently substituted.  Each component reports CUDA-event time and synchronized
host wall time; graph-level critical-path estimates use wall-time medians.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import torch
from torch import nn
from torch.nn import functional as F


PHYSICAL_PROPAGATION_MS = 0.714
PHASE_SLM_MS = 0.100
CCD_EXPOSURE_MS = 0.500
PHYSICAL_PASS_MS = PHYSICAL_PROPAGATION_MS + PHASE_SLM_MS + CCD_EXPOSURE_MS
OPTICAL_POWER_W = 80.388
GPU_RATED_POWER_W = 575.0


@dataclass(frozen=True)
class TaskSpec:
    task: str
    label: str
    language: bool
    physical_fields_per_call: int
    logical_samples_per_call: int
    feature_passes: int
    router_passes: int
    vision_tokens: int = 196
    language_tokens: int = 0


SPECS = {
    "t01": TaskSpec("t01", "Caltech101 retrieval", True, 1, 1, 4, 2, 196, 76),
    "t02": TaskSpec("t02", "LSP keypoint", False, 1, 1, 2, 1, 196, 0),
    "t03": TaskSpec("t03", "SALICON saliency", False, 1, 1, 2, 1, 196, 0),
    "t04": TaskSpec("t04", "OpenMoji interaction", True, 1, 1, 4, 2, 196, 64),
    "t06": TaskSpec("t06", "LGVQ MultiVideo-16x4", True, 1, 16, 4, 2, 196, 42),
}


def _git_value(*args: str) -> str | None:
    try:
        return subprocess.run(
            ["git", *args], check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


@torch.inference_mode()
def benchmark(
    name: str,
    function: Callable[[], torch.Tensor | tuple[torch.Tensor, ...]],
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
    cuda_ms: list[float] = []
    wall_ms: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        wall_start = time.perf_counter_ns()
        start.record()
        function()
        end.record()
        end.synchronize()
        wall_ms.append((time.perf_counter_ns() - wall_start) / 1.0e6)
        cuda_ms.append(float(start.elapsed_time(end)))
    return {
        "component": name,
        "shape_contract": shape_contract,
        "warmup_calls": warmup,
        "measured_calls": repeats,
        "logical_samples_per_call": logical_samples_per_call,
        "physical_fields_per_call": physical_fields_per_call,
        "logical_samples_measured": repeats * logical_samples_per_call,
        "physical_fields_measured": repeats * physical_fields_per_call,
        "cuda_event_ms": {
            "mean": statistics.fmean(cuda_ms),
            "median": statistics.median(cuda_ms),
            "p95": _percentile(cuda_ms, 0.95),
            "minimum": min(cuda_ms),
            "maximum": max(cuda_ms),
        },
        "synchronized_wall_ms": {
            "mean": statistics.fmean(wall_ms),
            "median": statistics.median(wall_ms),
            "p95": _percentile(wall_ms, 0.95),
            "minimum": min(wall_ms),
            "maximum": max(wall_ms),
        },
    }


class StandardCCDReadout(nn.Module):
    """478x478 raw intensity -> valid token rows x 192, matching T01--T04."""

    def __init__(self, token_count: int, width: int = 192) -> None:
        super().__init__()
        self.token_count = token_count
        self.pool = nn.AdaptiveAvgPool2d((224, 224))
        self.norm = nn.LayerNorm(224, elementwise_affine=False)
        self.output = nn.Linear(224, width)

    def forward(self, detector: torch.Tensor) -> torch.Tensor:
        value = detector.float().clamp_min(0.0)
        frame_mean = value.mean((-2, -1), keepdim=True).clamp_min(1.0e-6)
        value = torch.log1p((value / frame_mean).clamp_max(12.0))
        pooled = self.pool(value.unsqueeze(1)).squeeze(1)
        rows = F.relu(self.norm(pooled))[:, : self.token_count]
        return self.output(rows)


class ScaleMatchedFusion(nn.Module):
    def __init__(self, alpha: float = 0.50, epsilon: float = 1.0e-6) -> None:
        super().__init__()
        self.alpha = alpha
        self.epsilon = epsilon

    def forward(
        self, electronic: torch.Tensor, optical: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        valid = mask.unsqueeze(-1).float()
        token_axes = tuple(range(1, valid.ndim - 1))
        value_axes = tuple(range(1, electronic.ndim))
        denominator = (
            valid.sum(token_axes).reshape(electronic.shape[0])
            * electronic.shape[-1]
        ).clamp_min(1.0)

        def rms(value: torch.Tensor) -> torch.Tensor:
            squared = (value.float().square() * valid).sum(value_axes)
            shape = (value.shape[0],) + (1,) * (value.ndim - 1)
            return (
                (squared / denominator).sqrt().clamp_min(self.epsilon).reshape(shape)
            )

        re = rms(electronic).detach()
        ro = rms(optical).detach()
        mixture = (1.0 - self.alpha) * electronic.float() / re
        mixture = mixture + self.alpha * optical.float() / ro
        mixture = mixture * valid
        return re * mixture / rms(mixture).detach()


class StandardReload(nn.Module):
    """Fused 192-D tokens -> routed 4-expert amplitude canvas for the next pass."""

    def __init__(self, token_count: int) -> None:
        super().__init__()
        self.token_count = token_count
        self.projection = nn.Linear(192, 224)
        self.norm = nn.LayerNorm(224)
        indices = []
        for top in (20, 274):
            for left in (20, 274):
                yy, xx = torch.meshgrid(
                    torch.arange(top, top + 224),
                    torch.arange(left, left + 224),
                    indexing="ij",
                )
                indices.append((yy * 518 + xx).reshape(-1))
        self.register_buffer("indices", torch.stack(indices), persistent=False)

    def forward(self, tokens: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        projected = F.softplus(self.norm(self.projection(tokens.float())))
        field = projected.new_zeros(tokens.shape[0], 224, 224)
        field[:, : self.token_count] = projected
        field = field / field.square().mean((-2, -1), keepdim=True).sqrt().clamp_min(1.0e-6)
        values = (field[:, None] * weights[:, :, None, None]).reshape(tokens.shape[0], -1)
        canvas = field.new_zeros(tokens.shape[0], 518 * 518)
        canvas.scatter_(1, self.indices.reshape(1, -1).expand(tokens.shape[0], -1), values)
        return canvas.reshape(tokens.shape[0], 518, 518)


class StandardRouterPost(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        masks = torch.zeros(4, 478, 478)
        index = 0
        for top, bottom in ((164, 223), (255, 314)):
            for left, right in ((164, 223), (255, 314)):
                masks[index, top:bottom, left:right] = 1.0
                index += 1
        self.register_buffer("masks", masks, persistent=False)

    def forward(self, detector: torch.Tensor) -> torch.Tensor:
        energy = torch.einsum("bhw,ehw->be", detector, self.masks)
        centered = energy - energy.mean(-1, keepdim=True)
        logits = centered / centered.square().mean(-1, keepdim=True).add(1.0e-8).sqrt()
        probabilities = torch.softmax(logits / 2.0, -1)
        _, indices = torch.topk(probabilities, 2, dim=-1)
        selected = torch.zeros_like(probabilities, dtype=torch.bool).scatter(1, indices, True)
        sparse = probabilities * selected
        return sparse / sparse.square().sum(-1, keepdim=True).sqrt().clamp_min(1.0e-8)


class StandardResidualBlock(nn.Module):
    def __init__(self, *, causal: bool, kernel_size: int) -> None:
        super().__init__()
        self.causal = causal
        self.kernel_size = kernel_size
        self.token_norm = nn.LayerNorm(192)
        if causal:
            self.depthwise = nn.Conv1d(192, 192, kernel_size, groups=192, bias=False)
        else:
            self.depthwise = nn.Conv2d(192, 192, kernel_size, groups=192, bias=False)
        self.pointwise = nn.Linear(192, 192)
        self.token_gate = nn.Parameter(torch.logit(torch.tensor(0.10)))
        self.norm = nn.LayerNorm(192)
        self.mlp = nn.Sequential(nn.Linear(192, 384), nn.GELU(), nn.Linear(384, 192))
        self.mlp_gate = nn.Parameter(torch.logit(torch.tensor(0.10)))

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        token_input = self.token_norm(hidden).masked_fill(~mask.unsqueeze(-1), 0.0)
        if self.causal:
            sequence = F.pad(token_input.transpose(1, 2), (self.kernel_size - 1, 0))
            update = self.depthwise(sequence).transpose(1, 2)
        else:
            batch, count, width = token_input.shape
            if count != 196:
                raise RuntimeError("Formal vision residual expects 196 block-major tokens")
            image = (
                token_input.view(batch, 7, 7, 2, 2, width)
                .permute(0, 5, 1, 3, 2, 4)
                .reshape(batch, width, 14, 14)
            )
            pad = self.kernel_size // 2
            mixed = self.depthwise(F.pad(image, (pad, pad, pad, pad)))
            update = (
                mixed.view(batch, width, 7, 2, 7, 2)
                .permute(0, 2, 4, 3, 5, 1)
                .reshape(batch, count, width)
            )
        hidden = hidden + torch.sigmoid(self.token_gate) * self.pointwise(F.gelu(update))
        hidden = hidden.masked_fill(~mask.unsqueeze(-1), 0.0)
        hidden = hidden + torch.sigmoid(self.mlp_gate) * self.mlp(self.norm(hidden))
        return hidden.masked_fill(~mask.unsqueeze(-1), 0.0)


class RetrievalHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(224)
        self.projection = nn.Linear(224, 64)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.projection(self.norm(value.float())), p=2, dim=-1)


class DepthwiseResidual2D(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        groups = min(8, channels)
        while channels % groups:
            groups -= 1
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.GroupNorm(groups, channels),
            nn.GELU(),
        )
        self.gate = nn.Parameter(torch.logit(torch.tensor(0.10)))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + torch.sigmoid(self.gate) * self.block(value)


class UpsampleBlock(nn.Module):
    def __init__(self, before: int, after: int) -> None:
        super().__init__()
        groups = min(8, after)
        while after % groups:
            groups -= 1
        self.block = nn.Sequential(
            nn.Conv2d(before, before, 3, padding=1, groups=before, bias=False),
            nn.Conv2d(before, after, 1, bias=False),
            nn.GroupNorm(groups, after),
            nn.GELU(),
        )

    def forward(self, value: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        return self.block(F.interpolate(value, size=size, mode="bilinear", align_corners=False))


class ProgressiveHeadBase(nn.Module):
    def __init__(self, projection: int, channels: tuple[int, ...], output_size: int) -> None:
        super().__init__()
        self.output_size = output_size
        self.norm = nn.LayerNorm(192)
        self.projection = nn.Linear(192, projection)
        self.body = nn.Sequential(DepthwiseResidual2D(projection), DepthwiseResidual2D(projection))
        values = (projection, *channels)
        self.upsample = nn.ModuleList(
            UpsampleBlock(before, after) for before, after in zip(values[:-1], values[1:])
        )

    def features(self, spatial: torch.Tensor) -> torch.Tensor:
        value = self.projection(self.norm(spatial.permute(0, 2, 3, 1).float()))
        value = self.body(value.permute(0, 3, 1, 2))
        height, width = value.shape[-2:]
        for block in self.upsample:
            height, width = min(self.output_size, height * 2), min(self.output_size, width * 2)
            value = block(value, (height, width))
        if value.shape[-2:] != (self.output_size, self.output_size):
            value = F.interpolate(value, (self.output_size, self.output_size), mode="bilinear", align_corners=False)
        return value


class PoseHead(ProgressiveHeadBase):
    def __init__(self) -> None:
        super().__init__(160, (128, 96), 56)
        self.refine = DepthwiseResidual2D(96)
        self.predictor = nn.Conv2d(96, 14, 1)

    def forward(self, spatial: torch.Tensor) -> torch.Tensor:
        return self.predictor(self.refine(self.features(spatial)))


class SaliencyHead(ProgressiveHeadBase):
    def __init__(self) -> None:
        super().__init__(128, (96, 64, 32, 16), 224)
        self.refine = DepthwiseResidual2D(16)
        self.predictor = nn.Conv2d(16, 1, 1)

    def forward(self, spatial: torch.Tensor) -> torch.Tensor:
        return self.predictor(self.refine(self.features(spatial)))


class ConditionedResidual(nn.Module):
    def __init__(self, dilation: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(8, 192)
        self.depthwise = nn.Conv2d(192, 192, 3, padding=dilation, dilation=dilation, groups=192, bias=False)
        self.pointwise = nn.Conv2d(192, 192, 1, bias=False)
        self.condition = nn.Linear(192, 384)
        self.gate = nn.Parameter(torch.logit(torch.tensor(0.10)))

    def forward(self, value: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.condition(condition).chunk(2, -1)
        hidden = self.norm(value)
        hidden = hidden * (1.0 + 0.1 * torch.tanh(gamma)[:, :, None, None])
        hidden = hidden + 0.1 * torch.tanh(beta)[:, :, None, None]
        hidden = self.pointwise(F.gelu(self.depthwise(hidden)))
        return value + torch.sigmoid(self.gate) * hidden


class OpenMojiBridge(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.language_pool = nn.Sequential(nn.LayerNorm(384), nn.Linear(384, 192), nn.GELU())
        self.prompt_to_vision = nn.Sequential(nn.LayerNorm(192), nn.Linear(192, 1024))
        self.gate = nn.Parameter(torch.logit(torch.tensor(0.055)))

    def forward(self, language: torch.Tensor, vision: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        condition = self.language_pool(torch.cat((language.mean(1), language.amax(1)), -1))
        bias = torch.tanh(self.prompt_to_vision(condition))
        return condition, vision + torch.sigmoid(self.gate) * bias[:, None]


class OpenMojiHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.post_film = nn.Linear(192, 384)
        self.coordinate_projection = nn.Conv2d(2, 192, 1)
        self.editor = nn.ModuleList(ConditionedResidual(d) for d in (1, 2, 4))
        self.pre = nn.Sequential(
            nn.Conv2d(192, 192, 3, padding=1, groups=192, bias=False),
            nn.Conv2d(192, 192, 1, bias=False),
            nn.GroupNorm(8, 192),
            nn.GELU(),
        )
        self.category = nn.Conv2d(192, 17, 1)
        self.edit = nn.Conv2d(192, 1, 1)
        self.task = nn.Sequential(nn.LayerNorm(192), nn.Linear(192, 4))

    def forward(self, spatial: torch.Tensor, condition: torch.Tensor) -> tuple[torch.Tensor, ...]:
        gamma, beta = self.post_film(condition).chunk(2, -1)
        spatial = spatial * (1.0 + 0.1 * torch.tanh(gamma)[:, :, None, None])
        spatial = spatial + 0.1 * torch.tanh(beta)[:, :, None, None]
        axis = torch.linspace(-1.0, 1.0, spatial.shape[-1], device=spatial.device, dtype=spatial.dtype)
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        coords = torch.stack((xx, yy), 0).unsqueeze(0).expand(len(spatial), -1, -1, -1)
        spatial = spatial + self.coordinate_projection(coords)
        for block in self.editor:
            spatial = block(spatial, condition)
        value = F.adaptive_avg_pool2d(self.pre(spatial), (6, 6))
        return self.category(value), self.edit(value).squeeze(1), self.task(condition)


class T06VisionResidual(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(192)
        self.depthwise = nn.Conv2d(192, 192, 5, padding=2, groups=192, bias=False)
        self.pointwise = nn.Conv2d(192, 192, 1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        batch, frames, tokens, width = value.shape
        image = self.norm(value).reshape(batch * frames, 7, 7, width).permute(0, 3, 1, 2)
        output = self.pointwise(F.gelu(self.depthwise(image)))
        return output.permute(0, 2, 3, 1).reshape(batch, frames, tokens, width)


class T06LanguageResidual(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(192)
        self.depthwise = nn.Conv1d(192, 192, 5, groups=192, bias=False)
        self.pointwise = nn.Conv1d(192, 192, 1)

    def forward(self, value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        sequence = self.norm(value).masked_fill(~mask.unsqueeze(-1), 0.0).transpose(1, 2)
        output = self.pointwise(F.gelu(self.depthwise(F.pad(sequence, (4, 0))))).transpose(1, 2)
        return output.masked_fill(~mask.unsqueeze(-1), 0.0)


def _normalize_patch(value: torch.Tensor, clip: float = 8.0) -> torch.Tensor:
    value = value.float().clamp_min(0.0)
    mean = value.mean((-2, -1), keepdim=True).clamp_min(1.0e-6)
    return torch.log1p((value / mean).clamp_max(clip))


class T06FrameReadout(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((49, 96))
        self.norm = nn.LayerNorm(96)
        self.output = nn.Linear(96, 192)
        self.origins = tuple((3 + 119 * v, 3 + 119 * h) for v in range(4) for h in range(4))
        self.frame_origins = ((0, 0), (0, 59), (59, 0), (59, 59))

    def forward(self, detector: torch.Tensor) -> torch.Tensor:
        patches = []
        for video_top, video_left in self.origins:
            for frame_top, frame_left in self.frame_origins:
                y, x = 20 + video_top + frame_top, 20 + video_left + frame_left
                patches.append(_normalize_patch(detector[:, y : y + 56, x : x + 56]))
        stacked = torch.stack(patches, 1).flatten(0, 1)
        pooled = self.pool(stacked.unsqueeze(1)).squeeze(1)
        value = self.output(F.softplus(self.norm(pooled)))
        return value.reshape(detector.shape[0], 16, 4, 49, 192)


class T06VideoReadout(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(96)
        self.output = nn.Linear(96, 192)
        self.origins = tuple((3 + 119 * v, 3 + 119 * h) for v in range(4) for h in range(4))

    def forward(self, detector: torch.Tensor) -> torch.Tensor:
        patches = []
        for top, left in self.origins:
            y, x = 20 + top, 20 + left
            patches.append(_normalize_patch(detector[:, y : y + 115, x : x + 115]))
        stacked = torch.stack(patches, 1).flatten(0, 1)
        pooled = F.adaptive_avg_pool2d(stacked.unsqueeze(1), (42, 96)).squeeze(1)
        value = self.output(F.softplus(self.norm(pooled)))
        return value.reshape(detector.shape[0], 16, 42, 192)


class T06FrameReload(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.width_to_field = nn.Linear(192, 27)
        self.tokens_to_field = nn.Linear(49, 27)
        self.origins = tuple((3 + 119 * v, 3 + 119 * h) for v in range(4) for h in range(4))
        self.frame_origins = ((0, 0), (0, 59), (59, 0), (59, 59))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        encoded = F.softplus(self.width_to_field(tokens.float()))
        field = F.softplus(self.tokens_to_field(encoded.transpose(-2, -1))).transpose(-2, -1)
        field = field / field.square().mean((-2, -1), keepdim=True).sqrt().clamp_min(1.0e-6)
        canvas = field.new_zeros(field.shape[0], 518, 518)
        for video, (top, left) in enumerate(self.origins):
            for frame, (dy, dx) in enumerate(self.frame_origins):
                y, x = 20 + top + dy + 14, 20 + left + dx + 14
                canvas[:, y : y + 27, x : x + 27] = field[:, video, frame]
        return canvas


class T06VideoReload(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.width_to_field = nn.Linear(192, 56)
        self.origins = tuple((3 + 119 * v, 3 + 119 * h) for v in range(4) for h in range(4))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        encoded = F.softplus(self.width_to_field(tokens.float()))
        field = encoded.new_zeros(tokens.shape[0], 16, 56, 56)
        field[:, :, : tokens.shape[2]] = encoded
        field = field / field.square().mean((-2, -1), keepdim=True).sqrt().clamp_min(1.0e-6)
        canvas = field.new_zeros(tokens.shape[0], 518, 518)
        for video, (top, left) in enumerate(self.origins):
            y, x = 20 + top + 29, 20 + left + 29
            canvas[:, y : y + 56, x : x + 56] = field[:, video]
        return canvas


def _sparse_top2(probabilities: torch.Tensor) -> torch.Tensor:
    _, indices = probabilities.topk(2, dim=-1)
    selected = torch.zeros_like(probabilities, dtype=torch.bool).scatter(-1, indices, True)
    sparse = probabilities * selected
    return sparse / sparse.square().sum(-1, keepdim=True).sqrt().clamp_min(1.0e-8)


class T06FrameRouterPost(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.origins = tuple((3 + 119 * v, 3 + 119 * h) for v in range(4) for h in range(4))
        self.frames = ((0, 0), (0, 59), (59, 0), (59, 59))
        self.intervals = ((12, 26), (30, 44))

    def forward(self, detector: torch.Tensor, fields: torch.Tensor) -> torch.Tensor:
        rows = []
        for top, left in self.origins:
            for dy, dx in self.frames:
                y, x = 20 + top + dy, 20 + left + dx
                tile = detector[:, y : y + 56, x : x + 56]
                rows.append(torch.stack([tile[:, y0:y1, x0:x1].sum((-2, -1)) for y0, y1 in self.intervals for x0, x1 in self.intervals], -1))
        energy = torch.stack(rows, 1).reshape(detector.shape[0], 16, 4, 4)
        centered = energy - energy.mean(-1, keepdim=True)
        probabilities = torch.softmax(centered / centered.square().mean(-1, keepdim=True).add(1.0e-8).sqrt() / 1.1, -1)
        weights = _sparse_top2(probabilities)
        canvas = detector.new_zeros(detector.shape[0], 518, 518)
        for video, (top, left) in enumerate(self.origins):
            for frame, (dy, dx) in enumerate(self.frames):
                expert = 0
                for ey, ex in ((0, 0), (0, 29), (29, 0), (29, 29)):
                    y, x = 20 + top + dy + ey, 20 + left + dx + ex
                    canvas[:, y : y + 27, x : x + 27] = fields[:, video, frame] * weights[:, video, frame, expert, None, None]
                    expert += 1
        return canvas


class T06VideoRouterPost(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.origins = tuple((3 + 119 * v, 3 + 119 * h) for v in range(4) for h in range(4))
        self.intervals = ((23, 50), (65, 92))

    def forward(self, detector: torch.Tensor, fields: torch.Tensor) -> torch.Tensor:
        rows = []
        for top, left in self.origins:
            y, x = 20 + top, 20 + left
            tile = detector[:, y : y + 115, x : x + 115]
            rows.append(torch.stack([tile[:, y0:y1, x0:x1].sum((-2, -1)) for y0, y1 in self.intervals for x0, x1 in self.intervals], -1))
        energy = torch.stack(rows, 1)
        centered = energy - energy.mean(-1, keepdim=True)
        probabilities = torch.softmax(centered / centered.square().mean(-1, keepdim=True).add(1.0e-8).sqrt() / 1.1, -1)
        weights = _sparse_top2(probabilities)
        canvas = detector.new_zeros(detector.shape[0], 518, 518)
        for video, (top, left) in enumerate(self.origins):
            expert = 0
            for ey, ex in ((0, 0), (0, 59), (59, 0), (59, 59)):
                y, x = 20 + top + ey, 20 + left + ex
                canvas[:, y : y + 56, x : x + 56] = fields[:, video] * weights[:, video, expert, None, None]
                expert += 1
        return canvas


class T06Bridge(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.frame_merger = nn.Sequential(nn.LayerNorm(384), nn.Linear(384, 192), nn.GELU())
        self.serial_field = nn.Linear(192, 56)
        self.router_width = nn.Linear(192, 56)
        self.router_frames = nn.Linear(4, 56)
        self.frame_position = nn.Parameter(torch.zeros(1, 1, 4, 192))
        self.sequence_position = nn.Parameter(torch.zeros(1, 64, 192))

    def forward(self, vision: torch.Tensor, prompt: torch.Tensor, prompt_mask: torch.Tensor) -> tuple[torch.Tensor, ...]:
        image = self.frame_merger(torch.cat((vision.mean(3), vision.amax(3)), -1)) + self.frame_position
        prompt16 = prompt[:, None].expand(-1, 16, -1, -1)
        mask = torch.cat((torch.ones(vision.shape[0], 16, 4, dtype=torch.bool, device=vision.device), prompt_mask[:, None].expand(-1, 16, -1)), 2)
        sequence = torch.cat((image, prompt16), 2)
        sequence = (sequence + self.sequence_position[:, None, : sequence.shape[2]]).masked_fill(~mask.unsqueeze(-1), 0.0)
        serial = F.softplus(self.serial_field(sequence.float()))
        video_field = serial.new_zeros(sequence.shape[0], 16, 56, 56)
        video_field[:, :, : sequence.shape[2]] = serial
        router_encoded = F.softplus(self.router_width(image.float()))
        router_field = F.softplus(self.router_frames(router_encoded.transpose(-2, -1))).transpose(-2, -1)
        return sequence, mask, video_field, router_field


def _masked_statistics(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = mask.unsqueeze(-1).to(value.dtype)
    count = valid.sum(1).clamp_min(1.0)
    mean = (value * valid).sum(1) / count
    centered = (value - mean[:, None]) * valid
    std = (centered.square().sum(1) / count).sqrt()
    maximum = value.masked_fill(~mask.unsqueeze(-1), torch.finfo(value.dtype).min).amax(1)
    return torch.cat((mean, std, maximum), -1)


class T06Head(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.frame_norm = nn.LayerNorm(384)
        self.temporal3 = nn.Conv1d(384, 384, 3, padding=1, groups=384, bias=False)
        self.temporal5 = nn.Conv1d(384, 384, 5, padding=2, groups=384, bias=False)
        self.temporal_projection = nn.Conv1d(768, 512, 1)
        self.language = nn.Sequential(nn.LayerNorm(576), nn.Linear(576, 512), nn.GELU())
        self.output = nn.Sequential(nn.LayerNorm(4096), nn.Linear(4096, 1024), nn.GELU(), nn.Dropout(0.1), nn.Linear(1024, 1))

    def forward(self, vision: torch.Tensor, language: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        frame = torch.cat((vision.mean(2), vision.amax(2)), -1)
        normalized = self.frame_norm(frame).transpose(1, 2)
        sequence = self.temporal_projection(torch.cat((F.gelu(self.temporal3(normalized)), F.gelu(self.temporal5(normalized))), 1)).transpose(1, 2)
        difference1 = (sequence[:, 1:] - sequence[:, :-1]).abs()
        difference2 = (sequence[:, 2:] - sequence[:, :-2]).abs()
        summary = torch.cat((sequence.mean(1), sequence.float().std(1, unbiased=False), sequence.amax(1), difference1.mean(1), difference1.amax(1), difference2.mean(1), difference2.amax(1)), -1)
        prompt = self.language(_masked_statistics(language, mask))
        return self.output(torch.cat((summary, prompt), -1)).squeeze(-1)


def _standard_task(task: str, device: torch.device, warmup: int, repeats: int) -> dict[str, Any]:
    spec = SPECS[task]
    detector = torch.rand(1, 478, 478, device=device)
    weights_seed = torch.softmax(torch.rand(1, 4, device=device), -1)
    weights_seed = _sparse_top2(weights_seed)
    results = []

    def measure_modality(label: str, tokens: int, causal: bool, kernel: int) -> dict[str, float]:
        electronic = torch.randn(1, tokens, 192, device=device)
        mask = torch.ones(1, tokens, dtype=torch.bool, device=device)
        readout = StandardCCDReadout(tokens).to(device).eval()
        fusion = ScaleMatchedFusion().to(device).eval()
        reload_layer = StandardReload(tokens).to(device).eval()
        residual = StandardResidualBlock(causal=causal, kernel_size=kernel).to(device).eval()

        def fused_only() -> torch.Tensor:
            return fusion(electronic, readout(detector), mask)

        def fused_reload() -> torch.Tensor:
            return reload_layer(fused_only(), weights_seed)

        results.append(benchmark(f"{label}_ccd_to_fusion", fused_only, warmup=warmup, repeats=repeats, logical_samples_per_call=1, physical_fields_per_call=1, shape_contract=f"CCD [1,478,478] + E [1,{tokens},192] -> fused [1,{tokens},192]"))
        results.append(benchmark(f"{label}_ccd_to_next_slm", fused_reload, warmup=warmup, repeats=repeats, logical_samples_per_call=1, physical_fields_per_call=1, shape_contract=f"CCD -> fused tokens -> routed amplitude [1,518,518]; {tokens} valid rows"))
        results.append(benchmark(f"{label}_parallel_residual", lambda: residual(electronic, mask), warmup=warmup, repeats=repeats, logical_samples_per_call=1, physical_fields_per_call=1, shape_contract=f"E [1,{tokens},192], {'causal Conv1D k5' if causal else 'block-major Conv2D k3'} + MLP 192-384-192"))
        return {
            "fused": results[-3]["synchronized_wall_ms"]["median"],
            "next": results[-2]["synchronized_wall_ms"]["median"],
            "residual": results[-1]["synchronized_wall_ms"]["median"],
        }

    vision = measure_modality("vision", spec.vision_tokens, False, 3)
    language = measure_modality("language", spec.language_tokens, True, 5) if spec.language else None
    router = StandardRouterPost().to(device).eval()
    router_reload = StandardReload(196 if not spec.language else spec.vision_tokens).to(device).eval()
    router_tokens = torch.randn(1, 196 if not spec.language else spec.vision_tokens, 192, device=device)

    def router_to_expert() -> torch.Tensor:
        return router_reload(router_tokens, router(detector))

    results.append(benchmark("router_ccd_to_expert_slm", router_to_expert, warmup=warmup, repeats=repeats, logical_samples_per_call=1, physical_fields_per_call=1, shape_contract="CCD [1,478,478] -> four ROI energies -> Top-2 power-L2 -> [1,518,518] expert amplitude"))
    router_ms = results[-1]["synchronized_wall_ms"]["median"]

    bridge_ms = 0.0
    if task == "t01":
        head = RetrievalHead().to(device).eval()
        feature = torch.rand(1, 224, device=device)
        results.append(benchmark("task_head", lambda: head(feature), warmup=warmup, repeats=repeats, logical_samples_per_call=1, physical_fields_per_call=1, shape_contract="last valid CCD row [1,224] -> LN -> Linear 64 -> L2 embedding"))
    elif task == "t02":
        head = PoseHead().to(device).eval()
        spatial = torch.randn(1, 192, 14, 14, device=device)
        results.append(benchmark("task_head", lambda: head(spatial), warmup=warmup, repeats=repeats, logical_samples_per_call=1, physical_fields_per_call=1, shape_contract="[1,192,14,14] -> progressive decoder -> [1,14,56,56]"))
    elif task == "t03":
        head = SaliencyHead().to(device).eval()
        spatial = torch.randn(1, 192, 14, 14, device=device)
        results.append(benchmark("task_head", lambda: head(spatial), warmup=warmup, repeats=repeats, logical_samples_per_call=1, physical_fields_per_call=1, shape_contract="[1,192,14,14] -> progressive decoder -> [1,1,224,224]"))
    else:
        bridge = OpenMojiBridge().to(device).eval()
        language_value = torch.randn(1, 64, 192, device=device)
        vision_value = torch.randn(1, 196, 1024, device=device)
        condition, _ = bridge(language_value, vision_value)
        results.append(benchmark("language_to_vision_bridge", lambda: bridge(language_value, vision_value), warmup=warmup, repeats=repeats, logical_samples_per_call=1, physical_fields_per_call=1, shape_contract="Language [1,64,192] mean/max -> condition [1,192] -> visual bias [1,196,1024]"))
        bridge_ms = results[-1]["synchronized_wall_ms"]["median"]
        head = OpenMojiHead().to(device).eval()
        spatial = torch.randn(1, 192, 14, 14, device=device)
        results.append(benchmark("task_head", lambda: head(spatial, condition), warmup=warmup, repeats=repeats, logical_samples_per_call=1, physical_fields_per_call=1, shape_contract="[1,192,14,14]+condition -> 3 residual blocks -> category [1,17,6,6], edit [1,6,6], task [1,4]"))

    head_ms = results[-1]["synchronized_wall_ms"]["median"]
    physical = PHYSICAL_PASS_MS
    critical = spec.router_passes * (physical + router_ms)
    critical += max(physical, vision["residual"]) + vision["next"]
    critical += max(physical, vision["residual"]) + vision["fused"]
    if language is not None:
        critical += max(physical, language["residual"]) + language["next"]
        critical += max(physical, language["residual"]) + language["fused"]
    critical += bridge_ms + head_ms
    all_residual_covered = vision["residual"] <= physical and (language is None or language["residual"] <= physical)
    return {
        "specification": asdict(spec),
        "components": results,
        "critical_path": {
            "physical_pass_ms": physical,
            "all_parallel_residuals_covered": all_residual_covered,
            "estimated_wall_ms_per_call": critical,
            "logical_samples_per_call": spec.logical_samples_per_call,
            "estimated_wall_ms_per_logical_sample": critical / spec.logical_samples_per_call,
            "optical_rig_wall_energy_proxy_j_per_call": OPTICAL_POWER_W * critical / 1000.0,
            "physical_only_optical_energy_j_per_call": OPTICAL_POWER_W * (spec.feature_passes + spec.router_passes) * physical / 1000.0,
            "legacy_six_pass_9p084ms_energy_j": OPTICAL_POWER_W * 9.084 / 1000.0 if spec.feature_passes + spec.router_passes == 6 else None,
            "method": "router passes: physical+router post; feature passes: max(physical,residual)+serialized CCD/fusion/reload; then bridge and task head",
        },
    }


def _t06_task(device: torch.device, warmup: int, repeats: int) -> dict[str, Any]:
    spec = SPECS["t06"]
    detector = torch.rand(1, 478, 478, device=device)
    vision = torch.randn(1, 16, 4, 49, 192, device=device)
    sequence = torch.randn(1, 16, 42, 192, device=device)
    sequence_mask = torch.ones(1, 16, 42, dtype=torch.bool, device=device)
    prompt = torch.randn(1, 38, 192, device=device)
    prompt_mask = torch.ones(1, 38, dtype=torch.bool, device=device)
    frame_readout = T06FrameReadout().to(device).eval()
    video_readout = T06VideoReadout().to(device).eval()
    fusion = ScaleMatchedFusion().to(device).eval()
    frame_reload = T06FrameReload().to(device).eval()
    video_reload = T06VideoReload().to(device).eval()
    vision_residual = T06VisionResidual().to(device).eval()
    language_residual = T06LanguageResidual().to(device).eval()
    frame_router = T06FrameRouterPost().to(device).eval()
    video_router = T06VideoRouterPost().to(device).eval()
    bridge = T06Bridge().to(device).eval()
    head = T06Head().to(device).eval()
    frame_fields = torch.rand(1, 16, 4, 27, 27, device=device)
    video_fields = torch.rand(1, 16, 56, 56, device=device)
    results = []

    flat_vision_mask = torch.ones(16, 4, 49, dtype=torch.bool, device=device)
    flat_sequence_mask = sequence_mask.flatten(0, 1)

    def frame_fused() -> torch.Tensor:
        optical = frame_readout(detector)
        return fusion(vision.flatten(0, 1), optical.flatten(0, 1), flat_vision_mask).reshape_as(vision)

    def frame_next() -> torch.Tensor:
        return frame_reload(frame_fused())

    def video_fused() -> torch.Tensor:
        optical = video_readout(detector)
        return fusion(sequence.flatten(0, 1), optical.flatten(0, 1), flat_sequence_mask).reshape_as(sequence)

    def video_next() -> torch.Tensor:
        return video_reload(video_fused())

    calls = [
        ("frame_router_ccd_to_expert_slm", lambda: frame_router(detector, frame_fields), "CCD [1,478,478] -> 64x4 energies/Top-2 -> frame expert amplitude [1,518,518]"),
        ("frame_ccd_to_fusion", frame_fused, "CCD -> 64 lanes -> [1,16,4,49,192] -> RMS fusion"),
        ("frame_ccd_to_next_slm", frame_next, "frame CCD/fusion -> 64 encoded 27x27 fields -> [1,518,518]"),
        ("frame_parallel_residual", lambda: vision_residual(vision.flatten(0, 1)), "16 videos x 4 frames x 49 tokens x 192; Conv2D k5"),
        ("frame_to_video_bridge", lambda: bridge(vision, prompt, prompt_mask), "[1,16,4,49,192]+38 prompt tokens -> 16 sequences of 42 tokens and router/feature fields"),
        ("video_router_ccd_to_expert_slm", lambda: video_router(detector, video_fields), "CCD [1,478,478] -> 16x4 energies/Top-2 -> video expert amplitude [1,518,518]"),
        ("video_ccd_to_fusion", video_fused, "CCD -> 16 macro tiles -> [1,16,42,192] -> RMS fusion"),
        ("video_ccd_to_next_slm", video_next, "video CCD/fusion -> 16 encoded 56x56 fields -> [1,518,518]"),
        ("video_parallel_residual", lambda: language_residual(sequence.flatten(0, 1), flat_sequence_mask), "16 videos x 42 sequence tokens x 192; causal Conv1D k5"),
        ("task_head", lambda: head(vision.flatten(0, 1), sequence.flatten(0, 1), flat_sequence_mask), "16 videos: vision [16,4,49,192]+sequence [16,42,192] -> 16 MOS"),
    ]
    for name, function, shape in calls:
        results.append(benchmark(name, function, warmup=warmup, repeats=repeats, logical_samples_per_call=16, physical_fields_per_call=1, shape_contract=shape))
    times = {row["component"]: row["synchronized_wall_ms"]["median"] for row in results}
    physical = PHYSICAL_PASS_MS
    critical = physical + times["frame_router_ccd_to_expert_slm"]
    critical += max(physical, times["frame_parallel_residual"]) + times["frame_ccd_to_next_slm"]
    critical += max(physical, times["frame_parallel_residual"]) + times["frame_ccd_to_fusion"]
    critical += times["frame_to_video_bridge"]
    critical += physical + times["video_router_ccd_to_expert_slm"]
    critical += max(physical, times["video_parallel_residual"]) + times["video_ccd_to_next_slm"]
    critical += max(physical, times["video_parallel_residual"]) + times["video_ccd_to_fusion"]
    critical += times["task_head"]
    covered = times["frame_parallel_residual"] <= physical and times["video_parallel_residual"] <= physical
    return {
        "specification": asdict(spec),
        "components": results,
        "critical_path": {
            "physical_pass_ms": physical,
            "all_parallel_residuals_covered": covered,
            "estimated_wall_ms_per_call": critical,
            "logical_samples_per_call": 16,
            "estimated_wall_ms_per_logical_sample": critical / 16.0,
            "optical_rig_wall_energy_proxy_j_per_call": OPTICAL_POWER_W * critical / 1000.0,
            "physical_only_optical_energy_j_per_call": OPTICAL_POWER_W * 6 * physical / 1000.0,
            "legacy_six_pass_9p084ms_energy_j": OPTICAL_POWER_W * 9.084 / 1000.0,
            "method": "exact six-pass 16-video graph: two optical routers, four feature stages, frame-to-video bridge and TemporalReadout",
        },
    }


def _environment(device: torch.device) -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(device)
    return {
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": torch.cuda.get_device_name(device),
        "device_total_memory_bytes": properties.total_memory,
        "gpu_rated_power_w": GPU_RATED_POWER_W,
        "optical_power_w": OPTICAL_POWER_W,
        "git_commit": _git_value("rev-parse", "HEAD"),
        "git_status_porcelain": _git_value("status", "--porcelain"),
        "precision": "float32",
        "execution": "eager torch.inference_mode; no torch.compile; synchronized each measured call",
    }


def write_csv(path: Path, report: dict[str, Any]) -> None:
    rows = []
    for task, payload in report["tasks"].items():
        for component in payload["components"]:
            rows.append({
                "task": task,
                "task_label": payload["specification"]["label"],
                "component": component["component"],
                "shape_contract": component["shape_contract"],
                "warmup_calls": component["warmup_calls"],
                "measured_calls": component["measured_calls"],
                "logical_samples_measured": component["logical_samples_measured"],
                "physical_fields_measured": component["physical_fields_measured"],
                "cuda_median_ms": component["cuda_event_ms"]["median"],
                "cuda_p95_ms": component["cuda_event_ms"]["p95"],
                "wall_median_ms": component["synchronized_wall_ms"]["median"],
                "wall_p95_ms": component["synchronized_wall_ms"]["p95"],
            })
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", choices=sorted(SPECS), default=sorted(SPECS))
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=1000)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the formal RTX 5090 D benchmark")
    if args.warmup < 0 or args.repeats < 20:
        raise ValueError("warmup must be nonnegative and repeats must be at least 20")
    device = torch.device("cuda:0")
    torch.manual_seed(20260907)
    torch.cuda.manual_seed_all(20260907)
    tasks = {}
    for task in args.tasks:
        print(f"[profile] {task}: {SPECS[task].label}", flush=True)
        tasks[task] = _t06_task(device, args.warmup, args.repeats) if task == "t06" else _standard_task(task, device, args.warmup, args.repeats)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 1,
        "benchmark_scope": "post-CCD electronics, parallel residual routes, inter-stage reloads, task heads; optical propagation excluded and inserted from measured constants",
        "constants": {
            "propagation_ms_per_pass": PHYSICAL_PROPAGATION_MS,
            "phase_slm_ms_per_pass": PHASE_SLM_MS,
            "ccd_exposure_ms_per_pass": CCD_EXPOSURE_MS,
            "physical_pass_ms": PHYSICAL_PASS_MS,
            "optical_power_w": OPTICAL_POWER_W,
            "gpu_rated_power_w": GPU_RATED_POWER_W,
        },
        "environment": _environment(device),
        "tasks": tasks,
        "not_measured": {
            "t05": "planned task; no runnable formal optical MoE graph",
            "t07": "placeholder/migration task; no runnable formal optical MoE graph",
            "t08": "placeholder/migration task; no runnable formal optical MoE graph",
        },
    }
    json_path = output / "optical_moe_electronics_5090d.json"
    csv_path = output / "optical_moe_electronics_5090d.csv"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(csv_path, report)
    manifest = {
        "schema_version": 1,
        "files": [
            {"path": json_path.name, "sha256": _file_sha256(json_path), "bytes": json_path.stat().st_size},
            {"path": csv_path.name, "sha256": _file_sha256(csv_path), "bytes": csv_path.stat().st_size},
        ],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "output_dir": str(output), "tasks": list(tasks)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
