"""Exact six-pass simulation-to-hardware contract for the single-metric model.

The model owns four optical/electronic fusion stages.  Both expert stages use
an optical energy router first, therefore one physical inference uses six CCD
captures in the order declared by :data:`OPTICAL_PASSES`.  This module keeps
the numerical forward path and the measured-CCD replacement path in one place
so laboratory fine-tuning cannot silently use the older four-frame model.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping

import torch
from torch.nn import functional as F

from .modeling import LGVQSingleMetricOEO16, _alignment, _routing_statistics, _sparse_top2


OPTICAL_PASSES = (
    "vision_router",
    "vision_expert",
    "vision_global",
    "language_router",
    "language_expert",
    "language_global",
)

PASS_DIRECTORIES = {
    name: f"{index:02d}_{name}" for index, name in enumerate(OPTICAL_PASSES, 1)
}

FUSION_STAGES = {
    "vision_expert": ("vision_router", "vision_expert"),
    "vision_global": ("vision_router", "vision_expert", "vision_global"),
    "language_expert": (
        "vision_router",
        "vision_expert",
        "vision_global",
        "language_router",
        "language_expert",
    ),
    "language_global": OPTICAL_PASSES,
}


def _active(value: torch.Tensor, model: LGVQSingleMetricOEO16) -> torch.Tensor:
    margin = model.settings.geometry.active_margin
    size = model.settings.geometry.active_size
    return value[..., margin : margin + size, margin : margin + size]


def parallel_center_canvas(
    fields: torch.Tensor, model: LGVQSingleMetricOEO16
) -> torch.Tensor:
    geometry = model.settings.geometry
    margin = geometry.active_margin
    local = (geometry.lane_size - geometry.parallel_expert_size) // 2
    canvas = fields.new_zeros(fields.shape[0], geometry.canvas_size, geometry.canvas_size)
    for lane, (top, left) in enumerate(geometry.lane_origins):
        y, x = margin + top + local, margin + left + local
        size = geometry.parallel_expert_size
        canvas[:, y : y + size, x : x + size] = fields[:, lane]
    return _active(canvas, model)


def parallel_expert_canvas(
    fields: torch.Tensor, weights: torch.Tensor, model: LGVQSingleMetricOEO16
) -> torch.Tensor:
    geometry = model.settings.geometry
    margin = geometry.active_margin
    size = geometry.parallel_expert_size
    canvas = fields.new_zeros(fields.shape[0], geometry.canvas_size, geometry.canvas_size)
    for lane, (lane_top, lane_left) in enumerate(geometry.lane_origins):
        expert = 0
        for local_top, local_left in geometry.parallel_expert_origins:
            y, x = margin + lane_top + local_top, margin + lane_left + local_left
            canvas[:, y : y + size, x : x + size] = (
                fields[:, lane] * weights[:, lane, expert, None, None]
            )
            expert += 1
    return _active(canvas, model)


def serial_center_canvas(
    field: torch.Tensor,
    model: LGVQSingleMetricOEO16,
    *,
    router: bool = False,
    token_count: int | None = None,
) -> torch.Tensor:
    geometry = model.settings.geometry
    size = geometry.serial_expert_size
    value = field
    target = size
    if router and model.settings.serial_router_input_size != size:
        if token_count is None or not 0 < token_count <= size:
            raise ValueError("Compact serial router requires the valid token count")
        target = model.settings.serial_router_input_size
        value = F.adaptive_avg_pool2d(
            field[:, :token_count].unsqueeze(1), (target, target)
        ).squeeze(1)
    offset = (geometry.canvas_size - target) // 2
    canvas = F.pad(value, (offset, geometry.canvas_size - target - offset) * 2)
    return _active(canvas, model)


def serial_expert_canvas(
    field: torch.Tensor, weights: torch.Tensor, model: LGVQSingleMetricOEO16
) -> torch.Tensor:
    geometry = model.settings.geometry
    margin = geometry.active_margin
    size = geometry.serial_expert_size
    canvas = field.new_zeros(field.shape[0], geometry.canvas_size, geometry.canvas_size)
    for expert, (top, left) in enumerate(geometry.serial_expert_origins):
        y, x = margin + top, margin + left
        canvas[:, y : y + size, x : x + size] = (
            field * weights[:, expert, None, None]
        )
    return _active(canvas, model)


def _router_result(
    probabilities: torch.Tensor,
    energy: torch.Tensor,
    total: torch.Tensor,
    implementation: str,
) -> dict[str, Any]:
    weights, selected, indices = _sparse_top2(probabilities)
    return {
        "probabilities": probabilities,
        "weights": weights,
        "selected_mask": selected,
        "selected_indices": indices,
        "capture_fraction": energy.sum(-1) / total.clamp_min(1.0e-8),
        "router_implementation": implementation,
        **_routing_statistics(probabilities, selected),
    }


def parallel_router_from_ccd(
    active: torch.Tensor, model: LGVQSingleMetricOEO16
) -> dict[str, Any]:
    geometry = model.settings.geometry
    expected = (geometry.active_size, geometry.active_size)
    if active.ndim != 3 or tuple(active.shape[-2:]) != expected:
        raise ValueError(f"Parallel router CCD must be [B,{expected[0]},{expected[1]}]")
    rows, totals = [], []
    for top, left in geometry.lane_origins:
        lane = active[:, top : top + geometry.lane_size, left : left + geometry.lane_size]
        totals.append(lane.sum((-2, -1)))
        rows.append(
            torch.stack(
                [
                    lane[:, y0:y1, x0:x1].sum((-2, -1))
                    for y0, y1 in model.settings.parallel_router_intervals
                    for x0, x1 in model.settings.parallel_router_intervals
                ],
                -1,
            )
        )
    energy = torch.stack(rows, 1)
    centered = energy - energy.mean(-1, keepdim=True)
    logits = centered / centered.square().mean(-1, keepdim=True).add(1.0e-8).sqrt()
    probabilities = torch.softmax(logits / model.settings.router_temperature, -1)
    return _router_result(
        probabilities,
        energy,
        torch.stack(totals, 1),
        f"measured_optical_parallel{model.settings.frame_count}_energy_top2",
    )


def serial_router_from_ccd(
    active: torch.Tensor, model: LGVQSingleMetricOEO16
) -> dict[str, Any]:
    geometry = model.settings.geometry
    expected = (geometry.active_size, geometry.active_size)
    if active.ndim != 3 or tuple(active.shape[-2:]) != expected:
        raise ValueError(f"Serial router CCD must be [B,{expected[0]},{expected[1]}]")
    energy = torch.stack(
        [
            active[:, y0:y1, x0:x1].sum((-2, -1))
            for y0, y1 in model.settings.serial_router_intervals
            for x0, x1 in model.settings.serial_router_intervals
        ],
        -1,
    )
    signal = energy
    if model.serial_router.channel_standardizer is not None:
        signal = model.serial_router.channel_standardizer(torch.log(energy.clamp_min(1.0e-8)))
    centered = signal - signal.mean(-1, keepdim=True)
    logits = centered / centered.square().mean(-1, keepdim=True).add(1.0e-8).sqrt()
    probabilities = torch.softmax(logits / model.settings.router_temperature, -1)
    return _router_result(
        probabilities,
        energy,
        active.sum((-2, -1)),
        "measured_optical_serial_energy_top2",
    )


def parallel_features_from_ccd(
    active: torch.Tensor, model: LGVQSingleMetricOEO16, *, stage: str
) -> torch.Tensor:
    if stage not in {"vision_expert", "vision_global"}:
        raise ValueError("Invalid parallel readout stage")
    geometry = model.settings.geometry
    lanes = [
        model.parallel_optics._normalize(
            active[:, top : top + geometry.lane_size, left : left + geometry.lane_size]
        )
        for top, left in geometry.lane_origins
    ]
    stacked = torch.stack(lanes, 1)
    readout = (
        model.parallel_optics.expert_readout
        if stage == "vision_expert"
        else model.parallel_optics.global_readout
    )
    output = readout(stacked.flatten(0, 1))
    return output.reshape(
        active.shape[0],
        model.settings.frame_count,
        model.settings.token_count,
        model.settings.model_width,
    )


def serial_features_from_ccd(
    active: torch.Tensor,
    model: LGVQSingleMetricOEO16,
    *,
    stage: str,
    token_count: int,
) -> torch.Tensor:
    if stage not in {"language_expert", "language_global"}:
        raise ValueError("Invalid serial readout stage")
    value = active.float().clamp_min(0.0)
    mean = value.mean((-2, -1), keepdim=True).clamp_min(1.0e-6)
    normalized = torch.log1p(
        model.settings.ccd_log_compression
        * (value / mean).clamp_max(model.settings.ccd_relative_clip)
    )
    readout = (
        model.serial_optics.expert_readout
        if stage == "language_expert"
        else model.serial_optics.global_readout
    )
    pooled = F.adaptive_avg_pool2d(
        normalized.unsqueeze(1),
        (token_count, model.settings.detector_projection_size),
    ).squeeze(1)
    return readout.output(F.softplus(readout.norm(pooled)))


MeasurementLoader = Callable[[str, list[str], torch.device], torch.Tensor]


@dataclass
class HardwareForward:
    prediction: torch.Tensor | None
    normalized_prediction: torch.Tensor | None
    amplitudes: OrderedDict[str, torch.Tensor]
    routing: dict[str, dict[str, Any]]
    optical_alignment_loss: torch.Tensor | None


def forward_hardware(
    model: LGVQSingleMetricOEO16,
    batch: Mapping[str, Any],
    sample_keys: Iterable[str],
    *,
    measured_passes: Iterable[str] = (),
    measurement_loader: MeasurementLoader | None = None,
    stop_before: str | None = None,
) -> HardwareForward:
    """Forward with a contiguous simulated/measured optical-pass prefix."""

    keys = [str(value) for value in sample_keys]
    selected = tuple(measured_passes)
    if selected != OPTICAL_PASSES[: len(selected)]:
        raise ValueError("Measured optical passes must be a contiguous prefix")
    if stop_before is not None and stop_before not in OPTICAL_PASSES:
        raise ValueError(f"Unknown optical pass {stop_before!r}")
    if selected and measurement_loader is None:
        raise ValueError("Measured passes require a CCD loader")
    device = next(model.parameters()).device

    def tensor(name: str, *, optional: bool = False) -> torch.Tensor | None:
        value = batch.get(name)
        if value is None:
            if optional:
                return None
            raise KeyError(name)
        return value.to(device, non_blocking=True)

    def measured(name: str) -> torch.Tensor:
        assert measurement_loader is not None
        value = measurement_loader(name, keys, device).float()
        size = model.settings.geometry.active_size
        if tuple(value.shape) != (len(keys), size, size):
            raise ValueError(f"CCD batch for {name} has shape {tuple(value.shape)}")
        return value

    vision_tokens = tensor("vision_tokens")
    quality_tokens = tensor("quality_tokens")
    language_tokens = tensor("language_tokens")
    language_mask = tensor("language_mask").bool()
    raw_frames = tensor("raw_frames", optional=True)
    vgg_tokens = tensor("vgg_tokens", optional=True)
    if model.frame_stem is not None:
        if raw_frames is None:
            raise ValueError("Trainable frame stem requires raw frames")
        quality_tokens = model.frame_stem(raw_frames).to(torch.float16).float()
    prompt_tokens = model.language_adapter(language_tokens.float())
    prompt_valid = language_mask.unsqueeze(-1)
    prompt_summary = (prompt_tokens * prompt_valid).sum(1) / prompt_valid.sum(1).clamp_min(1)
    prompt_scale, prompt_shift = model.prompt_to_visual(prompt_summary).chunk(2, -1)
    qwen_vision = model.vision_adapter(vision_tokens.float())
    quality = model.quality_refiner(model.quality_adapter(quality_tokens.float()))
    raw_qwen_gate = getattr(model, "raw_qwen_gate", None)
    qwen_gate = qwen_vision.new_ones(()) if raw_qwen_gate is None else torch.sigmoid(raw_qwen_gate)
    vision = qwen_gate * qwen_vision + torch.sigmoid(model.raw_quality_gate) * quality
    vgg_correction = torch.zeros_like(qwen_vision)
    if model.vgg_correction is not None:
        if vgg_tokens is None:
            raise ValueError("This checkpoint requires VGG tokens")
        vgg_correction = model.vgg_correction(vgg_tokens.float())
        vision = vision + vgg_correction
    vision = model.visual_input_norm(
        vision * (1.0 + 0.10 * torch.tanh(prompt_scale[:, None, None]))
        + 0.10 * prompt_shift[:, None, None]
    )
    pre_optical_vision = vision
    amplitudes: OrderedDict[str, torch.Tensor] = OrderedDict()
    routing: dict[str, dict[str, Any]] = {}
    alignments: list[torch.Tensor] = []

    fields1 = model.parallel_optics.fields(vision)
    electronic1 = model.vision_routes[0](vision)
    amplitudes["vision_router"] = parallel_center_canvas(fields1, model)
    if stop_before == "vision_router":
        return HardwareForward(None, None, amplitudes, routing, None)
    routing["vision"] = (
        parallel_router_from_ccd(measured("vision_router"), model)
        if "vision_router" in selected
        else model.parallel_router(fields1)
    )
    amplitudes["vision_expert"] = parallel_expert_canvas(
        fields1, routing["vision"]["weights"], model
    )
    if stop_before == "vision_expert":
        return HardwareForward(None, None, amplitudes, routing, None)
    optical1 = (
        parallel_features_from_ccd(measured("vision_expert"), model, stage="vision_expert")
        if "vision_expert" in selected
        else model.parallel_optics.expert(fields1, routing["vision"]["weights"])
    )
    vision = model.fusions[0](electronic1, optical1)
    alignments.append(_alignment(electronic1, optical1))

    fields2 = model.parallel_optics.fields(vision)
    electronic2 = model.vision_routes[1](vision)
    amplitudes["vision_global"] = parallel_center_canvas(fields2, model)
    if stop_before == "vision_global":
        return HardwareForward(None, None, amplitudes, routing, torch.stack(alignments).mean())
    optical2 = (
        parallel_features_from_ccd(measured("vision_global"), model, stage="vision_global")
        if "vision_global" in selected
        else model.parallel_optics.global_path(fields2)
    )
    vision = model.fusions[1](electronic2, optical2)
    alignments.append(_alignment(electronic2, optical2))

    image_tokens = model.frame_merger(torch.cat((vision.mean(2), vision.amax(2)), -1)) + model.frame_position
    sequence = torch.cat((image_tokens, prompt_tokens), 1)
    mask = torch.cat(
        (torch.ones(sequence.shape[0], model.settings.frame_count, dtype=torch.bool, device=device), language_mask),
        1,
    )
    token_count = sequence.shape[1]
    sequence = (sequence + model.sequence_position[:, :token_count]).masked_fill(~mask.unsqueeze(-1), 0.0)

    fields3 = model.serial_optics.fields(sequence)
    electronic3 = model.language_routes[0](sequence, mask)
    amplitudes["language_router"] = serial_center_canvas(
        fields3, model, router=True, token_count=token_count
    )
    if stop_before == "language_router":
        return HardwareForward(None, None, amplitudes, routing, torch.stack(alignments).mean())
    routing["language"] = (
        serial_router_from_ccd(measured("language_router"), model)
        if "language_router" in selected
        else model.serial_router(fields3, token_count)
    )
    amplitudes["language_expert"] = serial_expert_canvas(
        fields3, routing["language"]["weights"], model
    )
    if stop_before == "language_expert":
        return HardwareForward(None, None, amplitudes, routing, torch.stack(alignments).mean())
    optical3 = (
        serial_features_from_ccd(measured("language_expert"), model, stage="language_expert", token_count=token_count)
        if "language_expert" in selected
        else model.serial_optics.expert(fields3, routing["language"]["weights"], token_count)
    )
    sequence = model.fusions[2](electronic3, optical3, mask)
    alignments.append(_alignment(electronic3, optical3))

    fields4 = model.serial_optics.fields(sequence)
    electronic4 = model.language_routes[1](sequence, mask)
    amplitudes["language_global"] = serial_center_canvas(fields4, model)
    if stop_before == "language_global":
        return HardwareForward(None, None, amplitudes, routing, torch.stack(alignments).mean())
    optical4 = (
        serial_features_from_ccd(measured("language_global"), model, stage="language_global", token_count=token_count)
        if "language_global" in selected
        else model.serial_optics.global_path(fields4, token_count)
    )
    sequence = model.fusions[3](electronic4, optical4, mask)
    alignments.append(_alignment(electronic4, optical4))
    normalized = model.readout(vision, sequence, mask)
    if model.late_input_correction is not None:
        normalized = normalized + model.late_input_correction(pre_optical_vision)
    prediction = normalized * model.target_std + model.target_mean
    return HardwareForward(
        prediction,
        normalized,
        amplitudes,
        routing,
        torch.stack(alignments).mean(),
    )


__all__ = [
    "FUSION_STAGES",
    "HardwareForward",
    "OPTICAL_PASSES",
    "PASS_DIRECTORIES",
    "forward_hardware",
    "parallel_features_from_ccd",
    "parallel_router_from_ccd",
    "serial_features_from_ccd",
    "serial_router_from_ccd",
]
