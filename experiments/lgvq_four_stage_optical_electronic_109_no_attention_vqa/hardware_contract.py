"""Exact simulation-to-hardware contract for the six physical optical passes.

The learned network has four optical/electronic fusion stages, but stage 1 and
stage 3 each need a separate optical router exposure before their expert
exposure.  Consequently one end-to-end hardware run contains six SLM/CCD
passes in the fixed order declared by :data:`OPTICAL_PASSES`.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping

import torch
from torch.nn import functional as F

from .modeling import (
    LGVQFourStageOEO,
    _alignment,
    _phase,
    _routing_statistics,
    _sparse_top2,
    deterministic_bridge,
)


OPTICAL_PASSES = (
    "stage1_router",
    "stage1_expert",
    "stage2_global",
    "stage3_router",
    "stage3_expert",
    "stage4_global",
)

PASS_DIRECTORIES = {
    name: f"{index:02d}_{name}"
    for index, name in enumerate(OPTICAL_PASSES, 1)
}

FUSION_STAGES = {
    "stage1": ("stage1_router", "stage1_expert"),
    "stage2": ("stage1_router", "stage1_expert", "stage2_global"),
    "stage3": (
        "stage1_router",
        "stage1_expert",
        "stage2_global",
        "stage3_router",
        "stage3_expert",
    ),
    "stage4": OPTICAL_PASSES,
}


def _zeros(model: LGVQFourStageOEO) -> torch.Tensor:
    geometry = model.settings.geometry
    reference = next(model.parameters())
    return reference.new_zeros(geometry.canvas_size, geometry.canvas_size)


def _active_crop(value: torch.Tensor, model: LGVQFourStageOEO) -> torch.Tensor:
    geometry = model.settings.geometry
    margin = geometry.active_margin
    return value[
        ..., margin : margin + geometry.active_size,
        margin : margin + geometry.active_size,
    ]


def phase_canvases(model: LGVQFourStageOEO) -> OrderedDict[str, torch.Tensor]:
    """Return six logical 478x478 physical-aperture phase planes.

    The simulator uses a 20-pixel zero-padding margin around this aperture for
    the 518-point FFT.  That numerical margin is not sent to either SLM.
    """

    geometry = model.settings.geometry
    margin = geometry.active_margin
    local = (geometry.quadrant_size - geometry.expert_size) // 2
    result: OrderedDict[str, torch.Tensor] = OrderedDict()

    parallel_router = _zeros(model)
    router_phase = _phase(model.parallel_router.raw_router_phase.detach())
    for lane, (top, left) in enumerate(geometry.lane_origins):
        y, x = margin + top + local, margin + left + local
        parallel_router[y : y + geometry.expert_size, x : x + geometry.expert_size] = router_phase[lane]
    result["stage1_router"] = parallel_router

    parallel_expert = _zeros(model)
    expert_phase = _phase(model.parallel_optics.raw_expert_phase.detach())
    index = 0
    for lane_top, lane_left in geometry.lane_origins:
        for local_top in (0, geometry.expert_pitch):
            for local_left in (0, geometry.expert_pitch):
                y, x = margin + lane_top + local_top, margin + lane_left + local_left
                parallel_expert[y : y + geometry.expert_size, x : x + geometry.expert_size] = expert_phase[index]
                index += 1
    result["stage1_expert"] = parallel_expert

    parallel_global = _zeros(model)
    parallel_global[
        margin : margin + geometry.active_size,
        margin : margin + geometry.active_size,
    ] = _phase(model.parallel_optics.raw_global_phase.detach())
    result["stage2_global"] = parallel_global

    serial_router = _zeros(model)
    centered = (geometry.canvas_size - geometry.expert_size) // 2
    serial_router[
        centered : centered + geometry.expert_size,
        centered : centered + geometry.expert_size,
    ] = _phase(model.serial_router.raw_router_phase.detach())
    result["stage3_router"] = serial_router

    serial_expert = _zeros(model)
    serial_expert_phase = _phase(model.serial_optics.raw_expert_phase.detach())
    index = 0
    positions = (geometry.expert_pitch, 2 * geometry.expert_pitch)
    for top in positions:
        for left in positions:
            y, x = margin + top, margin + left
            serial_expert[y : y + geometry.expert_size, x : x + geometry.expert_size] = serial_expert_phase[index]
            index += 1
    result["stage3_expert"] = serial_expert

    serial_global = _zeros(model)
    serial_global[
        margin : margin + geometry.active_size,
        margin : margin + geometry.active_size,
    ] = _phase(model.serial_optics.raw_global_phase.detach())
    result["stage4_global"] = serial_global
    return OrderedDict((name, _active_crop(value, model)) for name, value in result.items())


def parallel_center_canvas(fields: torch.Tensor, model: LGVQFourStageOEO) -> torch.Tensor:
    geometry = model.settings.geometry
    margin = geometry.active_margin
    local = (geometry.quadrant_size - geometry.expert_size) // 2
    canvas = fields.new_zeros(fields.shape[0], geometry.canvas_size, geometry.canvas_size)
    for lane, (top, left) in enumerate(geometry.lane_origins):
        y, x = margin + top + local, margin + left + local
        canvas[:, y : y + geometry.expert_size, x : x + geometry.expert_size] = fields[:, lane]
    return _active_crop(canvas, model)


def parallel_expert_canvas(
    fields: torch.Tensor, weights: torch.Tensor, model: LGVQFourStageOEO
) -> torch.Tensor:
    geometry = model.settings.geometry
    margin = geometry.active_margin
    canvas = fields.new_zeros(fields.shape[0], geometry.canvas_size, geometry.canvas_size)
    for lane, (lane_top, lane_left) in enumerate(geometry.lane_origins):
        expert = 0
        for local_top in (0, geometry.expert_pitch):
            for local_left in (0, geometry.expert_pitch):
                y, x = margin + lane_top + local_top, margin + lane_left + local_left
                canvas[:, y : y + geometry.expert_size, x : x + geometry.expert_size] = (
                    fields[:, lane] * weights[:, lane, expert, None, None]
                )
                expert += 1
    return _active_crop(canvas, model)


def serial_center_canvas(field: torch.Tensor, model: LGVQFourStageOEO) -> torch.Tensor:
    geometry = model.settings.geometry
    offset = (geometry.canvas_size - geometry.expert_size) // 2
    canvas = F.pad(
        field,
        (
            offset,
            geometry.canvas_size - geometry.expert_size - offset,
            offset,
            geometry.canvas_size - geometry.expert_size - offset,
        ),
    )
    return _active_crop(canvas, model)


def serial_expert_canvas(
    field: torch.Tensor, weights: torch.Tensor, model: LGVQFourStageOEO
) -> torch.Tensor:
    geometry = model.settings.geometry
    margin = geometry.active_margin
    canvas = field.new_zeros(field.shape[0], geometry.canvas_size, geometry.canvas_size)
    expert = 0
    for top in (geometry.expert_pitch, 2 * geometry.expert_pitch):
        for left in (geometry.expert_pitch, 2 * geometry.expert_pitch):
            y, x = margin + top, margin + left
            canvas[:, y : y + geometry.expert_size, x : x + geometry.expert_size] = (
                field * weights[:, expert, None, None]
            )
            expert += 1
    return _active_crop(canvas, model)


def _router_result(probabilities: torch.Tensor, energy: torch.Tensor, total: torch.Tensor, implementation: str) -> dict[str, Any]:
    weights, selected, indices = _sparse_top2(probabilities)
    statistics = _routing_statistics(probabilities, selected)
    return {
        "probabilities": probabilities,
        "weights": weights,
        "selected_mask": selected,
        "selected_indices": indices,
        "capture_fraction": energy.sum(-1) / total.clamp_min(1.0e-8),
        "router_implementation": implementation,
        **statistics,
    }


def parallel_router_from_ccd(active: torch.Tensor, model: LGVQFourStageOEO) -> dict[str, Any]:
    """Recover four per-frame Top-2 decisions from canonical 478x478 CCDs."""

    geometry = model.settings.geometry
    if active.ndim != 3 or tuple(active.shape[-2:]) != (geometry.active_size, geometry.active_size):
        raise ValueError("Parallel router CCD must be [B,478,478]")
    rows, totals = [], []
    for top, left in geometry.lane_origins:
        lane = active[:, top : top + geometry.quadrant_size, left : left + geometry.quadrant_size]
        totals.append(lane.sum((-2, -1)))
        rows.append(
            torch.stack(
                [
                    lane[:, y0:y1, x0:x1].sum((-2, -1))
                    for y0, y1 in model.settings.parallel_detector_intervals
                    for x0, x1 in model.settings.parallel_detector_intervals
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
        "measured_optical_parallel_energy_top2",
    )


def serial_router_from_ccd(active: torch.Tensor, model: LGVQFourStageOEO) -> dict[str, Any]:
    geometry = model.settings.geometry
    if active.ndim != 3 or tuple(active.shape[-2:]) != (geometry.active_size, geometry.active_size):
        raise ValueError("Serial router CCD must be [B,478,478]")
    energy = torch.stack(
        [
            active[:, y0:y1, x0:x1].sum((-2, -1))
            for y0, y1 in model.settings.serial_detector_intervals
            for x0, x1 in model.settings.serial_detector_intervals
        ],
        -1,
    )
    centered = energy - energy.mean(-1, keepdim=True)
    logits = centered / centered.square().mean(-1, keepdim=True).add(1.0e-8).sqrt()
    probabilities = torch.softmax(logits / model.settings.router_temperature, -1)
    return _router_result(
        probabilities,
        energy,
        active.sum((-2, -1)),
        "measured_optical_serial_energy_top2",
    )


def parallel_features_from_ccd(
    active: torch.Tensor, model: LGVQFourStageOEO, *, stage: str
) -> torch.Tensor:
    geometry = model.settings.geometry
    if stage not in {"stage1", "stage2"}:
        raise ValueError("Parallel CCD readout stage must be stage1 or stage2")
    lanes = []
    for top, left in geometry.lane_origins:
        lane = active[:, top : top + geometry.quadrant_size, left : left + geometry.quadrant_size]
        lanes.append(model.parallel_optics._normalize(lane))
    stacked = torch.stack(lanes, 1)
    readout = (
        model.parallel_optics.expert_readout
        if stage == "stage1"
        else model.parallel_optics.global_readout
    )
    result = readout(stacked.flatten(0, 1))
    return result.reshape(active.shape[0], 4, model.settings.token_count, model.settings.width)


def serial_features_from_ccd(
    active: torch.Tensor, model: LGVQFourStageOEO, *, stage: str
) -> torch.Tensor:
    if stage not in {"stage3", "stage4"}:
        raise ValueError("Serial CCD readout stage must be stage3 or stage4")
    value = active.float().clamp_min(0.0)
    mean = value.mean((-2, -1), keepdim=True).clamp_min(1.0e-6)
    normalized = torch.log1p(
        model.settings.ccd_log_compression
        * (value / mean).clamp_max(model.settings.ccd_relative_clip)
    )
    readout = (
        model.serial_optics.expert_readout
        if stage == "stage3"
        else model.serial_optics.global_readout
    )
    return readout(normalized)


MeasurementLoader = Callable[[str, list[str], torch.device], torch.Tensor]


@dataclass
class HardwareForward:
    prediction: torch.Tensor | None
    normalized_prediction: torch.Tensor | None
    amplitudes: OrderedDict[str, torch.Tensor]
    routing: dict[str, dict[str, Any]]
    optical_alignment_loss: torch.Tensor | None


def forward_hardware(
    model: LGVQFourStageOEO,
    frames: torch.Tensor,
    sample_keys: Iterable[str],
    *,
    measured_passes: Iterable[str] = (),
    measurement_loader: MeasurementLoader | None = None,
    stop_before: str | None = None,
) -> HardwareForward:
    """Run the exact model while optionally replacing optical passes by CCDs.

    ``stop_before`` returns the amplitude that must be played for that pass and
    avoids requiring its CCD.  All measured passes must form a prefix of the
    physical pass order; this prevents accidental mixing of incompatible
    simulator and hardware states during sequential deployment.
    """

    keys = [str(value) for value in sample_keys]
    selected = tuple(measured_passes)
    if any(name not in OPTICAL_PASSES for name in selected):
        raise ValueError("Unknown measured optical pass")
    if selected != OPTICAL_PASSES[: len(selected)]:
        raise ValueError("Measured optical passes must be a contiguous prefix")
    if stop_before is not None and stop_before not in OPTICAL_PASSES:
        raise ValueError(f"Unknown stop_before pass {stop_before!r}")
    if selected and measurement_loader is None:
        raise ValueError("Measured optical passes require a measurement_loader")

    def measured(name: str) -> torch.Tensor:
        assert measurement_loader is not None
        value = measurement_loader(name, keys, frames.device).float()
        expected = model.settings.geometry.active_size
        if tuple(value.shape) != (len(keys), expected, expected):
            raise ValueError(f"CCD batch for {name} has shape {tuple(value.shape)}")
        return value

    amplitudes: OrderedDict[str, torch.Tensor] = OrderedDict()
    routing: dict[str, dict[str, Any]] = {}
    alignments: list[torch.Tensor] = []

    electronic1 = model.frame_stem(frames)
    raw_fields = model.parallel_optics.raw_fields(frames)
    amplitudes["stage1_router"] = parallel_center_canvas(raw_fields, model)
    if stop_before == "stage1_router":
        return HardwareForward(None, None, amplitudes, routing, None)
    routing["stage1"] = (
        parallel_router_from_ccd(measured("stage1_router"), model)
        if "stage1_router" in selected
        else model.parallel_router(raw_fields)
    )
    amplitudes["stage1_expert"] = parallel_expert_canvas(
        raw_fields, routing["stage1"]["weights"], model
    )
    if stop_before == "stage1_expert":
        return HardwareForward(None, None, amplitudes, routing, None)
    optical1 = (
        parallel_features_from_ccd(measured("stage1_expert"), model, stage="stage1")
        if "stage1_expert" in selected
        else model.parallel_optics.expert(raw_fields, routing["stage1"]["weights"])
    )
    value = model.fusions[0](electronic1, optical1)
    alignments.append(_alignment(electronic1, optical1))

    electronic2 = model.electronic_stage2(value)
    fields2 = model.parallel_optics.token_fields(value)
    amplitudes["stage2_global"] = parallel_center_canvas(fields2, model)
    if stop_before == "stage2_global":
        return HardwareForward(None, None, amplitudes, routing, torch.stack(alignments).mean())
    optical2 = (
        parallel_features_from_ccd(measured("stage2_global"), model, stage="stage2")
        if "stage2_global" in selected
        else model.parallel_optics.global_path(fields2)
    )
    value = model.fusions[1](electronic2, optical2)
    alignments.append(_alignment(electronic2, optical2))

    sequence = deterministic_bridge(value, model.settings.bridge_pool)
    electronic3 = model.electronic_stage3(sequence)
    fields3 = model.serial_optics.fields(sequence)
    amplitudes["stage3_router"] = serial_center_canvas(fields3, model)
    if stop_before == "stage3_router":
        return HardwareForward(None, None, amplitudes, routing, torch.stack(alignments).mean())
    routing["stage3"] = (
        serial_router_from_ccd(measured("stage3_router"), model)
        if "stage3_router" in selected
        else model.serial_router(fields3)
    )
    amplitudes["stage3_expert"] = serial_expert_canvas(
        fields3, routing["stage3"]["weights"], model
    )
    if stop_before == "stage3_expert":
        return HardwareForward(None, None, amplitudes, routing, torch.stack(alignments).mean())
    optical3 = (
        serial_features_from_ccd(measured("stage3_expert"), model, stage="stage3")
        if "stage3_expert" in selected
        else model.serial_optics.expert(fields3, routing["stage3"]["weights"])
    )
    sequence = model.fusions[2](electronic3, optical3)
    alignments.append(_alignment(electronic3, optical3))

    electronic4 = model.electronic_stage4(sequence)
    fields4 = model.serial_optics.fields(sequence)
    amplitudes["stage4_global"] = serial_center_canvas(fields4, model)
    if stop_before == "stage4_global":
        return HardwareForward(None, None, amplitudes, routing, torch.stack(alignments).mean())
    optical4 = (
        serial_features_from_ccd(measured("stage4_global"), model, stage="stage4")
        if "stage4_global" in selected
        else model.serial_optics.global_path(fields4)
    )
    sequence = model.fusions[3](electronic4, optical4)
    alignments.append(_alignment(electronic4, optical4))
    normalized = model.readout(sequence)
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
    "phase_canvases",
    "serial_features_from_ccd",
    "serial_router_from_ccd",
]
