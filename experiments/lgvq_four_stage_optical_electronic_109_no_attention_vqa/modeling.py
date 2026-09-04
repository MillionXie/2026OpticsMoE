from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .settings import ExperimentSettings


def _phase(raw: torch.Tensor) -> torch.Tensor:
    return 2.0 * math.pi * torch.sigmoid(raw)


def _translate(value: torch.Tensor, shift_y: int, shift_x: int, *, fill: float | complex) -> torch.Tensor:
    result = torch.roll(value, (int(shift_y), int(shift_x)), dims=(-2, -1))
    if shift_y > 0:
        result[..., :shift_y, :] = fill
    elif shift_y < 0:
        result[..., shift_y:, :] = fill
    if shift_x > 0:
        result[..., :, :shift_x] = fill
    elif shift_x < 0:
        result[..., :, shift_x:] = fill
    return result


def _random_shift(maximum: int, training: bool) -> tuple[int, int]:
    if not training or maximum <= 0:
        return (0, 0)
    values = torch.randint(-maximum, maximum + 1, (2,))
    return int(values[0]), int(values[1])


def _phase_modulation(raw: torch.Tensor, *, probability: float, cell_size: int, training: bool) -> torch.Tensor:
    modulation = torch.exp(1j * _phase(raw)).to(torch.complex64)
    if not training or probability <= 0.0:
        return modulation
    leading = raw.shape[:-2]
    low_h = math.ceil(raw.shape[-2] / cell_size)
    low_w = math.ceil(raw.shape[-1] / cell_size)
    keep = torch.rand(*leading, low_h, low_w, device=raw.device) >= probability
    keep = keep.repeat_interleave(cell_size, -2).repeat_interleave(cell_size, -1)
    keep = keep[..., : raw.shape[-2], : raw.shape[-1]].to(torch.complex64)
    return keep * modulation + (1.0 - keep)


def _initialize_resampler(layer: nn.Linear) -> None:
    source, target = int(layer.in_features), int(layer.out_features)
    coordinates = ((torch.arange(target, dtype=torch.float32) + 0.5) * (source / target) - 0.5).clamp(0.0, source - 1.0)
    lower = coordinates.floor().long()
    upper = (lower + 1).clamp_max(source - 1)
    upper_weight = coordinates - lower.float()
    with torch.no_grad():
        layer.weight.zero_()
        rows = torch.arange(target)
        layer.weight[rows, lower] += 1.0 - upper_weight
        layer.weight[rows, upper] += upper_weight
        if layer.bias is not None:
            layer.bias.zero_()


class AngularSpectrum(nn.Module):
    def __init__(self, settings: ExperimentSettings) -> None:
        super().__init__()
        size = settings.geometry.canvas_size
        frequency = torch.fft.fftfreq(size, d=settings.pixel_pitch_um * 1.0e-6, dtype=torch.float64)
        fy, fx = torch.meshgrid(frequency, frequency, indexing="ij")
        wavelength = settings.wavelength_nm * 1.0e-9
        argument = (2.0 * math.pi) ** 2 * ((1.0 / wavelength) ** 2 - fx.square() - fy.square())
        propagating = argument >= 0.0
        if settings.k_space_enabled:
            radial = 2.0 * math.pi * torch.sqrt(fx.square() + fy.square())
            cutoff = (2.0 * math.pi / wavelength) * math.sin(math.radians(settings.theta_max_deg))
            propagating &= radial <= cutoff
        transfer = torch.exp(1j * settings.distance_m * torch.sqrt(argument.clamp_min(0.0))).to(torch.complex64)
        self.size = size
        self.register_buffer("transfer", torch.where(propagating, transfer, torch.zeros_like(transfer)), persistent=False)

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        if field.ndim != 3 or tuple(field.shape[-2:]) != (self.size, self.size):
            raise ValueError(f"Expected [B,{self.size},{self.size}], got {tuple(field.shape)}")
        return torch.fft.ifft2(torch.fft.fft2(field.to(torch.complex64)) * self.transfer)


def _sparse_top2(probabilities: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    values, indices = torch.topk(probabilities, 2, dim=-1)
    selected = torch.zeros_like(probabilities).scatter(-1, indices, 1.0)
    sparse = probabilities * selected
    sparse = sparse / sparse.square().sum(-1, keepdim=True).sqrt().clamp_min(1.0e-8)
    weights = sparse + probabilities - probabilities.detach()
    return weights, selected.bool(), indices


def _routing_statistics(probabilities: torch.Tensor, selected: torch.Tensor) -> dict[str, torch.Tensor]:
    flat_probabilities = probabilities.reshape(-1, 4)
    flat_selected = selected.float().reshape(-1, 4)
    importance = flat_probabilities.mean(0)
    load = flat_selected.mean(0) / 2.0
    balance = 4.0 * torch.sum(importance * load)
    importance_loss = 4.0 * importance.square().sum() - 1.0
    entropy = -(flat_probabilities.clamp_min(1.0e-8).log() * flat_probabilities).sum(-1).mean() / math.log(4.0)
    return {"importance": importance, "load": load, "balance_loss": balance, "importance_loss": importance_loss, "normalized_entropy": entropy}


def _spot_phase(size: int, detector_size: int, intervals: tuple[tuple[int, int], tuple[int, int]], settings: ExperimentSettings) -> torch.Tensor:
    detector_center = 0.5 * (detector_size - 1)
    centers = [0.5 * (left + right - 1) for left, right in intervals]
    coordinates = (torch.arange(size, dtype=torch.float64) - 0.5 * (size - 1)) * settings.pixel_pitch_um * 1.0e-6
    yy, xx = torch.meshgrid(coordinates, coordinates, indexing="ij")
    wavelength = settings.wavelength_nm * 1.0e-9
    pitch = settings.pixel_pitch_um * 1.0e-6
    phasors = torch.zeros_like(xx, dtype=torch.complex128)
    for target_y in centers:
        for target_x in centers:
            offset_x = (target_x - detector_center) * pitch
            offset_y = (target_y - detector_center) * pitch
            angle = 2.0 * math.pi * (offset_x * xx + offset_y * yy) / (wavelength * settings.distance_m)
            phasors += torch.exp(1j * angle)
    phase = torch.remainder(torch.angle(phasors), 2.0 * math.pi)
    normalized = (phase / (2.0 * math.pi)).clamp(1.0e-4, 1.0 - 1.0e-4)
    return torch.logit(normalized).float()


class OpticalRouterParallel(nn.Module):
    def __init__(self, settings: ExperimentSettings) -> None:
        super().__init__()
        self.settings = settings
        self.geometry = settings.geometry
        self.intervals = settings.parallel_detector_intervals
        initial = _spot_phase(self.geometry.expert_size, self.geometry.quadrant_size, self.intervals, settings)
        self.raw_router_phase = nn.Parameter(initial.unsqueeze(0).repeat(4, 1, 1))
        self.propagation = AngularSpectrum(settings)

    def forward(self, fields: torch.Tensor) -> dict[str, Any]:
        size = self.geometry.expert_size
        if tuple(fields.shape[1:]) != (4, size, size):
            raise ValueError(f"Parallel optical router expects [B,4,{size},{size}]")
        margin = self.geometry.active_margin
        local = (self.geometry.quadrant_size - size) // 2
        canvas = fields.new_zeros(fields.shape[0], self.geometry.canvas_size, self.geometry.canvas_size)
        phase_canvas = torch.ones_like(canvas, dtype=torch.complex64)
        shifted_fields = _translate(fields, *_random_shift(self.settings.input_shift_pixels, self.training), fill=0.0)
        phase = _translate(
            _phase_modulation(self.raw_router_phase, probability=self.settings.phase_dropout_p, cell_size=self.settings.phase_dropout_cell_size, training=self.training),
            *_random_shift(self.settings.phase_shift_pixels, self.training), fill=1.0 + 0.0j,
        )
        for lane, (top, left) in enumerate(self.geometry.lane_origins):
            y, x = margin + top + local, margin + left + local
            canvas[:, y : y + size, x : x + size] = shifted_fields[:, lane]
            phase_canvas[:, y : y + size, x : x + size] = phase[lane]
        detector = self.propagation(canvas.to(torch.complex64) * phase_canvas).abs().square().float()
        detector = _translate(detector, *_random_shift(self.settings.ccd_shift_pixels, self.training), fill=0.0)
        rows, lane_energy = [], []
        for top, left in self.geometry.lane_origins:
            lane = detector[:, margin + top : margin + top + self.geometry.quadrant_size, margin + left : margin + left + self.geometry.quadrant_size]
            lane_energy.append(lane.sum((-2, -1)))
            experts = []
            for y0, y1 in self.intervals:
                for x0, x1 in self.intervals:
                    experts.append(lane[:, y0:y1, x0:x1].sum((-2, -1)))
            rows.append(torch.stack(experts, -1))
        energy = torch.stack(rows, 1)
        centered = energy - energy.mean(-1, keepdim=True)
        logits = centered / centered.square().mean(-1, keepdim=True).add(1.0e-8).sqrt()
        if self.training and self.settings.router_noise_std > 0.0:
            logits = logits + torch.randn_like(logits) * self.settings.router_noise_std
        probabilities = torch.softmax(logits / self.settings.router_temperature, -1)
        weights, selected, indices = _sparse_top2(probabilities)
        statistics = _routing_statistics(probabilities, selected)
        captured = energy.sum(-1) / torch.stack(lane_energy, 1).clamp_min(1.0e-8)
        return {"probabilities": probabilities, "weights": weights, "selected_mask": selected, "selected_indices": indices, "capture_fraction": captured, "router_implementation": "optical_parallel_energy_top2", **statistics}


class OpticalRouterSerial(nn.Module):
    def __init__(self, settings: ExperimentSettings) -> None:
        super().__init__()
        self.settings = settings
        self.geometry = settings.geometry
        self.intervals = settings.serial_detector_intervals
        self.raw_router_phase = nn.Parameter(_spot_phase(self.geometry.expert_size, self.geometry.active_size, self.intervals, settings))
        self.propagation = AngularSpectrum(settings)

    def forward(self, field: torch.Tensor) -> dict[str, Any]:
        size, canvas_size = self.geometry.expert_size, self.geometry.canvas_size
        if tuple(field.shape[1:]) != (size, size):
            raise ValueError(f"Serial optical router expects [B,{size},{size}]")
        offset = (canvas_size - size) // 2
        shifted = _translate(field, *_random_shift(self.settings.input_shift_pixels, self.training), fill=0.0)
        canvas = F.pad(shifted, (offset, canvas_size - size - offset, offset, canvas_size - size - offset)).to(torch.complex64)
        phase_canvas = torch.ones_like(canvas)
        phase_canvas[:, offset : offset + size, offset : offset + size] = _translate(
            _phase_modulation(self.raw_router_phase, probability=self.settings.phase_dropout_p, cell_size=self.settings.phase_dropout_cell_size, training=self.training),
            *_random_shift(self.settings.phase_shift_pixels, self.training), fill=1.0 + 0.0j,
        )
        detector = self.propagation(canvas * phase_canvas).abs().square().float()
        detector = _translate(detector, *_random_shift(self.settings.ccd_shift_pixels, self.training), fill=0.0)
        margin = self.geometry.active_margin
        active = detector[:, margin : margin + self.geometry.active_size, margin : margin + self.geometry.active_size]
        energy = torch.stack([active[:, y0:y1, x0:x1].sum((-2, -1)) for y0, y1 in self.intervals for x0, x1 in self.intervals], -1)
        centered = energy - energy.mean(-1, keepdim=True)
        logits = centered / centered.square().mean(-1, keepdim=True).add(1.0e-8).sqrt()
        if self.training and self.settings.router_noise_std > 0.0:
            logits = logits + torch.randn_like(logits) * self.settings.router_noise_std
        probabilities = torch.softmax(logits / self.settings.router_temperature, -1)
        weights, selected, indices = _sparse_top2(probabilities)
        statistics = _routing_statistics(probabilities, selected)
        captured = energy.sum(-1) / active.sum((-2, -1)).clamp_min(1.0e-8)
        return {"probabilities": probabilities, "weights": weights, "selected_mask": selected, "selected_indices": indices, "capture_fraction": captured, "router_implementation": "optical_serial_energy_top2", **statistics}


class CcdReadout(nn.Module):
    def __init__(self, token_count: int, detector_width: int, feature_width: int) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((token_count, detector_width))
        self.norm = nn.LayerNorm(detector_width)
        self.output = nn.Linear(detector_width, feature_width)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        pooled = self.pool(value.unsqueeze(1)).squeeze(1)
        return self.output(F.softplus(self.norm(pooled)))


class OpticalPathParallel(nn.Module):
    def __init__(self, settings: ExperimentSettings) -> None:
        super().__init__()
        self.settings = settings
        self.geometry = settings.geometry
        self.width_to_field = nn.Linear(settings.width, self.geometry.expert_size)
        self.tokens_to_field = nn.Linear(settings.token_count, self.geometry.expert_size)
        _initialize_resampler(self.tokens_to_field)
        self.raw_expert_phase = nn.Parameter(torch.empty(16, self.geometry.expert_size, self.geometry.expert_size))
        self.raw_global_phase = nn.Parameter(torch.empty(self.geometry.active_size, self.geometry.active_size))
        nn.init.normal_(self.raw_expert_phase, 0.0, settings.phase_init_std)
        nn.init.normal_(self.raw_global_phase, 0.0, settings.phase_init_std)
        self.propagation = AngularSpectrum(settings)
        self.expert_readout = CcdReadout(settings.token_count, settings.detector_projection_size, settings.width)
        self.global_readout = CcdReadout(settings.token_count, settings.detector_projection_size, settings.width)
        self.last_ccd: dict[str, torch.Tensor] = {}

    def raw_fields(self, frames: torch.Tensor) -> torch.Tensor:
        frames = frames.float().div(255.0)
        gray = 0.2989 * frames[:, :, 0] + 0.5870 * frames[:, :, 1] + 0.1140 * frames[:, :, 2]
        resized = F.interpolate(gray.flatten(0, 1).unsqueeze(1), (self.geometry.expert_size, self.geometry.expert_size), mode="bilinear", align_corners=False).squeeze(1)
        field = resized.reshape(frames.shape[0], 4, self.geometry.expert_size, self.geometry.expert_size).clamp_min(0.0)
        return field / field.square().mean((-2, -1), keepdim=True).sqrt().clamp_min(1.0e-6)

    def token_fields(self, tokens: torch.Tensor) -> torch.Tensor:
        encoded = F.softplus(self.width_to_field(tokens.float()))
        field = F.softplus(self.tokens_to_field(encoded.transpose(-2, -1))).transpose(-2, -1)
        return field / field.square().mean((-2, -1), keepdim=True).sqrt().clamp_min(1.0e-6)

    def _normalize(self, value: torch.Tensor) -> torch.Tensor:
        value = value.float().clamp_min(0.0)
        mean = value.mean((-2, -1), keepdim=True).clamp_min(1.0e-6)
        return torch.log1p(self.settings.ccd_log_compression * (value / mean).clamp_max(self.settings.ccd_relative_clip))

    def _read(self, detector: torch.Tensor, *, stage: str) -> torch.Tensor:
        margin = self.geometry.active_margin
        lanes = []
        for top, left in self.geometry.lane_origins:
            lanes.append(self._normalize(detector[:, margin + top : margin + top + self.geometry.quadrant_size, margin + left : margin + left + self.geometry.quadrant_size]))
        stacked = torch.stack(lanes, 1)
        self.last_ccd[stage] = stacked.detach()
        readout = self.expert_readout if stage == "stage1" else self.global_readout
        result = readout(stacked.flatten(0, 1))
        return result.reshape(detector.shape[0], 4, self.settings.token_count, self.settings.width)

    def expert(self, fields: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        margin = self.geometry.active_margin
        canvas = fields.new_zeros(fields.shape[0], self.geometry.canvas_size, self.geometry.canvas_size)
        shifted = _translate(fields, *_random_shift(self.settings.input_shift_pixels, self.training), fill=0.0)
        for lane, (lane_top, lane_left) in enumerate(self.geometry.lane_origins):
            expert = 0
            for local_top in (0, self.geometry.expert_pitch):
                for local_left in (0, self.geometry.expert_pitch):
                    top, left = margin + lane_top + local_top, margin + lane_left + local_left
                    canvas[:, top : top + self.geometry.expert_size, left : left + self.geometry.expert_size] = shifted[:, lane] * weights[:, lane, expert, None, None]
                    expert += 1
        modulation = _translate(
            _phase_modulation(self.raw_expert_phase, probability=self.settings.phase_dropout_p, cell_size=self.settings.phase_dropout_cell_size, training=self.training),
            *_random_shift(self.settings.phase_shift_pixels, self.training), fill=1.0 + 0.0j,
        )
        field = canvas.to(torch.complex64)
        index = 0
        for lane_top, lane_left in self.geometry.lane_origins:
            for local_top in (0, self.geometry.expert_pitch):
                for local_left in (0, self.geometry.expert_pitch):
                    top, left = margin + lane_top + local_top, margin + lane_left + local_left
                    field[:, top : top + self.geometry.expert_size, left : left + self.geometry.expert_size] *= modulation[index]
                    index += 1
        detector = self.propagation(field).abs().square().float()
        detector = _translate(detector, *_random_shift(self.settings.ccd_shift_pixels, self.training), fill=0.0)
        return self._read(detector, stage="stage1")

    def global_path(self, fields: torch.Tensor) -> torch.Tensor:
        margin = self.geometry.active_margin
        local = (self.geometry.quadrant_size - self.geometry.expert_size) // 2
        canvas = fields.new_zeros(fields.shape[0], self.geometry.canvas_size, self.geometry.canvas_size)
        shifted = _translate(fields, *_random_shift(self.settings.input_shift_pixels, self.training), fill=0.0)
        for lane, (top, left) in enumerate(self.geometry.lane_origins):
            y, x = margin + top + local, margin + left + local
            canvas[:, y : y + self.geometry.expert_size, x : x + self.geometry.expert_size] = shifted[:, lane]
        phase = _translate(
            _phase_modulation(self.raw_global_phase, probability=self.settings.phase_dropout_p, cell_size=self.settings.phase_dropout_cell_size, training=self.training),
            *_random_shift(self.settings.phase_shift_pixels, self.training), fill=1.0 + 0.0j,
        )
        field = canvas.to(torch.complex64)
        field[:, margin : margin + self.geometry.active_size, margin : margin + self.geometry.active_size] *= phase
        detector = self.propagation(field).abs().square().float()
        detector = _translate(detector, *_random_shift(self.settings.ccd_shift_pixels, self.training), fill=0.0)
        return self._read(detector, stage="stage2")


class OpticalPathSerial(nn.Module):
    def __init__(self, settings: ExperimentSettings) -> None:
        super().__init__()
        self.settings = settings
        self.geometry = settings.geometry
        self.width_to_field = nn.Linear(settings.width, self.geometry.expert_size)
        self.raw_expert_phase = nn.Parameter(torch.empty(4, self.geometry.expert_size, self.geometry.expert_size))
        self.raw_global_phase = nn.Parameter(torch.empty(self.geometry.active_size, self.geometry.active_size))
        nn.init.normal_(self.raw_expert_phase, 0.0, settings.phase_init_std)
        nn.init.normal_(self.raw_global_phase, 0.0, settings.phase_init_std)
        self.propagation = AngularSpectrum(settings)
        self.expert_readout = CcdReadout(settings.serial_token_count, settings.detector_projection_size, settings.width)
        self.global_readout = CcdReadout(settings.serial_token_count, settings.detector_projection_size, settings.width)
        self.last_ccd: dict[str, torch.Tensor] = {}

    def fields(self, tokens: torch.Tensor) -> torch.Tensor:
        encoded = F.softplus(self.width_to_field(tokens.float()))
        field = encoded.new_zeros(tokens.shape[0], self.geometry.expert_size, self.geometry.expert_size)
        field[:, : tokens.shape[1]] = encoded
        return field / field.square().mean((-2, -1), keepdim=True).sqrt().clamp_min(1.0e-6)

    def _read(self, detector: torch.Tensor, *, stage: str) -> torch.Tensor:
        margin = self.geometry.active_margin
        active = detector[:, margin : margin + self.geometry.active_size, margin : margin + self.geometry.active_size]
        active = active.float().clamp_min(0.0)
        mean = active.mean((-2, -1), keepdim=True).clamp_min(1.0e-6)
        normalized = torch.log1p(self.settings.ccd_log_compression * (active / mean).clamp_max(self.settings.ccd_relative_clip))
        self.last_ccd[stage] = normalized.detach()
        readout = self.expert_readout if stage == "stage3" else self.global_readout
        return readout(normalized)

    def expert(self, field: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        margin = self.geometry.active_margin
        positions = (self.geometry.expert_pitch, 2 * self.geometry.expert_pitch)
        canvas = field.new_zeros(field.shape[0], self.geometry.canvas_size, self.geometry.canvas_size)
        shifted = _translate(field, *_random_shift(self.settings.input_shift_pixels, self.training), fill=0.0)
        expert = 0
        for top in positions:
            for left in positions:
                y, x = margin + top, margin + left
                canvas[:, y : y + self.geometry.expert_size, x : x + self.geometry.expert_size] = shifted * weights[:, expert, None, None]
                expert += 1
        phase = _translate(
            _phase_modulation(self.raw_expert_phase, probability=self.settings.phase_dropout_p, cell_size=self.settings.phase_dropout_cell_size, training=self.training),
            *_random_shift(self.settings.phase_shift_pixels, self.training), fill=1.0 + 0.0j,
        )
        propagated = canvas.to(torch.complex64)
        expert = 0
        for top in positions:
            for left in positions:
                y, x = margin + top, margin + left
                propagated[:, y : y + self.geometry.expert_size, x : x + self.geometry.expert_size] *= phase[expert]
                expert += 1
        detector = self.propagation(propagated).abs().square().float()
        detector = _translate(detector, *_random_shift(self.settings.ccd_shift_pixels, self.training), fill=0.0)
        return self._read(detector, stage="stage3")

    def global_path(self, field: torch.Tensor) -> torch.Tensor:
        margin = self.geometry.active_margin
        offset = (self.geometry.canvas_size - self.geometry.expert_size) // 2
        shifted = _translate(field, *_random_shift(self.settings.input_shift_pixels, self.training), fill=0.0)
        canvas = F.pad(shifted, (offset, self.geometry.canvas_size - self.geometry.expert_size - offset, offset, self.geometry.canvas_size - self.geometry.expert_size - offset)).to(torch.complex64)
        phase = _translate(
            _phase_modulation(self.raw_global_phase, probability=self.settings.phase_dropout_p, cell_size=self.settings.phase_dropout_cell_size, training=self.training),
            *_random_shift(self.settings.phase_shift_pixels, self.training), fill=1.0 + 0.0j,
        )
        canvas[:, margin : margin + self.geometry.active_size, margin : margin + self.geometry.active_size] *= phase
        detector = self.propagation(canvas).abs().square().float()
        detector = _translate(detector, *_random_shift(self.settings.ccd_shift_pixels, self.training), fill=0.0)
        return self._read(detector, stage="stage4")


class FrameStem(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        if width != 192:
            raise ValueError("Frame stem output width must be 192")
        self.input_channels = 14
        self.conv1 = nn.Conv2d(14, 48, 3, stride=2, padding=1)
        self.norm1 = nn.GroupNorm(8, 48)
        self.conv2 = nn.Conv2d(48, 64, 3, stride=2, padding=1)
        self.norm2 = nn.GroupNorm(8, 64)
        self.conv3 = nn.Conv2d(64, 96, 3, stride=2, padding=1)
        self.norm3 = nn.GroupNorm(12, 96)
        self.conv4 = nn.Conv2d(96, 96, 3, stride=1, padding=1)
        self.norm4 = nn.GroupNorm(12, 96)
        self.conv5 = nn.Conv2d(96, 192, 3, stride=2, padding=1)
        self.norm5 = nn.GroupNorm(24, 192)
        sobel_x = torch.tensor(((-1.0, 0.0, 1.0), (-2.0, 0.0, 2.0), (-1.0, 0.0, 1.0))) / 4.0
        sobel_y = sobel_x.t().contiguous()
        laplacian = torch.tensor(((0.0, 1.0, 0.0), (1.0, -4.0, 1.0), (0.0, 1.0, 0.0))) / 4.0
        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3), persistent=False)
        self.register_buffer("sobel_y", sobel_y.view(1, 1, 3, 3), persistent=False)
        self.register_buffer("laplacian", laplacian.view(1, 1, 3, 3), persistent=False)

    def quality_channels(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.ndim != 5 or frames.shape[1:3] != (4, 3):
            raise ValueError("Frame stem expects [B,4,3,H,W]")
        batch, frame_count, _, height, width = frames.shape
        rgb = frames.float().div(255.0)
        luminance = 0.2989 * rgb[:, :, 0:1] + 0.5870 * rgb[:, :, 1:2] + 0.1140 * rgb[:, :, 2:3]
        flat_luminance = luminance.flatten(0, 1)
        padded3 = F.pad(flat_luminance, (1, 1, 1, 1), mode="reflect")
        sobel_x = F.conv2d(padded3, self.sobel_x)
        sobel_y = F.conv2d(padded3, self.sobel_y)
        gradient = torch.sqrt(sobel_x.square() + sobel_y.square() + 1.0e-12)
        laplacian = F.conv2d(padded3, self.laplacian).abs()
        padded5 = F.pad(flat_luminance, (2, 2, 2, 2), mode="reflect")
        local_mean = F.avg_pool2d(padded5, 5, stride=1)
        local_square_mean = F.avg_pool2d(padded5.square(), 5, stride=1)
        local_std = (local_square_mean - local_mean.square()).clamp_min(0.0).sqrt()
        shape = (batch, frame_count, 1, height, width)
        sobel_x = sobel_x.reshape(shape)
        sobel_y = sobel_y.reshape(shape)
        gradient = gradient.reshape(shape)
        laplacian = laplacian.reshape(shape)
        local_std = local_std.reshape(shape)
        saturation = rgb.amax(2, keepdim=True) - rgb.amin(2, keepdim=True)
        temporal = torch.zeros_like(luminance)
        temporal[:, 1:] = (luminance[:, 1:] - luminance[:, :-1]).abs()
        y = torch.linspace(-1.0, 1.0, height, device=rgb.device, dtype=rgb.dtype).view(1, 1, 1, height, 1).expand(batch, frame_count, 1, height, width)
        x = torch.linspace(-1.0, 1.0, width, device=rgb.device, dtype=rgb.dtype).view(1, 1, 1, 1, width).expand(batch, frame_count, 1, height, width)
        time = torch.linspace(-1.0, 1.0, frame_count, device=rgb.device, dtype=rgb.dtype).view(1, frame_count, 1, 1, 1).expand(batch, frame_count, 1, height, width)
        return torch.cat(
            (rgb, luminance, sobel_x, sobel_y, gradient, laplacian, local_std, saturation, temporal, x, y, time),
            2,
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        batch, frame_count = frames.shape[:2]
        value = self.quality_channels(frames).flatten(0, 1)
        value = F.gelu(self.norm1(self.conv1(value)))
        value = F.gelu(self.norm2(self.conv2(value)))
        value = F.gelu(self.norm3(self.conv3(value)))
        value = F.gelu(self.norm4(self.conv4(value)))
        value = F.gelu(self.norm5(self.conv5(value)))
        return value.flatten(2).transpose(1, 2).reshape(batch, frame_count, -1, value.shape[1])


class ElectronicGridRoute(nn.Module):
    def __init__(self, width: int, grid: int) -> None:
        super().__init__()
        self.width, self.grid = width, grid
        self.norm = nn.LayerNorm(width)
        self.depthwise = nn.Conv2d(width, width, 5, padding=2, groups=width, bias=False)
        self.pointwise = nn.Conv2d(width, width, 1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        batch, frames, tokens, width = value.shape
        if tokens != self.grid * self.grid or width != self.width:
            raise ValueError("Electronic grid shape changed")
        image = self.norm(value).reshape(batch * frames, self.grid, self.grid, width).permute(0, 3, 1, 2)
        result = self.pointwise(F.gelu(self.depthwise(image)))
        return result.permute(0, 2, 3, 1).reshape(batch, frames, tokens, width)


class ElectronicSequenceRoute(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.depthwise = nn.Conv1d(width, width, 5, padding=2, groups=width, bias=False)
        self.pointwise = nn.Conv1d(width, width, 1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        sequence = self.norm(value).transpose(1, 2)
        return self.pointwise(F.gelu(self.depthwise(sequence))).transpose(1, 2)


class RmsConvexFusion(nn.Module):
    def __init__(self, settings: ExperimentSettings) -> None:
        super().__init__()
        self.minimum, self.maximum, self.epsilon = settings.alpha_min, settings.alpha_max, settings.fusion_epsilon
        initial = (settings.alpha_initial - self.minimum) / (self.maximum - self.minimum)
        self.raw_alpha = nn.Parameter(torch.logit(torch.tensor(initial)))
        self.last_diagnostics: dict[str, float] = {}

    @property
    def alpha(self) -> torch.Tensor:
        return self.minimum + (self.maximum - self.minimum) * torch.sigmoid(self.raw_alpha)

    def forward(self, electronic: torch.Tensor, optical: torch.Tensor) -> torch.Tensor:
        if electronic.shape != optical.shape:
            raise ValueError("Fusion inputs must have identical shape")
        axes = tuple(range(1, electronic.ndim))
        re = electronic.float().square().mean(axes, keepdim=True).sqrt().clamp_min(self.epsilon).detach()
        ro = optical.float().square().mean(axes, keepdim=True).sqrt().clamp_min(self.epsilon).detach()
        mixture = (1.0 - self.alpha) * electronic.float() / re + self.alpha * optical.float() / ro
        rm = mixture.square().mean(axes, keepdim=True).sqrt().clamp_min(self.epsilon).detach()
        result = (re * mixture / rm).to(electronic.dtype)
        self.last_diagnostics = {"alpha": float(self.alpha.detach()), "electronic_rms": float(re.mean()), "optical_rms": float(ro.mean()), "output_to_electronic_rms": float((result.float().square().mean(axes).sqrt() / re.flatten()).mean().detach())}
        return result


def deterministic_bridge(value: torch.Tensor, pool_size: int) -> torch.Tensor:
    batch, frames, tokens, width = value.shape
    grid = int(math.isqrt(tokens))
    if grid * grid != tokens:
        raise ValueError("Bridge input must be a square grid")
    images = value.reshape(batch * frames, grid, grid, width).permute(0, 3, 1, 2)
    pooled = F.adaptive_avg_pool2d(images, (pool_size, pool_size)).permute(0, 2, 3, 1).reshape(batch, frames * pool_size * pool_size, width)
    frame_means = value.mean(2)
    return torch.cat((pooled, frame_means), 1)


class QualityReadout(nn.Module):
    def __init__(self, settings: ExperimentSettings) -> None:
        super().__init__()
        width = settings.width
        self.frame_count = settings.frame_count
        self.tokens_per_frame = settings.bridge_pool * settings.bridge_pool
        self.spatial_statistics_pooling = settings.spatial_statistics_pooling
        if self.spatial_statistics_pooling:
            self.spatial_statistics_norm = nn.LayerNorm(width * 3)
            self.spatial_statistics_projection = nn.Linear(width * 3, width)
        self.spatial_norm = nn.LayerNorm(width * 2)
        self.temporal_norm = nn.LayerNorm(width)
        self.temporal_depthwise = nn.Conv1d(width, width, 3, padding=1, groups=width, bias=False)
        self.temporal_pointwise = nn.Conv1d(width, width, 1)
        self.spatial_head = nn.Sequential(nn.Linear(width * 2, settings.head_width), nn.GELU(), nn.Dropout(settings.dropout), nn.Linear(settings.head_width, 1))
        self.temporal_head = nn.Sequential(nn.Linear(width * 3, settings.head_width), nn.GELU(), nn.Dropout(settings.dropout), nn.Linear(settings.head_width, 1))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        count = self.frame_count * self.tokens_per_frame
        frame_tokens = value[:, :count].reshape(value.shape[0], self.frame_count, self.tokens_per_frame, value.shape[-1])
        frame = frame_tokens.mean(2)
        if self.spatial_statistics_pooling:
            frame_std = frame_tokens.float().std(2, unbiased=False).to(frame_tokens.dtype)
            frame_max = frame_tokens.amax(2)
            spatial_frame = F.gelu(
                self.spatial_statistics_projection(
                    self.spatial_statistics_norm(torch.cat((frame, frame_std, frame_max), -1))
                )
            )
            spatial_summary = self.spatial_norm(
                torch.cat((spatial_frame.mean(1), spatial_frame.amax(1)), -1)
            )
        else:
            spatial_summary = self.spatial_norm(torch.cat((frame.mean(1), frame.amax(1)), -1))
        temporal = self.temporal_pointwise(F.gelu(self.temporal_depthwise(self.temporal_norm(frame).transpose(1, 2)))).transpose(1, 2)
        differences = temporal[:, 1:] - temporal[:, :-1]
        temporal_summary = torch.cat((temporal.mean(1), differences.abs().mean(1), differences.abs().amax(1)), -1)
        return torch.cat((self.spatial_head(spatial_summary), self.temporal_head(temporal_summary)), -1)


def _alignment(electronic: torch.Tensor, optical: torch.Tensor) -> torch.Tensor:
    left = F.normalize(electronic.float().flatten(1), dim=-1)
    right = F.normalize(optical.float().flatten(1), dim=-1)
    return (1.0 - (left * right).sum(-1)).mean()


class LGVQFourStageOEO(nn.Module):
    def __init__(self, settings: ExperimentSettings) -> None:
        super().__init__()
        self.settings = settings
        self.frame_stem = FrameStem(settings.width)
        self.electronic_stage2 = ElectronicGridRoute(settings.width, settings.token_grid)
        self.electronic_stage3 = ElectronicSequenceRoute(settings.width)
        self.electronic_stage4 = ElectronicSequenceRoute(settings.width)
        self.parallel_optics = OpticalPathParallel(settings)
        self.serial_optics = OpticalPathSerial(settings)
        self.parallel_router = OpticalRouterParallel(settings)
        self.serial_router = OpticalRouterSerial(settings)
        self.fusions = nn.ModuleList([RmsConvexFusion(settings) for _ in range(4)])
        self.readout = QualityReadout(settings)
        self.register_buffer("target_mean", torch.zeros(2))
        self.register_buffer("target_std", torch.ones(2))

    def set_target_statistics(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.target_mean.copy_(mean)
        self.target_std.copy_(std.clamp_min(1.0e-6))

    def forward(self, frames: torch.Tensor, *, optical_enabled: bool = True) -> dict[str, Any]:
        electronic1 = self.frame_stem(frames)
        routing: dict[str, dict[str, Any]] = {}
        alignments: list[torch.Tensor] = []
        if optical_enabled:
            raw_fields = self.parallel_optics.raw_fields(frames)
            routing["stage1"] = self.parallel_router(raw_fields)
            optical1 = self.parallel_optics.expert(raw_fields, routing["stage1"]["weights"])
            value = self.fusions[0](electronic1, optical1)
            alignments.append(_alignment(electronic1, optical1))
        else:
            value = electronic1

        electronic2 = self.electronic_stage2(value)
        if optical_enabled:
            optical2 = self.parallel_optics.global_path(self.parallel_optics.token_fields(value))
            value = self.fusions[1](electronic2, optical2)
            alignments.append(_alignment(electronic2, optical2))
        else:
            value = electronic2

        sequence = deterministic_bridge(value, self.settings.bridge_pool)
        electronic3 = self.electronic_stage3(sequence)
        if optical_enabled:
            serial_fields = self.serial_optics.fields(sequence)
            routing["stage3"] = self.serial_router(serial_fields)
            optical3 = self.serial_optics.expert(serial_fields, routing["stage3"]["weights"])
            sequence = self.fusions[2](electronic3, optical3)
            alignments.append(_alignment(electronic3, optical3))
        else:
            sequence = electronic3

        electronic4 = self.electronic_stage4(sequence)
        if optical_enabled:
            optical4 = self.serial_optics.global_path(self.serial_optics.fields(sequence))
            sequence = self.fusions[3](electronic4, optical4)
            alignments.append(_alignment(electronic4, optical4))
        else:
            sequence = electronic4

        normalized_prediction = self.readout(sequence)
        prediction = normalized_prediction * self.target_std + self.target_mean
        if routing:
            balance = torch.stack([item["balance_loss"] for item in routing.values()]).mean()
            importance = torch.stack([item["importance_loss"] for item in routing.values()]).mean()
            capture = torch.stack([(1.0 - item["capture_fraction"].clamp(0.0, 1.0)).mean() for item in routing.values()]).mean()
        else:
            balance = normalized_prediction.new_zeros(())
            importance = normalized_prediction.new_zeros(())
            capture = normalized_prediction.new_zeros(())
        return {
            "prediction": prediction,
            "normalized_prediction": normalized_prediction,
            "routing": routing,
            "optical_enabled": optical_enabled,
            "optical_alignment_loss": torch.stack(alignments).mean() if alignments else normalized_prediction.new_zeros(()),
            "router_balance_loss": balance,
            "router_importance_loss": importance,
            "router_capture_loss": capture,
        }

    def fusion_diagnostics(self) -> dict[str, dict[str, float]]:
        return {f"stage{index + 1}": dict(layer.last_diagnostics) for index, layer in enumerate(self.fusions)}

    def parameter_breakdown(self) -> dict[str, Any]:
        groups = {
            "electronic_front": self.frame_stem,
            "electronic_stage2": self.electronic_stage2,
            "electronic_stage3": self.electronic_stage3,
            "electronic_stage4": self.electronic_stage4,
            "optical_feature_paths": nn.ModuleList([self.parallel_optics, self.serial_optics]),
            "optical_routers": nn.ModuleList([self.parallel_router, self.serial_router]),
            "fusion": self.fusions,
            "electronic_readout": self.readout,
        }
        report = {name: sum(parameter.numel() for parameter in module.parameters()) for name, module in groups.items()}
        report["total_trainable"] = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        report["total_frozen"] = sum(parameter.numel() for parameter in self.parameters() if not parameter.requires_grad)
        return report


def build_model(settings: ExperimentSettings) -> LGVQFourStageOEO:
    return LGVQFourStageOEO(settings)


__all__ = ["LGVQFourStageOEO", "build_model", "deterministic_bridge"]
