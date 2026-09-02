from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_router_retrieval.router import (
    sparsify_probabilities,
)
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.optics.physical import (
    AngularSpectrumPropagator,
)

from .settings import ExperimentSettings, OpticalGeometry


def _phase(raw: torch.Tensor) -> torch.Tensor:
    return 2.0 * math.pi * torch.sigmoid(raw)


def _range_logit(initial: float, minimum: float, maximum: float) -> torch.Tensor:
    value = (float(initial) - float(minimum)) / (float(maximum) - float(minimum))
    if not 0.0 < value < 1.0:
        raise ValueError("Initial alpha must be strictly inside the configured range")
    return torch.logit(torch.tensor(value))


def _range_gate(raw: torch.Tensor, minimum: float, maximum: float) -> torch.Tensor:
    return float(minimum) + (float(maximum) - float(minimum)) * torch.sigmoid(raw)


def _router_statistics(
    probabilities: torch.Tensor,
    selected: torch.Tensor,
    *,
    top_k: int,
    eps: float = 1.0e-8,
) -> dict[str, torch.Tensor]:
    flat_probabilities = probabilities.reshape(-1, probabilities.shape[-1])
    flat_selected = selected.reshape(-1, selected.shape[-1])
    importance = flat_probabilities.mean(0)
    load = flat_selected.float().mean(0) / float(top_k)
    experts = probabilities.shape[-1]
    balance = float(experts) * torch.sum(importance * load)
    importance_loss = float(experts) * torch.sum(importance.square()) - 1.0
    entropy = -(
        flat_probabilities.clamp_min(eps).log() * flat_probabilities
    ).sum(-1).mean() / math.log(float(experts))
    return {
        "importance": importance,
        "load": load,
        "balance_loss": balance,
        "importance_loss": importance_loss,
        "normalized_entropy": entropy,
    }


def _four_spot_initial_phase(
    *,
    input_size: int,
    detector_size: int,
    intervals: tuple[tuple[int, int], tuple[int, int]],
    pixel_pitch_um: float,
    wavelength_nm: float,
    distance_m: float,
) -> torch.Tensor:
    """Caltech router's deterministic four-order phase initialization."""

    detector_center = 0.5 * (detector_size - 1)
    centers = [0.5 * (left + right - 1) for left, right in intervals]
    coordinates = (
        torch.arange(input_size, dtype=torch.float64) - 0.5 * (input_size - 1)
    ) * (pixel_pitch_um * 1.0e-6)
    yy, xx = torch.meshgrid(coordinates, coordinates, indexing="ij")
    wavelength_m = wavelength_nm * 1.0e-9
    pitch_m = pixel_pitch_um * 1.0e-6
    phasors = torch.zeros_like(xx, dtype=torch.complex128)
    for target_y in centers:
        for target_x in centers:
            offset_x_m = (target_x - detector_center) * pitch_m
            offset_y_m = (target_y - detector_center) * pitch_m
            frequency_x = offset_x_m / (wavelength_m * distance_m)
            frequency_y = offset_y_m / (wavelength_m * distance_m)
            angle = 2.0 * math.pi * (frequency_x * xx + frequency_y * yy)
            phasors = phasors + torch.exp(1j * angle)
    return torch.remainder(torch.angle(phasors), 2.0 * math.pi).float()


def _phase_to_raw(initial_phase: torch.Tensor) -> torch.Tensor:
    normalized = (initial_phase / (2.0 * math.pi)).clamp(1.0e-4, 1.0 - 1.0e-4)
    return torch.logit(normalized).float()


def _translate_no_wrap(
    value: torch.Tensor, shift_y: int, shift_x: int, *, fill: float | complex
) -> torch.Tensor:
    shifted = torch.roll(value, (int(shift_y), int(shift_x)), dims=(-2, -1))
    if shift_y > 0:
        shifted[..., :shift_y, :] = fill
    elif shift_y < 0:
        shifted[..., shift_y:, :] = fill
    if shift_x > 0:
        shifted[..., :, :shift_x] = fill
    elif shift_x < 0:
        shifted[..., :, shift_x:] = fill
    return shifted


def _draw_shift(maximum: int, training: bool) -> tuple[int, int]:
    if not training or maximum <= 0:
        return (0, 0)
    return tuple(int(value) for value in torch.randint(-maximum, maximum + 1, (2,)))


def _block_phase_modulation(
    raw: torch.Tensor, *, dropout_p: float, block_size: int, training: bool
) -> torch.Tensor:
    modulation = torch.exp(1j * _phase(raw)).to(torch.complex64)
    if not training or dropout_p <= 0.0:
        return modulation
    leading = raw.shape[:-2]
    low_h = math.ceil(raw.shape[-2] / block_size)
    low_w = math.ceil(raw.shape[-1] / block_size)
    keep = torch.rand(*leading, low_h, low_w, device=raw.device) >= dropout_p
    keep = keep.repeat_interleave(block_size, -2).repeat_interleave(block_size, -1)
    keep = keep[..., : raw.shape[-2], : raw.shape[-1]].to(torch.complex64)
    return keep * modulation + (1.0 - keep)


class ElectronicLaneRouter(nn.Module):
    """Shared 788-parameter router, applied independently to all four frames."""

    implementation_name = "electronic_shared_per_frame_moe4"

    def __init__(self, settings: ExperimentSettings) -> None:
        super().__init__()
        size = int(settings.router_pool_size)
        self.pool_size = size
        self.top_k = int(settings.top_k)
        self.temperature = float(settings.router_temperature)
        self.noise_std = float(settings.router_noise_std)
        self.norm = nn.LayerNorm(size * size, elementwise_affine=False)
        self.gate = nn.Linear(size * size, 4)
        nn.init.normal_(self.gate.weight, 0.0, 0.01)
        nn.init.zeros_(self.gate.bias)

    def forward(self, fields: torch.Tensor) -> dict[str, torch.Tensor | str | bool]:
        if fields.ndim != 4 or fields.shape[1] != 4:
            raise ValueError("Electronic frame router expects [B,4,H,W]")
        batch = fields.shape[0]
        pooled = F.adaptive_avg_pool2d(
            fields.reshape(batch * 4, 1, *fields.shape[-2:]),
            (self.pool_size, self.pool_size),
        ).flatten(1)
        logits = self.gate(self.norm(pooled)).reshape(batch, 4, 4)
        if self.training and self.noise_std > 0.0:
            logits = logits + torch.randn_like(logits) * self.noise_std
        probabilities = torch.softmax(logits / self.temperature, -1)
        weights, selected, indices = sparsify_probabilities(
            probabilities.reshape(-1, 4),
            self.top_k,
            normalization="power_l2",
            straight_through=True,
        )
        weights = weights.reshape(batch, 4, 4)
        selected = selected.reshape(batch, 4, 4)
        indices = indices.reshape(batch, 4, self.top_k)
        return {
            "logits": logits,
            "probabilities": probabilities,
            "weights": weights,
            "selected_mask": selected,
            "selected_indices": indices,
            **_router_statistics(probabilities, selected, top_k=self.top_k),
            "router_implementation": self.implementation_name,
            "weight_normalization": "power_l2",
            "straight_through": True,
        }


class OpticalParallelLaneRouter(nn.Module):
    """One 518-plane exposure gives four independent optical Top-2 decisions.

    Each frame is centered inside one fixed 232x232 lane. Four independent
    109x109 router phases modulate those inputs. After one 10 cm propagation,
    sixteen 29x29 detector windows yield [B,4 lanes,4 expert scores].
    No new ROI, resize, or electronic trainable scoring head is introduced.
    """

    implementation_name = "optical_4lane_16detector_energy_moe4"

    def __init__(self, settings: ExperimentSettings) -> None:
        super().__init__()
        self.geometry = settings.geometry
        self.top_k = int(settings.top_k)
        self.temperature = float(settings.router_temperature)
        self.noise_std = float(settings.router_noise_std)
        self.phase_dropout_p = float(settings.phase_dropout_p)
        self.phase_dropout_block_size = settings.router_phase_dropout_block_size
        self.input_shift_pixels = settings.router_input_shift_pixels
        self.phase_shift_pixels = settings.router_phase_shift_pixels
        self.ccd_shift_pixels = settings.router_ccd_shift_pixels
        self.energy_eps = settings.router_energy_eps
        self.capture_loss_scale = settings.router_capture_loss_scale
        self.detector_intervals = settings.router_detector_intervals
        initial_phase = _four_spot_initial_phase(
            input_size=self.geometry.expert_size,
            detector_size=self.geometry.quadrant_size,
            intervals=self.detector_intervals,
            pixel_pitch_um=settings.pixel_pitch_um,
            wavelength_nm=settings.wavelength_nm,
            distance_m=settings.optical_distance_m,
        )
        self.raw_router_phase = nn.Parameter(
            _phase_to_raw(initial_phase).unsqueeze(0).repeat(4, 1, 1)
        )
        self.propagator = AngularSpectrumPropagator(
            wavelength_m=settings.wavelength_nm * 1.0e-9,
            pixel_size_m=settings.pixel_pitch_um * 1.0e-6,
            grid_size=self.geometry.canvas_size,
            distance_m=settings.optical_distance_m,
            k_space_constraint_enabled=settings.k_space_enabled,
            theta_max_deg=settings.theta_max_deg,
        )
        self.last_detector_intensity: torch.Tensor | None = None
        self.last_detector_energy: torch.Tensor | None = None

    def _input_canvas(self, fields: torch.Tensor) -> torch.Tensor:
        geometry = self.geometry
        margin = geometry.active_margin
        center_offset = (geometry.quadrant_size - geometry.expert_size) // 2
        canvas = fields.new_zeros(
            fields.shape[0], geometry.canvas_size, geometry.canvas_size
        )
        phase_canvas = torch.ones_like(canvas, dtype=torch.complex64)
        modulation = _block_phase_modulation(
            self.raw_router_phase,
            dropout_p=self.phase_dropout_p,
            block_size=self.phase_dropout_block_size,
            training=self.training,
        )
        input_shift = _draw_shift(self.input_shift_pixels, self.training)
        phase_shift = _draw_shift(self.phase_shift_pixels, self.training)
        for lane, (lane_top, lane_left) in enumerate(geometry.lane_origins):
            top = margin + lane_top + center_offset
            left = margin + lane_left + center_offset
            canvas[:, top : top + geometry.expert_size, left : left + geometry.expert_size] = _translate_no_wrap(
                fields[:, lane], *input_shift, fill=0.0
            )
            phase_canvas[:, top : top + geometry.expert_size, left : left + geometry.expert_size] = _translate_no_wrap(
                modulation[lane], *phase_shift, fill=1.0 + 0.0j
            )
        return canvas.to(torch.complex64) * phase_canvas

    def _detector_energy(self, intensity: torch.Tensor) -> torch.Tensor:
        geometry = self.geometry
        margin = geometry.active_margin
        rows: list[torch.Tensor] = []
        intervals = self.detector_intervals
        for lane_top, lane_left in geometry.lane_origins:
            experts: list[torch.Tensor] = []
            for local_top, local_bottom in intervals:
                for local_left, local_right in intervals:
                    crop = intensity[
                        :,
                        margin + lane_top + local_top : margin + lane_top + local_bottom,
                        margin + lane_left + local_left : margin + lane_left + local_right,
                    ]
                    experts.append(crop.sum(dim=(-2, -1)))
            rows.append(torch.stack(experts, -1))
        return torch.stack(rows, 1)

    def _lane_energy(self, intensity: torch.Tensor) -> torch.Tensor:
        """Total detector energy for each independent 232x232 Vision lane."""

        geometry = self.geometry
        margin = geometry.active_margin
        values = []
        for lane_top, lane_left in geometry.lane_origins:
            lane = intensity[
                :,
                margin + lane_top : margin + lane_top + geometry.quadrant_size,
                margin + lane_left : margin + lane_left + geometry.quadrant_size,
            ]
            values.append(lane.sum((-2, -1)))
        return torch.stack(values, 1)

    def forward(self, fields: torch.Tensor) -> dict[str, torch.Tensor | str | bool]:
        if fields.ndim != 4 or fields.shape[1] != 4 or tuple(fields.shape[-2:]) != (
            self.geometry.expert_size,
            self.geometry.expert_size,
        ):
            size = self.geometry.expert_size
            raise ValueError(f"Optical parallel router expects [B,4,{size},{size}]")
        detector = self.propagator(self._input_canvas(fields)).abs().square().float()
        detector = _translate_no_wrap(
            detector, *_draw_shift(self.ccd_shift_pixels, self.training), fill=0.0
        )
        energy = self._detector_energy(detector)
        centered = energy - energy.mean(-1, keepdim=True)
        logits = centered / centered.square().mean(-1, keepdim=True).add(self.energy_eps).sqrt()
        if self.training and self.noise_std > 0.0:
            logits = logits + torch.randn_like(logits) * self.noise_std
        probabilities = torch.softmax(logits / self.temperature, -1)
        batch = fields.shape[0]
        weights, selected, indices = sparsify_probabilities(
            probabilities.reshape(-1, 4),
            self.top_k,
            normalization="power_l2",
            straight_through=True,
        )
        weights = weights.reshape(batch, 4, 4)
        selected = selected.reshape(batch, 4, 4)
        indices = indices.reshape(batch, 4, self.top_k)
        self.last_detector_intensity = detector.detach()
        self.last_detector_energy = energy.detach()
        # Capture is assessed per optical lane. Dividing every lane by the
        # complete 518 detector would incorrectly penalize the four-way
        # parallel layout by roughly a factor of four.
        captured = energy.sum(-1) / self._lane_energy(detector).clamp_min(
            self.energy_eps
        )
        capture_loss = (1.0 - captured.clamp(0.0, 1.0)).mean()
        statistics = _router_statistics(probabilities, selected, top_k=self.top_k)
        return {
            "logits": logits,
            "probabilities": probabilities,
            "weights": weights,
            "selected_mask": selected,
            "selected_indices": indices,
            "detector_energy": energy,
            **statistics,
            "balance_loss": statistics["balance_loss"] + self.capture_loss_scale * capture_loss,
            "capture_loss": capture_loss,
            "capture_fraction": captured,
            "capture_loss_scale": fields.new_tensor(self.capture_loss_scale),
            "score_normalization": "standardized_region_energy",
            "router_implementation": self.implementation_name,
            "weight_normalization": "power_l2",
            "straight_through": True,
        }


@dataclass(frozen=True)
class SerialGeometry:
    canvas_size: int = 518
    active_size: int = 478
    quadrant_size: int = 478
    expert_size: int = 109
    expert_pitch: int = 123

    @property
    def active_margin(self) -> int:
        return (self.canvas_size - self.active_size) // 2

    @property
    def lane_origins(self) -> tuple[tuple[int, int], ...]:
        return ((0, 0),)


def _serial_geometry(settings: ExperimentSettings) -> SerialGeometry:
    return SerialGeometry(
        canvas_size=settings.geometry.canvas_size,
        active_size=settings.geometry.active_size,
        quadrant_size=settings.geometry.active_size,
        expert_size=settings.geometry.expert_size,
        expert_pitch=settings.geometry.expert_pitch,
    )


class SerialElectronicRouter(nn.Module):
    def __init__(self, settings: ExperimentSettings) -> None:
        super().__init__()
        self.top_k = settings.top_k
        self.temperature = settings.router_temperature
        self.noise_std = settings.router_noise_std
        size = settings.router_pool_size
        self.norm = nn.LayerNorm(size * size, elementwise_affine=False)
        self.gate = nn.Linear(size * size, 4)
        self.size = size
        nn.init.normal_(self.gate.weight, 0.0, 0.01)
        nn.init.zeros_(self.gate.bias)

    def forward(self, field: torch.Tensor) -> dict[str, torch.Tensor | str | bool]:
        pooled = F.adaptive_avg_pool2d(field[:, None], (self.size, self.size)).flatten(1)
        logits = self.gate(self.norm(pooled))
        if self.training and self.noise_std > 0.0:
            logits = logits + torch.randn_like(logits) * self.noise_std
        probabilities = torch.softmax(logits / self.temperature, -1)
        weights, selected, indices = sparsify_probabilities(
            probabilities,
            self.top_k,
            normalization="power_l2",
            straight_through=True,
        )
        return {
            "logits": logits,
            "probabilities": probabilities,
            "weights": weights,
            "selected_mask": selected,
            "selected_indices": indices,
            **_router_statistics(probabilities[:, None], selected[:, None], top_k=self.top_k),
            "router_implementation": "electronic_language_moe4",
            "weight_normalization": "power_l2",
            "straight_through": True,
        }


class SerialOpticalRouter(nn.Module):
    """Language-side Caltech-compatible center input -> four detector scores."""

    def __init__(self, settings: ExperimentSettings) -> None:
        super().__init__()
        self.geometry = _serial_geometry(settings)
        self.top_k = settings.top_k
        self.temperature = settings.router_temperature
        self.noise_std = settings.router_noise_std
        self.detector_intervals = settings.language_router_detector_intervals
        self.phase_dropout_p = settings.phase_dropout_p
        self.phase_dropout_block_size = settings.router_phase_dropout_block_size
        self.input_shift_pixels = settings.router_input_shift_pixels
        self.phase_shift_pixels = settings.router_phase_shift_pixels
        self.ccd_shift_pixels = settings.router_ccd_shift_pixels
        self.energy_eps = settings.router_energy_eps
        self.capture_loss_scale = settings.router_capture_loss_scale
        initial_phase = _four_spot_initial_phase(
            input_size=self.geometry.expert_size,
            detector_size=self.geometry.quadrant_size,
            intervals=self.detector_intervals,
            pixel_pitch_um=settings.pixel_pitch_um,
            wavelength_nm=settings.wavelength_nm,
            distance_m=settings.optical_distance_m,
        )
        self.raw_router_phase = nn.Parameter(_phase_to_raw(initial_phase))
        self.propagator = AngularSpectrumPropagator(
            wavelength_m=settings.wavelength_nm * 1.0e-9,
            pixel_size_m=settings.pixel_pitch_um * 1.0e-6,
            grid_size=self.geometry.canvas_size,
            distance_m=settings.optical_distance_m,
            k_space_constraint_enabled=settings.k_space_enabled,
            theta_max_deg=settings.theta_max_deg,
        )

    def forward(self, field: torch.Tensor) -> dict[str, torch.Tensor | str | bool]:
        geometry = self.geometry
        if tuple(field.shape[-2:]) != (geometry.expert_size, geometry.expert_size):
            size = geometry.expert_size
            raise ValueError(f"Language optical router requires [B,{size},{size}]")
        input_margin = (geometry.canvas_size - geometry.expert_size) // 2
        shifted_field = _translate_no_wrap(
            field, *_draw_shift(self.input_shift_pixels, self.training), fill=0.0
        )
        canvas = F.pad(
            shifted_field,
            (
                input_margin,
                geometry.canvas_size - geometry.expert_size - input_margin,
                input_margin,
                geometry.canvas_size - geometry.expert_size - input_margin,
            ),
        ).to(torch.complex64)
        modulation = torch.ones_like(canvas)
        modulation[
            :,
            input_margin : input_margin + geometry.expert_size,
            input_margin : input_margin + geometry.expert_size,
        ] = _translate_no_wrap(
            _block_phase_modulation(
                self.raw_router_phase,
                dropout_p=self.phase_dropout_p,
                block_size=self.phase_dropout_block_size,
                training=self.training,
            ),
            *_draw_shift(self.phase_shift_pixels, self.training),
            fill=1.0 + 0.0j,
        )
        detector = self.propagator(canvas * modulation).abs().square().float()
        detector = _translate_no_wrap(
            detector, *_draw_shift(self.ccd_shift_pixels, self.training), fill=0.0
        )
        margin = geometry.active_margin
        energies = []
        for top, bottom in self.detector_intervals:
            for left, right in self.detector_intervals:
                energies.append(
                    detector[
                        :,
                        margin + top : margin + bottom,
                        margin + left : margin + right,
                    ].sum((-2, -1))
                )
        energy = torch.stack(energies, -1)
        centered = energy - energy.mean(-1, keepdim=True)
        logits = centered / centered.square().mean(-1, keepdim=True).add(self.energy_eps).sqrt()
        if self.training and self.noise_std > 0.0:
            logits = logits + torch.randn_like(logits) * self.noise_std
        probabilities = torch.softmax(logits / self.temperature, -1)
        weights, selected, indices = sparsify_probabilities(
            probabilities,
            self.top_k,
            normalization="power_l2",
            straight_through=True,
        )
        active = detector[
            :,
            margin : margin + geometry.active_size,
            margin : margin + geometry.active_size,
        ]
        captured = energy.sum(-1) / active.sum((-2, -1)).clamp_min(self.energy_eps)
        capture_loss = (1.0 - captured.clamp(0.0, 1.0)).mean()
        statistics = _router_statistics(probabilities[:, None], selected[:, None], top_k=self.top_k)
        return {
            "logits": logits,
            "probabilities": probabilities,
            "weights": weights,
            "selected_mask": selected,
            "selected_indices": indices,
            "detector_energy": energy,
            **statistics,
            "balance_loss": statistics["balance_loss"] + self.capture_loss_scale * capture_loss,
            "capture_loss": capture_loss,
            "capture_fraction": captured,
            "capture_loss_scale": field.new_tensor(self.capture_loss_scale),
            "score_normalization": "standardized_region_energy",
            "router_implementation": "optical_language_center_to_moe4",
            "weight_normalization": "power_l2",
            "straight_through": True,
        }


class OpticalReadout(nn.Module):
    def __init__(self, tokens: int, projection_size: int, width: int) -> None:
        super().__init__()
        self.tokens = int(tokens)
        self.projection_size = int(projection_size)
        self.pool = nn.AdaptiveAvgPool2d((self.tokens, self.projection_size))
        self.norm = nn.LayerNorm(self.projection_size)
        self.output = nn.Linear(self.projection_size, width)

    def forward(self, lane_intensity: torch.Tensor) -> torch.Tensor:
        pooled = self.pool(lane_intensity.unsqueeze(1)).squeeze(1)
        return self.output(F.softplus(self.norm(pooled)))


def _initialize_linear_resampler(layer: nn.Linear) -> None:
    """Initialize a Linear layer as 1-D center-aligned linear resampling."""

    source_count = int(layer.in_features)
    target_count = int(layer.out_features)
    coordinates = (
        (torch.arange(target_count, dtype=torch.float32) + 0.5)
        * (float(source_count) / float(target_count))
        - 0.5
    ).clamp(0.0, float(source_count - 1))
    lower = coordinates.floor().long()
    upper = (lower + 1).clamp_max(source_count - 1)
    upper_weight = coordinates - lower.float()
    lower_weight = 1.0 - upper_weight
    with torch.no_grad():
        layer.weight.zero_()
        rows = torch.arange(target_count)
        layer.weight[rows, lower] += lower_weight
        layer.weight[rows, upper] += upper_weight
        if layer.bias is not None:
            layer.bias.zero_()


class OpticalFeaturePath(nn.Module):
    """Shared implementation for parallel Vision or serial Language optics."""

    def __init__(
        self,
        settings: ExperimentSettings,
        *,
        parallel_frames: bool,
        token_count: int,
    ) -> None:
        super().__init__()
        self.parallel_frames = bool(parallel_frames)
        self.geometry: Any = (
            settings.geometry if parallel_frames else _serial_geometry(settings)
        )
        self.token_count = int(token_count)
        self.width = int(settings.model_width)
        self.field_adapter = nn.Linear(self.width, self.geometry.expert_size)
        self.token_compressor = (
            nn.Linear(self.token_count, self.geometry.expert_size)
            if self.parallel_frames
            else None
        )
        if self.token_compressor is not None:
            _initialize_linear_resampler(self.token_compressor)
        lane_count = 4 if parallel_frames else 1
        expert_count = lane_count * 4
        self.raw_expert_phase = nn.Parameter(
            torch.empty(expert_count, self.geometry.expert_size, self.geometry.expert_size)
        )
        self.raw_global_phase = nn.Parameter(
            torch.empty(self.geometry.active_size, self.geometry.active_size)
        )
        nn.init.normal_(self.raw_expert_phase, 0.0, settings.phase_init_std)
        nn.init.normal_(self.raw_global_phase, 0.0, settings.phase_init_std)
        self.phase_dropout_p = settings.phase_dropout_p
        self.phase_dropout_block_size = settings.router_phase_dropout_block_size
        # Reuse the audited robust optical perturbation contract for every
        # feature stage; translations are zero-filled (never torch.roll wrap).
        self.input_shift_pixels = settings.router_input_shift_pixels
        self.phase_shift_pixels = settings.router_phase_shift_pixels
        self.ccd_shift_pixels = settings.router_ccd_shift_pixels
        self.propagator = AngularSpectrumPropagator(
            wavelength_m=settings.wavelength_nm * 1.0e-9,
            pixel_size_m=settings.pixel_pitch_um * 1.0e-6,
            grid_size=self.geometry.canvas_size,
            distance_m=settings.optical_distance_m,
            k_space_constraint_enabled=settings.k_space_enabled,
            theta_max_deg=settings.theta_max_deg,
        )
        self.readout1 = OpticalReadout(
            self.token_count, settings.detector_projection_size, self.width
        )
        self.readout2 = OpticalReadout(
            self.token_count, settings.detector_projection_size, self.width
        )
        self.relative_clip = settings.ccd_relative_clip
        self.log_compression = settings.ccd_log_compression
        self.last_ccd: dict[str, torch.Tensor] = {}

    def encode_fields(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.parallel_frames:
            if tokens.ndim != 4 or tokens.shape[1] != 4:
                raise ValueError("Vision optical encoder expects [B,4,T,D]")
            batch, lanes, count, _ = tokens.shape
            if count != self.token_count or self.token_compressor is None:
                raise ValueError("Vision token compressor contract changed")
            encoded = F.softplus(self.field_adapter(tokens.float()))
            # Keep all 196 tokens while mapping them to the native 109x109 SLM
            # tile: channels 192->109, then a shared token-axis 196->109 map.
            field = F.softplus(
                self.token_compressor(encoded.transpose(-2, -1))
            ).transpose(-2, -1)
        else:
            if tokens.ndim != 3:
                raise ValueError("Language optical encoder expects [B,L,D]")
            batch, count, _ = tokens.shape
            encoded = F.softplus(self.field_adapter(tokens.float()))
            field = encoded.new_zeros(
                batch, self.geometry.expert_size, self.geometry.expert_size
            )
            field[:, :count] = encoded
        rms = field.square().mean(dim=(-2, -1), keepdim=True).sqrt().clamp_min(1.0e-6)
        return field / rms

    def _expert_local_positions(self) -> tuple[int, int]:
        if self.parallel_frames:
            return (0, self.geometry.expert_pitch)
        group_span = self.geometry.expert_pitch + self.geometry.expert_size
        offset = (self.geometry.active_size - group_span) // 2
        return (offset, offset + self.geometry.expert_pitch)

    def _pack(self, fields: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        geometry = self.geometry
        batch = fields.shape[0]
        canvas = fields.new_zeros(batch, geometry.canvas_size, geometry.canvas_size)
        margin = geometry.active_margin
        lane_origins = geometry.lane_origins
        fields = _translate_no_wrap(
            fields,
            *_draw_shift(self.input_shift_pixels, self.training),
            fill=0.0,
        )
        if not self.parallel_frames:
            fields = fields[:, None]
            weights = weights[:, None]
        positions = self._expert_local_positions()
        for lane, (lane_top, lane_left) in enumerate(lane_origins):
            expert = 0
            for local_top in positions:
                for local_left in positions:
                    top = margin + lane_top + local_top
                    left = margin + lane_left + local_left
                    canvas[:, top : top + geometry.expert_size, left : left + geometry.expert_size] = (
                        fields[:, lane] * weights[:, lane, expert, None, None]
                    )
                    expert += 1
        return canvas

    def _expert_modulation(self, canvas: torch.Tensor) -> torch.Tensor:
        geometry = self.geometry
        output = canvas.to(torch.complex64).clone()
        modulation = _block_phase_modulation(
            self.raw_expert_phase,
            dropout_p=self.phase_dropout_p,
            block_size=self.phase_dropout_block_size,
            training=self.training,
        )
        modulation = _translate_no_wrap(
            modulation,
            *_draw_shift(self.phase_shift_pixels, self.training),
            fill=1.0 + 0.0j,
        )
        index = 0
        positions = self._expert_local_positions()
        for lane_top, lane_left in geometry.lane_origins:
            for local_top in positions:
                for local_left in positions:
                    top = geometry.active_margin + lane_top + local_top
                    left = geometry.active_margin + lane_left + local_left
                    output[:, top : top + geometry.expert_size, left : left + geometry.expert_size] *= modulation[index]
                    index += 1
        return output

    def _global_modulation(self, canvas: torch.Tensor) -> torch.Tensor:
        geometry = self.geometry
        output = canvas.to(torch.complex64).clone()
        margin = geometry.active_margin
        modulation = _block_phase_modulation(
            self.raw_global_phase,
            dropout_p=self.phase_dropout_p,
            block_size=self.phase_dropout_block_size,
            training=self.training,
        )
        modulation = _translate_no_wrap(
            modulation,
            *_draw_shift(self.phase_shift_pixels, self.training),
            fill=1.0 + 0.0j,
        )
        output[
            :,
            margin : margin + geometry.active_size,
            margin : margin + geometry.active_size,
        ] *= modulation
        return output

    def _normalize_lane(self, value: torch.Tensor) -> torch.Tensor:
        value = value.float().clamp_min(0.0)
        mean = value.mean((-2, -1), keepdim=True).clamp_min(1.0e-6)
        return torch.log1p(
            self.log_compression * (value / mean).clamp_max(self.relative_clip)
        )

    def _read_lanes(self, detector: torch.Tensor, *, stage: int) -> torch.Tensor:
        geometry = self.geometry
        lanes = []
        for top, left in geometry.lane_origins:
            crop = detector[
                :,
                geometry.active_margin + top : geometry.active_margin + top + geometry.quadrant_size,
                geometry.active_margin + left : geometry.active_margin + left + geometry.quadrant_size,
            ]
            lanes.append(self._normalize_lane(crop))
        stacked = torch.stack(lanes, 1)
        readout = self.readout1 if stage == 1 else self.readout2
        result = readout(stacked.reshape(-1, *stacked.shape[-2:]))
        self.last_ccd[f"stage{stage}"] = stacked.detach()
        if self.parallel_frames:
            return result.reshape(detector.shape[0], 4, self.token_count, self.width)
        return result.reshape(detector.shape[0], self.token_count, self.width)

    def forward_stage(
        self,
        fields: torch.Tensor,
        weights: torch.Tensor,
        *,
        stage: int,
    ) -> torch.Tensor:
        canvas = self._pack(fields, weights)
        modulation = (
            self._expert_modulation(canvas) if stage == 1 else self._global_modulation(canvas)
        )
        detector = self.propagator(modulation).abs().square().float()
        detector = _translate_no_wrap(
            detector,
            *_draw_shift(self.ccd_shift_pixels, self.training),
            fill=0.0,
        )
        return self._read_lanes(detector, stage=stage)


class VisionMixerBlock(nn.Module):
    """Attention-free 2D depthwise token mixer and channel MLP."""

    def __init__(self, settings: ExperimentSettings) -> None:
        super().__init__()
        width = settings.model_width
        hidden = int(round(width * settings.mixer_expansion))
        self.height = settings.token_grid_height
        self.width_tokens = settings.token_grid_width
        self.token_norm = nn.LayerNorm(width)
        self.depthwise = nn.Conv2d(width, width, 5, padding=2, groups=width, bias=False)
        self.pointwise = nn.Conv2d(width, width, 1)
        self.channel_norm = nn.LayerNorm(width)
        self.mlp = nn.Sequential(
            nn.Linear(width, hidden),
            nn.GELU(),
            nn.Dropout(settings.mixer_dropout),
            nn.Linear(hidden, width),
            nn.Dropout(settings.mixer_dropout),
        )
        self.token_gate = nn.Parameter(torch.logit(torch.tensor(0.20)))
        self.channel_gate = nn.Parameter(torch.logit(torch.tensor(0.20)))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        batch, frames, tokens, width = value.shape
        if tokens != self.height * self.width_tokens:
            raise ValueError("Vision token grid does not match mixer configuration")
        normalized = self.token_norm(value).reshape(
            batch * frames, self.height, self.width_tokens, width
        ).permute(0, 3, 1, 2)
        update = self.pointwise(F.gelu(self.depthwise(normalized))).permute(0, 2, 3, 1)
        update = update.reshape(batch, frames, tokens, width)
        value = value + torch.sigmoid(self.token_gate) * update
        return value + torch.sigmoid(self.channel_gate) * self.mlp(self.channel_norm(value))


class LanguageMixerBlock(nn.Module):
    """Causal depthwise Conv1D + channel MLP; no attention is added."""

    def __init__(self, settings: ExperimentSettings) -> None:
        super().__init__()
        width = settings.model_width
        hidden = int(round(width * settings.mixer_expansion))
        self.norm = nn.LayerNorm(width)
        self.depthwise = nn.Conv1d(width, width, 5, groups=width, bias=False)
        self.pointwise = nn.Linear(width, width)
        self.mlp_norm = nn.LayerNorm(width)
        self.mlp = nn.Sequential(
            nn.Linear(width, hidden), nn.GELU(), nn.Linear(hidden, width)
        )
        self.token_gate = nn.Parameter(torch.logit(torch.tensor(0.20)))
        self.channel_gate = nn.Parameter(torch.logit(torch.tensor(0.20)))

    def forward(self, value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        source = self.norm(value).masked_fill(~mask.unsqueeze(-1), 0.0).transpose(1, 2)
        update = self.depthwise(F.pad(source, (4, 0))).transpose(1, 2)
        value = value + torch.sigmoid(self.token_gate) * self.pointwise(F.gelu(update))
        value = value + torch.sigmoid(self.channel_gate) * self.mlp(self.mlp_norm(value))
        return value.masked_fill(~mask.unsqueeze(-1), 0.0)


class ScaleMatchedFusion(nn.Module):
    """Audited balanced residual: equal RMS, convex mix, common post-rescale."""

    def __init__(self, settings: ExperimentSettings) -> None:
        super().__init__()
        self.minimum = settings.fusion_alpha_min
        self.maximum = settings.fusion_alpha_max
        self.epsilon = settings.fusion_rms_epsilon
        self.raw_alpha = nn.Parameter(
            _range_logit(settings.fusion_alpha_initial, self.minimum, self.maximum)
        )
        self.last_diagnostics: dict[str, float] = {}

    @property
    def alpha(self) -> torch.Tensor:
        return _range_gate(self.raw_alpha, self.minimum, self.maximum)

    def forward(
        self,
        electronic: torch.Tensor,
        optical: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        if electronic.shape != optical.shape or valid_mask.shape != electronic.shape[:-1]:
            raise ValueError("Balanced fusion requires shape-matched E/O/mask")
        mask = valid_mask.unsqueeze(-1).float()
        denominator = (valid_mask.sum(tuple(range(1, valid_mask.ndim)), keepdim=True).float() * electronic.shape[-1]).clamp_min(1.0)
        while denominator.ndim < electronic.ndim:
            denominator = denominator.unsqueeze(-1)

        def rms(value: torch.Tensor) -> torch.Tensor:
            axes = tuple(range(1, value.ndim))
            return ((value.float().square() * mask).sum(axes, keepdim=True) / denominator).sqrt().clamp_min(self.epsilon)

        e32, o32 = electronic.float(), optical.float()
        re, ro = rms(e32).detach(), rms(o32).detach()
        mixture = (1.0 - self.alpha) * (e32 / re) + self.alpha * (o32 / ro)
        rm = rms(mixture).detach()
        fused = (re * mixture / rm).to(electronic.dtype)
        fused = fused.masked_fill(~valid_mask.unsqueeze(-1), 0.0)
        self.last_diagnostics = {
            "alpha": float(self.alpha.detach()),
            "electronic_rms": float(re.mean()),
            "optical_rms": float(ro.mean()),
            "mixture_rms_before_post_rescale": float(rm.mean()),
            "fused_to_electronic_rms": float(
                (rms(fused.float()) / re).mean().detach()
            ),
        }
        return fused


class LightweightTemporalHead(nn.Module):
    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        frame_width = width * 2
        self.norm = nn.LayerNorm(frame_width)
        self.temporal_depthwise = nn.Conv1d(
            frame_width, frame_width, 3, padding=1, groups=frame_width, bias=False
        )
        self.temporal_pointwise = nn.Linear(frame_width, 128)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        frame = torch.cat((tokens.mean(2), tokens.amax(2)), -1)
        mixed = self.temporal_depthwise(self.norm(frame).transpose(1, 2)).transpose(1, 2)
        frame_feature = self.dropout(F.gelu(self.temporal_pointwise(mixed)))
        return torch.cat((frame_feature.mean(1), frame_feature.amax(1)), -1)


class DualTaskQualityReadout(nn.Module):
    """Attention-free task-specific electronics after all four optical layers."""

    def __init__(self, settings: ExperimentSettings) -> None:
        super().__init__()
        width = settings.model_width
        hidden = settings.quality_head_width
        dropout = settings.quality_head_dropout
        self.spatial_frame = nn.Sequential(
            nn.LayerNorm(width * 3),
            nn.Linear(width * 3, width),
            nn.GELU(),
        )
        temporal_width = width * 2
        self.temporal_norm = nn.LayerNorm(temporal_width)
        self.temporal_k3 = nn.Conv1d(
            temporal_width, temporal_width, 3, padding=1,
            groups=temporal_width, bias=False,
        )
        self.temporal_k5 = nn.Conv1d(
            temporal_width, temporal_width, 5, padding=2,
            groups=temporal_width, bias=False,
        )
        self.temporal_project = nn.Sequential(
            nn.Linear(temporal_width * 3, width),
            nn.GELU(),
        )
        self.spatial_head = nn.Sequential(
            nn.LayerNorm(width * 2 + 128),
            nn.Linear(width * 2 + 128, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.temporal_head = nn.Sequential(
            nn.LayerNorm(width * 4 + 128),
            nn.Linear(width * 4 + 128, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(
        self, vision: torch.Tensor, language_summary: torch.Tensor
    ) -> torch.Tensor:
        frame_mean = vision.mean(2)
        frame_max = vision.amax(2)
        frame_std = vision.float().std(2, unbiased=False).to(vision.dtype)
        spatial_frame = self.spatial_frame(
            torch.cat((frame_mean, frame_std, frame_max), -1)
        )
        spatial_video = torch.cat(
            (spatial_frame.mean(1), spatial_frame.amax(1)), -1
        )
        spatial = self.spatial_head(
            torch.cat((spatial_video, language_summary), -1)
        )

        temporal_frame = torch.cat((frame_mean, frame_max), -1)
        normalized = self.temporal_norm(temporal_frame).transpose(1, 2)
        mixed = torch.cat(
            (
                temporal_frame,
                self.temporal_k3(normalized).transpose(1, 2),
                self.temporal_k5(normalized).transpose(1, 2),
            ),
            -1,
        )
        temporal_frame = self.temporal_project(mixed)
        sequence_summary = torch.cat(
            (temporal_frame.mean(1), temporal_frame.amax(1)), -1
        )
        differences = temporal_frame[:, 1:] - temporal_frame[:, :-1]
        difference_summary = torch.cat(
            (differences.mean(1), differences.abs().amax(1)), -1
        )
        temporal = self.temporal_head(
            torch.cat((sequence_summary, difference_summary, language_summary), -1)
        )
        return torch.cat((spatial, temporal), -1)


class LGVQSpatiotemporalModel(nn.Module):
    """Qwen cached Vision+Language -> four optical layers -> two MOS values."""

    def __init__(self, settings: ExperimentSettings) -> None:
        super().__init__()
        self.settings = settings
        self.vision_adapter = nn.Sequential(
            nn.LayerNorm(settings.input_width),
            nn.Linear(settings.input_width, settings.model_width),
        )
        self.language_adapter = nn.Sequential(
            nn.LayerNorm(settings.language_input_width),
            nn.Linear(settings.language_input_width, settings.model_width),
        )
        # Four sample-dependent image tokens are prepended before Language
        # Block 1. This replaces the native single-image Qwen merger for the
        # four-frame parallel contract; it is not DeepStack.
        self.lightweight_frame_merger = nn.Sequential(
            nn.LayerNorm(settings.model_width * 2),
            nn.Linear(settings.model_width * 2, settings.model_width),
            nn.GELU(),
        )
        self.vision_mixers = nn.ModuleList([VisionMixerBlock(settings) for _ in range(2)])
        self.language_mixers = nn.ModuleList([LanguageMixerBlock(settings) for _ in range(2)])
        self.vision_optics = OpticalFeaturePath(
            settings, parallel_frames=True, token_count=settings.token_count
        )
        self.language_optics = OpticalFeaturePath(
            settings,
            parallel_frames=False,
            token_count=settings.language_token_count + settings.frame_count,
        )
        self.vision_router = OpticalParallelLaneRouter(settings)
        self.language_router = SerialOpticalRouter(settings)
        self.vision_fusions = nn.ModuleList([ScaleMatchedFusion(settings) for _ in range(2)])
        self.language_fusions = nn.ModuleList([ScaleMatchedFusion(settings) for _ in range(2)])
        self.vision_output_norm = nn.LayerNorm(settings.model_width)
        self.language_output_norm = nn.LayerNorm(settings.model_width)
        self.language_pool = nn.Sequential(
            nn.LayerNorm(settings.model_width * 2),
            nn.Linear(settings.model_width * 2, 128),
            nn.GELU(),
        )
        self.quality_readout = DualTaskQualityReadout(settings)
        self.register_buffer("target_mean", torch.zeros(2))
        self.register_buffer("target_std", torch.ones(2))
        self.last_routing: dict[str, dict[str, Any]] = {}

    def set_target_statistics(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        if tuple(mean.shape) != (2,) or tuple(std.shape) != (2,):
            raise ValueError("Target mean/std must both be [2]")
        self.target_mean.copy_(mean.float())
        self.target_std.copy_(std.float().clamp_min(1.0e-6))

    def initialize_without_spaq(self) -> dict[str, Any]:
        """SPAQ assets are optional; preserve Qwen cache geometry at init."""

        for adapter in (self.vision_adapter[1], self.language_adapter[1]):
            nn.init.orthogonal_(adapter.weight)
            nn.init.zeros_(adapter.bias)
        return {
            "mode": "qwen_backbone_cache_orthogonal_projection",
            "spaq_checkpoint_required": False,
            "vision_projection": [self.settings.input_width, self.settings.model_width],
            "language_projection": [
                self.settings.language_input_width,
                self.settings.model_width,
            ],
            "attention_added": False,
        }

    def _language_pool(self, value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        count = mask.sum(1, keepdim=True).clamp_min(1).float()
        mean = (value * mask.unsqueeze(-1)).sum(1) / count
        maximum = value.masked_fill(~mask.unsqueeze(-1), -torch.inf).amax(1)
        maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
        return self.language_pool(torch.cat((mean, maximum), -1))

    @staticmethod
    def _optical_alignment(
        electronic: torch.Tensor,
        optical: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask = valid_mask.unsqueeze(-1).float()
        axes = tuple(range(1, electronic.ndim))
        denominator = (
            valid_mask.sum(tuple(range(1, valid_mask.ndim)), keepdim=True).float()
            * electronic.shape[-1]
        ).clamp_min(1.0)
        while denominator.ndim < electronic.ndim:
            denominator = denominator.unsqueeze(-1)

        def normalize(value: torch.Tensor) -> torch.Tensor:
            value = value.float()
            rms = (
                (value.square() * mask).sum(axes, keepdim=True) / denominator
            ).sqrt().clamp_min(1.0e-6)
            return value / rms

        target = normalize(electronic).detach()
        prediction = normalize(optical)
        smooth = (
            F.smooth_l1_loss(prediction, target, reduction="none") * mask
        ).sum() / mask.sum().clamp_min(1.0) / electronic.shape[-1]
        flattened_target = (target * mask).flatten(1)
        flattened_prediction = (prediction * mask).flatten(1)
        cosine = 1.0 - F.cosine_similarity(
            flattened_prediction, flattened_target, dim=-1, eps=1.0e-6
        ).mean()
        return 0.5 * (smooth + cosine)

    def forward(
        self,
        frame_tokens: torch.Tensor,
        language_tokens: torch.Tensor,
        language_mask: torch.Tensor,
    ) -> dict[str, Any]:
        expected_vision = (
            self.settings.frame_count,
            self.settings.token_count,
            self.settings.input_width,
        )
        if frame_tokens.ndim != 4 or tuple(frame_tokens.shape[1:]) != expected_vision:
            raise ValueError(f"frame_tokens must be [B,{expected_vision}], got {tuple(frame_tokens.shape)}")
        if language_tokens.ndim != 3 or language_tokens.shape[0] != frame_tokens.shape[0]:
            raise ValueError("language_tokens must be [B,L,D]")
        if language_tokens.shape[-1] != self.settings.language_input_width:
            raise ValueError("Cached language hidden width does not match config")
        if tuple(language_mask.shape) != tuple(language_tokens.shape[:2]):
            raise ValueError("language_mask must be [B,L]")

        vision = self.vision_adapter(frame_tokens.float())
        vision_mask = torch.ones(
            vision.shape[:-1], dtype=torch.bool, device=vision.device
        )
        vision_fields = self.vision_optics.encode_fields(vision)
        vision_routing = self.vision_router(vision_fields)

        # The four-stage replacement order is strictly Vision expert -> Vision
        # global -> multimodal merger -> Language expert -> Language global.
        electronic_v1 = self.vision_mixers[0](vision)
        optical_v1 = self.vision_optics.forward_stage(
            vision_fields, vision_routing["weights"], stage=1
        )
        alignment_losses = [
            self._optical_alignment(electronic_v1, optical_v1, vision_mask)
        ]
        vision = self.vision_fusions[0](electronic_v1, optical_v1, vision_mask)
        vision_fields2 = self.vision_optics.encode_fields(vision)
        electronic_v2 = self.vision_mixers[1](vision)
        optical_v2 = self.vision_optics.forward_stage(
            vision_fields2, vision_routing["weights"], stage=2
        )
        alignment_losses.append(
            self._optical_alignment(electronic_v2, optical_v2, vision_mask)
        )
        vision = self.vision_output_norm(
            self.vision_fusions[1](electronic_v2, optical_v2, vision_mask)
        )

        prompt_language = self.language_adapter(language_tokens.float()).masked_fill(
            ~language_mask.unsqueeze(-1), 0.0
        )
        frame_image_tokens = self.lightweight_frame_merger(
            torch.cat((vision.mean(2), vision.amax(2)), -1)
        )
        language = torch.cat((frame_image_tokens, prompt_language), 1)
        language_mask = torch.cat(
            (
                torch.ones(
                    language_mask.shape[0],
                    self.settings.frame_count,
                    dtype=torch.bool,
                    device=language_mask.device,
                ),
                language_mask,
            ),
            1,
        )
        language_fields = self.language_optics.encode_fields(language)
        language_routing = self.language_router(language_fields)
        electronic_l1 = self.language_mixers[0](language, language_mask)
        optical_l1 = self.language_optics.forward_stage(
            language_fields, language_routing["weights"], stage=1
        )
        alignment_losses.append(
            self._optical_alignment(
                electronic_l1, optical_l1[:, : language.shape[1]], language_mask
            )
        )
        language = self.language_fusions[0](
            electronic_l1, optical_l1[:, : language.shape[1]], language_mask
        )
        language_fields2 = self.language_optics.encode_fields(language)
        electronic_l2 = self.language_mixers[1](language, language_mask)
        optical_l2 = self.language_optics.forward_stage(
            language_fields2, language_routing["weights"], stage=2
        )
        alignment_losses.append(
            self._optical_alignment(
                electronic_l2, optical_l2[:, : language.shape[1]], language_mask
            )
        )
        language = self.language_output_norm(
            self.language_fusions[1](
                electronic_l2, optical_l2[:, : language.shape[1]], language_mask
            )
        ).masked_fill(~language_mask.unsqueeze(-1), 0.0)
        self.last_routing = {
            "vision": vision_routing,
            "language": language_routing,
        }

        language_summary = self._language_pool(language, language_mask)
        normalized_prediction = self.quality_readout(vision, language_summary)
        prediction = normalized_prediction * self.target_std + self.target_mean
        zero = normalized_prediction.new_zeros(())
        vision_capture = vision_routing.get("capture_loss", zero)
        language_capture = language_routing.get("capture_loss", zero)
        return {
            "prediction": prediction,
            "normalized_prediction": normalized_prediction,
            "router_balance_loss": 0.5
            * (vision_routing["balance_loss"] + language_routing["balance_loss"]),
            "router_importance_loss": 0.5
            * (vision_routing["importance_loss"] + language_routing["importance_loss"]),
            "router_capture_loss": 0.5 * (vision_capture + language_capture),
            "optical_alignment_loss": torch.stack(alignment_losses).mean(),
            "routing": self.last_routing,
        }

    def fusion_diagnostics(self) -> dict[str, dict[str, float]]:
        names = (
            ("vision_expert", self.vision_fusions[0]),
            ("vision_global", self.vision_fusions[1]),
            ("language_expert", self.language_fusions[0]),
            ("language_global", self.language_fusions[1]),
        )
        return {name: dict(module.last_diagnostics) for name, module in names}

    def parameter_breakdown(self) -> dict[str, Any]:
        def count(module: nn.Module) -> int:
            return sum(parameter.numel() for parameter in module.parameters())

        router_parameter_ids = {
            id(parameter)
            for module in (self.vision_router, self.language_router)
            for parameter in module.parameters()
        }
        phase = sum(
            parameter.numel()
            for name, parameter in self.named_parameters()
            if "raw_" in name and "phase" in name and id(parameter) not in router_parameter_ids
        )
        router = count(self.vision_router) + count(self.language_router)
        total = count(self)
        return {
            "architecture": self.settings.architecture_label,
            "total": total,
            "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "phase": phase,
            "router": router,
            "other_electronic": total - phase - router,
            "outputs": ["spatial", "temporal"],
            "alignment_output": False,
            "vision_optical_layers": 2,
            "language_optical_layers": 2,
            "vision_frame_parallelism": 4,
        }


def _resize_wrapped_phase(raw: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    phase = _phase(raw.float())
    leading = phase.shape[:-2]
    source = phase.reshape(-1, 1, *phase.shape[-2:])
    size = target.shape[-2:]
    cosine = F.interpolate(
        source.cos(), size=size, mode="bicubic", align_corners=False
    )
    sine = F.interpolate(
        source.sin(), size=size, mode="bicubic", align_corners=False
    )
    resized = torch.atan2(sine, cosine).remainder(2.0 * math.pi)
    return _phase_to_raw(resized).reshape(*leading, *size).to(target.dtype)


def _resize_linear_rows(value: torch.Tensor, rows: int) -> torch.Tensor:
    if value.ndim == 2:
        return F.interpolate(
            value.float().t().unsqueeze(0),
            size=rows,
            mode="linear",
            align_corners=False,
        ).squeeze(0).t().to(value.dtype)
    if value.ndim == 1:
        return F.interpolate(
            value.float()[None, None],
            size=rows,
            mode="linear",
            align_corners=False,
        )[0, 0].to(value.dtype)
    raise ValueError("Only Linear output rows/bias may be resized")


def _resize_vector(value: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if value.ndim != 1 or target.ndim != 1:
        raise ValueError("Vector migration requires one-dimensional tensors")
    return F.interpolate(
        value.float()[None, None],
        size=target.numel(),
        mode="linear",
        align_corners=False,
    )[0, 0].to(target.dtype)


def _resize_linear_weight(value: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Resize both Linear axes and preserve each output row's L2 scale."""

    if value.ndim != 2 or target.ndim != 2:
        raise ValueError("Linear migration requires two-dimensional weights")
    resized = F.interpolate(
        value.float()[None, None],
        size=tuple(target.shape),
        mode="bilinear",
        align_corners=False,
    )[0, 0]
    source_row_norm = value.float().square().sum(-1).sqrt()
    if source_row_norm.numel() != target.shape[0]:
        source_row_norm = F.interpolate(
            source_row_norm[None, None],
            size=target.shape[0],
            mode="linear",
            align_corners=False,
        )[0, 0]
    target_row_norm = resized.square().sum(-1).sqrt().clamp_min(1.0e-8)
    resized = resized * (source_row_norm / target_row_norm).unsqueeze(-1)
    return resized.to(target.dtype)


def build_model(settings: ExperimentSettings) -> tuple[LGVQSpatiotemporalModel, dict[str, Any]]:
    model = LGVQSpatiotemporalModel(settings)
    fallback_report = model.initialize_without_spaq()
    checkpoint = settings.initialization_checkpoint or settings.optional_sister_checkpoint
    if checkpoint is None:
        return model, fallback_report
    if not checkpoint.exists():
        if settings.initialization_checkpoint is not None:
            raise FileNotFoundError(
                f"Configured O2 initialization checkpoint is missing: {checkpoint}"
            )
        return model, {
            **fallback_report,
            "optional_checkpoint_missing": str(checkpoint),
        }
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
    if not isinstance(state, dict):
        raise RuntimeError("Initialization checkpoint does not contain a state_dict")
    if not any(key in model.state_dict() for key in state) and all(
        str(key).startswith("module.") for key in state
    ):
        state = {str(key)[7:]: value for key, value in state.items()}

    required_o2_phase = {
        "vision_optics.raw_expert_phase",
        "vision_optics.raw_global_phase",
        "language_optics.raw_expert_phase",
        "language_optics.raw_global_phase",
        "vision_router.raw_router_phase",
        "language_router.raw_router_phase",
    }
    required_o2_adapters = {
        "vision_adapter.0.weight",
        "vision_adapter.0.bias",
        "vision_adapter.1.weight",
        "vision_adapter.1.bias",
        "language_adapter.0.weight",
        "language_adapter.0.bias",
        "language_adapter.1.weight",
        "language_adapter.1.bias",
        "vision_optics.field_adapter.weight",
        "vision_optics.field_adapter.bias",
        "language_optics.field_adapter.weight",
        "language_optics.field_adapter.bias",
    }
    if settings.initialization_checkpoint is not None:
        absent = sorted((required_o2_phase | required_o2_adapters) - set(state))
        if absent:
            raise RuntimeError(
                "The configured initialization is not a complete prior O2 checkpoint; "
                f"missing migration tensors: {absent}"
            )

    target_state = model.state_dict()
    compatible = {}
    resized: list[str] = []
    exact: list[str] = []
    for key, value in state.items():
        if (
            key not in target_state
            or key.endswith("raw_alpha")
            or not torch.is_tensor(value)
        ):
            continue
        target = target_state[key]
        if tuple(value.shape) == tuple(target.shape):
            compatible[key] = value.to(target.dtype)
            exact.append(key)
        elif "raw_" in key and "phase" in key and value.ndim >= 2:
            if tuple(value.shape[:-2]) == tuple(target.shape[:-2]):
                compatible[key] = _resize_wrapped_phase(value, target)
                resized.append(key)
        elif key.endswith("field_adapter.weight") and value.shape[1] == target.shape[1]:
            compatible[key] = _resize_linear_rows(value, target.shape[0])
            resized.append(key)
        elif key.endswith("field_adapter.bias"):
            compatible[key] = _resize_linear_rows(value, target.shape[0])
            resized.append(key)
        elif key in {
            "vision_adapter.0.weight",
            "vision_adapter.0.bias",
            "language_adapter.0.weight",
            "language_adapter.0.bias",
        }:
            compatible[key] = _resize_vector(value, target)
            resized.append(key)
        elif key in {
            "vision_adapter.1.weight",
            "language_adapter.1.weight",
        }:
            compatible[key] = _resize_linear_weight(value, target)
            resized.append(key)

    if settings.initialization_checkpoint is not None:
        failed = sorted(
            key
            for key in required_o2_phase | required_o2_adapters
            if key not in compatible
        )
        if failed:
            raise RuntimeError(f"Required O2 migration tensors could not be loaded: {failed}")
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected compatible initialization keys: {unexpected}")
    return model, {
        "mode": "o2_checkpoint_migration_to_hardware109",
        "path": str(checkpoint),
        "source_architecture": (
            payload.get("architecture") if isinstance(payload, dict) else None
        ),
        "source_epoch": payload.get("epoch") if isinstance(payload, dict) else None,
        "loaded_tensor_count": len(compatible),
        "exact_tensor_count": len(exact),
        "exact_tensors": sorted(exact),
        "complex_phase_or_row_resized": resized,
        "required_phase_tensors_migrated": sorted(required_o2_phase & set(compatible)),
        "required_adapter_tensors_migrated": sorted(
            required_o2_adapters & set(compatible)
        ),
        "missing_new_tensor_count": len(missing),
        "new_or_unmapped_tensors": sorted(missing),
        "spaq_checkpoint_required": False,
        "alpha_reset_to_configured_initial": True,
        "warning": (
            "Exact tensors are reused; phase tensors are resized through cos/sin "
            "and atan2. The old 1024-wide Vision adapter is deterministically "
            "resized to the full-Qwen 2048-wide cache; the dual readout and "
            "Vision token compressor remain new."
        ),
    }


__all__ = [
    "LGVQSpatiotemporalModel",
    "OpticalParallelLaneRouter",
    "ScaleMatchedFusion",
    "build_model",
]


