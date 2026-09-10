from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .settings import ExperimentSettings


def _phase(raw: torch.Tensor) -> torch.Tensor:
    return 2.0 * math.pi * torch.sigmoid(raw)


def _translate(
    value: torch.Tensor,
    shift_y: int,
    shift_x: int,
    *,
    fill: float | complex,
) -> torch.Tensor:
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
    sample = torch.randint(-maximum, maximum + 1, (2,))
    return int(sample[0]), int(sample[1])


def _phase_modulation(
    raw: torch.Tensor,
    *,
    settings: ExperimentSettings,
    training: bool,
) -> torch.Tensor:
    phase = _phase(raw)
    levels = settings.phase_quantization_levels
    if levels:
        step = (2.0 * math.pi) / float(levels - 1)
        quantized = torch.round(phase / step) * step
        # The forward pass sees exactly the 8-bit phase levels exported to the
        # physical SLM.  The straight-through gradient still updates raw_phase.
        phase = phase + (quantized - phase).detach() if training else quantized
    modulation = torch.exp(1j * phase).to(torch.complex64)
    if training and settings.phase_dropout_p > 0.0:
        leading = raw.shape[:-2]
        cell_size = settings.phase_dropout_cell_size
        low_h = math.ceil(raw.shape[-2] / cell_size)
        low_w = math.ceil(raw.shape[-1] / cell_size)
        keep = (
            torch.rand(*leading, low_h, low_w, device=raw.device)
            >= settings.phase_dropout_p
        )
        keep = keep.repeat_interleave(cell_size, -2).repeat_interleave(cell_size, -1)
        keep = keep[..., : raw.shape[-2], : raw.shape[-1]].to(torch.complex64)
        # A dropped phase cell becomes zero phase, not zero optical amplitude.
        modulation = keep * modulation + (1.0 - keep)

    if training and (
        settings.unmodulated_power_fraction_max
        > settings.unmodulated_power_fraction_min
    ):
        fraction = settings.unmodulated_power_fraction_min + (
            settings.unmodulated_power_fraction_max
            - settings.unmodulated_power_fraction_min
        ) * torch.rand((), device=raw.device)
    else:
        fraction = raw.new_tensor(settings.unmodulated_power_fraction_eval)
    # Coherent zero-order leakage: ``fraction`` is the nominal unmodulated
    # optical-power coefficient, hence its field coefficient is sqrt(fraction).
    # Propagation remains one physical 10-cm pass because it is linear in the
    # complex field. Interference makes the observed CCD share position-varying,
    # as it is in the real setup.
    return (
        torch.sqrt(1.0 - fraction) * modulation
        + torch.sqrt(fraction).to(torch.complex64)
    )


def _initialize_resampler(layer: nn.Linear) -> None:
    source, target = int(layer.in_features), int(layer.out_features)
    coordinate = (
        (torch.arange(target, dtype=torch.float32) + 0.5) * (source / target) - 0.5
    ).clamp(0.0, source - 1.0)
    lower = coordinate.floor().long()
    upper = (lower + 1).clamp_max(source - 1)
    upper_weight = coordinate - lower.float()
    with torch.no_grad():
        layer.weight.zero_()
        rows = torch.arange(target)
        layer.weight[rows, lower] += 1.0 - upper_weight
        layer.weight[rows, upper] += upper_weight
        if layer.bias is not None:
            layer.bias.zero_()


class AngularSpectrum(nn.Module):
    """Fixed 532 nm / 17 um / 10 cm angular-spectrum propagation."""

    def __init__(self, settings: ExperimentSettings) -> None:
        super().__init__()
        size = settings.geometry.canvas_size
        frequency = torch.fft.fftfreq(
            size, d=settings.pixel_pitch_um * 1.0e-6, dtype=torch.float64
        )
        fy, fx = torch.meshgrid(frequency, frequency, indexing="ij")
        wavelength = settings.wavelength_nm * 1.0e-9
        argument = (2.0 * math.pi) ** 2 * (
            (1.0 / wavelength) ** 2 - fx.square() - fy.square()
        )
        valid = argument >= 0.0
        if settings.k_space_enabled:
            radial = 2.0 * math.pi * torch.sqrt(fx.square() + fy.square())
            cutoff = (2.0 * math.pi / wavelength) * math.sin(
                math.radians(settings.theta_max_deg)
            )
            valid &= radial <= cutoff
        transfer = torch.exp(
            1j * settings.distance_m * torch.sqrt(argument.clamp_min(0.0))
        ).to(torch.complex64)
        self.size = size
        self.register_buffer(
            "transfer",
            torch.where(valid, transfer, torch.zeros_like(transfer)),
            persistent=False,
        )

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        expected = (self.size, self.size)
        if field.ndim != 3 or tuple(field.shape[-2:]) != expected:
            raise ValueError(f"Expected [B,{self.size},{self.size}], got {tuple(field.shape)}")
        return torch.fft.ifft2(
            torch.fft.fft2(field.to(torch.complex64)) * self.transfer
        )


def _sparse_top2(
    probabilities: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    values, indices = torch.topk(probabilities, 2, dim=-1)
    selected = torch.zeros_like(probabilities).scatter(-1, indices, 1.0)
    sparse = probabilities * selected
    sparse = sparse / sparse.square().sum(-1, keepdim=True).sqrt().clamp_min(1.0e-8)
    # Strict Top-2 in the forward pass; dense softmax gradient in the backward pass.
    weights = sparse + probabilities - probabilities.detach()
    return weights, selected.bool(), indices


def _routing_statistics(
    probabilities: torch.Tensor, selected: torch.Tensor
) -> dict[str, torch.Tensor]:
    flat_p = probabilities.reshape(-1, 4)
    flat_s = selected.float().reshape(-1, 4)
    importance = flat_p.mean(0)
    load = flat_s.mean(0) / 2.0
    conditional_entropy = -(
        flat_p.clamp_min(1.0e-8).log() * flat_p
    ).sum(-1).mean() / math.log(4.0)
    marginal_entropy = -(
        importance.clamp_min(1.0e-8).log() * importance
    ).sum() / math.log(4.0)
    return {
        "importance": importance,
        "load": load,
        "balance_loss": 4.0 * torch.sum(importance * load),
        "importance_loss": 4.0 * importance.square().sum() - 1.0,
        "normalized_entropy": conditional_entropy,
        "marginal_entropy": marginal_entropy,
        # Negative normalized mutual information. Minimizing this term asks
        # different samples to make different, confident optical routing
        # decisions; unlike hard load counts, it remains differentiable.
        "diversity_loss": conditional_entropy - marginal_entropy,
    }


def _spot_phase(
    phase_size: int,
    detector_size: int,
    intervals: tuple[tuple[int, int], tuple[int, int]],
    settings: ExperimentSettings,
) -> torch.Tensor:
    detector_center = 0.5 * (detector_size - 1)
    centers = [0.5 * (left + right - 1) for left, right in intervals]
    coordinate = (
        torch.arange(phase_size, dtype=torch.float64) - 0.5 * (phase_size - 1)
    ) * settings.pixel_pitch_um * 1.0e-6
    yy, xx = torch.meshgrid(coordinate, coordinate, indexing="ij")
    wavelength = settings.wavelength_nm * 1.0e-9
    pitch = settings.pixel_pitch_um * 1.0e-6
    phasors = torch.zeros_like(xx, dtype=torch.complex128)
    for target_y in centers:
        for target_x in centers:
            dx = (target_x - detector_center) * pitch
            dy = (target_y - detector_center) * pitch
            angle = 2.0 * math.pi * (dx * xx + dy * yy) / (
                wavelength * settings.distance_m
            )
            phasors += torch.exp(1j * angle)
    normalized = (
        torch.remainder(torch.angle(phasors), 2.0 * math.pi) / (2.0 * math.pi)
    ).clamp(1.0e-4, 1.0 - 1.0e-4)
    return torch.logit(normalized).float()


class OpticalRouterParallel16(nn.Module):
    def __init__(self, settings: ExperimentSettings) -> None:
        super().__init__()
        self.settings = settings
        self.geometry = settings.geometry
        initial = _spot_phase(
            self.geometry.parallel_expert_size,
            self.geometry.lane_size,
            settings.parallel_router_intervals,
            settings,
        )
        self.raw_router_phase = nn.Parameter(
            initial.unsqueeze(0).repeat(settings.frame_count, 1, 1)
        )
        self.propagation = AngularSpectrum(settings)

    def forward(self, fields: torch.Tensor) -> dict[str, Any]:
        size = self.geometry.parallel_expert_size
        expected = (self.settings.frame_count, size, size)
        if tuple(fields.shape[1:]) != expected:
            raise ValueError(f"Parallel router expects [B,{expected}], got {tuple(fields.shape)}")
        margin = self.geometry.active_margin
        local = (self.geometry.lane_size - size) // 2
        canvas = fields.new_zeros(
            fields.shape[0], self.geometry.canvas_size, self.geometry.canvas_size
        )
        phase_canvas = torch.ones_like(canvas, dtype=torch.complex64)
        shifted_fields = _translate(
            fields,
            *_random_shift(self.settings.input_shift_pixels, self.training),
            fill=0.0,
        )
        shifted_phase = _translate(
            _phase_modulation(
                self.raw_router_phase,
                settings=self.settings,
                training=self.training,
            ),
            *_random_shift(self.settings.phase_shift_pixels, self.training),
            fill=1.0 + 0.0j,
        )
        for lane, (top, left) in enumerate(self.geometry.lane_origins):
            y, x = margin + top + local, margin + left + local
            canvas[:, y : y + size, x : x + size] = shifted_fields[:, lane]
            phase_canvas[:, y : y + size, x : x + size] = shifted_phase[lane]
        detector = self.propagation(canvas.to(torch.complex64) * phase_canvas)
        detector = _translate(
            detector.abs().square().float(),
            *_random_shift(self.settings.ccd_shift_pixels, self.training),
            fill=0.0,
        )
        rows, lane_energy = [], []
        for top, left in self.geometry.lane_origins:
            lane = detector[
                :,
                margin + top : margin + top + self.geometry.lane_size,
                margin + left : margin + left + self.geometry.lane_size,
            ]
            lane_energy.append(lane.sum((-2, -1)))
            regions = [
                lane[:, y0:y1, x0:x1].sum((-2, -1))
                for y0, y1 in self.settings.parallel_router_intervals
                for x0, x1 in self.settings.parallel_router_intervals
            ]
            rows.append(torch.stack(regions, -1))
        energy = torch.stack(rows, 1)
        centered = energy - energy.mean(-1, keepdim=True)
        logits = centered / centered.square().mean(-1, keepdim=True).add(1.0e-8).sqrt()
        if self.training and self.settings.router_noise_std > 0.0:
            logits = logits + torch.randn_like(logits) * self.settings.router_noise_std
        probabilities = torch.softmax(
            logits / self.settings.parallel_router_temperature, dim=-1
        )
        weights, selected, indices = _sparse_top2(probabilities)
        captured = energy.sum(-1) / torch.stack(lane_energy, 1).clamp_min(1.0e-8)
        return {
            "probabilities": probabilities,
            "weights": weights,
            "selected_mask": selected,
            "selected_indices": indices,
            "capture_fraction": captured,
            "router_implementation": (
                f"optical_parallel{self.settings.frame_count}_energy_top2"
            ),
            **_routing_statistics(probabilities, selected),
        }


class OpticalRouterSerial(nn.Module):
    def __init__(self, settings: ExperimentSettings) -> None:
        super().__init__()
        self.settings = settings
        self.geometry = settings.geometry
        self.raw_router_phase = nn.Parameter(
            _spot_phase(
                self.geometry.serial_expert_size,
                self.geometry.active_size,
                settings.serial_router_intervals,
                settings,
            )
        )
        self.propagation = AngularSpectrum(settings)
        self.channel_standardizer = (
            nn.BatchNorm1d(4, affine=False, momentum=0.05)
            if settings.serial_router_channel_standardization
            else None
        )

    def forward(
        self, field: torch.Tensor, token_count: int | None = None
    ) -> dict[str, Any]:
        size = self.geometry.serial_expert_size
        if tuple(field.shape[1:]) != (size, size):
            raise ValueError(f"Serial router expects [B,{size},{size}]")
        router_input_size = self.settings.serial_router_input_size
        router_field = field
        if router_input_size != size:
            if token_count is None or not 0 < token_count <= size:
                raise ValueError(
                    "Compact serial router requires the actual sequence length"
                )
            # The serial field stores valid tokens in its first rows and pads
            # the rest with zeros. Crop those valid rows before resampling so
            # the 20% coherent zero-order footprint is centered rather than
            # biased toward the upper two detector windows.
            router_field = field[:, :token_count]
            router_field = F.adaptive_avg_pool2d(
                router_field.unsqueeze(1), (router_input_size, router_input_size)
            ).squeeze(1)
        input_offset = (self.geometry.canvas_size - router_input_size) // 2
        phase_offset = (self.geometry.canvas_size - size) // 2
        shifted = _translate(
            router_field,
            *_random_shift(self.settings.input_shift_pixels, self.training),
            fill=0.0,
        )
        canvas = F.pad(
            shifted,
            (
                input_offset,
                self.geometry.canvas_size - router_input_size - input_offset,
            )
            * 2,
        ).to(torch.complex64)
        phase_canvas = torch.ones_like(canvas)
        phase_canvas[
            :,
            phase_offset : phase_offset + size,
            phase_offset : phase_offset + size,
        ] = _translate(
            _phase_modulation(
                self.raw_router_phase,
                settings=self.settings,
                training=self.training,
            ),
            *_random_shift(self.settings.phase_shift_pixels, self.training),
            fill=1.0 + 0.0j,
        )
        detector = self.propagation(canvas * phase_canvas).abs().square().float()
        detector = _translate(
            detector,
            *_random_shift(self.settings.ccd_shift_pixels, self.training),
            fill=0.0,
        )
        margin = self.geometry.active_margin
        active = detector[
            :, margin : margin + self.geometry.active_size, margin : margin + self.geometry.active_size
        ]
        raw_energy = torch.stack(
            [
                active[:, y0:y1, x0:x1].sum((-2, -1))
                for y0, y1 in self.settings.serial_router_intervals
                for x0, x1 in self.settings.serial_router_intervals
            ],
            -1,
        )
        energy = raw_energy
        if self.settings.serial_router_flatfield_calibration:
            # A real setup obtains these four scalar gains once after loading
            # the router phase by displaying a spatially uniform amplitude
            # reference.  Recomputing that reference here keeps the simulated
            # gains consistent while the phase mask is still trainable.  This
            # is detector flat-field calibration, not an electronic router:
            # the sample-dependent routing signal remains the four measured
            # optical region energies.
            reference_canvas = torch.zeros_like(canvas[:1])
            reference_canvas[
                :,
                input_offset : input_offset + router_input_size,
                input_offset : input_offset + router_input_size,
            ] = 1.0
            reference_detector = self.propagation(
                reference_canvas * phase_canvas[:1]
            ).abs().square().float()
            reference_active = reference_detector[
                :,
                margin : margin + self.geometry.active_size,
                margin : margin + self.geometry.active_size,
            ]
            reference_energy = torch.stack(
                [
                    reference_active[:, y0:y1, x0:x1].sum((-2, -1))
                    for y0, y1 in self.settings.serial_router_intervals
                    for x0, x1 in self.settings.serial_router_intervals
                ],
                -1,
            )
            relative_gain = reference_energy / reference_energy.mean(
                -1, keepdim=True
            ).clamp_min(1.0e-8)
            energy = raw_energy / relative_gain.clamp_min(1.0e-4)
        router_signal = energy
        if self.channel_standardizer is not None:
            # Four-channel detector calibration only: no affine/trainable
            # parameters and no sample-dependent electronic routing network.
            # Running moments are stored with the checkpoint; on hardware they
            # can be re-estimated from a small calibration subset.
            router_signal = self.channel_standardizer(
                torch.log(energy.clamp_min(1.0e-8))
            )
        centered = router_signal - router_signal.mean(-1, keepdim=True)
        logits = centered / centered.square().mean(-1, keepdim=True).add(1.0e-8).sqrt()
        if self.training and self.settings.router_noise_std > 0.0:
            logits = logits + torch.randn_like(logits) * self.settings.router_noise_std
        probabilities = torch.softmax(
            logits / self.settings.serial_router_temperature, dim=-1
        )
        weights, selected, indices = _sparse_top2(probabilities)
        captured = raw_energy.sum(-1) / active.sum((-2, -1)).clamp_min(1.0e-8)
        return {
            "probabilities": probabilities,
            "weights": weights,
            "selected_mask": selected,
            "selected_indices": indices,
            "capture_fraction": captured,
            "router_implementation": (
                "optical_serial_energy_flatfield_standardized_top2"
                if self.settings.serial_router_flatfield_calibration
                and self.settings.serial_router_channel_standardization
                else (
                    "optical_serial_energy_flatfield_top2"
                    if self.settings.serial_router_flatfield_calibration
                    else (
                        "optical_serial_energy_channel_standardized_top2"
                        if self.settings.serial_router_channel_standardization
                        else "optical_serial_energy_top2"
                    )
                )
            ),
            **_routing_statistics(probabilities, selected),
        }


class CcdReadout(nn.Module):
    def __init__(self, token_count: int, detector_width: int, width: int) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((token_count, detector_width))
        self.norm = nn.LayerNorm(detector_width)
        self.output = nn.Linear(detector_width, width)

    def forward(self, intensity: torch.Tensor) -> torch.Tensor:
        pooled = self.pool(intensity.unsqueeze(1)).squeeze(1)
        return self.output(F.softplus(self.norm(pooled)))


class ParallelOpticalFeaturePath(nn.Module):
    def __init__(self, settings: ExperimentSettings) -> None:
        super().__init__()
        self.settings = settings
        self.geometry = settings.geometry
        size = self.geometry.parallel_expert_size
        self.width_to_field = nn.Linear(settings.model_width, size)
        self.tokens_to_field = nn.Linear(settings.token_count, size)
        _initialize_resampler(self.tokens_to_field)
        self.raw_expert_phase = nn.Parameter(
            torch.empty(settings.frame_count * 4, size, size)
        )
        self.raw_global_phase = nn.Parameter(
            torch.empty(self.geometry.active_size, self.geometry.active_size)
        )
        nn.init.normal_(self.raw_expert_phase, 0.0, settings.phase_init_std)
        nn.init.normal_(self.raw_global_phase, 0.0, settings.phase_init_std)
        self.propagation = AngularSpectrum(settings)
        self.expert_readout = CcdReadout(
            settings.token_count,
            settings.detector_projection_size,
            settings.model_width,
        )
        self.global_readout = CcdReadout(
            settings.token_count,
            settings.detector_projection_size,
            settings.model_width,
        )
        self.last_ccd: dict[str, torch.Tensor] = {}

    def fields(self, tokens: torch.Tensor) -> torch.Tensor:
        expected = (
            self.settings.frame_count,
            self.settings.token_count,
            self.settings.model_width,
        )
        if tuple(tokens.shape[1:]) != expected:
            raise ValueError(f"Vision tokens must be [B,{expected}]")
        encoded = F.softplus(self.width_to_field(tokens.float()))
        field = F.softplus(
            self.tokens_to_field(encoded.transpose(-2, -1))
        ).transpose(-2, -1)
        return field / field.square().mean((-2, -1), keepdim=True).sqrt().clamp_min(1.0e-6)

    def _normalize(self, value: torch.Tensor) -> torch.Tensor:
        value = value.float().clamp_min(0.0)
        mean = value.mean((-2, -1), keepdim=True).clamp_min(1.0e-6)
        return torch.log1p(
            self.settings.ccd_log_compression
            * (value / mean).clamp_max(self.settings.ccd_relative_clip)
        )

    def _read(self, detector: torch.Tensor, stage: str) -> torch.Tensor:
        margin = self.geometry.active_margin
        lanes = [
            self._normalize(
                detector[
                    :,
                    margin + top : margin + top + self.geometry.lane_size,
                    margin + left : margin + left + self.geometry.lane_size,
                ]
            )
            for top, left in self.geometry.lane_origins
        ]
        stacked = torch.stack(lanes, 1)
        self.last_ccd[stage] = stacked.detach()
        readout = self.expert_readout if stage == "vision_expert" else self.global_readout
        output = readout(stacked.flatten(0, 1))
        return output.reshape(
            detector.shape[0],
            self.settings.frame_count,
            self.settings.token_count,
            self.settings.model_width,
        )

    def expert(self, fields: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        size = self.geometry.parallel_expert_size
        margin = self.geometry.active_margin
        canvas = fields.new_zeros(
            fields.shape[0], self.geometry.canvas_size, self.geometry.canvas_size
        )
        shifted = _translate(
            fields,
            *_random_shift(self.settings.input_shift_pixels, self.training),
            fill=0.0,
        )
        for lane, (lane_top, lane_left) in enumerate(self.geometry.lane_origins):
            expert = 0
            for local_top in (0, self.geometry.parallel_expert_pitch):
                for local_left in (0, self.geometry.parallel_expert_pitch):
                    top = margin + lane_top + local_top
                    left = margin + lane_left + local_left
                    canvas[:, top : top + size, left : left + size] = (
                        shifted[:, lane] * weights[:, lane, expert, None, None]
                    )
                    expert += 1
        modulation = _translate(
            _phase_modulation(
                self.raw_expert_phase,
                settings=self.settings,
                training=self.training,
            ),
            *_random_shift(self.settings.phase_shift_pixels, self.training),
            fill=1.0 + 0.0j,
        )
        propagated = canvas.to(torch.complex64)
        index = 0
        for lane_top, lane_left in self.geometry.lane_origins:
            for local_top in (0, self.geometry.parallel_expert_pitch):
                for local_left in (0, self.geometry.parallel_expert_pitch):
                    top = margin + lane_top + local_top
                    left = margin + lane_left + local_left
                    propagated[:, top : top + size, left : left + size] *= modulation[index]
                    index += 1
        detector = self.propagation(propagated).abs().square().float()
        detector = _translate(
            detector,
            *_random_shift(self.settings.ccd_shift_pixels, self.training),
            fill=0.0,
        )
        return self._read(detector, "vision_expert")

    def global_path(self, fields: torch.Tensor) -> torch.Tensor:
        size = self.geometry.parallel_expert_size
        margin = self.geometry.active_margin
        local = (self.geometry.lane_size - size) // 2
        canvas = fields.new_zeros(
            fields.shape[0], self.geometry.canvas_size, self.geometry.canvas_size
        )
        shifted = _translate(
            fields,
            *_random_shift(self.settings.input_shift_pixels, self.training),
            fill=0.0,
        )
        for lane, (top, left) in enumerate(self.geometry.lane_origins):
            y, x = margin + top + local, margin + left + local
            canvas[:, y : y + size, x : x + size] = shifted[:, lane]
        phase = _translate(
            _phase_modulation(
                self.raw_global_phase,
                settings=self.settings,
                training=self.training,
            ),
            *_random_shift(self.settings.phase_shift_pixels, self.training),
            fill=1.0 + 0.0j,
        )
        propagated = canvas.to(torch.complex64)
        propagated[
            :, margin : margin + self.geometry.active_size, margin : margin + self.geometry.active_size
        ] *= phase
        detector = self.propagation(propagated).abs().square().float()
        detector = _translate(
            detector,
            *_random_shift(self.settings.ccd_shift_pixels, self.training),
            fill=0.0,
        )
        return self._read(detector, "vision_global")


class SerialOpticalFeaturePath(nn.Module):
    def __init__(self, settings: ExperimentSettings) -> None:
        super().__init__()
        self.settings = settings
        self.geometry = settings.geometry
        size = self.geometry.serial_expert_size
        self.width_to_field = nn.Linear(settings.model_width, size)
        self.raw_expert_phase = nn.Parameter(torch.empty(4, size, size))
        self.raw_global_phase = nn.Parameter(
            torch.empty(self.geometry.active_size, self.geometry.active_size)
        )
        nn.init.normal_(self.raw_expert_phase, 0.0, settings.phase_init_std)
        nn.init.normal_(self.raw_global_phase, 0.0, settings.phase_init_std)
        self.propagation = AngularSpectrum(settings)
        self.expert_readout = CcdReadout(
            settings.maximum_language_tokens,
            settings.detector_projection_size,
            settings.model_width,
        )
        self.global_readout = CcdReadout(
            settings.maximum_language_tokens,
            settings.detector_projection_size,
            settings.model_width,
        )
        self.last_ccd: dict[str, torch.Tensor] = {}

    def fields(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3 or tokens.shape[-1] != self.settings.model_width:
            raise ValueError("Language tokens must be [B,T,192]")
        if tokens.shape[1] > self.geometry.serial_expert_size:
            raise ValueError("Language token sequence does not fit the 109-row optical field")
        encoded = F.softplus(self.width_to_field(tokens.float()))
        field = encoded.new_zeros(
            tokens.shape[0],
            self.geometry.serial_expert_size,
            self.geometry.serial_expert_size,
        )
        field[:, : tokens.shape[1]] = encoded
        return field / field.square().mean((-2, -1), keepdim=True).sqrt().clamp_min(1.0e-6)

    def _read(
        self, detector: torch.Tensor, stage: str, token_count: int
    ) -> torch.Tensor:
        margin = self.geometry.active_margin
        active = detector[
            :, margin : margin + self.geometry.active_size, margin : margin + self.geometry.active_size
        ].float().clamp_min(0.0)
        mean = active.mean((-2, -1), keepdim=True).clamp_min(1.0e-6)
        normalized = torch.log1p(
            self.settings.ccd_log_compression
            * (active / mean).clamp_max(self.settings.ccd_relative_clip)
        )
        self.last_ccd[stage] = normalized.detach()
        readout = self.expert_readout if stage == "language_expert" else self.global_readout
        # The module is constructed for the formal maximum. Pooling directly to the
        # actual token count avoids padding tokens becoming learned observations.
        pooled = F.adaptive_avg_pool2d(
            normalized.unsqueeze(1),
            (token_count, self.settings.detector_projection_size),
        ).squeeze(1)
        return readout.output(F.softplus(readout.norm(pooled)))

    def expert(
        self, field: torch.Tensor, weights: torch.Tensor, token_count: int
    ) -> torch.Tensor:
        size = self.geometry.serial_expert_size
        margin = self.geometry.active_margin
        positions = (
            self.geometry.serial_expert_pitch,
            2 * self.geometry.serial_expert_pitch,
        )
        canvas = field.new_zeros(
            field.shape[0], self.geometry.canvas_size, self.geometry.canvas_size
        )
        shifted = _translate(
            field,
            *_random_shift(self.settings.input_shift_pixels, self.training),
            fill=0.0,
        )
        expert = 0
        for top in positions:
            for left in positions:
                y, x = margin + top, margin + left
                canvas[:, y : y + size, x : x + size] = (
                    shifted * weights[:, expert, None, None]
                )
                expert += 1
        phase = _translate(
            _phase_modulation(
                self.raw_expert_phase,
                settings=self.settings,
                training=self.training,
            ),
            *_random_shift(self.settings.phase_shift_pixels, self.training),
            fill=1.0 + 0.0j,
        )
        propagated = canvas.to(torch.complex64)
        expert = 0
        for top in positions:
            for left in positions:
                y, x = margin + top, margin + left
                propagated[:, y : y + size, x : x + size] *= phase[expert]
                expert += 1
        detector = self.propagation(propagated).abs().square().float()
        detector = _translate(
            detector,
            *_random_shift(self.settings.ccd_shift_pixels, self.training),
            fill=0.0,
        )
        return self._read(detector, "language_expert", token_count)

    def global_path(self, field: torch.Tensor, token_count: int) -> torch.Tensor:
        size = self.geometry.serial_expert_size
        margin = self.geometry.active_margin
        offset = (self.geometry.canvas_size - size) // 2
        shifted = _translate(
            field,
            *_random_shift(self.settings.input_shift_pixels, self.training),
            fill=0.0,
        )
        canvas = F.pad(
            shifted,
            (offset, self.geometry.canvas_size - size - offset) * 2,
        ).to(torch.complex64)
        phase = _translate(
            _phase_modulation(
                self.raw_global_phase,
                settings=self.settings,
                training=self.training,
            ),
            *_random_shift(self.settings.phase_shift_pixels, self.training),
            fill=1.0 + 0.0j,
        )
        canvas[
            :, margin : margin + self.geometry.active_size, margin : margin + self.geometry.active_size
        ] *= phase
        detector = self.propagation(canvas).abs().square().float()
        detector = _translate(
            detector,
            *_random_shift(self.settings.ccd_shift_pixels, self.training),
            fill=0.0,
        )
        return self._read(detector, "language_global", token_count)


class VisionElectronicRoute(nn.Module):
    """One explicit attention-free 7x7 depthwise/pointwise route."""

    def __init__(self, settings: ExperimentSettings) -> None:
        super().__init__()
        width = settings.model_width
        self.grid = settings.token_grid
        self.width = width
        self.norm = nn.LayerNorm(width)
        self.depthwise = nn.Conv2d(
            width, width, 5, padding=2, groups=width, bias=False
        )
        self.pointwise = nn.Conv2d(width, width, 1)
        self.skip_max = float(settings.electronic_skip_max)
        if settings.electronic_skip_enabled:
            ratio = settings.electronic_skip_initial / self.skip_max
            self.raw_skip = nn.Parameter(torch.atanh(torch.tensor(ratio)))

    @property
    def skip(self) -> torch.Tensor | None:
        raw = getattr(self, "raw_skip", None)
        return None if raw is None else self.skip_max * torch.tanh(raw)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        batch, frames, tokens, width = value.shape
        if tokens != self.grid * self.grid or width != self.width:
            raise ValueError("Vision electronic grid contract changed")
        image = self.norm(value).reshape(
            batch * frames, self.grid, self.grid, width
        ).permute(0, 3, 1, 2)
        output = self.pointwise(F.gelu(self.depthwise(image)))
        output = output.permute(0, 2, 3, 1).reshape(batch, frames, tokens, width)
        skip = self.skip
        return output if skip is None else output + skip * value


class LanguageElectronicRoute(nn.Module):
    """One explicit causal depthwise/pointwise Conv1D route; no attention."""

    def __init__(
        self,
        width: int,
        *,
        skip_enabled: bool = False,
        skip_initial: float = 0.0,
        skip_max: float = 1.0,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.depthwise = nn.Conv1d(width, width, 5, groups=width, bias=False)
        self.pointwise = nn.Conv1d(width, width, 1)
        self.skip_max = float(skip_max)
        if skip_enabled:
            ratio = float(skip_initial) / self.skip_max
            self.raw_skip = nn.Parameter(torch.atanh(torch.tensor(ratio)))

    @property
    def skip(self) -> torch.Tensor | None:
        raw = getattr(self, "raw_skip", None)
        return None if raw is None else self.skip_max * torch.tanh(raw)

    def forward(self, value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        sequence = self.norm(value).masked_fill(~mask.unsqueeze(-1), 0.0).transpose(1, 2)
        sequence = F.pad(sequence, (4, 0))
        output = self.pointwise(F.gelu(self.depthwise(sequence))).transpose(1, 2)
        skip = self.skip
        if skip is not None:
            output = output + skip * value
        return output.masked_fill(~mask.unsqueeze(-1), 0.0)


class _VisionResidualConvBlock(nn.Module):
    """One sequential attention-free block inside the sole electronic route."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.depthwise = nn.Conv2d(
            width, width, 5, padding=2, groups=width, bias=False
        )
        self.expand = nn.Conv2d(width, width * 2, 1)
        self.project = nn.Conv2d(width * 2, width, 1)
        self.raw_scale = nn.Parameter(torch.tensor(-2.0))
        # A newly appended residual block must be an exact identity at warm
        # start. Older blocks are restored from the checkpoint by exact name;
        # only genuinely new blocks keep this zero-output initialization.
        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.project.bias)

    def forward(self, value: torch.Tensor, grid: int) -> torch.Tensor:
        batch, frames, _, width = value.shape
        image = self.norm(value).reshape(
            batch * frames, grid, grid, width
        ).permute(0, 3, 1, 2)
        residual = self.project(F.gelu(self.expand(self.depthwise(image))))
        residual = residual.permute(0, 2, 3, 1).reshape_as(value)
        return value + torch.sigmoid(self.raw_scale) * residual


class _GlobalResponseNorm2d(nn.Module):
    """ConvNeXt-V2 style response normalization; no attention or token mixing."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        response = torch.linalg.vector_norm(value.float(), dim=(-2, -1), keepdim=True)
        normalized = response / response.mean(1, keepdim=True).clamp_min(1.0e-6)
        normalized = normalized.to(value.dtype)
        return value + self.gamma * (value * normalized) + self.beta


class _VisionLargeKernelResidualBlock(nn.Module):
    """One lightweight 7x7 ConvNeXt-style block inside the electronic route."""

    def __init__(self, width: int) -> None:
        super().__init__()
        expanded = width * 4
        self.norm = nn.LayerNorm(width)
        self.depthwise = nn.Conv2d(
            width, width, 7, padding=3, groups=width, bias=False
        )
        self.expand = nn.Conv2d(width, expanded, 1)
        self.grn = _GlobalResponseNorm2d(expanded)
        self.project = nn.Conv2d(expanded, width, 1)
        self.raw_scale = nn.Parameter(torch.tensor(-2.0))
        # Exact identity at checkpoint load. This makes the architecture search
        # reversible and prevents a random new block from erasing the formal run.
        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.project.bias)

    def forward(self, value: torch.Tensor, grid: int) -> torch.Tensor:
        batch, frames, _, width = value.shape
        image = self.norm(value).reshape(
            batch * frames, grid, grid, width
        ).permute(0, 3, 1, 2)
        residual = self.depthwise(image)
        residual = self.grn(F.gelu(self.expand(residual)))
        residual = self.project(residual)
        residual = residual.permute(0, 2, 3, 1).reshape_as(value)
        return value + torch.sigmoid(self.raw_scale) * residual


class VisionElectronicResidualRoute(nn.Module):
    """One electronic route with a checkpoint-compatible convolutional stem."""

    def __init__(self, settings: ExperimentSettings) -> None:
        super().__init__()
        self.grid = settings.token_grid
        self.width = settings.model_width
        # Keep these names and tensor shapes identical to VisionElectronicRoute.
        # The formal two-branch model can therefore inherit the already trained
        # electronic transform instead of silently randomizing the whole E path.
        self.norm = nn.LayerNorm(self.width)
        self.depthwise = nn.Conv2d(
            self.width,
            self.width,
            5,
            padding=2,
            groups=self.width,
            bias=False,
        )
        self.pointwise = nn.Conv2d(self.width, self.width, 1)
        blocks: list[nn.Module] = []
        for index in range(settings.electronic_route_depth - 1):
            # Block 0 retains the checkpoint-compatible 5x5 implementation.
            # Only newly appended blocks use the larger lightweight kernel.
            block_type = (
                _VisionLargeKernelResidualBlock
                if settings.electronic_route_variant == "residual_convnext"
                and index >= 1
                else _VisionResidualConvBlock
            )
            blocks.append(block_type(self.width))
        self.blocks = nn.ModuleList(blocks)
        self.skip_max = float(settings.electronic_skip_max)
        if settings.electronic_skip_enabled:
            ratio = settings.electronic_skip_initial / self.skip_max
            self.raw_skip = nn.Parameter(torch.atanh(torch.tensor(ratio)))

    @property
    def skip(self) -> torch.Tensor | None:
        raw = getattr(self, "raw_skip", None)
        return None if raw is None else self.skip_max * torch.tanh(raw)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        batch, frames, tokens, width = value.shape
        if tokens != self.grid * self.grid or width != self.width:
            raise ValueError("Vision electronic grid contract changed")
        identity = value
        image = self.norm(value).reshape(
            batch * frames, self.grid, self.grid, width
        ).permute(0, 3, 1, 2)
        value = self.pointwise(F.gelu(self.depthwise(image)))
        value = value.permute(0, 2, 3, 1).reshape(
            batch, frames, tokens, width
        )
        for block in self.blocks:
            value = block(value, self.grid)
        skip = self.skip
        if skip is not None:
            value = value + skip * identity
        return value


class _LanguageResidualConvBlock(nn.Module):
    """One sequential causal block inside the sole sequence electronic route."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.depthwise = nn.Conv1d(width, width, 5, groups=width, bias=False)
        self.expand = nn.Conv1d(width, width * 2, 1)
        self.project = nn.Conv1d(width * 2, width, 1)
        self.raw_scale = nn.Parameter(torch.tensor(-2.0))
        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.project.bias)

    def forward(self, value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        sequence = self.norm(value).masked_fill(~mask.unsqueeze(-1), 0.0)
        sequence = F.pad(sequence.transpose(1, 2), (4, 0))
        residual = self.project(F.gelu(self.expand(self.depthwise(sequence))))
        result = value + torch.sigmoid(self.raw_scale) * residual.transpose(1, 2)
        return result.masked_fill(~mask.unsqueeze(-1), 0.0)


class _LanguageLargeKernelResidualBlock(nn.Module):
    """Causal 7-tap counterpart of the lightweight vision residual block."""

    def __init__(self, width: int) -> None:
        super().__init__()
        expanded = width * 4
        self.norm = nn.LayerNorm(width)
        self.depthwise = nn.Conv1d(width, width, 7, groups=width, bias=False)
        self.expand = nn.Conv1d(width, expanded, 1)
        self.project = nn.Conv1d(expanded, width, 1)
        self.raw_scale = nn.Parameter(torch.tensor(-2.0))
        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.project.bias)

    def forward(self, value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        sequence = self.norm(value).masked_fill(~mask.unsqueeze(-1), 0.0)
        sequence = F.pad(sequence.transpose(1, 2), (6, 0))
        residual = self.project(F.gelu(self.expand(self.depthwise(sequence))))
        result = value + torch.sigmoid(self.raw_scale) * residual.transpose(1, 2)
        return result.masked_fill(~mask.unsqueeze(-1), 0.0)


class LanguageElectronicResidualRoute(nn.Module):
    """One causal electronic route with a compatible convolutional stem."""

    def __init__(self, settings: ExperimentSettings) -> None:
        super().__init__()
        self.width = settings.model_width
        # These three modules exactly match LanguageElectronicRoute so the
        # warm-start keeps the trained causal electronic transform.
        self.norm = nn.LayerNorm(self.width)
        self.depthwise = nn.Conv1d(
            self.width, self.width, 5, groups=self.width, bias=False
        )
        self.pointwise = nn.Conv1d(self.width, self.width, 1)
        blocks: list[nn.Module] = []
        for index in range(settings.electronic_route_depth - 1):
            block_type = (
                _LanguageLargeKernelResidualBlock
                if settings.electronic_route_variant == "residual_convnext"
                and index >= 1
                else _LanguageResidualConvBlock
            )
            blocks.append(block_type(self.width))
        self.blocks = nn.ModuleList(blocks)
        self.skip_max = float(settings.electronic_skip_max)
        if settings.electronic_skip_enabled:
            ratio = settings.electronic_skip_initial / self.skip_max
            self.raw_skip = nn.Parameter(torch.atanh(torch.tensor(ratio)))

    @property
    def skip(self) -> torch.Tensor | None:
        raw = getattr(self, "raw_skip", None)
        return None if raw is None else self.skip_max * torch.tanh(raw)

    def forward(self, value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        identity = value
        sequence = self.norm(value).masked_fill(
            ~mask.unsqueeze(-1), 0.0
        ).transpose(1, 2)
        sequence = F.pad(sequence, (4, 0))
        value = self.pointwise(F.gelu(self.depthwise(sequence))).transpose(1, 2)
        value = value.masked_fill(~mask.unsqueeze(-1), 0.0)
        for block in self.blocks:
            value = block(value, mask)
        skip = self.skip
        if skip is not None:
            value = value + skip * identity
        return value


class RmsConvexFusion(nn.Module):
    def __init__(self, settings: ExperimentSettings) -> None:
        super().__init__()
        self.minimum = settings.alpha_min
        self.maximum = settings.alpha_max
        self.epsilon = settings.fusion_epsilon
        initial = (settings.alpha_initial - self.minimum) / (
            self.maximum - self.minimum
        )
        self.raw_alpha = nn.Parameter(torch.logit(torch.tensor(initial)))
        self.last_diagnostics: dict[str, float] = {}

    @property
    def alpha(self) -> torch.Tensor:
        return self.minimum + (self.maximum - self.minimum) * torch.sigmoid(
            self.raw_alpha
        )

    def forward(
        self,
        electronic: torch.Tensor,
        optical: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if electronic.shape != optical.shape:
            raise ValueError("Fusion inputs must have identical shapes")
        if mask is None:
            valid = torch.ones_like(electronic[..., :1], dtype=torch.bool)
        else:
            if tuple(mask.shape) != tuple(electronic.shape[:-1]):
                raise ValueError("Fusion mask shape differs from token shape")
            valid = mask.unsqueeze(-1)
        valid_float = valid.float()
        denominator = (valid_float.sum(tuple(range(1, electronic.ndim))) * electronic.shape[-1]).clamp_min(1.0)

        def rms(value: torch.Tensor) -> torch.Tensor:
            numerator = (value.float().square() * valid_float).sum(
                tuple(range(1, value.ndim)), keepdim=True
            )
            shape = (value.shape[0],) + (1,) * (value.ndim - 1)
            return (numerator / denominator.reshape(shape)).sqrt().clamp_min(self.epsilon)

        re = rms(electronic).detach()
        ro = rms(optical).detach()
        mixture = (1.0 - self.alpha) * electronic.float() / re + self.alpha * optical.float() / ro
        mixture = mixture * valid_float
        rm = rms(mixture).detach()
        result = (re * mixture / rm).to(electronic.dtype)
        self.last_diagnostics = {
            "alpha": float(self.alpha.detach()),
            "electronic_rms": float(re.mean()),
            "optical_rms": float(ro.mean()),
            "output_to_electronic_rms": float((rms(result) / re).mean().detach()),
        }
        return result.masked_fill(~valid, 0.0)


def _masked_statistics(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = mask.unsqueeze(-1)
    count = valid.sum(1).clamp_min(1)
    mean = (value * valid).sum(1) / count
    variance = ((value - mean[:, None]).square() * valid).sum(1) / count
    maximum = value.masked_fill(~valid, -torch.inf).amax(1)
    maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
    return torch.cat((mean, variance.clamp_min(0.0).sqrt(), maximum), -1)


class SpatialReadout(nn.Module):
    def __init__(self, settings: ExperimentSettings) -> None:
        super().__init__()
        width, hidden = settings.model_width, settings.head_width
        self.frame = nn.Sequential(
            nn.LayerNorm(width * 3),
            nn.Linear(width * 3, hidden),
            nn.GELU(),
        )
        self.language = nn.Sequential(
            nn.LayerNorm(width * 3),
            nn.Linear(width * 3, hidden),
            nn.GELU(),
        )
        self.output = nn.Sequential(
            nn.LayerNorm(hidden * 4),
            nn.Linear(hidden * 4, hidden * 2),
            nn.GELU(),
            nn.Dropout(settings.dropout),
            nn.Linear(hidden * 2, 1),
        )

    def forward(
        self, vision: torch.Tensor, language: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        frame = torch.cat(
            (
                vision.mean(2),
                vision.float().std(2, unbiased=False).to(vision.dtype),
                vision.amax(2),
            ),
            -1,
        )
        frame = self.frame(frame)
        video = torch.cat(
            (
                frame.mean(1),
                frame.float().std(1, unbiased=False).to(frame.dtype),
                frame.amax(1),
            ),
            -1,
        )
        prompt = self.language(_masked_statistics(language, mask))
        return self.output(torch.cat((video, prompt), -1)).squeeze(-1)


class SpatialGridReadout(nn.Module):
    """Attention-free electronic head preserving the configured square token grid."""

    def __init__(self, settings: ExperimentSettings) -> None:
        super().__init__()
        width, hidden = settings.model_width, settings.head_width
        self.grid = settings.token_grid
        self.frame_count = settings.frame_count
        self.image_focus_max = settings.spatial_readout_image_focus_max
        if self.image_focus_max > 0.0:
            # Zero gives an exact warm start from the established all-sequence
            # readout. Fine-tuning may then emphasize the four sample-varying
            # image tokens without deleting the text-conditioned optical path.
            self.raw_image_focus = nn.Parameter(torch.zeros(()))
        spatial_width = 64
        self.token_norm = nn.LayerNorm(width)
        self.spatial_depthwise = nn.Conv2d(
            width, width, kernel_size=3, padding=1, groups=width, bias=False
        )
        self.spatial_projection = nn.Conv2d(width, spatial_width, kernel_size=1)
        self.frame = nn.Sequential(
            nn.LayerNorm(spatial_width * 3 * 3 * 2),
            nn.Linear(spatial_width * 3 * 3 * 2, hidden),
            nn.GELU(),
        )
        self.language = nn.Sequential(
            nn.LayerNorm(width * 3),
            nn.Linear(width * 3, hidden),
            nn.GELU(),
        )
        self.output = nn.Sequential(
            nn.LayerNorm(hidden * 4),
            nn.Linear(hidden * 4, hidden * 2),
            nn.GELU(),
            nn.Dropout(settings.dropout),
            nn.Linear(hidden * 2, 1),
        )

    def forward(
        self, vision: torch.Tensor, language: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        batch, frames, tokens, width = vision.shape
        if tokens != self.grid * self.grid:
            raise ValueError("Spatial-grid readout requires the configured square token grid")
        grid = self.token_norm(vision).reshape(
            batch * frames, self.grid, self.grid, width
        ).permute(0, 3, 1, 2)
        grid = F.gelu(self.spatial_depthwise(grid))
        grid = F.gelu(self.spatial_projection(grid))
        pooled = torch.cat(
            (
                F.adaptive_avg_pool2d(grid, 3),
                F.adaptive_max_pool2d(grid, 3),
            ),
            1,
        ).flatten(1)
        frame = self.frame(pooled).reshape(batch, frames, -1)
        video = torch.cat(
            (
                frame.mean(1),
                frame.float().std(1, unbiased=False).to(frame.dtype),
                frame.amax(1),
            ),
            -1,
        )
        all_sequence = self.language(_masked_statistics(language, mask))
        prompt = all_sequence
        if self.image_focus_max > 0.0:
            image_sequence = self.language(
                _masked_statistics(
                    language[:, : self.frame_count],
                    mask[:, : self.frame_count],
                )
            )
            image_focus = self.image_focus_max * torch.tanh(self.raw_image_focus)
            prompt = all_sequence + image_focus * (image_sequence - all_sequence)
        return self.output(torch.cat((video, prompt), -1)).squeeze(-1)


class SpatialMultiscaleReadout(nn.Module):
    """Attention-free multi-scale readout of the post-optical 2-D token field.

    The input remains the result of all four optical/electronic fusion stages.
    This head adds only depthwise convolutions, fixed local differences, pooling,
    and MLP layers after the optical graph; it cannot bypass any optical stage.
    """

    def __init__(self, settings: ExperimentSettings) -> None:
        super().__init__()
        width, hidden = settings.model_width, settings.head_width
        self.grid = settings.token_grid
        spatial_width = 96
        self.token_norm = nn.LayerNorm(width)
        self.depthwise3 = nn.Conv2d(
            width, width, kernel_size=3, padding=1, groups=width, bias=False
        )
        self.depthwise5 = nn.Conv2d(
            width, width, kernel_size=5, padding=2, groups=width, bias=False
        )
        self.spatial_projection = nn.Conv2d(width * 4, spatial_width, kernel_size=1)
        # Average and maximum summaries retain both global content and local
        # extrema at 1x1, 2x2, and 4x4 scales.
        pooled_width = spatial_width * 2 * (1 + 4 + 16)
        self.frame = nn.Sequential(
            nn.LayerNorm(pooled_width),
            nn.Linear(pooled_width, hidden * 2),
            nn.GELU(),
            nn.Dropout(settings.dropout),
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
        )
        self.language = nn.Sequential(
            nn.LayerNorm(width * 3),
            nn.Linear(width * 3, hidden),
            nn.GELU(),
        )
        self.output = nn.Sequential(
            nn.LayerNorm(hidden * 7),
            nn.Linear(hidden * 7, hidden * 2),
            nn.GELU(),
            nn.Dropout(settings.dropout),
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(
        self, vision: torch.Tensor, language: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        batch, frames, tokens, width = vision.shape
        if tokens != self.grid * self.grid:
            raise ValueError("Spatial multi-scale readout requires a square token grid")
        grid = self.token_norm(vision).reshape(
            batch * frames, self.grid, self.grid, width
        ).permute(0, 3, 1, 2)
        smooth = F.avg_pool2d(grid, kernel_size=3, stride=1, padding=1)
        local_residual = grid - smooth
        features = torch.cat(
            (
                grid,
                F.gelu(self.depthwise3(grid)),
                F.gelu(self.depthwise5(grid)),
                local_residual,
            ),
            1,
        )
        features = F.gelu(self.spatial_projection(features))
        pooled = torch.cat(
            tuple(
                pool(features, size).flatten(1)
                for size in (1, 2, 4)
                for pool in (F.adaptive_avg_pool2d, F.adaptive_max_pool2d)
            ),
            1,
        )
        frame = self.frame(pooled).reshape(batch, frames, -1)
        frame_difference = (frame[:, 1:] - frame[:, :-1]).abs()
        video = torch.cat(
            (
                frame.mean(1),
                frame.float().std(1, unbiased=False).to(frame.dtype),
                frame.amax(1),
                frame.amin(1),
                frame_difference.mean(1),
                frame_difference.amax(1),
            ),
            -1,
        )
        prompt = self.language(_masked_statistics(language, mask))
        return self.output(torch.cat((video, prompt), -1)).squeeze(-1)


class SpatialGridResidualReadout(SpatialGridReadout):
    """Existing trained grid readout plus a zero-start local correction.

    Attribute names and shapes inherited from :class:`SpatialGridReadout` are
    intentionally unchanged, so exact-name/shape warm start restores the full
    established predictor. The new branch starts at zero and can only learn a
    post-optical correction; no pre-optical feature bypass is introduced.
    """

    def __init__(self, settings: ExperimentSettings) -> None:
        super().__init__(settings)
        width, hidden = settings.model_width, settings.head_width
        self.residual_max = float(settings.spatial_residual_max)
        residual_width = 64
        self.residual_projection = nn.Conv2d(width * 3, residual_width, 1)
        self.residual_frame = nn.Sequential(
            nn.LayerNorm(residual_width * 3 + width * 3),
            nn.Linear(residual_width * 3 + width * 3, hidden),
            nn.GELU(),
            nn.Dropout(settings.dropout),
        )
        self.residual_output = nn.Sequential(
            nn.LayerNorm(hidden * 4),
            nn.Linear(hidden * 4, hidden),
            nn.GELU(),
            nn.Dropout(settings.dropout),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.residual_output[-1].weight)
        nn.init.zeros_(self.residual_output[-1].bias)

    def forward(
        self, vision: torch.Tensor, language: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        base_prediction = super().forward(vision, language, mask)
        batch, frames, tokens, width = vision.shape
        grid = self.token_norm(vision).reshape(
            batch * frames, self.grid, self.grid, width
        ).permute(0, 3, 1, 2)
        smooth = F.avg_pool2d(grid, kernel_size=3, stride=1, padding=1)
        local = grid - smooth
        gradient_x = F.pad(grid[..., 1:] - grid[..., :-1], (0, 1, 0, 0))
        detail = F.gelu(
            self.residual_projection(torch.cat((local, gradient_x, grid), 1))
        )
        frame_summary = torch.cat(
            (
                detail.mean((-2, -1)),
                detail.float().std((-2, -1), unbiased=False).to(detail.dtype),
                detail.amax((-2, -1)),
                local.square().mean((-2, -1)).sqrt(),
                gradient_x.abs().mean((-2, -1)),
                gradient_x.abs().amax((-2, -1)),
            ),
            -1,
        )
        frame = self.residual_frame(frame_summary).reshape(batch, frames, -1)
        video = torch.cat(
            (
                frame.mean(1),
                frame.float().std(1, unbiased=False).to(frame.dtype),
                frame.amax(1),
            ),
            -1,
        )
        prompt = self.language(_masked_statistics(language, mask))
        correction = self.residual_output(torch.cat((video, prompt), -1)).squeeze(-1)
        bounded_correction = self.residual_max * torch.tanh(
            correction / self.residual_max
        )
        return base_prediction + bounded_correction


class SpatialPyramidResidualReadout(SpatialGridReadout):
    """Warm-start grid predictor plus a richer post-optical local correction.

    The branch receives only the Vision output after the first two O/E/O
    stages and the multimodal sequence after all four stages.  It therefore
    cannot bypass the optical network.  Operations are limited to depthwise
    convolution, pointwise projection, fixed pooling and MLPs.
    """

    def __init__(self, settings: ExperimentSettings) -> None:
        super().__init__(settings)
        width, hidden = settings.model_width, settings.head_width
        self.residual_max = float(settings.spatial_residual_max)
        residual_width = 96
        self.residual_depthwise3 = nn.Conv2d(
            width, width, 3, padding=1, groups=width, bias=False
        )
        self.residual_depthwise5 = nn.Conv2d(
            width, width, 5, padding=2, groups=width, bias=False
        )
        self.residual_projection = nn.Conv2d(width * 3, residual_width, 1)
        pooled_width = residual_width * 2 * (1 + 4 + 16)
        self.residual_frame = nn.Sequential(
            nn.LayerNorm(pooled_width),
            nn.Linear(pooled_width, hidden),
            nn.GELU(),
            nn.Dropout(settings.dropout),
        )
        self.residual_output = nn.Sequential(
            nn.LayerNorm(hidden * 7),
            nn.Linear(hidden * 7, hidden * 2),
            nn.GELU(),
            nn.Dropout(settings.dropout),
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.residual_output[-1].weight)
        nn.init.zeros_(self.residual_output[-1].bias)

    def forward(
        self, vision: torch.Tensor, language: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        base_prediction = super().forward(vision, language, mask)
        batch, frames, tokens, width = vision.shape
        grid = self.token_norm(vision).reshape(
            batch * frames, self.grid, self.grid, width
        ).permute(0, 3, 1, 2)
        features = F.gelu(
            self.residual_projection(
                torch.cat(
                    (
                        grid,
                        F.gelu(self.residual_depthwise3(grid)),
                        F.gelu(self.residual_depthwise5(grid)),
                    ),
                    1,
                )
            )
        )
        pooled = torch.cat(
            tuple(
                pool(features, size).flatten(1)
                for size in (1, 2, 4)
                for pool in (F.adaptive_avg_pool2d, F.adaptive_max_pool2d)
            ),
            1,
        )
        frame = self.residual_frame(pooled).reshape(batch, frames, -1)
        difference = (frame[:, 1:] - frame[:, :-1]).abs()
        video = torch.cat(
            (
                frame.mean(1),
                frame.float().std(1, unbiased=False).to(frame.dtype),
                frame.amax(1),
                frame.amin(1),
                difference.mean(1),
                difference.amax(1),
            ),
            -1,
        )
        prompt = self.language(_masked_statistics(language, mask))
        correction = self.residual_output(torch.cat((video, prompt), -1)).squeeze(-1)
        bounded_correction = self.residual_max * torch.tanh(
            correction / self.residual_max
        )
        return base_prediction + bounded_correction


class _DilatedResidualConvStack(nn.Module):
    """Local-to-global 15x15 field with explicit local residual retention."""

    def __init__(self, input_width: int, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(input_width, channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=2, dilation=2)
        self.norm2 = nn.GroupNorm(8, channels)
        self.conv3 = nn.Conv2d(channels, channels, 3, padding=4, dilation=4)
        self.norm3 = nn.GroupNorm(8, channels)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        local = F.gelu(self.norm1(self.conv1(value)))
        medium = local + F.gelu(self.norm2(self.conv2(local)))
        return medium + F.gelu(self.norm3(self.conv3(medium)))


class _ReadoutLargeKernelRefiner(nn.Module):
    """A zero-start 7x7 depthwise refinement inside the existing MOS head."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            channels, channels, 7, padding=3, groups=channels, bias=False
        )
        self.norm = nn.GroupNorm(8, channels)
        self.expand = nn.Conv2d(channels, channels * 2, 1)
        self.project = nn.Conv2d(channels * 2, channels, 1)
        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.project.bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        correction = self.project(
            F.gelu(self.expand(self.norm(self.depthwise(value))))
        )
        return value + correction


class SpatialDeepResidualReadout(SpatialGridReadout):
    """Higher-capacity convolutional correction after all optical stages.

    This is a plain convolution/pooling/MLP readout. It contains no attention,
    recurrent unit, or Transformer.  The branch receives only tensors already
    produced by the four-stage optical/electronic graph.  Its final layer is
    zero initialized, preserving the warm-start predictor exactly.
    """

    def __init__(self, settings: ExperimentSettings) -> None:
        super().__init__(settings)
        width, hidden = settings.model_width, settings.head_width
        self.residual_max = float(settings.spatial_residual_max)
        channels = 128
        if settings.spatial_residual_receptive_field == "hybrid15":
            self.residual_conv = _DilatedResidualConvStack(width, channels)
        else:
            dilations = (
                (1, 2, 4)
                if settings.spatial_residual_receptive_field == "dilated15"
                else (1, 1, 1)
            )
            self.residual_conv = nn.Sequential(
                nn.Conv2d(
                    width, channels, 3, padding=dilations[0], dilation=dilations[0]
                ),
                nn.GroupNorm(8, channels),
                nn.GELU(),
                nn.Conv2d(
                    channels,
                    channels,
                    3,
                    padding=dilations[1],
                    dilation=dilations[1],
                ),
                nn.GroupNorm(8, channels),
                nn.GELU(),
                nn.Conv2d(
                    channels,
                    channels,
                    3,
                    padding=dilations[2],
                    dilation=dilations[2],
                ),
                nn.GroupNorm(8, channels),
                nn.GELU(),
            )
        self.large_kernel_refiner = (
            _ReadoutLargeKernelRefiner(channels)
            if settings.spatial_readout_refiner_enabled
            else nn.Identity()
        )
        self.moment_frame = None
        if settings.spatial_readout_moment_refiner_enabled:
            # Channel-wise spatial contrast and gradient energy are useful IQA
            # cues. The zero output projection makes this an exact warm start.
            moment_hidden = min(192, hidden // 2)
            self.moment_frame = nn.Sequential(
                nn.LayerNorm(channels * 3),
                nn.Linear(channels * 3, moment_hidden),
                nn.GELU(),
                nn.Dropout(settings.dropout),
                nn.Linear(moment_hidden, hidden),
            )
            nn.init.zeros_(self.moment_frame[-1].weight)
            nn.init.zeros_(self.moment_frame[-1].bias)
        pooled_width = channels * 2 * (1 + 4 + 16)
        self.residual_frame = nn.Sequential(
            nn.LayerNorm(pooled_width),
            nn.Linear(pooled_width, hidden),
            nn.GELU(),
            nn.Dropout(settings.dropout),
        )
        self.residual_language = nn.Sequential(
            nn.LayerNorm(width * 3),
            nn.Linear(width * 3, hidden // 2),
            nn.GELU(),
        )
        self.residual_output = nn.Sequential(
            nn.LayerNorm(hidden * 6 + hidden // 2),
            nn.Linear(hidden * 6 + hidden // 2, hidden * 2),
            nn.GELU(),
            nn.Dropout(settings.dropout),
            nn.Linear(hidden * 2, 1),
        )
        nn.init.zeros_(self.residual_output[-1].weight)
        nn.init.zeros_(self.residual_output[-1].bias)

    def _residual_features(
        self, vision: torch.Tensor, language: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        batch, frames, tokens, width = vision.shape
        grid = self.token_norm(vision).reshape(
            batch * frames, self.grid, self.grid, width
        ).permute(0, 3, 1, 2)
        feature = self.large_kernel_refiner(self.residual_conv(grid))
        pooled = torch.cat(
            tuple(
                pool(feature, size).flatten(1)
                for size in (1, 2, 4)
                for pool in (F.adaptive_avg_pool2d, F.adaptive_max_pool2d)
            ),
            1,
        )
        frame = self.residual_frame(pooled).reshape(batch, frames, -1)
        if self.moment_frame is not None:
            centered = feature - feature.mean((-2, -1), keepdim=True)
            contrast = centered.float().square().mean((-2, -1)).sqrt().to(feature.dtype)
            gradient_x = (feature[..., 1:] - feature[..., :-1]).abs().mean((-2, -1))
            gradient_y = (feature[..., 1:, :] - feature[..., :-1, :]).abs().mean((-2, -1))
            moments = torch.cat((contrast, gradient_x, gradient_y), -1)
            frame = frame + self.moment_frame(moments).reshape(batch, frames, -1)
        difference = (frame[:, 1:] - frame[:, :-1]).abs()
        video = torch.cat(
            (
                frame.mean(1),
                frame.float().std(1, unbiased=False).to(frame.dtype),
                frame.amax(1),
                frame.amin(1),
                difference.mean(1),
                difference.amax(1),
            ),
            -1,
        )
        prompt = self.residual_language(_masked_statistics(language, mask))
        return torch.cat((video, prompt), -1)

    def forward(
        self, vision: torch.Tensor, language: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        base_prediction = super().forward(vision, language, mask)
        correction = self.residual_output(
            self._residual_features(vision, language, mask)
        ).squeeze(-1)
        correction = self.residual_max * torch.tanh(correction / self.residual_max)
        return base_prediction + correction


class SpatialWeightedLevelResidualReadout(SpatialDeepResidualReadout):
    """One post-optical head using five ordered levels for a bounded correction.

    This mirrors the frozen-Qwen baseline's useful inductive bias: softmax
    probabilities over Bad/Poor/Fair/Good/Excellent become a continuous score
    by a weighted sum.  The five scores correct the already trained scalar
    readout, giving an exact warm start. Every input is after the four optical
    stages; no raw-frame, pre-optical, attention, or Transformer bypass exists.
    """

    def __init__(self, settings: ExperimentSettings) -> None:
        super().__init__(settings)
        input_width = self.residual_output[-1].in_features
        self.residual_output[-1] = nn.Linear(input_width, 5)
        nn.init.zeros_(self.residual_output[-1].weight)
        nn.init.zeros_(self.residual_output[-1].bias)
        self.register_buffer(
            "level_scores",
            torch.linspace(-self.residual_max, self.residual_max, 5),
        )
        self.last_level_logits: torch.Tensor | None = None
        self.last_level_base_prediction: torch.Tensor | None = None

    def forward(
        self, vision: torch.Tensor, language: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        base_prediction = SpatialGridReadout.forward(self, vision, language, mask)
        logits = self.residual_output(
            self._residual_features(vision, language, mask)
        )
        probabilities = logits.float().softmax(-1).to(logits.dtype)
        correction = probabilities @ self.level_scores.to(probabilities.dtype)
        self.last_level_logits = logits
        self.last_level_base_prediction = base_prediction
        return base_prediction + correction


class SpatialCompactWeightedReadout(nn.Module):
    """Compact five-level MOS head operating only after all optical stages.

    The former deep readout expanded every frame to a 5,376-value pyramid and
    then used two wide fully-connected mappings.  This replacement keeps a
    small spatial convolutional field, pools only to 1x1 and 2x2, and retains
    mean/std/extrema/change summaries over the four frames.  It contains only
    project-owned convolution, pooling and fully-connected operations: no
    attention, Transformer, recurrent unit, or named pretrained backbone.
    """

    def __init__(self, settings: ExperimentSettings) -> None:
        super().__init__()
        width = settings.model_width
        channels = settings.spatial_compact_channels
        frame_width = settings.spatial_compact_frame_width
        language_width = settings.spatial_compact_language_width
        head_width = settings.spatial_compact_head_width
        self.grid = settings.token_grid
        self.token_norm = nn.LayerNorm(width)
        self.spatial_projection = nn.Conv2d(width, channels, 1)
        self.spatial_local = nn.Conv2d(
            channels, channels, 5, padding=2, groups=channels, bias=False
        )
        self.spatial_context = nn.Conv2d(
            channels,
            channels,
            3,
            padding=2,
            dilation=2,
            groups=channels,
            bias=False,
        )
        pooled_width = channels * 2 * (1 + 4)
        self.frame = nn.Sequential(
            nn.LayerNorm(pooled_width),
            nn.Linear(pooled_width, frame_width),
            nn.GELU(),
            nn.Dropout(settings.dropout),
        )
        self.language = nn.Sequential(
            nn.LayerNorm(width * 3),
            nn.Linear(width * 3, language_width),
            nn.GELU(),
        )
        joint_width = frame_width * 6 + language_width
        self.output = nn.Sequential(
            nn.LayerNorm(joint_width),
            nn.Linear(joint_width, head_width),
            nn.GELU(),
            nn.Dropout(settings.dropout),
            nn.Linear(head_width, 5),
        )
        self.register_buffer(
            "level_scores",
            torch.linspace(
                settings.spatial_level_score_min,
                settings.spatial_level_score_max,
                5,
            ),
        )
        self.last_level_logits: torch.Tensor | None = None
        self.last_level_base_prediction: torch.Tensor | None = None

    def forward(
        self, vision: torch.Tensor, language: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        batch, frames, tokens, width = vision.shape
        if tokens != self.grid * self.grid:
            raise ValueError("Compact Spatial readout requires a square token grid")
        grid = self.token_norm(vision).reshape(
            batch * frames, self.grid, self.grid, width
        ).permute(0, 3, 1, 2)
        feature = F.gelu(self.spatial_projection(grid))
        feature = feature + F.gelu(self.spatial_local(feature))
        feature = feature + F.gelu(self.spatial_context(feature))
        pooled = torch.cat(
            tuple(
                pool(feature, size).flatten(1)
                for size in (1, 2)
                for pool in (F.adaptive_avg_pool2d, F.adaptive_max_pool2d)
            ),
            1,
        )
        frame = self.frame(pooled).reshape(batch, frames, -1)
        difference = (frame[:, 1:] - frame[:, :-1]).abs()
        video = torch.cat(
            (
                frame.mean(1),
                frame.float().std(1, unbiased=False).to(frame.dtype),
                frame.amax(1),
                frame.amin(1),
                difference.mean(1),
                difference.amax(1),
            ),
            -1,
        )
        prompt = self.language(_masked_statistics(language, mask))
        logits = self.output(torch.cat((video, prompt), -1))
        probabilities = logits.float().softmax(-1).to(logits.dtype)
        prediction = probabilities @ self.level_scores.to(probabilities.dtype)
        self.last_level_logits = logits
        self.last_level_base_prediction = torch.zeros_like(prediction)
        return prediction


class SpatialPrunedGridCompactResidualReadout(SpatialGridReadout):
    """Structured-pruned grid head plus a small post-optical correction.

    The established grid branch is retained, but its 1,024-neuron terminal
    hidden layer is reduced to ``spatial_compact_head_width`` neurons.  The
    training utility initializes those neurons from the most influential
    source units.  A zero-start convolutional correction then recovers detail
    using only tensors that have already traversed the four optical stages.
    """

    def __init__(self, settings: ExperimentSettings) -> None:
        super().__init__(settings)
        width, hidden = settings.model_width, settings.head_width
        pruned_width = settings.spatial_compact_head_width
        self.output[1] = nn.Linear(hidden * 4, pruned_width)
        self.output[-1] = nn.Linear(pruned_width, 1)
        self.residual_max = float(settings.spatial_residual_max)
        self.residual_scale = float(settings.spatial_compact_residual_scale)
        channels, frame_width = 64, 128
        self.compact_projection = nn.Conv2d(width, channels, 1)
        self.compact_local = nn.Conv2d(
            channels, channels, 5, padding=2, groups=channels, bias=False
        )
        pooled_width = channels * 2 * (1 + 4)
        self.compact_frame = nn.Sequential(
            nn.LayerNorm(pooled_width),
            nn.Linear(pooled_width, frame_width),
            nn.GELU(),
            nn.Dropout(settings.dropout),
        )
        self.compact_output = nn.Sequential(
            nn.LayerNorm(frame_width * 6 + hidden),
            nn.Linear(frame_width * 6 + hidden, 256),
            nn.GELU(),
            nn.Dropout(settings.dropout),
            nn.Linear(256, 1),
        )
        nn.init.zeros_(self.compact_output[-1].weight)
        nn.init.zeros_(self.compact_output[-1].bias)

    def forward(
        self, vision: torch.Tensor, language: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        base_prediction = SpatialGridReadout.forward(self, vision, language, mask)
        batch, frames, tokens, width = vision.shape
        grid = self.token_norm(vision).reshape(
            batch * frames, self.grid, self.grid, width
        ).permute(0, 3, 1, 2)
        feature = F.gelu(self.compact_projection(grid))
        feature = feature + F.gelu(self.compact_local(feature))
        pooled = torch.cat(
            tuple(
                pool(feature, size).flatten(1)
                for size in (1, 2)
                for pool in (F.adaptive_avg_pool2d, F.adaptive_max_pool2d)
            ),
            1,
        )
        frame = self.compact_frame(pooled).reshape(batch, frames, -1)
        difference = (frame[:, 1:] - frame[:, :-1]).abs()
        video = torch.cat(
            (
                frame.mean(1),
                frame.float().std(1, unbiased=False).to(frame.dtype),
                frame.amax(1),
                frame.amin(1),
                difference.mean(1),
                difference.amax(1),
            ),
            -1,
        )
        prompt = self.language(_masked_statistics(language, mask))
        correction = self.compact_output(torch.cat((video, prompt), -1)).squeeze(-1)
        correction = self.residual_max * torch.tanh(correction / self.residual_max)
        return base_prediction + self.residual_scale * correction


class _CrossFrameSpatialBlock(nn.Module):
    """A small ConvNeXt-style block without attention or a new input branch."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            channels, channels, 5, padding=2, groups=channels, bias=False
        )
        self.norm = nn.GroupNorm(8, channels)
        self.expand = nn.Conv2d(channels, channels * 2, 1)
        self.project = nn.Conv2d(channels * 2, channels, 1)
        self.scale = nn.Parameter(torch.tensor(0.10))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = self.project(
            F.gelu(self.expand(self.norm(self.depthwise(value))))
        )
        return value + self.scale * residual


class SpatialCrossFrameResidualReadout(SpatialWeightedLevelResidualReadout):
    """Warm-start weighted readout plus a lightweight spatial-video correction.

    Every input has already passed through all four optical/electronic stages.
    Cross-frame statistics are computed at corresponding 14x14 locations before
    spatial pooling, retaining persistent blur/noise/texture evidence that is
    lost when each frame is pooled independently. The final layer starts at
    exactly zero, so loading the formal weighted checkpoint is functionally
    identical until this small correction is trained.
    """

    def __init__(self, settings: ExperimentSettings) -> None:
        super().__init__(settings)
        width = settings.model_width
        channels = 32
        hidden = min(128, settings.head_width // 2)
        prompt_width = 64
        self.crossframe_norm = nn.LayerNorm(width)
        # Per-location mean/std/max/min plus mean absolute consecutive change.
        self.crossframe_projection = nn.Conv2d(width * 5, channels, 1)
        self.crossframe_blocks = nn.Sequential(
            _CrossFrameSpatialBlock(channels),
            _CrossFrameSpatialBlock(channels),
        )
        pooled_width = channels * 2 * (1 + 4 + 16)
        self.crossframe_spatial = nn.Sequential(
            nn.LayerNorm(pooled_width),
            nn.Linear(pooled_width, hidden),
            nn.GELU(),
            nn.Dropout(settings.dropout),
        )
        self.crossframe_language = nn.Sequential(
            nn.LayerNorm(width * 3),
            nn.Linear(width * 3, prompt_width),
            nn.GELU(),
        )
        self.crossframe_output = nn.Sequential(
            nn.LayerNorm(hidden + prompt_width + 1),
            nn.Linear(hidden + prompt_width + 1, hidden),
            nn.GELU(),
            nn.Dropout(settings.dropout),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.crossframe_output[-1].weight)
        nn.init.zeros_(self.crossframe_output[-1].bias)

    def forward(
        self, vision: torch.Tensor, language: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        base_prediction = super().forward(vision, language, mask)
        batch, frames, tokens, width = vision.shape
        if tokens != self.grid * self.grid:
            raise ValueError("Cross-frame readout requires a square token grid")
        grid = self.crossframe_norm(vision).reshape(
            batch, frames, self.grid, self.grid, width
        ).permute(0, 1, 4, 2, 3)
        consecutive = (
            (grid[:, 1:] - grid[:, :-1]).abs().mean(1)
            if frames > 1
            else torch.zeros_like(grid[:, 0])
        )
        summary = torch.cat(
            (
                grid.mean(1),
                grid.float().std(1, unbiased=False).to(grid.dtype),
                grid.amax(1),
                grid.amin(1),
                consecutive,
            ),
            1,
        )
        feature = F.gelu(self.crossframe_projection(summary))
        feature = self.crossframe_blocks(feature)
        pooled = torch.cat(
            tuple(
                pool(feature, size).flatten(1)
                for size in (1, 2, 4)
                for pool in (F.adaptive_avg_pool2d, F.adaptive_max_pool2d)
            ),
            1,
        )
        spatial = self.crossframe_spatial(pooled)
        prompt = self.crossframe_language(_masked_statistics(language, mask))
        raw = self.crossframe_output(
            torch.cat((spatial, prompt, base_prediction.unsqueeze(-1)), -1)
        ).squeeze(-1)
        correction = self.residual_max * torch.tanh(raw / self.residual_max)
        return base_prediction + correction


class SpatialDualLevelResidualReadout(SpatialWeightedLevelResidualReadout):
    """Five-level correction plus a small complementary scalar regressor.

    Both predictions consume exactly the same post-optical tensor inside one
    readout.  The scalar output is zero initialized, so this is an exact
    warm-start of the formal five-level model rather than another branch.
    """

    def __init__(self, settings: ExperimentSettings) -> None:
        super().__init__(settings)
        input_width = self.residual_output[1].in_features
        hidden = min(256, settings.head_width)
        self.dual_output = nn.Sequential(
            nn.LayerNorm(input_width),
            nn.Linear(input_width, hidden),
            nn.GELU(),
            nn.Dropout(settings.dropout),
            nn.Linear(hidden, 1),
        )
        self.dual_gate_logit = nn.Parameter(torch.tensor(-1.38629436))
        nn.init.zeros_(self.dual_output[-1].weight)
        nn.init.zeros_(self.dual_output[-1].bias)

    def forward(
        self, vision: torch.Tensor, language: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        base_prediction = SpatialGridReadout.forward(self, vision, language, mask)
        features = self._residual_features(vision, language, mask)
        logits = self.residual_output(features)
        probabilities = logits.float().softmax(-1).to(logits.dtype)
        level_correction = probabilities @ self.level_scores.to(probabilities.dtype)
        scalar_raw = self.dual_output(features).squeeze(-1)
        scalar_correction = self.residual_max * torch.tanh(
            scalar_raw / self.residual_max
        )
        scalar_gate = self.dual_gate_logit.sigmoid().to(scalar_correction.dtype)
        self.last_level_logits = logits
        self.last_level_base_prediction = base_prediction
        return base_prediction + level_correction + scalar_gate * scalar_correction


class SpatialWeightedLevelAbsoluteReadout(SpatialDeepResidualReadout):
    """Five ordered logits directly predict normalized MOS.

    This is the strict logits-only counterpart of
    :class:`SpatialWeightedLevelResidualReadout`: it uses the same post-optical
    convolutional features but does not consume or fuse the legacy scalar
    prediction. The fixed anchors span the training MOS range in normalized
    coordinates and keep the result continuous and ordered.
    """

    def __init__(self, settings: ExperimentSettings) -> None:
        super().__init__(settings)
        input_width = self.residual_output[-1].in_features
        self.residual_output[-1] = nn.Linear(input_width, 5)
        nn.init.zeros_(self.residual_output[-1].weight)
        nn.init.zeros_(self.residual_output[-1].bias)
        self.register_buffer(
            "level_scores",
            torch.linspace(
                settings.spatial_level_score_min,
                settings.spatial_level_score_max,
                5,
            ),
        )
        self.last_level_logits: torch.Tensor | None = None
        self.last_level_base_prediction: torch.Tensor | None = None

    def forward(
        self, vision: torch.Tensor, language: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        logits = self.residual_output(
            self._residual_features(vision, language, mask)
        )
        probabilities = logits.float().softmax(-1).to(logits.dtype)
        prediction = probabilities @ self.level_scores.to(probabilities.dtype)
        self.last_level_logits = logits
        self.last_level_base_prediction = torch.zeros_like(prediction)
        return prediction


class SpatialWeightedLevelBlendReadout(SpatialDeepResidualReadout):
    """Convex fusion of the legacy scalar and an absolute five-level score."""

    def __init__(self, settings: ExperimentSettings) -> None:
        super().__init__(settings)
        input_width = self.residual_output[-1].in_features
        self.residual_output[-1] = nn.Linear(input_width, 5)
        nn.init.zeros_(self.residual_output[-1].weight)
        nn.init.zeros_(self.residual_output[-1].bias)
        self.register_buffer(
            "level_scores",
            torch.linspace(
                settings.spatial_level_score_min,
                settings.spatial_level_score_max,
                5,
            ),
        )
        initial = float(settings.spatial_level_blend_initial)
        self.residual_blend_logit = nn.Parameter(
            torch.tensor(math.log(initial / (1.0 - initial)))
        )
        self.last_level_logits: torch.Tensor | None = None
        self.last_level_base_prediction: torch.Tensor | None = None
        self.last_level_blend: torch.Tensor | None = None

    def forward(
        self, vision: torch.Tensor, language: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        base_prediction = SpatialGridReadout.forward(self, vision, language, mask)
        logits = self.residual_output(
            self._residual_features(vision, language, mask)
        )
        probabilities = logits.float().softmax(-1).to(logits.dtype)
        level_prediction = probabilities @ self.level_scores.to(probabilities.dtype)
        blend = self.residual_blend_logit.sigmoid().to(level_prediction.dtype)
        prediction = base_prediction.lerp(level_prediction, blend)
        self.last_level_logits = logits
        self.last_level_base_prediction = torch.zeros_like(prediction)
        self.last_level_blend = blend
        return prediction


class QualitySpatialAdapter(nn.Module):
    """Small attention-free 2-D input head for the cached 14 quality maps.

    The cache still comes from the same decoded frames and contains no learned
    output.  Convolutions here are trainable and run before the first optical
    layer, so quality evidence cannot bypass the four-stage O/E/O path.
    """

    def __init__(self, settings: ExperimentSettings) -> None:
        super().__init__()
        self.grid = settings.token_grid
        self.input_width = settings.quality_input_width
        hidden = 64
        self.input_norm = nn.GroupNorm(2, self.input_width)
        self.conv1 = nn.Conv2d(self.input_width, hidden, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, hidden)
        self.conv2 = nn.Conv2d(hidden, hidden, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, hidden)
        self.project = nn.Conv2d(hidden, settings.model_width, 1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        batch, frames, tokens, channels = value.shape
        if tokens != self.grid * self.grid or channels != self.input_width:
            raise ValueError("Quality-spatial adapter input contract changed")
        image = value.reshape(batch * frames, self.grid, self.grid, channels).permute(
            0, 3, 1, 2
        )
        image = F.gelu(self.norm1(self.conv1(self.input_norm(image))))
        image = F.gelu(self.norm2(self.conv2(image)))
        image = self.project(image)
        return image.permute(0, 2, 3, 1).reshape(
            batch, frames, tokens, self.project.out_channels
        )


class CompactFrameStemRefiner(nn.Module):
    """Cheap 14x14 spatial correction used for progressive compression.

    The unit is a pair of depthwise convolutions followed by pointwise mixing.
    Its last projection starts at zero, so inserting any number of units leaves
    the established Conv5 output bit-identical before distillation.  It is not
    a predictor and has no path around the four optical/electronic stages.
    """

    def __init__(self, width: int = 192, hidden: int = 256) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(24, width)
        self.depthwise3 = nn.Conv2d(
            width, width, 3, padding=1, groups=width, bias=False
        )
        self.depthwise5 = nn.Conv2d(
            width, width, 5, padding=2, groups=width, bias=False
        )
        self.mix = nn.Conv2d(width * 2, hidden, 1)
        self.project = nn.Conv2d(hidden, width, 1)
        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.project.bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        normalized = self.norm(value)
        context = torch.cat(
            (self.depthwise3(normalized), self.depthwise5(normalized)), dim=1
        )
        return value + self.project(F.gelu(self.mix(context)))


class TrainableQualityFrameStem(nn.Module):
    """Five plain convolutions mapping four RGB frames to 14x14x192 tokens."""

    def __init__(self, refiner_depth: int = 0) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(14, 48, 3, stride=2, padding=1)
        self.norm1 = nn.GroupNorm(8, 48)
        self.conv2 = nn.Conv2d(48, 64, 3, stride=2, padding=1)
        self.norm2 = nn.GroupNorm(8, 64)
        self.conv3 = nn.Conv2d(64, 96, 3, stride=2, padding=1)
        self.norm3 = nn.GroupNorm(12, 96)
        self.conv4 = nn.Conv2d(96, 96, 3, padding=1)
        self.norm4 = nn.GroupNorm(12, 96)
        self.conv5 = nn.Conv2d(96, 192, 3, stride=2, padding=1)
        self.norm5 = nn.GroupNorm(24, 192)
        self.refiners = nn.ModuleList(
            [CompactFrameStemRefiner() for _ in range(refiner_depth)]
        )
        sobel_x = torch.tensor(
            ((-1.0, 0.0, 1.0), (-2.0, 0.0, 2.0), (-1.0, 0.0, 1.0))
        ) / 4.0
        sobel_y = sobel_x.t().contiguous()
        laplacian = torch.tensor(
            ((0.0, 1.0, 0.0), (1.0, -4.0, 1.0), (0.0, 1.0, 0.0))
        ) / 4.0
        self.register_buffer(
            "sobel_x", sobel_x.view(1, 1, 3, 3), persistent=False
        )
        self.register_buffer(
            "sobel_y", sobel_y.view(1, 1, 3, 3), persistent=False
        )
        self.register_buffer(
            "laplacian", laplacian.view(1, 1, 3, 3), persistent=False
        )

    def quality_channels(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.ndim != 5 or tuple(frames.shape[1:3]) != (4, 3):
            raise ValueError("Frame stem expects [B,4,3,H,W]")
        batch, frame_count, _, height, width = frames.shape
        rgb = frames.float().div(255.0)
        luminance = (
            0.2989 * rgb[:, :, 0:1]
            + 0.5870 * rgb[:, :, 1:2]
            + 0.1140 * rgb[:, :, 2:3]
        )
        flat = luminance.flatten(0, 1)
        padded3 = F.pad(flat, (1, 1, 1, 1), mode="reflect")
        sobel_x = F.conv2d(padded3, self.sobel_x)
        sobel_y = F.conv2d(padded3, self.sobel_y)
        gradient = torch.sqrt(sobel_x.square() + sobel_y.square() + 1.0e-12)
        laplacian = F.conv2d(padded3, self.laplacian).abs()
        padded5 = F.pad(flat, (2, 2, 2, 2), mode="reflect")
        local_mean = F.avg_pool2d(padded5, 5, stride=1)
        local_square_mean = F.avg_pool2d(padded5.square(), 5, stride=1)
        local_std = (local_square_mean - local_mean.square()).clamp_min(0.0).sqrt()
        shape = (batch, frame_count, 1, height, width)
        sobel_x, sobel_y = sobel_x.reshape(shape), sobel_y.reshape(shape)
        gradient, laplacian = gradient.reshape(shape), laplacian.reshape(shape)
        local_std = local_std.reshape(shape)
        saturation = rgb.amax(2, keepdim=True) - rgb.amin(2, keepdim=True)
        temporal = torch.zeros_like(luminance)
        temporal[:, 1:] = (luminance[:, 1:] - luminance[:, :-1]).abs()
        y = torch.linspace(
            -1.0, 1.0, height, device=rgb.device, dtype=rgb.dtype
        ).view(1, 1, 1, height, 1).expand(batch, frame_count, 1, height, width)
        x = torch.linspace(
            -1.0, 1.0, width, device=rgb.device, dtype=rgb.dtype
        ).view(1, 1, 1, 1, width).expand(batch, frame_count, 1, height, width)
        time = torch.linspace(
            -1.0, 1.0, frame_count, device=rgb.device, dtype=rgb.dtype
        ).view(1, frame_count, 1, 1, 1).expand(
            batch, frame_count, 1, height, width
        )
        return torch.cat(
            (
                rgb,
                luminance,
                sobel_x,
                sobel_y,
                gradient,
                laplacian,
                local_std,
                saturation,
                temporal,
                x,
                y,
                time,
            ),
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
        for refiner in self.refiners:
            value = refiner(value)
        return value.flatten(2).transpose(1, 2).reshape(
            batch, frame_count, -1, value.shape[1]
        )


class QualitySpatialRefiner(nn.Module):
    """Zero-start spatial correction for a declared quality tensor.

    In the strict two-branch profile this module lives entirely inside E1 and
    refines only the existing electronic quality residual.  Its final
    projection is initialized to zero, so adding it to a warm-started checkpoint
    preserves every prediction exactly until optimization begins.  It contains
    only normalization and convolutions; no attention or MOS-readout bypass is
    introduced.
    """

    def __init__(self, settings: ExperimentSettings) -> None:
        super().__init__()
        self.grid = settings.token_grid
        self.width = settings.model_width
        self.maximum = float(settings.quality_refiner_max)
        self.norm = nn.LayerNorm(self.width)
        self.depthwise3 = nn.Conv2d(
            self.width, self.width, 3, padding=1, groups=self.width, bias=False
        )
        self.depthwise5 = nn.Conv2d(
            self.width, self.width, 5, padding=2, groups=self.width, bias=False
        )
        self.project = nn.Conv2d(self.width * 2, self.width, 1)
        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.project.bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        batch, frames, tokens, width = value.shape
        if tokens != self.grid * self.grid or width != self.width:
            raise ValueError("Quality refiner input contract changed")
        grid = self.norm(value).reshape(
            batch * frames, self.grid, self.grid, width
        ).permute(0, 3, 1, 2)
        correction = self.project(
            torch.cat(
                (
                    F.gelu(self.depthwise3(grid)),
                    F.gelu(self.depthwise5(grid)),
                ),
                1,
            )
        )
        correction = self.maximum * torch.tanh(correction / self.maximum)
        correction = correction.permute(0, 2, 3, 1).reshape_as(value)
        return value + correction


class FrozenVGGSpatialCorrection(nn.Module):
    """Bounded, zero-start correction from a frozen plain-convolution front.

    The cached 14x14 VGG16 tokens are produced only by sequential convolution,
    ReLU and max-pooling layers.  This adapter contains no attention or
    Transformer and injects its result before optical stage one, so it cannot
    bypass the required four-stage optical/electronic path.
    """

    def __init__(self, settings: ExperimentSettings) -> None:
        super().__init__()
        width = settings.model_width
        self.maximum = float(settings.vgg_correction_max)
        self.mode = settings.vgg_correction_mode
        input_width = 512 if self.mode == "local" else 512 * 4
        self.adapter = nn.Sequential(
            nn.LayerNorm(input_width),
            nn.Linear(input_width, width * 2),
            nn.GELU(),
            nn.Linear(width * 2, width),
        )
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        value = tokens.float()
        if self.mode == "context":
            batch, frames, token_count, channels = value.shape
            grid_size = math.isqrt(token_count)
            if grid_size * grid_size != token_count:
                raise ValueError("Context VGG correction requires a square token grid")
            grid = value.reshape(batch * frames, grid_size, grid_size, channels).permute(0, 3, 1, 2)
            local = F.avg_pool2d(grid, 3, stride=1, padding=1).permute(0, 2, 3, 1).reshape_as(value)
            mean = value.mean(2, keepdim=True).expand_as(value)
            maximum = value.amax(2, keepdim=True).expand_as(value)
            value = torch.cat((value, local, mean, maximum), -1)
        correction = self.adapter(value)
        return self.maximum * torch.tanh(correction / self.maximum)


class FrozenResNetElectronicCorrection(nn.Module):
    """Zero-start adapter from frozen ResNet18-L3 tokens into electronic E1.

    The adapted tensor is added only to the existing electronic residual route
    and must traverse O2 plus both language stages before the single readout.
    It is therefore not a direct MOS or pre-optical-output bypass.
    """

    def __init__(self, settings: ExperimentSettings) -> None:
        super().__init__()
        self.maximum = float(settings.resnet_electronic_max)
        hidden = 384
        self.adapter = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256, hidden),
            nn.GELU(),
            nn.Linear(hidden, settings.model_width),
        )
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        correction = self.adapter(tokens.float())
        return self.maximum * torch.tanh(correction / self.maximum)


class FrozenMobileNetElectronicCorrection(nn.Module):
    """Small E1 adapter for a truncated pretrained MobileNetV2 front.

    Width 64 corresponds to blocks 0..10 (0.289 M total); width 96 corresponds
    to blocks 0..11 (0.362 M total). It has no classifier, score head,
    attention, or path around the optical stages.
    """

    def __init__(self, settings: ExperimentSettings) -> None:
        super().__init__()
        self.maximum = float(settings.mobilenet_electronic_max)
        self.input_width = settings.mobilenet_feature_width
        self.adapter = nn.Sequential(
            nn.LayerNorm(self.input_width),
            nn.Linear(self.input_width, settings.model_width),
            nn.GELU(),
            nn.Linear(settings.model_width, settings.model_width),
        )
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        correction = self.adapter(tokens.float())
        return self.maximum * torch.tanh(correction / self.maximum)


class CustomConvE1Correction(nn.Module):
    """Project-owned RGB convolutional correction inside electronic E1.

    This is intentionally written from elementary Conv2d, GroupNorm, GELU and
    Linear layers.  It has no imported backbone, classifier, attention,
    pooling-to-score path, or route around the four optical/electronic stages.
    Four stride-2 convolutions map each 224x224 frame to the model's 14x14
    token grid; one local residual pair increases spatial context before a
    per-token projection produces the bounded E1 correction.
    """

    def __init__(self, settings: ExperimentSettings) -> None:
        super().__init__()
        self.grid = settings.token_grid
        self.maximum = float(settings.custom_conv_electronic_max)
        self.downsample = nn.Sequential(
            nn.Conv2d(3, 24, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(6, 24),
            nn.GELU(),
            nn.Conv2d(24, 48, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 48),
            nn.GELU(),
            nn.Conv2d(48, 64, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, 96, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(12, 96),
            nn.GELU(),
        )
        self.local_residual = nn.Sequential(
            nn.Conv2d(96, 96, 3, padding=1, bias=False),
            nn.GroupNorm(12, 96),
            nn.GELU(),
            nn.Conv2d(96, 96, 3, padding=1, bias=False),
            nn.GroupNorm(12, 96),
        )
        self.token_projection = nn.Sequential(
            nn.LayerNorm(96),
            nn.Linear(96, settings.model_width),
            nn.GELU(),
            nn.Linear(settings.model_width, settings.model_width),
        )
        nn.init.zeros_(self.token_projection[-1].weight)
        nn.init.zeros_(self.token_projection[-1].bias)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.ndim != 5 or tuple(frames.shape[1:3]) != (4, 3):
            raise ValueError("Custom Conv E1 correction requires [B,4,3,H,W]")
        batch, frame_count = frames.shape[:2]
        value = frames.float().div(127.5).sub(1.0).flatten(0, 1)
        value = self.downsample(value)
        value = F.gelu(value + self.local_residual(value))
        if tuple(value.shape[-2:]) != (self.grid, self.grid):
            raise ValueError("Custom Conv E1 correction expects 224x224 input frames")
        tokens = value.flatten(2).transpose(1, 2)
        correction = self.token_projection(tokens)
        correction = self.maximum * torch.tanh(correction / self.maximum)
        return correction.reshape(
            batch, frame_count, self.grid * self.grid, -1
        )


class TinyRgbE1Adapter(nn.Module):
    """Tiny, zero-start RGB front contained inside the E1 residual route.

    This module has no pretrained weights and no score output.  Three
    depthwise-separable downsampling steps map 224x224 RGB frames to the same
    14x14 token grid as the optical path.  The zero-initialized projection is
    injected only into E1 before the first optical/electronic fusion.
    """

    def __init__(self, settings: ExperimentSettings) -> None:
        super().__init__()
        self.grid = settings.token_grid
        self.maximum = float(settings.tiny_rgb_electronic_adapter_max)
        self.stem = nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(4, 16)
        self.depthwise2 = nn.Conv2d(
            16, 16, 3, stride=2, padding=1, groups=16, bias=False
        )
        self.pointwise2 = nn.Conv2d(16, 24, 1, bias=False)
        self.norm2 = nn.GroupNorm(6, 24)
        self.depthwise3 = nn.Conv2d(
            24, 24, 3, stride=2, padding=1, groups=24, bias=False
        )
        self.pointwise3 = nn.Conv2d(24, 32, 1, bias=False)
        self.norm3 = nn.GroupNorm(8, 32)
        self.depthwise4 = nn.Conv2d(
            32, 32, 3, stride=2, padding=1, groups=32, bias=False
        )
        self.pointwise4 = nn.Conv2d(32, 48, 1, bias=False)
        self.norm4 = nn.GroupNorm(8, 48)
        self.local5 = nn.Conv2d(48, 48, 5, padding=2, groups=48, bias=False)
        self.context3 = nn.Conv2d(
            48, 48, 3, padding=2, dilation=2, groups=48, bias=False
        )
        self.project = nn.Conv2d(96, settings.model_width, 1)
        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.project.bias)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.ndim != 5 or tuple(frames.shape[1:3]) != (4, 3):
            raise ValueError("Tiny RGB E1 adapter requires [B,4,3,H,W]")
        batch, frame_count = frames.shape[:2]
        value = frames.float().div(127.5).sub(1.0).flatten(0, 1)
        value = F.gelu(self.norm1(self.stem(value)))
        value = F.gelu(self.norm2(self.pointwise2(self.depthwise2(value))))
        value = F.gelu(self.norm3(self.pointwise3(self.depthwise3(value))))
        base = F.gelu(self.norm4(self.pointwise4(self.depthwise4(value))))
        if tuple(base.shape[-2:]) != (self.grid, self.grid):
            raise ValueError("Tiny RGB E1 adapter expects 224x224 input frames")
        correction = self.project(
            torch.cat((F.gelu(self.local5(base)), F.gelu(self.context3(base))), 1)
        )
        correction = self.maximum * torch.tanh(correction / self.maximum)
        return correction.flatten(2).transpose(1, 2).reshape(
            batch, frame_count, self.grid * self.grid, -1
        )


class SpatialLateInputCorrection(nn.Module):
    """Bounded final correction from the declared pre-optical multimodal field.

    The input already contains Qwen patch/position features, the convolutional
    quality feature, and prompt conditioning. This plain convolution/pooling
    head is applied only after the four optical/electronic stages produce the
    main score. Its zero-initialized last layer preserves a warm-start exactly,
    while the explicit bound prevents it from replacing the optical predictor.
    """

    def __init__(self, settings: ExperimentSettings) -> None:
        super().__init__()
        self.grid = settings.token_grid
        self.maximum = float(settings.late_input_correction_max)
        width, hidden, channels = settings.model_width, settings.head_width, 64
        self.norm = nn.LayerNorm(width)
        self.depthwise3 = nn.Conv2d(
            width, width, 3, padding=1, groups=width, bias=False
        )
        self.depthwise5 = nn.Conv2d(
            width, width, 5, padding=2, groups=width, bias=False
        )
        self.project = nn.Conv2d(width * 3, channels, 1)
        pooled_width = channels * 2 * (1 + 4 + 16)
        self.frame = nn.Sequential(
            nn.LayerNorm(pooled_width),
            nn.Linear(pooled_width, hidden),
            nn.GELU(),
            nn.Dropout(settings.dropout),
        )
        self.output = nn.Sequential(
            nn.LayerNorm(hidden * 6),
            nn.Linear(hidden * 6, hidden * 2),
            nn.GELU(),
            nn.Dropout(settings.dropout),
            nn.Linear(hidden * 2, 1),
        )
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        batch, frames, tokens, width = value.shape
        if tokens != self.grid * self.grid:
            raise ValueError("Late Spatial correction requires a square token grid")
        grid = self.norm(value).reshape(
            batch * frames, self.grid, self.grid, width
        ).permute(0, 3, 1, 2)
        feature = F.gelu(
            self.project(
                torch.cat(
                    (
                        grid,
                        F.gelu(self.depthwise3(grid)),
                        F.gelu(self.depthwise5(grid)),
                    ),
                    1,
                )
            )
        )
        pooled = torch.cat(
            tuple(
                pool(feature, size).flatten(1)
                for size in (1, 2, 4)
                for pool in (F.adaptive_avg_pool2d, F.adaptive_max_pool2d)
            ),
            1,
        )
        frame = self.frame(pooled).reshape(batch, frames, -1)
        difference = (frame[:, 1:] - frame[:, :-1]).abs()
        video = torch.cat(
            (
                frame.mean(1),
                frame.float().std(1, unbiased=False).to(frame.dtype),
                frame.amax(1),
                frame.amin(1),
                difference.mean(1),
                difference.amax(1),
            ),
            -1,
        )
        correction = self.output(video).squeeze(-1)
        return self.maximum * torch.tanh(correction / self.maximum)


class TemporalReadout(nn.Module):
    def __init__(self, settings: ExperimentSettings) -> None:
        super().__init__()
        width, hidden = settings.model_width, settings.head_width
        frame_width = width * 2
        self.frame_norm = nn.LayerNorm(frame_width)
        self.temporal3 = nn.Conv1d(
            frame_width, frame_width, 3, padding=1, groups=frame_width, bias=False
        )
        self.temporal5 = nn.Conv1d(
            frame_width, frame_width, 5, padding=2, groups=frame_width, bias=False
        )
        self.temporal_projection = nn.Conv1d(frame_width * 2, hidden, 1)
        self.language = nn.Sequential(
            nn.LayerNorm(width * 3),
            nn.Linear(width * 3, hidden),
            nn.GELU(),
        )
        self.output = nn.Sequential(
            nn.LayerNorm(hidden * 8),
            nn.Linear(hidden * 8, hidden * 2),
            nn.GELU(),
            nn.Dropout(settings.dropout),
            nn.Linear(hidden * 2, 1),
        )

    def forward(
        self, vision: torch.Tensor, language: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        frame = torch.cat((vision.mean(2), vision.amax(2)), -1)
        normalized = self.frame_norm(frame).transpose(1, 2)
        sequence = self.temporal_projection(
            torch.cat((F.gelu(self.temporal3(normalized)), F.gelu(self.temporal5(normalized))), 1)
        ).transpose(1, 2)
        difference1 = (sequence[:, 1:] - sequence[:, :-1]).abs()
        difference2 = (sequence[:, 2:] - sequence[:, :-2]).abs()
        summary = torch.cat(
            (
                sequence.mean(1),
                sequence.float().std(1, unbiased=False).to(sequence.dtype),
                sequence.amax(1),
                difference1.mean(1),
                difference1.amax(1),
                difference2.mean(1),
                difference2.amax(1),
            ),
            -1,
        )
        prompt = self.language(_masked_statistics(language, mask))
        return self.output(torch.cat((summary, prompt), -1)).squeeze(-1)


def _alignment(electronic: torch.Tensor, optical: torch.Tensor) -> torch.Tensor:
    left = F.normalize(electronic.float().flatten(1), dim=-1)
    right = F.normalize(optical.float().flatten(1), dim=-1)
    return (1.0 - (left * right).sum(-1)).mean()


class LGVQSingleMetricOEO16(nn.Module):
    """Qwen-front, text-conditioned, single-target, four-stage O/E/O network."""

    def __init__(self, settings: ExperimentSettings) -> None:
        super().__init__()
        self.settings = settings
        self.vision_adapter = nn.Sequential(
            nn.LayerNorm(settings.vision_input_width),
            nn.Linear(settings.vision_input_width, settings.model_width),
        )
        # The formal two-branch profile disables every auxiliary visual source:
        # Qwen patch+position is then the one shared input to the E and O paths.
        self.quality_adapter: nn.Module | None
        if not settings.quality_branch_enabled:
            self.quality_adapter = None
        elif settings.quality_adapter_mode == "spatial_conv":
            self.quality_adapter = QualitySpatialAdapter(settings)
        elif settings.quality_adapter_mode == "identity":
            self.quality_adapter = nn.Identity()
        else:
            self.quality_adapter = nn.Sequential(
                nn.LayerNorm(settings.quality_input_width),
                nn.Linear(settings.quality_input_width, settings.model_width),
            )
        self.quality_refiner = (
            QualitySpatialRefiner(settings)
            if settings.quality_refiner_enabled
            else nn.Identity()
        )
        self.frame_stem = (
            TrainableQualityFrameStem(settings.frame_stem_refiner_depth)
            if settings.trainable_frame_stem_enabled
            else None
        )
        self.vgg_correction = (
            FrozenVGGSpatialCorrection(settings)
            if settings.vgg_feature_cache_path is not None
            else None
        )
        self.resnet_electronic_correction = (
            FrozenResNetElectronicCorrection(settings)
            if settings.resnet_feature_cache_path is not None
            else None
        )
        self.mobilenet_electronic_correction = (
            FrozenMobileNetElectronicCorrection(settings)
            if settings.mobilenet_feature_cache_path is not None
            else None
        )
        self.custom_conv_electronic_correction = (
            CustomConvE1Correction(settings)
            if settings.custom_conv_electronic_enabled
            else None
        )
        self.tiny_rgb_electronic_adapter = (
            TinyRgbE1Adapter(settings)
            if settings.tiny_rgb_electronic_adapter_enabled
            else None
        )
        if settings.quality_branch_enabled:
            self.raw_quality_gate = nn.Parameter(
                torch.logit(torch.tensor(settings.quality_gate_initial))
            )
        if settings.qwen_gate_enabled:
            self.raw_qwen_gate = nn.Parameter(
                torch.logit(torch.tensor(settings.qwen_gate_initial))
            )
        self.visual_input_norm = nn.LayerNorm(settings.model_width)
        self.language_adapter = nn.Sequential(
            nn.LayerNorm(settings.language_input_width),
            nn.Linear(settings.language_input_width, settings.model_width),
        )
        # Attention-free prompt conditioning before the first optical pass.
        # This makes text part of the actual feature path, rather than merely
        # appending a constant prompt summary at the final readout.
        self.prompt_to_visual = nn.Sequential(
            nn.LayerNorm(settings.model_width),
            nn.Linear(settings.model_width, settings.model_width * 2),
        )
        if settings.electronic_route_variant in {
            "residual_conv",
            "residual_convnext",
        }:
            self.vision_routes = nn.ModuleList(
                [
                    VisionElectronicResidualRoute(settings),
                    VisionElectronicResidualRoute(settings),
                ]
            )
            self.language_routes = nn.ModuleList(
                [
                    LanguageElectronicResidualRoute(settings),
                    LanguageElectronicResidualRoute(settings),
                ]
            )
        else:
            self.vision_routes = nn.ModuleList(
                [VisionElectronicRoute(settings), VisionElectronicRoute(settings)]
            )
            self.language_routes = nn.ModuleList(
                [
                    LanguageElectronicRoute(
                        settings.model_width,
                        skip_enabled=settings.electronic_skip_enabled,
                        skip_initial=settings.electronic_skip_initial,
                        skip_max=settings.electronic_skip_max,
                    ),
                    LanguageElectronicRoute(
                        settings.model_width,
                        skip_enabled=settings.electronic_skip_enabled,
                        skip_initial=settings.electronic_skip_initial,
                        skip_max=settings.electronic_skip_max,
                    ),
                ]
            )
        self.electronic_quality_norm = (
            nn.LayerNorm(settings.quality_input_width)
            if settings.electronic_quality_residual_enabled
            else None
        )
        if settings.electronic_quality_residual_enabled:
            self.raw_electronic_quality_scale = nn.Parameter(
                torch.logit(
                    torch.tensor(settings.electronic_quality_residual_initial)
                )
            )
        if settings.electronic_quality_reinjection_enabled:
            # Zero is an exact warm start. Unlike a sigmoid gate, tanh permits
            # the optimizer to learn either a corrective addition or removal.
            self.raw_electronic_quality_reinjection = nn.Parameter(
                torch.zeros(())
            )
        if settings.electronic_cross_stage_skip_enabled:
            self.raw_vision_cross_stage_skip = nn.Parameter(torch.zeros(()))
            self.raw_language_cross_stage_skip = nn.Parameter(torch.zeros(()))
        self.parallel_optics = ParallelOpticalFeaturePath(settings)
        self.serial_optics = SerialOpticalFeaturePath(settings)
        self.parallel_router = OpticalRouterParallel16(settings)
        self.serial_router = OpticalRouterSerial(settings)
        self.fusions = nn.ModuleList([RmsConvexFusion(settings) for _ in range(4)])
        self.frame_merger = nn.Sequential(
            nn.LayerNorm(settings.model_width * 2),
            nn.Linear(settings.model_width * 2, settings.model_width),
            nn.GELU(),
        )
        self.frame_position = nn.Parameter(
            torch.zeros(1, settings.frame_count, settings.model_width)
        )
        self.sequence_position = nn.Parameter(
            torch.zeros(1, settings.maximum_language_tokens, settings.model_width)
        )
        nn.init.normal_(self.frame_position, std=0.02)
        nn.init.normal_(self.sequence_position, std=0.02)
        self.readout: nn.Module
        if settings.target_name == "spatial":
            if settings.spatial_readout_mode == "spatial_grid":
                self.readout = SpatialGridReadout(settings)
            elif settings.spatial_readout_mode == "spatial_multiscale":
                self.readout = SpatialMultiscaleReadout(settings)
            elif settings.spatial_readout_mode == "spatial_grid_residual":
                self.readout = SpatialGridResidualReadout(settings)
            elif settings.spatial_readout_mode == "spatial_pyramid_residual":
                self.readout = SpatialPyramidResidualReadout(settings)
            elif settings.spatial_readout_mode == "spatial_deep_residual":
                self.readout = SpatialDeepResidualReadout(settings)
            elif settings.spatial_readout_mode == "spatial_weighted_level_residual":
                self.readout = SpatialWeightedLevelResidualReadout(settings)
            elif settings.spatial_readout_mode == "spatial_crossframe_residual":
                self.readout = SpatialCrossFrameResidualReadout(settings)
            elif settings.spatial_readout_mode == "spatial_dual_level_residual":
                self.readout = SpatialDualLevelResidualReadout(settings)
            elif settings.spatial_readout_mode == "spatial_weighted_level_absolute":
                self.readout = SpatialWeightedLevelAbsoluteReadout(settings)
            elif settings.spatial_readout_mode == "spatial_weighted_level_blend":
                self.readout = SpatialWeightedLevelBlendReadout(settings)
            elif settings.spatial_readout_mode == "spatial_compact_weighted":
                self.readout = SpatialCompactWeightedReadout(settings)
            elif settings.spatial_readout_mode == "spatial_pruned_grid_compact_residual":
                self.readout = SpatialPrunedGridCompactResidualReadout(settings)
            else:
                self.readout = SpatialReadout(settings)
        elif settings.target_name == "temporal":
            self.readout = TemporalReadout(settings)
        else:
            raise ValueError("target_name must be spatial or temporal")
        self.late_input_correction = (
            SpatialLateInputCorrection(settings)
            if settings.late_input_correction_enabled
            else None
        )
        self.register_buffer("target_mean", torch.tensor(0.0))
        self.register_buffer("target_std", torch.tensor(1.0))

    def set_target_statistics(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.target_mean.copy_(mean.reshape(()))
        self.target_std.copy_(std.reshape(()).clamp_min(1.0e-6))

    def forward(
        self,
        vision_tokens: torch.Tensor,
        quality_tokens: torch.Tensor,
        language_tokens: torch.Tensor,
        language_mask: torch.Tensor,
        raw_frames: torch.Tensor | None = None,
        *,
        vgg_tokens: torch.Tensor | None = None,
        resnet_tokens: torch.Tensor | None = None,
        mobilenet_tokens: torch.Tensor | None = None,
        optical_enabled: bool = True,
    ) -> dict[str, Any]:
        if self.frame_stem is not None:
            if raw_frames is None:
                raise ValueError("The trainable frame stem requires raw_frames")
            # The original warm-start cache was persisted as float16. Keeping
            # the same quantization boundary makes epoch 0 reproducible while
            # retaining gradients through the cast during fine-tuning.
            quality_tokens = self.frame_stem(raw_frames).to(torch.float16).float()
        quality_is_used = (
            self.settings.quality_branch_enabled
            or self.settings.electronic_quality_residual_enabled
        )
        if quality_is_used and tuple(
            vision_tokens.shape[:-1]
        ) != tuple(quality_tokens.shape[:-1]):
            raise ValueError("Qwen and fixed-quality token grids must match")
        if vision_tokens.shape[-1] != self.settings.vision_input_width:
            raise ValueError("Qwen Vision front width must be 1024")
        if (
            quality_is_used
            and quality_tokens.shape[-1] != self.settings.quality_input_width
        ):
            raise ValueError("Quality token width differs from the configured width")
        if tuple(language_mask.shape) != tuple(language_tokens.shape[:-1]):
            raise ValueError("Language mask must match the prompt token sequence")

        prompt_tokens = self.language_adapter(language_tokens.float())
        prompt_valid = language_mask.bool().unsqueeze(-1)
        prompt_summary = (prompt_tokens * prompt_valid).sum(1) / prompt_valid.sum(1).clamp_min(1)
        prompt_scale, prompt_shift = self.prompt_to_visual(prompt_summary).chunk(2, -1)

        qwen_vision = self.vision_adapter(vision_tokens.float())
        quality_gate = qwen_vision.new_zeros(())
        quality = qwen_vision.new_zeros(qwen_vision.shape)
        if self.quality_adapter is not None:
            quality = self.quality_refiner(
                self.quality_adapter(quality_tokens.float())
            )
            quality_gate = torch.sigmoid(self.raw_quality_gate)
        raw_qwen_gate = getattr(self, "raw_qwen_gate", None)
        qwen_gate = (
            qwen_vision.new_ones(())
            if raw_qwen_gate is None
            else torch.sigmoid(raw_qwen_gate)
        )
        vision = qwen_gate * qwen_vision + quality_gate * quality
        vgg_correction = qwen_vision.new_zeros(qwen_vision.shape)
        if self.vgg_correction is not None:
            if vgg_tokens is None:
                raise ValueError("The plain-VGG correction requires vgg_tokens")
            if tuple(vgg_tokens.shape[:-1]) != tuple(qwen_vision.shape[:-1]) or vgg_tokens.shape[-1] != 512:
                raise ValueError("Plain-VGG token contract must be [B,4,196,512]")
            vgg_correction = self.vgg_correction(vgg_tokens)
            vision = vision + vgg_correction
        vision = self.visual_input_norm(
            vision * (1.0 + 0.10 * torch.tanh(prompt_scale[:, None, None]))
            + 0.10 * prompt_shift[:, None, None]
        )
        pre_optical_vision = vision
        routing: dict[str, dict[str, Any]] = {}
        alignments: list[torch.Tensor] = []

        fields1 = self.parallel_optics.fields(vision)
        electronic1 = self.vision_routes[0](vision)
        tiny_rgb_electronic = electronic1.new_zeros(electronic1.shape)
        if self.tiny_rgb_electronic_adapter is not None:
            if raw_frames is None:
                raise ValueError("The tiny RGB E1 adapter requires raw_frames")
            tiny_rgb_electronic = self.tiny_rgb_electronic_adapter(raw_frames)
            electronic1 = electronic1 + tiny_rgb_electronic
        resnet_electronic = electronic1.new_zeros(electronic1.shape)
        if self.resnet_electronic_correction is not None:
            if resnet_tokens is None:
                raise ValueError("The ResNet18 electronic residual requires resnet_tokens")
            if tuple(resnet_tokens.shape[:-1]) != tuple(electronic1.shape[:-1]) or resnet_tokens.shape[-1] != 256:
                raise ValueError("ResNet18 token contract must be [B,4,196,256]")
            resnet_electronic = self.resnet_electronic_correction(resnet_tokens)
            electronic1 = electronic1 + resnet_electronic
        mobilenet_electronic = electronic1.new_zeros(electronic1.shape)
        if self.mobilenet_electronic_correction is not None:
            if mobilenet_tokens is None:
                raise ValueError(
                    "The MobileNetV2 electronic residual requires mobilenet_tokens"
                )
            if (
                tuple(mobilenet_tokens.shape[:-1]) != tuple(electronic1.shape[:-1])
                or mobilenet_tokens.shape[-1]
                != self.settings.mobilenet_feature_width
            ):
                raise ValueError(
                    "MobileNetV2 token width differs from the configured boundary"
                )
            mobilenet_electronic = self.mobilenet_electronic_correction(
                mobilenet_tokens
            )
            electronic1 = electronic1 + mobilenet_electronic
        custom_conv_electronic = electronic1.new_zeros(electronic1.shape)
        if self.custom_conv_electronic_correction is not None:
            if raw_frames is None:
                raise ValueError("The custom Conv E1 correction requires raw_frames")
            custom_conv_electronic = self.custom_conv_electronic_correction(
                raw_frames
            )
            electronic1 = electronic1 + custom_conv_electronic
        electronic_quality_scale = electronic1.new_zeros(())
        if self.electronic_quality_norm is not None:
            electronic_quality_scale = torch.sigmoid(
                self.raw_electronic_quality_scale
            )
            electronic_quality = self.electronic_quality_norm(
                quality_tokens.float()
            )
            if (
                self.settings.quality_refiner_enabled
                and self.quality_adapter is None
            ):
                electronic_quality = self.quality_refiner(electronic_quality)
            electronic1 = (
                electronic1 + electronic_quality_scale * electronic_quality
            )
        if optical_enabled:
            routing["vision"] = self.parallel_router(fields1)
            optical1 = self.parallel_optics.expert(fields1, routing["vision"]["weights"])
            vision = self.fusions[0](electronic1, optical1)
            alignments.append(_alignment(electronic1, optical1))
        else:
            vision = electronic1

        fields2 = self.parallel_optics.fields(vision)
        electronic2 = self.vision_routes[1](vision)
        vision_cross_stage_skip = electronic2.new_zeros(())
        raw_vision_cross_stage_skip = getattr(
            self, "raw_vision_cross_stage_skip", None
        )
        if raw_vision_cross_stage_skip is not None:
            vision_cross_stage_skip = (
                self.settings.electronic_cross_stage_skip_max
                * torch.tanh(raw_vision_cross_stage_skip)
            )
            electronic2 = electronic2 + vision_cross_stage_skip * electronic1
        electronic_quality_reinjection = electronic2.new_zeros(())
        raw_reinjection = getattr(
            self, "raw_electronic_quality_reinjection", None
        )
        if raw_reinjection is not None:
            electronic_quality_reinjection = (
                self.settings.electronic_quality_reinjection_max
                * torch.tanh(raw_reinjection)
            )
            # This is a skip connection inside the sole electronic residual
            # route. The result must still traverse O2 and both language O/E
            # stages before the single MOS head, so it is not a third branch.
            electronic2 = (
                electronic2
                + electronic_quality_reinjection * electronic_quality
            )
        if optical_enabled:
            optical2 = self.parallel_optics.global_path(fields2)
            vision = self.fusions[1](electronic2, optical2)
            alignments.append(_alignment(electronic2, optical2))
        else:
            vision = electronic2

        image_tokens = self.frame_merger(
            torch.cat((vision.mean(2), vision.amax(2)), -1)
        ) + self.frame_position
        sequence = torch.cat((image_tokens, prompt_tokens), 1)
        mask = torch.cat(
            (
                torch.ones(
                    sequence.shape[0],
                    self.settings.frame_count,
                    dtype=torch.bool,
                    device=sequence.device,
                ),
                language_mask.bool(),
            ),
            1,
        )
        if sequence.shape[1] > self.settings.maximum_language_tokens:
            raise ValueError("Image plus prompt tokens exceed the formal sequence limit")
        sequence = (
            sequence + self.sequence_position[:, : sequence.shape[1]]
        ).masked_fill(~mask.unsqueeze(-1), 0.0)

        fields3 = self.serial_optics.fields(sequence)
        electronic3 = self.language_routes[0](sequence, mask)
        if optical_enabled:
            router_fields3 = fields3
            if self.settings.serial_router_visual_token_gain != 1.0:
                # The language-stage optical router sees the same physical
                # sequence field, but the four image summary rows are exposed
                # more strongly than the fixed prompt rows. This is a fixed
                # amplitude encoding rule, not an electronic router or a new
                # inference branch; the feature-producing expert path below
                # still receives the unmodified sequence.
                router_sequence = sequence.clone()
                router_sequence[:, : self.settings.frame_count] *= (
                    self.settings.serial_router_visual_token_gain
                )
                router_fields3 = self.serial_optics.fields(router_sequence)
            routing["language"] = self.serial_router(
                router_fields3, sequence.shape[1]
            )
            optical3 = self.serial_optics.expert(
                fields3, routing["language"]["weights"], sequence.shape[1]
            )
            sequence = self.fusions[2](electronic3, optical3, mask)
            alignments.append(_alignment(electronic3, optical3))
        else:
            sequence = electronic3

        fields4 = self.serial_optics.fields(sequence)
        electronic4 = self.language_routes[1](sequence, mask)
        language_cross_stage_skip = electronic4.new_zeros(())
        raw_language_cross_stage_skip = getattr(
            self, "raw_language_cross_stage_skip", None
        )
        if raw_language_cross_stage_skip is not None:
            language_cross_stage_skip = (
                self.settings.electronic_cross_stage_skip_max
                * torch.tanh(raw_language_cross_stage_skip)
            )
            electronic4 = electronic4 + language_cross_stage_skip * electronic3
        if optical_enabled:
            optical4 = self.serial_optics.global_path(fields4, sequence.shape[1])
            sequence = self.fusions[3](electronic4, optical4, mask)
            alignments.append(_alignment(electronic4, optical4))
        else:
            sequence = electronic4

        normalized = self.readout(vision, sequence, mask)
        quality_level_logits = getattr(self.readout, "last_level_logits", None)
        quality_level_scores = getattr(self.readout, "level_scores", None)
        quality_level_base_prediction = getattr(
            self.readout, "last_level_base_prediction", None
        )
        readout_image_focus = normalized.new_zeros(())
        if hasattr(self.readout, "raw_image_focus"):
            readout_image_focus = (
                self.settings.spatial_readout_image_focus_max
                * torch.tanh(self.readout.raw_image_focus)
            )
        input_correction = normalized.new_zeros(normalized.shape)
        if self.late_input_correction is not None:
            input_correction = self.late_input_correction(pre_optical_vision)
            normalized = normalized + input_correction
        prediction = normalized * self.target_std + self.target_mean
        if routing:
            balance = torch.stack(
                [value["balance_loss"] for value in routing.values()]
            ).mean()
            importance = torch.stack(
                [value["importance_loss"] for value in routing.values()]
            ).mean()
            capture = torch.stack(
                [
                    (1.0 - value["capture_fraction"].clamp(0.0, 1.0)).mean()
                    for value in routing.values()
                ]
            ).mean()
        else:
            zero = normalized.new_zeros(())
            balance = importance = capture = zero
        return {
            "prediction": prediction,
            "normalized_prediction": normalized,
            "quality_level_logits": quality_level_logits,
            "quality_level_scores": quality_level_scores,
            "quality_level_base_prediction": quality_level_base_prediction,
            "target_name": self.settings.target_name,
            "quality_gate": quality_gate,
            "electronic_quality_residual_scale": electronic_quality_scale,
            "electronic_quality_reinjection_scale": electronic_quality_reinjection,
            "vision_cross_stage_skip_scale": vision_cross_stage_skip,
            "language_cross_stage_skip_scale": language_cross_stage_skip,
            "qwen_gate": qwen_gate,
            "late_input_correction": input_correction,
            "spatial_readout_image_focus": readout_image_focus,
            "vgg_correction_rms": vgg_correction.float().square().mean().sqrt(),
            "resnet_electronic_rms": resnet_electronic.float().square().mean().sqrt(),
            "mobilenet_electronic_rms": mobilenet_electronic.float().square().mean().sqrt(),
            "custom_conv_electronic_rms": custom_conv_electronic.float().square().mean().sqrt(),
            "tiny_rgb_electronic_rms": tiny_rgb_electronic.float().square().mean().sqrt(),
            "routing": routing,
            "optical_enabled": optical_enabled,
            "optical_alignment_loss": torch.stack(alignments).mean()
            if alignments
            else normalized.new_zeros(()),
            "router_balance_loss": balance,
            "router_importance_loss": importance,
            "router_capture_loss": capture,
        }

    def fusion_diagnostics(self) -> dict[str, dict[str, float]]:
        names = ("vision_expert", "vision_global", "language_expert", "language_global")
        return {
            name: dict(layer.last_diagnostics)
            for name, layer in zip(names, self.fusions)
        }

    def parameter_breakdown(self) -> dict[str, int]:
        groups = {
            "qwen_boundary_adapters": nn.ModuleList(
                [
                    self.vision_adapter,
                    self.visual_input_norm,
                    self.language_adapter,
                    self.prompt_to_visual,
                ]
            ),
            "electronic_routes": nn.ModuleList(
                [*self.vision_routes, *self.language_routes]
            ),
            "optical_feature_paths": nn.ModuleList(
                [self.parallel_optics, self.serial_optics]
            ),
            "optical_routers": nn.ModuleList(
                [self.parallel_router, self.serial_router]
            ),
            "fusion": self.fusions,
            "multimodal_bridge": self.frame_merger,
            "position_parameters": nn.ParameterList(
                [self.frame_position, self.sequence_position]
            ),
            "single_metric_readout": self.readout,
        }
        if self.quality_adapter is not None:
            groups["quality_input_adapter"] = self.quality_adapter
        if self.electronic_quality_norm is not None:
            components = [self.electronic_quality_norm]
            if (
                self.settings.quality_refiner_enabled
                and self.quality_adapter is None
            ):
                components.append(self.quality_refiner)
            groups["electronic_quality_residual"] = nn.ModuleList(components)
        if (
            self.settings.quality_refiner_enabled
            and self.quality_adapter is not None
        ):
            groups["quality_input_refiner"] = self.quality_refiner
        if self.frame_stem is not None:
            groups["trainable_quality_frame_stem"] = self.frame_stem
        if self.vgg_correction is not None:
            groups["plain_vgg16_spatial_correction"] = self.vgg_correction
        if self.resnet_electronic_correction is not None:
            groups["resnet18_layer3_electronic_residual"] = (
                self.resnet_electronic_correction
            )
        if self.mobilenet_electronic_correction is not None:
            block_name = {
                64: "mobilenetv2_b10_electronic_residual",
                96: "mobilenetv2_b11_electronic_residual",
            }.get(
                self.settings.mobilenet_feature_width,
                "mobilenetv2_electronic_residual",
            )
            groups[block_name] = (
                self.mobilenet_electronic_correction
            )
        if self.custom_conv_electronic_correction is not None:
            groups["custom_conv_e1_correction"] = (
                self.custom_conv_electronic_correction
            )
        if self.tiny_rgb_electronic_adapter is not None:
            groups["tiny_rgb_e1_adapter"] = self.tiny_rgb_electronic_adapter
        if self.late_input_correction is not None:
            groups["late_input_correction"] = self.late_input_correction
        result = {
            name: sum(parameter.numel() for parameter in module.parameters())
            for name, module in groups.items()
        }
        if hasattr(self, "raw_electronic_quality_scale"):
            result["electronic_quality_residual"] += (
                self.raw_electronic_quality_scale.numel()
            )
        if hasattr(self, "raw_vision_cross_stage_skip"):
            result["electronic_cross_stage_skips"] = (
                self.raw_vision_cross_stage_skip.numel()
                + self.raw_language_cross_stage_skip.numel()
            )
        result["total_trainable"] = sum(
            parameter.numel() for parameter in self.parameters() if parameter.requires_grad
        )
        result["total_frozen_in_student"] = sum(
            parameter.numel() for parameter in self.parameters() if not parameter.requires_grad
        )
        return result


def build_model(settings: ExperimentSettings) -> LGVQSingleMetricOEO16:
    return LGVQSingleMetricOEO16(settings)


__all__ = ["CustomConvE1Correction", "LGVQSingleMetricOEO16", "build_model"]
