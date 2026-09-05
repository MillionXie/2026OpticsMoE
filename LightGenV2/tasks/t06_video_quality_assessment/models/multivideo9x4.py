"""True full-field 9-video x 4-frame optical Temporal-VQA model.

All nine videos share each of six 518x518 coherent propagations.  Optical
results are cropped and normalized per video before a shared electronic
readout emits one scalar MOS for each video.  There is no attention or
Transformer module in the trainable student.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from experiments.qwen3_vl_2b_lgvq_single_metric_o2_16frame_54.modeling import (
    AngularSpectrum,
    CcdReadout,
    LanguageElectronicRoute,
    RmsConvexFusion,
    TemporalReadout,
    VisionElectronicRoute,
    _alignment,
    _initialize_resampler,
    _phase_modulation,
    _random_shift,
    _sparse_top2,
    _spot_phase,
    _translate,
)

from ..multivideo_settings import MultiVideoSettings


def _slotwise_routing_statistics(
    probabilities: torch.Tensor, selected: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Balance every physical lane across samples, not only the whole field.

    A global average can be perfectly uniform when slot 0 always chooses
    experts 0/1 and slot 1 always chooses 2/3.  Averaging over the batch while
    preserving the physical decision index prevents that false balance.
    """

    flat_p = probabilities.reshape(probabilities.shape[0], -1, 4)
    flat_s = selected.float().reshape(selected.shape[0], -1, 4)
    importance = flat_p.mean(0)
    load = flat_s.mean(0) / 2.0
    entropy = -(
        flat_p.clamp_min(1.0e-8).log() * flat_p
    ).sum(-1).mean() / math.log(4.0)
    codes = (flat_s * flat_s.new_tensor((1.0, 2.0, 4.0, 8.0))).sum(-1).long()
    counts = torch.stack(
        [(codes == code).sum(0) for code in range(16)], 0
    )
    modal_fraction = counts.amax(0).float() / max(1, codes.shape[0])
    unique_patterns = (counts > 0).sum(0).float()
    return {
        "importance": importance.mean(0),
        "load": load.mean(0),
        "balance_loss": (4.0 * (importance * load).sum(-1)).mean(),
        "importance_loss": (4.0 * importance.square().sum(-1) - 1.0).mean(),
        "normalized_entropy": entropy,
        "conditional_probability_std": flat_p.std(0, unbiased=False).mean(),
        "selection_variation_fraction": (1.0 - modal_fraction).mean(),
        "unique_selection_patterns_mean": unique_patterns.mean(),
    }


def _standardized_router(
    energy: torch.Tensor, settings: MultiVideoSettings, *, training: bool
) -> dict[str, Any]:
    centered = energy - energy.mean(-1, keepdim=True)
    logits = centered / centered.square().mean(-1, keepdim=True).add(1.0e-8).sqrt()
    if settings.router_noise_std > 0 and training:
        logits = logits + settings.router_noise_std * torch.randn_like(logits)
    probabilities = torch.softmax(logits / settings.router_temperature, dim=-1)
    weights, selected, indices = _sparse_top2(probabilities)
    return {
        "probabilities": probabilities,
        "weights": weights,
        "selected_mask": selected,
        "selected_indices": indices,
        **_slotwise_routing_statistics(probabilities, selected),
    }


class _FullFieldBase(nn.Module):
    def __init__(self, settings: MultiVideoSettings) -> None:
        super().__init__()
        self.settings = settings
        self.geometry = settings.geometry
        self.propagation = AngularSpectrum(settings)
        support = torch.zeros(
            self.geometry.canvas_size, self.geometry.canvas_size, dtype=torch.bool
        )
        margin = self.geometry.active_margin
        for top, left in self.geometry.video_origins:
            support[
                margin + top : margin + top + self.geometry.video_tile_size,
                margin + left : margin + left + self.geometry.video_tile_size,
            ] = True
        self.register_buffer("video_support", support, persistent=False)

    def _guard_energy(self, detector: torch.Tensor) -> torch.Tensor:
        total = detector.sum((-2, -1)).clamp_min(1.0e-8)
        guard = detector.masked_fill(self.video_support, 0.0).sum((-2, -1))
        return (guard / total).mean()

    def _detector(self, field: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        detector = self.propagation(field).abs().square().float()
        detector = _translate(
            detector,
            *_random_shift(self.settings.ccd_shift_pixels, self.training),
            fill=0.0,
        )
        return detector, self._guard_energy(detector)

    def _normalize(self, patch: torch.Tensor) -> torch.Tensor:
        value = patch.float().clamp_min(0.0)
        mean = value.mean((-2, -1), keepdim=True).clamp_min(1.0e-6)
        return torch.log1p(
            self.settings.ccd_log_compression
            * (value / mean).clamp_max(self.settings.ccd_relative_clip)
        )


class FrameOpticalRouter(_FullFieldBase):
    def __init__(self, settings: MultiVideoSettings) -> None:
        super().__init__(settings)
        size = self.geometry.frame_expert_size
        initial = _spot_phase(
            size,
            self.geometry.frame_lane_size,
            settings.frame_router_intervals,
            settings,
        )
        self.raw_router_phase = nn.Parameter(
            initial.unsqueeze(0).repeat(
                self.geometry.video_count * self.geometry.frames_per_video, 1, 1
            )
        )

    def forward(self, fields: torch.Tensor) -> dict[str, Any]:
        expected = (
            self.geometry.video_count,
            self.geometry.frames_per_video,
            self.geometry.frame_expert_size,
            self.geometry.frame_expert_size,
        )
        if tuple(fields.shape[1:]) != expected:
            raise ValueError(f"Frame router expects [B,{expected}], got {tuple(fields.shape)}")
        size = self.geometry.frame_expert_size
        local = (self.geometry.frame_lane_size - size) // 2
        margin = self.geometry.active_margin
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
                self.raw_router_phase, settings=self.settings, training=self.training
            ),
            *_random_shift(self.settings.phase_shift_pixels, self.training),
            fill=1.0 + 0.0j,
        )
        index = 0
        for video, (video_top, video_left) in enumerate(self.geometry.video_origins):
            for frame, (frame_top, frame_left) in enumerate(
                self.geometry.frame_origins_local
            ):
                top = margin + video_top + frame_top + local
                left = margin + video_left + frame_left + local
                canvas[:, top : top + size, left : left + size] = shifted_fields[
                    :, video, frame
                ]
                phase_canvas[:, top : top + size, left : left + size] = shifted_phase[
                    index
                ]
                index += 1
        detector, guard = self._detector(canvas.to(torch.complex64) * phase_canvas)
        rows, lane_totals = [], []
        intervals = self.settings.frame_router_intervals
        for video_top, video_left in self.geometry.video_origins:
            video_rows, video_totals = [], []
            for frame_top, frame_left in self.geometry.frame_origins_local:
                top = margin + video_top + frame_top
                left = margin + video_left + frame_left
                lane = detector[
                    :,
                    top : top + self.geometry.frame_lane_size,
                    left : left + self.geometry.frame_lane_size,
                ]
                video_totals.append(lane.sum((-2, -1)))
                video_rows.append(
                    torch.stack(
                        [
                            lane[:, y0:y1, x0:x1].sum((-2, -1))
                            for y0, y1 in intervals
                            for x0, x1 in intervals
                        ],
                        -1,
                    )
                )
            rows.append(torch.stack(video_rows, 1))
            lane_totals.append(torch.stack(video_totals, 1))
        energy = torch.stack(rows, 1)
        result = _standardized_router(energy, self.settings, training=self.training)
        result.update(
            {
                "capture_fraction": energy.sum(-1)
                / torch.stack(lane_totals, 1).clamp_min(1.0e-8),
                "guard_energy_fraction": guard,
                "router_implementation": "optical_fullfield_9video_4frame_energy_top2",
            }
        )
        return result


class FrameOpticalPath(_FullFieldBase):
    def __init__(self, settings: MultiVideoSettings) -> None:
        super().__init__(settings)
        size = self.geometry.frame_expert_size
        self.width_to_field = nn.Linear(settings.model_width, size)
        self.tokens_to_field = nn.Linear(settings.token_count, size)
        _initialize_resampler(self.tokens_to_field)
        self.raw_expert_phase = nn.Parameter(
            torch.empty(
                self.geometry.video_count * self.geometry.frames_per_video * 4,
                size,
                size,
            )
        )
        self.raw_global_phase = nn.Parameter(
            torch.empty(
                self.geometry.video_count,
                self.geometry.video_phase_tile_size,
                self.geometry.video_phase_tile_size,
            )
        )
        nn.init.normal_(self.raw_expert_phase, 0.0, settings.phase_init_std)
        nn.init.normal_(self.raw_global_phase, 0.0, settings.phase_init_std)
        self.expert_readout = CcdReadout(
            settings.token_count, settings.detector_projection_size, settings.model_width
        )
        self.global_readout = CcdReadout(
            settings.token_count, settings.detector_projection_size, settings.model_width
        )
        self.last_ccd: dict[str, torch.Tensor] = {}

    def fields(self, tokens: torch.Tensor) -> torch.Tensor:
        expected = (
            self.geometry.video_count,
            self.geometry.frames_per_video,
            self.settings.token_count,
            self.settings.model_width,
        )
        if tuple(tokens.shape[1:]) != expected:
            raise ValueError(f"Frame tokens must be [B,{expected}]")
        encoded = F.softplus(self.width_to_field(tokens.float()))
        field = F.softplus(
            self.tokens_to_field(encoded.transpose(-2, -1))
        ).transpose(-2, -1)
        return field / field.square().mean((-2, -1), keepdim=True).sqrt().clamp_min(1.0e-6)

    def _read_frames(self, detector: torch.Tensor, stage: str) -> torch.Tensor:
        margin = self.geometry.active_margin
        rows = []
        for video_top, video_left in self.geometry.video_origins:
            frames = []
            for frame_top, frame_left in self.geometry.frame_origins_local:
                top = margin + video_top + frame_top
                left = margin + video_left + frame_left
                frames.append(
                    self._normalize(
                        detector[
                            :,
                            top : top + self.geometry.frame_lane_size,
                            left : left + self.geometry.frame_lane_size,
                        ]
                    )
                )
            rows.append(torch.stack(frames, 1))
        stacked = torch.stack(rows, 1)
        self.last_ccd[stage] = stacked.detach()
        readout = self.expert_readout if stage == "frame_expert" else self.global_readout
        output = readout(stacked.flatten(0, 2))
        return output.reshape(
            detector.shape[0],
            self.geometry.video_count,
            self.geometry.frames_per_video,
            self.settings.token_count,
            self.settings.model_width,
        )

    def expert(
        self, fields: torch.Tensor, weights: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        size = self.geometry.frame_expert_size
        margin = self.geometry.active_margin
        shifted = _translate(
            fields,
            *_random_shift(self.settings.input_shift_pixels, self.training),
            fill=0.0,
        )
        phase = _translate(
            _phase_modulation(
                self.raw_expert_phase, settings=self.settings, training=self.training
            ),
            *_random_shift(self.settings.phase_shift_pixels, self.training),
            fill=1.0 + 0.0j,
        )
        propagated = torch.zeros(
            fields.shape[0],
            self.geometry.canvas_size,
            self.geometry.canvas_size,
            device=fields.device,
            dtype=torch.complex64,
        )
        index = 0
        for video, (video_top, video_left) in enumerate(self.geometry.video_origins):
            for frame, (frame_top, frame_left) in enumerate(self.geometry.frame_origins_local):
                for expert, (local_top, local_left) in enumerate(self.geometry.frame_expert_origins_local):
                    top = margin + video_top + frame_top + local_top
                    left = margin + video_left + frame_left + local_left
                    propagated[:, top : top + size, left : left + size] = (
                        shifted[:, video, frame]
                        * weights[:, video, frame, expert, None, None]
                    ).to(torch.complex64) * phase[index]
                    index += 1
        detector, guard = self._detector(propagated)
        return self._read_frames(detector, "frame_expert"), guard

    def global_path(self, fields: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        size = self.geometry.frame_expert_size
        lane_local = (self.geometry.frame_lane_size - size) // 2
        margin = self.geometry.active_margin
        canvas = fields.new_zeros(
            fields.shape[0], self.geometry.canvas_size, self.geometry.canvas_size
        )
        shifted = _translate(
            fields,
            *_random_shift(self.settings.input_shift_pixels, self.training),
            fill=0.0,
        )
        for video, (video_top, video_left) in enumerate(self.geometry.video_origins):
            for frame, (frame_top, frame_left) in enumerate(self.geometry.frame_origins_local):
                top = margin + video_top + frame_top + lane_local
                left = margin + video_left + frame_left + lane_local
                canvas[:, top : top + size, left : left + size] = shifted[:, video, frame]
        phase = _translate(
            _phase_modulation(
                self.raw_global_phase, settings=self.settings, training=self.training
            ),
            *_random_shift(self.settings.phase_shift_pixels, self.training),
            fill=1.0 + 0.0j,
        )
        propagated = canvas.to(torch.complex64)
        offset = self.geometry.video_phase_offset
        for video, (top, left) in enumerate(self.geometry.video_origins):
            y, x = margin + top + offset, margin + left + offset
            tile = self.geometry.video_phase_tile_size
            propagated[:, y : y + tile, x : x + tile] *= phase[video]
        detector, guard = self._detector(propagated)
        return self._read_frames(detector, "frame_global"), guard


class VideoOpticalRouter(_FullFieldBase):
    def __init__(self, settings: MultiVideoSettings) -> None:
        super().__init__(settings)
        initial = _spot_phase(
            self.geometry.video_field_size,
            self.geometry.video_tile_size,
            settings.video_router_intervals,
            settings,
        )
        self.raw_router_phase = nn.Parameter(
            initial.unsqueeze(0).repeat(self.geometry.video_count, 1, 1)
        )

    def forward(self, fields: torch.Tensor) -> dict[str, Any]:
        size = self.geometry.video_field_size
        margin = self.geometry.active_margin
        offset = self.geometry.video_field_offset
        canvas = fields.new_zeros(
            fields.shape[0], self.geometry.canvas_size, self.geometry.canvas_size
        )
        phase_canvas = torch.ones_like(canvas, dtype=torch.complex64)
        shifted = _translate(
            fields,
            *_random_shift(self.settings.input_shift_pixels, self.training),
            fill=0.0,
        )
        phases = _translate(
            _phase_modulation(
                self.raw_router_phase, settings=self.settings, training=self.training
            ),
            *_random_shift(self.settings.phase_shift_pixels, self.training),
            fill=1.0 + 0.0j,
        )
        for video, (top, left) in enumerate(self.geometry.video_origins):
            y, x = margin + top + offset, margin + left + offset
            canvas[:, y : y + size, x : x + size] = shifted[:, video]
            phase_canvas[:, y : y + size, x : x + size] = phases[video]
        detector, guard = self._detector(canvas.to(torch.complex64) * phase_canvas)
        rows, totals = [], []
        intervals = self.settings.video_router_intervals
        for top, left in self.geometry.video_origins:
            y, x = margin + top, margin + left
            tile = detector[
                :, y : y + self.geometry.video_tile_size, x : x + self.geometry.video_tile_size
            ]
            totals.append(tile.sum((-2, -1)))
            rows.append(
                torch.stack(
                    [
                        tile[:, y0:y1, x0:x1].sum((-2, -1))
                        for y0, y1 in intervals
                        for x0, x1 in intervals
                    ],
                    -1,
                )
            )
        energy = torch.stack(rows, 1)
        result = _standardized_router(energy, self.settings, training=self.training)
        result.update(
            {
                "capture_fraction": energy.sum(-1)
                / torch.stack(totals, 1).clamp_min(1.0e-8),
                "guard_energy_fraction": guard,
                "router_implementation": "optical_fullfield_9video_energy_top2",
            }
        )
        return result


class VideoOpticalPath(_FullFieldBase):
    def __init__(self, settings: MultiVideoSettings) -> None:
        super().__init__(settings)
        size = self.geometry.video_field_size
        self.width_to_field = nn.Linear(settings.model_width, size)
        self.raw_expert_phase = nn.Parameter(
            torch.empty(self.geometry.video_count * 4, size, size)
        )
        self.raw_global_phase = nn.Parameter(
            torch.empty(
                self.geometry.video_count,
                self.geometry.video_phase_tile_size,
                self.geometry.video_phase_tile_size,
            )
        )
        nn.init.normal_(self.raw_expert_phase, 0.0, settings.phase_init_std)
        nn.init.normal_(self.raw_global_phase, 0.0, settings.phase_init_std)
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
        if tokens.ndim != 4 or tokens.shape[1] != self.geometry.video_count:
            raise ValueError("Video sequence tokens must be [B,9,S,192]")
        if tokens.shape[2] > self.geometry.video_field_size:
            raise ValueError("Video sequence exceeds the 72-row optical field")
        encoded = F.softplus(self.width_to_field(tokens.float()))
        field = encoded.new_zeros(
            tokens.shape[0],
            self.geometry.video_count,
            self.geometry.video_field_size,
            self.geometry.video_field_size,
        )
        field[:, :, : tokens.shape[2]] = encoded
        return field / field.square().mean((-2, -1), keepdim=True).sqrt().clamp_min(1.0e-6)

    def _read_videos(
        self, detector: torch.Tensor, stage: str, token_count: int
    ) -> torch.Tensor:
        margin = self.geometry.active_margin
        patches = []
        for top, left in self.geometry.video_origins:
            y, x = margin + top, margin + left
            patches.append(
                self._normalize(
                    detector[
                        :,
                        y : y + self.geometry.video_tile_size,
                        x : x + self.geometry.video_tile_size,
                    ]
                )
            )
        stacked = torch.stack(patches, 1)
        self.last_ccd[stage] = stacked.detach()
        readout = self.expert_readout if stage == "video_expert" else self.global_readout
        pooled = F.adaptive_avg_pool2d(
            stacked.flatten(0, 1).unsqueeze(1),
            (token_count, self.settings.detector_projection_size),
        ).squeeze(1)
        output = readout.output(F.softplus(readout.norm(pooled)))
        return output.reshape(
            detector.shape[0], self.geometry.video_count, token_count, self.settings.model_width
        )

    def expert(
        self, fields: torch.Tensor, weights: torch.Tensor, token_count: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        size = self.geometry.video_field_size
        margin = self.geometry.active_margin
        shifted = _translate(
            fields,
            *_random_shift(self.settings.input_shift_pixels, self.training),
            fill=0.0,
        )
        phases = _translate(
            _phase_modulation(
                self.raw_expert_phase, settings=self.settings, training=self.training
            ),
            *_random_shift(self.settings.phase_shift_pixels, self.training),
            fill=1.0 + 0.0j,
        )
        propagated = torch.zeros(
            fields.shape[0],
            self.geometry.canvas_size,
            self.geometry.canvas_size,
            device=fields.device,
            dtype=torch.complex64,
        )
        index = 0
        for video, (video_top, video_left) in enumerate(self.geometry.video_origins):
            for expert, (local_top, local_left) in enumerate(
                self.geometry.video_expert_origins_local
            ):
                top = margin + video_top + local_top
                left = margin + video_left + local_left
                propagated[:, top : top + size, left : left + size] = (
                    shifted[:, video] * weights[:, video, expert, None, None]
                ).to(torch.complex64) * phases[index]
                index += 1
        detector, guard = self._detector(propagated)
        return self._read_videos(detector, "video_expert", token_count), guard

    def global_path(
        self, fields: torch.Tensor, token_count: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        size = self.geometry.video_field_size
        margin = self.geometry.active_margin
        offset = self.geometry.video_field_offset
        shifted = _translate(
            fields,
            *_random_shift(self.settings.input_shift_pixels, self.training),
            fill=0.0,
        )
        propagated = torch.zeros(
            fields.shape[0],
            self.geometry.canvas_size,
            self.geometry.canvas_size,
            device=fields.device,
            dtype=torch.complex64,
        )
        for video, (top, left) in enumerate(self.geometry.video_origins):
            y, x = margin + top + offset, margin + left + offset
            propagated[:, y : y + size, x : x + size] = shifted[:, video].to(
                torch.complex64
            )
        phases = _translate(
            _phase_modulation(
                self.raw_global_phase, settings=self.settings, training=self.training
            ),
            *_random_shift(self.settings.phase_shift_pixels, self.training),
            fill=1.0 + 0.0j,
        )
        phase_offset = self.geometry.video_phase_offset
        tile_size = self.geometry.video_phase_tile_size
        for video, (top, left) in enumerate(self.geometry.video_origins):
            y, x = margin + top + phase_offset, margin + left + phase_offset
            propagated[:, y : y + tile_size, x : x + tile_size] *= phases[video]
        detector, guard = self._detector(propagated)
        return self._read_videos(detector, "video_global", token_count), guard


class MultiVideo9x4OpticalVQA(nn.Module):
    """Six-pass O/E/O student producing one Temporal MOS per video slot."""

    def __init__(self, settings: MultiVideoSettings) -> None:
        super().__init__()
        self.settings = settings
        self.vision_adapter = nn.Sequential(
            nn.LayerNorm(settings.vision_input_width),
            nn.Linear(settings.vision_input_width, settings.model_width),
        )
        self.quality_adapter = nn.Sequential(
            nn.LayerNorm(settings.quality_input_width),
            nn.Linear(settings.quality_input_width, settings.model_width),
        )
        self.raw_quality_gate = nn.Parameter(
            torch.logit(torch.tensor(settings.quality_gate_initial))
        )
        self.visual_input_norm = nn.LayerNorm(settings.model_width)
        self.language_adapter = nn.Sequential(
            nn.LayerNorm(settings.language_input_width),
            nn.Linear(settings.language_input_width, settings.model_width),
        )
        self.prompt_to_visual = nn.Sequential(
            nn.LayerNorm(settings.model_width),
            nn.Linear(settings.model_width, settings.model_width * 2),
        )
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
                )
                for _ in range(2)
            ]
        )
        # Names deliberately preserve shape-compatible warm-start keys.
        self.parallel_optics = FrameOpticalPath(settings)
        self.parallel_router = FrameOpticalRouter(settings)
        self.serial_optics = VideoOpticalPath(settings)
        self.serial_router = VideoOpticalRouter(settings)
        self.fusions = nn.ModuleList([RmsConvexFusion(settings) for _ in range(4)])
        self.frame_merger = nn.Sequential(
            nn.LayerNorm(settings.model_width * 2),
            nn.Linear(settings.model_width * 2, settings.model_width),
            nn.GELU(),
        )
        self.frame_position = nn.Parameter(
            torch.zeros(1, 1, settings.frame_count, settings.model_width)
        )
        self.sequence_position = nn.Parameter(
            torch.zeros(1, settings.maximum_language_tokens, settings.model_width)
        )
        nn.init.normal_(self.frame_position, std=0.02)
        nn.init.normal_(self.sequence_position, std=0.02)
        self.readout = TemporalReadout(settings)
        self.register_buffer("target_mean", torch.tensor(0.0))
        self.register_buffer("target_std", torch.tensor(1.0))

    def set_target_statistics(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.target_mean.copy_(mean.reshape(()))
        self.target_std.copy_(std.reshape(()).clamp_min(1.0e-6))

    def _vision_route(self, layer: nn.Module, value: torch.Tensor) -> torch.Tensor:
        shape = value.shape
        output = layer(value.flatten(0, 1))
        return output.reshape(shape)

    def _language_route(
        self, layer: nn.Module, value: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        shape = value.shape
        output = layer(value.flatten(0, 1), mask.flatten(0, 1))
        return output.reshape(shape)

    def _fuse(
        self,
        layer: RmsConvexFusion,
        electronic: torch.Tensor,
        optical: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        shape = electronic.shape
        flat_mask = None if mask is None else mask.flatten(0, 1)
        output = layer(
            electronic.flatten(0, 1), optical.flatten(0, 1), flat_mask
        )
        return output.reshape(shape)

    def forward(
        self,
        vision_tokens: torch.Tensor,
        quality_tokens: torch.Tensor,
        language_tokens: torch.Tensor,
        language_mask: torch.Tensor,
        *,
        optical_enabled: bool = True,
    ) -> dict[str, Any]:
        expected = (
            self.settings.videos_per_field,
            self.settings.frame_count,
            self.settings.token_count,
        )
        if tuple(vision_tokens.shape[1:-1]) != expected:
            raise ValueError(f"Vision input must be [B,{expected},1024]")
        if tuple(quality_tokens.shape[1:-1]) != expected:
            raise ValueError(f"Quality input must be [B,{expected},14]")
        batch = vision_tokens.shape[0]
        videos = self.settings.videos_per_field
        if language_tokens.ndim != 3 or language_tokens.shape[0] != batch:
            raise ValueError("One broadcastable prompt sequence is required per physical field")

        prompt_base = self.language_adapter(language_tokens.float())
        prompt_valid = language_mask.bool().unsqueeze(-1)
        prompt_summary = (prompt_base * prompt_valid).sum(1) / prompt_valid.sum(1).clamp_min(1)
        prompt_scale, prompt_shift = self.prompt_to_visual(prompt_summary).chunk(2, -1)
        prompt_tokens = prompt_base[:, None].expand(-1, videos, -1, -1)
        prompt_mask = language_mask.bool()[:, None].expand(-1, videos, -1)

        qwen = self.vision_adapter(vision_tokens.float())
        quality = self.quality_adapter(quality_tokens.float())
        vision = qwen + torch.sigmoid(self.raw_quality_gate) * quality
        vision = self.visual_input_norm(
            vision * (1 + 0.10 * torch.tanh(prompt_scale[:, None, None, None]))
            + 0.10 * prompt_shift[:, None, None, None]
        )
        routing: dict[str, dict[str, Any]] = {}
        alignments: list[torch.Tensor] = []
        guard_losses: list[torch.Tensor] = []

        fields1 = self.parallel_optics.fields(vision)
        electronic1 = self._vision_route(self.vision_routes[0], vision)
        if optical_enabled:
            routing["frame"] = self.parallel_router(fields1)
            optical1, guard1 = self.parallel_optics.expert(
                fields1, routing["frame"]["weights"]
            )
            vision = self._fuse(self.fusions[0], electronic1, optical1)
            alignments.append(_alignment(electronic1.flatten(0, 1), optical1.flatten(0, 1)))
            guard_losses.extend((routing["frame"]["guard_energy_fraction"], guard1))
        else:
            vision = electronic1

        fields2 = self.parallel_optics.fields(vision)
        electronic2 = self._vision_route(self.vision_routes[1], vision)
        if optical_enabled:
            optical2, guard2 = self.parallel_optics.global_path(fields2)
            vision = self._fuse(self.fusions[1], electronic2, optical2)
            alignments.append(_alignment(electronic2.flatten(0, 1), optical2.flatten(0, 1)))
            guard_losses.append(guard2)
        else:
            vision = electronic2

        image_tokens = self.frame_merger(
            torch.cat((vision.mean(3), vision.amax(3)), -1)
        ) + self.frame_position
        sequence = torch.cat((image_tokens, prompt_tokens), 2)
        mask = torch.cat(
            (
                torch.ones(
                    batch,
                    videos,
                    self.settings.frame_count,
                    dtype=torch.bool,
                    device=sequence.device,
                ),
                prompt_mask,
            ),
            2,
        )
        if sequence.shape[2] > self.settings.maximum_language_tokens:
            raise ValueError("Image+prompt tokens exceed the 72-pixel video field")
        sequence = (
            sequence + self.sequence_position[:, None, : sequence.shape[2]]
        ).masked_fill(~mask.unsqueeze(-1), 0.0)

        fields3 = self.serial_optics.fields(sequence)
        electronic3 = self._language_route(self.language_routes[0], sequence, mask)
        if optical_enabled:
            routing["video"] = self.serial_router(fields3)
            optical3, guard3 = self.serial_optics.expert(
                fields3, routing["video"]["weights"], sequence.shape[2]
            )
            sequence = self._fuse(self.fusions[2], electronic3, optical3, mask)
            alignments.append(_alignment(electronic3.flatten(0, 1), optical3.flatten(0, 1)))
            guard_losses.extend((routing["video"]["guard_energy_fraction"], guard3))
        else:
            sequence = electronic3

        fields4 = self.serial_optics.fields(sequence)
        electronic4 = self._language_route(self.language_routes[1], sequence, mask)
        if optical_enabled:
            optical4, guard4 = self.serial_optics.global_path(fields4, sequence.shape[2])
            sequence = self._fuse(self.fusions[3], electronic4, optical4, mask)
            alignments.append(_alignment(electronic4.flatten(0, 1), optical4.flatten(0, 1)))
            guard_losses.append(guard4)
        else:
            sequence = electronic4

        normalized = self.readout(
            vision.flatten(0, 1), sequence.flatten(0, 1), mask.flatten(0, 1)
        ).reshape(batch, videos)
        prediction = normalized * self.target_std + self.target_mean
        if routing:
            balance = torch.stack([v["balance_loss"] for v in routing.values()]).mean()
            importance = torch.stack([v["importance_loss"] for v in routing.values()]).mean()
            capture = torch.stack(
                [
                    (1 - v["capture_fraction"].clamp(0, 1)).mean()
                    for v in routing.values()
                ]
            ).mean()
            guard = torch.stack(guard_losses).mean()
        else:
            balance = importance = capture = guard = normalized.new_zeros(())
        return {
            "prediction": prediction,
            "normalized_prediction": normalized,
            "routing": routing,
            "optical_enabled": optical_enabled,
            "quality_gate": torch.sigmoid(self.raw_quality_gate),
            "optical_alignment_loss": torch.stack(alignments).mean()
            if alignments
            else normalized.new_zeros(()),
            "router_balance_loss": balance,
            "router_importance_loss": importance,
            "router_capture_loss": capture,
            "guard_energy_loss": guard,
        }

    def fusion_diagnostics(self) -> dict[str, dict[str, float]]:
        names = ("frame_expert", "frame_global", "video_expert", "video_global")
        return {name: dict(layer.last_diagnostics) for name, layer in zip(names, self.fusions)}

    def parameter_breakdown(self) -> dict[str, int]:
        groups = {
            "qwen_boundary_adapters": nn.ModuleList(
                [self.vision_adapter, self.quality_adapter, self.visual_input_norm, self.language_adapter, self.prompt_to_visual]
            ),
            "electronic_routes": nn.ModuleList([*self.vision_routes, *self.language_routes]),
            "optical_feature_paths": nn.ModuleList([self.parallel_optics, self.serial_optics]),
            "optical_routers": nn.ModuleList([self.parallel_router, self.serial_router]),
            "fusion": self.fusions,
            "multimodal_bridge": self.frame_merger,
            "position_parameters": nn.ParameterList([self.frame_position, self.sequence_position]),
            "single_metric_readout": self.readout,
        }
        result = {
            name: sum(parameter.numel() for parameter in module.parameters())
            for name, module in groups.items()
        }
        result["total_trainable"] = sum(p.numel() for p in self.parameters() if p.requires_grad)
        result["total_frozen_in_student"] = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        return result


def build_model(settings: MultiVideoSettings) -> MultiVideo9x4OpticalVQA:
    model = MultiVideo9x4OpticalVQA(settings)
    forbidden = [
        module.__class__.__name__
        for module in model.modules()
        if "attention" in module.__class__.__name__.lower()
        or "transformer" in module.__class__.__name__.lower()
    ]
    if forbidden:
        raise RuntimeError(f"Forbidden attention/Transformer modules: {forbidden}")
    return model


__all__ = ["MultiVideo9x4OpticalVQA", "build_model"]
