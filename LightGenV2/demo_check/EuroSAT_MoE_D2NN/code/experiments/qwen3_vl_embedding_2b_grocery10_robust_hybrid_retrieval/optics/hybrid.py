from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.optics.moe import (
    ExpertPhasePlane,
    GlobalPhasePlane,
    HomogeneousMoEOpticalCore,
    LanguageDeepStackHomogeneousMoE,
    VisionDeepStackHomogeneousMoE,
)
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.optics.physical import (
    PhaseLayer,
)
from ..settings import residual_logit


def translate_zero_fill(
    value: torch.Tensor,
    shift_y: int,
    shift_x: int,
    *,
    fill: float | complex = 0.0,
) -> torch.Tensor:
    """Integer translation on the last two axes without circular wraparound."""

    height, width = value.shape[-2:]
    shift_y, shift_x = int(shift_y), int(shift_x)
    output = torch.full_like(value, fill)
    if abs(shift_y) >= height or abs(shift_x) >= width:
        return output
    source_y0, source_y1 = max(0, -shift_y), min(height, height - shift_y)
    source_x0, source_x1 = max(0, -shift_x), min(width, width - shift_x)
    target_y0, target_y1 = source_y0 + shift_y, source_y1 + shift_y
    target_x0, target_x1 = source_x0 + shift_x, source_x1 + shift_x
    output[..., target_y0:target_y1, target_x0:target_x1] = value[
        ..., source_y0:source_y1, source_x0:source_x1
    ]
    return output


class _RandomRegistrationMixin:
    shift_max_px: int
    alignment_enabled: bool
    apply_during_eval: bool

    def _registration_active(self) -> bool:
        return bool(
            self.alignment_enabled and (self.training or self.apply_during_eval)
        )

    def _sample_shift(self, reference: torch.Tensor) -> tuple[int, int]:
        if not self._registration_active() or self.shift_max_px <= 0:
            return 0, 0
        values = torch.randint(
            -self.shift_max_px,
            self.shift_max_px + 1,
            (2,),
            device=reference.device,
        )
        return int(values[0].item()), int(values[1].item())


class RobustExpertPhasePlane(_RandomRegistrationMixin, ExpertPhasePlane):
    def __init__(self, geometry: Any, settings: Any) -> None:
        super().__init__(geometry, settings)
        self.shift_max_px = int(settings.phase_shift_max_px)
        self.alignment_enabled = bool(settings.alignment_augmentation_enabled)
        self.apply_during_eval = bool(settings.alignment_apply_during_eval)

    def _stacked_modulation(self, batch_size: int) -> torch.Tensor:
        modulation = super()._stacked_modulation(batch_size)
        shift_y, shift_x = self._sample_shift(modulation)
        return translate_zero_fill(
            modulation, shift_y, shift_x, fill=complex(1.0, 0.0)
        )


class RobustPhaseLayer(_RandomRegistrationMixin, PhaseLayer):
    def __init__(self, size: int, settings: Any) -> None:
        super().__init__(
            size,
            settings.phase_parameterization,
            settings.phase_init,
            settings.phase_init_std,
            settings.phase_dropout_mode,
            settings.phase_dropout_p,
            settings.phase_dropout_block_size,
            settings.phase_dropout_batch_shared,
        )
        self.shift_max_px = int(settings.phase_shift_max_px)
        self.alignment_enabled = bool(settings.alignment_augmentation_enabled)
        self.apply_during_eval = bool(settings.alignment_apply_during_eval)

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        modulation = torch.exp(1j * self.phase()).to(torch.complex64)
        if (
            self.training
            and self.dropout_active
            and self.dropout_mode != "none"
            and self.dropout_p > 0.0
        ):
            batch = 1 if self.dropout_batch_shared else field.shape[0]
            if self.dropout_mode == "phase_bypass":
                keep = torch.rand(
                    batch, self.size, self.size, device=field.device
                ) >= self.dropout_p
            elif self.dropout_mode == "block_phase_bypass":
                block = max(1, self.dropout_block_size)
                low = math.ceil(self.size / block)
                keep = torch.rand(batch, low, low, device=field.device) >= self.dropout_p
                keep = keep.repeat_interleave(block, -2).repeat_interleave(block, -1)
                keep = keep[:, : self.size, : self.size]
            else:
                raise RuntimeError(
                    f"Unsupported active phase dropout mode {self.dropout_mode!r}"
                )
            keep = keep.to(torch.complex64)
            modulation = keep * modulation.unsqueeze(0) + (1.0 - keep)
        shift_y, shift_x = self._sample_shift(modulation)
        modulation = translate_zero_fill(
            modulation, shift_y, shift_x, fill=complex(1.0, 0.0)
        )
        return field.to(torch.complex64) * modulation


class RobustGlobalPhasePlane(GlobalPhasePlane):
    def __init__(self, geometry: Any, settings: Any) -> None:
        nn.Module.__init__(self)
        self.geometry = geometry
        self.phase = RobustPhaseLayer(geometry.active_size, settings)


class LocalElectronicRefiner(nn.Module):
    """Small translation-aware refiner; parameter count is independent of CCD area."""

    def __init__(self, width: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.input_projection = nn.Conv2d(1, width, kernel_size=1)
        self.local = nn.Conv2d(
            width, width, kernel_size=3, padding=1, groups=width
        )
        self.context = nn.Conv2d(
            width,
            width,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            groups=width,
        )
        self.dropout = nn.Dropout2d(dropout)
        self.output_projection = nn.Conv2d(width, 1, kernel_size=1)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        hidden = F.gelu(self.local(self.input_projection(value)))
        hidden = self.dropout(F.gelu(self.context(hidden)))
        return self.output_projection(hidden)


class LearnableResidualFusion(nn.Module):
    """Convex input/optical blend followed by a local electronic correction."""

    def __init__(
        self,
        initial_input_weight: float,
        width: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_weight_logit = nn.Parameter(
            torch.tensor(residual_logit(initial_input_weight), dtype=torch.float32)
        )
        self.refiner = LocalElectronicRefiner(width, dilation, dropout)

    @staticmethod
    def _unit_rms(value: torch.Tensor) -> torch.Tensor:
        # Top-k routing leaves whole expert crops exactly zero.  Clamp the
        # mean-square *before* sqrt: clamping only the resulting scale still
        # executes SqrtBackward at zero and yields an infinite derivative,
        # which becomes NaN when multiplied by the zero crop gradient.
        mean_square = value.float().square().mean(
            dim=(-2, -1), keepdim=True
        )
        scale = mean_square.clamp_min(1.0e-12).sqrt()
        return value.float() / scale

    def input_weight(self) -> torch.Tensor:
        return torch.sigmoid(self.input_weight_logit)

    def forward(
        self, optical_output: torch.Tensor, residual_input: torch.Tensor
    ) -> torch.Tensor:
        if optical_output.shape != residual_input.shape or optical_output.ndim != 3:
            raise ValueError("Residual fusion expects matching [N,H,W] tensors")
        optical = self._unit_rms(optical_output)
        residual = self._unit_rms(residual_input)
        weight = self.input_weight()
        mixed = weight * residual + (1.0 - weight) * optical
        correction = self.refiner(mixed.unsqueeze(1)).squeeze(1)
        return F.relu(mixed + correction)


class RobustHybridOpticalCore(_RandomRegistrationMixin, HomogeneousMoEOpticalCore):
    """Two optical phase planes, each closed by lightweight electronic fusion."""

    def __init__(self, hidden_size: int, max_tokens: int, settings: Any) -> None:
        super().__init__(hidden_size, max_tokens, settings)
        self.expert_layers = nn.ModuleList(
            [RobustExpertPhasePlane(self.geometry, settings) for _ in range(settings.expert_layers)]
        )
        self.global_phase = RobustGlobalPhasePlane(self.geometry, settings)
        fusion_args = (
            settings.hybrid_residual_initial_weight,
            settings.hybrid_refiner_width,
            settings.hybrid_refiner_dilation,
            settings.hybrid_refiner_dropout,
        )
        self.expert_fusions = nn.ModuleList(
            [LearnableResidualFusion(*fusion_args) for _ in range(settings.expert_layers)]
        )
        self.detector_fusion = LearnableResidualFusion(*fusion_args)
        self.shift_max_px = int(settings.ccd_shift_max_px)
        self.input_shift_max_px = int(settings.input_shift_max_px)
        self.alignment_enabled = bool(settings.alignment_augmentation_enabled)
        self.apply_during_eval = bool(settings.alignment_apply_during_eval)
        self.current_input_fields: torch.Tensor | None = None

    def _sample_bounded_shift(
        self, reference: torch.Tensor, maximum: int
    ) -> tuple[int, int]:
        original = self.shift_max_px
        self.shift_max_px = int(maximum)
        try:
            return self._sample_shift(reference)
        finally:
            self.shift_max_px = original

    def begin(
        self, input_fields: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        self.current_input_fields = input_fields
        field, routing = super().begin(input_fields)
        shift_y, shift_x = self._sample_bounded_shift(
            field, self.input_shift_max_px
        )
        return translate_zero_fill(field, shift_y, shift_x), routing

    def _expert_crops(self, canvas: torch.Tensor) -> torch.Tensor:
        batch = canvas.shape[0]
        indices = self.expert_canvas_indices.reshape(-1)
        return canvas.flatten(1).index_select(1, indices).reshape(
            batch,
            self.geometry.num_experts,
            self.geometry.expert_size,
            self.geometry.expert_size,
        )

    def _scatter_expert_crops(self, crops: torch.Tensor) -> torch.Tensor:
        batch = crops.shape[0]
        indices = self.expert_canvas_indices.reshape(-1)
        return crops.new_zeros(
            (batch, self.geometry.canvas_size * self.geometry.canvas_size)
        ).scatter(
            1,
            indices.unsqueeze(0).expand(batch, -1),
            crops.reshape(batch, -1),
        ).reshape(batch, self.geometry.canvas_size, self.geometry.canvas_size)

    def run_stage(
        self,
        index: int,
        field: torch.Tensor,
        routing: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        replay = self.hardware_stage_reload_fields.get(int(index))
        if replay is not None:
            if tuple(replay.shape) != tuple(field.shape):
                raise RuntimeError("Hardware stage reload shape does not match optical field")
            field = replay.to(device=field.device, dtype=torch.complex64)
        else:
            field = self.propagator(self.expert_layers[index](field))
            shift_y, shift_x = self._sample_shift(field)
            field = translate_zero_fill(field, shift_y, shift_x)
            if self.interlayer_enabled:
                conversion = self.interlayer_conversions[index]
                field = conversion(
                    field,
                    selected_experts=(
                        routing["selected_mask"]
                        if self.interlayer_hard_route_mask
                        else None
                    ),
                    routing_weights=(
                        routing["weights"]
                        if self.interlayer_reapply_routing_weights
                        else None
                    ),
                    input_amplitude_scales=routing.get("amplitude_scales"),
                )
                self.last_expert_response = conversion.last_expert_response
                self.last_response_gain = conversion.last_response_gain

        # ``SquareDetectionLayerNormReload`` and hardware stage replay both
        # return a zero-phase, explicitly nonnegative amplitude.  Reading its
        # real component is therefore exact.  Do not use ``complex.abs()``
        # here: hard routing creates many exact complex zeros, and the
        # derivative of |z| at z=0 is undefined in PyTorch, which contaminates
        # the router and every expert phase with NaN on the first backward.
        optical = self._expert_crops(field.real.float().clamp_min(0.0))
        residual = self._expert_crops(routing["amplitude_slm_canvas"].float())
        batch, experts, height, width = optical.shape
        fused = self.expert_fusions[index](
            optical.reshape(batch * experts, height, width),
            residual.reshape(batch * experts, height, width),
        ).reshape_as(optical)
        fused = fused * routing["selected_mask"][..., None, None].to(fused.dtype)
        fused_canvas = self._scatter_expert_crops(fused)
        field = torch.complex(fused_canvas, torch.zeros_like(fused_canvas))
        if self.capture_intermediate_fields:
            count = min(self.capture_sample_count, len(field))
            self.last_stage_fields.append(field[:count].detach().cpu())
        return field

    def read_hidden(
        self,
        field: torch.Tensor,
        lengths: list[int],
        boundary_dtype: torch.dtype,
        *,
        final: bool = False,
    ) -> torch.Tensor:
        measured_intensity = self.hardware_final_detector_intensity if final else None
        if final and measured_intensity is None:
            field = self.propagator(self.global_phase(field))
            shift_y, shift_x = self._sample_shift(field)
            field = translate_zero_fill(field, shift_y, shift_x)
            if self.capture_intermediate_fields:
                count = min(self.capture_sample_count, len(field))
                aperture = self.geometry.detector_aperture
                self.last_detector_complex_field = field[
                    :count, aperture.y0 : aperture.y1, aperture.x0 : aperture.x1
                ].detach().cpu().to(torch.complex64)
        if measured_intensity is not None:
            readout, intensity = self.readout.forward_intensity(
                measured_intensity.to(field.device)
            )
        else:
            readout, intensity = self.readout(field)
        if final:
            if self.current_input_fields is None:
                raise RuntimeError("Final residual input is unavailable")
            readout = self.detector_fusion(readout, self.current_input_fields)
            if not torch.isfinite(readout).all():
                raise RuntimeError("Final robust detector readout contains NaN or Inf")
            self.current_detector_readout = readout
        if final and self.capture_intermediate_fields:
            count = min(self.capture_sample_count, len(field))
            self.last_detector_intensity = intensity[:count].detach().cpu()
            self.last_detector_readout = readout[:count].detach().cpu()
        packed = torch.cat(
            [readout[row, :length] for row, length in enumerate(lengths)], dim=0
        )
        return self.output_adapter(packed).to(boundary_dtype)

    def parameter_breakdown(self) -> dict[str, Any]:
        result = super().parameter_breakdown()
        fusion_parameters = sum(
            parameter.numel()
            for module in (*self.expert_fusions, self.detector_fusion)
            for parameter in module.parameters()
        )
        result["learnable_residual_fusion_parameters"] = fusion_parameters
        result["learnable_residual_weights"] = len(self.expert_fusions) + 1
        result["total_parameters"] = sum(p.numel() for p in self.parameters())
        result["trainable_parameters"] = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        return result


class RobustVisionOpticalMoE(VisionDeepStackHomogeneousMoE):
    def __init__(self, hidden_size: int, settings: Any) -> None:
        nn.Module.__init__(self)
        self.core = RobustHybridOpticalCore(
            hidden_size, settings.max_visual_tokens, settings
        )
        self.tap_stages = tuple(int(stage) for stage in settings.vision_tap_stages)
        self.last_token_counts: list[int] = []
        self.tap_outputs: list[torch.Tensor] = []
        self.last_output: torch.Tensor | None = None
        self.last_residual_base: torch.Tensor | None = None


class RobustLanguageOpticalMoE(LanguageDeepStackHomogeneousMoE):
    def __init__(self, hidden_size: int, settings: Any) -> None:
        nn.Module.__init__(self)
        self.core = RobustHybridOpticalCore(
            hidden_size, settings.max_language_tokens, settings
        )
        self.valid_mask: torch.Tensor | None = None
        self.field: torch.Tensor | None = None
        self.routing: dict[str, torch.Tensor] | None = None
        self.lengths: list[int] = []
        self.positions: list[torch.Tensor] = []
        self.last_hidden: torch.Tensor | None = None
        self.last_output: torch.Tensor | None = None
        self.residual_base: torch.Tensor | None = None
        self.deepstack_injection_count: int | None = None
