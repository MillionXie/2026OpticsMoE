from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from ..pca import FixedPCAProjection
from .geometry import MoEGeometry
from .physical import AngularSpectrumPropagator, PhaseLayer, aperture_linear_indices
from .router import ElectronicAmplitudeRouter


def lengths_from_cu(hidden: torch.Tensor, cu_seqlens: torch.Tensor | None) -> list[int]:
    if hidden.ndim != 2:
        raise ValueError(f"Packed vision hidden must be [sum(T),D], got {tuple(hidden.shape)}")
    if cu_seqlens is None:
        raise RuntimeError("Packed vision hidden requires per-image cu_seqlens")
    boundaries = cu_seqlens.detach().cpu().long().tolist()
    lengths = [end - start for start, end in zip(boundaries[:-1], boundaries[1:])]
    if not lengths or sum(lengths) != hidden.shape[0] or any(length <= 0 for length in lengths):
        raise RuntimeError("cu_seqlens do not match packed visual tokens")
    return lengths


class ExpertPhasePlane(nn.Module):
    def __init__(self, geometry: MoEGeometry, settings: Any) -> None:
        super().__init__()
        self.geometry = geometry
        self.experts = nn.ModuleList([
            PhaseLayer(
                geometry.expert_size,
                settings.phase_parameterization,
                settings.phase_init,
                settings.phase_init_std,
                settings.phase_dropout_mode,
                settings.phase_dropout_p,
                settings.phase_dropout_block_size,
                settings.phase_dropout_batch_shared,
            )
            for _ in range(geometry.num_experts)
        ])
        self.register_buffer(
            "aperture_indices",
            aperture_linear_indices(geometry.canvas_size, geometry.expert_apertures),
            persistent=False,
        )

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        batch = field.shape[0]
        indexes = self.aperture_indices.reshape(-1)
        crops = field.to(torch.complex64).flatten(1).index_select(1, indexes).reshape(
            batch,
            self.geometry.num_experts,
            self.geometry.expert_size,
            self.geometry.expert_size,
        )
        raw = torch.stack([expert.raw_phase for expert in self.experts], dim=0)
        reference = self.experts[0]
        if reference.parameterization == "sigmoid":
            phase = 2.0 * math.pi * torch.sigmoid(raw)
        elif reference.parameterization == "unconstrained":
            phase = raw
        else:
            raise ValueError(f"Unsupported phase parameterization {reference.parameterization!r}")
        modulation = torch.exp(1j * phase).to(torch.complex64)
        if (
            self.training
            and reference.dropout_active
            and reference.dropout_mode != "none"
            and reference.dropout_p > 0
        ):
            modulation = self._apply_dropout(modulation, reference, batch)
        modulated = crops * modulation
        return torch.zeros(
            (batch, self.geometry.canvas_size * self.geometry.canvas_size),
            dtype=torch.complex64,
            device=field.device,
        ).scatter(
            1,
            indexes.unsqueeze(0).expand(batch, -1),
            modulated.reshape(batch, -1),
        ).reshape(batch, self.geometry.canvas_size, self.geometry.canvas_size)

    def _apply_dropout(self, modulation: torch.Tensor, reference: PhaseLayer, batch: int) -> torch.Tensor:
        dropout_batch = 1 if reference.dropout_batch_shared else batch
        if reference.dropout_mode == "phase_bypass":
            keep = torch.rand(
                dropout_batch,
                len(self.experts),
                reference.size,
                reference.size,
                device=modulation.device,
            ) >= reference.dropout_p
        elif reference.dropout_mode == "block_phase_bypass":
            block = max(1, reference.dropout_block_size)
            low = math.ceil(reference.size / block)
            keep = torch.rand(
                dropout_batch,
                len(self.experts),
                low,
                low,
                device=modulation.device,
            ) >= reference.dropout_p
            keep = keep.repeat_interleave(block, -2).repeat_interleave(block, -1)
            keep = keep[..., : reference.size, : reference.size]
        else:
            raise RuntimeError(f"Unsupported active phase dropout mode {reference.dropout_mode!r}")
        keep = keep.to(torch.complex64)
        return keep * modulation.unsqueeze(0) + (1.0 - keep)

    def set_phase_dropout_active(self, active: bool) -> None:
        for expert in self.experts:
            expert.set_dropout_active(active)


class GlobalPhasePlane(nn.Module):
    def __init__(self, geometry: MoEGeometry, settings: Any) -> None:
        super().__init__()
        self.geometry = geometry
        self.phase = PhaseLayer(
            geometry.active_size,
            settings.phase_parameterization,
            settings.phase_init,
            settings.phase_init_std,
            settings.phase_dropout_mode,
            settings.phase_dropout_p,
            settings.phase_dropout_block_size,
            settings.phase_dropout_batch_shared,
        )

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        output = field.to(torch.complex64).clone()
        aperture = self.geometry.active_aperture
        output[:, aperture.y0:aperture.y1, aperture.x0:aperture.x1] = self.phase(
            output[:, aperture.y0:aperture.y1, aperture.x0:aperture.x1]
        )
        return output

    def set_phase_dropout_active(self, active: bool) -> None:
        self.phase.set_dropout_active(active)


class SignedDetectorReadout(nn.Module):
    """Square-law detector with separate signed and reload representations."""

    def __init__(self, geometry: MoEGeometry, settings: Any) -> None:
        super().__init__()
        self.geometry = geometry
        self.output_size = int(settings.detector_output_size)
        self.pool = nn.AdaptiveAvgPool2d((self.output_size, self.output_size))
        normalized_shape = (
            self.output_size
            if settings.detector_layernorm_scope == "per_token"
            else (self.output_size, self.output_size)
        )
        self.norm = nn.LayerNorm(
            normalized_shape,
            eps=settings.detector_layernorm_eps,
            elementwise_affine=settings.detector_layernorm_affine,
        )

    def forward(
        self,
        field: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        full_intensity = field.to(torch.complex64).abs().square().float()
        aperture = self.geometry.detector_aperture
        detector_intensity = full_intensity[
            :,
            aperture.y0:aperture.y1,
            aperture.x0:aperture.x1,
        ]
        pooled = self.pool(detector_intensity.unsqueeze(1)).squeeze(1)
        signed_readout = self.norm(pooled)
        reload_amplitude = F.relu(signed_readout)
        return signed_readout, reload_amplitude, detector_intensity


@dataclass
class OpticalState:
    routing: dict[str, torch.Tensor]
    field: torch.Tensor
    signed_readout: torch.Tensor | None = None
    reload_amplitude: torch.Tensor | None = None


class PCAHomogeneousMoEOpticalCore(nn.Module):
    """Four-stage MoE16 whose trainable path never leaves PCA latent space."""

    def __init__(self, max_tokens: int, settings: Any) -> None:
        super().__init__()
        self.latent_dim = int(settings.latent_dim)
        self.max_tokens = int(max_tokens)
        self.geometry = MoEGeometry(
            settings.canvas_size,
            settings.active_size,
            settings.expert_size,
            settings.expert_pitch,
            settings.num_experts,
            settings.expert_grid_rows,
            settings.expert_grid_cols,
        )
        self.geometry.validate()
        self.latent_input_norm = nn.LayerNorm(self.latent_dim)
        self.nonnegative = nn.Softplus()
        self.router = ElectronicAmplitudeRouter(
            self.geometry,
            settings.top_k,
            settings.router_pool_size,
            settings.router_temperature,
            settings.router_input_layernorm_enabled,
            settings.router_input_layernorm_eps,
        )
        self.amplitude_slm_weight_domain = settings.amplitude_slm_weight_domain
        self.amplitude_slm_input_normalization = settings.amplitude_slm_input_normalization
        self.register_buffer(
            "expert_canvas_indices",
            aperture_linear_indices(settings.canvas_size, self.geometry.expert_apertures),
            persistent=False,
        )
        wavelength_m = settings.wavelength_nm * 1e-9
        pixel_size_m = settings.pixel_pitch_um * 1e-6
        propagation = {
            "wavelength_m": wavelength_m,
            "pixel_size_m": pixel_size_m,
            "grid_size": settings.canvas_size,
            "k_space_constraint_enabled": settings.k_space_constraint_enabled,
            "theta_max_deg": settings.theta_max_deg,
        }
        self.expert_layers = nn.ModuleList([
            ExpertPhasePlane(self.geometry, settings)
            for _ in range(settings.expert_layers)
        ])
        self.interstage_propagator = AngularSpectrumPropagator(
            distance_m=settings.expert_interlayer_distance_m,
            **propagation,
        )
        self.last_expert_propagator = AngularSpectrumPropagator(
            distance_m=settings.last_expert_to_global_distance_m,
            **propagation,
        )
        self.detector_propagator = AngularSpectrumPropagator(
            distance_m=settings.global_to_detector_distance_m,
            **propagation,
        )
        self.global_phase = GlobalPhasePlane(self.geometry, settings)
        self.readout = SignedDetectorReadout(self.geometry, settings)
        self.last_input_fields: torch.Tensor | None = None
        self.last_signed_readouts: list[torch.Tensor] = []
        self.last_reload_amplitudes: list[torch.Tensor] = []
        self.last_detector_intensities: list[torch.Tensor] = []
        self.last_routing: dict[str, torch.Tensor] = {}
        self.capture_intermediate_fields = False

    def encode_groups(self, groups: list[torch.Tensor]) -> torch.Tensor:
        if not groups:
            raise ValueError("At least one PCA token group is required")
        counts = [len(group) for group in groups]
        for count in counts:
            if count > self.max_tokens:
                raise RuntimeError(
                    f"token count {count} exceeds optical field rows={self.max_tokens}; "
                    "no truncation, crop, pooling, or fallback resize is allowed."
                )
        packed = torch.cat(groups, dim=0)
        if packed.shape[-1] != self.latent_dim:
            raise ValueError(
                f"Expected PCA latent dimension {self.latent_dim}, got {packed.shape[-1]}"
            )
        projected = self.nonnegative(self.latent_input_norm(packed.float()))
        count_tensor = torch.tensor(counts, device=projected.device)
        valid = (
            torch.arange(self.latent_dim, device=projected.device)[None, :]
            < count_tensor[:, None]
        )
        mask = valid.unsqueeze(-1).expand(-1, -1, self.latent_dim)
        fields = projected.new_zeros((len(groups), self.latent_dim, self.latent_dim))
        return fields.masked_scatter(mask, projected.reshape(-1))

    def pack_fields(self, fields: torch.Tensor, lengths: list[int]) -> torch.Tensor:
        return torch.cat(
            [fields[index, :length] for index, length in enumerate(lengths)],
            dim=0,
        )

    def start(self, input_fields: torch.Tensor) -> OpticalState:
        routing = self.router(input_fields)
        self.last_routing = routing
        self.last_input_fields = (
            input_fields.detach().cpu() if self.capture_intermediate_fields else None
        )
        self.last_signed_readouts = []
        self.last_reload_amplitudes = []
        self.last_detector_intensities = []
        return OpticalState(routing=routing, field=self._fanout(input_fields, routing))

    def run_stage(
        self,
        stage: int,
        state: OpticalState,
        injection_field: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if stage < 0 or stage >= len(self.expert_layers):
            raise IndexError(stage)
        if stage > 0:
            if state.signed_readout is None:
                raise RuntimeError("Optical stages must execute in order")
            signed_for_reload = state.signed_readout
            if injection_field is not None:
                if injection_field.shape != signed_for_reload.shape:
                    raise RuntimeError("DeepStack PCA injection field shape mismatch")
                signed_for_reload = signed_for_reload + injection_field
            state.reload_amplitude = F.relu(signed_for_reload)
            state.field = self._fanout(state.reload_amplitude, state.routing)
        modulated = self.expert_layers[stage](state.field)
        final = stage == len(self.expert_layers) - 1
        if final:
            propagated = self.last_expert_propagator(modulated)
            propagated = self.detector_propagator(self.global_phase(propagated))
        else:
            propagated = self.interstage_propagator(modulated)
        signed, reload_amplitude, intensity = self.readout(propagated)
        state.field = propagated
        state.signed_readout = signed
        state.reload_amplitude = reload_amplitude
        if self.capture_intermediate_fields:
            self.last_signed_readouts.append(signed.detach().cpu())
            self.last_reload_amplitudes.append(reload_amplitude.detach().cpu())
            self.last_detector_intensities.append(intensity.detach().cpu())
        return signed

    def _fanout(
        self,
        input_fields: torch.Tensor,
        routing: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        amplitude = input_fields.float()
        if self.amplitude_slm_input_normalization == "per_sample_max":
            amplitude = amplitude / amplitude.amax(
                dim=(-2, -1), keepdim=True
            ).clamp_min(1e-8)
        elif self.amplitude_slm_input_normalization != "none":
            raise RuntimeError("Unsupported amplitude SLM input normalization")
        weights = routing["weights"].float().clamp_min(0.0)
        scales = weights if self.amplitude_slm_weight_domain == "amplitude" else weights.sqrt()
        values = (amplitude[:, None] * scales[:, :, None, None]).reshape(len(amplitude), -1)
        indexes = self.expert_canvas_indices.reshape(-1)
        canvas = amplitude.new_zeros(
            (len(amplitude), self.geometry.canvas_size * self.geometry.canvas_size)
        ).scatter(
            1,
            indexes.unsqueeze(0).expand(len(amplitude), -1),
            values,
        ).reshape(len(amplitude), self.geometry.canvas_size, self.geometry.canvas_size)
        return torch.complex(canvas, torch.zeros_like(canvas))

    def router_losses(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.last_routing["balance_loss"], self.last_routing["importance_loss"]

    def set_phase_dropout_active(self, active: bool) -> None:
        for layer in self.expert_layers:
            layer.set_phase_dropout_active(active)
        self.global_phase.set_phase_dropout_active(active)

    def parameter_breakdown(self) -> dict[str, int]:
        expert_phase = sum(parameter.numel() for layer in self.expert_layers for parameter in layer.parameters())
        global_phase = sum(parameter.numel() for parameter in self.global_phase.parameters())
        router = sum(parameter.numel() for parameter in self.router.parameters())
        latent_norm = sum(parameter.numel() for parameter in self.latent_input_norm.parameters())
        total = sum(parameter.numel() for parameter in self.parameters())
        return {
            "expert_phase_parameters": expert_phase,
            "global_phase_parameters": global_phase,
            "optical_phase_parameters": expert_phase + global_phase,
            "router_parameters": router,
            "latent_input_norm_parameters": latent_norm,
            "trainable_hidden_to_latent_linear_parameters": 0,
            "trainable_latent_to_hidden_linear_parameters": 0,
            "total_parameters": total,
            "trainable_parameters": sum(
                parameter.numel() for parameter in self.parameters() if parameter.requires_grad
            ),
        }


class VisionPCAOpticalMoE(nn.Module):
    def __init__(self, pca: FixedPCAProjection, settings: Any) -> None:
        super().__init__()
        self.pca = pca
        self.core = PCAHomogeneousMoEOpticalCore(settings.max_visual_tokens, settings)
        self.stage_latents: list[torch.Tensor] = []
        self.decoded_outputs: list[torch.Tensor] = []
        self.last_token_counts: list[int] = []

    def compute(self, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor | None) -> None:
        lengths = lengths_from_cu(hidden_states, cu_seqlens)
        self.last_token_counts = lengths
        latent = self.pca.encode(hidden_states)
        fields = self.core.encode_groups(list(latent.split(lengths)))
        state = self.core.start(fields)
        self.stage_latents = []
        self.decoded_outputs = []
        for stage in range(len(self.core.expert_layers)):
            signed = self.core.run_stage(stage, state)
            packed = self.core.pack_fields(signed, lengths)
            self.stage_latents.append(packed)
            self.decoded_outputs.append(self.pca.decode(packed).to(hidden_states.dtype))

    def output_for_slot(self, slot: int) -> torch.Tensor:
        if slot >= len(self.decoded_outputs):
            raise RuntimeError("Vision optical stages have not been computed")
        return self.decoded_outputs[slot]

    def router_losses(self):
        return self.core.router_losses()

    def parameter_breakdown(self):
        return self.core.parameter_breakdown()


class LanguagePCAOpticalMoE(nn.Module):
    def __init__(self, pca: FixedPCAProjection, settings: Any) -> None:
        super().__init__()
        self.pca = pca
        self.core = PCAHomogeneousMoEOpticalCore(settings.max_language_tokens, settings)
        self.valid_mask: torch.Tensor | None = None
        self.lengths: list[int] = []
        self.state: OpticalState | None = None
        self.last_decoded_hidden: torch.Tensor | None = None
        self.stage_latents: list[torch.Tensor] = []

    def set_attention_mask(self, attention_mask: torch.Tensor) -> None:
        lengths = [int(value) for value in attention_mask.long().sum(1).tolist()]
        maximum = max(lengths)
        if maximum > self.core.max_tokens:
            raise RuntimeError(
                f"language sequence length {maximum} exceeds optical field rows={self.core.max_tokens}; "
                "shorten the caption/prompt or lower the visual token budget. No truncation is allowed."
            )
        self.valid_mask = attention_mask.bool()
        self.lengths = lengths

    def _mask(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.valid_mask is None or self.valid_mask.shape != hidden.shape[:2]:
            raise RuntimeError("Call prepare_student_batch with the original attention mask")
        return self.valid_mask.to(hidden.device)

    def _groups(self, hidden: torch.Tensor) -> list[torch.Tensor]:
        return list(hidden[self._mask(hidden)].split(self.lengths))

    def _scatter(self, packed: torch.Tensor, template: torch.Tensor) -> torch.Tensor:
        output = torch.zeros_like(template)
        output[self._mask(template)] = packed.to(output.dtype)
        return output

    def _delta_field(self, delta: torch.Tensor) -> torch.Tensor:
        projected = self.pca.encode_additive_delta(delta[self._mask(delta)])
        groups = list(projected.split(self.lengths))
        fields = projected.new_zeros((len(groups), self.core.latent_dim, self.core.latent_dim))
        for index, group in enumerate(groups):
            fields[index, : len(group)] = group
        return fields

    def forward_stage(self, stage: int, hidden_states: torch.Tensor) -> torch.Tensor:
        injection = None
        if stage == 0:
            latent = self.pca.encode(hidden_states[self._mask(hidden_states)])
            fields = self.core.encode_groups(list(latent.split(self.lengths)))
            self.state = self.core.start(fields)
            self.stage_latents = []
        else:
            if self.state is None or self.last_decoded_hidden is None:
                raise RuntimeError("Language optical stages must execute in order")
            injection = self._delta_field(hidden_states - self.last_decoded_hidden)
        assert self.state is not None
        signed = self.core.run_stage(stage, self.state, injection)
        packed = self.core.pack_fields(signed, self.lengths)
        self.stage_latents.append(packed)
        decoded = self.pca.decode(packed)
        output = self._scatter(decoded, hidden_states)
        self.last_decoded_hidden = output.clone()
        return output

    def router_losses(self):
        return self.core.router_losses()

    def parameter_breakdown(self):
        return self.core.parameter_breakdown()
