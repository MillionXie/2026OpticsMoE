from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .geometry import MoEGeometry
from .physical import (AngularSpectrumPropagator, PhaseLayer,
                       SquareDetectionLayerNormReload, aperture_linear_indices)
from .router import ElectronicAmplitudeRouter


def lengths_from_cu(hidden: torch.Tensor, cu_seqlens: torch.Tensor | None) -> list[int]:
    if hidden.ndim != 2: raise ValueError(f"Packed vision hidden must be [sum(T),D], got {tuple(hidden.shape)}")
    if cu_seqlens is None: raise RuntimeError("Packed vision hidden requires per-image cu_seqlens")
    boundaries = cu_seqlens.detach().cpu().long().tolist()
    lengths = [end - start for start, end in zip(boundaries[:-1], boundaries[1:])]
    if not lengths or sum(lengths) != hidden.shape[0] or any(length <= 0 for length in lengths):
        raise RuntimeError("cu_seqlens do not match packed visual tokens")
    return lengths


class ExpertPhasePlane(nn.Module):
    def __init__(self, geometry: MoEGeometry, settings: Any) -> None:
        super().__init__(); self.geometry = geometry
        self.experts = nn.ModuleList([
            PhaseLayer(geometry.expert_size, settings.phase_parameterization, settings.phase_init,
                       settings.phase_init_std, settings.phase_dropout_mode, settings.phase_dropout_p,
                       settings.phase_dropout_block_size, settings.phase_dropout_batch_shared)
            for _ in range(geometry.num_experts)])
        self.register_buffer(
            "aperture_indices",
            aperture_linear_indices(geometry.canvas_size, geometry.expert_apertures),
            persistent=False,
        )

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        batch = field.shape[0]; flat_indices = self.aperture_indices.reshape(-1)
        crops = field.to(torch.complex64).flatten(1).index_select(1, flat_indices).reshape(
            batch, self.geometry.num_experts, self.geometry.expert_size, self.geometry.expert_size
        )
        # Preserve the checkpoint-compatible ModuleList of independent phase
        # parameters, but evaluate all 16 masks in one batched tensor path.
        # The former implementation launched sigmoid/exp/multiply once per
        # expert at every optical stage.
        modulated = crops * self._stacked_modulation(batch)
        return torch.zeros(
            (batch, self.geometry.canvas_size * self.geometry.canvas_size),
            dtype=torch.complex64,
            device=field.device,
        ).scatter(
            1,
            flat_indices.unsqueeze(0).expand(batch, -1),
            modulated.reshape(batch, -1),
        ).reshape(batch, self.geometry.canvas_size, self.geometry.canvas_size)

    def _stacked_modulation(self, batch_size: int) -> torch.Tensor:
        reference = self.experts[0]
        raw_phase = torch.stack([expert.raw_phase for expert in self.experts], dim=0)
        if reference.parameterization == "sigmoid":
            phase = 2.0 * math.pi * torch.sigmoid(raw_phase)
        elif reference.parameterization == "unconstrained":
            phase = raw_phase
        else:
            raise ValueError(f"Unsupported phase parameterization {reference.parameterization!r}")
        modulation = torch.exp(1j * phase).to(torch.complex64)

        dropout_active = (
            self.training
            and reference.dropout_active
            and reference.dropout_mode != "none"
            and reference.dropout_p > 0.0
        )
        if not dropout_active:
            return modulation
        if not all(
            expert.dropout_active == reference.dropout_active
            and expert.dropout_mode == reference.dropout_mode
            and expert.dropout_p == reference.dropout_p
            and expert.dropout_block_size == reference.dropout_block_size
            and expert.dropout_batch_shared == reference.dropout_batch_shared
            for expert in self.experts
        ):
            raise RuntimeError("All experts in one phase plane must share phase-dropout configuration")

        dropout_batch = 1 if reference.dropout_batch_shared else int(batch_size)
        if reference.dropout_mode == "phase_bypass":
            # Expert-major random layout reproduces the former sequence of one
            # random draw per expert while still applying modulation in one op.
            keep = torch.rand(
                len(self.experts),
                dropout_batch,
                reference.size,
                reference.size,
                device=raw_phase.device,
            ) >= reference.dropout_p
        elif reference.dropout_mode == "block_phase_bypass":
            block = max(1, reference.dropout_block_size)
            low = math.ceil(reference.size / block)
            keep = torch.rand(
                len(self.experts),
                dropout_batch,
                low,
                low,
                device=raw_phase.device,
            ) >= reference.dropout_p
            keep = keep.repeat_interleave(block, -2).repeat_interleave(block, -1)
            keep = keep[..., :reference.size, :reference.size]
        else:
            raise RuntimeError(f"Unsupported active phase dropout mode {reference.dropout_mode!r}")
        keep = keep.permute(1, 0, 2, 3).to(torch.complex64)
        return keep * modulation.unsqueeze(0) + (1.0 - keep)

    def set_phase_dropout_active(self, active: bool) -> None:
        for expert in self.experts: expert.set_dropout_active(active)


class GlobalPhasePlane(nn.Module):
    def __init__(self, geometry: MoEGeometry, settings: Any) -> None:
        super().__init__(); self.geometry = geometry
        self.phase = PhaseLayer(geometry.active_size, settings.phase_parameterization, settings.phase_init,
                                settings.phase_init_std, settings.phase_dropout_mode, settings.phase_dropout_p,
                                settings.phase_dropout_block_size, settings.phase_dropout_batch_shared)

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        output = field.to(torch.complex64).clone(); aperture = self.geometry.active_aperture
        output[:, aperture.y0:aperture.y1, aperture.x0:aperture.x1] = self.phase(
            field[:, aperture.y0:aperture.y1, aperture.x0:aperture.x1])
        return output

    def set_phase_dropout_active(self, active: bool) -> None: self.phase.set_dropout_active(active)


class FullPlaneReadout(nn.Module):
    """Read the physical CCD ROI, then map it to token rows electronically.

    Free-space propagation is evaluated on the padded 1026x1026 FFT canvas,
    while the CCD observes the aligned 986x986 active footprint.  The crop is
    therefore performed before any pooling or normalization.
    """

    def __init__(self, geometry: MoEGeometry, settings: Any) -> None:
        super().__init__()
        self.geometry = geometry
        self.output_size = int(settings.detector_output_size)
        self.pool = nn.AdaptiveAvgPool2d((self.output_size, self.output_size))
        self.layernorm_scope = settings.detector_layernorm_scope
        normalized_shape = (
            self.output_size
            if self.layernorm_scope == "per_token"
            else (self.output_size, self.output_size)
        )
        self.norm = nn.LayerNorm(
            normalized_shape,
            eps=settings.detector_layernorm_eps,
            elementwise_affine=settings.detector_layernorm_affine,
        )
        self.nonlinearity = settings.detector_nonlinearity

    def forward(self, field: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        full_intensity = field.to(torch.complex64).abs().square().float()
        aperture = self.geometry.detector_aperture
        detector_intensity = full_intensity[
            :,
            aperture.y0:aperture.y1,
            aperture.x0:aperture.x1,
        ]
        return self.forward_intensity(detector_intensity)

    def forward_intensity(
        self, detector_intensity: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        expected = (
            self.geometry.detector_aperture.height,
            self.geometry.detector_aperture.width,
        )
        if detector_intensity.ndim != 3 or tuple(detector_intensity.shape[-2:]) != expected:
            raise ValueError(
                f"Measured final CCD intensity must be [B,{expected[0]},{expected[1]}], "
                f"got {tuple(detector_intensity.shape)}"
            )
        detector_intensity = detector_intensity.float()
        if not torch.isfinite(detector_intensity).all():
            raise RuntimeError("Measured final CCD intensity contains NaN or Inf")
        if torch.any(detector_intensity < -1.0e-7):
            raise RuntimeError("Measured final CCD intensity must be nonnegative")
        detector_intensity = detector_intensity.clamp_min(0.0)
        pooled = self.pool(detector_intensity.unsqueeze(1)).squeeze(1)
        normalized = self.norm(pooled)
        readout = F.relu(normalized) if self.nonlinearity == "relu" else F.softplus(normalized)
        return readout, detector_intensity


class HomogeneousMoEOpticalCore(nn.Module):
    """One-expert-stage homogeneous optical MoE core."""

    def __init__(self, hidden_size: int, max_tokens: int, settings: Any) -> None:
        super().__init__(); self.hidden_size = int(hidden_size); self.max_tokens = int(max_tokens)
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
        self.input_adapter = nn.Linear(hidden_size, settings.input_adapter_dim)
        self.input_norm = nn.LayerNorm(settings.input_adapter_dim); self.nonnegative = nn.Softplus()
        wavelength_m = settings.wavelength_nm * 1e-9; pixel_m = settings.pixel_pitch_um * 1e-6
        self.router = ElectronicAmplitudeRouter(
            self.geometry,
            settings.top_k,
            settings.router_pool_size,
            settings.router_temperature,
            settings.router_input_layernorm_enabled,
            settings.router_input_layernorm_eps,
            noise_std=getattr(settings, "router_noise_std", 0.0),
            gate_init_std=getattr(settings, "router_gate_init_std", 0.01),
        )
        self.amplitude_slm_weight_domain = settings.amplitude_slm_weight_domain
        self.amplitude_slm_input_normalization = settings.amplitude_slm_input_normalization
        self.amplitude_phase_relay = settings.amplitude_phase_relay
        self.register_buffer(
            "expert_canvas_indices",
            aperture_linear_indices(settings.canvas_size, self.geometry.expert_apertures),
            persistent=False,
        )
        prop_kwargs = {"wavelength_m": wavelength_m, "pixel_size_m": pixel_m, "grid_size": settings.canvas_size,
                       "k_space_constraint_enabled": settings.k_space_constraint_enabled,
                       "theta_max_deg": settings.theta_max_deg}
        self.expert_layers = nn.ModuleList([ExpertPhasePlane(self.geometry, settings) for _ in range(settings.expert_layers)])
        # All three configured hops are required to share one distance in this
        # experiment, so one immutable transfer function can be reused.
        self.propagator = AngularSpectrumPropagator(
            distance_m=settings.expert_interlayer_distance_m,
            **prop_kwargs,
        )
        self.interlayer_enabled = bool(settings.interlayer_enabled)
        self.interlayer_hard_route_mask = bool(settings.interlayer_hard_route_mask)
        self.interlayer_reapply_routing_weights = bool(settings.interlayer_reapply_routing_weights)
        self.interlayer_conversions = nn.ModuleList([
            SquareDetectionLayerNormReload(settings.canvas_size, self.geometry.expert_apertures,
                                           settings.interlayer_layernorm_eps, settings.interlayer_nonlinearity,
                                           settings.interlayer_per_expert_enabled,
                                           settings.interlayer_elementwise_affine,
                                           settings.interlayer_detector_integration_factor,
                                           settings.oeo_preserve_response_amplitude,
                                           settings.oeo_response_gain_min,
                                           settings.oeo_response_gain_max)
            for _ in range(settings.expert_layers)]) if self.interlayer_enabled else nn.ModuleList()
        self.global_phase = GlobalPhasePlane(self.geometry, settings)
        self.readout = FullPlaneReadout(self.geometry, settings)
        self.output_adapter = nn.Linear(settings.input_adapter_dim, hidden_size)
        self.last_input_fields: torch.Tensor | None = None; self.last_routing: dict[str, torch.Tensor] = {}
        self.last_expert_response: torch.Tensor | None = None
        self.last_response_gain: torch.Tensor | None = None
        self.last_amplitude_slm_canvas: torch.Tensor | None = None
        self.last_stage_fields: list[torch.Tensor] = []; self.last_detector_intensity: torch.Tensor | None = None
        self.last_detector_complex_field: torch.Tensor | None = None
        self.last_detector_readout: torch.Tensor | None = None
        # Graph-carrying final detector readout. Unlike last_detector_readout,
        # this tensor is never detached and is consumed by the retrieval
        # embedding head. Shape: [batch, detector_output_size, detector_output_size].
        self.current_detector_readout: torch.Tensor | None = None
        self.capture_intermediate_fields = False
        self.capture_sample_count = 1
        self.hardware_stage_reload_fields: dict[int, torch.Tensor] = {}
        self.hardware_final_detector_intensity: torch.Tensor | None = None

    def encode_groups(self, groups: list[torch.Tensor], *, injection: bool = False) -> torch.Tensor:
        if not groups:
            raise ValueError("At least one token group is required")
        counts = [len(group) for group in groups]
        for group, count in zip(groups, counts):
            if count > self.max_tokens:
                kind = "language sequence length" if group.ndim == 2 and self.hidden_size > 1024 else "visual token count"
                hint = "Shorten the prompt or lower processor_max_pixels" if kind.startswith("language") else "Lower processor_max_pixels"
                raise RuntimeError(f"{kind} {count} exceeds optical field rows={self.max_tokens}. {hint}; no crop or resize is allowed.")

        # A single GEMM is substantially faster than one tiny adapter launch per
        # sample.  The packed order is the same row-major order used below by
        # masked_scatter, so this is mathematically identical to the former loop.
        packed = torch.cat(groups, dim=0)
        projected = self.nonnegative(self.input_norm(self.input_adapter(packed.float())))
        if injection:
            changed = packed.float().abs().sum(-1, keepdim=True).gt(0)
            projected = projected * changed
        count_tensor = torch.tensor(counts, device=projected.device)
        valid_rows = torch.arange(self.geometry.expert_size, device=projected.device)[None, :] < count_tensor[:, None]
        field_mask = valid_rows.unsqueeze(-1).expand(-1, -1, self.geometry.expert_size)
        empty = projected.new_zeros((len(groups), self.geometry.expert_size, self.geometry.expert_size))
        return empty.masked_scatter(field_mask, projected.reshape(-1))

    def _normalize_amplitude_slm_input(self, input_fields: torch.Tensor) -> torch.Tensor:
        if self.amplitude_slm_input_normalization == "none":
            return input_fields
        if self.amplitude_slm_input_normalization == "per_sample_max":
            scale = input_fields.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
            return input_fields / scale
        raise RuntimeError(f"Unsupported amplitude-SLM input normalization {self.amplitude_slm_input_normalization!r}")

    def _amplitude_scales(self, routing: dict[str, torch.Tensor]) -> torch.Tensor:
        weights = routing["weights"].float().clamp_min(0.0)
        if self.amplitude_slm_weight_domain == "amplitude":
            return weights
        if self.amplitude_slm_weight_domain == "power":
            return weights.sqrt()
        raise RuntimeError(f"Unsupported routing weight domain {self.amplitude_slm_weight_domain!r}")

    def _direct_amplitude_load(self, input_fields: torch.Tensor,
                               routing: dict[str, torch.Tensor]) -> torch.Tensor:
        """Place weighted image/feature copies directly on the amplitude SLM.

        The amplitude plane is ideally relayed by a 4f system onto the
        co-planar phase SLM, so there is no propagation or prompt phase here.
        """
        amplitude = self._normalize_amplitude_slm_input(input_fields.float())
        scales = self._amplitude_scales(routing)
        batch = len(input_fields)
        values = (amplitude[:, None] * scales[:, :, None, None]).reshape(batch, -1)
        flat_indices = self.expert_canvas_indices.reshape(-1)
        canvas = amplitude.new_zeros((batch, self.geometry.canvas_size * self.geometry.canvas_size)).scatter(
            1,
            flat_indices.unsqueeze(0).expand(batch, -1),
            values,
        ).reshape(batch, self.geometry.canvas_size, self.geometry.canvas_size)
        routing["amplitude_scales"] = scales
        routing["amplitude_slm_canvas"] = canvas
        return torch.complex(canvas, torch.zeros_like(canvas))

    def begin(self, input_fields: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        capture_count = min(self.capture_sample_count, len(input_fields))
        self.last_input_fields = (
            input_fields[:capture_count].detach().cpu()
            if self.capture_intermediate_fields else None
        )
        routing = self.router(input_fields); self.last_routing = routing
        self.last_expert_response = None
        self.last_response_gain = None
        field = self._direct_amplitude_load(input_fields, routing)
        self.last_amplitude_slm_canvas = (
            field.real[:capture_count].detach().cpu()
            if self.capture_intermediate_fields else None
        )
        self.last_stage_fields = []
        self.last_detector_intensity = None
        self.last_detector_complex_field = None
        self.last_detector_readout = None
        self.current_detector_readout = None
        return field, routing

    def fanout(self, input_fields: torch.Tensor, routing: dict[str, torch.Tensor]) -> torch.Tensor:
        return self._direct_amplitude_load(input_fields, routing)

    def run_stage(self, index: int, field: torch.Tensor, routing: dict[str, torch.Tensor]) -> torch.Tensor:
        replay = self.hardware_stage_reload_fields.get(int(index))
        if replay is not None:
            if tuple(replay.shape) != tuple(field.shape):
                raise RuntimeError(
                    f"Hardware stage-{index} reload field shape {tuple(replay.shape)} "
                    f"does not match optical field {tuple(field.shape)}"
                )
            field = replay.to(device=field.device, dtype=torch.complex64)
        else:
            field = self.propagator(self.expert_layers[index](field))
            if self.interlayer_enabled:
                field = self.interlayer_conversions[index](
                    field,
                    selected_experts=routing["selected_mask"] if self.interlayer_hard_route_mask else None,
                    routing_weights=routing["weights"] if self.interlayer_reapply_routing_weights else None,
                    input_amplitude_scales=routing.get("amplitude_scales"))
                response = self.interlayer_conversions[index].last_expert_response
                if response is not None:
                    self.last_expert_response = response
                response_gain = self.interlayer_conversions[index].last_response_gain
                if response_gain is not None:
                    self.last_response_gain = response_gain
        if self.capture_intermediate_fields:
            capture_count = min(self.capture_sample_count, len(field))
            self.last_stage_fields.append(field[:capture_count].detach().cpu())
        return field

    def read_hidden(self, field: torch.Tensor, lengths: list[int], boundary_dtype: torch.dtype,
                    *, final: bool = False) -> torch.Tensor:
        measured_intensity = self.hardware_final_detector_intensity if final else None
        if final and measured_intensity is None:
            field = self.propagator(self.global_phase(field))
            if self.capture_intermediate_fields:
                capture_count = min(self.capture_sample_count, len(field))
                aperture = self.geometry.detector_aperture
                self.last_detector_complex_field = field[
                    :capture_count,
                    aperture.y0:aperture.y1,
                    aperture.x0:aperture.x1,
                ].detach().cpu().to(torch.complex64)
        if measured_intensity is not None:
            if measured_intensity.shape[0] != field.shape[0]:
                raise RuntimeError(
                    "Hardware final CCD batch does not match the optical field batch: "
                    f"{measured_intensity.shape[0]} != {field.shape[0]}"
                )
            readout, intensity = self.readout.forward_intensity(
                measured_intensity.to(field.device)
            )
        else:
            readout, intensity = self.readout(field)
        if final:
            if not torch.isfinite(readout).all():
                raise RuntimeError("Final detector readout contains NaN or Inf")
            if torch.any(readout < 0):
                raise RuntimeError(
                    "Final detector readout must be nonnegative before the retrieval readout"
                )
            self.current_detector_readout = readout
        if final and self.capture_intermediate_fields:
            capture_count = min(self.capture_sample_count, len(field))
            self.last_detector_intensity = intensity[:capture_count].detach().cpu()
            self.last_detector_readout = readout[:capture_count].detach().cpu()
        packed_readout = torch.cat([readout[row, :length] for row, length in enumerate(lengths)], dim=0)
        return self.output_adapter(packed_readout).to(boundary_dtype)

    def router_losses(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.last_routing["balance_loss"], self.last_routing["importance_loss"]

    def router_response_consistency_loss(self) -> torch.Tensor:
        """Align sparse routing weights with the experts' ungated CCD response.

        The response target is detached: it supervises the electronic router
        without letting the optical phases reduce this auxiliary loss by
        changing the target itself. Only the already selected top-k experts
        participate, so diffraction leakage into inactive apertures cannot
        become a routing target.
        """
        weights = self.last_routing.get("weights")
        selected = self.last_routing.get("selected_mask")
        response = self.last_expert_response
        if weights is None or selected is None or response is None:
            if weights is None:
                return next(self.parameters()).new_zeros(())
            return weights.new_zeros(())
        selected_float = selected.to(response.dtype)
        target = response * selected_float
        target = target / target.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
        difference = (weights.float() - target.float()).square() * selected_float
        return difference.sum(dim=-1).div(
            selected_float.sum(dim=-1).clamp_min(1.0)
        ).mean()

    def set_phase_dropout_active(self, active: bool) -> None:
        for layer in self.expert_layers: layer.set_phase_dropout_active(active)
        self.global_phase.set_phase_dropout_active(active)

    def set_intermediate_field_capture(self, enabled: bool, sample_count: int = 1) -> None:
        if sample_count <= 0:
            raise ValueError("Intermediate-field capture sample_count must be positive")
        self.capture_intermediate_fields = bool(enabled)
        self.capture_sample_count = int(sample_count)
        for conversion in self.interlayer_conversions:
            conversion.set_intermediate_capture(enabled, sample_count)

    def set_hardware_replay(
        self,
        *,
        stage_reload_fields: dict[int, torch.Tensor] | None = None,
        final_detector_intensity: torch.Tensor | None = None,
    ) -> None:
        """Substitute measured electronic reload/final CCD tensors for propagation.

        Reload fields are complex zero-phase amplitudes on the full numerical
        canvas. Final CCD intensity is the registered physical active ROI. The
        measured tensors remain differentiable only with respect to downstream
        electronics; this path is intended for eval/deployment replay.
        """
        self.hardware_stage_reload_fields = {
            int(index): value for index, value in (stage_reload_fields or {}).items()
        }
        self.hardware_final_detector_intensity = final_detector_intensity

    def clear_hardware_replay(self) -> None:
        self.hardware_stage_reload_fields = {}
        self.hardware_final_detector_intensity = None

    def parameter_breakdown(self) -> dict[str, Any]:
        expert_phase = sum(
            parameter.numel()
            for layer in self.expert_layers
            for parameter in layer.parameters()
        )
        global_phase = sum(parameter.numel() for parameter in self.global_phase.parameters())
        phase = expert_phase + global_phase
        router = sum(p.numel() for p in self.router.parameters())
        input_adapter = sum(parameter.numel() for parameter in self.input_adapter.parameters())
        adapter_norm = sum(parameter.numel() for parameter in self.input_norm.parameters())
        output_adapter = sum(parameter.numel() for parameter in self.output_adapter.parameters())
        adapters = input_adapter + adapter_norm + output_adapter
        total = sum(p.numel() for p in self.parameters())
        return {
            "expert_phase_parameters": expert_phase,
            "global_phase_parameters": global_phase,
            "optical_phase_parameters": phase,
            "router_parameters": router,
            "input_adapter_parameters": input_adapter,
            "adapter_norm_parameters": adapter_norm,
            "output_adapter_parameters": output_adapter,
            "adapter_parameters": adapters,
            "optoelectronic_interlayer_parameters": sum(
                parameter.numel() for module in self.interlayer_conversions for parameter in module.parameters()
            ),
            "total_parameters": total,
            "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "router_implementation": "electronic_amplitude_topk",
            "phase_prompt_parameters": 0,
        }


class VisionDeepStackHomogeneousMoE(nn.Module):
    def __init__(self, hidden_size: int, settings: Any) -> None:
        super().__init__(); self.core = HomogeneousMoEOpticalCore(hidden_size, settings.max_visual_tokens, settings)
        self.tap_stages = tuple(int(stage) for stage in settings.vision_tap_stages)
        self.last_token_counts: list[int] = []; self.tap_outputs: list[torch.Tensor] = []
        self.last_output: torch.Tensor | None = None; self.last_residual_base: torch.Tensor | None = None

    def compute(self, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor | None,
                residual_base: torch.Tensor | None = None) -> None:
        lengths = lengths_from_cu(hidden_states, cu_seqlens); self.last_token_counts = lengths
        if residual_base is not None and residual_base.shape != hidden_states.shape:
            raise RuntimeError(
                f"Vision residual shape {tuple(residual_base.shape)} does not match optical input "
                f"{tuple(hidden_states.shape)}"
            )
        self.last_residual_base = residual_base
        inputs = self.core.encode_groups(list(hidden_states.split(lengths))); field, routing = self.core.begin(inputs)
        taps: dict[int, torch.Tensor] = {}
        for index in range(len(self.core.expert_layers)):
            field = self.core.run_stage(index, field, routing)
            stage = index + 1
            if stage in self.tap_stages:
                delta = self.core.read_hidden(field, lengths, hidden_states.dtype)
                taps[stage] = delta if residual_base is None else residual_base + delta
        self.tap_outputs = [taps[stage] for stage in self.tap_stages]
        delta = self.core.read_hidden(field, lengths, hidden_states.dtype, final=True)
        self.last_output = delta if residual_base is None else residual_base + delta

    def output_for_slot(self, slot: int) -> torch.Tensor:
        if slot < len(self.tap_outputs): return self.tap_outputs[slot]
        if slot == len(self.tap_outputs) and self.last_output is not None: return self.last_output
        raise RuntimeError("Vision optical taps have not been computed for this batch")

    def router_losses(self): return self.core.router_losses()
    def set_phase_dropout_active(self, active: bool): self.core.set_phase_dropout_active(active)
    def set_intermediate_field_capture(self, enabled: bool, sample_count: int = 1):
        self.core.set_intermediate_field_capture(enabled, sample_count)
    def parameter_breakdown(self): return self.core.parameter_breakdown()


class LanguageDeepStackHomogeneousMoE(nn.Module):
    def __init__(self, hidden_size: int, settings: Any) -> None:
        super().__init__(); self.core = HomogeneousMoEOpticalCore(hidden_size, settings.max_language_tokens, settings)
        self.valid_mask: torch.Tensor | None = None; self.field: torch.Tensor | None = None
        self.routing: dict[str, torch.Tensor] | None = None; self.lengths: list[int] = []; self.positions: list[torch.Tensor] = []
        self.last_hidden: torch.Tensor | None = None; self.last_output: torch.Tensor | None = None
        self.residual_base: torch.Tensor | None = None
        self.deepstack_injection_count: int | None = None

    def set_attention_mask(self, mask: torch.Tensor) -> None:
        lengths = mask.long().sum(1).tolist()
        if max(lengths) > self.core.max_tokens:
            raise RuntimeError(f"language sequence length {max(lengths)} exceeds optical field rows={self.core.max_tokens}. "
                               "Shorten the prompt or lower processor_max_pixels; no crop or resize is allowed.")
        self.valid_mask = mask.bool(); self.lengths = [int(value) for value in lengths]
        self.positions = []

    def set_deepstack_injection_count(self, count: int) -> None:
        if count < 0:
            raise ValueError("DeepStack injection count must be non-negative")
        # In the one-stage baseline the sole selected DeepStack injection
        # occurs in a bypassed Qwen language slot before the optical language
        # stage. Therefore this count is placement metadata, not an
        # optical-stage bound.
        self.deepstack_injection_count = int(count)

    def _mask_on(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.valid_mask is None or self.valid_mask.shape != hidden.shape[:2]:
            raise RuntimeError("Call prepare_student_batch with the original 2-D attention mask before forward")
        if self.valid_mask.device != hidden.device:
            self.valid_mask = self.valid_mask.to(hidden.device, non_blocking=True)
        return self.valid_mask

    def _groups(self, hidden: torch.Tensor) -> list[torch.Tensor]:
        packed = hidden[self._mask_on(hidden)]
        return list(packed.split(self.lengths))

    def _scatter(self, packed: torch.Tensor, template: torch.Tensor) -> torch.Tensor:
        output = torch.zeros_like(template)
        output[self._mask_on(template)] = packed
        return output

    def forward_stage(self, stage: int, hidden_states: torch.Tensor,
                      optical_input: torch.Tensor | None = None,
                      residual_base: torch.Tensor | None = None) -> torch.Tensor:
        if stage == 0:
            branch_input = hidden_states if optical_input is None else optical_input
            if branch_input.shape != hidden_states.shape:
                raise RuntimeError(
                    f"Language optical input shape {tuple(branch_input.shape)} does not match hidden "
                    f"{tuple(hidden_states.shape)}"
                )
            if residual_base is not None and residual_base.shape != hidden_states.shape:
                raise RuntimeError(
                    f"Language residual shape {tuple(residual_base.shape)} does not match hidden "
                    f"{tuple(hidden_states.shape)}"
                )
            self.residual_base = residual_base
            fields = self.core.encode_groups(self._groups(branch_input)); self.field, self.routing = self.core.begin(fields)
        else:
            if self.field is None or self.routing is None or self.last_hidden is None:
                raise RuntimeError("Language optical stages must execute in order")
            # Native Qwen adds one DeepStack visual tensor after each of its
            # first N language layers. Replacement records N explicitly, which
            # avoids a count_nonzero GPU->CPU synchronization at every stage.
            has_injection = (stage <= self.deepstack_injection_count
                             if self.deepstack_injection_count is not None else None)
            if has_injection is not False:
                delta = hidden_states - self.last_hidden
                if self.residual_base is not None:
                    self.residual_base = self.residual_base + delta
                delta_fields = self.core.encode_groups(self._groups(delta), injection=True)
                if has_injection is True or torch.count_nonzero(delta_fields):
                    self.field = self.field + self.core.fanout(delta_fields, self.routing)
        assert self.field is not None and self.routing is not None
        self.field = self.core.run_stage(stage, self.field, self.routing)
        packed = self.core.read_hidden(self.field, self.lengths, hidden_states.dtype,
                                       final=stage == len(self.core.expert_layers) - 1)
        optical_delta = self._scatter(packed, hidden_states)
        output = optical_delta if self.residual_base is None else self.residual_base + optical_delta
        # Qwen's native _deepstack_process updates the returned tensor in-place.
        # Keep an explicit pre-injection copy so the next optical stage can
        # recover exactly the DeepStack delta that was added between layers.
        self.last_hidden = output.clone()
        if stage == len(self.core.expert_layers) - 1: self.last_output = output
        return output

    def router_losses(self): return self.core.router_losses()
    def set_phase_dropout_active(self, active: bool): self.core.set_phase_dropout_active(active)
    def set_intermediate_field_capture(self, enabled: bool, sample_count: int = 1):
        self.core.set_intermediate_field_capture(enabled, sample_count)
    def parameter_breakdown(self): return self.core.parameter_breakdown()

    def retrieval_detector_features(self) -> torch.Tensor:
        """Return the last valid token row of the nonnegative final CCD readout.

        The optical detector readout is [B, S_max, D] with D equal to the
        configured optical channel/readout width (224).  Each valid language
        token owns one row. Retrieval follows the official Qwen embedding
        pooling rule and selects the final non-padding token row.
        """
        readout = self.core.current_detector_readout
        if readout is None:
            raise RuntimeError(
                "Final language detector features are unavailable; run a complete student forward first"
            )
        if len(self.lengths) != readout.shape[0]:
            raise RuntimeError(
                f"Language token lengths ({len(self.lengths)}) do not match detector batch "
                f"({readout.shape[0]})"
            )
        rows = []
        for sample_index, length in enumerate(self.lengths):
            if length <= 0 or length > readout.shape[1]:
                raise RuntimeError(
                    f"Invalid language sequence length {length} for detector rows={readout.shape[1]}"
                )
            rows.append(readout[sample_index, length - 1])
        features = torch.stack(rows, dim=0)
        if features.ndim != 2 or features.shape[1] != self.core.readout.output_size:
            raise RuntimeError(f"Unexpected detector feature shape {tuple(features.shape)}")
        if not torch.isfinite(features).all() or torch.any(features < 0):
            raise RuntimeError("Retrieval detector features must be finite and nonnegative")
        return features
