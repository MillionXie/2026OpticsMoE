from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .geometry import MoEGeometry
from .physical import AngularSpectrumPropagator, PhaseLayer, SquareDetectionLayerNormReload
from .router import GlobalRouterPrompt


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

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        output = torch.zeros_like(field, dtype=torch.complex64)
        for aperture, phase in zip(self.geometry.expert_apertures, self.experts):
            crop = field[:, aperture.y0:aperture.y1, aperture.x0:aperture.x1]
            output[:, aperture.y0:aperture.y1, aperture.x0:aperture.x1] = phase(crop)
        return output

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
    def __init__(self, settings: Any) -> None:
        super().__init__(); size = settings.canvas_size // settings.detector_pool_kernel
        self.pool = nn.AvgPool2d(settings.detector_pool_kernel, settings.detector_pool_kernel)
        self.norm = nn.LayerNorm((size, size), eps=settings.detector_layernorm_eps, elementwise_affine=False)
        self.nonlinearity = settings.detector_nonlinearity

    def forward(self, field: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        intensity = field.to(torch.complex64).abs().square().float()
        pooled = self.pool(intensity.unsqueeze(1)).squeeze(1); normalized = self.norm(pooled)
        return (F.relu(normalized) if self.nonlinearity == "relu" else F.softplus(normalized)), intensity


class HomogeneousMoEOpticalCore(nn.Module):
    """Verified MoE9x5 physical core shared structurally, but not parametrically, by vision/language."""

    def __init__(self, hidden_size: int, max_tokens: int, settings: Any) -> None:
        super().__init__(); self.hidden_size = int(hidden_size); self.max_tokens = int(max_tokens)
        self.geometry = MoEGeometry(settings.canvas_size, settings.active_size, settings.expert_size,
                                    settings.expert_pitch, settings.num_experts); self.geometry.validate()
        self.input_adapter = nn.Linear(hidden_size, settings.input_adapter_dim)
        self.input_norm = nn.LayerNorm(settings.input_adapter_dim); self.nonnegative = nn.Softplus()
        wavelength_m = settings.wavelength_nm * 1e-9; pixel_m = settings.pixel_pitch_um * 1e-6
        self.prompt = GlobalRouterPrompt(self.geometry, wavelength_m, pixel_m, settings.prompt_focal_length_m,
                                         settings.top_k, settings.router_pool_size, settings.router_temperature,
                                         settings.router_input_layernorm_enabled, settings.router_input_layernorm_eps)
        prop_kwargs = {"wavelength_m": wavelength_m, "pixel_size_m": pixel_m, "grid_size": settings.canvas_size,
                       "k_space_constraint_enabled": settings.k_space_constraint_enabled,
                       "theta_max_deg": settings.theta_max_deg}
        self.expert_layers = nn.ModuleList([ExpertPhasePlane(self.geometry, settings) for _ in range(settings.expert_layers)])
        self.propagations = nn.ModuleList([
            AngularSpectrumPropagator(distance_m=(settings.expert_interlayer_distance_m
                                                   if index < settings.expert_layers - 1
                                                   else settings.last_expert_to_global_distance_m), **prop_kwargs)
            for index in range(settings.expert_layers)])
        self.interlayer_enabled = bool(settings.interlayer_enabled)
        self.interlayer_hard_route_mask = bool(settings.interlayer_hard_route_mask)
        self.interlayer_reapply_routing_weights = bool(settings.interlayer_reapply_routing_weights)
        self.interlayer_conversions = nn.ModuleList([
            SquareDetectionLayerNormReload(settings.canvas_size, self.geometry.expert_apertures,
                                           settings.interlayer_layernorm_eps, settings.interlayer_nonlinearity,
                                           settings.interlayer_per_expert_enabled,
                                           settings.interlayer_elementwise_affine)
            for _ in range(settings.expert_layers)]) if self.interlayer_enabled else nn.ModuleList()
        self.global_phase = GlobalPhasePlane(self.geometry, settings)
        self.to_detector = AngularSpectrumPropagator(distance_m=settings.global_to_detector_distance_m, **prop_kwargs)
        self.readout = FullPlaneReadout(settings); self.output_adapter = nn.Linear(settings.input_adapter_dim, hidden_size)
        self.last_input_fields: torch.Tensor | None = None; self.last_routing: dict[str, torch.Tensor] = {}
        self.last_stage_fields: list[torch.Tensor] = []; self.last_detector_intensity: torch.Tensor | None = None
        self.last_detector_readout: torch.Tensor | None = None

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

    def _fanout(self, input_fields: torch.Tensor, routing: dict[str, torch.Tensor]) -> torch.Tensor:
        canvas = input_fields.new_zeros((len(input_fields), self.geometry.canvas_size, self.geometry.canvas_size))
        aperture = self.geometry.input_aperture
        canvas[:, aperture.y0:aperture.y1, aperture.x0:aperture.x1] = input_fields
        canvas_spectrum = torch.fft.fft2(torch.flip(canvas.to(torch.complex64), (-2, -1)))
        # All DeepStack injections in one forward use the same sample-dependent
        # routing prompt. Cache only within that forward/autograd graph.
        transmission_spectrum = routing.get("_transmission_spectrum")
        if transmission_spectrum is None:
            transmission_spectrum = torch.fft.fft2(routing["transmission"].to(torch.complex64))
            routing["_transmission_spectrum"] = transmission_spectrum
        return torch.fft.fftshift(torch.fft.ifft2(canvas_spectrum * transmission_spectrum), (-2, -1))

    def begin(self, input_fields: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        self.last_input_fields = input_fields
        routing = self.prompt(input_fields); self.last_routing = routing
        field = self._fanout(input_fields, routing)
        self.last_stage_fields = []
        return field, routing

    def fanout(self, input_fields: torch.Tensor, routing: dict[str, torch.Tensor]) -> torch.Tensor:
        return self._fanout(input_fields, routing)

    def run_stage(self, index: int, field: torch.Tensor, routing: dict[str, torch.Tensor]) -> torch.Tensor:
        field = self.propagations[index](self.expert_layers[index](field))
        if self.interlayer_enabled:
            field = self.interlayer_conversions[index](
                field,
                selected_experts=routing["selected_mask"] if self.interlayer_hard_route_mask else None,
                routing_weights=routing["weights"] if self.interlayer_reapply_routing_weights else None)
        self.last_stage_fields.append(field)
        return field

    def read_hidden(self, field: torch.Tensor, lengths: list[int], boundary_dtype: torch.dtype,
                    *, final: bool = False) -> torch.Tensor:
        if final: field = self.to_detector(self.global_phase(field))
        readout, intensity = self.readout(field)
        if final:
            self.last_detector_intensity = intensity; self.last_detector_readout = readout
        packed_readout = torch.cat([readout[row, :length] for row, length in enumerate(lengths)], dim=0)
        return self.output_adapter(packed_readout).to(boundary_dtype)

    def router_losses(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.last_routing["balance_loss"], self.last_routing["importance_loss"]

    def set_phase_dropout_active(self, active: bool) -> None:
        for layer in self.expert_layers: layer.set_phase_dropout_active(active)
        self.global_phase.set_phase_dropout_active(active)

    def parameter_breakdown(self) -> dict[str, int]:
        phase = sum(p.numel() for name, p in self.named_parameters() if "raw_phase" in name)
        router = sum(p.numel() for p in self.prompt.router.parameters())
        adapters = sum(p.numel() for module in (self.input_adapter, self.input_norm, self.output_adapter)
                       for p in module.parameters())
        total = sum(p.numel() for p in self.parameters())
        return {"optical_phase_parameters": phase, "router_parameters": router,
                "adapter_parameters": adapters, "total_parameters": total,
                "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad)}


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
                f"Vision residual shape {tuple(residual_base.shape)} does not match optical input {tuple(hidden_states.shape)}"
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
        if count < 0 or count >= len(self.core.expert_layers):
            raise ValueError(f"DeepStack injection count must be in [0,{len(self.core.expert_layers) - 1}], got {count}")
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
                    f"Language optical input shape {tuple(branch_input.shape)} does not match hidden {tuple(hidden_states.shape)}"
                )
            if residual_base is not None and residual_base.shape != hidden_states.shape:
                raise RuntimeError(
                    f"Language residual shape {tuple(residual_base.shape)} does not match hidden {tuple(hidden_states.shape)}"
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
    def parameter_breakdown(self): return self.core.parameter_breakdown()
