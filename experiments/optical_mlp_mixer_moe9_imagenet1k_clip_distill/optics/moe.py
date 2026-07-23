from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .geometry import MoEGeometry
from .physical import (
    AngularSpectrumPropagator,
    ExpertSquareDetectionReload,
    PhaseLayer,
    aperture_linear_indices,
)
from .router import ElectronicTopKRouter


class ExpertPhasePlane(nn.Module):
    def __init__(self, geometry: MoEGeometry, settings: Any) -> None:
        super().__init__()
        self.geometry = geometry
        phase = settings.optics
        dropout = settings.phase_dropout
        self.experts = nn.ModuleList(
            [
                PhaseLayer(
                    geometry.expert_size,
                    parameterization=phase.phase_parameterization,
                    init=phase.phase_init,
                    init_std=phase.phase_init_std,
                    dropout_mode=dropout.mode,
                    dropout_p=dropout.p,
                    dropout_block_size=dropout.block_size,
                    dropout_batch_shared=dropout.batch_shared,
                )
                for _ in range(geometry.num_experts)
            ]
        )
        self.register_buffer(
            "aperture_indices",
            aperture_linear_indices(geometry.canvas_size, geometry.expert_apertures),
            persistent=False,
        )

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        batch = field.shape[0]
        flat_indices = self.aperture_indices.reshape(-1)
        crops = field.to(torch.complex64).flatten(1).index_select(1, flat_indices)
        crops = crops.reshape(
            batch,
            self.geometry.num_experts,
            self.geometry.expert_size,
            self.geometry.expert_size,
        )
        modulated = torch.stack(
            [phase(crops[:, index]) for index, phase in enumerate(self.experts)],
            dim=1,
        )
        output = torch.zeros(
            batch,
            self.geometry.canvas_size * self.geometry.canvas_size,
            dtype=torch.complex64,
            device=field.device,
        )
        return output.scatter(
            1,
            flat_indices.unsqueeze(0).expand(batch, -1),
            modulated.reshape(batch, -1),
        ).reshape(batch, self.geometry.canvas_size, self.geometry.canvas_size)

    def set_phase_dropout_active(self, active: bool) -> None:
        for expert in self.experts:
            expert.set_dropout_active(active)


class GlobalPhasePlane(nn.Module):
    def __init__(self, geometry: MoEGeometry, settings: Any) -> None:
        super().__init__()
        self.geometry = geometry
        phase = settings.optics
        dropout = settings.phase_dropout
        self.phase = PhaseLayer(
            geometry.active_size,
            parameterization=phase.phase_parameterization,
            init=phase.phase_init,
            init_std=phase.phase_init_std,
            dropout_mode=dropout.mode,
            dropout_p=dropout.p,
            dropout_block_size=dropout.block_size,
            dropout_batch_shared=dropout.batch_shared,
        )

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        aperture = self.geometry.active_aperture
        output = field.to(torch.complex64).clone()
        output[:, aperture.y0 : aperture.y1, aperture.x0 : aperture.x1] = self.phase(
            field[:, aperture.y0 : aperture.y1, aperture.x0 : aperture.x1]
        )
        return output

    def set_phase_dropout_active(self, active: bool) -> None:
        self.phase.set_dropout_active(active)


class OpticalMoECore(nn.Module):
    """Five phase stages and one shared global phase for a folded Mixer block."""

    def __init__(self, settings: Any) -> None:
        super().__init__()
        geometry_cfg = settings.geometry
        self.geometry = MoEGeometry(
            canvas_size=geometry_cfg.canvas_size,
            active_size=geometry_cfg.active_size,
            expert_size=geometry_cfg.expert_size,
            expert_pitch=geometry_cfg.expert_pitch,
            num_experts=geometry_cfg.num_experts,
        )
        self.geometry.validate()
        router_cfg = settings.router
        self.router = ElectronicTopKRouter(
            num_experts=self.geometry.num_experts,
            top_k=router_cfg.top_k,
            pool_size=router_cfg.pool_size,
            temperature=router_cfg.temperature,
            input_layernorm_enabled=router_cfg.input_layernorm_enabled,
            input_layernorm_eps=router_cfg.input_layernorm_eps,
        )
        self.amplitude_weight_domain = router_cfg.amplitude_weight_domain
        self.amplitude_input_normalization = router_cfg.amplitude_input_normalization
        self.register_buffer(
            "expert_canvas_indices",
            aperture_linear_indices(self.geometry.canvas_size, self.geometry.expert_apertures),
            persistent=False,
        )
        self.phase_planes = nn.ModuleList(
            [ExpertPhasePlane(self.geometry, settings) for _ in range(settings.model.phase_layers_per_block)]
        )
        optics = settings.optics
        propagation_kwargs = {
            "wavelength_m": optics.wavelength_nm * 1e-9,
            "pixel_size_m": optics.pixel_pitch_um * 1e-6,
            "grid_size": self.geometry.canvas_size,
            "k_space_constraint_enabled": optics.k_space_constraint_enabled,
            "theta_max_deg": optics.theta_max_deg,
        }
        self.stage_propagations = nn.ModuleList(
            [
                AngularSpectrumPropagator(
                    distance_m=optics.inter_layer_distance_m,
                    **propagation_kwargs,
                )
                for _ in self.phase_planes
            ]
        )
        oeo = settings.oeo
        self.oeo_enabled = bool(oeo.enabled)
        self.hard_route_mask = bool(oeo.hard_route_mask)
        self.reapply_routing_weights = bool(oeo.reapply_routing_weights)
        self.reloads = nn.ModuleList(
            [
                ExpertSquareDetectionReload(
                    self.geometry.canvas_size,
                    self.geometry.expert_apertures,
                    eps=oeo.layernorm_eps,
                    nonlinearity=oeo.nonlinearity,
                    per_expert_enabled=oeo.per_expert_enabled,
                    elementwise_affine=oeo.elementwise_affine,
                )
                for _ in self.phase_planes
            ]
        )
        self.to_global = AngularSpectrumPropagator(
            distance_m=optics.readout_to_global_distance_m,
            **propagation_kwargs,
        )
        self.global_phase = GlobalPhasePlane(self.geometry, settings)
        self.to_detector = AngularSpectrumPropagator(
            distance_m=optics.global_to_detector_distance_m,
            **propagation_kwargs,
        )
        self.detector_layernorm_eps = settings.model.detector_layernorm_eps
        self.capture_debug = False
        self.last_debug: dict[str, Any] = {}
        self.last_routing: dict[str, torch.Tensor] = {}

    def _amplitude_scales(self, weights: torch.Tensor) -> torch.Tensor:
        weights = weights.float().clamp_min(0)
        if self.amplitude_weight_domain == "amplitude":
            return weights
        if self.amplitude_weight_domain == "power":
            return weights.sqrt()
        raise RuntimeError(f"Unsupported routing weight domain {self.amplitude_weight_domain!r}")

    def _normalize_input(self, fields: torch.Tensor) -> torch.Tensor:
        if self.amplitude_input_normalization == "none":
            return fields
        if self.amplitude_input_normalization == "per_sample_max":
            return fields / fields.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
        raise RuntimeError(f"Unsupported input normalization {self.amplitude_input_normalization!r}")

    def route(self, fields: torch.Tensor) -> dict[str, torch.Tensor]:
        routing = self.router(fields)
        self.last_routing = routing
        return routing

    def direct_amplitude_load(
        self, fields: torch.Tensor, routing: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Directly address selected amplitude-SLM expert regions; no prompt phase."""

        amplitude = self._normalize_input(fields.float())
        scales = self._amplitude_scales(routing["weights"])
        values = (amplitude[:, None] * scales[:, :, None, None]).reshape(len(fields), -1)
        flat_indices = self.expert_canvas_indices.reshape(-1)
        canvas = amplitude.new_zeros(
            len(fields), self.geometry.canvas_size * self.geometry.canvas_size
        )
        canvas = canvas.scatter(
            1,
            flat_indices.unsqueeze(0).expand(len(fields), -1),
            values,
        ).reshape(len(fields), self.geometry.canvas_size, self.geometry.canvas_size)
        if self.capture_debug:
            self.last_debug.setdefault("amplitude_loads", []).append(canvas.detach().cpu())
        return torch.complex(canvas, torch.zeros_like(canvas))

    def begin(self, fields: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        self.last_debug = {}
        if self.capture_debug:
            self.last_debug["token_optical_input"] = fields.detach().cpu()
        routing = self.route(fields)
        if self.capture_debug:
            self.last_debug["routing_weights"] = routing["weights"].detach().cpu()
            self.last_debug["selected_indices"] = (
                routing["selected_indices"].detach().cpu()
            )
        return self.direct_amplitude_load(fields, routing), routing

    def reload_with_same_routing(
        self, fields: torch.Tensor, routing: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        if self.capture_debug:
            self.last_debug["channel_optical_input"] = fields.detach().cpu()
        return self.direct_amplitude_load(fields, routing)

    def run_stage(
        self, stage_index: int, field: torch.Tensor, routing: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        field = self.stage_propagations[stage_index](self.phase_planes[stage_index](field))
        detector_intensity = field.abs().square()
        if self.oeo_enabled:
            field = self.reloads[stage_index](
                field,
                selected_experts=(
                    routing["selected_mask"] if self.hard_route_mask else None
                ),
                routing_weights=(
                    routing["weights"] if self.reapply_routing_weights else None
                ),
            )
        if self.capture_debug:
            self.last_debug.setdefault("stage_detector_intensity", []).append(
                detector_intensity.detach().cpu()
            )
            self.last_debug.setdefault("stage_reloaded_amplitude", []).append(
                field.real.detach().cpu()
            )
        return field

    def global_readout(self, field: torch.Tensor, name: str) -> torch.Tensor:
        detector_field = self.to_detector(self.global_phase(self.to_global(field)))
        intensity = detector_field.abs().square().float()
        aperture = self.geometry.detector_aperture
        roi = intensity[
            :, aperture.y0 : aperture.y1, aperture.x0 : aperture.x1
        ]
        normalized = F.layer_norm(
            roi,
            roi.shape[-2:],
            weight=None,
            bias=None,
            eps=self.detector_layernorm_eps,
        )
        if self.capture_debug:
            self.last_debug[f"{name}_detector_intensity_full"] = intensity.detach().cpu()
            self.last_debug[f"{name}_detector_roi"] = roi.detach().cpu()
            self.last_debug[f"{name}_detector_readout"] = normalized.detach().cpu()
        return normalized

    def set_phase_dropout_active(self, active: bool) -> None:
        for plane in self.phase_planes:
            plane.set_phase_dropout_active(active)
        self.global_phase.set_phase_dropout_active(active)

    def set_debug_capture(self, enabled: bool) -> None:
        self.capture_debug = bool(enabled)

    def parameter_breakdown(self) -> dict[str, int]:
        expert_phase = sum(
            parameter.numel()
            for name, parameter in self.named_parameters()
            if "phase_planes" in name and name.endswith("raw_phase")
        )
        global_phase = sum(
            parameter.numel()
            for name, parameter in self.named_parameters()
            if "global_phase" in name and name.endswith("raw_phase")
        )
        router = sum(parameter.numel() for parameter in self.router.parameters())
        oeo = sum(parameter.numel() for parameter in self.reloads.parameters())
        return {
            "expert_phase_parameters": expert_phase,
            "global_phase_parameters": global_phase,
            "optical_phase_parameters": expert_phase + global_phase,
            "router_parameters": router,
            "oeo_parameters": oeo,
            "total_parameters": sum(parameter.numel() for parameter in self.parameters()),
            "trainable_parameters": sum(
                parameter.numel() for parameter in self.parameters() if parameter.requires_grad
            ),
        }
