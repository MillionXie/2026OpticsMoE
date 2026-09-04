from __future__ import annotations

import copy
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.optics.moe import (
    HomogeneousMoEOpticalCore,
)
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.optics.physical import (
    AngularSpectrumPropagator,
    PhaseLayer,
)
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.optics.router import (
    ElectronicAmplitudeRouter,
)


class FinalCCDScaleNormalizer(nn.Module):
    """Apply the same scale-only detector conditioning to every ablation."""

    def __init__(self, relative_clip: float, log_compression: float) -> None:
        super().__init__()
        self.relative_clip = float(relative_clip)
        self.log_compression = float(log_compression)

    def forward(self, intensity: torch.Tensor) -> torch.Tensor:
        value = intensity.float().clamp_min(0.0)
        if not torch.isfinite(value).all():
            raise RuntimeError("CCD intensity contains NaN or Inf")
        frame_mean = value.mean(dim=(-2, -1), keepdim=True).clamp_min(1.0e-6)
        relative = (value / frame_mean).clamp_max(self.relative_clip)
        return torch.log1p(self.log_compression * relative)


class PerExpertSigmoidReload(nn.Module):
    """Square-law CCD, per-expert LayerNorm, sigmoid and zero-phase reload.

    LayerNorm is required before sigmoid because raw optical intensity has no
    fixed absolute scale.  It is non-affine by default.  Routing weights are
    applied *after* normalization/sigmoid, so normalization cannot erase the
    sparse MoE allocation.
    """

    def __init__(self, core: HomogeneousMoEOpticalCore, settings: Any) -> None:
        super().__init__()
        self.canvas_size = int(core.geometry.canvas_size)
        self.num_experts = int(core.geometry.num_experts)
        self.expert_size = int(core.geometry.expert_size)
        self.eps = float(settings.multiplane_oeo_eps)
        self.gain = float(settings.multiplane_oeo_gain)
        self.affine = bool(settings.multiplane_oeo_affine)
        self.register_buffer(
            "aperture_indices", core.expert_canvas_indices.clone(), persistent=False
        )
        if self.affine:
            self.affine_scale = nn.Parameter(torch.ones(self.num_experts, 1, 1))
            self.affine_bias = nn.Parameter(torch.zeros(self.num_experts, 1, 1))
        else:
            self.register_parameter("affine_scale", None)
            self.register_parameter("affine_bias", None)
        self.last_intensity: torch.Tensor | None = None
        self.last_normalized: torch.Tensor | None = None
        self.last_amplitude: torch.Tensor | None = None
        self.last_input_power: torch.Tensor | None = None
        self.last_output_power: torch.Tensor | None = None

    def detect(self, field: torch.Tensor) -> torch.Tensor:
        batch = field.shape[0]
        crops = (
            field.to(torch.complex64)
            .abs()
            .square()
            .flatten(1)
            .index_select(1, self.aperture_indices.reshape(-1))
            .reshape(batch, self.num_experts, self.expert_size, self.expert_size)
            .float()
        )
        mean = crops.mean(dim=(-2, -1), keepdim=True)
        variance = (crops - mean).square().mean(dim=(-2, -1), keepdim=True)
        normalized = (crops - mean) * torch.rsqrt(variance + self.eps)
        if self.affine:
            normalized = (
                normalized * self.affine_scale.unsqueeze(0)
                + self.affine_bias.unsqueeze(0)
            )
        amplitude = torch.sigmoid(self.gain * normalized)
        self.last_intensity = crops.detach()
        self.last_normalized = normalized.detach()
        self.last_amplitude = amplitude.detach()
        self.last_input_power = crops.sum(dim=(-2, -1)).detach()
        return amplitude

    def reload(
        self,
        amplitude: torch.Tensor,
        routing: dict[str, torch.Tensor],
        amplitude_scales: torch.Tensor,
    ) -> torch.Tensor:
        batch = amplitude.shape[0]
        selected = routing["selected_mask"].to(amplitude.dtype)
        values = amplitude * selected[:, :, None, None]
        values = values * amplitude_scales[:, :, None, None]
        flat = amplitude.new_zeros(batch, self.canvas_size * self.canvas_size)
        flat = flat.scatter(
            1,
            self.aperture_indices.reshape(-1).unsqueeze(0).expand(batch, -1),
            values.reshape(batch, -1),
        )
        canvas = flat.reshape(batch, self.canvas_size, self.canvas_size)
        self.last_output_power = canvas.square().sum(dim=(-2, -1)).detach()
        return torch.complex(canvas, torch.zeros_like(canvas))

    @staticmethod
    def collapse_for_next_router(
        amplitude: torch.Tensor, routing: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        selected = routing["selected_mask"].to(amplitude.dtype)
        count = selected.sum(dim=-1, keepdim=True).clamp_min(1.0)
        # The old routing weight is deliberately removed before the next
        # router.  The next independent router owns the new sparse allocation.
        return (
            amplitude * selected[:, :, None, None]
        ).sum(dim=1) / count[:, :, None]


class FullApertureSigmoidReload(nn.Module):
    """Square-law CCD, full-aperture normalization and zero-phase reload.

    This is the conventional D2NN OEO boundary.  Unlike the MoE converter it
    has no expert crops, routing mask or routing weights: the complete 224x224
    field is detected together and reloaded as one nonnegative amplitude.
    """

    def __init__(self, settings: Any) -> None:
        super().__init__()
        self.eps = float(settings.multiplane_oeo_eps)
        self.gain = float(settings.multiplane_oeo_gain)
        self.affine = bool(settings.multiplane_oeo_affine)
        if self.affine:
            self.affine_scale = nn.Parameter(torch.ones(()))
            self.affine_bias = nn.Parameter(torch.zeros(()))
        else:
            self.register_parameter("affine_scale", None)
            self.register_parameter("affine_bias", None)
        self.last_intensity: torch.Tensor | None = None
        self.last_normalized: torch.Tensor | None = None
        self.last_amplitude: torch.Tensor | None = None
        self.last_input_power: torch.Tensor | None = None
        self.last_output_power: torch.Tensor | None = None

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        intensity = field.to(torch.complex64).abs().square().float()
        mean = intensity.mean(dim=(-2, -1), keepdim=True)
        variance = (intensity - mean).square().mean(dim=(-2, -1), keepdim=True)
        normalized = (intensity - mean) * torch.rsqrt(variance + self.eps)
        if self.affine:
            normalized = normalized * self.affine_scale + self.affine_bias
        amplitude = torch.sigmoid(self.gain * normalized)
        if not torch.isfinite(amplitude).all():
            raise RuntimeError("D2NN OEO reload amplitude contains NaN or Inf")
        self.last_intensity = intensity.detach()
        self.last_normalized = normalized.detach()
        self.last_amplitude = amplitude.detach()
        self.last_input_power = intensity.sum(dim=(-2, -1)).detach()
        self.last_output_power = amplitude.square().sum(dim=(-2, -1)).detach()
        return torch.complex(amplitude, torch.zeros_like(amplitude))


class MultiplaneMoECore(HomogeneousMoEOpticalCore):
    """Four MoE4 phase planes followed by one global phase and final CCD."""

    def __init__(self, hidden_size: int, max_tokens: int, settings: Any) -> None:
        optical_settings = copy.copy(settings)
        optical_settings.expert_layers = int(settings.multiplane_expert_planes)
        optical_settings.interlayer_enabled = False
        optical_settings.detector_output_size = int(settings.input_adapter_dim)
        optical_settings.expert_interlayer_distance_m = float(
            settings.multiplane_interplane_distance_m
        )
        super().__init__(hidden_size, max_tokens, optical_settings)
        self.variant = str(settings.multiplane_variant)
        self.dynamic_router = self.variant == "moe_oeo_dynamic_router"
        self.oeo_enabled = self.variant in {
            "moe_oeo_fixed_router",
            "moe_oeo_dynamic_router",
        }
        self.final_normalizer = FinalCCDScaleNormalizer(
            settings.language_optical_normalization_clip,
            settings.language_optical_log_compression,
        )
        self.detector_propagator = AngularSpectrumPropagator(
            wavelength_m=float(settings.wavelength_nm) * 1.0e-9,
            pixel_size_m=float(settings.pixel_pitch_um) * 1.0e-6,
            grid_size=int(settings.canvas_size),
            distance_m=float(settings.multiplane_detector_distance_m),
            k_space_constraint_enabled=bool(settings.k_space_constraint_enabled),
            theta_max_deg=float(settings.theta_max_deg),
        )
        self.oeo_layers = nn.ModuleList(
            [PerExpertSigmoidReload(self, settings) for _ in self.expert_layers]
            if self.oeo_enabled
            else []
        )
        self.additional_routers = nn.ModuleList()
        if self.dynamic_router:
            self.additional_routers.extend(
                [
                    ElectronicAmplitudeRouter(
                        self.geometry,
                        settings.top_k,
                        settings.router_pool_size,
                        settings.router_temperature,
                        settings.router_input_layernorm_enabled,
                        settings.router_input_layernorm_eps,
                        noise_std=getattr(settings, "router_noise_std", 0.0),
                        gate_init_std=getattr(settings, "router_gate_init_std", 0.01),
                    )
                    for _ in range(len(self.expert_layers) - 1)
                ]
            )
        self.stage_routings: list[dict[str, torch.Tensor]] = []
        self.stage_diagnostics: list[dict[str, torch.Tensor]] = []
        self._field: torch.Tensor | None = None
        self._routing: dict[str, torch.Tensor] | None = None
        self._lengths: list[int] = []
        self._padding_mask: torch.Tensor | None = None
        self.last_raw_detector_intensity: torch.Tensor | None = None
        self.reset_analysis_accumulators()

    def begin(self, input_fields: torch.Tensor):
        field, routing = super().begin(input_fields)
        self.stage_routings = [routing]
        self.stage_diagnostics = []
        return field, routing

    def _run_plane(
        self, index: int, field: torch.Tensor, routing: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        input_power = field.abs().square().sum(dim=(-2, -1))
        field = self.propagator(self.expert_layers[index](field))
        propagated_power = field.abs().square().sum(dim=(-2, -1))
        diagnostic = {
            "input_power": input_power.detach(),
            "propagated_power": propagated_power.detach(),
        }
        if self.oeo_enabled:
            converter = self.oeo_layers[index]
            amplitude = converter.detect(field)
            if self.dynamic_router and index < len(self.expert_layers) - 1:
                canonical = converter.collapse_for_next_router(amplitude, routing)
                routing = self.additional_routers[index](canonical)
                self.stage_routings.append(routing)
                field = self._direct_amplitude_load(canonical, routing)
            else:
                scales = self._amplitude_scales(routing)
                field = converter.reload(amplitude, routing, scales)
            diagnostic["reload_power"] = field.abs().square().sum(
                dim=(-2, -1)
            ).detach()
        self.stage_diagnostics.append(diagnostic)
        self.last_routing = routing
        return field, routing

    def _finalize(
        self, field: torch.Tensor, lengths: list[int], dtype: torch.dtype
    ) -> torch.Tensor:
        field = self.detector_propagator(self.global_phase(field))
        aperture = self.geometry.detector_aperture
        intensity = field[
            :, aperture.y0 : aperture.y1, aperture.x0 : aperture.x1
        ].abs().square().float()
        self.last_raw_detector_intensity = intensity.detach()
        conditioned = self.final_normalizer(intensity)
        readout, _ = self.readout.forward_intensity(conditioned)
        if not torch.isfinite(readout).all():
            raise RuntimeError("Final MoE CCD readout contains NaN or Inf")
        self.current_detector_readout = readout
        self.last_detector_intensity = intensity.detach()
        self.last_detector_readout = readout.detach()
        self._accumulate_analysis()
        packed = torch.cat(
            [readout[row, :length] for row, length in enumerate(lengths)], dim=0
        )
        return self.output_adapter(packed).to(dtype)

    def forward_groups(self, groups: list[torch.Tensor]) -> torch.Tensor:
        lengths = [len(group) for group in groups]
        field, routing = self.begin(self.encode_groups(groups))
        for index in range(len(self.expert_layers)):
            field, routing = self._run_plane(index, field, routing)
        return self._finalize(field, lengths, groups[0].dtype)

    def start_staged(self, groups: list[torch.Tensor], padding_mask: torch.Tensor) -> None:
        self._lengths = [len(group) for group in groups]
        self._field, self._routing = self.begin(self.encode_groups(groups))
        self._padding_mask = padding_mask

    def forward_staged_plane(self, stage: int, dtype: torch.dtype) -> torch.Tensor | None:
        if self._field is None or self._routing is None:
            raise RuntimeError("Language optical stack was not initialized")
        self._field, self._routing = self._run_plane(
            stage, self._field, self._routing
        )
        if stage != len(self.expert_layers) - 1:
            return None
        return self._finalize(self._field, self._lengths, dtype)

    def router_losses(self) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.stage_routings:
            zero = next(self.parameters()).new_zeros(())
            return zero, zero
        balance = torch.stack([item["balance_loss"] for item in self.stage_routings]).mean()
        importance = torch.stack(
            [item["importance_loss"] for item in self.stage_routings]
        ).mean()
        return balance, importance

    def all_router_parameters(self) -> list[nn.Parameter]:
        return [
            *self.router.parameters(),
            *(parameter for module in self.additional_routers for parameter in module.parameters()),
        ]

    def reset_analysis_accumulators(self) -> None:
        self.analysis_samples = 0
        self.analysis_router_selection: list[torch.Tensor] = []
        self.analysis_router_importance: list[torch.Tensor] = []
        self.analysis_router_weight: list[torch.Tensor] = []
        self.analysis_stage_input_power: list[torch.Tensor] = []
        self.analysis_stage_output_power: list[torch.Tensor] = []
        self.analysis_final_ccd_mean: torch.Tensor | None = None

    @torch.no_grad()
    def _accumulate_analysis(self) -> None:
        if self.current_detector_readout is None:
            return
        batch = int(self.current_detector_readout.shape[0])
        if not self.analysis_router_selection:
            self.analysis_router_selection = [
                routing["selected_mask"].new_zeros(routing["selected_mask"].shape[1], dtype=torch.float32)
                for routing in self.stage_routings
            ]
            self.analysis_router_importance = [
                routing["importance"].new_zeros(routing["importance"].shape, dtype=torch.float32)
                for routing in self.stage_routings
            ]
            self.analysis_router_weight = [
                routing["weights"].new_zeros(routing["weights"].shape[1], dtype=torch.float32)
                for routing in self.stage_routings
            ]
            self.analysis_stage_input_power = [
                values["input_power"].new_zeros((), dtype=torch.float32)
                for values in self.stage_diagnostics
            ]
            self.analysis_stage_output_power = [
                values["propagated_power"].new_zeros((), dtype=torch.float32)
                for values in self.stage_diagnostics
            ]
            self.analysis_final_ccd_mean = self.current_detector_readout.new_zeros((), dtype=torch.float32)
        for index, routing in enumerate(self.stage_routings):
            self.analysis_router_selection[index].add_(
                routing["selected_mask"].float().sum(dim=0)
            )
            self.analysis_router_importance[index].add_(
                routing["importance"].float(), alpha=batch
            )
            self.analysis_router_weight[index].add_(
                routing["weights"].float().sum(dim=0)
            )
        for index, values in enumerate(self.stage_diagnostics):
            self.analysis_stage_input_power[index].add_(
                values["input_power"].float().sum()
            )
            self.analysis_stage_output_power[index].add_(
                values.get("reload_power", values["propagated_power"]).float().sum()
            )
        assert self.analysis_final_ccd_mean is not None
        self.analysis_final_ccd_mean.add_(
            self.last_raw_detector_intensity.float().mean(dim=(-2, -1)).sum()
        )
        self.analysis_samples += batch

    @torch.no_grad()
    def analysis_summary(self) -> dict[str, Any]:
        denominator = max(1, self.analysis_samples)
        return {
            "samples": self.analysis_samples,
            "routers": [
                {
                    "stage": index + 1,
                    "selection_rate": (selection / denominator).cpu().tolist(),
                    "mean_probability_importance": (self.analysis_router_importance[index] / denominator).cpu().tolist(),
                    "mean_sparse_weight": (self.analysis_router_weight[index] / denominator).cpu().tolist(),
                }
                for index, selection in enumerate(self.analysis_router_selection)
            ],
            "optical_stages": [
                {
                    "stage": index + 1,
                    "mean_input_power": float(self.analysis_stage_input_power[index].cpu() / denominator),
                    "mean_output_or_reload_power": float(self.analysis_stage_output_power[index].cpu() / denominator),
                }
                for index in range(len(self.analysis_stage_input_power))
            ],
            "final_ccd_mean": (
                None
                if self.analysis_final_ccd_mean is None
                else float(self.analysis_final_ccd_mean.cpu() / denominator)
            ),
        }


class D2NNFivePlaneCore(nn.Module):
    """Five 224x224 D2NN planes, optionally separated by sigmoid OEO."""

    def __init__(self, hidden_size: int, max_tokens: int, settings: Any) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.max_tokens = int(max_tokens)
        self.size = int(settings.expert_size)
        self.variant = str(settings.multiplane_variant)
        self.oeo_enabled = self.variant == "d2nn_oeo_sigmoid"
        self.input_adapter = nn.Linear(self.hidden_size, self.size)
        self.input_norm = nn.LayerNorm(self.size)
        self.nonnegative = nn.Softplus()
        self.expert_layers = nn.ModuleList(
            [
                PhaseLayer(
                    self.size,
                    settings.phase_parameterization,
                    settings.phase_init,
                    settings.phase_init_std,
                    settings.phase_dropout_mode,
                    settings.phase_dropout_p,
                    settings.phase_dropout_block_size,
                    settings.phase_dropout_batch_shared,
                )
                for _ in range(settings.multiplane_d2nn_planes)
            ]
        )
        propagation = dict(
            wavelength_m=float(settings.wavelength_nm) * 1.0e-9,
            pixel_size_m=float(settings.pixel_pitch_um) * 1.0e-6,
            grid_size=self.size,
            k_space_constraint_enabled=bool(settings.k_space_constraint_enabled),
            theta_max_deg=float(settings.theta_max_deg),
        )
        self.propagator = AngularSpectrumPropagator(
            distance_m=float(settings.multiplane_interplane_distance_m),
            **propagation,
        )
        self.detector_propagator = AngularSpectrumPropagator(
            distance_m=float(settings.multiplane_detector_distance_m),
            **propagation,
        )
        self.detector_norm = nn.LayerNorm(
            self.size,
            eps=settings.detector_layernorm_eps,
            elementwise_affine=settings.detector_layernorm_affine,
        )
        self.detector_nonlinearity = str(settings.detector_nonlinearity)
        self.output_adapter = nn.Linear(self.size, self.hidden_size)
        self.final_normalizer = FinalCCDScaleNormalizer(
            settings.language_optical_normalization_clip,
            settings.language_optical_log_compression,
        )
        # Four conversion boundaries sit between the five phase planes.  The
        # fifth propagation terminates at the common final task CCD and is not
        # needlessly detected and re-encoded a second time.
        self.oeo_layers = nn.ModuleList(
            [
                FullApertureSigmoidReload(settings)
                for _ in range(max(0, len(self.expert_layers) - 1))
            ]
            if self.oeo_enabled
            else []
        )
        self.current_detector_readout: torch.Tensor | None = None
        self.last_raw_detector_intensity: torch.Tensor | None = None
        self.last_routing: dict[str, torch.Tensor] = {}
        self.stage_routings: list[dict[str, torch.Tensor]] = []
        self.stage_diagnostics: list[dict[str, torch.Tensor]] = []
        self._field: torch.Tensor | None = None
        self._lengths: list[int] = []
        self.reset_analysis_accumulators()

    def encode_groups(self, groups: list[torch.Tensor]) -> torch.Tensor:
        counts = [len(group) for group in groups]
        if not groups or any(count <= 0 or count > self.size for count in counts):
            raise RuntimeError(f"Invalid D2NN token counts {counts}")
        packed = torch.cat(groups, dim=0)
        projected = self.nonnegative(self.input_norm(self.input_adapter(packed.float())))
        count_tensor = torch.tensor(counts, device=projected.device)
        valid = torch.arange(self.size, device=projected.device)[None] < count_tensor[:, None]
        mask = valid.unsqueeze(-1).expand(-1, -1, self.size)
        fields = projected.new_zeros(len(groups), self.size, self.size)
        return fields.masked_scatter(mask, projected.reshape(-1))

    def _identity_routing(self, batch: int, device: torch.device) -> dict[str, torch.Tensor]:
        zero = self.input_adapter.weight.new_zeros(())
        return {
            "weights": torch.ones(batch, 1, device=device),
            "selected_mask": torch.ones(batch, 1, dtype=torch.bool, device=device),
            "selected_indices": torch.zeros(batch, 1, dtype=torch.long, device=device),
            "importance": torch.ones(1, device=device),
            "normalized_entropy": zero,
            "balance_loss": zero,
            "importance_loss": zero,
        }

    def _begin(self, groups: list[torch.Tensor]) -> torch.Tensor:
        amplitude = self.encode_groups(groups)
        routing = self._identity_routing(len(groups), amplitude.device)
        self.last_routing = routing
        self.stage_routings = [routing]
        self.stage_diagnostics = []
        return torch.complex(amplitude, torch.zeros_like(amplitude))

    def _run_plane(self, stage: int, field: torch.Tensor) -> torch.Tensor:
        input_power = field.abs().square().sum(dim=(-2, -1))
        field = self.expert_layers[stage](field)
        propagator = (
            self.detector_propagator
            if stage == len(self.expert_layers) - 1
            else self.propagator
        )
        field = propagator(field)
        diagnostic = {
            "input_power": input_power.detach(),
            "propagated_power": field.abs().square().sum(
                dim=(-2, -1)
            ).detach(),
        }
        if self.oeo_enabled and stage < len(self.expert_layers) - 1:
            field = self.oeo_layers[stage](field)
            diagnostic["reload_power"] = field.abs().square().sum(
                dim=(-2, -1)
            ).detach()
        self.stage_diagnostics.append(diagnostic)
        return field

    def _finalize(self, field: torch.Tensor, lengths: list[int], dtype: torch.dtype) -> torch.Tensor:
        intensity = field.abs().square().float()
        self.last_raw_detector_intensity = intensity.detach()
        conditioned = self.final_normalizer(intensity)
        normalized = self.detector_norm(conditioned)
        readout = (
            F.relu(normalized)
            if self.detector_nonlinearity == "relu"
            else F.softplus(normalized)
        )
        self.current_detector_readout = readout
        self._accumulate_analysis()
        packed = torch.cat(
            [readout[row, :length] for row, length in enumerate(lengths)], dim=0
        )
        return self.output_adapter(packed).to(dtype)

    def forward_groups(self, groups: list[torch.Tensor]) -> torch.Tensor:
        field = self._begin(groups)
        for stage in range(len(self.expert_layers)):
            field = self._run_plane(stage, field)
        return self._finalize(field, [len(group) for group in groups], groups[0].dtype)

    def start_staged(self, groups: list[torch.Tensor], _padding_mask: torch.Tensor) -> None:
        self._lengths = [len(group) for group in groups]
        self._field = self._begin(groups)

    def forward_staged_plane(self, stage: int, dtype: torch.dtype) -> torch.Tensor | None:
        if self._field is None:
            raise RuntimeError("Language D2NN stack was not initialized")
        self._field = self._run_plane(stage, self._field)
        if stage != len(self.expert_layers) - 1:
            return None
        return self._finalize(self._field, self._lengths, dtype)

    def router_losses(self) -> tuple[torch.Tensor, torch.Tensor]:
        zero = self.input_adapter.weight.new_zeros(())
        return zero, zero

    def router_response_consistency_loss(self) -> torch.Tensor:
        return self.input_adapter.weight.new_zeros(())

    def set_phase_dropout_active(self, active: bool) -> None:
        for phase in self.expert_layers:
            phase.set_dropout_active(active)

    def set_intermediate_field_capture(self, _enabled: bool, _sample_count: int = 1) -> None:
        return None

    def all_router_parameters(self) -> list[nn.Parameter]:
        return []

    def reset_analysis_accumulators(self) -> None:
        self.analysis_samples = 0
        self.analysis_stage_input_power: list[torch.Tensor] = []
        self.analysis_stage_output_power: list[torch.Tensor] = []
        self.analysis_final_ccd_mean: torch.Tensor | None = None

    @torch.no_grad()
    def _accumulate_analysis(self) -> None:
        if self.current_detector_readout is None:
            return
        batch = int(self.current_detector_readout.shape[0])
        if not self.analysis_stage_input_power:
            self.analysis_stage_input_power = [
                values["input_power"].new_zeros((), dtype=torch.float32)
                for values in self.stage_diagnostics
            ]
            self.analysis_stage_output_power = [
                values.get("reload_power", values["propagated_power"]).new_zeros(
                    (), dtype=torch.float32
                )
                for values in self.stage_diagnostics
            ]
            self.analysis_final_ccd_mean = self.current_detector_readout.new_zeros((), dtype=torch.float32)
        for index, values in enumerate(self.stage_diagnostics):
            self.analysis_stage_input_power[index].add_(values["input_power"].float().sum())
            self.analysis_stage_output_power[index].add_(
                values.get("reload_power", values["propagated_power"]).float().sum()
            )
        assert self.analysis_final_ccd_mean is not None
        self.analysis_final_ccd_mean.add_(
            self.last_raw_detector_intensity.float().mean(dim=(-2, -1)).sum()
        )
        self.analysis_samples += batch

    @torch.no_grad()
    def analysis_summary(self) -> dict[str, Any]:
        denominator = max(1, self.analysis_samples)
        return {
            "samples": self.analysis_samples,
            "routers": [],
            "optical_stages": [
                {
                    "stage": index + 1,
                    "mean_input_power": float(self.analysis_stage_input_power[index].cpu() / denominator),
                    "mean_output_or_reload_power": float(self.analysis_stage_output_power[index].cpu() / denominator),
                }
                for index in range(len(self.analysis_stage_input_power))
            ],
            "final_ccd_mean": (
                None
                if self.analysis_final_ccd_mean is None
                else float(self.analysis_final_ccd_mean.cpu() / denominator)
            ),
        }


def build_physical_core(hidden_size: int, max_tokens: int, settings: Any) -> nn.Module:
    if settings.multiplane_variant.startswith("d2nn"):
        return D2NNFivePlaneCore(hidden_size, max_tokens, settings)
    return MultiplaneMoECore(hidden_size, max_tokens, settings)


__all__ = [
    "D2NNFivePlaneCore",
    "FullApertureSigmoidReload",
    "MultiplaneMoECore",
    "PerExpertSigmoidReload",
    "build_physical_core",
]
