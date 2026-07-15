from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .geometry import MoEGeometry
from .physical import AngularSpectrumPropagator, PhaseLayer, SquareDetectionLayerNormReload
from .router import GlobalRouterPrompt


def lengths_from_cu(hidden: torch.Tensor, cu_seqlens: torch.Tensor | None) -> list[int]:
    if hidden.ndim != 2:
        raise ValueError(f"Packed vision hidden must be [sum(T),D], got {tuple(hidden.shape)}")
    if cu_seqlens is None:
        raise RuntimeError("Packed vision hidden requires cu_seqlens; batches cannot share one optical field")
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
            PhaseLayer(geometry.expert_size, settings.phase_parameterization, settings.phase_init, settings.phase_init_std)
            for _ in range(geometry.num_experts)
        ])

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        output = torch.zeros_like(field, dtype=torch.complex64)
        for aperture, phase in zip(self.geometry.expert_apertures, self.experts):
            crop = field[:, aperture.y0:aperture.y1, aperture.x0:aperture.x1]
            output[:, aperture.y0:aperture.y1, aperture.x0:aperture.x1] = phase(crop)
        return output


class GlobalPhasePlane(nn.Module):
    def __init__(self, geometry: MoEGeometry, settings: Any) -> None:
        super().__init__()
        self.geometry = geometry
        self.phase = PhaseLayer(geometry.active_size, settings.phase_parameterization, settings.phase_init, settings.phase_init_std)

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        output = field.to(torch.complex64).clone()
        aperture = self.geometry.active_aperture
        crop = field[:, aperture.y0:aperture.y1, aperture.x0:aperture.x1]
        output[:, aperture.y0:aperture.y1, aperture.x0:aperture.x1] = self.phase(crop)
        return output


class FullPlaneDetectorReadout(nn.Module):
    """480x480 square-law detector -> fixed average pool -> non-affine LN -> activation."""

    def __init__(self, settings: Any) -> None:
        super().__init__()
        size = settings.canvas_size // settings.detector_pool_kernel
        self.pool = nn.AvgPool2d(settings.detector_pool_kernel, settings.detector_pool_kernel)
        self.norm = nn.LayerNorm((size, size), eps=settings.detector_layernorm_eps, elementwise_affine=False)
        self.nonlinearity = settings.detector_nonlinearity

    def forward(self, detector_field: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        intensity = detector_field.to(torch.complex64).abs().square().float()
        pooled = self.pool(intensity.unsqueeze(1)).squeeze(1)
        normalized = self.norm(pooled)
        readout = F.relu(normalized) if self.nonlinearity == "relu" else F.softplus(normalized)
        return readout, intensity


class VisionHomogeneousMoESurrogate(nn.Module):
    """Replace all Qwen vision blocks with the verified homogeneous top-3 optical MoE."""

    def __init__(self, hidden_size: int, settings: Any) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.max_visual_tokens = int(settings.max_visual_tokens)
        self.geometry = MoEGeometry(settings.canvas_size, settings.active_size, settings.expert_size,
                                    settings.expert_pitch, settings.num_experts)
        self.geometry.validate()
        self.input_adapter = nn.Linear(hidden_size, settings.input_adapter_dim)
        self.input_norm = nn.LayerNorm(settings.input_adapter_dim)
        self.nonnegative = nn.Softplus()
        wavelength_m = settings.wavelength_nm * 1e-9
        pixel_m = settings.pixel_pitch_um * 1e-6
        self.prompt = GlobalRouterPrompt(self.geometry, wavelength_m, pixel_m, settings.prompt_focal_length_m,
                                         settings.top_k, settings.router_pool_size, settings.router_temperature)
        prop_kwargs = {"wavelength_m": wavelength_m, "pixel_size_m": pixel_m, "grid_size": settings.canvas_size,
                       "k_space_constraint_enabled": settings.k_space_constraint_enabled, "theta_max_deg": settings.theta_max_deg}
        self.expert_layers = nn.ModuleList([ExpertPhasePlane(self.geometry, settings) for _ in range(settings.expert_layers)])
        self.propagations = nn.ModuleList([
            AngularSpectrumPropagator(distance_m=(settings.expert_interlayer_distance_m if index < settings.expert_layers - 1 else settings.last_expert_to_global_distance_m), **prop_kwargs)
            for index in range(settings.expert_layers)
        ])
        self.interlayer_conversions = nn.ModuleList([
            SquareDetectionLayerNormReload(self.geometry.expert_apertures, settings.interlayer_layernorm_eps, settings.interlayer_nonlinearity)
            for _ in range(settings.expert_layers)
        ])
        self.global_phase = GlobalPhasePlane(self.geometry, settings)
        self.to_detector = AngularSpectrumPropagator(distance_m=settings.global_to_detector_distance_m, **prop_kwargs)
        self.detector_readout = FullPlaneDetectorReadout(settings)
        self.output_adapter = nn.Linear(settings.input_adapter_dim, hidden_size)
        self.last_token_counts: list[int] = []
        self.last_input_fields: torch.Tensor | None = None
        self.last_detector_intensity: torch.Tensor | None = None
        self.last_detector_readout: torch.Tensor | None = None
        self.last_output: torch.Tensor | None = None
        self.last_routing: dict[str, torch.Tensor] = {}

    def encode_groups(self, groups: list[torch.Tensor]) -> torch.Tensor:
        fields = []
        for group in groups:
            token_count = len(group)
            if token_count > self.max_visual_tokens:
                raise RuntimeError(
                    f"visual token count {token_count} exceeds max_visual_tokens={self.max_visual_tokens}. "
                    "Lower processor_max_pixels. No crop, interpolation, pooling, or token truncation is allowed."
                )
            projected = self.nonnegative(self.input_norm(self.input_adapter(group.float())))
            field = projected.new_zeros((self.geometry.expert_size, self.geometry.expert_size))
            field[:token_count, :] = projected
            fields.append(field)
        return torch.stack(fields)

    @staticmethod
    def global_fanout_convolution(field: torch.Tensor, transmission: torch.Tensor) -> torch.Tensor:
        flipped = torch.flip(field.to(torch.complex64), dims=(-2, -1))
        return torch.fft.fftshift(torch.fft.ifft2(torch.fft.fft2(flipped) * torch.fft.fft2(transmission.to(torch.complex64))), dim=(-2, -1))

    def forward(self, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor | None = None, **_: Any) -> torch.Tensor:
        lengths = lengths_from_cu(hidden_states, cu_seqlens)
        self.last_token_counts = lengths
        groups = list(hidden_states.split(lengths, dim=0))
        input_fields = self.encode_groups(groups)
        self.last_input_fields = input_fields
        canvas = input_fields.new_zeros((len(groups), self.geometry.canvas_size, self.geometry.canvas_size))
        aperture = self.geometry.input_aperture
        canvas[:, aperture.y0:aperture.y1, aperture.x0:aperture.x1] = input_fields
        routing = self.prompt(input_fields)
        self.last_routing = routing
        field = self.global_fanout_convolution(canvas.to(torch.complex64), routing["transmission"])
        for phase, propagation, conversion in zip(self.expert_layers, self.propagations, self.interlayer_conversions):
            field = conversion(propagation(phase(field)))
        detector_field = self.to_detector(self.global_phase(field))
        readout, intensity = self.detector_readout(detector_field)
        self.last_detector_intensity = intensity
        self.last_detector_readout = readout
        outputs = [self.output_adapter(readout[index, :length, :]) for index, length in enumerate(lengths)]
        output = torch.cat(outputs, dim=0).to(hidden_states.dtype)
        self.last_output = output
        return output

    def router_losses(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.last_routing["balance_loss"], self.last_routing["importance_loss"]

    def parameter_breakdown(self) -> dict[str, int]:
        phase = sum(parameter.numel() for name, parameter in self.named_parameters() if "raw_phase" in name)
        router = sum(parameter.numel() for parameter in self.prompt.router.parameters())
        input_adapter = sum(parameter.numel() for parameter in self.input_adapter.parameters())
        input_norm = sum(parameter.numel() for parameter in self.input_norm.parameters())
        output_adapter = sum(parameter.numel() for parameter in self.output_adapter.parameters())
        total = sum(parameter.numel() for parameter in self.parameters())
        return {"optical_phase_parameters": phase, "router_parameters": router,
                "input_adapter_parameters": input_adapter, "input_adapter_norm_parameters": input_norm,
                "output_adapter_parameters": output_adapter,
                "adapter_parameters": input_adapter + input_norm + output_adapter,
                "detector_layernorm_parameters": sum(parameter.numel() for parameter in self.detector_readout.norm.parameters()),
                "surrogate_total_parameters": total, "surrogate_trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad)}

