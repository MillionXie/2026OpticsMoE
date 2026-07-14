"""Deep, purely linear complex-field experts for the heterogeneous MoE."""

from __future__ import annotations

import math
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from optics import AngularSpectrumPropagator, PhaseLayer


EXPERT_TYPES = ("d2nn", "fourier", "fiber")


def _inverse_sigmoid(value: float) -> float:
    value = min(max(float(value), 1.0e-6), 1.0 - 1.0e-6)
    return math.log(value / (1.0 - value))


class OpticalExpert(nn.Module):
    expert_type = "base"

    def __init__(self, size: int, scalar_cfg: dict | None = None) -> None:
        super().__init__()
        self.size = int(size)
        scalar_cfg = scalar_cfg or {}
        self.output_gain_enabled = bool(scalar_cfg.get("gain_enabled", False))
        self.output_phase_bias_enabled = bool(scalar_cfg.get("phase_bias_enabled", False))
        self.gain_min = float(scalar_cfg.get("gain_min", 0.0))
        self.gain_max = float(scalar_cfg.get("gain_max", 2.0))
        gain_init = float(scalar_cfg.get("gain_init", 1.0))
        if not self.gain_min < gain_init < self.gain_max:
            raise ValueError("expert_bank.output_scalar.gain_init must lie strictly inside its bounds")
        normalized = (gain_init - self.gain_min) / (self.gain_max - self.gain_min)
        raw_gain = torch.tensor(_inverse_sigmoid(normalized), dtype=torch.float32)
        phase_bias = torch.tensor(float(scalar_cfg.get("phase_bias_init_rad", 0.0)), dtype=torch.float32)
        if self.output_gain_enabled:
            self.raw_output_gain = nn.Parameter(raw_gain)
        else:
            self.register_buffer("raw_output_gain", raw_gain, persistent=True)
        if self.output_phase_bias_enabled:
            self.output_phase_bias = nn.Parameter(phase_bias)
        else:
            self.register_buffer("output_phase_bias", phase_bias, persistent=True)

    def _check_input(self, field: torch.Tensor) -> torch.Tensor:
        if field.ndim != 3 or tuple(field.shape[-2:]) != (self.size, self.size):
            raise ValueError(f"{type(self).__name__} expects [B,{self.size},{self.size}], got {tuple(field.shape)}")
        return field.to(torch.complex64)

    def output_gain(self) -> torch.Tensor:
        return self.gain_min + (self.gain_max - self.gain_min) * torch.sigmoid(self.raw_output_gain)

    def apply_output_scalar(self, field: torch.Tensor) -> torch.Tensor:
        gain = self.output_gain() if self.output_gain_enabled else field.real.new_tensor(1.0)
        phase = self.output_phase_bias if self.output_phase_bias_enabled else field.real.new_tensor(0.0)
        scalar = gain.to(field.device) * torch.exp(1j * phase.to(field.device))
        return (field.to(torch.complex64) * scalar.to(torch.complex64)).to(torch.complex64)

    def parameter_summary(self) -> dict:
        return {
            "type": self.expert_type,
            "parameters": int(sum(parameter.numel() for parameter in self.parameters())),
            "trainable_parameters": int(sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)),
            "output_gain": float(self.output_gain().detach().cpu()),
            "output_phase_bias_rad": float(self.output_phase_bias.detach().cpu()),
        }


class PaddedLocalPropagator(nn.Module):
    """Zero padding + angular spectrum propagation + finite centre crop."""

    def __init__(self, size: int, padding: int, **propagator_kwargs) -> None:
        super().__init__()
        self.size = int(size)
        self.padding = int(padding)
        if self.padding < 0:
            raise ValueError("propagation_padding must be non-negative")
        self.propagator = AngularSpectrumPropagator(
            grid_size=self.size + 2 * self.padding,
            **propagator_kwargs,
        )

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        if self.padding:
            field = F.pad(field, (self.padding,) * 4, mode="constant", value=0.0)
        field = self.propagator(field.to(torch.complex64))
        if self.padding:
            field = field[:, self.padding : self.padding + self.size, self.padding : self.padding + self.size]
        return field.to(torch.complex64)


def _phase_layer(size: int, cfg: dict, optics_cfg: dict) -> PhaseLayer:
    return PhaseLayer(
        size,
        parameterization=str(cfg.get("phase_param", optics_cfg.get("phase_param", "sigmoid"))),
        init=str(cfg.get("phase_init", optics_cfg.get("phase_init", "zeros"))),
        init_std=float(cfg.get("phase_init_std", optics_cfg.get("init_std", 0.02))),
        phase_dropout_mode="none",
        phase_dropout_p=0.0,
    )


def _propagator_kwargs(cfg: dict, optics_cfg: dict, distance_key: str) -> dict:
    return {
        "wavelength_m": float(optics_cfg.get("wavelength_m", 5.32e-7)),
        "pixel_size_m": float(optics_cfg.get("pixel_size_m", 16.0e-6)),
        "distance_m": float(cfg.get(distance_key, 0.05)),
        "evanescent_mode": str(optics_cfg.get("evanescent_mode", "zero")),
        "k_space_constraint_enabled": bool(optics_cfg.get("k_space_constraint_enabled", False)),
        "theta_max_deg": float(optics_cfg.get("theta_max_deg", 1.0)),
    }


class D2NNExpert(OpticalExpert):
    expert_type = "d2nn"

    def __init__(self, size: int, cfg: dict, optics_cfg: dict, scalar_cfg: dict | None = None) -> None:
        super().__init__(size, scalar_cfg)
        self.num_layers = int(cfg.get("num_layers", 5))
        self.propagation_padding = int(cfg.get("propagation_padding", 30))
        if self.num_layers < 1:
            raise ValueError("expert_bank.d2nn.num_layers must be positive")
        self.phase_layers = nn.ModuleList([_phase_layer(size, cfg, optics_cfg) for _ in range(self.num_layers)])
        common = _propagator_kwargs(cfg, optics_cfg, "inter_layer_distance_m")
        self.propagators = nn.ModuleList(
            [PaddedLocalPropagator(size, self.propagation_padding, **common) for _ in range(self.num_layers - 1)]
        )

    def forward(self, field: torch.Tensor, return_details: bool = False, capture_fields: bool = True):
        field = self._check_input(field)
        intermediates = []
        for index, phase in enumerate(self.phase_layers):
            field = phase(field)
            if index < len(self.propagators):
                field = self.propagators[index](field)
            if return_details and capture_fields:
                intermediates.append(field)
        output = self.apply_output_scalar(field)
        if not return_details:
            return output
        return output, {"spatial_layer_fields": intermediates}

    def phase_stack(self) -> torch.Tensor:
        return torch.stack([layer.get_phase_wrapped() for layer in self.phase_layers])

    def parameter_summary(self) -> dict:
        value = super().parameter_summary()
        value.update(
            {
                "phase_masks": self.num_layers,
                "phase_mask_parameters": int(sum(layer.raw_phase.numel() for layer in self.phase_layers)),
                "propagation_padding": self.propagation_padding,
            }
        )
        return value


class FourierConvolutionBlock(nn.Module):
    """Finite-aperture centred Fourier transform with a 120x120 phase mask."""

    def __init__(self, size: int, padding: int, cfg: dict) -> None:
        super().__init__()
        self.size = int(size)
        self.padding = int(padding)
        self.raw_frequency_phase = nn.Parameter(torch.empty(size, size, dtype=torch.float32))
        init = str(cfg.get("phase_init", "zeros")).lower()
        init_std = float(cfg.get("phase_init_std", 0.02))
        if init in {"zero", "zeros", "identity"}:
            nn.init.zeros_(self.raw_frequency_phase)
        elif init in {"uniform", "uniform_0_2pi"}:
            nn.init.uniform_(self.raw_frequency_phase, 0.0, 2.0 * math.pi)
        elif init in {"normal", "gaussian", "small_normal"}:
            nn.init.normal_(self.raw_frequency_phase, 0.0, init_std)
        else:
            raise ValueError(f"Unsupported Fourier phase_init: {init}")

    def frequency_phase(self) -> torch.Tensor:
        return torch.remainder(self.raw_frequency_phase, 2.0 * math.pi)

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        if self.padding:
            field = F.pad(field, (self.padding,) * 4, mode="constant", value=0.0)
        dims = (-2, -1)
        spectrum = torch.fft.fftshift(
            torch.fft.fft2(torch.fft.ifftshift(field, dim=dims), dim=dims, norm="ortho"),
            dim=dims,
        )
        start = self.padding
        stop = start + self.size
        # A finite 120x120 Fourier aperture is explicit: frequencies outside
        # it are blocked, not passed through an implicit larger phase plate.
        finite_spectrum = torch.zeros_like(spectrum)
        central = spectrum[:, start:stop, start:stop]
        frequency_mask = torch.exp(1j * self.frequency_phase()).to(torch.complex64)
        finite_spectrum[:, start:stop, start:stop] = central * frequency_mask
        spatial = torch.fft.fftshift(
            torch.fft.ifft2(torch.fft.ifftshift(finite_spectrum, dim=dims), dim=dims, norm="ortho"),
            dim=dims,
        )
        # The centre crop is the finite spatial expert aperture.  It is a
        # non-diagonal operation in Fourier space and prevents mask folding.
        return spatial[:, start:stop, start:stop].to(torch.complex64)


class FourierExpert(OpticalExpert):
    expert_type = "fourier"

    def __init__(self, size: int, cfg: dict, optics_cfg: dict, scalar_cfg: dict | None = None) -> None:
        super().__init__(size, scalar_cfg)
        if not bool(cfg.get("phase_only", True)):
            raise ValueError("Deep FourierExpert requires phase_only=true")
        self.num_conv_blocks = int(cfg.get("num_conv_blocks", 3))
        self.num_tail_spatial_layers = int(cfg.get("num_tail_spatial_layers", 2))
        self.propagation_padding = int(cfg.get("propagation_padding", 30))
        if self.num_conv_blocks < 1 or self.num_tail_spatial_layers < 1:
            raise ValueError("Fourier conv block and tail spatial layer counts must be positive")
        self.convolution_blocks = nn.ModuleList(
            [FourierConvolutionBlock(size, self.propagation_padding, cfg) for _ in range(self.num_conv_blocks)]
        )
        self.tail_spatial_layers = nn.ModuleList(
            [_phase_layer(size, cfg, optics_cfg) for _ in range(self.num_tail_spatial_layers)]
        )
        common = _propagator_kwargs(cfg, optics_cfg, "inter_block_distance_m")
        self.inter_block_propagators = nn.ModuleList(
            [PaddedLocalPropagator(size, self.propagation_padding, **common) for _ in range(self.num_conv_blocks - 1)]
        )
        tail_common = dict(common)
        tail_common["distance_m"] = float(cfg.get("tail_spatial_distance_m", cfg.get("inter_block_distance_m", 0.05)))
        self.tail_propagators = nn.ModuleList(
            [PaddedLocalPropagator(size, self.propagation_padding, **tail_common) for _ in range(self.num_tail_spatial_layers - 1)]
        )

    def forward(self, field: torch.Tensor, return_details: bool = False, capture_fields: bool = True):
        field = self._check_input(field)
        block_fields = []
        tail_fields = []
        for index, block in enumerate(self.convolution_blocks):
            field = block(field)
            if return_details and capture_fields:
                block_fields.append(field)
            if index < len(self.inter_block_propagators):
                field = self.inter_block_propagators[index](field)
        for index, phase in enumerate(self.tail_spatial_layers):
            field = phase(field)
            if index < len(self.tail_propagators):
                field = self.tail_propagators[index](field)
            if return_details and capture_fields:
                tail_fields.append(field)
        output = self.apply_output_scalar(field)
        if not return_details:
            return output
        return output, {"fourier_block_fields": block_fields, "tail_spatial_fields": tail_fields}

    def frequency_phase_stack(self) -> torch.Tensor:
        return torch.stack([block.frequency_phase() for block in self.convolution_blocks])

    def spatial_phase_stack(self) -> torch.Tensor:
        return torch.stack([layer.get_phase_wrapped() for layer in self.tail_spatial_layers])

    def parameter_summary(self) -> dict:
        value = super().parameter_summary()
        value.update(
            {
                "fourier_conv_blocks": self.num_conv_blocks,
                "tail_spatial_layers": self.num_tail_spatial_layers,
                "frequency_phase_parameters": int(sum(block.raw_frequency_phase.numel() for block in self.convolution_blocks)),
                "tail_spatial_phase_parameters": int(sum(layer.raw_phase.numel() for layer in self.tail_spatial_layers)),
                "propagation_padding": self.propagation_padding,
                "finite_aperture_between_blocks": True,
            }
        )
        return value


class FiberArrayExpert(OpticalExpert):
    expert_type = "fiber"

    def __init__(self, size: int, cfg: dict, optics_cfg: dict, scalar_cfg: dict | None = None) -> None:
        super().__init__(size, scalar_cfg)
        self.num_pre_layers = int(cfg.get("num_pre_d2nn_layers", 2))
        self.num_post_layers = int(cfg.get("num_post_d2nn_layers", 2))
        if self.num_pre_layers < 1 or self.num_post_layers < 1:
            raise ValueError("Fiber encoder and decoder must each contain at least one D2NN phase layer")
        self.propagation_padding = int(cfg.get("propagation_padding", 30))
        self.pre_layers = nn.ModuleList([_phase_layer(size, cfg, optics_cfg) for _ in range(self.num_pre_layers)])
        self.post_layers = nn.ModuleList([_phase_layer(size, cfg, optics_cfg) for _ in range(self.num_post_layers)])
        common = _propagator_kwargs(cfg, optics_cfg, "inter_layer_distance_m")
        self.pre_propagators = nn.ModuleList(
            [PaddedLocalPropagator(size, self.propagation_padding, **common) for _ in range(self.num_pre_layers - 1)]
        )
        self.post_propagators = nn.ModuleList(
            [PaddedLocalPropagator(size, self.propagation_padding, **common) for _ in range(self.num_post_layers - 1)]
        )
        fibers_per_axis = int(cfg.get("fibers_per_axis", 10))
        if fibers_per_axis < 1:
            raise ValueError("expert_bank.fiber.fibers_per_axis must be positive")
        self.mode_grid = (fibers_per_axis, fibers_per_axis)
        self.num_modes = fibers_per_axis**2
        self.mode_sigma_pixels = float(cfg.get("mode_sigma_px", 3.0))
        margin = float(cfg.get("mode_center_margin_px", 6.0))
        axis = torch.linspace(margin, size - 1.0 - margin, fibers_per_axis)
        yy, xx = torch.meshgrid(torch.arange(size, dtype=torch.float32), torch.arange(size, dtype=torch.float32), indexing="ij")
        modes = []
        centres = []
        for y in axis:
            for x in axis:
                mode = torch.exp(-((yy - y).square() + (xx - x).square()) / (2.0 * self.mode_sigma_pixels**2))
                mode = mode / mode.square().sum().sqrt().clamp_min(1.0e-12)
                modes.append(mode)
                centres.append(torch.stack((y, x)))
        mode_bank = torch.stack(modes).to(torch.complex64)
        if bool(cfg.get("mode_bank_trainable", False)):
            self.mode_bank = nn.Parameter(mode_bank)
        else:
            self.register_buffer("mode_bank", mode_bank, persistent=False)
        self.register_buffer("mode_centers_yx", torch.stack(centres), persistent=True)
        phase_value = torch.zeros(self.num_modes, dtype=torch.float32)
        amplitude_min = float(cfg.get("amplitude_min", 0.0))
        amplitude_max = float(cfg.get("amplitude_max", 1.0))
        amplitude_init = float(cfg.get("amplitude_init", 0.95))
        if not amplitude_min < amplitude_init < amplitude_max:
            raise ValueError("Fiber amplitude_init must lie strictly inside its bounds")
        self.amplitude_min = amplitude_min
        self.amplitude_max = amplitude_max
        normalized = (amplitude_init - amplitude_min) / (amplitude_max - amplitude_min)
        amplitude_value = torch.full((self.num_modes,), _inverse_sigmoid(normalized), dtype=torch.float32)
        if bool(cfg.get("trainable_mode_phase", True)):
            self.raw_mode_phase = nn.Parameter(phase_value)
        else:
            self.register_buffer("raw_mode_phase", phase_value, persistent=True)
        if bool(cfg.get("trainable_mode_amplitude", True)):
            self.raw_mode_amplitude = nn.Parameter(amplitude_value)
        else:
            self.register_buffer("raw_mode_amplitude", amplitude_value, persistent=True)

    def mode_phase(self) -> torch.Tensor:
        return torch.remainder(self.raw_mode_phase, 2.0 * math.pi)

    def mode_amplitude(self) -> torch.Tensor:
        return self.amplitude_min + (self.amplitude_max - self.amplitude_min) * torch.sigmoid(self.raw_mode_amplitude)

    def phase_stack(self) -> torch.Tensor:
        return torch.stack(
            [layer.get_phase_wrapped() for layer in self.pre_layers]
            + [layer.get_phase_wrapped() for layer in self.post_layers]
        )

    def forward(self, field: torch.Tensor, return_details: bool = False, capture_fields: bool = True):
        field = self._check_input(field)
        encoder_fields = []
        decoder_fields = []
        for index, phase in enumerate(self.pre_layers):
            field = phase(field)
            if index < len(self.pre_propagators):
                field = self.pre_propagators[index](field)
            if return_details and capture_fields:
                encoder_fields.append(field)
        encoded_field = field
        modes = self.mode_bank.to(field.device)
        coefficients = torch.einsum("mhw,bhw->bm", modes.conj(), encoded_field)
        mode_power = coefficients.abs().square()
        input_power = encoded_field.abs().square().sum((-2, -1)).clamp_min(1.0e-12)
        coupling_efficiency = mode_power.sum(-1) / input_power
        modal_total = mode_power.sum(-1, keepdim=True)
        mode_distribution = mode_power / modal_total.clamp_min(1.0e-12)
        effective_mode_number = torch.where(
            modal_total[:, 0] > 1.0e-12,
            1.0 / mode_distribution.square().sum(-1).clamp_min(1.0e-12),
            torch.zeros_like(modal_total[:, 0]),
        )
        transmission = self.mode_amplitude().to(field.device) * torch.exp(1j * self.mode_phase().to(field.device))
        reconstructed = torch.einsum("bm,mhw->bhw", coefficients * transmission.to(torch.complex64), modes).to(torch.complex64)
        field = reconstructed
        for index, phase in enumerate(self.post_layers):
            field = phase(field)
            if index < len(self.post_propagators):
                field = self.post_propagators[index](field)
            if return_details and capture_fields:
                decoder_fields.append(field)
        output = self.apply_output_scalar(field)
        if not return_details:
            return output
        return output, {
            "encoder_fields": encoder_fields,
            "encoded_field": encoded_field if capture_fields else None,
            "reconstructed_field": reconstructed if capture_fields else None,
            "decoder_fields": decoder_fields,
            "coupling_efficiency": coupling_efficiency,
            "mode_power_distribution": mode_distribution,
            "effective_mode_number": effective_mode_number,
        }

    def parameter_summary(self) -> dict:
        value = super().parameter_summary()
        value.update(
            {
                "pre_d2nn_layers": self.num_pre_layers,
                "post_d2nn_layers": self.num_post_layers,
                "d2nn_phase_parameters": int(sum(layer.raw_phase.numel() for layer in list(self.pre_layers) + list(self.post_layers))),
                "mode_grid": list(self.mode_grid),
                "num_modes": self.num_modes,
                "mode_sigma_pixels": self.mode_sigma_pixels,
                "mode_phase_parameters": int(self.raw_mode_phase.numel()) if isinstance(self.raw_mode_phase, nn.Parameter) else 0,
                "mode_amplitude_parameters": int(self.raw_mode_amplitude.numel()) if isinstance(self.raw_mode_amplitude, nn.Parameter) else 0,
                "mode_bank_trainable_parameters": int(self.mode_bank.numel()) if isinstance(self.mode_bank, nn.Parameter) else 0,
                "amplitude_bounds": [self.amplitude_min, self.amplitude_max],
                "propagation_padding": self.propagation_padding,
            }
        )
        return value


class HeterogeneousExpertBank(nn.Module):
    def __init__(self, layout, bank_cfg: dict, optics_cfg: dict) -> None:
        super().__init__()
        self.layout = layout
        assignments = [str(value).lower() for value in bank_cfg.get("assignments", [])]
        if len(assignments) != layout.num_experts:
            raise ValueError(f"expert_bank.assignments must have {layout.num_experts} entries")
        unknown = sorted(set(assignments) - set(EXPERT_TYPES))
        if unknown:
            raise ValueError(f"Unsupported expert types: {unknown}")
        self.expert_types = assignments
        scalar_cfg = bank_cfg.get("output_scalar", {})
        experts = []
        for expert_type in assignments:
            if expert_type == "d2nn":
                expert = D2NNExpert(layout.expert_size, bank_cfg.get("d2nn", {}), optics_cfg, scalar_cfg)
            elif expert_type == "fourier":
                expert = FourierExpert(layout.expert_size, bank_cfg.get("fourier", {}), optics_cfg, scalar_cfg)
            else:
                expert = FiberArrayExpert(layout.expert_size, bank_cfg.get("fiber", {}), optics_cfg, scalar_cfg)
            experts.append(expert)
        self.experts = nn.ModuleList(experts)
        self.max_fiber_modes = max((expert.num_modes for expert in experts if expert.expert_type == "fiber"), default=0)

    def forward(self, field: torch.Tensor, return_details: bool = False, capture_outputs: bool = True):
        output = torch.zeros_like(field, dtype=torch.complex64)
        local_outputs = []
        expert_details = []
        input_powers = []
        output_powers = []
        batch = field.shape[0]
        coupling = torch.full((batch, len(self.experts)), float("nan"), device=field.device)
        effective_modes = torch.full_like(coupling, float("nan"))
        mode_distributions = torch.zeros(batch, len(self.experts), self.max_fiber_modes, device=field.device)
        for index, (aperture, expert) in enumerate(zip(self.layout.expert_apertures, self.experts)):
            local = field[:, aperture.y0 : aperture.y1, aperture.x0 : aperture.x1].to(torch.complex64)
            need_details = return_details and (capture_outputs or expert.expert_type == "fiber")
            if need_details:
                local_output, details = expert(local, return_details=True, capture_fields=capture_outputs)
            else:
                local_output = expert(local)
                details = {}
            if local_output.shape != local.shape or local_output.dtype != torch.complex64:
                raise RuntimeError(f"Expert {index} violated the [B,120,120] complex64 interface")
            output[:, aperture.y0 : aperture.y1, aperture.x0 : aperture.x1] = local_output
            if capture_outputs:
                local_outputs.append(local_output)
                expert_details.append(details)
            input_powers.append(local.abs().square().sum((-2, -1)))
            output_powers.append(local_output.abs().square().sum((-2, -1)))
            if expert.expert_type == "fiber" and details:
                coupling[:, index] = details["coupling_efficiency"].detach()
                effective_modes[:, index] = details["effective_mode_number"].detach()
                mode_distributions[:, index, : expert.num_modes] = details["mode_power_distribution"].detach()
        if not return_details:
            return output
        return output, {
            "local_outputs": torch.stack(local_outputs, dim=1) if local_outputs else None,
            "expert_details": expert_details if capture_outputs else None,
            "input_power": torch.stack(input_powers, dim=1),
            "output_power": torch.stack(output_powers, dim=1),
            "fiber_coupling_efficiency": coupling,
            "fiber_effective_mode_number": effective_modes,
            "fiber_mode_power_distribution": mode_distributions,
            "expert_types": list(self.expert_types),
        }

    def parameter_report(self) -> dict:
        per_expert = []
        by_type = defaultdict(lambda: {"count": 0, "parameters": 0, "trainable_parameters": 0})
        for index, expert in enumerate(self.experts):
            value = expert.parameter_summary()
            value.update({"index": index, "row": index // 3, "column": index % 3})
            per_expert.append(value)
            aggregate = by_type[value["type"]]
            aggregate["count"] += 1
            aggregate["parameters"] += value["parameters"]
            aggregate["trainable_parameters"] += value["trainable_parameters"]
        return {"per_expert": per_expert, "by_type": dict(by_type)}

    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
