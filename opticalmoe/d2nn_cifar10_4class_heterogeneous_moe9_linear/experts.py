"""Linear complex-field experts for the heterogeneous optical MoE.

Every expert maps ``[B, H, W] complex64`` to the same shape.  None of the
experts detects intensity, normalizes a sample's power, or reloads a detected
signal.  They are therefore linear in the incident complex optical field for
fixed trainable parameters.
"""

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
    """Uniform expert interface with optional sample-independent output scalar."""

    expert_type = "base"

    def __init__(self, size: int, scalar_cfg: dict | None = None) -> None:
        super().__init__()
        self.size = int(size)
        scalar_cfg = scalar_cfg or {}
        self.output_gain_enabled = bool(scalar_cfg.get("gain_enabled", False))
        self.output_phase_bias_enabled = bool(scalar_cfg.get("phase_bias_enabled", False))
        self.gain_min = float(scalar_cfg.get("gain_min", 0.0))
        self.gain_max = float(scalar_cfg.get("gain_max", 2.0))
        if not self.gain_min < self.gain_max:
            raise ValueError("experts.output_scalar.gain_min must be smaller than gain_max")
        gain_init = float(scalar_cfg.get("gain_init", 1.0))
        if not self.gain_min < gain_init < self.gain_max:
            raise ValueError("experts.output_scalar.gain_init must lie strictly inside its bounds")
        normalized_gain = (gain_init - self.gain_min) / (self.gain_max - self.gain_min)
        raw_gain = torch.tensor(_inverse_sigmoid(normalized_gain), dtype=torch.float32)
        phase_bias = torch.tensor(float(scalar_cfg.get("phase_bias_init_rad", 0.0)), dtype=torch.float32)
        if self.output_gain_enabled:
            self.raw_output_gain = nn.Parameter(raw_gain)
        else:
            self.register_buffer("raw_output_gain", raw_gain, persistent=True)
        if self.output_phase_bias_enabled:
            self.output_phase_bias = nn.Parameter(phase_bias)
        else:
            self.register_buffer("output_phase_bias", phase_bias, persistent=True)

    def output_gain(self) -> torch.Tensor:
        return self.gain_min + (self.gain_max - self.gain_min) * torch.sigmoid(self.raw_output_gain)

    def apply_output_scalar(self, field: torch.Tensor) -> torch.Tensor:
        gain = self.output_gain() if self.output_gain_enabled else field.real.new_tensor(1.0)
        phase = self.output_phase_bias if self.output_phase_bias_enabled else field.real.new_tensor(0.0)
        scalar = gain.to(field.device) * torch.exp(1j * phase.to(field.device))
        return (field.to(torch.complex64) * scalar.to(torch.complex64)).to(torch.complex64)

    def _check_input(self, field: torch.Tensor) -> torch.Tensor:
        if field.ndim != 3 or tuple(field.shape[-2:]) != (self.size, self.size):
            raise ValueError(f"{type(self).__name__} expects [B,{self.size},{self.size}], got {tuple(field.shape)}")
        return field.to(torch.complex64)

    def parameter_summary(self) -> dict:
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        return {
            "type": self.expert_type,
            "parameters": int(total),
            "trainable_parameters": int(trainable),
            "output_gain": float(self.output_gain().detach().cpu()),
            "output_phase_bias_rad": float(self.output_phase_bias.detach().cpu()),
        }


class PaddedLocalPropagator(nn.Module):
    """Zero-pad a local field, propagate it, then centre-crop it."""

    def __init__(self, size: int, padding: int, **propagator_kwargs) -> None:
        super().__init__()
        self.size = int(size)
        self.padding = int(padding)
        if self.padding < 0:
            raise ValueError("D2NN propagation_padding must be non-negative")
        self.propagator = AngularSpectrumPropagator(
            grid_size=self.size + 2 * self.padding,
            **propagator_kwargs,
        )

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        if self.padding:
            field = F.pad(field, (self.padding,) * 4, mode="constant", value=0.0)
        field = self.propagator(field)
        if self.padding:
            field = field[:, self.padding : self.padding + self.size, self.padding : self.padding + self.size]
        return field.to(torch.complex64)


class D2NNExpert(OpticalExpert):
    expert_type = "d2nn"

    def __init__(self, size: int, cfg: dict, optics_cfg: dict, scalar_cfg: dict | None = None) -> None:
        super().__init__(size, scalar_cfg)
        self.num_layers = int(cfg.get("num_layers", 5))
        if self.num_layers < 1:
            raise ValueError("experts.d2nn.num_layers must be positive")
        self.propagation_padding = int(cfg.get("propagation_padding", 30))
        self.phase_layers = nn.ModuleList(
            [
                PhaseLayer(
                    size,
                    parameterization=str(cfg.get("phase_param", optics_cfg.get("phase_param", "sigmoid"))),
                    init=str(cfg.get("phase_init", optics_cfg.get("phase_init", "zeros"))),
                    init_std=float(cfg.get("phase_init_std", optics_cfg.get("init_std", 0.02))),
                    phase_dropout_mode="none",
                    phase_dropout_p=0.0,
                )
                for _ in range(self.num_layers)
            ]
        )
        common = {
            "wavelength_m": float(optics_cfg.get("wavelength_m", 5.32e-7)),
            "pixel_size_m": float(optics_cfg.get("pixel_size_m", 16.0e-6)),
            "distance_m": float(cfg.get("inter_layer_distance_m", 0.05)),
            "evanescent_mode": str(optics_cfg.get("evanescent_mode", "zero")),
            "k_space_constraint_enabled": bool(optics_cfg.get("k_space_constraint_enabled", False)),
            "theta_max_deg": float(optics_cfg.get("theta_max_deg", 1.0)),
        }
        self.propagators = nn.ModuleList(
            [PaddedLocalPropagator(size, self.propagation_padding, **common) for _ in range(self.num_layers - 1)]
        )

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        field = self._check_input(field)
        for index, phase_layer in enumerate(self.phase_layers):
            field = phase_layer(field)
            if index < len(self.propagators):
                field = self.propagators[index](field)
        return self.apply_output_scalar(field)

    def phase_stack(self) -> torch.Tensor:
        return torch.stack([layer.get_phase_wrapped() for layer in self.phase_layers])

    def parameter_summary(self) -> dict:
        summary = super().parameter_summary()
        summary.update(
            {
                "phase_masks": self.num_layers,
                "phase_mask_parameters": int(sum(layer.raw_phase.numel() for layer in self.phase_layers)),
                "propagation_padding": self.propagation_padding,
            }
        )
        return summary


class FourierExpert(OpticalExpert):
    expert_type = "fourier"

    def __init__(self, size: int, cfg: dict, scalar_cfg: dict | None = None) -> None:
        super().__init__(size, scalar_cfg)
        self.raw_fourier_phase = nn.Parameter(torch.empty(size, size, dtype=torch.float32))
        init = str(cfg.get("phase_init", "zeros")).lower()
        init_std = float(cfg.get("phase_init_std", 0.02))
        if init in {"zeros", "zero", "identity"}:
            nn.init.zeros_(self.raw_fourier_phase)
        elif init in {"uniform", "uniform_0_2pi"}:
            nn.init.uniform_(self.raw_fourier_phase, 0.0, 2.0 * math.pi)
        elif init in {"normal", "gaussian", "small_normal"}:
            nn.init.normal_(self.raw_fourier_phase, 0.0, init_std)
        else:
            raise ValueError(f"Unsupported Fourier phase_init: {init}")

    def fourier_phase(self) -> torch.Tensor:
        return torch.remainder(self.raw_fourier_phase, 2.0 * math.pi)

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        field = self._check_input(field)
        dims = (-2, -1)
        spectrum = torch.fft.fftshift(
            torch.fft.fft2(torch.fft.ifftshift(field, dim=dims), dim=dims, norm="ortho"),
            dim=dims,
        )
        mask = torch.exp(1j * self.fourier_phase()).to(torch.complex64)
        output = torch.fft.fftshift(
            torch.fft.ifft2(torch.fft.ifftshift(spectrum * mask, dim=dims), dim=dims, norm="ortho"),
            dim=dims,
        )
        return self.apply_output_scalar(output.to(torch.complex64))

    def parameter_summary(self) -> dict:
        summary = super().parameter_summary()
        summary.update({"fourier_phase_parameters": int(self.raw_fourier_phase.numel()), "fft_norm": "ortho"})
        return summary


class FiberArrayExpert(OpticalExpert):
    expert_type = "fiber"

    def __init__(self, size: int, cfg: dict, scalar_cfg: dict | None = None) -> None:
        super().__init__(size, scalar_cfg)
        rows, cols = [int(value) for value in cfg.get("mode_grid", [10, 10])]
        if rows < 1 or cols < 1:
            raise ValueError("experts.fiber.mode_grid values must be positive")
        self.mode_grid = (rows, cols)
        self.num_modes = rows * cols
        self.mode_sigma_pixels = float(cfg.get("mode_sigma_pixels", 4.0))
        if self.mode_sigma_pixels <= 0:
            raise ValueError("experts.fiber.mode_sigma_pixels must be positive")
        margin = float(cfg.get("mode_center_margin_pixels", 6.0))
        ys = torch.linspace(margin, size - 1.0 - margin, rows)
        xs = torch.linspace(margin, size - 1.0 - margin, cols)
        yy, xx = torch.meshgrid(torch.arange(size, dtype=torch.float32), torch.arange(size, dtype=torch.float32), indexing="ij")
        modes = []
        centers = []
        for y in ys:
            for x in xs:
                mode = torch.exp(-((yy - y).square() + (xx - x).square()) / (2.0 * self.mode_sigma_pixels**2))
                mode = mode / mode.square().sum().sqrt().clamp_min(1.0e-12)
                modes.append(mode)
                centers.append(torch.stack((y, x)))
        self.register_buffer("mode_bank", torch.stack(modes).to(torch.complex64), persistent=False)
        self.register_buffer("mode_centers_yx", torch.stack(centers), persistent=True)
        self.raw_mode_phase = nn.Parameter(torch.zeros(self.num_modes, dtype=torch.float32))
        self.amplitude_min = float(cfg.get("amplitude_min", 0.0))
        self.amplitude_max = float(cfg.get("amplitude_max", 1.0))
        amplitude_init = float(cfg.get("amplitude_init", 0.95))
        if not self.amplitude_min < amplitude_init < self.amplitude_max:
            raise ValueError("Fiber amplitude_init must lie strictly inside amplitude_min/amplitude_max")
        normalized = (amplitude_init - self.amplitude_min) / (self.amplitude_max - self.amplitude_min)
        self.raw_mode_amplitude = nn.Parameter(
            torch.full((self.num_modes,), _inverse_sigmoid(normalized), dtype=torch.float32)
        )

    def mode_phase(self) -> torch.Tensor:
        return torch.remainder(self.raw_mode_phase, 2.0 * math.pi)

    def mode_amplitude(self) -> torch.Tensor:
        return self.amplitude_min + (self.amplitude_max - self.amplitude_min) * torch.sigmoid(self.raw_mode_amplitude)

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        field = self._check_input(field)
        modes = self.mode_bank.to(field.device)
        coefficients = torch.einsum("mhw,bhw->bm", modes.conj(), field)
        transmission = self.mode_amplitude().to(field.device) * torch.exp(1j * self.mode_phase().to(field.device))
        reconstructed = torch.einsum("bm,mhw->bhw", coefficients * transmission.to(torch.complex64), modes)
        return self.apply_output_scalar(reconstructed.to(torch.complex64))

    def parameter_summary(self) -> dict:
        summary = super().parameter_summary()
        summary.update(
            {
                "mode_grid": list(self.mode_grid),
                "num_modes": self.num_modes,
                "mode_sigma_pixels": self.mode_sigma_pixels,
                "mode_phase_parameters": int(self.raw_mode_phase.numel()),
                "mode_amplitude_parameters": int(self.raw_mode_amplitude.numel()),
                "amplitude_bounds": [self.amplitude_min, self.amplitude_max],
            }
        )
        return summary


class HeterogeneousExpertBank(nn.Module):
    """Crop nine local fields, apply configured experts, and reassemble canvas."""

    def __init__(self, layout, experts_cfg: dict, optics_cfg: dict) -> None:
        super().__init__()
        self.layout = layout
        configured_types = [str(value).strip().lower() for value in experts_cfg.get("types", [])]
        if len(configured_types) != layout.num_experts:
            raise ValueError(f"experts.types must contain {layout.num_experts} entries, got {len(configured_types)}")
        unknown = sorted(set(configured_types) - set(EXPERT_TYPES))
        if unknown:
            raise ValueError(f"Unsupported expert types: {unknown}; expected {list(EXPERT_TYPES)}")
        self.expert_types = configured_types
        scalar_cfg = experts_cfg.get("output_scalar", {})
        modules = []
        for expert_type in configured_types:
            if expert_type == "d2nn":
                module = D2NNExpert(layout.expert_size, experts_cfg.get("d2nn", {}), optics_cfg, scalar_cfg)
            elif expert_type == "fourier":
                module = FourierExpert(layout.expert_size, experts_cfg.get("fourier", {}), scalar_cfg)
            else:
                module = FiberArrayExpert(layout.expert_size, experts_cfg.get("fiber", {}), scalar_cfg)
            modules.append(module)
        self.experts = nn.ModuleList(modules)

    def forward(self, field: torch.Tensor, return_details: bool = False, capture_outputs: bool = True):
        if field.ndim != 3 or tuple(field.shape[-2:]) != (self.layout.canvas_size, self.layout.canvas_size):
            raise ValueError(f"HeterogeneousExpertBank expects [B,{self.layout.canvas_size},{self.layout.canvas_size}]")
        output = torch.zeros_like(field, dtype=torch.complex64)
        local_outputs = []
        input_powers = []
        output_powers = []
        for aperture, expert in zip(self.layout.expert_apertures, self.experts):
            local = field[:, aperture.y0 : aperture.y1, aperture.x0 : aperture.x1].to(torch.complex64)
            local_output = expert(local)
            if local_output.dtype != torch.complex64 or local_output.shape != local.shape:
                raise RuntimeError(
                    f"Expert {type(expert).__name__} violated interface: {local_output.shape}/{local_output.dtype}"
                )
            output[:, aperture.y0 : aperture.y1, aperture.x0 : aperture.x1] = local_output
            if capture_outputs:
                local_outputs.append(local_output)
            input_powers.append(local.abs().square().sum(dim=(-2, -1)))
            output_powers.append(local_output.abs().square().sum(dim=(-2, -1)))
        if not return_details:
            return output
        return output, {
            "local_outputs": torch.stack(local_outputs, dim=1) if local_outputs else None,
            "input_power": torch.stack(input_powers, dim=1),
            "output_power": torch.stack(output_powers, dim=1),
            "expert_types": list(self.expert_types),
        }

    def parameter_report(self) -> dict:
        per_expert = []
        by_type = defaultdict(lambda: {"count": 0, "parameters": 0, "trainable_parameters": 0})
        for index, expert in enumerate(self.experts):
            summary = expert.parameter_summary()
            summary["index"] = index
            summary["row"] = index // 3
            summary["column"] = index % 3
            per_expert.append(summary)
            type_summary = by_type[summary["type"]]
            type_summary["count"] += 1
            type_summary["parameters"] += summary["parameters"]
            type_summary["trainable_parameters"] += summary["trainable_parameters"]
        return {"per_expert": per_expert, "by_type": dict(by_type)}

    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
