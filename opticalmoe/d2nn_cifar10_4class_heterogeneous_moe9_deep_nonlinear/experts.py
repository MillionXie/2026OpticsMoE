"""Five-stage heterogeneous experts with selective stage-global OEO nonlinearities."""

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

    def nonlinear_enabled(self, stage_index: int) -> bool:
        return bool(self.nonlinear_schedule[int(stage_index)])


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


class StageGlobalOEO(nn.Module):
    """Parameter-free stage-global intensity LayerNorm followed by ReLU.

    Mean and standard deviation are recomputed for each sample and stage over
    every OEO-enabled expert pixel. They are never learned and are not shared
    across stages. Disabled regions retain their original complex field.
    """

    def __init__(self, cfg: dict, stage_index: int) -> None:
        super().__init__()
        self.stage_index = int(stage_index)
        normalization = cfg.get("normalization", {})
        reencoding = cfg.get("reencoding", {})
        if str(cfg.get("type", "intensity_layernorm_relu")) != "intensity_layernorm_relu":
            raise ValueError("nonlinearity.type must be intensity_layernorm_relu")
        if str(normalization.get("type", "stage_global_layernorm")) != "stage_global_layernorm":
            raise ValueError("nonlinearity.normalization.type must be stage_global_layernorm")
        if str(normalization.get("aperture", "nonlinear_enabled_expert_regions")) != "nonlinear_enabled_expert_regions":
            raise ValueError("normalization aperture must be nonlinear_enabled_expert_regions")
        if bool(normalization.get("elementwise_affine", False)):
            raise ValueError("Stage-global optical LayerNorm must use elementwise_affine=false")
        if str(cfg.get("activation", {}).get("type", "relu")) != "relu":
            raise ValueError("nonlinearity.activation.type must be relu")
        if not bool(reencoding.get("zero_phase", True)):
            raise ValueError("OEO re-encoding must use zero_phase=true")
        if str(reencoding.get("amplitude_source", "relu_output")) != "relu_output":
            raise ValueError("nonlinearity.reencoding.amplitude_source must be relu_output")
        self.eps = float(normalization.get("eps", 1.0e-6))
        if self.eps <= 0:
            raise ValueError("nonlinearity.normalization.eps must be positive")

    def forward(self, fields: list[torch.Tensor], enabled: list[bool], capture_fields: bool = False):
        enabled_indices = [index for index, value in enumerate(enabled) if value]
        if not enabled_indices:
            return fields, {
                "enabled": list(enabled),
                "normalization_mean": None,
                "normalization_std": None,
                "pre_power": None,
                "normalized_power": None,
                "output_power": None,
                "active_ratio": None,
                "pre_intensity": None,
                "normalized_intensity": None,
                "activation": None,
                "reencoded_amplitude": None,
            }
        selected = torch.stack([fields[index].to(torch.complex64) for index in enabled_indices], dim=1)
        intensity = selected.abs().square().float()
        mean = intensity.mean(dim=(1, 2, 3), keepdim=True)
        variance = (intensity - mean).square().mean(dim=(1, 2, 3), keepdim=True)
        std = torch.sqrt(variance + self.eps)
        normalized = (intensity - mean) / std
        activated = torch.relu(normalized)
        amplitude = activated
        reencoded = torch.complex(amplitude, torch.zeros_like(amplitude))
        outputs = list(fields)
        for local_index, expert_index in enumerate(enabled_indices):
            outputs[expert_index] = reencoded[:, local_index].to(torch.complex64)
        batch = intensity.shape[0]
        expert_count = len(fields)
        pre_power = torch.full((batch, expert_count), float("nan"), device=intensity.device)
        normalized_power = torch.full_like(pre_power, float("nan"))
        output_power = torch.full_like(pre_power, float("nan"))
        active_ratio = torch.full_like(pre_power, float("nan"))
        for local_index, expert_index in enumerate(enabled_indices):
            pre_power[:, expert_index] = intensity[:, local_index].sum((-2, -1))
            normalized_power[:, expert_index] = normalized[:, local_index].square().sum((-2, -1))
            output_power[:, expert_index] = amplitude[:, local_index].square().sum((-2, -1))
            active_ratio[:, expert_index] = (activated[:, local_index] > 0).float().mean((-2, -1))
        return outputs, {
            "enabled": list(enabled),
            "enabled_indices": enabled_indices,
            "normalization_mean": mean[:, 0, 0, 0],
            "normalization_std": std[:, 0, 0, 0],
            "pre_power": pre_power,
            "normalized_power": normalized_power,
            "output_power": output_power,
            "active_ratio": active_ratio,
            "pre_intensity": intensity if capture_fields else None,
            "normalized_intensity": normalized if capture_fields else None,
            "activation": activated if capture_fields else None,
            "reencoded_amplitude": amplitude if capture_fields else None,
        }


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
        if self.num_layers != 5:
            raise ValueError("The staged nonlinear D2NNExpert requires exactly five layers/stages")
        self.num_stages = self.num_layers
        self.nonlinear_schedule = [bool(value) for value in cfg.get("nonlinear_schedule", [True] * 5)]
        if len(self.nonlinear_schedule) != self.num_stages:
            raise ValueError("expert_bank.d2nn.nonlinear_schedule must contain five booleans")
        self.phase_layers = nn.ModuleList([_phase_layer(size, cfg, optics_cfg) for _ in range(self.num_layers)])
        common = _propagator_kwargs(cfg, optics_cfg, "inter_layer_distance_m")
        self.propagators = nn.ModuleList(
            [PaddedLocalPropagator(size, self.propagation_padding, **common) for _ in range(self.num_layers)]
        )

    def forward_stage(self, stage_index: int, field: torch.Tensor, return_details: bool = False):
        stage_index = int(stage_index)
        if not 0 <= stage_index < self.num_stages:
            raise IndexError(f"D2NN stage_index must be in [0,{self.num_stages - 1}]")
        field = self._check_input(field)
        output = self.propagators[stage_index](self.phase_layers[stage_index](field))
        if not return_details:
            return output
        return output, {"stage_index": stage_index, "stage_kind": "d2nn_phase", "linear_field": output}

    def forward(self, field: torch.Tensor, return_details: bool = False, capture_fields: bool = True):
        field = self._check_input(field)
        intermediates = []
        for index in range(self.num_stages):
            field = self.forward_stage(index, field)
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
                "num_stages": self.num_stages,
                "nonlinear_schedule": list(self.nonlinear_schedule),
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
        if self.num_conv_blocks + self.num_tail_spatial_layers != 5:
            raise ValueError("The staged nonlinear FourierExpert requires five stages in total")
        self.num_stages = 5
        self.nonlinear_schedule = [bool(value) for value in cfg.get("nonlinear_schedule", [True] * 5)]
        if len(self.nonlinear_schedule) != self.num_stages:
            raise ValueError("expert_bank.fourier.nonlinear_schedule must contain five booleans")
        self.convolution_blocks = nn.ModuleList(
            [FourierConvolutionBlock(size, self.propagation_padding, cfg) for _ in range(self.num_conv_blocks)]
        )
        self.tail_spatial_layers = nn.ModuleList(
            [_phase_layer(size, cfg, optics_cfg) for _ in range(self.num_tail_spatial_layers)]
        )
        common = _propagator_kwargs(cfg, optics_cfg, "inter_block_distance_m")
        self.inter_block_propagators = nn.ModuleList(
            [PaddedLocalPropagator(size, self.propagation_padding, **common) for _ in range(self.num_conv_blocks)]
        )
        tail_common = dict(common)
        tail_common["distance_m"] = float(cfg.get("tail_spatial_distance_m", cfg.get("inter_block_distance_m", 0.05)))
        self.tail_propagators = nn.ModuleList(
            [PaddedLocalPropagator(size, self.propagation_padding, **tail_common) for _ in range(self.num_tail_spatial_layers)]
        )

    def forward_stage(self, stage_index: int, field: torch.Tensor, return_details: bool = False):
        stage_index = int(stage_index)
        if not 0 <= stage_index < self.num_stages:
            raise IndexError(f"Fourier stage_index must be in [0,{self.num_stages - 1}]")
        field = self._check_input(field)
        if stage_index < self.num_conv_blocks:
            before_propagation = self.convolution_blocks[stage_index](field)
            output = self.inter_block_propagators[stage_index](before_propagation)
            kind = "fourier_convolution"
        else:
            local_index = stage_index - self.num_conv_blocks
            before_propagation = self.tail_spatial_layers[local_index](field)
            output = self.tail_propagators[local_index](before_propagation)
            kind = "spatial_phase"
        if not return_details:
            return output
        return output, {
            "stage_index": stage_index,
            "stage_kind": kind,
            "before_propagation": before_propagation,
            "linear_field": output,
        }

    def forward(self, field: torch.Tensor, return_details: bool = False, capture_fields: bool = True):
        field = self._check_input(field)
        block_fields = []
        tail_fields = []
        for index in range(self.num_stages):
            field = self.forward_stage(index, field)
            if return_details and capture_fields:
                if index < self.num_conv_blocks:
                    block_fields.append(field)
                else:
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
                "num_stages": self.num_stages,
                "nonlinear_schedule": list(self.nonlinear_schedule),
            }
        )
        return value


class FiberArrayExpert(OpticalExpert):
    expert_type = "fiber"

    def __init__(self, size: int, cfg: dict, optics_cfg: dict, scalar_cfg: dict | None = None) -> None:
        super().__init__(size, scalar_cfg)
        self.num_pre_layers = int(cfg.get("num_pre_d2nn_layers", 2))
        self.num_post_layers = int(cfg.get("num_post_d2nn_layers", 2))
        if self.num_pre_layers != 2 or self.num_post_layers != 2:
            raise ValueError("The staged nonlinear FiberArrayExpert requires a 2-layer encoder and 2-layer decoder")
        self.num_stages = 5
        self.nonlinear_schedule = [bool(value) for value in cfg.get("nonlinear_schedule", [True, False, True, True, True])]
        if len(self.nonlinear_schedule) != self.num_stages:
            raise ValueError("expert_bank.fiber.nonlinear_schedule must contain five booleans")
        if self.nonlinear_schedule[1]:
            raise ValueError("Fiber Stage2 must bypass OEO nonlinearity to preserve the complex field")
        self.propagation_padding = int(cfg.get("propagation_padding", 30))
        self.pre_layers = nn.ModuleList([_phase_layer(size, cfg, optics_cfg) for _ in range(self.num_pre_layers)])
        self.post_layers = nn.ModuleList([_phase_layer(size, cfg, optics_cfg) for _ in range(self.num_post_layers)])
        common = _propagator_kwargs(cfg, optics_cfg, "inter_layer_distance_m")
        self.pre_propagators = nn.ModuleList(
            [PaddedLocalPropagator(size, self.propagation_padding, **common) for _ in range(self.num_pre_layers)]
        )
        self.post_propagators = nn.ModuleList(
            [PaddedLocalPropagator(size, self.propagation_padding, **common) for _ in range(self.num_post_layers)]
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

    def _fiber_bottleneck(self, encoded_field: torch.Tensor, return_details: bool = False):
        modes = self.mode_bank.to(encoded_field.device)
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
        transmission = self.mode_amplitude().to(encoded_field.device) * torch.exp(
            1j * self.mode_phase().to(encoded_field.device)
        )
        reconstructed = torch.einsum(
            "bm,mhw->bhw", coefficients * transmission.to(torch.complex64), modes
        ).to(torch.complex64)
        if not return_details:
            return reconstructed
        return reconstructed, {
            "encoded_field": encoded_field,
            "reconstructed_field": reconstructed,
            "coupling_efficiency": coupling_efficiency,
            "mode_power": mode_power,
            "mode_power_distribution": mode_distribution,
            "effective_mode_number": effective_mode_number,
            "reconstruction_power": reconstructed.abs().square().sum((-2, -1)),
            "stage2_bypasses_nonlinearity": True,
        }

    def forward_stage(self, stage_index: int, field: torch.Tensor, return_details: bool = False):
        stage_index = int(stage_index)
        if not 0 <= stage_index < self.num_stages:
            raise IndexError(f"Fiber stage_index must be in [0,{self.num_stages - 1}]")
        field = self._check_input(field)
        details = {"stage_index": stage_index}
        if stage_index < 2:
            output = self.pre_propagators[stage_index](self.pre_layers[stage_index](field))
            details.update({"stage_kind": "d2nn_encoder", "linear_field": output})
        elif stage_index == 2:
            result = self._fiber_bottleneck(field, return_details=return_details)
            if return_details:
                output, fiber_details = result
                details.update(fiber_details)
            else:
                output = result
            details.update({"stage_kind": "fiber_bottleneck", "linear_field": output})
        else:
            local_index = stage_index - 3
            output = self.post_propagators[local_index](self.post_layers[local_index](field))
            details.update({"stage_kind": "d2nn_decoder", "linear_field": output})
        if not return_details:
            return output
        return output, details

    def forward(self, field: torch.Tensor, return_details: bool = False, capture_fields: bool = True):
        field = self._check_input(field)
        encoder_fields = []
        decoder_fields = []
        fiber_details = {}
        for index in range(self.num_stages):
            result = self.forward_stage(index, field, return_details=return_details and index == 2)
            if return_details and index == 2:
                field, fiber_details = result
            else:
                field = result
            if return_details and capture_fields:
                if index < 2:
                    encoder_fields.append(field)
                elif index > 2:
                    decoder_fields.append(field)
        output = self.apply_output_scalar(field)
        if not return_details:
            return output
        return output, {
            "encoder_fields": encoder_fields,
            "encoded_field": fiber_details.get("encoded_field") if capture_fields else None,
            "reconstructed_field": fiber_details.get("reconstructed_field") if capture_fields else None,
            "decoder_fields": decoder_fields,
            "coupling_efficiency": fiber_details["coupling_efficiency"],
            "mode_power_distribution": fiber_details["mode_power_distribution"],
            "effective_mode_number": fiber_details["effective_mode_number"],
            "reconstruction_power": fiber_details["reconstruction_power"],
            "stage2_bypasses_nonlinearity": True,
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
                "num_stages": self.num_stages,
                "nonlinear_schedule": list(self.nonlinear_schedule),
                "extra_fiber_coupling_phase_parameters": 0,
            }
        )
        return value


class HeterogeneousExpertBank(nn.Module):
    """Nine experts advanced synchronously through five stages.

    The linear stage is expert-specific.  OEO is then performed once over all
    enabled regions, which is what makes normalization stage-global instead of
    independently normalizing away each routed expert's relative power.
    """

    def __init__(self, layout, bank_cfg: dict, optics_cfg: dict, nonlinearity_cfg: dict) -> None:
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
        self.num_stages = 5
        self.nonlinearity_enabled = bool(nonlinearity_cfg.get("enabled", True))
        self.stage_nonlinearities = nn.ModuleList(
            [StageGlobalOEO(nonlinearity_cfg, stage_index) for stage_index in range(self.num_stages)]
        )
        self.max_fiber_modes = max((expert.num_modes for expert in experts if expert.expert_type == "fiber"), default=0)

    def forward(self, field: torch.Tensor, return_details: bool = False, capture_outputs: bool = True):
        locals_ = [
            field[:, aperture.y0 : aperture.y1, aperture.x0 : aperture.x1].to(torch.complex64)
            for aperture in self.layout.expert_apertures
        ]
        input_powers = torch.stack([value.abs().square().sum((-2, -1)) for value in locals_], dim=1)
        stage_details = []
        batch = field.shape[0]
        coupling = torch.full((batch, len(self.experts)), float("nan"), device=field.device)
        effective_modes = torch.full_like(coupling, float("nan"))
        reconstruction_power = torch.full_like(coupling, float("nan"))
        mode_distributions = torch.zeros(batch, len(self.experts), self.max_fiber_modes, device=field.device)
        for stage_index in range(self.num_stages):
            linear_input_power = torch.stack([value.abs().square().sum((-2, -1)) for value in locals_], dim=1)
            linear_outputs = []
            linear_details = []
            for expert_index, expert in enumerate(self.experts):
                need_details = return_details and (capture_outputs or (expert.expert_type == "fiber" and stage_index == 2))
                result = expert.forward_stage(stage_index, locals_[expert_index], return_details=need_details)
                if need_details:
                    local_output, details = result
                else:
                    local_output, details = result, {}
                if local_output.shape != locals_[expert_index].shape or local_output.dtype != torch.complex64:
                    raise RuntimeError(f"Expert {expert_index} violated the [B,120,120] complex64 stage interface")
                linear_outputs.append(local_output)
                linear_details.append(details if capture_outputs else None)
                if expert.expert_type == "fiber" and stage_index == 2 and details:
                    coupling[:, expert_index] = details["coupling_efficiency"].detach()
                    effective_modes[:, expert_index] = details["effective_mode_number"].detach()
                    reconstruction_power[:, expert_index] = details["reconstruction_power"].detach()
                    mode_distributions[:, expert_index, : expert.num_modes] = details["mode_power_distribution"].detach()
            enabled = [self.nonlinearity_enabled and expert.nonlinear_enabled(stage_index) for expert in self.experts]
            linear_output_power = torch.stack(
                [value.abs().square().sum((-2, -1)) for value in linear_outputs], dim=1
            )
            locals_, oeo = self.stage_nonlinearities[stage_index](linear_outputs, enabled, capture_fields=capture_outputs)
            if return_details:
                stage_details.append(
                    {
                        "stage_index": stage_index,
                        "nonlinear_enabled": enabled,
                        "linear_input_power": linear_input_power,
                        "linear_output_power": linear_output_power,
                        "linear_fields": torch.stack(linear_outputs, dim=1) if capture_outputs else None,
                        "output_fields": torch.stack(locals_, dim=1) if capture_outputs else None,
                        "linear_details": linear_details if capture_outputs else None,
                        "oeo": oeo,
                    }
                )
        output = torch.zeros_like(field, dtype=torch.complex64)
        final_locals = []
        for aperture, local_output, expert in zip(self.layout.expert_apertures, locals_, self.experts):
            local_output = expert.apply_output_scalar(local_output)
            final_locals.append(local_output)
            output[:, aperture.y0 : aperture.y1, aperture.x0 : aperture.x1] = local_output
        output_powers = torch.stack([value.abs().square().sum((-2, -1)) for value in final_locals], dim=1)
        if not return_details:
            return output
        return output, {
            "local_outputs": torch.stack(final_locals, dim=1) if capture_outputs else None,
            "stage_details": stage_details,
            "input_power": input_powers,
            "output_power": output_powers,
            "fiber_coupling_efficiency": coupling,
            "fiber_effective_mode_number": effective_modes,
            "fiber_reconstruction_power": reconstruction_power,
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

    def nonlinearity_parameter_report(self) -> dict:
        per_stage = []
        for index, module in enumerate(self.stage_nonlinearities):
            per_stage.append(
                {
                    "stage": index + 1,
                    "normalization": "per_sample_stage_global_layernorm",
                    "elementwise_affine": False,
                    "activation": "relu",
                    "reencoding": "relu_output_as_zero_phase_amplitude",
                    "parameters": int(sum(value.numel() for value in module.parameters())),
                    "trainable_parameters": int(sum(value.numel() for value in module.parameters() if value.requires_grad)),
                }
            )
        return {
            "enabled": self.nonlinearity_enabled,
            "learned_gain": False,
            "learned_threshold": False,
            "per_stage": per_stage,
            "parameters": int(sum(value.numel() for value in self.stage_nonlinearities.parameters())),
            "trainable_parameters": int(sum(value.numel() for value in self.stage_nonlinearities.parameters() if value.requires_grad)),
        }

    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
