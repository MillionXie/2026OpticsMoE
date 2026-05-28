from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .angular_spectrum import AngularSpectrumPropagator
from .detectors import DetectorArray
from .phase_layers import PhaseLayer
from .prompts import IdentityPrompt, PromptModule
from .readout import ElectronicReadout


class OpticalClassifier(nn.Module):
    """Diffractive optical neural network classifier with a reserved prompt plane."""

    def __init__(
        self,
        num_classes: int,
        wavelength_m: float = 532e-9,
        pixel_size_m: float = 8e-6,
        input_size: int = 200,
        padding: int = 200,
        grid_size: int = 600,
        num_layers: int = 5,
        distances_m: Optional[Dict[str, float]] = None,
        phase_param: str = "unconstrained",
        phase_init: str = "uniform",
        detector_size: int = 32,
        detector_layout: str = "grid",
        readout_type: str = "optical_only",
        normalize_detector_energy: bool = True,
        logit_scale: float = 10.0,
        readout_hidden_dim: int = 64,
        readout_activation: str = "relu",
        prompt: Optional[PromptModule] = None,
        evanescent_mode: str = "zero",
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.wavelength_m = float(wavelength_m)
        self.pixel_size_m = float(pixel_size_m)
        self.input_size = int(input_size)
        self.padding = int(padding)
        self.grid_size = int(grid_size)
        self.num_layers = int(num_layers)
        self.phase_param = phase_param

        expected_grid = self.input_size + 2 * self.padding
        if self.grid_size != expected_grid:
            raise ValueError(
                f"grid_size must equal input_size + 2 * padding: "
                f"{self.grid_size} != {expected_grid}"
            )

        distances_m = distances_m or {
            "input_to_prompt": 0.05,
            "prompt_to_first_layer": 0.05,
            "inter_layer": 0.05,
            "last_layer_to_detector": 0.05,
        }
        self.distances_m = dict(distances_m)

        self.prompt = prompt if prompt is not None else IdentityPrompt()
        self.phase_layers = nn.ModuleList(
            [
                PhaseLayer(
                    grid_size=self.grid_size,
                    parameterization=phase_param,
                    init=phase_init,
                )
                for _ in range(self.num_layers)
            ]
        )

        self.propagators = nn.ModuleList(self._build_propagators(evanescent_mode))
        if len(self.propagators) != self.num_layers + 2:
            raise RuntimeError("Number of propagation segments must equal num_layers + 2.")

        self.detector = DetectorArray(
            num_classes=num_classes,
            grid_size=self.grid_size,
            detector_size=detector_size,
            layout=detector_layout,
            normalize_total_energy=normalize_detector_energy,
        )
        self.readout = ElectronicReadout(
            num_classes=num_classes,
            readout_type=readout_type,
            logit_scale=logit_scale,
            hidden_dim=readout_hidden_dim,
            activation=readout_activation,
        )

    @property
    def num_propagation_segments(self) -> int:
        return len(self.propagators)

    def _build_propagators(self, evanescent_mode: str) -> list:
        distances = [
            self.distances_m["input_to_prompt"],
            self.distances_m["prompt_to_first_layer"],
        ]
        distances.extend([self.distances_m["inter_layer"]] * max(0, self.num_layers - 1))
        distances.append(self.distances_m["last_layer_to_detector"])

        return [
            AngularSpectrumPropagator(
                wavelength_m=self.wavelength_m,
                pixel_size_m=self.pixel_size_m,
                grid_size=self.grid_size,
                distance_m=distance,
                evanescent_mode=evanescent_mode,
            )
            for distance in distances
        ]

    def _prepare_amplitude(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            if x.shape[1] != 1:
                raise ValueError(f"Expected one channel input, got shape {tuple(x.shape)}")
            x = x[:, 0]
        elif x.ndim != 3:
            raise ValueError(f"Expected input shape [B, 1, H, W] or [B, H, W], got {tuple(x.shape)}")

        x = x.float()
        if x.shape[-2:] != (self.input_size, self.input_size):
            x = F.interpolate(
                x.unsqueeze(1),
                size=(self.input_size, self.input_size),
                mode="bilinear",
                align_corners=False,
            )[:, 0]
        return torch.clamp(x, 0.0, 1.0)

    def _pad_to_grid(self, amplitude: torch.Tensor) -> torch.Tensor:
        padded = F.pad(
            amplitude,
            (self.padding, self.padding, self.padding, self.padding),
            mode="constant",
            value=0.0,
        )
        return padded.to(torch.complex64)

    def optical_parameter_count(self) -> int:
        modules = [self.prompt, self.phase_layers]
        return sum(
            p.numel()
            for module in modules
            for p in module.parameters()
            if p.requires_grad
        )

    def electronic_parameter_count(self) -> int:
        return sum(p.numel() for p in self.readout.parameters() if p.requires_grad)

    def forward(self, x: torch.Tensor, return_intermediates: bool = False):
        intermediates = {} if return_intermediates else None

        amplitude = self._prepare_amplitude(x)
        if return_intermediates:
            intermediates["input_amplitude"] = amplitude.detach()

        field = self._pad_to_grid(amplitude)
        if return_intermediates:
            intermediates["padded_input"] = field.detach()

        field = self.propagators[0](field)
        if return_intermediates:
            intermediates["after_input_to_prompt"] = field.detach()

        field = self.prompt(field)
        if return_intermediates:
            intermediates["after_prompt"] = field.detach()

        field = self.propagators[1](field)
        if return_intermediates:
            intermediates["after_prompt_to_first_layer"] = field.detach()

        for layer_idx, layer in enumerate(self.phase_layers):
            one_based_idx = layer_idx + 1
            field = layer(field)
            if return_intermediates:
                intermediates[f"after_layer_{one_based_idx}_modulation"] = field.detach()

            if layer_idx < self.num_layers - 1:
                field = self.propagators[2 + layer_idx](field)
            else:
                field = self.propagators[-1](field)

            if return_intermediates:
                intermediates[f"after_layer_{one_based_idx}_propagation"] = field.detach()

        detector_field = field
        detector_intensity = torch.abs(detector_field) ** 2
        detector_energies = self.detector(detector_field)
        logits = self.readout(detector_energies)

        if return_intermediates:
            intermediates["detector_field"] = detector_field.detach()
            intermediates["detector_intensity"] = detector_intensity.detach()
            intermediates["detector_energies"] = detector_energies.detach()
            return logits, intermediates

        return logits
