import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .angular_spectrum import AngularSpectrumPropagator
from .detectors import DetectorArray
from .phase_layers import PhaseLayer
from .readout import ElectronicReadout


class ParameterMatchedFullCanvasPhaseMask(nn.Module):
    """Full-canvas phase mask generated from a compact trainable phase grid.

    The compact grid keeps the six-layer control close to the trainable optical
    parameter count of the four-expert model. Its unit-magnitude complex
    modulation is interpolated to the full canvas, so there is no expert
    partition and no blocked gap.
    """

    def __init__(
        self,
        canvas_shape: Tuple[int, int],
        parameter_grid_size: int,
        phase_param: str = "unconstrained",
        phase_init: str = "uniform_0_2pi",
        init_std: float = 0.02,
    ) -> None:
        super().__init__()
        self.canvas_shape = tuple(int(value) for value in canvas_shape)
        self.parameter_grid_size = int(parameter_grid_size)
        self.phase = PhaseLayer(
            grid_size=self.parameter_grid_size,
            parameterization=phase_param,
            init=phase_init,
            init_std=init_std,
        )

    def complex_modulation(self) -> torch.Tensor:
        local_phase = self.phase.get_phase()
        real = torch.cos(local_phase)[None, None]
        imag = torch.sin(local_phase)[None, None]
        real = F.interpolate(
            real,
            size=self.canvas_shape,
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        imag = F.interpolate(
            imag,
            size=self.canvas_shape,
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        magnitude = torch.sqrt(real.square() + imag.square()).clamp_min(1e-8)
        return torch.complex(real / magnitude, imag / magnitude).to(
            torch.complex64
        )

    def get_phase_wrapped(self) -> torch.Tensor:
        modulation = self.complex_modulation()
        return torch.remainder(torch.angle(modulation), 2.0 * math.pi)

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        return field.to(torch.complex64) * self.complex_modulation().unsqueeze(0)


class SixLayerNoPromptControl(nn.Module):
    """Parameter-matched six-mask control without prompt or expert partitions."""

    def __init__(
        self,
        num_classes: int = 10,
        canvas_shape: Tuple[int, int] = (700, 700),
        input_size: int = 200,
        num_masks: int = 6,
        parameter_grid_size: int = 464,
        wavelength_m: float = 532e-9,
        pixel_size_m: float = 8e-6,
        distances_m: Optional[Dict[str, float]] = None,
        phase_param: str = "unconstrained",
        phase_init: str = "uniform_0_2pi",
        phase_init_std: float = 0.02,
        detector_size: int = 32,
        detector_layout: str = "grid",
        normalize_detector_energy: bool = True,
        readout_type: str = "optical_only",
        logit_scale: float = 10.0,
        readout_hidden_dim: int = 64,
        readout_activation: str = "relu",
        evanescent_mode: str = "zero",
    ) -> None:
        super().__init__()
        if int(num_masks) != 6:
            raise ValueError("This control experiment is defined with six masks.")
        self.num_classes = int(num_classes)
        self.canvas_shape = tuple(int(value) for value in canvas_shape)
        self.input_size = int(input_size)
        self.num_masks = int(num_masks)
        self.parameter_grid_size = int(parameter_grid_size)

        defaults = {
            "input_to_identity_prompt": 0.20,
            "identity_prompt_to_first_mask": 0.20,
            "inter_mask": 0.05,
            "last_mask_to_detector": 0.05,
        }
        self.distances_m = dict(defaults)
        if distances_m:
            self.distances_m.update(
                {key: float(value) for key, value in distances_m.items()}
            )

        self.input_to_identity_prompt = AngularSpectrumPropagator(
            wavelength_m=wavelength_m,
            pixel_size_m=pixel_size_m,
            grid_size=self.canvas_shape,
            distance_m=self.distances_m["input_to_identity_prompt"],
            evanescent_mode=evanescent_mode,
        )
        self.identity_prompt_to_first_mask = AngularSpectrumPropagator(
            wavelength_m=wavelength_m,
            pixel_size_m=pixel_size_m,
            grid_size=self.canvas_shape,
            distance_m=self.distances_m["identity_prompt_to_first_mask"],
            evanescent_mode=evanescent_mode,
        )
        self.phase_masks = nn.ModuleList(
            [
                ParameterMatchedFullCanvasPhaseMask(
                    canvas_shape=self.canvas_shape,
                    parameter_grid_size=self.parameter_grid_size,
                    phase_param=phase_param,
                    phase_init=phase_init,
                    init_std=phase_init_std,
                )
                for _ in range(self.num_masks)
            ]
        )
        self.inter_mask_propagators = nn.ModuleList(
            [
                AngularSpectrumPropagator(
                    wavelength_m=wavelength_m,
                    pixel_size_m=pixel_size_m,
                    grid_size=self.canvas_shape,
                    distance_m=self.distances_m["inter_mask"],
                    evanescent_mode=evanescent_mode,
                )
                for _ in range(self.num_masks - 1)
            ]
        )
        self.last_mask_to_detector = AngularSpectrumPropagator(
            wavelength_m=wavelength_m,
            pixel_size_m=pixel_size_m,
            grid_size=self.canvas_shape,
            distance_m=self.distances_m["last_mask_to_detector"],
            evanescent_mode=evanescent_mode,
        )
        self.detector = DetectorArray(
            num_classes=self.num_classes,
            grid_size=self.canvas_shape,
            detector_size=detector_size,
            layout=detector_layout,
            normalize_total_energy=normalize_detector_energy,
        )
        self.readout = ElectronicReadout(
            num_classes=self.num_classes,
            readout_type=readout_type,
            logit_scale=logit_scale,
            hidden_dim=readout_hidden_dim,
            activation=readout_activation,
        )

    @property
    def num_propagation_segments(self) -> int:
        return self.num_masks + 2

    def prepare_canvas_input(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim == 3:
            images = images.unsqueeze(1)
        if images.ndim != 4:
            raise ValueError(f"Expected [B,C,H,W], got {tuple(images.shape)}")
        images = images.float()
        if images.shape[1] != 1:
            images = images.mean(dim=1, keepdim=True)
        if tuple(images.shape[-2:]) != (self.input_size, self.input_size):
            images = F.interpolate(
                images,
                size=(self.input_size, self.input_size),
                mode="bilinear",
                align_corners=False,
            )
        images = images.clamp(0.0, 1.0)
        canvas = torch.zeros(
            images.shape[0],
            self.canvas_shape[0],
            self.canvas_shape[1],
            dtype=torch.float32,
            device=images.device,
        )
        y0 = (self.canvas_shape[0] - self.input_size) // 2
        x0 = (self.canvas_shape[1] - self.input_size) // 2
        canvas[:, y0 : y0 + self.input_size, x0 : x0 + self.input_size] = (
            images[:, 0]
        )
        return canvas.to(torch.complex64)

    def forward(self, images: torch.Tensor, return_intermediates: bool = False):
        input_field = self.prepare_canvas_input(images)
        after_input_to_prompt = self.input_to_identity_prompt(input_field)
        # The physical prompt plane is retained, but its transmission is one.
        after_identity_prompt = after_input_to_prompt
        field = self.identity_prompt_to_first_mask(after_identity_prompt)
        first_mask_entrance = field
        after_each_mask = []
        for index, phase_mask in enumerate(self.phase_masks):
            field = phase_mask(field)
            if return_intermediates:
                after_each_mask.append(field)
            if index < self.num_masks - 1:
                field = self.inter_mask_propagators[index](field)
        detector_field = self.last_mask_to_detector(field)
        detector_intensity = torch.abs(detector_field).square()
        detector_energies = self.detector(detector_field)
        logits = self.readout(detector_energies)
        if not return_intermediates:
            return logits
        return logits, {
            "input_amplitude": input_field.real,
            "after_input_to_prompt": after_input_to_prompt,
            "after_identity_prompt": after_identity_prompt,
            "first_mask_entrance": first_mask_entrance,
            "after_each_mask": after_each_mask,
            "detector_field": detector_field,
            "detector_intensity": detector_intensity,
            "detector_energies": detector_energies,
            "logits": logits,
        }

    def get_phase_masks_wrapped(self) -> torch.Tensor:
        return torch.stack(
            [mask.get_phase_wrapped() for mask in self.phase_masks],
            dim=0,
        )

    def optical_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.phase_masks.parameters())

    def electronic_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.readout.parameters())

