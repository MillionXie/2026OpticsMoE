from __future__ import annotations

from typing import Any, Sequence

import torch
from PIL import Image
from torch import nn
from torchvision.transforms import functional as TF

from .optics import (
    AngularSpectrumPropagator,
    PhaseOnlyMask,
    TenRegionDetector,
)
from .settings import Settings


def _encode_rgb_tensor(
    rgb: torch.Tensor, encoding: str = "grayscale_amplitude"
) -> torch.Tensor:
    if rgb.ndim != 3 or rgb.shape[0] != 3 or rgb.shape[-2] != rgb.shape[-1]:
        raise ValueError("RGB tensor must have shape [3,H,H]")
    if encoding != "grayscale_amplitude":
        raise ValueError(
            "This D2NN baseline is a scalar grayscale optical system; "
            "input.encoding must be grayscale_amplitude"
        )
    weights = rgb.new_tensor([0.2989, 0.5870, 0.1140]).view(3, 1, 1)
    return (rgb * weights).sum(0, keepdim=True).clamp(0.0, 1.0)


def pil_images_to_amplitude(
    images: Sequence[Image.Image], encoding: str
) -> torch.Tensor:
    """Encode RGB images as a fixed, one-shot nonnegative scalar amplitude."""

    tensors = []
    for image in images:
        rgb = TF.pil_to_tensor(image.convert("RGB")).float().div_(255.0)
        tensors.append(_encode_rgb_tensor(rgb, encoding))
    if not tensors:
        raise ValueError("At least one input image is required")
    return torch.stack(tensors)


def pil_images_to_grayscale_amplitude(images: Sequence[Image.Image]) -> torch.Tensor:
    """Backward-compatible explicit grayscale helper."""

    return pil_images_to_amplitude(images, "grayscale_amplitude")


class TwoPlaneD2NNClassifier(nn.Module):
    """Input amplitude -> local phase -> 10 cm -> global phase -> 10 cm -> CCD."""

    def __init__(self, settings: Settings) -> None:
        super().__init__()
        self.settings = settings
        self.num_classes = len(settings.selected_skus)
        self.canvas_size = settings.canvas_size
        self.active_size = settings.active_size
        self.input_size = settings.image_size
        self.first_phase_size = settings.first_phase_size
        self.input_start = (self.canvas_size - self.input_size) // 2
        self.input_end = self.input_start + self.input_size
        self.active_start = (self.canvas_size - self.active_size) // 2
        self.active_end = self.active_start + self.active_size

        mask_kwargs = {
            "parameterization": settings.phase_parameterization,
            "init": settings.phase_init,
            "init_std": settings.phase_init_std,
        }
        self.first_phase = PhaseOnlyMask(self.first_phase_size, **mask_kwargs)
        self.second_global_phase = PhaseOnlyMask(self.active_size, **mask_kwargs)

        propagation_kwargs = {
            "wavelength_m": settings.wavelength_nm * 1e-9,
            "pixel_size_m": settings.pixel_pitch_um * 1e-6,
            "grid_size": settings.canvas_size,
            "k_space_constraint_enabled": settings.k_space_constraint_enabled,
            "theta_max_deg": settings.theta_max_deg,
        }
        self.input_propagator = (
            AngularSpectrumPropagator(
                distance_m=settings.input_to_first_phase_distance_m,
                **propagation_kwargs,
            )
            if settings.input_to_first_phase_distance_m > 0
            else None
        )
        self.first_to_second = AngularSpectrumPropagator(
            distance_m=settings.first_to_second_phase_distance_m,
            **propagation_kwargs,
        )
        self.second_to_detector = AngularSpectrumPropagator(
            distance_m=settings.second_phase_to_detector_distance_m,
            **propagation_kwargs,
        )
        self.detector = TenRegionDetector(
            canvas_size=settings.canvas_size,
            active_size=settings.active_size,
            row_layout=settings.detector_row_layout,
            region_size=settings.detector_size,
            horizontal_gap=settings.detector_horizontal_gap,
            vertical_gap=settings.detector_vertical_gap,
            normalize_total_energy=settings.detector_normalize_total_energy,
            eps=settings.detector_eps,
        )

    def prepare_input(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim == 3:
            images = images.unsqueeze(1)
        if images.ndim != 4 or images.shape[1] not in {1, 3}:
            raise ValueError("Images must have shape [B,1,H,W] or [B,3,H,W]")
        if tuple(images.shape[-2:]) != (self.input_size, self.input_size):
            raise ValueError(
                f"Input must already be {self.input_size}x{self.input_size}; "
                "silent resizing inside the optical model is forbidden"
            )
        images = images.float()
        if images.shape[1] == 3:
            images = torch.stack(
                [
                    _encode_rgb_tensor(image, self.settings.input_encoding)
                    for image in images
                ]
            )
        amplitude = images[:, 0].clamp(0.0, 1.0)
        canvas = amplitude.new_zeros(
            amplitude.shape[0], self.canvas_size, self.canvas_size
        )
        canvas[
            :,
            self.input_start : self.input_end,
            self.input_start : self.input_end,
        ] = amplitude
        return torch.complex(canvas, torch.zeros_like(canvas))

    @staticmethod
    def _apply_centered_mask(
        field: torch.Tensor, mask: PhaseOnlyMask, start: int, end: int
    ) -> torch.Tensor:
        output = field.to(torch.complex64).clone()
        output[:, start:end, start:end] = mask(field[:, start:end, start:end])
        return output

    def forward(
        self,
        images: torch.Tensor,
        *,
        return_intermediates: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
        input_field = self.prepare_input(images)
        field = (
            self.input_propagator(input_field)
            if self.input_propagator is not None
            else input_field
        )
        after_first_phase = self._apply_centered_mask(
            field, self.first_phase, self.input_start, self.input_end
        )
        before_second_phase = self.first_to_second(after_first_phase)
        after_second_phase = self._apply_centered_mask(
            before_second_phase,
            self.second_global_phase,
            self.active_start,
            self.active_end,
        )
        detector_field = self.second_to_detector(after_second_phase)
        energies, detector_intensity, full_intensity = self.detector(detector_field)
        if not torch.isfinite(energies).all() or torch.any(energies < 0):
            raise RuntimeError("D2NN detector energies must be finite and nonnegative")
        if not return_intermediates:
            return energies
        return energies, {
            "input_amplitude_canvas": input_field.real,
            "after_first_phase": after_first_phase,
            "before_second_phase": before_second_phase,
            "after_second_phase": after_second_phase,
            "detector_field": detector_field,
            "detector_intensity": detector_intensity,
            "full_detector_intensity": full_intensity,
            "detector_region_energies": energies,
        }

    def parameter_report(self) -> dict[str, Any]:
        first = self.first_phase.raw_phase.numel()
        second = self.second_global_phase.raw_phase.numel()
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )
        return {
            "model_type": "two_plane_phase_only_d2nn_classifier",
            "task": "grocery10_closed_set_classification",
            "phase_planes": 2,
            "first_local_phase_parameters": first,
            "second_global_phase_parameters": second,
            "optical_phase_parameters": first + second,
            "electronic_trainable_parameters": 0,
            "total_parameters": total,
            "total_trainable_parameters": trainable,
            "input_shape": [None, 1, self.input_size, self.input_size],
            "input_encoding": self.settings.input_encoding,
            "canvas_shape": [None, self.canvas_size, self.canvas_size],
            "first_phase_shape": [self.first_phase_size, self.first_phase_size],
            "second_phase_shape": [self.active_size, self.active_size],
            "detector_regions": 10,
            "detector_layout": list(self.settings.detector_row_layout),
            "output_shape": [None, 10],
            "intermediate_oeo_conversions": 0,
            "moe_experts": 0,
            "similarity_embedding_dim": None,
            "classification_readout": "ten fixed square-law detector regions",
        }
