from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .settings import Settings


class AngularSpectrumPropagator(nn.Module):
    """One fixed-distance, band-limited angular-spectrum propagation."""

    def __init__(
        self,
        *,
        grid_size: int,
        wavelength_nm: float,
        pixel_pitch_um: float,
        distance_m: float,
    ) -> None:
        super().__init__()
        self.grid_size = int(grid_size)
        frequency = torch.fft.fftfreq(
            self.grid_size, d=float(pixel_pitch_um) * 1.0e-6, dtype=torch.float64
        )
        fy, fx = torch.meshgrid(frequency, frequency, indexing="ij")
        wavelength_m = float(wavelength_nm) * 1.0e-9
        argument = (1.0 / wavelength_m) ** 2 - fx.square() - fy.square()
        propagating = argument >= 0.0
        phase = 2.0 * math.pi * float(distance_m) * torch.sqrt(
            argument.clamp_min(0.0)
        )
        transfer = torch.exp(1j * phase).to(torch.complex64)
        self.register_buffer(
            "transfer_function",
            torch.where(propagating, transfer, torch.zeros_like(transfer)),
            persistent=False,
        )

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        expected = (self.grid_size, self.grid_size)
        if field.ndim != 3 or tuple(field.shape[-2:]) != expected:
            raise ValueError(f"Expected [B,{self.grid_size},{self.grid_size}], got {tuple(field.shape)}")
        spectrum = torch.fft.fft2(field.to(torch.complex64), dim=(-2, -1))
        return torch.fft.ifft2(
            spectrum * self.transfer_function, dim=(-2, -1)
        ).to(torch.complex64)


class SingleLayerMNIST4D2NN(nn.Module):
    """Co-planar amplitude/phase input followed by one 10 cm propagation."""

    def __init__(self, settings: Settings) -> None:
        super().__init__()
        self.settings = settings
        self.raw_phase = nn.Parameter(
            torch.zeros(settings.active_size, settings.active_size, dtype=torch.float32)
        )
        masks = torch.zeros(
            len(settings.classes),
            settings.active_size,
            settings.active_size,
            dtype=torch.float32,
        )
        for index, (left, top, right, bottom) in enumerate(
            settings.detector_bounds()
        ):
            masks[index, top:bottom, left:right] = 1.0
        self.register_buffer("detector_masks", masks, persistent=False)
        self.propagator = AngularSpectrumPropagator(
            grid_size=settings.propagation_grid_size,
            wavelength_nm=settings.wavelength_nm,
            pixel_pitch_um=settings.logical_pixel_pitch_um,
            distance_m=settings.detector_distance_m,
        )

    def phase(self) -> torch.Tensor:
        return 2.0 * math.pi * torch.sigmoid(self.raw_phase)

    def prepare_active_amplitude(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim == 3:
            images = images.unsqueeze(1)
        if images.ndim != 4 or images.shape[1] != 1:
            raise ValueError(f"Expected grayscale [B,1,H,W], got {tuple(images.shape)}")
        if tuple(images.shape[-2:]) != (
            self.settings.input_size,
            self.settings.input_size,
        ):
            images = F.interpolate(
                images.float(),
                size=(self.settings.input_size, self.settings.input_size),
                mode="bilinear",
                align_corners=False,
            )
        guard = self.settings.input_guard
        return F.pad(images[:, 0].float(), (guard, guard, guard, guard))

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        active_amplitude = self.prepare_active_amplitude(images)
        modulated = active_amplitude.to(torch.complex64) * torch.exp(
            1j * self.phase()
        ).to(torch.complex64)
        guard = self.settings.canvas_guard
        physical_canvas_field = F.pad(modulated, (guard, guard, guard, guard))
        propagation_guard = self.settings.propagation_guard
        numerical_field = F.pad(
            physical_canvas_field,
            (propagation_guard, propagation_guard, propagation_guard, propagation_guard),
        )
        numerical_detector_field = self.propagator(numerical_field)
        numerical_intensity = numerical_detector_field.abs().square().float()
        detector_intensity_canvas = numerical_intensity[
            :,
            propagation_guard : propagation_guard + self.settings.canvas_size,
            propagation_guard : propagation_guard + self.settings.canvas_size,
        ]
        detector_intensity = detector_intensity_canvas[
            :, guard : guard + self.settings.active_size,
            guard : guard + self.settings.active_size,
        ]
        detector_energy = torch.einsum(
            "bhw,chw->bc", detector_intensity, self.detector_masks
        )
        total_energy = detector_intensity_canvas.sum(dim=(-2, -1)).clamp_min(
            self.settings.loss_eps
        )
        detector_fraction = detector_energy / total_energy[:, None]
        logits = torch.log(detector_fraction.clamp_min(self.settings.loss_eps))
        return {
            "logits": logits,
            "detector_energy": detector_energy,
            "detector_fraction": detector_fraction,
            "detector_intensity": detector_intensity,
            "detector_intensity_canvas": detector_intensity_canvas,
            "active_amplitude": active_amplitude,
        }

    def optical_routing_loss(
        self, output: dict[str, torch.Tensor], targets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rows = torch.arange(len(targets), device=targets.device)
        fractions = output["detector_fraction"]
        detector_capture = fractions.sum(dim=1).clamp_min(self.settings.loss_eps)
        class_probability = fractions / detector_capture[:, None]
        detector_ce = -torch.log(
            class_probability[rows, targets].clamp_min(self.settings.loss_eps)
        ).mean()
        # Faithfully preserve the reference github_D2NN_mnist4 objective:
        # 100*MSE over the whole output intensity plane against a binary target
        # detector template. Detector CE remains an optional zero-weight
        # ablation term and is always logged as a diagnostic.
        target_active = self.detector_masks[targets]
        guard = self.settings.canvas_guard
        target_canvas = F.pad(target_active, (guard, guard, guard, guard))
        intensity = output["detector_intensity_canvas"]
        template_mse = 100.0 * F.mse_loss(intensity, target_canvas)
        total = (
            self.settings.template_mse_loss_weight * template_mse
            + self.settings.detector_ce_loss_weight * detector_ce
        )
        return total, template_mse, detector_ce

    @torch.no_grad()
    def phase_statistics(self) -> dict[str, float]:
        phase = self.phase().float()
        raw = self.raw_phase.float()
        return {
            "raw_phase_mean": float(raw.mean()),
            "raw_phase_std": float(raw.std()),
            "phase_mean_rad": float(phase.mean()),
            "phase_std_rad": float(phase.std()),
            "phase_min_rad": float(phase.min()),
            "phase_max_rad": float(phase.max()),
        }
