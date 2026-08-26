from __future__ import annotations

import math
from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from .settings import V2Settings


Shift = tuple[int, int]
ShiftMap = dict[str, Shift]
SHIFT_KEYS = ("input", "phase", "pre_ccd")


def translate_zero_fill(tensor: torch.Tensor, *, dy: int, dx: int) -> torch.Tensor:
    """Translate the last two axes without circular wrap-around."""

    if tensor.ndim < 2:
        raise ValueError("translate_zero_fill expects at least two dimensions")
    height, width = tensor.shape[-2:]
    if abs(int(dy)) >= height or abs(int(dx)) >= width:
        return torch.zeros_like(tensor)
    source_top = max(0, -int(dy))
    source_bottom = min(height, height - int(dy))
    source_left = max(0, -int(dx))
    source_right = min(width, width - int(dx))
    target_top = max(0, int(dy))
    target_left = max(0, int(dx))
    target_bottom = target_top + source_bottom - source_top
    target_right = target_left + source_right - source_left
    result = torch.zeros_like(tensor)
    result[..., target_top:target_bottom, target_left:target_right] = tensor[
        ..., source_top:source_bottom, source_left:source_right
    ]
    return result


class AngularSpectrumKSpacePropagator(nn.Module):
    """Padded ASM with a fixed circular angular cutoff in k-space."""

    def __init__(
        self,
        *,
        grid_size: int,
        wavelength_nm: float,
        pixel_pitch_um: float,
        distance_m: float,
        k_space_enabled: bool,
        theta_max_deg: float,
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
        if k_space_enabled:
            cutoff = math.sin(math.radians(float(theta_max_deg))) / wavelength_m
            k_space_mask = fx.square() + fy.square() <= cutoff**2
        else:
            k_space_mask = torch.ones_like(propagating)
        pass_mask = propagating & k_space_mask
        phase = 2.0 * math.pi * float(distance_m) * torch.sqrt(
            argument.clamp_min(0.0)
        )
        transfer = torch.exp(1j * phase).to(torch.complex64)
        self.register_buffer(
            "transfer_function",
            torch.where(pass_mask, transfer, torch.zeros_like(transfer)),
            persistent=False,
        )
        self.register_buffer("k_space_pass_mask", pass_mask, persistent=False)
        self.pass_fraction = float(pass_mask.double().mean())

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        expected = (self.grid_size, self.grid_size)
        if field.ndim != 3 or tuple(field.shape[-2:]) != expected:
            raise ValueError(
                f"Expected [B,{self.grid_size},{self.grid_size}], got {tuple(field.shape)}"
            )
        spectrum = torch.fft.fft2(field.to(torch.complex64), dim=(-2, -1))
        return torch.fft.ifft2(
            spectrum * self.transfer_function, dim=(-2, -1)
        ).to(torch.complex64)


class RobustRawCCDMNIST4D2NN(nn.Module):
    """One phase mask, robust pre-CCD geometry, and an untouched raw CCD plane."""

    def __init__(self, settings: V2Settings) -> None:
        super().__init__()
        self.settings = settings
        self.robustness_training_active = True
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
        self.propagator = AngularSpectrumKSpacePropagator(
            grid_size=settings.propagation_grid_size,
            wavelength_nm=settings.wavelength_nm,
            pixel_pitch_um=settings.logical_pixel_pitch_um,
            distance_m=settings.detector_distance_m,
            k_space_enabled=settings.k_space_enabled,
            theta_max_deg=settings.k_space_theta_max_deg,
        )

    def phase(self) -> torch.Tensor:
        return 2.0 * math.pi * torch.sigmoid(self.raw_phase)

    def set_robustness_training_active(self, active: bool) -> None:
        """Enable train-time geometry jitter without changing eval behavior/state."""

        self.robustness_training_active = bool(active)

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

    @staticmethod
    def _sample_cardinal(max_pixels: int, probability: float) -> Shift:
        if int(max_pixels) <= 0 or float(torch.rand(())) >= float(probability):
            return (0, 0)
        magnitude = int(torch.randint(1, int(max_pixels) + 1, ()).item())
        direction = int(torch.randint(0, 4, ()).item())
        return (
            (-magnitude, 0),
            (magnitude, 0),
            (0, -magnitude),
            (0, magnitude),
        )[direction]

    def _resolve_shifts(
        self, forced_shifts: Mapping[str, Shift] | None
    ) -> ShiftMap:
        if forced_shifts is not None:
            extra = set(forced_shifts) - set(SHIFT_KEYS)
            if extra:
                raise ValueError(f"Unknown forced shift keys: {sorted(extra)}")
            return {
                key: tuple(int(value) for value in forced_shifts.get(key, (0, 0)))
                for key in SHIFT_KEYS
            }
        if (
            not self.training
            or not self.settings.robustness_enabled
            or not self.robustness_training_active
        ):
            return {key: (0, 0) for key in SHIFT_KEYS}
        probability = self.settings.robustness_probability
        return {
            "input": self._sample_cardinal(
                self.settings.input_shift_max_px, probability
            ),
            "phase": self._sample_cardinal(
                self.settings.phase_shift_max_px, probability
            ),
            "pre_ccd": self._sample_cardinal(
                self.settings.pre_ccd_shift_max_px, probability
            ),
        }

    def forward(
        self,
        images: torch.Tensor,
        *,
        forced_shifts: Mapping[str, Shift] | None = None,
    ) -> dict[str, torch.Tensor | ShiftMap]:
        shifts = self._resolve_shifts(forced_shifts)
        active_amplitude = self.prepare_active_amplitude(images)
        active_amplitude = translate_zero_fill(
            active_amplitude, dy=shifts["input"][0], dx=shifts["input"][1]
        )
        physical_phase = translate_zero_fill(
            self.phase(), dy=shifts["phase"][0], dx=shifts["phase"][1]
        )
        modulated = active_amplitude.to(torch.complex64) * torch.exp(
            1j * physical_phase
        ).to(torch.complex64)
        canvas_guard = self.settings.canvas_guard
        canvas_field = F.pad(
            modulated, (canvas_guard, canvas_guard, canvas_guard, canvas_guard)
        )
        propagation_guard = self.settings.propagation_guard
        numerical_field = F.pad(
            canvas_field,
            (
                propagation_guard,
                propagation_guard,
                propagation_guard,
                propagation_guard,
            ),
        )
        numerical_detector_field = self.propagator(numerical_field)
        detector_canvas_field = numerical_detector_field[
            :,
            propagation_guard : propagation_guard + self.settings.canvas_size,
            propagation_guard : propagation_guard + self.settings.canvas_size,
        ]
        raw_ccd_field = detector_canvas_field[
            :,
            canvas_guard : canvas_guard + self.settings.active_size,
            canvas_guard : canvas_guard + self.settings.active_size,
        ]
        raw_ccd_field = translate_zero_fill(
            raw_ccd_field, dy=shifts["pre_ccd"][0], dx=shifts["pre_ccd"][1]
        )
        # |E|^2 is the CCD photodetection itself. From this point onward no
        # activation, normalization, log compression, clipping, or background
        # subtraction is applied.
        ccd_intensity = raw_ccd_field.abs().square().float()
        detector_energy = torch.einsum(
            "bhw,chw->bc", ccd_intensity, self.detector_masks
        )
        return {
            "ccd_intensity": ccd_intensity,
            "detector_energy": detector_energy,
            "active_amplitude": active_amplitude,
            "applied_shifts": shifts,
        }

    def raw_ccd_loss(
        self,
        output: Mapping[str, torch.Tensor | ShiftMap],
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        intensity = output["ccd_intensity"]
        if not isinstance(intensity, torch.Tensor):
            raise TypeError("ccd_intensity must be a tensor")
        target_mask = self.detector_masks[targets]
        background_mask = 1.0 - target_mask
        target_area = target_mask[0].sum().clamp_min(1.0)
        background_area = background_mask[0].sum().clamp_min(1.0)
        target_region_mse = (
            (intensity - 1.0).square().mul(target_mask).sum()
            / (len(targets) * target_area)
        )
        background_mse = (
            intensity.square().mul(background_mask).sum()
            / (len(targets) * background_area)
        )
        detector_energy = torch.einsum(
            "bhw,chw->bc", intensity, self.detector_masks
        )
        # Training-only discriminative diagnostic/loss. Using log(raw energy)
        # as logits makes softmax exactly E_c / sum_j(E_j); it does not alter
        # the CCD tensor or add any inference-time nonlinearity.
        detector_ce = F.cross_entropy(
            detector_energy.clamp_min(self.settings.loss_eps).log(), targets
        )
        if self.settings.loss_mode == "notebook_full_plane_mse":
            # This is the audited notebook objective, now evaluated on the
            # proportionally mapped 478x478 CCD plane. The target is one in the
            # correct detector and zero everywhere else, so the large optical
            # background receives its proper pixel-count weight.
            total = self.settings.notebook_full_plane_mse_scale * (
                intensity - target_mask
            ).square().mean()
            total = total + self.settings.detector_ce_loss_weight * detector_ce
        elif self.settings.loss_mode == "legacy_balanced_region_mse":
            total = (
                self.settings.target_region_mse_weight * target_region_mse
                + self.settings.background_mse_weight * background_mse
            )
        else:  # Settings validation should make this unreachable.
            raise RuntimeError(f"Unsupported raw CCD loss mode: {self.settings.loss_mode}")
        return total, target_region_mse, background_mse, detector_ce

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
