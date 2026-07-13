import math
from typing import Tuple, Union

import torch
import torch.nn as nn


GridSize = Union[int, Tuple[int, int]]
PHASE_DROPOUT_MODES = {"none", "phase_bypass", "block_phase_bypass"}


class AngularSpectrumPropagator(nn.Module):
    """Fixed-distance angular spectrum free-space propagation."""

    def __init__(
        self,
        wavelength_m: float,
        pixel_size_m: float,
        grid_size: GridSize,
        distance_m: float,
        evanescent_mode: str = "zero",
        k_space_constraint_enabled: bool = False,
        theta_max_deg: float = 1.0,
    ):
        super().__init__()
        if evanescent_mode != "zero":
            raise ValueError("Only evanescent_mode='zero' is supported.")
        if isinstance(grid_size, int):
            height = width = grid_size
        else:
            height, width = grid_size
        self.wavelength_m = float(wavelength_m)
        self.pixel_size_m = float(pixel_size_m)
        self.grid_size = (int(height), int(width))
        self.distance_m = float(distance_m)
        self.k_space_constraint_enabled = bool(k_space_constraint_enabled)
        self.theta_max_deg = float(theta_max_deg)
        if self.k_space_constraint_enabled and not (0.0 < self.theta_max_deg <= 90.0):
            raise ValueError("theta_max_deg must be in (0, 90] when k-space constraint is enabled.")
        transfer_function, k_space_mask, max_angle_deg = self._build_transfer_function()
        self.max_sampled_angle_deg = float(max_angle_deg)
        self.k_space_pass_fraction = float(k_space_mask.to(torch.float64).mean().item())
        self.register_buffer("transfer_function", transfer_function, persistent=False)
        self.register_buffer("k_space_mask", k_space_mask, persistent=False)

    def _build_transfer_function(self):
        height, width = self.grid_size
        # Match the notebook/NumPy kernel construction in float64.  Computing
        # the multi-million-radian propagation phase in float32 introduces a
        # visible phase error even though shifted and unshifted FFT forms are
        # mathematically equivalent.
        fy = torch.fft.fftfreq(height, d=self.pixel_size_m, dtype=torch.float64)
        fx = torch.fft.fftfreq(width, d=self.pixel_size_m, dtype=torch.float64)
        fy_grid, fx_grid = torch.meshgrid(fy, fx, indexing="ij")
        argument = (2.0 * math.pi) ** 2 * (
            (1.0 / self.wavelength_m) ** 2 - fx_grid.square() - fy_grid.square()
        )
        propagating = argument >= 0.0
        radial_frequency = torch.sqrt(fx_grid.square() + fy_grid.square())
        wave_number = 2.0 * math.pi / self.wavelength_m
        radial_wave_number = 2.0 * math.pi * radial_frequency
        angle = torch.asin((radial_wave_number / wave_number).clamp(0.0, 1.0))
        max_angle_deg = float(torch.rad2deg(angle.max()).item())
        if self.k_space_constraint_enabled:
            cutoff = wave_number * math.sin(math.radians(self.theta_max_deg))
            k_space_mask = radial_wave_number <= cutoff
        else:
            k_space_mask = torch.ones_like(propagating, dtype=torch.bool)
        phase = self.distance_m * torch.sqrt(argument.clamp_min(0.0))
        transfer = torch.exp(1j * phase).to(torch.complex64)
        pass_mask = propagating & k_space_mask
        return torch.where(pass_mask, transfer, torch.zeros_like(transfer)), k_space_mask, max_angle_deg

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        if field.ndim != 3:
            raise ValueError(f"Expected [B,H,W], got {tuple(field.shape)}")
        if tuple(field.shape[-2:]) != self.grid_size:
            raise ValueError(f"Expected grid {self.grid_size}, got {tuple(field.shape[-2:])}")
        field = field.to(torch.complex64)
        # The notebook applies the first mask directly to the input.  Returning
        # here also avoids an unnecessary FFT/IFFT round trip at z=0.
        if self.distance_m == 0.0:
            return field
        spectrum = torch.fft.fft2(field, dim=(-2, -1))
        return torch.fft.ifft2(spectrum * self.transfer_function, dim=(-2, -1)).to(torch.complex64)


class PhaseLayer(nn.Module):
    """Phase-only modulation with optional training-time phase bypass dropout."""

    def __init__(
        self,
        grid_size: GridSize,
        parameterization: str = "unconstrained",
        init: str = "uniform_0_2pi",
        init_std: float = 0.02,
        phase_dropout_mode: str = "none",
        phase_dropout_p: float = 0.0,
        phase_dropout_block_size: int = 8,
        phase_dropout_batch_shared: bool = True,
    ):
        super().__init__()
        if isinstance(grid_size, int):
            height = width = grid_size
        else:
            height, width = grid_size
        if phase_dropout_mode not in PHASE_DROPOUT_MODES:
            raise ValueError(f"Unsupported phase dropout mode: {phase_dropout_mode}")
        self.grid_size = (int(height), int(width))
        self.parameterization = parameterization
        self.phase_dropout_mode = phase_dropout_mode
        self.phase_dropout_p = float(phase_dropout_p)
        self.phase_dropout_block_size = int(phase_dropout_block_size)
        self.phase_dropout_batch_shared = bool(phase_dropout_batch_shared)
        self.phase_dropout_active = True
        self.raw_phase = nn.Parameter(torch.empty(self.grid_size, dtype=torch.float32))
        self.reset_parameters(init, init_std)

    def reset_parameters(self, init: str, init_std: float) -> None:
        if self.parameterization == "sigmoid":
            self._reset_sigmoid_parameters(init, init_std)
            return
        if init in {"identity", "zeros"}:
            nn.init.zeros_(self.raw_phase)
        elif init in {"uniform", "uniform_0_2pi"}:
            nn.init.uniform_(self.raw_phase, 0.0, 2.0 * math.pi)
        elif init in {"normal", "small_normal"}:
            nn.init.normal_(self.raw_phase, 0.0, init_std)
        else:
            raise ValueError(f"Unsupported phase init: {init}")

    @staticmethod
    def _logit(value: torch.Tensor) -> torch.Tensor:
        return torch.log(value / (1.0 - value))

    def _reset_sigmoid_parameters(self, init: str, init_std: float) -> None:
        if init in {"identity", "zeros", "zero"}:
            # With phase = 2*pi*sigmoid(raw_phase), an exact effective phase of
            # zero would require raw_phase -> -inf and would saturate sigmoid,
            # killing gradients. raw_phase=0 gives a spatially constant pi
            # phase, which is optically equivalent to an identity mask up to a
            # global phase factor while keeping the sigmoid derivative maximal.
            nn.init.zeros_(self.raw_phase)
            return
        if init in {"uniform", "uniform_0_2pi"}:
            nn.init.uniform_(self.raw_phase, 0.0, 2.0 * math.pi)
            return
        if init in {"normal", "small_normal", "gaussian"}:
            nn.init.normal_(self.raw_phase, 0.0, float(init_std))
            return
        raise ValueError(f"Unsupported phase init: {init}")

    def get_phase(self) -> torch.Tensor:
        if self.parameterization == "unconstrained":
            return self.raw_phase
        if self.parameterization == "sigmoid":
            return 2.0 * math.pi * torch.sigmoid(self.raw_phase)
        raise ValueError(f"Unsupported phase parameterization: {self.parameterization}")

    def get_phase_wrapped(self) -> torch.Tensor:
        return torch.remainder(self.get_phase(), 2.0 * math.pi)

    def set_phase_dropout_active(self, active: bool) -> None:
        self.phase_dropout_active = bool(active)

    def _dropout_enabled(self) -> bool:
        return self.training and self.phase_dropout_active and self.phase_dropout_mode != "none" and self.phase_dropout_p > 0.0

    def _sample_keep_mask(self, batch_size: int, height: int, width: int, device: torch.device) -> torch.Tensor:
        mask_batch = 1 if self.phase_dropout_batch_shared else int(batch_size)
        keep_prob = 1.0 - self.phase_dropout_p
        if self.phase_dropout_mode == "phase_bypass":
            return (torch.rand((mask_batch, height, width), device=device) < keep_prob).float()
        if self.phase_dropout_mode == "block_phase_bypass":
            block = max(1, self.phase_dropout_block_size)
            low_h = int(math.ceil(height / block))
            low_w = int(math.ceil(width / block))
            mask = (torch.rand((mask_batch, low_h, low_w), device=device) < keep_prob).float()
            return mask.repeat_interleave(block, -2).repeat_interleave(block, -1)[:, :height, :width]
        raise RuntimeError(f"Unexpected phase dropout mode: {self.phase_dropout_mode}")

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        field = field.to(torch.complex64)
        phase = self.get_phase().to(device=field.device, dtype=torch.float32)
        modulation = torch.exp(1j * phase).to(torch.complex64)
        if not self._dropout_enabled():
            return field * modulation
        keep = self._sample_keep_mask(field.shape[0], phase.shape[-2], phase.shape[-1], field.device)
        keep_complex = keep.to(torch.complex64)
        effective_modulation = keep_complex * modulation.unsqueeze(0) + (1.0 - keep_complex)
        return field * effective_modulation


class DetectorArray(nn.Module):
    def __init__(
        self,
        num_classes: int,
        grid_size: GridSize,
        detector_size: int = 32,
        layout: str = "grid",
        normalize_total_energy: bool = True,
        eps: float = 1e-8,
        start_pos_x: int = 75,
        start_pos_y: int = 75,
        n_det_sets=None,
        det_steps_x=None,
        det_steps_y: int = 150,
    ):
        super().__init__()
        if isinstance(grid_size, int):
            height = width = grid_size
        else:
            height, width = grid_size
        self.num_classes = int(num_classes)
        self.grid_size = (int(height), int(width))
        self.detector_size = int(detector_size)
        self.layout = layout
        self.normalize_total_energy = bool(normalize_total_energy)
        self.eps = float(eps)
        self.start_pos_x = int(start_pos_x)
        self.start_pos_y = int(start_pos_y)
        self.n_det_sets = list(n_det_sets) if n_det_sets is not None else [2, 2]
        self.det_steps_x = list(det_steps_x) if det_steps_x is not None else [150 for _ in self.n_det_sets]
        self.det_steps_y = int(det_steps_y)
        self.register_buffer("masks", self._build_masks(), persistent=False)

    def _centers(self):
        height, width = self.grid_size
        if self.layout != "grid":
            raise ValueError(f"Unsupported detector layout: {self.layout}")
        rows = int(math.ceil(math.sqrt(self.num_classes)))
        cols = int(math.ceil(float(self.num_classes) / rows))
        ys = torch.linspace(self.detector_size // 2, height - self.detector_size // 2 - 1, steps=rows)
        xs = torch.linspace(self.detector_size // 2, width - self.detector_size // 2 - 1, steps=cols)
        centers = []
        for y in ys:
            for x in xs:
                centers.append((int(round(float(y))), int(round(float(x)))))
                if len(centers) == self.num_classes:
                    return centers
        return centers

    def _build_masks(self) -> torch.Tensor:
        height, width = self.grid_size
        masks = torch.zeros(self.num_classes, height, width, dtype=torch.float32)
        if self.layout == "fixed_2x2":
            positions = []
            for row_index, detectors_in_row in enumerate(self.n_det_sets):
                # Match github_D2NN_mnist4.ipynb exactly: det_steps_x/y are
                # clear gaps *between* detector squares, not top-left strides.
                y0 = self.start_pos_y + row_index * (self.detector_size + self.det_steps_y)
                gap_x = self.det_steps_x[min(row_index, len(self.det_steps_x) - 1)]
                for col_index in range(int(detectors_in_row)):
                    x0 = self.start_pos_x + col_index * (self.detector_size + int(gap_x))
                    positions.append((int(y0), int(x0)))
            if len(positions) < self.num_classes:
                raise ValueError(f"Detector layout supplies {len(positions)} regions for {self.num_classes} classes.")
            for index, (y0, x0) in enumerate(positions[: self.num_classes]):
                y1 = min(height, y0 + self.detector_size)
                x1 = min(width, x0 + self.detector_size)
                if y0 < 0 or x0 < 0 or y1 <= y0 or x1 <= x0:
                    raise ValueError(f"Invalid detector region {index}: y[{y0}:{y1}] x[{x0}:{x1}]")
                masks[index, y0:y1, x0:x1] = 1.0
            return masks
        half = self.detector_size // 2
        for index, (cy, cx) in enumerate(self._centers()):
            y0 = max(0, cy - half)
            x0 = max(0, cx - half)
            masks[index, y0 : min(height, y0 + self.detector_size), x0 : min(width, x0 + self.detector_size)] = 1.0
        return masks

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        intensity = torch.abs(field.to(torch.complex64)).square()
        energies = torch.einsum("bhw,chw->bc", intensity, self.masks)
        if self.normalize_total_energy:
            energies = energies / (intensity.sum(dim=(-2, -1), keepdim=False).unsqueeze(1) + self.eps)
        return energies
