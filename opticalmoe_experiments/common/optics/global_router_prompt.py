import math
from typing import Dict, List

import torch
import torch.nn as nn

from .expert_layout import ExpertLayout


class GlobalRouterPrompt(nn.Module):
    """Prompt-plane AS global router.

    This is a global grating-order router, not a spatially partitioned prompt.
    Amplitude weights enter the prompt-plane complex router sum.
    """

    def __init__(
        self,
        layout: ExpertLayout,
        wavelength_m: float,
        pixel_size_m: float,
        prompt_to_expert_m: float,
        focal_length_m: float,
        mode: str = "complex_order_router",
        amplitude_init_logits: float = 2.0,
        train_amplitudes: bool = True,
        train_phase_biases: bool = True,
        grating_scale: float = 1.0,
        grating_sign_x: float = 1.0,
        grating_sign_y: float = 1.0,
        normalize: str = "sum_amplitude",
    ) -> None:
        super().__init__()
        if mode not in {"complex_order_router", "phase_only_angle_sum"}:
            raise ValueError("prompt mode must be complex_order_router or phase_only_angle_sum.")
        if normalize not in {"sum_amplitude", "max_abs"}:
            raise ValueError("normalize must be sum_amplitude or max_abs.")
        layout.validate()
        self.layout = layout
        self.mode = mode
        self.normalize = normalize
        self.wavelength_m = float(wavelength_m)
        self.pixel_size_m = float(pixel_size_m)
        self.prompt_to_expert_m = float(prompt_to_expert_m)
        self.focal_length_m = float(focal_length_m)
        self.grating_scale = float(grating_scale)
        self.grating_sign_x = float(grating_sign_x)
        self.grating_sign_y = float(grating_sign_y)

        self.amplitude_logits = nn.Parameter(
            torch.full((layout.num_experts,), float(amplitude_init_logits), dtype=torch.float32),
            requires_grad=bool(train_amplitudes),
        )
        self.phase_biases = nn.Parameter(
            torch.zeros(layout.num_experts, dtype=torch.float32),
            requires_grad=bool(train_phase_biases),
        )

        y_grid, x_grid = layout.physical_grids(pixel_size_m)
        k = 2.0 * math.pi / self.wavelength_m
        lens_phase = -k / (2.0 * self.focal_length_m) * (x_grid.square() + y_grid.square())
        grating_phases = []
        for cy, cx in layout.expert_centers:
            dy_px = float(cy - layout.canvas_center[0])
            dx_px = float(cx - layout.canvas_center[1])
            fx = dx_px * self.pixel_size_m / (self.wavelength_m * self.prompt_to_expert_m)
            fy = dy_px * self.pixel_size_m / (self.wavelength_m * self.prompt_to_expert_m)
            grating_phases.append(
                2.0
                * math.pi
                * (
                    self.grating_sign_x * self.grating_scale * fx * x_grid
                    + self.grating_sign_y * self.grating_scale * fy * y_grid
                )
            )
        self.register_buffer("lens_phase", lens_phase.float(), persistent=False)
        self.register_buffer("grating_phases", torch.stack(grating_phases, dim=0).float(), persistent=False)
        self.register_buffer("prompt_mask", layout.prompt_aperture_mask().float(), persistent=False)

    def amplitudes(self) -> torch.Tensor:
        return torch.sigmoid(self.amplitude_logits)

    def powers(self) -> torch.Tensor:
        return self.amplitudes().square()

    def normalized_powers(self) -> torch.Tensor:
        powers = self.powers()
        return powers / (powers.sum() + 1e-8)

    def router(self) -> torch.Tensor:
        amplitudes = self.amplitudes().to(self.grating_phases.device)
        phase_biases = self.phase_biases.to(self.grating_phases.device)
        channel_phase = self.grating_phases + phase_biases.view(-1, 1, 1)
        router_sum = torch.sum(amplitudes.view(-1, 1, 1) * torch.exp(1j * channel_phase), dim=0).to(torch.complex64)
        if self.normalize == "sum_amplitude":
            router = router_sum / (amplitudes.sum() + 1e-8)
        else:
            router = router_sum / torch.clamp(router_sum.abs().max(), min=1.0)
        if self.mode == "phase_only_angle_sum":
            router = torch.exp(1j * torch.angle(router)).to(torch.complex64)
        return router.to(torch.complex64)

    def transmission(self) -> torch.Tensor:
        lens = torch.exp(1j * self.lens_phase).to(torch.complex64)
        return (self.prompt_mask.to(torch.complex64) * lens * self.router()).to(torch.complex64)

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        return field.to(torch.complex64) * self.transmission().unsqueeze(0)

    def prompt_maps(self) -> Dict[str, torch.Tensor]:
        router = self.router()
        total = self.transmission()
        mask = self.prompt_mask.to(device=router.device, dtype=router.real.dtype)
        router_amplitude = router.abs() * mask
        router_phase = torch.remainder(torch.angle(router), 2.0 * math.pi) * mask
        total_amplitude = total.abs() * mask
        total_phase = torch.remainder(torch.angle(total), 2.0 * math.pi) * mask
        aperture = self.layout.prompt_aperture
        return {
            "prompt_router_amplitude": router_amplitude,
            "prompt_router_phase": router_phase,
            "prompt_total_amplitude": total_amplitude,
            "prompt_total_phase": total_phase,
            "prompt_aperture_mask": self.prompt_mask,
            "prompt_aperture_region": aperture.to_dict(),
            "prompt_aperture_bounds": [aperture.y0, aperture.y1, aperture.x0, aperture.x1],
        }

    def channel_table(self) -> List[Dict[str, float]]:
        rows = []
        for index, ((cy, cx), aperture) in enumerate(zip(self.layout.expert_centers, self.layout.expert_apertures)):
            dy_px = float(cy - self.layout.canvas_center[0])
            dx_px = float(cx - self.layout.canvas_center[1])
            fx = dx_px * self.pixel_size_m / (self.wavelength_m * self.prompt_to_expert_m)
            fy = dy_px * self.pixel_size_m / (self.wavelength_m * self.prompt_to_expert_m)
            rows.append(
                {
                    "index": index,
                    "expert_id": aperture.name,
                    "dx_px": dx_px,
                    "dy_px": dy_px,
                    "fx_cycles_per_m": fx,
                    "fy_cycles_per_m": fy,
                    "grating_period_x_px": math.inf if abs(fx) < 1e-20 else 1.0 / (abs(fx) * self.pixel_size_m),
                    "grating_period_y_px": math.inf if abs(fy) < 1e-20 else 1.0 / (abs(fy) * self.pixel_size_m),
                    "predicted_shift_x_px": dx_px,
                    "predicted_shift_y_px": dy_px,
                }
            )
        return rows
