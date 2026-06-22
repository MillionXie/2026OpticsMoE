import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .nine_expert_geometry import NineExpertFair134Layout


class ASGlobalRouterPromptBank(nn.Module):
    """Task-specific complex-amplitude prompt for AS global routing.

    Each task owns 9 amplitude logits and 9 phase biases. The optical lens and
    grating carrier phases are shared fixed geometry. The amplitude weights are
    applied on the prompt plane inside the complex router sum.
    """

    def __init__(
        self,
        task_names: Sequence[str],
        layout: NineExpertFair134Layout,
        wavelength_m: float,
        pixel_size_m: float,
        prompt_to_expert_m: float,
        focal_length_m: float,
        amplitude_init_logits: float = 2.0,
        train_phase_biases: bool = True,
        prompt_mode: str = "complex_order_router",
        grating_scale: float = 1.0,
        grating_sign_x: float = 1.0,
        grating_sign_y: float = 1.0,
        normalize: str = "sum_amplitude",
    ) -> None:
        super().__init__()
        names = [str(name).lower() for name in task_names]
        if not names or len(names) != len(set(names)):
            raise ValueError("task_names must be non-empty and unique.")
        if prompt_mode not in {"complex_order_router", "phase_only_angle_sum"}:
            raise ValueError(
                "prompt_mode must be complex_order_router or phase_only_angle_sum."
            )
        if normalize not in {"sum_amplitude", "max_abs"}:
            raise ValueError("normalize must be sum_amplitude or max_abs.")

        layout.validate()
        self.task_names = names
        self.layout = layout
        self.wavelength_m = float(wavelength_m)
        self.pixel_size_m = float(pixel_size_m)
        self.prompt_to_expert_m = float(prompt_to_expert_m)
        self.focal_length_m = float(focal_length_m)
        self.prompt_mode = prompt_mode
        self.grating_scale = float(grating_scale)
        self.grating_sign_x = float(grating_sign_x)
        self.grating_sign_y = float(grating_sign_y)
        self.normalize = normalize

        self.amplitude_logits = nn.ParameterDict(
            {
                name: nn.Parameter(
                    torch.full((9,), float(amplitude_init_logits), dtype=torch.float32)
                )
                for name in names
            }
        )
        self.phase_biases = nn.ParameterDict(
            {
                name: nn.Parameter(
                    torch.zeros(9, dtype=torch.float32),
                    requires_grad=bool(train_phase_biases),
                )
                for name in names
            }
        )

        y_grid, x_grid = layout.physical_grids(pixel_size_m)
        k = 2.0 * math.pi / self.wavelength_m
        lens_phase = -k / (2.0 * self.focal_length_m) * (x_grid.square() + y_grid.square())
        grating_phases = []
        for center_y, center_x in layout.expert_centers:
            dy_px = float(center_y - layout.canvas_center[0])
            dx_px = float(center_x - layout.canvas_center[1])
            fx = dx_px * self.pixel_size_m / (self.wavelength_m * self.prompt_to_expert_m)
            fy = dy_px * self.pixel_size_m / (self.wavelength_m * self.prompt_to_expert_m)
            fx_eff = self.grating_sign_x * self.grating_scale * fx
            fy_eff = self.grating_sign_y * self.grating_scale * fy
            grating_phases.append(2.0 * math.pi * (fx_eff * x_grid + fy_eff * y_grid))

        self.register_buffer("lens_phase", lens_phase.to(torch.float32), persistent=False)
        self.register_buffer(
            "grating_phases",
            torch.stack(grating_phases, dim=0).to(torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "prompt_mask",
            layout.prompt_aperture_mask().to(torch.float32),
            persistent=False,
        )

    def normalize_name(self, task_name: str) -> str:
        name = str(task_name).lower()
        if name not in self.amplitude_logits:
            raise KeyError(f"Unknown task {task_name!r}. Available: {self.task_names}")
        return name

    def resolve_name(self, task_name: Optional[str] = None, task_id: Optional[int] = None) -> str:
        if task_name is not None and task_id is not None:
            raise ValueError("Provide task_name or task_id, not both.")
        if task_name is not None:
            return self.normalize_name(task_name)
        if task_id is None:
            raise ValueError("task_name or task_id is required.")
        index = int(task_id)
        if index < 0 or index >= len(self.task_names):
            raise IndexError(f"task_id {index} is outside the task list.")
        return self.task_names[index]

    def controls(self, task_name: str) -> Tuple[torch.Tensor, torch.Tensor]:
        name = self.normalize_name(task_name)
        return self.amplitude_logits[name], self.phase_biases[name]

    def amplitudes(self, task_name: str) -> torch.Tensor:
        logits, _ = self.controls(task_name)
        return torch.sigmoid(logits)

    def powers(self, task_name: str) -> torch.Tensor:
        return self.amplitudes(task_name).square()

    def normalized_powers(self, task_name: str) -> torch.Tensor:
        powers = self.powers(task_name)
        return powers / (powers.sum() + 1e-8)

    def router(self, task_name: str) -> torch.Tensor:
        amplitudes, phase_biases = self.controls(task_name)
        amplitudes = torch.sigmoid(amplitudes).to(self.grating_phases.device)
        phase_biases = phase_biases.to(self.grating_phases.device)
        channel_phase = self.grating_phases + phase_biases.view(9, 1, 1)
        router_sum = torch.sum(
            amplitudes.view(9, 1, 1) * torch.exp(1j * channel_phase),
            dim=0,
        ).to(torch.complex64)
        if self.normalize == "sum_amplitude":
            router = router_sum / (amplitudes.sum() + 1e-8)
        else:
            router = router_sum / torch.clamp(router_sum.abs().max(), min=1.0)
        if self.prompt_mode == "phase_only_angle_sum":
            router = torch.exp(1j * torch.angle(router)).to(torch.complex64)
        return router.to(torch.complex64)

    def transmission(self, task_name: str) -> torch.Tensor:
        router = self.router(task_name)
        lens = torch.exp(1j * self.lens_phase).to(torch.complex64)
        return (self.prompt_mask.to(torch.complex64) * lens * router).to(torch.complex64)

    def forward(self, field: torch.Tensor, task_name: str) -> torch.Tensor:
        return field.to(torch.complex64) * self.transmission(task_name).unsqueeze(0)

    def prompt_maps(self, task_name: str) -> Dict[str, torch.Tensor]:
        router = self.router(task_name)
        total = self.transmission(task_name)
        return {
            "prompt_router_amplitude": router.abs(),
            "prompt_router_phase": torch.remainder(torch.angle(router), 2.0 * math.pi),
            "prompt_total_amplitude": total.abs(),
            "prompt_total_phase": torch.remainder(torch.angle(total), 2.0 * math.pi),
        }

    def channel_table(self) -> List[Dict[str, float]]:
        rows = []
        for index, ((cy, cx), aperture) in enumerate(
            zip(self.layout.expert_centers, self.layout.expert_apertures)
        ):
            dy_px = float(cy - self.layout.canvas_center[0])
            dx_px = float(cx - self.layout.canvas_center[1])
            fx = dx_px * self.pixel_size_m / (self.wavelength_m * self.prompt_to_expert_m)
            fy = dy_px * self.pixel_size_m / (self.wavelength_m * self.prompt_to_expert_m)
            period_x_px = (
                math.inf
                if abs(fx) < 1e-20
                else 1.0 / (abs(fx) * self.pixel_size_m)
            )
            period_y_px = (
                math.inf
                if abs(fy) < 1e-20
                else 1.0 / (abs(fy) * self.pixel_size_m)
            )
            rows.append(
                {
                    "index": index,
                    "expert_id": aperture.name,
                    "target_center_y": float(cy),
                    "target_center_x": float(cx),
                    "dy_px": dy_px,
                    "dx_px": dx_px,
                    "fx_cycles_per_m": fx,
                    "fy_cycles_per_m": fy,
                    "grating_period_x_px": period_x_px,
                    "grating_period_y_px": period_y_px,
                    "predicted_shift_x_px": dx_px,
                    "predicted_shift_y_px": dy_px,
                    "grating_scale": self.grating_scale,
                    "grating_sign_x": self.grating_sign_x,
                    "grating_sign_y": self.grating_sign_y,
                }
            )
        return rows
