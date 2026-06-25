import torch
import torch.nn as nn

from .expert_layout import ExpertLayout
from .phase_layers import PhaseLayer


class ExpertPhaseLayer(nn.Module):
    """Local phase masks embedded into a global expert aperture layout."""

    def __init__(
        self,
        layout: ExpertLayout,
        phase_param: str = "unconstrained",
        phase_init: str = "identity",
        init_std: float = 0.02,
        aperture_mode: str = "hard",
        phase_dropout_mode: str = "none",
        phase_dropout_p: float = 0.0,
        phase_dropout_block_size: int = 8,
        phase_dropout_batch_shared: bool = True,
    ) -> None:
        super().__init__()
        if aperture_mode not in {"hard", "transparent"}:
            raise ValueError("aperture_mode must be hard or transparent.")
        self.layout = layout
        self.layout.validate()
        self.aperture_mode = aperture_mode
        self.local_phases = nn.ModuleList(
            [
                PhaseLayer(
                    grid_size=(layout.expert_size, layout.expert_size),
                    parameterization=phase_param,
                    init=phase_init,
                    init_std=init_std,
                    phase_dropout_mode=phase_dropout_mode,
                    phase_dropout_p=phase_dropout_p,
                    phase_dropout_block_size=phase_dropout_block_size,
                    phase_dropout_batch_shared=phase_dropout_batch_shared,
                )
                for _ in range(layout.num_experts)
            ]
        )

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        if field.ndim != 3:
            raise ValueError(f"Expected [B,H,W], got {tuple(field.shape)}")
        output = torch.zeros_like(field, dtype=torch.complex64) if self.aperture_mode == "hard" else field.to(torch.complex64).clone()
        for aperture, phase_layer in zip(self.layout.expert_apertures, self.local_phases):
            local = field[:, aperture.y0:aperture.y1, aperture.x0:aperture.x1]
            output[:, aperture.y0:aperture.y1, aperture.x0:aperture.x1] = phase_layer(local)
        return output

    def get_phase_wrapped(self) -> torch.Tensor:
        return torch.stack([layer.get_phase_wrapped() for layer in self.local_phases], dim=0)

    def set_phase_dropout_active(self, active: bool) -> None:
        for layer in self.local_phases:
            layer.set_phase_dropout_active(active)


class GlobalFCPhaseMask(nn.Module):
    def __init__(
        self,
        canvas_shape,
        phase_size=None,
        phase_mode: str = "center_window",
        padding_mode: str = "transparent",
        phase_param: str = "unconstrained",
        phase_init: str = "identity",
        init_std: float = 0.02,
        phase_dropout_mode: str = "none",
        phase_dropout_p: float = 0.0,
        phase_dropout_block_size: int = 8,
        phase_dropout_batch_shared: bool = True,
    ) -> None:
        super().__init__()
        if isinstance(canvas_shape, int):
            canvas_shape = (canvas_shape, canvas_shape)
        self.canvas_shape = (int(canvas_shape[0]), int(canvas_shape[1]))
        self.phase_mode = str(phase_mode or "center_window")
        self.padding_mode = str(padding_mode or "transparent")
        if self.phase_mode not in {"center_window", "full_canvas"}:
            raise ValueError("global FC phase_mode must be center_window or full_canvas.")
        if self.padding_mode != "transparent":
            raise ValueError("Only transparent global FC padding is currently supported.")
        if self.phase_mode == "full_canvas":
            self.phase_size = self.canvas_shape
            grid_size = self.canvas_shape
            self.y0, self.y1 = 0, self.canvas_shape[0]
            self.x0, self.x1 = 0, self.canvas_shape[1]
        else:
            if phase_size is None:
                phase_size = min(self.canvas_shape)
            if isinstance(phase_size, (tuple, list)):
                height, width = int(phase_size[0]), int(phase_size[1])
            else:
                height = width = int(phase_size)
            if height > self.canvas_shape[0] or width > self.canvas_shape[1]:
                raise ValueError("global FC phase_size cannot exceed canvas_shape.")
            self.phase_size = (height, width)
            self.y0 = (self.canvas_shape[0] - height) // 2
            self.y1 = self.y0 + height
            self.x0 = (self.canvas_shape[1] - width) // 2
            self.x1 = self.x0 + width
            grid_size = self.phase_size
        self.phase = PhaseLayer(
            grid_size,
            parameterization=phase_param,
            init=phase_init,
            init_std=init_std,
            phase_dropout_mode=phase_dropout_mode,
            phase_dropout_p=phase_dropout_p,
            phase_dropout_block_size=phase_dropout_block_size,
            phase_dropout_batch_shared=phase_dropout_batch_shared,
        )

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        field = field.to(torch.complex64)
        if self.phase_mode == "full_canvas":
            return self.phase(field)
        local = field[:, self.y0:self.y1, self.x0:self.x1]
        local_out = self.phase(local)
        embedded = torch.zeros_like(field, dtype=torch.complex64)
        embedded[:, self.y0:self.y1, self.x0:self.x1] = local_out
        outside = torch.ones(field.shape[-2:], dtype=torch.float32, device=field.device)
        outside[self.y0:self.y1, self.x0:self.x1] = 0.0
        return field * outside.unsqueeze(0).to(torch.complex64) + embedded

    def get_phase_wrapped(self) -> torch.Tensor:
        return self.phase.get_phase_wrapped()

    def get_phase_canvas_wrapped(self) -> torch.Tensor:
        phase = torch.zeros(self.canvas_shape, dtype=self.get_phase_wrapped().dtype, device=self.get_phase_wrapped().device)
        phase[self.y0:self.y1, self.x0:self.x1] = self.get_phase_wrapped()
        return phase

    def phase_region(self):
        return [int(self.y0), int(self.y1), int(self.x0), int(self.x1)]

    def trainable_parameter_count(self) -> int:
        return int(self.phase.raw_phase.numel())

    def set_phase_dropout_active(self, active: bool) -> None:
        self.phase.set_phase_dropout_active(active)
