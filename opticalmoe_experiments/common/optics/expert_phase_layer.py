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
        phase_param: str = "unconstrained",
        phase_init: str = "identity",
        init_std: float = 0.02,
        phase_dropout_mode: str = "none",
        phase_dropout_p: float = 0.0,
        phase_dropout_block_size: int = 8,
        phase_dropout_batch_shared: bool = True,
    ) -> None:
        super().__init__()
        self.phase = PhaseLayer(
            canvas_shape,
            parameterization=phase_param,
            init=phase_init,
            init_std=init_std,
            phase_dropout_mode=phase_dropout_mode,
            phase_dropout_p=phase_dropout_p,
            phase_dropout_block_size=phase_dropout_block_size,
            phase_dropout_batch_shared=phase_dropout_batch_shared,
        )

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        return self.phase(field)

    def get_phase_wrapped(self) -> torch.Tensor:
        return self.phase.get_phase_wrapped()

    def set_phase_dropout_active(self, active: bool) -> None:
        self.phase.set_phase_dropout_active(active)

