from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .conversion import OpticalConversion


class VisionOpticalStackSurrogate(nn.Module):
    """Checkpoint-compatible vision surrogate with a propagation-free encoder API."""

    def __init__(self, hidden_size: int, optical_dim: int, conversions: int, field_size: int,
                 padding_size: int, wavelength_nm: float, pixel_pitch_um: float,
                 distance_cm: float, amplitude_mask_enabled: bool, phase_init: str = "zeros",
                 phase_init_std: float = 0.02, residual_enabled: bool = True,
                 identity_scale_init: float = 1.0, modulated_scale_init: float = 0.1,
                 identity_scale_trainable: bool = False,
                 modulated_scale_trainable: bool = True) -> None:
        super().__init__()
        if optical_dim != field_size:
            raise ValueError("Direct token-row mapping requires optical_dim == field_size")
        self.hidden_size = int(hidden_size)
        self.optical_dim = int(optical_dim)
        self.field_size = int(field_size)
        self.residual_enabled = bool(residual_enabled)
        self.input_adapter = nn.Linear(hidden_size, optical_dim)
        self.adapter_norm = nn.LayerNorm(optical_dim)
        self.nonnegative = nn.Softplus()
        # These modules preserve source checkpoint compatibility. The probe API never calls them.
        self.conversions = nn.ModuleList([
            OpticalConversion(field_size, padding_size, wavelength_nm, pixel_pitch_um,
                              distance_cm, amplitude_mask_enabled, phase_init, phase_init_std)
            for _ in range(conversions)
        ])
        self.output_adapter = nn.Linear(optical_dim, hidden_size)
        self._make_scale("identity_scale", identity_scale_init, identity_scale_trainable)
        self._make_scale("modulated_scale", modulated_scale_init, modulated_scale_trainable)

    def _make_scale(self, name: str, initial: float, trainable: bool) -> None:
        value = torch.tensor(float(initial), dtype=torch.float32)
        if trainable:
            setattr(self, name, nn.Parameter(value))
        else:
            self.register_buffer(name, value)

    def _check_token_count(self, count: int) -> None:
        if count > self.field_size:
            raise RuntimeError(
                f"visual token count {count} exceeds optical_field_size={self.field_size}. "
                "Lower processor_max_pixels or rebuild the source experiment. "
                "No crop, truncation, pooling, or fallback resize is allowed."
            )

    def encode_groups_to_input_fields(self, groups: list[torch.Tensor]) -> torch.Tensor:
        """Run only input_adapter -> LayerNorm -> Softplus -> strict zero padding."""
        if not groups:
            raise ValueError("At least one vision token group is required")
        fields: list[torch.Tensor] = []
        for group in groups:
            if group.ndim != 2 or group.shape[1] != self.hidden_size:
                raise ValueError(
                    f"Expected [T,{self.hidden_size}] vision hidden, got {tuple(group.shape)}"
                )
            self._check_token_count(len(group))
            projected = self.nonnegative(self.adapter_norm(self.input_adapter(group.float())))
            field = projected.new_zeros((self.field_size, self.field_size))
            field[: len(group)] = projected
            fields.append(field)
        return torch.stack(fields)

    def forward(self, *_: Any, **__: Any) -> torch.Tensor:
        raise RuntimeError(
            "Vision-field probe must call encode_groups_to_input_fields(); optical propagation is disabled"
        )

