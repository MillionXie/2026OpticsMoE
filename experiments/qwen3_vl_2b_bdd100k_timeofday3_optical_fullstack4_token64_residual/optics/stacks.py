from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .conversion import OpticalConversion


def _lengths_from_cu(hidden: torch.Tensor, cu_seqlens: torch.Tensor | None) -> list[int]:
    if hidden.ndim != 2:
        raise ValueError(f"Packed vision hidden must be 2D, got {tuple(hidden.shape)}")
    if cu_seqlens is None:
        raise RuntimeError("Packed vision hidden requires cu_seqlens; batch tokens cannot share one optical field")
    boundaries = cu_seqlens.detach().cpu().long().tolist()
    lengths = [b - a for a, b in zip(boundaries[:-1], boundaries[1:])]
    if not lengths or sum(lengths) != hidden.shape[0] or any(length <= 0 for length in lengths):
        raise ValueError("cu_seqlens do not match packed vision tokens")
    return lengths


class _OpticalStackBase(nn.Module):
    """Project token rows directly onto a square optical field without resizing."""

    token_kind = "token count"
    overflow_advice = "Reduce the input sequence length."

    def __init__(
        self,
        hidden_size: int,
        optical_dim: int,
        conversions: int,
        field_size: int,
        padding_size: int,
        wavelength_nm: float,
        pixel_pitch_um: float,
        distance_cm: float,
        amplitude_mask_enabled: bool,
        phase_init: str = "zeros",
        phase_init_std: float = 0.02,
        residual_enabled: bool = True,
        identity_scale_init: float = 1.0,
        modulated_scale_init: float = 0.1,
        identity_scale_trainable: bool = False,
        modulated_scale_trainable: bool = True,
    ) -> None:
        super().__init__()
        if conversions != 4:
            raise ValueError("fullstack4 requires four optical conversions")
        if optical_dim != field_size:
            raise ValueError(
                "Direct token-row optical mapping requires optical_dim == optical_field_size "
                f"(got optical_dim={optical_dim}, optical_field_size={field_size})"
            )
        self.hidden_size = int(hidden_size)
        self.optical_dim = int(optical_dim)
        self.field_size = int(field_size)
        self.residual_enabled = bool(residual_enabled)
        self.input_adapter = nn.Linear(hidden_size, optical_dim)
        self.adapter_norm = nn.LayerNorm(optical_dim)
        self.nonnegative = nn.Softplus()
        self.conversions = nn.ModuleList(
            [
                OpticalConversion(
                    field_size,
                    padding_size,
                    wavelength_nm,
                    pixel_pitch_um,
                    distance_cm,
                    amplitude_mask_enabled,
                    phase_init,
                    phase_init_std,
                )
                for _ in range(conversions)
            ]
        )
        self.output_adapter = nn.Linear(optical_dim, hidden_size)
        self._make_scale("identity_scale", identity_scale_init, identity_scale_trainable)
        self._make_scale("modulated_scale", modulated_scale_init, modulated_scale_trainable)
        self.last_input_fields: torch.Tensor | None = None
        self.last_fields: list[torch.Tensor] = []
        self.last_output: torch.Tensor | None = None
        self.last_delta: torch.Tensor | None = None
        self.last_projected_tokens: list[torch.Tensor] = []
        self._last_delta_groups: list[torch.Tensor] = []
        self.last_token_counts: list[int] = []

    def _make_scale(self, name: str, init: float, trainable: bool) -> None:
        value = torch.tensor(float(init), dtype=torch.float32)
        if trainable:
            setattr(self, name, nn.Parameter(value))
        else:
            self.register_buffer(name, value)

    def scale_values(self) -> dict[str, float]:
        return {
            "identity_scale": float(self.identity_scale.detach().cpu()),
            "modulated_scale": float(self.modulated_scale.detach().cpu()),
        }

    def _check_token_count(self, token_count: int) -> None:
        if token_count > self.field_size:
            raise RuntimeError(
                f"{self.token_kind} {token_count} exceeds optical_field_size={self.field_size}. "
                f"{self.overflow_advice} No crop, truncation, pooling, or fallback resize is allowed."
            )

    def _combine_residual(self, input_hidden: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        if not self.residual_enabled:
            return delta
        boundary_dtype = input_hidden.dtype
        output = self.identity_scale.float() * input_hidden.float()
        output = output + self.modulated_scale.float() * delta.float()
        return output.to(boundary_dtype)

    def _convert_groups(self, groups: list[torch.Tensor], boundary_dtype: torch.dtype) -> list[torch.Tensor]:
        fields: list[torch.Tensor] = []
        self.last_projected_tokens = []
        for group in groups:
            self._check_token_count(len(group))
            projected = self.input_adapter(group.float())
            projected = self.nonnegative(self.adapter_norm(projected))
            self.last_projected_tokens.append(projected)
            field = projected.new_zeros((self.field_size, self.field_size))
            field[: len(group), :] = projected
            fields.append(field)
        stacked_fields = torch.stack(fields)
        self.last_input_fields = stacked_fields
        self.last_fields = []
        for conversion in self.conversions:
            stacked_fields = conversion(stacked_fields)
            self.last_fields.append(stacked_fields)
        outputs: list[torch.Tensor] = []
        self._last_delta_groups = []
        for field, group in zip(stacked_fields, groups):
            valid_rows = field[: len(group), :]
            delta = self.output_adapter(valid_rows).to(boundary_dtype)
            self._last_delta_groups.append(delta)
            outputs.append(self._combine_residual(group, delta))
        return outputs


class VisionOpticalStackSurrogate(_OpticalStackBase):
    """Replace the complete packed vision transformer stack with optical4."""

    token_kind = "visual token count"
    overflow_advice = "Lower processor_max_pixels so each image produces fewer pre-merge visual tokens."

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        **_: Any,
    ) -> torch.Tensor:
        lengths = _lengths_from_cu(hidden_states, cu_seqlens)
        self.last_token_counts = lengths
        groups = list(hidden_states.split(lengths, dim=0))
        output = torch.cat(self._convert_groups(groups, hidden_states.dtype), dim=0)
        self.last_delta = torch.cat(self._last_delta_groups, dim=0)
        self.last_output = output
        return output


class LanguageOpticalStackSurrogate(_OpticalStackBase):
    """Drop-in first decoder layer replacing the complete language stack."""

    token_kind = "language sequence length"
    overflow_advice = (
        "Shorten the classification prompt or lower the visual token budget / processor_max_pixels."
    )

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._valid_token_masks: torch.Tensor | None = None
        self.return_tuple = False

    def set_attention_mask(self, attention_mask: torch.Tensor) -> None:
        if attention_mask.ndim != 2:
            raise ValueError("Original 2D attention_mask is required for language optical boundaries")
        self._valid_token_masks = attention_mask.detach().bool()

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None, **_: Any):
        if hidden_states.ndim != 3:
            raise ValueError(f"Language hidden must be [B,S,D], got {tuple(hidden_states.shape)}")
        batch, sequence, _ = hidden_states.shape
        masks = self._valid_token_masks
        if masks is None and attention_mask is not None and attention_mask.ndim == 2:
            masks = attention_mask.detach().bool()
        if masks is None:
            masks = torch.ones(batch, sequence, dtype=torch.bool, device=hidden_states.device)
        masks = masks.to(hidden_states.device)
        lengths = masks.sum(dim=1).detach().cpu().long().tolist()
        if len(lengths) != batch or any(length <= 0 or length > sequence for length in lengths):
            raise RuntimeError("Language sequence boundaries do not match the current batch")
        self.last_token_counts = list(lengths)
        groups = [hidden_states[index, masks[index]] for index in range(batch)]
        converted = self._convert_groups(groups, hidden_states.dtype)
        output = hidden_states.new_zeros(hidden_states.shape)
        delta_output = hidden_states.new_zeros(hidden_states.shape)
        for index, value in enumerate(converted):
            output[index, masks[index]] = value
            delta_output[index, masks[index]] = self._last_delta_groups[index]
        self.last_delta = delta_output
        self.last_output = output
        return (output,) if self.return_tuple else output


class VisionBypass(nn.Module):
    def forward(self, hidden_states: torch.Tensor, **_: Any) -> torch.Tensor:
        return hidden_states


class LanguageBypass(nn.Module):
    def __init__(self, return_tuple: bool = False) -> None:
        super().__init__()
        self.return_tuple = return_tuple

    def forward(self, hidden_states: torch.Tensor, **_: Any):
        return (hidden_states,) if self.return_tuple else hidden_states
