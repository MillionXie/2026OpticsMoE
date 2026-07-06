from __future__ import annotations

from typing import Any
import torch
from torch import nn
from torch.nn import functional as F

from .conversion import OpticalConversion


def _tokens_to_field(tokens: torch.Tensor, field_size: int) -> torch.Tensor:
    return F.interpolate(tokens[None, None], size=(field_size, field_size), mode="bilinear", align_corners=False)[0, 0]


def _field_to_tokens(field: torch.Tensor, token_count: int, optical_dim: int) -> torch.Tensor:
    return F.interpolate(field[None, None], size=(token_count, optical_dim), mode="bilinear", align_corners=False)[0, 0]


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
    def __init__(self, hidden_size: int, optical_dim: int, conversions: int, field_size: int,
                 padding_size: int, wavelength_nm: float, pixel_pitch_um: float,
                 distance_cm: float, amplitude_mask_enabled: bool, phase_init: str = "zeros",
                 phase_init_std: float = 0.02) -> None:
        super().__init__()
        if conversions != 4:
            raise ValueError("fullstack4 requires four optical conversions")
        self.hidden_size = int(hidden_size); self.optical_dim = int(optical_dim); self.field_size = int(field_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.input_adapter = nn.Linear(hidden_size, optical_dim)
        self.conversions = nn.ModuleList([
            OpticalConversion(
                field_size, padding_size, wavelength_nm, pixel_pitch_um, distance_cm,
                amplitude_mask_enabled, phase_init, phase_init_std,
            )
            for _ in range(conversions)
        ])
        self.output_adapter = nn.Linear(optical_dim, hidden_size)
        self.last_fields: list[torch.Tensor] = []
        self.last_output: torch.Tensor | None = None
        self.last_token_counts: list[int] = []

    def _convert_groups(self, groups: list[torch.Tensor], boundary_dtype: torch.dtype) -> list[torch.Tensor]:
        projected = [F.relu(self.input_adapter(self.norm(group.float()))) for group in groups]
        fields = torch.stack([_tokens_to_field(group, self.field_size) for group in projected])
        self.last_fields = []
        for conversion in self.conversions:
            fields = conversion(fields)
            self.last_fields.append(fields)
        outputs = [self.output_adapter(_field_to_tokens(field, len(group), self.optical_dim)).to(boundary_dtype)
                   for field, group in zip(fields, groups)]
        return outputs


class VisionOpticalStackSurrogate(_OpticalStackBase):
    """Replace the complete packed vision transformer stack with optical4."""
    def forward(self, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor | None = None, **_: Any) -> torch.Tensor:
        lengths = _lengths_from_cu(hidden_states, cu_seqlens)
        self.last_token_counts = lengths
        groups = list(hidden_states.split(lengths, dim=0))
        output = torch.cat(self._convert_groups(groups, hidden_states.dtype), dim=0)
        self.last_output = output
        return output


class LanguageOpticalStackSurrogate(_OpticalStackBase):
    """Drop-in first decoder layer replacing the complete language stack."""
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
        for index, value in enumerate(converted):
            output[index, masks[index]] = value
        self.last_output = output
        return (output,) if self.return_tuple else output


class VisionBypass(nn.Module):
    def forward(self, hidden_states: torch.Tensor, **_: Any) -> torch.Tensor:
        return hidden_states


class LanguageBypass(nn.Module):
    def __init__(self, return_tuple: bool = False) -> None:
        super().__init__(); self.return_tuple = return_tuple
    def forward(self, hidden_states: torch.Tensor, **_: Any):
        return (hidden_states,) if self.return_tuple else hidden_states
