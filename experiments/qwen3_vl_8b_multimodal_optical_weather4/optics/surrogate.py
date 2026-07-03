from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import torch
from torch import nn
from torch.nn import functional as F

from .optical_group import OpticalGroup


class OpticalVisionBlockSurrogate(nn.Module):
    """Drop-in replacement for one packed Qwen3-VL vision transformer block."""

    def __init__(
        self,
        hidden_size: int,
        optical_dim: int,
        optical_layers: int,
        optical_field_size: int,
        optical_padding_size: int,
        wavelength_nm: float,
        pixel_pitch_um: float,
        mask_distance_cm: float,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.optical_dim = int(optical_dim)
        self.norm = nn.LayerNorm(hidden_size)
        self.input_adapter = nn.Linear(hidden_size, optical_dim)
        self.optical_group = OpticalGroup(
            optical_layers,
            optical_field_size,
            optical_padding_size,
            wavelength_nm,
            pixel_pitch_um,
            mask_distance_cm,
        )
        self.output_adapter = nn.Linear(optical_dim, hidden_size)
        self.last_input: torch.Tensor | None = None
        self.last_output: torch.Tensor | None = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        **_: Any,
    ) -> torch.Tensor:
        self.last_input = hidden_states
        original_shape = hidden_states.shape
        packed, lengths = _pack_hidden_states(hidden_states, cu_seqlens)
        amplitude_tokens = F.relu(self.input_adapter(self.norm(packed)))
        token_groups = list(amplitude_tokens.split(lengths, dim=0))
        fields = torch.stack([_tokens_to_field(group, self.optical_group.field_size) for group in token_groups])
        detected = self.optical_group(fields)
        optical_tokens = torch.cat(
            [
                _field_to_tokens(field, length, self.optical_dim)
                for field, length in zip(detected, lengths)
            ],
            dim=0,
        )
        output = packed + self.output_adapter(optical_tokens)
        if len(original_shape) == 3:
            output = output.reshape(original_shape)
        self.last_output = output
        return output


def _pack_hidden_states(
    hidden_states: torch.Tensor, cu_seqlens: torch.Tensor | None
) -> tuple[torch.Tensor, list[int]]:
    if hidden_states.ndim == 3:
        batch, tokens, hidden = hidden_states.shape
        return hidden_states.reshape(batch * tokens, hidden), [tokens] * batch
    if hidden_states.ndim != 2:
        raise ValueError(f"Expected packed 2D or batched 3D hidden states, got {hidden_states.shape}")
    if cu_seqlens is None:
        return hidden_states, [hidden_states.shape[0]]
    boundaries = cu_seqlens.detach().to(device="cpu", dtype=torch.long).tolist()
    lengths = [right - left for left, right in zip(boundaries[:-1], boundaries[1:])]
    if sum(lengths) != hidden_states.shape[0]:
        raise ValueError("cu_seqlens do not match packed vision token count")
    return hidden_states, lengths


def _tokens_to_field(tokens: torch.Tensor, field_size: int) -> torch.Tensor:
    field = tokens.unsqueeze(0).unsqueeze(0)
    return F.interpolate(
        field,
        size=(field_size, field_size),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0).squeeze(0)


def _field_to_tokens(field: torch.Tensor, token_count: int, optical_dim: int) -> torch.Tensor:
    tokens = F.interpolate(
        field.unsqueeze(0).unsqueeze(0),
        size=(token_count, optical_dim),
        mode="bilinear",
        align_corners=False,
    )
    return tokens.squeeze(0).squeeze(0)


@dataclass
class TeacherBlockCapture:
    input_hidden: torch.Tensor | None = None
    output_hidden: torch.Tensor | None = None

    def clear(self) -> None:
        self.input_hidden = None
        self.output_hidden = None


class VisionBlockReplacement:
    """Switch one loaded Qwen model between electronic teacher and optical student."""

    def __init__(
        self,
        model: nn.Module,
        block_index: int,
        surrogate: OpticalVisionBlockSurrogate,
    ) -> None:
        self.visual = _locate_visual_model(model)
        self.blocks = self.visual.blocks
        if not 0 <= block_index < len(self.blocks):
            raise IndexError(
                f"Vision block index {block_index} is outside [0, {len(self.blocks) - 1}]"
            )
        self.block_index = int(block_index)
        self.original_block = self.blocks[self.block_index]
        self.surrogate = surrogate
        self.capture = TeacherBlockCapture()
        self._pre_handle = self.original_block.register_forward_pre_hook(self._capture_input)
        self._post_handle = self.original_block.register_forward_hook(self._capture_output)

    def use_teacher(self) -> None:
        self.blocks[self.block_index] = self.original_block

    def use_student(self) -> None:
        self.blocks[self.block_index] = self.surrogate

    def trainable_parameters(self) -> Iterable[nn.Parameter]:
        return self.surrogate.parameters()

    def close(self) -> None:
        self.use_teacher()
        self._pre_handle.remove()
        self._post_handle.remove()

    def _capture_input(self, _module: nn.Module, args: tuple[Any, ...]) -> None:
        self.capture.input_hidden = args[0].detach()

    def _capture_output(
        self, _module: nn.Module, _args: tuple[Any, ...], output: torch.Tensor
    ) -> None:
        self.capture.output_hidden = output.detach()


def _locate_visual_model(model: nn.Module) -> nn.Module:
    visual = getattr(model, "visual", None)
    if visual is not None and hasattr(visual, "blocks"):
        return visual
    core = getattr(model, "model", None)
    visual = getattr(core, "visual", None)
    if visual is None or not hasattr(visual, "blocks"):
        raise RuntimeError("Unable to locate Qwen3-VL visual.blocks")
    return visual

