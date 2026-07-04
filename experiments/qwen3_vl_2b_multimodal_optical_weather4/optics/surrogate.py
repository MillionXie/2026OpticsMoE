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
        self.last_token_counts: list[int] | None = None
        self.last_field_shapes: list[list[int]] | None = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        **_: Any,
    ) -> torch.Tensor:
        self.last_input = hidden_states
        boundary_dtype = hidden_states.dtype
        original_shape = hidden_states.shape
        packed, lengths = _pack_hidden_states(hidden_states, cu_seqlens)
        self.last_token_counts = list(lengths)
        # FFT-based optical propagation requires complex64/complex128. PyTorch
        # cannot construct a complex tensor from BF16, so the trainable optical
        # surrogate deliberately forms an FP32 numerical island inside a BF16
        # Qwen backbone and casts only its block output back at the boundary.
        packed = packed.to(dtype=self.norm.weight.dtype)
        amplitude_tokens = F.relu(self.input_adapter(self.norm(packed)))
        token_groups = list(amplitude_tokens.split(lengths, dim=0))
        fields = torch.stack([_tokens_to_field(group, self.optical_group.field_size) for group in token_groups])
        self.last_field_shapes = [list(field.shape) for field in fields]
        detected = self.optical_group(fields)
        optical_tokens = torch.cat(
            [
                _field_to_tokens(field, length, self.optical_dim)
                for field, length in zip(detected, lengths)
            ],
            dim=0,
        )
        # No electronic residual bypass is permitted: deployed optical hardware
        # only exposes the detected/read-out optical path.
        output = self.output_adapter(optical_tokens).to(boundary_dtype)
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
        raise RuntimeError(
            "Packed vision hidden states require cu_seqlens; refusing to merge a batch into one optical field"
        )
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
    block_start: int
    block_end: int
    input_hidden: torch.Tensor | None = None
    output_hidden: torch.Tensor | None = None
    cu_seqlens: torch.Tensor | None = None

    def clear(self) -> None:
        self.input_hidden = None
        self.output_hidden = None
        self.cu_seqlens = None


class VisionBlockBypass(nn.Module):
    """Preserve Qwen's block-list indices while removing an electronic block."""

    def forward(self, hidden_states: torch.Tensor, **_: Any) -> torch.Tensor:
        return hidden_states


class VisionBlockReplacement:
    """Switch grouped electronic teacher blocks to optical student conversions.

    Each inclusive ``(start, end)`` group is represented by one optical
    surrogate at the group's entrance. Later indices in that group become
    identity placeholders. Keeping the original block-list length preserves
    Qwen's index-dependent visual/deep-stack control flow while ensuring
    deep-stack taps observe optically transformed features.
    """

    def __init__(
        self,
        model: nn.Module,
        block_groups: int | list[tuple[int, int]],
        surrogates: OpticalVisionBlockSurrogate | list[OpticalVisionBlockSurrogate],
    ) -> None:
        self.visual = _locate_visual_model(model)
        self.blocks = self.visual.blocks
        if isinstance(block_groups, int):
            block_groups = [(block_groups, block_groups)]
        if isinstance(surrogates, OpticalVisionBlockSurrogate):
            surrogates = [surrogates]
        self.block_groups = [(int(start), int(end)) for start, end in block_groups]
        self.surrogates = list(surrogates)
        if len(self.block_groups) != len(self.surrogates):
            raise ValueError("Every teacher block group requires one optical surrogate")
        _validate_block_groups(self.block_groups, len(self.blocks))
        replaced_indices = [
            index
            for start, end in self.block_groups
            for index in range(start, end + 1)
        ]
        self.original_blocks = {index: self.blocks[index] for index in replaced_indices}
        self.bypasses = {
            index: VisionBlockBypass()
            for start, end in self.block_groups
            for index in range(start + 1, end + 1)
        }
        self.captures = [
            TeacherBlockCapture(start, end) for start, end in self.block_groups
        ]
        self._handles: list[Any] = []
        for capture in self.captures:
            self._handles.append(
                self.original_blocks[capture.block_start].register_forward_pre_hook(
                    _capture_input_for(capture), with_kwargs=True
                )
            )
            self._handles.append(
                self.original_blocks[capture.block_end].register_forward_hook(
                    _capture_output_for(capture)
                )
            )

    @property
    def replaced_block_indices(self) -> list[int]:
        return [
            index
            for start, end in self.block_groups
            for index in range(start, end + 1)
        ]

    def use_teacher(self) -> None:
        for index, original in self.original_blocks.items():
            self.blocks[index] = original

    def use_student(self) -> None:
        for (start, end), surrogate in zip(self.block_groups, self.surrogates):
            self.blocks[start] = surrogate
            for index in range(start + 1, end + 1):
                self.blocks[index] = self.bypasses[index]

    def trainable_parameters(self) -> Iterable[nn.Parameter]:
        for surrogate in self.surrogates:
            yield from surrogate.parameters()

    def clear_captures(self) -> None:
        for capture in self.captures:
            capture.clear()

    def teacher_outputs(self) -> list[torch.Tensor]:
        outputs = [capture.output_hidden for capture in self.captures]
        if any(output is None for output in outputs):
            missing = [
                list(self.block_groups[index])
                for index, output in enumerate(outputs)
                if output is None
            ]
            raise RuntimeError(f"Teacher hooks did not capture block groups: {missing}")
        return [output for output in outputs if output is not None]

    def teacher_inputs(self) -> list[torch.Tensor]:
        inputs = [capture.input_hidden for capture in self.captures]
        if any(value is None for value in inputs):
            raise RuntimeError("Teacher hooks did not capture all group inputs")
        return [value for value in inputs if value is not None]

    def teacher_token_counts(self) -> list[int]:
        boundaries = self.captures[0].cu_seqlens
        if boundaries is None:
            raise RuntimeError(
                "Teacher block call did not expose cu_seqlens; per-image cache boundaries are unavailable"
            )
        values = boundaries.detach().to(device="cpu", dtype=torch.long).tolist()
        counts = [right - left for left, right in zip(values[:-1], values[1:])]
        for capture in self.captures[1:]:
            current = capture.cu_seqlens
            if current is None or current.detach().cpu().tolist() != boundaries.detach().cpu().tolist():
                raise RuntimeError("Teacher block groups reported inconsistent per-image boundaries")
        return counts

    def student_outputs(self) -> list[torch.Tensor]:
        outputs = [surrogate.last_output for surrogate in self.surrogates]
        if any(output is None for output in outputs):
            missing = [
                list(self.block_groups[index])
                for index, output in enumerate(outputs)
                if output is None
            ]
            raise RuntimeError(f"Optical surrogates did not expose group outputs: {missing}")
        return [output for output in outputs if output is not None]

    def set_surrogates_trainable(self, trainable: bool) -> None:
        for surrogate in self.surrogates:
            surrogate.requires_grad_(trainable)

    def train(self, mode: bool = True) -> None:
        for surrogate in self.surrogates:
            surrogate.train(mode)

    def eval(self) -> None:
        self.train(False)

    def cpu_state_dicts(self) -> list[dict[str, torch.Tensor]]:
        return [
            {key: value.detach().cpu().clone() for key, value in surrogate.state_dict().items()}
            for surrogate in self.surrogates
        ]

    def load_state_dicts(self, states: list[dict[str, torch.Tensor]]) -> None:
        if len(states) != len(self.surrogates):
            raise ValueError(
                f"Optical checkpoint has {len(states)} conversions; expected {len(self.surrogates)}"
            )
        for surrogate, state in zip(self.surrogates, states):
            surrogate.load_state_dict(state)

    def close(self) -> None:
        self.use_teacher()
        for handle in self._handles:
            handle.remove()


def _capture_input_for(capture: TeacherBlockCapture):
    def hook(
        _module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> None:
        capture.input_hidden = args[0].detach()
        cu_seqlens = kwargs.get("cu_seqlens")
        if cu_seqlens is None and len(args) > 1 and torch.is_tensor(args[1]):
            cu_seqlens = args[1]
        if cu_seqlens is not None:
            capture.cu_seqlens = cu_seqlens.detach()

    return hook


def _capture_output_for(capture: TeacherBlockCapture):
    def hook(
        _module: nn.Module, _args: tuple[Any, ...], output: torch.Tensor
    ) -> None:
        capture.output_hidden = output.detach()

    return hook


def _validate_block_groups(groups: list[tuple[int, int]], block_count: int) -> None:
    previous_end = -1
    for start, end in groups:
        if start > end:
            raise ValueError(f"Invalid teacher block group: {(start, end)}")
        if start <= previous_end:
            raise ValueError("Teacher block groups must be ordered and non-overlapping")
        if start < 0 or end >= block_count:
            raise IndexError(
                f"Teacher block group {(start, end)} is outside [0, {block_count - 1}]"
            )
        previous_end = end


def _locate_visual_model(model: nn.Module) -> nn.Module:
    visual = getattr(model, "visual", None)
    if visual is not None and hasattr(visual, "blocks"):
        return visual
    core = getattr(model, "model", None)
    visual = getattr(core, "visual", None)
    if visual is None or not hasattr(visual, "blocks"):
        raise RuntimeError("Unable to locate Qwen3-VL visual.blocks")
    return visual
