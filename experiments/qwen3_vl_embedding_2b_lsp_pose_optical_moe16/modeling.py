from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from experiments.qwen3_vl_embedding_2b_fss1000_vision_optical_saliency.modeling import (
    LoadedVisionBackbone,
    load_vision_backbone,
    preprocess_vision,
    restore_detector_spatial,
    restore_packed_spatial,
)
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.optics.moe import (
    HomogeneousMoEOpticalCore,
    lengths_from_cu,
)


NUM_JOINTS = 14


class LightweightPoseHead(nn.Module):
    """A shared lightweight spatial decoder for electronic and optical features."""

    def __init__(
        self,
        input_dim: int,
        projection_dim: int = 128,
        decoder_channels: tuple[int, ...] = (128, 64),
        groupnorm_groups: int = 8,
        heatmap_size: int = 56,
        num_joints: int = NUM_JOINTS,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.heatmap_size = int(heatmap_size)
        self.num_joints = int(num_joints)
        self.token_norm = nn.LayerNorm(self.input_dim)
        self.token_projection = nn.Linear(self.input_dim, int(projection_dim))
        layers: list[nn.Module] = []
        current = int(projection_dim)
        for channels in decoder_channels:
            channels = int(channels)
            groups = min(int(groupnorm_groups), channels)
            while channels % groups:
                groups -= 1
            layers.extend([
                nn.Conv2d(current, channels, 3, padding=1, bias=False),
                nn.GroupNorm(groups, channels),
                nn.GELU(),
            ])
            current = channels
        self.decoder = nn.Sequential(*layers)
        self.predictor = nn.Conv2d(current, self.num_joints, 1)

    def forward(self, spatial_features: torch.Tensor) -> torch.Tensor:
        if spatial_features.ndim != 4 or spatial_features.shape[1] != self.input_dim:
            raise RuntimeError(
                f"Pose head expects [B,{self.input_dim},H,W], got {tuple(spatial_features.shape)}"
            )
        tokens = spatial_features.permute(0, 2, 3, 1)
        projected = self.token_projection(self.token_norm(tokens.float()))
        decoded = self.decoder(projected.permute(0, 3, 1, 2))
        decoded = F.interpolate(
            decoded, size=(self.heatmap_size, self.heatmap_size),
            mode="bilinear", align_corners=False,
        )
        heatmaps = self.predictor(decoded)
        expected = (
            spatial_features.shape[0], self.num_joints,
            self.heatmap_size, self.heatmap_size,
        )
        if tuple(heatmaps.shape) != expected:
            raise RuntimeError(f"Pose heatmap shape {tuple(heatmaps.shape)} != {expected}")
        return heatmaps

    def specification(self) -> dict[str, Any]:
        return {
            "type": "lightweight_qwen_spatial_pose_head",
            "input_dim": self.input_dim,
            "heatmap_size": self.heatmap_size,
            "num_joints": self.num_joints,
            "parameters": sum(p.numel() for p in self.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad),
        }


class FrozenQwenVisionPoseTeacher(nn.Module):
    """Frozen native Qwen Vision plus a trainable 14-joint heatmap head."""

    def __init__(self, loaded: LoadedVisionBackbone, head: LightweightPoseHead) -> None:
        super().__init__()
        self.visual = loaded.visual
        self.head = head
        self._captured: torch.Tensor | None = None
        if not hasattr(self.visual, "blocks") or not len(self.visual.blocks):
            raise RuntimeError("Qwen visual module has no native blocks")
        self._hook = self.visual.blocks[-1].register_forward_hook(self._capture)

    def _capture(self, _module: nn.Module, _inputs: Any, output: Any) -> None:
        hidden = output[0] if isinstance(output, tuple) else output
        if not torch.is_tensor(hidden) or hidden.ndim != 2:
            raise RuntimeError("Final Qwen Vision block must return packed [tokens,hidden]")
        self._captured = hidden

    def train(self, mode: bool = True) -> "FrozenQwenVisionPoseTeacher":
        super().train(mode)
        self.visual.eval()
        self.head.train(mode)
        return self

    def forward(
        self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.visual.eval()
        with torch.no_grad():
            self._captured = None
            dtype = next(self.visual.patch_embed.parameters()).dtype
            self.visual(pixel_values.to(dtype), grid_thw=image_grid_thw)
            if self._captured is None:
                raise RuntimeError("Failed to capture final native Qwen Vision hidden")
            packed = self._captured.detach()
        spatial = restore_packed_spatial(packed, image_grid_thw)
        return self.head(spatial), spatial

    def close(self) -> None:
        self._hook.remove()


class _VisionBypass(nn.Module):
    def forward(self, hidden_states: torch.Tensor, **_: Any) -> torch.Tensor:
        return hidden_states


class _OpticalCaptureBlock(nn.Module):
    """Replace all native Vision blocks while retaining only patch/position input."""

    def __init__(self, core: HomogeneousMoEOpticalCore) -> None:
        super().__init__()
        self.core = core
        self.token_counts: list[int] = []

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        **_: Any,
    ) -> torch.Tensor:
        lengths = lengths_from_cu(hidden_states, cu_seqlens)
        self.token_counts = lengths
        fields = self.core.encode_groups(list(hidden_states.split(lengths)))
        field, routing = self.core.begin(fields)
        for index in range(len(self.core.expert_layers)):
            field = self.core.run_stage(index, field, routing)
        # One global phase plane then 10 cm propagation to the square-law CCD.
        field = self.core.propagator(self.core.global_phase(field))
        readout, intensity = self.core.readout(field)
        if not torch.isfinite(readout).all() or torch.any(readout < 0):
            raise RuntimeError("Vision CCD readout must be finite and nonnegative")
        self.core.current_detector_readout = readout
        if self.core.capture_intermediate_fields:
            count = min(self.core.capture_sample_count, len(field))
            self.core.last_detector_complex_field = field[:count].detach().cpu()
            self.core.last_detector_intensity = intensity[:count].detach().cpu()
            self.core.last_detector_readout = readout[:count].detach().cpu()
        # Native blocks are all bypassed; this return exists only to let the
        # frozen Qwen visual wrapper finish its bookkeeping.
        return hidden_states


class VisionOpticalPoseStudent(nn.Module):
    """Frozen Qwen patch stem -> single-layer MoE16 -> global phase/CCD -> pose head."""

    def __init__(
        self,
        loaded: LoadedVisionBackbone,
        settings: Any,
        head: LightweightPoseHead,
    ) -> None:
        super().__init__()
        self.visual = loaded.visual
        self.original_blocks = list(self.visual.blocks)
        self.core = HomogeneousMoEOpticalCore(
            settings.vision_hidden_size, settings.max_visual_tokens, settings,
        ).to(loaded.device)
        self.core.output_adapter.requires_grad_(False)
        self.capture_block = _OpticalCaptureBlock(self.core)
        self.student_blocks = nn.ModuleList(
            [self.capture_block] + [_VisionBypass() for _ in self.original_blocks[1:]]
        )
        self.head = head
        self._active = False

    def activate(self) -> None:
        for index, block in enumerate(self.student_blocks):
            self.visual.blocks[index] = block
        self._active = True

    def restore_native(self) -> None:
        for index, block in enumerate(self.original_blocks):
            self.visual.blocks[index] = block
        self._active = False

    def train(self, mode: bool = True) -> "VisionOpticalPoseStudent":
        super().train(mode)
        self.visual.eval()
        self.core.train(mode)
        self.head.train(mode)
        return self

    def forward(
        self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self._active:
            self.activate()
        dtype = next(self.visual.patch_embed.parameters()).dtype
        self.visual(pixel_values.to(dtype), grid_thw=image_grid_thw)
        detector = self.core.current_detector_readout
        if detector is None:
            raise RuntimeError("Optical Vision path did not produce a CCD readout")
        spatial = restore_detector_spatial(
            detector, image_grid_thw, self.capture_block.token_counts,
        )
        return self.head(spatial), spatial, detector

    def router_losses(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.core.router_losses()


def build_teacher(
    loaded: LoadedVisionBackbone, settings: Any,
) -> FrozenQwenVisionPoseTeacher:
    head = LightweightPoseHead(
        settings.vision_hidden_size,
        settings.pose_projection_dim,
        settings.pose_decoder_channels,
        settings.pose_groupnorm_groups,
        settings.heatmap_size,
    ).to(loaded.device)
    return FrozenQwenVisionPoseTeacher(loaded, head)


def build_student(
    loaded: LoadedVisionBackbone, settings: Any,
) -> VisionOpticalPoseStudent:
    loaded.model.requires_grad_(False)
    head = LightweightPoseHead(
        settings.detector_output_size,
        settings.pose_projection_dim,
        settings.pose_decoder_channels,
        settings.pose_groupnorm_groups,
        settings.heatmap_size,
    ).to(loaded.device)
    student = VisionOpticalPoseStudent(loaded, settings, head)
    student.core.requires_grad_(True)
    student.core.output_adapter.requires_grad_(False)
    student.head.requires_grad_(True)
    return student


def trainable_parameter_report(module: nn.Module, prefix: str) -> dict[str, Any]:
    rows, seen = [], set()
    for name, parameter in module.named_parameters():
        if parameter.requires_grad and id(parameter) not in seen:
            rows.append({
                "name": f"{prefix}.{name}",
                "shape": list(parameter.shape),
                "parameters": parameter.numel(),
            })
            seen.add(id(parameter))
    return {
        "trainable_parameters": sum(row["parameters"] for row in rows),
        "trainable_tensors": len(rows),
        "trainable_parameter_list": rows,
    }


__all__ = [
    "LoadedVisionBackbone", "LightweightPoseHead", "FrozenQwenVisionPoseTeacher",
    "VisionOpticalPoseStudent", "load_vision_backbone", "preprocess_vision",
    "restore_packed_spatial", "restore_detector_spatial", "build_teacher",
    "build_student", "trainable_parameter_report",
]
