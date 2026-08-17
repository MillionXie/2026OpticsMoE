from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval.optical_blocks import (
    VisionTwoBlockOpticalCore,
)
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.optics.moe import (
    lengths_from_cu,
)


def restore_qwen_block_major_spatial(
    packed: torch.Tensor,
    image_grid_thw: torch.Tensor,
    *,
    merge_size: int = 2,
) -> torch.Tensor:
    """Restore Qwen's block-major patch order to true [B,C,H,W] order."""

    if packed.ndim != 2:
        raise RuntimeError(f"Packed features must be [sum(T),C], got {packed.shape}")
    if image_grid_thw.ndim != 2 or image_grid_thw.shape[1] != 3:
        raise RuntimeError("image_grid_thw must have shape [B,3]")
    rows: list[torch.Tensor] = []
    offset = 0
    for sample_index, (frames, height, width) in enumerate(
        image_grid_thw.detach().cpu().long().tolist()
    ):
        if frames != 1:
            raise RuntimeError(
                f"Dense image task requires one frame, sample {sample_index} has {frames}"
            )
        if height % merge_size or width % merge_size:
            raise RuntimeError(
                f"Qwen grid {(height, width)} is not divisible by merge_size={merge_size}"
            )
        count = int(frames * height * width)
        group = packed[offset : offset + count]
        if len(group) != count:
            raise RuntimeError("Packed feature count does not match image_grid_thw")
        grid = (
            group.view(
                frames,
                height // merge_size,
                width // merge_size,
                merge_size,
                merge_size,
                packed.shape[-1],
            )
            .permute(0, 5, 1, 3, 2, 4)
            .reshape(frames, packed.shape[-1], height, width)
        )
        rows.append(grid[0])
        offset += count
    if offset != packed.shape[0]:
        raise RuntimeError(
            f"Unused packed rows: consumed {offset}, provided {packed.shape[0]}"
        )
    if len({tuple(value.shape[-2:]) for value in rows}) != 1:
        raise RuntimeError("A dense batch cannot contain different spatial grids")
    return torch.stack(rows)


class _VisionBypass(nn.Module):
    def forward(self, hidden_states: torch.Tensor, **_: Any) -> torch.Tensor:
        return hidden_states


class DenseVision2Core(nn.Module):
    """Compatibility wrapper around the retrieval-tested Vision2 core."""

    def __init__(self, hidden_size: int, settings: Any) -> None:
        super().__init__()
        self.hybrid = VisionTwoBlockOpticalCore(
            hidden_size, settings.max_visual_tokens, settings
        )

    @property
    def optical_branch(self) -> nn.Module:
        return self.hybrid.optical_branch

    @property
    def router(self) -> nn.Module:
        return self.optical_branch.core.router

    @property
    def expert_layers(self) -> nn.Module:
        return self.optical_branch.core.expert_layers

    @property
    def global_phase(self) -> nn.Module:
        return self.optical_branch.core.global_phase

    @property
    def last_routing(self) -> dict[str, torch.Tensor]:
        return self.hybrid.last_routing

    @property
    def last_latent_groups(self) -> list[torch.Tensor]:
        return self.hybrid.last_latent_groups

    @property
    def last_input_fields(self) -> torch.Tensor | None:
        return self.optical_branch.core.last_input_fields

    @property
    def last_amplitude_slm_canvas(self) -> torch.Tensor | None:
        return self.optical_branch.core.last_amplitude_slm_canvas

    @property
    def last_stage_fields(self) -> list[torch.Tensor]:
        return self.optical_branch.core.last_stage_fields

    @property
    def last_detector_intensity(self) -> torch.Tensor | None:
        return self.optical_branch.core.last_detector_intensity

    @property
    def last_detector_readout(self) -> torch.Tensor | None:
        return self.optical_branch.core.last_detector_readout

    def forward_groups(
        self,
        groups: list[torch.Tensor],
        spatial_shapes: list[tuple[int, int, int]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.hybrid.forward_groups(
            groups, causal=False, spatial_shapes=spatial_shapes
        )

    def router_losses(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.optical_branch.core.router_losses()

    def operating_loss(self) -> torch.Tensor:
        value = self.optical_branch.current_operating_loss
        return self.hybrid.block2_optical_fusion_logit.new_zeros(()) if value is None else value

    def parameter_breakdown(self) -> dict[str, Any]:
        result = self.hybrid.parameter_breakdown()
        result["dense_output"] = "second_fusion_latent_192_before_qwen_merger"
        return result

    def set_phase_dropout_active(self, active: bool) -> None:
        self.optical_branch.set_phase_dropout_active(active)

    def set_intermediate_field_capture(
        self, enabled: bool, sample_count: int = 1
    ) -> None:
        self.optical_branch.core.set_intermediate_field_capture(enabled, sample_count)


class _HybridCaptureBlock(nn.Module):
    def __init__(self, core: DenseVision2Core) -> None:
        super().__init__()
        self.core = core
        self.spatial_shapes: list[tuple[int, int, int]] | None = None

    def set_grid(self, image_grid_thw: torch.Tensor) -> None:
        shapes = []
        for frames, height, width in image_grid_thw.detach().cpu().long().tolist():
            if frames != 1:
                raise RuntimeError("Dense Vision2 supports still images only")
            shapes.append((1, int(height), int(width)))
        self.spatial_shapes = shapes

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        **_: Any,
    ) -> torch.Tensor:
        if self.spatial_shapes is None:
            raise RuntimeError("image_grid_thw must be set before Vision forward")
        lengths = lengths_from_cu(hidden_states, cu_seqlens)
        if len(lengths) != len(self.spatial_shapes):
            raise RuntimeError("Qwen sequence boundaries do not match image grid batch")
        self.core.forward_groups(
            list(hidden_states.split(lengths)), self.spatial_shapes
        )
        # Native blocks and merger are irrelevant to the task head. Returning
        # the input only lets Qwen finish its fixed wrapper bookkeeping.
        return hidden_states


class _DepthwiseResidual2D(nn.Module):
    def __init__(self, channels: int, groups: int = 8) -> None:
        super().__init__()
        norm_groups = min(groups, channels)
        while channels % norm_groups:
            norm_groups -= 1
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.GroupNorm(norm_groups, channels),
            nn.GELU(),
        )
        self.gate = nn.Parameter(torch.logit(torch.tensor(0.10)))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + torch.sigmoid(self.gate) * self.block(value)


class _UpsampleBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        groups = min(8, output_channels)
        while output_channels % groups:
            groups -= 1
        self.block = nn.Sequential(
            nn.Conv2d(
                input_channels,
                input_channels,
                3,
                padding=1,
                groups=input_channels,
                bias=False,
            ),
            nn.Conv2d(input_channels, output_channels, 1, bias=False),
            nn.GroupNorm(groups, output_channels),
            nn.GELU(),
        )

    def forward(self, value: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        value = F.interpolate(value, size=size, mode="bilinear", align_corners=False)
        return self.block(value)


class _ProgressiveDecoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        projection_dim: int,
        channels: tuple[int, ...],
        output_size: int,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_size = int(output_size)
        self.token_norm = nn.LayerNorm(input_dim)
        self.token_projection = nn.Linear(input_dim, projection_dim)
        self.body = nn.Sequential(
            _DepthwiseResidual2D(projection_dim),
            _DepthwiseResidual2D(projection_dim),
        )
        values = (projection_dim, *channels)
        self.upsample = nn.ModuleList(
            _UpsampleBlock(before, after)
            for before, after in zip(values[:-1], values[1:])
        )
        self.output_channels = values[-1]

    def forward_features(self, spatial: torch.Tensor) -> torch.Tensor:
        if spatial.ndim != 4 or spatial.shape[1] != self.input_dim:
            raise RuntimeError(
                f"Decoder expects [B,{self.input_dim},H,W], got {tuple(spatial.shape)}"
            )
        tokens = spatial.permute(0, 2, 3, 1)
        value = self.token_projection(self.token_norm(tokens.float()))
        value = self.body(value.permute(0, 3, 1, 2))
        height, width = value.shape[-2:]
        for block in self.upsample:
            height = min(self.output_size, height * 2)
            width = min(self.output_size, width * 2)
            value = block(value, (height, width))
        if tuple(value.shape[-2:]) != (self.output_size, self.output_size):
            value = F.interpolate(
                value,
                size=(self.output_size, self.output_size),
                mode="bilinear",
                align_corners=False,
            )
        return value


class SaliencyDensityDecoder(_ProgressiveDecoder):
    def __init__(self, input_dim: int = 192, output_size: int = 224) -> None:
        super().__init__(input_dim, 128, (96, 64, 32, 16), output_size)
        self.refine = _DepthwiseResidual2D(16)
        self.classifier = nn.Conv2d(16, 1, 1)

    def forward(self, spatial: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.refine(self.forward_features(spatial)))

    def specification(self) -> dict[str, Any]:
        return _head_spec(self, "progressive_saliency_density_decoder", 1)


class LesionSegmentationDecoder(_ProgressiveDecoder):
    def __init__(self, input_dim: int = 192, output_size: int = 224) -> None:
        super().__init__(input_dim, 128, (96, 64, 32, 16), output_size)
        self.boundary_refine = nn.Sequential(
            _DepthwiseResidual2D(16),
            _DepthwiseResidual2D(16),
        )
        self.classifier = nn.Conv2d(16, 1, 1)

    def forward(self, spatial: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.boundary_refine(self.forward_features(spatial)))

    def specification(self) -> dict[str, Any]:
        return _head_spec(self, "progressive_lesion_boundary_decoder", 1)


class PoseHeatmapDecoder(_ProgressiveDecoder):
    def __init__(
        self,
        input_dim: int = 192,
        heatmap_size: int = 56,
        num_joints: int = 14,
    ) -> None:
        super().__init__(input_dim, 160, (128, 96), heatmap_size)
        self.num_joints = int(num_joints)
        self.refine = _DepthwiseResidual2D(96)
        self.predictor = nn.Conv2d(96, self.num_joints, 1)

    def forward(self, spatial: torch.Tensor) -> torch.Tensor:
        return self.predictor(self.refine(self.forward_features(spatial)))

    def specification(self) -> dict[str, Any]:
        return _head_spec(
            self, "progressive_pose_heatmap_decoder", self.num_joints
        )


def _head_spec(module: nn.Module, name: str, outputs: int) -> dict[str, Any]:
    return {
        "type": name,
        "input_dim": int(getattr(module, "input_dim")),
        "output_size": int(getattr(module, "output_size")),
        "output_channels": int(outputs),
        "attention_enabled": False,
        "parameters": sum(parameter.numel() for parameter in module.parameters()),
        "trainable_parameters": sum(
            parameter.numel()
            for parameter in module.parameters()
            if parameter.requires_grad
        ),
    }


class Vision2HybridDenseStudent(nn.Module):
    """Frozen Qwen stem, two hybrid Vision stages, and a task decoder."""

    def __init__(self, loaded: Any, settings: Any, head: nn.Module) -> None:
        super().__init__()
        self.visual = loaded.visual
        self.device = loaded.device
        self.original_blocks = list(self.visual.blocks)
        self.original_deepstack_indexes = tuple(
            int(value) for value in getattr(self.visual, "deepstack_visual_indexes", ())
        )
        self.core = DenseVision2Core(settings.vision_hidden_size, settings).to(
            loaded.device
        )
        self.capture_block = _HybridCaptureBlock(self.core)
        self.student_blocks = nn.ModuleList(
            [self.capture_block]
            + [_VisionBypass() for _ in self.original_blocks[1:]]
        )
        self.head = head.to(loaded.device)
        self._active = False
        loaded.model.requires_grad_(False).eval()
        self.core.requires_grad_(True)
        # Dense heads consume the 192-D latent directly; this adapter only
        # feeds the ignored Qwen merger path and must not appear trainable.
        self.core.hybrid.output_adapter.requires_grad_(False)
        self.head.requires_grad_(True)

    def activate(self) -> None:
        for index, block in enumerate(self.student_blocks):
            self.visual.blocks[index] = block
        if hasattr(self.visual, "deepstack_visual_indexes"):
            self.visual.deepstack_visual_indexes = []
        self._active = True

    def restore_native(self) -> None:
        for index, block in enumerate(self.original_blocks):
            self.visual.blocks[index] = block
        if hasattr(self.visual, "deepstack_visual_indexes"):
            self.visual.deepstack_visual_indexes = list(
                self.original_deepstack_indexes
            )
        self._active = False

    def train(self, mode: bool = True) -> "Vision2HybridDenseStudent":
        super().train(mode)
        self.visual.eval()
        self.core.train(mode)
        self.head.train(mode)
        return self

    def forward(
        self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self._active:
            self.activate()
        self.capture_block.set_grid(image_grid_thw)
        dtype = next(self.visual.patch_embed.parameters()).dtype
        self.visual(pixel_values.to(dtype), grid_thw=image_grid_thw)
        groups = self.core.last_latent_groups
        if len(groups) != len(image_grid_thw):
            raise RuntimeError("Vision2 core did not retain one latent group per image")
        spatial = restore_qwen_block_major_spatial(
            torch.cat(groups, dim=0), image_grid_thw
        )
        detector = self.core.optical_branch.core.current_detector_readout
        if detector is None:
            raise RuntimeError("Vision2 global CCD readout is unavailable")
        return self.head(spatial), spatial, detector

    def router_losses(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.core.router_losses()

    def operating_loss(self) -> torch.Tensor:
        return self.core.operating_loss()


__all__ = [
    "DenseVision2Core",
    "LesionSegmentationDecoder",
    "PoseHeatmapDecoder",
    "SaliencyDensityDecoder",
    "Vision2HybridDenseStudent",
    "restore_qwen_block_major_spatial",
]
