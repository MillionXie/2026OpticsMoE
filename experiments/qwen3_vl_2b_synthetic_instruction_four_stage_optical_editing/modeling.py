from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from torch import nn
from torch.nn import functional as F

from experiments.qwen3_vl_embedding_2b_caltech101_electronic_retrieval.electronic_blocks import (
    ElectronicSequenceCore,
)
from experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval.optical_blocks import (
    LanguageTwoBlockOpticalCore,
    VisionTwoBlockOpticalCore,
)
from experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval.settings import (
    load_settings as load_four_stage_settings,
)

from .settings import Settings


def build_compact_settings(settings: Settings) -> Any:
    compact = load_four_stage_settings(settings.optical_base_config)
    compact.electronic_width = settings.electronic_width
    compact.electronic_layers = 2
    compact.electronic_token_mixer_enabled = True
    compact.electronic_vision_token_mixer_type = "depthwise_conv2d"
    compact.electronic_vision_token_mixer_kernel_size = 3
    compact.electronic_language_token_mixer_type = "depthwise_conv1d"
    compact.electronic_language_token_mixer_kernel_size = 5
    compact.max_visual_tokens = 196
    compact.max_language_tokens = settings.max_language_tokens
    compact.optical_fusion_initial = settings.optical_fusion_initial
    compact.language_optical_max_shift_pixels = settings.optical_shift_pixels
    compact.language_optical_phase_shift_pixels = settings.optical_shift_pixels
    compact.language_optical_ccd_shift_pixels = settings.optical_shift_pixels
    compact.max_shift_pixels = settings.optical_shift_pixels
    compact.phase_shift_pixels = settings.optical_shift_pixels
    compact.ccd_shift_pixels = settings.optical_shift_pixels
    compact.phase_dropout_p = settings.phase_dropout_p
    compact.language_optical_phase_dropout_p = settings.phase_dropout_p
    compact.output_dir = settings.output_dir
    return compact


def _weights_path(checkpoint: Path) -> Path:
    direct = checkpoint / "model.safetensors"
    if direct.exists():
        return direct
    index_path = checkpoint / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        filename = index["weight_map"]["model.visual.patch_embed.proj.weight"]
        return checkpoint / filename
    raise FileNotFoundError(f"Qwen safetensors not found under {checkpoint}")


class FrozenQwenVisionStem(nn.Module):
    """Only Qwen3-VL's pretrained patch and learned position embedding."""

    def __init__(self, checkpoint: Path, image_size: int = 224) -> None:
        super().__init__()
        if image_size != 224:
            raise ValueError("FrozenQwenVisionStem currently supports 224x224 only")
        self.image_size = image_size
        self.patch_size = 16
        self.temporal_patch_size = 2
        self.hidden_size = 1024
        self.grid_size = image_size // self.patch_size
        self.merge_size = 2
        path = _weights_path(checkpoint)
        with safe_open(str(path), framework="pt", device="cpu") as values:
            weight = values.get_tensor("model.visual.patch_embed.proj.weight").float()
            bias = values.get_tensor("model.visual.patch_embed.proj.bias").float()
            position = values.get_tensor("model.visual.pos_embed.weight").float()
        self.proj = nn.Conv3d(
            3,
            self.hidden_size,
            kernel_size=(self.temporal_patch_size, self.patch_size, self.patch_size),
            stride=(self.temporal_patch_size, self.patch_size, self.patch_size),
            bias=True,
        )
        self.proj.weight.data.copy_(weight)
        self.proj.bias.data.copy_(bias)
        self.position = nn.Parameter(position, requires_grad=False)
        self.proj.requires_grad_(False)

    def _interpolated_position(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        side = int(math.sqrt(len(self.position)))
        if side * side != len(self.position):
            raise RuntimeError("Qwen position table is not square")
        table = self.position.view(side, side, self.hidden_size).permute(2, 0, 1).unsqueeze(0)
        resized = F.interpolate(
            table,
            size=(self.grid_size, self.grid_size),
            mode="bilinear",
            align_corners=True,
        )
        return resized[0].to(device=device, dtype=dtype)

    def _to_block_major(self, grid: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = grid.shape
        merge = self.merge_size
        return (
            grid.view(batch, channels, height // merge, merge, width // merge, merge)
            .permute(0, 2, 4, 3, 5, 1)
            .reshape(batch, height * width, channels)
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or tuple(images.shape[-2:]) != (self.image_size, self.image_size):
            raise RuntimeError(f"Vision stem expects [B,3,{self.image_size},{self.image_size}]")
        normalized = images.float().mul(2.0).sub(1.0)
        video = normalized.unsqueeze(2).expand(-1, -1, self.temporal_patch_size, -1, -1)
        grid = self.proj(video).squeeze(2)
        grid = grid + self._interpolated_position(grid.device, grid.dtype).unsqueeze(0)
        return self._to_block_major(grid)


def restore_block_major(tokens: torch.Tensor, merge_size: int = 2) -> torch.Tensor:
    if tokens.ndim != 3:
        raise RuntimeError("Expected [B,T,C] block-major tokens")
    batch, count, channels = tokens.shape
    side = int(math.sqrt(count))
    if side * side != count or side % merge_size:
        raise RuntimeError(f"Cannot restore {count} tokens to a merge-{merge_size} square")
    return (
        tokens.view(
            batch,
            side // merge_size,
            side // merge_size,
            merge_size,
            merge_size,
            channels,
        )
        .permute(0, 5, 1, 3, 2, 4)
        .reshape(batch, channels, side, side)
    )


class ConditionedResidual2D(nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            3,
            padding=dilation,
            dilation=dilation,
            groups=channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(channels, channels, 1, bias=False)
        self.condition = nn.Linear(channels, channels * 2)
        self.gate = nn.Parameter(torch.logit(torch.tensor(0.10)))

    def forward(self, value: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.condition(condition).chunk(2, dim=-1)
        hidden = self.norm(value)
        hidden = hidden * (1.0 + 0.1 * torch.tanh(gamma)[:, :, None, None])
        hidden = hidden + 0.1 * torch.tanh(beta)[:, :, None, None]
        hidden = self.pointwise(F.gelu(self.depthwise(hidden)))
        return value + torch.sigmoid(self.gate) * hidden


class UpsampleBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        groups = min(8, output_channels)
        while output_channels % groups:
            groups -= 1
        self.block = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, 3, padding=1, groups=input_channels, bias=False),
            nn.Conv2d(input_channels, output_channels, 1, bias=False),
            nn.GroupNorm(groups, output_channels),
            nn.GELU(),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = F.interpolate(value, scale_factor=2.0, mode="bilinear", align_corners=False)
        return self.block(value)


class StructuredCanvasDecoder(nn.Module):
    def __init__(self, input_channels: int = 192, palette_classes: int = 8) -> None:
        super().__init__()
        self.input_projection = nn.Sequential(
            nn.Conv2d(input_channels, 128, 1, bias=False),
            nn.GroupNorm(8, 128),
            nn.GELU(),
        )
        self.condition = nn.Linear(input_channels, 256)
        self.upsample = nn.ModuleList(
            [
                UpsampleBlock(128, 112),
                UpsampleBlock(112, 80),
                UpsampleBlock(80, 48),
                UpsampleBlock(48, 32),
            ]
        )
        self.refine = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1, groups=32, bias=False),
            nn.Conv2d(32, 32, 1, bias=False),
            nn.GroupNorm(8, 32),
            nn.GELU(),
        )
        self.palette_head = nn.Conv2d(32, palette_classes, 1)
        self.edit_head = nn.Conv2d(32, 1, 1)

    def forward(self, spatial: torch.Tensor, condition: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        value = self.input_projection(spatial)
        gamma, beta = self.condition(condition).chunk(2, dim=-1)
        value = value * (1.0 + 0.1 * torch.tanh(gamma)[:, :, None, None])
        value = value + 0.1 * torch.tanh(beta)[:, :, None, None]
        for block in self.upsample:
            value = block(value)
        value = self.refine(value)
        return self.palette_head(value), self.edit_head(value).squeeze(1)


class InstructionOpticalEditor(nn.Module):
    def __init__(self, settings: Settings) -> None:
        super().__init__()
        self.settings = settings
        compact = build_compact_settings(settings)
        core_class_language: type[nn.Module]
        core_class_vision: type[nn.Module]
        if settings.optical_enabled:
            self.language_core = LanguageTwoBlockOpticalCore(
                2048, settings.max_language_tokens, compact
            )
            self.vision_core = VisionTwoBlockOpticalCore(1024, 196, compact)
        else:
            self.language_core = ElectronicSequenceCore(
                2048,
                settings.max_language_tokens,
                compact,
                "depthwise_conv1d",
                5,
            )
            self.vision_core = ElectronicSequenceCore(
                1024,
                196,
                compact,
                "depthwise_conv2d",
                3,
            )
        self.vision_stem = FrozenQwenVisionStem(settings.qwen_checkpoint, settings.image_size)
        self.language_pool = nn.Sequential(
            nn.LayerNorm(settings.electronic_width * 2),
            nn.Linear(settings.electronic_width * 2, settings.electronic_width),
            nn.GELU(),
        )
        self.prompt_to_vision = nn.Sequential(
            nn.LayerNorm(settings.electronic_width),
            nn.Linear(settings.electronic_width, 1024),
        )
        self.prompt_vision_gate = nn.Parameter(
            torch.logit(torch.tensor(settings.optical_fusion_initial))
        )
        self.post_film = nn.Linear(settings.electronic_width, settings.electronic_width * 2)
        nn.init.zeros_(self.post_film.weight)
        nn.init.zeros_(self.post_film.bias)
        self.coordinate_projection = nn.Conv2d(2, settings.electronic_width, 1)
        self.editor = nn.ModuleList(
            [ConditionedResidual2D(settings.electronic_width, dilation) for dilation in (1, 2, 4, 1)]
        )
        self.decoder = StructuredCanvasDecoder(
            settings.electronic_width, settings.palette_classes
        )
        self.task_head = nn.Sequential(
            nn.LayerNorm(settings.electronic_width),
            nn.Linear(settings.electronic_width, 4),
        )

    def _language_condition(self, groups: list[torch.Tensor]) -> torch.Tensor:
        self.language_core.forward_groups(groups, causal=True)
        latent_groups = self.language_core.last_latent_groups
        if len(latent_groups) != len(groups):
            raise RuntimeError("Language core did not retain every prompt group")
        pooled = torch.stack(
            [torch.cat((group.mean(0), group.amax(0)), dim=0) for group in latent_groups]
        )
        return self.language_pool(pooled)

    @staticmethod
    def _coordinates(batch: int, size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        axis = torch.linspace(-1.0, 1.0, size, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        return torch.stack((xx, yy), dim=0).unsqueeze(0).expand(batch, -1, -1, -1)

    def forward(
        self,
        source_images: torch.Tensor,
        prompt_hidden: list[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        device = source_images.device
        groups = [group.to(device=device, dtype=torch.float32) for group in prompt_hidden]
        condition = self._language_condition(groups)
        visual = self.vision_stem(source_images)
        prompt_bias = torch.tanh(self.prompt_to_vision(condition))
        visual = visual + torch.sigmoid(self.prompt_vision_gate) * prompt_bias[:, None, :]
        if self.settings.optical_enabled:
            _, latent = self.vision_core.forward_groups(
                [row for row in visual],
                causal=False,
                spatial_shapes=[(1, 14, 14)] * len(visual),
            )
        else:
            _, latent = self.vision_core.forward_groups(
                [row for row in visual],
                causal=False,
                spatial_shapes=[(1, 14, 14)] * len(visual),
            )
        spatial = restore_block_major(latent[:, :196])
        gamma, beta = self.post_film(condition).chunk(2, dim=-1)
        spatial = spatial * (1.0 + 0.1 * torch.tanh(gamma)[:, :, None, None])
        spatial = spatial + 0.1 * torch.tanh(beta)[:, :, None, None]
        coords = self._coordinates(len(spatial), spatial.shape[-1], spatial.device, spatial.dtype)
        spatial = spatial + self.coordinate_projection(coords)
        for block in self.editor:
            spatial = block(spatial, condition)
        palette_logits, edit_logits = self.decoder(spatial, condition)
        return {
            "palette_logits": palette_logits,
            "edit_logits": edit_logits,
            "task_logits": self.task_head(condition),
            "condition": condition,
            "spatial": spatial,
            "ccd_operating_loss": self.ccd_operating_loss(),
            "router_balance_loss": self.router_balance_loss(),
        }

    def _optical_paths(self) -> list[Any]:
        if not self.settings.optical_enabled:
            return []
        return [self.language_core.optical_branch, self.vision_core.optical_branch]

    def ccd_operating_loss(self) -> torch.Tensor:
        values = [path.current_operating_loss for path in self._optical_paths()]
        present = [value for value in values if value is not None]
        parameter = next(self.parameters())
        return parameter.new_zeros(()) if not present else torch.stack(present).mean()

    def router_balance_loss(self) -> torch.Tensor:
        losses = [path.core.router_losses()[0] for path in self._optical_paths()]
        parameter = next(self.parameters())
        return parameter.new_zeros(()) if not losses else torch.stack(losses).mean()

    def set_phase_trainable(self, enabled: bool) -> None:
        for name, parameter in self.named_parameters():
            if "raw_phase" in name:
                parameter.requires_grad_(enabled)

    def phase_parameters(self) -> list[nn.Parameter]:
        return [parameter for name, parameter in self.named_parameters() if "raw_phase" in name]

    def architecture_report(self) -> dict[str, Any]:
        groups: dict[str, int] = {
            "frozen_qwen_vision_stem": sum(p.numel() for p in self.vision_stem.parameters()),
            "language_core": sum(p.numel() for p in self.language_core.parameters()),
            "vision_core": sum(p.numel() for p in self.vision_core.parameters()),
            "condition_and_editor": sum(
                p.numel()
                for module in (
                    self.language_pool,
                    self.prompt_to_vision,
                    self.post_film,
                    self.coordinate_projection,
                    self.editor,
                    self.task_head,
                )
                for p in module.parameters()
            ),
            "decoder": sum(p.numel() for p in self.decoder.parameters()),
        }
        return {
            "type": "frozen_qwen_lm_language2_vision2_prompt_conditioned_optical_editor",
            "qwen_language_model": "frozen contextual hidden cached offline",
            "qwen_vision": "frozen patch and learned position stem only",
            "optical_enabled": self.settings.optical_enabled,
            "physical_stage_order": (
                ["language_expert", "language_global", "vision_expert", "vision_global"]
                if self.settings.optical_enabled
                else []
            ),
            "prompt_is_only_task_input": True,
            "task_head_is_auxiliary_supervision_only": True,
            "palette_classes": self.settings.palette_classes,
            "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "total_parameters_in_compact_runtime": sum(p.numel() for p in self.parameters()),
            "breakdown": groups,
        }


def build_model(settings: Settings, device: torch.device) -> InstructionOpticalEditor:
    model = InstructionOpticalEditor(settings).to(device)
    model.vision_stem.requires_grad_(False).eval()
    return model


__all__ = [
    "FrozenQwenVisionStem",
    "InstructionOpticalEditor",
    "StructuredCanvasDecoder",
    "build_compact_settings",
    "build_model",
    "restore_block_major",
]
