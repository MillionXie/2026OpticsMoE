from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from experiments.qwen3_vl_2b_synthetic_instruction_four_stage_optical_editing.modeling import (
    ConditionedResidual2D,
    FrozenQwenVisionStem,
    build_compact_settings,
    restore_block_major,
)
from experiments.qwen3_vl_embedding_2b_caltech101_electronic_retrieval.electronic_blocks import (
    ElectronicSequenceCore,
)
from experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval.optical_blocks import (
    LanguageTwoBlockOpticalCore,
    VisionTwoBlockOpticalCore,
)

from .settings import Settings


class SemanticGridDecoder(nn.Module):
    def __init__(self, width: int, grid_size: int, categories: int) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.pre = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1, groups=width, bias=False),
            nn.Conv2d(width, width, 1, bias=False),
            nn.GroupNorm(8, width),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((grid_size, grid_size))
        self.category_head = nn.Conv2d(width, categories + 1, 1)
        self.edit_head = nn.Conv2d(width, 1, 1)

    def forward(self, spatial: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        value = self.pool(self.pre(spatial))
        return self.category_head(value), self.edit_head(value).squeeze(1)


class OpenMojiOpticalEditor(nn.Module):
    def __init__(self, settings: Settings) -> None:
        super().__init__()
        self.settings = settings
        compact = build_compact_settings(settings)
        if settings.optical_enabled:
            self.language_core = LanguageTwoBlockOpticalCore(
                2048, settings.max_language_tokens, compact
            )
            self.vision_core = VisionTwoBlockOpticalCore(1024, 196, compact)
        else:
            self.language_core = ElectronicSequenceCore(
                2048, settings.max_language_tokens, compact, "depthwise_conv1d", 5
            )
            self.vision_core = ElectronicSequenceCore(
                1024, 196, compact, "depthwise_conv2d", 3
            )
        self.vision_stem = FrozenQwenVisionStem(settings.qwen_checkpoint, settings.image_size)
        width = settings.electronic_width
        self.language_pool = nn.Sequential(
            nn.LayerNorm(width * 2), nn.Linear(width * 2, width), nn.GELU()
        )
        self.prompt_to_vision = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1024))
        self.prompt_vision_gate = nn.Parameter(
            torch.logit(torch.tensor(settings.optical_fusion_initial))
        )
        self.post_film = nn.Linear(width, width * 2)
        nn.init.zeros_(self.post_film.weight)
        nn.init.zeros_(self.post_film.bias)
        self.coordinate_projection = nn.Conv2d(2, width, 1)
        self.editor = nn.ModuleList(
            [ConditionedResidual2D(width, dilation) for dilation in (1, 2, 4)]
        )
        self.decoder = SemanticGridDecoder(width, settings.grid_size, settings.icon_classes)
        self.task_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 4))

    def _language_condition(self, groups: list[torch.Tensor]) -> torch.Tensor:
        self.language_core.forward_groups(groups, causal=True)
        latent = self.language_core.last_latent_groups
        pooled = torch.stack(
            [torch.cat((group.mean(0), group.amax(0)), dim=0) for group in latent]
        )
        return self.language_pool(pooled)

    @staticmethod
    def _coordinates(batch: int, size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        axis = torch.linspace(-1.0, 1.0, size, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        return torch.stack((xx, yy), dim=0).unsqueeze(0).expand(batch, -1, -1, -1)

    def forward(
        self, source_images: torch.Tensor, prompt_hidden: list[torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        groups = [value.to(device=source_images.device, dtype=torch.float32) for value in prompt_hidden]
        condition = self._language_condition(groups)
        visual = self.vision_stem(source_images)
        bias = torch.tanh(self.prompt_to_vision(condition))
        visual = visual + torch.sigmoid(self.prompt_vision_gate) * bias[:, None, :]
        _, latent = self.vision_core.forward_groups(
            [row for row in visual],
            causal=False,
            spatial_shapes=[(1, 14, 14)] * len(visual),
        )
        spatial = restore_block_major(latent[:, :196])
        gamma, beta = self.post_film(condition).chunk(2, dim=-1)
        spatial = spatial * (1.0 + 0.1 * torch.tanh(gamma)[:, :, None, None])
        spatial = spatial + 0.1 * torch.tanh(beta)[:, :, None, None]
        coordinates = self._coordinates(
            len(spatial), spatial.shape[-1], spatial.device, spatial.dtype
        )
        spatial = spatial + self.coordinate_projection(coordinates)
        for block in self.editor:
            spatial = block(spatial, condition)
        category_logits, edit_logits = self.decoder(spatial)
        return {
            "category_logits": category_logits,
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
        values = [path.core.router_losses()[0] for path in self._optical_paths()]
        parameter = next(self.parameters())
        return parameter.new_zeros(()) if not values else torch.stack(values).mean()

    def set_phase_trainable(self, enabled: bool) -> None:
        for name, parameter in self.named_parameters():
            if "raw_phase" in name:
                parameter.requires_grad_(enabled)

    def architecture_report(self) -> dict[str, Any]:
        return {
            "type": "frozen_qwen_language2_vision2_openmoji_semantic_grid_editor",
            "qwen_language_model": "full frozen contextual hidden cached offline",
            "qwen_vision": "frozen patch and position stem",
            "physical_stage_order": (
                ["language_expert", "language_global", "vision_expert", "vision_global"]
                if self.settings.optical_enabled
                else []
            ),
            "decoder": "electronic 6x6 category grid plus edit grid; fixed OpenMoji compositor",
            "prompt_is_only_task_input": True,
            "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "total_parameters": sum(p.numel() for p in self.parameters()),
        }


def build_model(settings: Settings, device: torch.device) -> OpenMojiOpticalEditor:
    model = OpenMojiOpticalEditor(settings).to(device)
    model.vision_stem.requires_grad_(False).eval()
    return model


__all__ = ["OpenMojiOpticalEditor", "SemanticGridDecoder", "build_model"]
