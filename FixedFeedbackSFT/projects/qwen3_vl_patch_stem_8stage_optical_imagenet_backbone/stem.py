from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


STEM_FORMAT = "qwen3-vl-static-image-patch-position-stem-v1"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def merge_order(value: torch.Tensor, merge_size: int = 2) -> torch.Tensor:
    """Use the native Qwen3-VL 2x2 spatial-merger token order.

    ``value`` is [B,C,H,W]. Qwen groups every 2x2 patch block before it enters
    the vision stack, so the flattened order is block-row, block-column,
    intra-row, intra-column rather than ordinary row-major order.
    """

    if value.ndim != 4:
        raise ValueError(f"Expected [B,C,H,W], got {tuple(value.shape)}")
    batch, channels, height, width = value.shape
    merge = int(merge_size)
    if height % merge or width % merge:
        raise ValueError("Patch grid must be divisible by spatial_merge_size")
    return (
        value.view(batch, channels, height // merge, merge, width // merge, merge)
        .permute(0, 2, 4, 3, 5, 1)
        .reshape(batch, height * width, channels)
    )


class StaticQwenPatchStem(nn.Module):
    """The extracted Qwen3-VL patch/position stem for 224x224 still images.

    It contains no Transformer, attention, spatial merger or language model.
    All tensors are registered as buffers and are therefore strictly frozen.
    """

    def __init__(self, checkpoint: str | Path) -> None:
        super().__init__()
        checkpoint = Path(checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing extracted Qwen stem: {checkpoint}")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if payload.get("format") != STEM_FORMAT:
            raise RuntimeError(
                f"Unsupported stem format {payload.get('format')!r}; expected {STEM_FORMAT!r}"
            )
        weight = payload["conv2d_weight"].float().contiguous()
        bias = payload["conv2d_bias"].float().contiguous()
        position = payload["position_embedding"].float().contiguous()
        metadata = dict(payload.get("metadata", {}))
        patch_size = int(metadata.get("patch_size", weight.shape[-1]))
        image_size = int(metadata.get("image_size", 224))
        token_count = (image_size // patch_size) ** 2
        expected_weight = (1024, 3, patch_size, patch_size)
        if tuple(weight.shape) != expected_weight:
            raise RuntimeError(
                f"Expected collapsed patch weight {expected_weight}, got {tuple(weight.shape)}"
            )
        if tuple(bias.shape) != (1024,) or tuple(position.shape) != (token_count, 1024):
            raise RuntimeError(
                "Extracted Qwen stem bias/position shapes do not match the fixed 224 setup"
            )
        self.image_size = image_size
        self.patch_size = patch_size
        self.token_count = token_count
        self.hidden_size = 1024
        self.merge_size = int(metadata.get("spatial_merge_size", 2))
        self.metadata = metadata
        self.checkpoint_path = str(checkpoint)
        self.checkpoint_sha256 = sha256_file(checkpoint)
        self.register_buffer("conv2d_weight", weight, persistent=True)
        self.register_buffer("conv2d_bias", bias, persistent=True)
        self.register_buffer("position_embedding", position, persistent=True)
        self.register_buffer(
            "qwen_mean",
            torch.tensor(metadata.get("image_mean", [0.5, 0.5, 0.5]), dtype=torch.float32).view(1, 3, 1, 1),
            persistent=True,
        )
        self.register_buffer(
            "qwen_std",
            torch.tensor(metadata.get("image_std", [0.5, 0.5, 0.5]), dtype=torch.float32).view(1, 3, 1, 1),
            persistent=True,
        )

    def forward(self, images_01: torch.Tensor) -> torch.Tensor:
        if images_01.ndim != 4 or images_01.shape[1] != 3:
            raise ValueError(f"Expected [B,3,H,W], got {tuple(images_01.shape)}")
        if tuple(images_01.shape[-2:]) != (self.image_size, self.image_size):
            raise ValueError(
                f"Static Qwen stem requires {self.image_size}x{self.image_size} input, "
                f"got {tuple(images_01.shape[-2:])}"
            )
        value = images_01.float()
        value = (value - self.qwen_mean) / self.qwen_std
        patches = F.conv2d(
            value,
            self.conv2d_weight,
            self.conv2d_bias,
            stride=self.patch_size,
        )
        tokens = merge_order(patches, self.merge_size)
        return tokens + self.position_embedding.unsqueeze(0)

    def parameter_report(self) -> dict[str, Any]:
        return {
            "format": STEM_FORMAT,
            "checkpoint": self.checkpoint_path,
            "checkpoint_sha256": self.checkpoint_sha256,
            "frozen_parameters": int(
                self.conv2d_weight.numel()
                + self.conv2d_bias.numel()
                + self.position_embedding.numel()
            ),
            "image_size": self.image_size,
            "patch_size": self.patch_size,
            "token_count": self.token_count,
            "hidden_size": self.hidden_size,
            "contains_transformer": False,
            "contains_attention": False,
            "metadata": self.metadata,
        }
