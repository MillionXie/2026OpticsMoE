"""Cache a frozen, target-trained five-convolution quality input head.

This is an auxiliary input asset, not an independent prediction branch.  In
the current strict two-branch Spatial model the official Qwen patch+position
tokens remain the shared visual input.  These cached tokens are injected only
inside the first electronic residual E1, immediately before the E1/O1 fusion;
they are never fed directly to O1 or to the final MOS readout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


CONTRACT = "lgvq_quality_conv5_feature_cache_v1"
STEM_ASSET_CONTRACT = "lgvq_quality_conv5_stem_state_v1"


class FrameStem(nn.Module):
    """Exact frozen Conv5 input transform used by the formal Spatial-4 model.

    Keeping the small module here makes a delivery independent of the earlier
    LGVQ prototype from which the weights were warm-started.  It produces an
    auxiliary E1 input; it is not a MOS-prediction branch.
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(14, 48, 3, stride=2, padding=1)
        self.norm1 = nn.GroupNorm(8, 48)
        self.conv2 = nn.Conv2d(48, 64, 3, stride=2, padding=1)
        self.norm2 = nn.GroupNorm(8, 64)
        self.conv3 = nn.Conv2d(64, 96, 3, stride=2, padding=1)
        self.norm3 = nn.GroupNorm(12, 96)
        self.conv4 = nn.Conv2d(96, 96, 3, stride=1, padding=1)
        self.norm4 = nn.GroupNorm(12, 96)
        self.conv5 = nn.Conv2d(96, 192, 3, stride=2, padding=1)
        self.norm5 = nn.GroupNorm(24, 192)
        sobel_x = torch.tensor(
            ((-1.0, 0.0, 1.0), (-2.0, 0.0, 2.0), (-1.0, 0.0, 1.0))
        ) / 4.0
        sobel_y = sobel_x.t().contiguous()
        laplacian = torch.tensor(
            ((0.0, 1.0, 0.0), (1.0, -4.0, 1.0), (0.0, 1.0, 0.0))
        ) / 4.0
        self.register_buffer(
            "sobel_x", sobel_x.view(1, 1, 3, 3), persistent=False
        )
        self.register_buffer(
            "sobel_y", sobel_y.view(1, 1, 3, 3), persistent=False
        )
        self.register_buffer(
            "laplacian", laplacian.view(1, 1, 3, 3), persistent=False
        )

    def quality_channels(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.ndim != 5 or tuple(frames.shape[1:3]) != (4, 3):
            raise ValueError("Frame stem expects [B,4,3,H,W]")
        batch, frame_count, _, height, width = frames.shape
        rgb = frames.float().div(255.0)
        luminance = (
            0.2989 * rgb[:, :, 0:1]
            + 0.5870 * rgb[:, :, 1:2]
            + 0.1140 * rgb[:, :, 2:3]
        )
        flat_luminance = luminance.flatten(0, 1)
        padded3 = F.pad(flat_luminance, (1, 1, 1, 1), mode="reflect")
        sobel_x = F.conv2d(padded3, self.sobel_x)
        sobel_y = F.conv2d(padded3, self.sobel_y)
        gradient = torch.sqrt(sobel_x.square() + sobel_y.square() + 1.0e-12)
        laplacian = F.conv2d(padded3, self.laplacian).abs()
        padded5 = F.pad(flat_luminance, (2, 2, 2, 2), mode="reflect")
        local_mean = F.avg_pool2d(padded5, 5, stride=1)
        local_square_mean = F.avg_pool2d(padded5.square(), 5, stride=1)
        local_std = (
            local_square_mean - local_mean.square()
        ).clamp_min(0.0).sqrt()
        shape = (batch, frame_count, 1, height, width)
        sobel_x = sobel_x.reshape(shape)
        sobel_y = sobel_y.reshape(shape)
        gradient = gradient.reshape(shape)
        laplacian = laplacian.reshape(shape)
        local_std = local_std.reshape(shape)
        saturation = rgb.amax(2, keepdim=True) - rgb.amin(2, keepdim=True)
        temporal = torch.zeros_like(luminance)
        temporal[:, 1:] = (luminance[:, 1:] - luminance[:, :-1]).abs()
        y = torch.linspace(
            -1.0, 1.0, height, device=rgb.device, dtype=rgb.dtype
        ).view(1, 1, 1, height, 1).expand(batch, frame_count, 1, height, width)
        x = torch.linspace(
            -1.0, 1.0, width, device=rgb.device, dtype=rgb.dtype
        ).view(1, 1, 1, 1, width).expand(batch, frame_count, 1, height, width)
        time = torch.linspace(
            -1.0, 1.0, frame_count, device=rgb.device, dtype=rgb.dtype
        ).view(1, frame_count, 1, 1, 1).expand(
            batch, frame_count, 1, height, width
        )
        return torch.cat(
            (
                rgb,
                luminance,
                sobel_x,
                sobel_y,
                gradient,
                laplacian,
                local_std,
                saturation,
                temporal,
                x,
                y,
                time,
            ),
            2,
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        batch, frame_count = frames.shape[:2]
        value = self.quality_channels(frames).flatten(0, 1)
        value = F.gelu(self.norm1(self.conv1(value)))
        value = F.gelu(self.norm2(self.conv2(value)))
        value = F.gelu(self.norm3(self.conv3(value)))
        value = F.gelu(self.norm4(self.conv4(value)))
        value = F.gelu(self.norm5(self.conv5(value)))
        return value.flatten(2).transpose(1, 2).reshape(
            batch, frame_count, -1, value.shape[1]
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path, *, mmap: bool = False) -> Any:
    kwargs: dict[str, Any] = {"map_location": "cpu", "weights_only": False}
    if mmap:
        kwargs["mmap"] = True
    try:
        return torch.load(path, **kwargs)
    except TypeError:
        kwargs.pop("mmap", None)
        return torch.load(path, **kwargs)


def build_cache(
    *, frame_cache: Path, checkpoint: Path, output: Path, batch_size: int, device: str
) -> dict[str, Any]:
    frame_cache = frame_cache.expanduser().resolve()
    checkpoint = checkpoint.expanduser().resolve()
    output = output.expanduser().resolve()
    frames_payload = _load(frame_cache, mmap=True)
    frames = frames_payload.get("frames")
    sample_ids = list(map(str, frames_payload.get("sample_ids", [])))
    if not torch.is_tensor(frames) or frames.dtype != torch.uint8:
        raise ValueError("Source frame cache must contain uint8 frames")
    if tuple(frames.shape[1:]) != (4, 3, 224, 224) or len(sample_ids) != frames.shape[0]:
        raise ValueError("Source frame cache must be [N,4,3,224,224] with matching IDs")

    saved = _load(checkpoint)
    state = saved.get("state_dict", saved.get("model", saved))
    if not isinstance(state, dict):
        raise ValueError("Source checkpoint has no state_dict")
    if saved.get("contract") == STEM_ASSET_CONTRACT:
        stem_state = state
    else:
        prefix = "frame_stem."
        stem_state = {
            name[len(prefix) :]: value
            for name, value in state.items()
            if name.startswith(prefix)
        }
    if not stem_state:
        raise ValueError("No FrameStem weights found in preprocessing asset")
    stem = FrameStem()
    stem.load_state_dict(stem_state, strict=True)
    target_device = torch.device(device if torch.cuda.is_available() else "cpu")
    stem.to(target_device).eval().requires_grad_(False)
    quality = torch.empty(frames.shape[0], 4, 196, 192, dtype=torch.float16)
    with torch.inference_mode():
        for start in range(0, frames.shape[0], batch_size):
            stop = min(frames.shape[0], start + batch_size)
            value = stem(frames[start:stop].to(target_device, non_blocking=True))
            if tuple(value.shape) != (stop - start, 4, 196, 192):
                raise RuntimeError(f"Unexpected FrameStem shape {tuple(value.shape)}")
            quality[start:stop].copy_(value.detach().cpu().half())
            print(f"[quality-stem] {stop}/{frames.shape[0]}", flush=True)
    payload = {
        "schema_version": 1,
        "contract": CONTRACT,
        "quality_tokens": quality,
        "sample_ids": sample_ids,
        "shape": list(quality.shape),
        "dtype": str(quality.dtype),
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256": _sha256(checkpoint),
        "source_frame_cache": str(frame_cache),
        "source_frame_cache_sha256": _sha256(frame_cache),
        "interpretation": (
            "Frozen five-convolution quality input head; in the strict two-branch "
            "Spatial model it is injected only into electronic residual E1 before "
            "the E1/O1 fusion, never directly into O1 or the MOS readout"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(output)
    report = {key: value for key, value in payload.items() if key != "quality_tokens"}
    report["path"] = str(output)
    report["sha256"] = _sha256(output)
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame-cache", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    print(json.dumps(build_cache(**vars(args)), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
