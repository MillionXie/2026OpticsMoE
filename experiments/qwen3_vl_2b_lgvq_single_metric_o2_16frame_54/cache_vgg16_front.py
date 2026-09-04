"""Cache frozen plain-VGG16 14x14 features for the four Spatial frames."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from torchvision.models import VGG16_Weights, vgg16


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build(
    *, frame_cache: Path, output: Path, batch_size: int, device: str
) -> dict[str, Any]:
    frame_cache = frame_cache.expanduser().resolve()
    raw = torch.load(frame_cache, map_location="cpu", weights_only=False, mmap=True)
    frames = raw.get("frames")
    sample_ids = list(map(str, raw.get("sample_ids", [])))
    if (
        not torch.is_tensor(frames)
        or frames.dtype != torch.uint8
        or tuple(frames.shape[1:]) != (4, 3, 224, 224)
        or len(sample_ids) != frames.shape[0]
    ):
        raise ValueError("frame cache must contain uint8 [N,4,3,224,224]")
    target_device = torch.device(device if torch.cuda.is_available() else "cpu")
    weights = VGG16_Weights.IMAGENET1K_V1
    # features[:24] ends at the fourth max-pool: [N,512,14,14]. This is a
    # sequential Conv/ReLU/Pool graph with no attention or Transformer.
    front = vgg16(weights=weights).features[:24].to(target_device).eval()
    front.requires_grad_(False)
    mean = torch.tensor((0.485, 0.456, 0.406), device=target_device).view(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225), device=target_device).view(1, 3, 1, 1)
    tokens = torch.empty(frames.shape[0], 4, 196, 512, dtype=torch.float16)
    with torch.inference_mode():
        for start in range(0, frames.shape[0], batch_size):
            stop = min(frames.shape[0], start + batch_size)
            value = frames[start:stop].to(target_device, non_blocking=True)
            count = stop - start
            value = value.flatten(0, 1).float().div_(255.0)
            value = (value - mean) / std
            value = front(value)
            value = value.flatten(2).transpose(1, 2).reshape(count, 4, 196, 512)
            tokens[start:stop].copy_(value.cpu().half())
            print(f"[vgg16-front] {stop}/{frames.shape[0]}", flush=True)
    payload = {
        "schema_version": 1,
        "contract": "lgvq_frozen_plain_vgg16_4f_14x14x512_v1",
        "tokens": tokens,
        "sample_ids": sample_ids,
        "shape": list(tokens.shape),
        "dtype": str(tokens.dtype),
        "source_frame_cache": str(frame_cache),
        "source_frame_cache_sha256": _sha256(frame_cache),
        "torchvision_weights": str(weights),
        "front": "vgg16.features[:24] sequential Conv/ReLU/MaxPool only",
        "attention_or_transformer": False,
    }
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(output)
    report = {key: value for key, value in payload.items() if key != "tokens"}
    report.update({"path": str(output), "sha256": _sha256(output)})
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame-cache", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    print(json.dumps(build(**vars(args)), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
