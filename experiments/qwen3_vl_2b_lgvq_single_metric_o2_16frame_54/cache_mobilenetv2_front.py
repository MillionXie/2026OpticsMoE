"""Cache a 0.29M/0.36M pretrained MobileNetV2 electronic input front.

Only torchvision MobileNetV2 feature blocks 0..10 or 0..11 are retained. They
contain plain/inverted-residual convolutions and produce 14x14 tokens; the
original classifier and all later blocks are discarded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build(
    *, frame_cache: Path, output: Path, batch_size: int, device: str,
    last_block: int = 10
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
    weights = MobileNet_V2_Weights.IMAGENET1K_V2
    source = mobilenet_v2(weights=weights)
    if last_block not in {10, 11}:
        raise ValueError("last_block must be 10 or 11")
    front = source.features[: last_block + 1].to(target_device).eval()
    front.requires_grad_(False)
    parameter_count = sum(parameter.numel() for parameter in front.parameters())
    expected_parameters = {10: 239_360, 11: 305_984}[last_block]
    output_width = {10: 64, 11: 96}[last_block]
    if parameter_count != expected_parameters:
        raise RuntimeError(f"Unexpected MobileNetV2 front size: {parameter_count}")
    mean = torch.tensor((0.485, 0.456, 0.406), device=target_device).view(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225), device=target_device).view(1, 3, 1, 1)
    tokens = torch.empty(
        frames.shape[0], 4, 196, output_width, dtype=torch.float16
    )
    with torch.inference_mode():
        for start in range(0, frames.shape[0], batch_size):
            stop = min(frames.shape[0], start + batch_size)
            count = stop - start
            value = frames[start:stop].to(target_device, non_blocking=True)
            value = value.flatten(0, 1).float().div_(255.0)
            value = front((value - mean) / std)
            if tuple(value.shape[1:]) != (output_width, 14, 14):
                raise RuntimeError(f"Unexpected MobileNetV2 output {tuple(value.shape)}")
            value = value.flatten(2).transpose(1, 2).reshape(
                count, 4, 196, output_width
            )
            tokens[start:stop].copy_(value.cpu().half())
            print(
                f"[mobilenetv2-block{last_block}] {stop}/{frames.shape[0]}",
                flush=True,
            )
    payload = {
        "schema_version": 1,
        "contract": (
            f"lgvq_frozen_mobilenetv2_b{last_block}_4f_14x14x{output_width}_v1"
        ),
        "tokens": tokens,
        "sample_ids": sample_ids,
        "shape": list(tokens.shape),
        "dtype": str(tokens.dtype),
        "source_frame_cache": str(frame_cache),
        "source_frame_cache_sha256": _sha256(frame_cache),
        "torchvision_weights": str(weights),
        "front": (
            f"mobilenet_v2 features[0:{last_block + 1}], blocks 0..{last_block}"
        ),
        "last_block": last_block,
        "front_parameters": parameter_count,
        "classifier_or_later_blocks_retained": False,
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
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--last-block", type=int, default=10, choices=(10, 11))
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    print(json.dumps(build(**vars(args)), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
