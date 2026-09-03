"""Cache a frozen, target-trained five-convolution quality input head.

This is a warm-start asset, not a hidden inference bypass: the cached tokens
are fused with the official Qwen patch+position tokens before optical stage 1
and therefore traverse all four O/E fusion stages in the deployed model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch


CONTRACT = "lgvq_quality_conv5_feature_cache_v1"


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
    from experiments.lgvq_four_stage_optical_electronic_109_no_attention_vqa.modeling import (
        FrameStem,
    )

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
    prefix = "frame_stem."
    stem_state = {
        name[len(prefix) :]: value
        for name, value in state.items()
        if name.startswith(prefix)
    }
    stem = FrameStem(192)
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
            "Frozen five-convolution quality input head; fused with Qwen tokens "
            "before optical stage 1; never connected directly to the MOS readout"
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

