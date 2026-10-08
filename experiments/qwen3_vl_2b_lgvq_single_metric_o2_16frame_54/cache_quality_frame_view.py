"""Cache a Conv5 quality view for an alternate uniform four-frame selection."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .cache_qwen_front import decode_frames, frame_fractions
from .data import read_manifest


CONTRACT = "lgvq_quality_conv5_feature_cache_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> Any:
    return torch.load(path, map_location="cpu", weights_only=False)


def _load_stem(checkpoint: Path, device: torch.device) -> torch.nn.Module:
    from experiments.lgvq_four_stage_optical_electronic_109_no_attention_vqa.modeling import (
        FrameStem,
    )

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
    return stem.to(device).eval().requires_grad_(False)


def _decode_video(path: Path, *, sampling_offset: float) -> torch.Tensor:
    images = decode_frames(
        path,
        4,
        sampling_offset=sampling_offset,
        output_size=224,
    )
    return torch.stack(
        [
            torch.from_numpy(np.asarray(image, dtype=np.uint8).copy()).permute(2, 0, 1)
            for image in images
        ]
    )


def build_cache(
    *,
    manifest: Path,
    checkpoint: Path,
    output: Path,
    sampling_offset: float,
    batch_size: int,
    device_name: str,
) -> dict[str, Any]:
    fractions = frame_fractions(4, sampling_offset=sampling_offset)
    rows = read_manifest(manifest)
    checkpoint = checkpoint.expanduser().resolve()
    output = output.expanduser().resolve()
    device = torch.device(
        device_name
        if not device_name.startswith("cuda") or torch.cuda.is_available()
        else "cpu"
    )
    stem = _load_stem(checkpoint, device)
    quality = torch.empty(len(rows), 4, 196, 192, dtype=torch.float16)
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            stop = min(len(rows), start + batch_size)
            frames = torch.stack(
                [
                    _decode_video(
                        Path(row.video_path), sampling_offset=sampling_offset
                    )
                    for row in rows[start:stop]
                ]
            ).to(device, non_blocking=True)
            value = stem(frames)
            expected = (stop - start, 4, 196, 192)
            if tuple(value.shape) != expected:
                raise RuntimeError(
                    f"Unexpected FrameStem shape {tuple(value.shape)}; expected {expected}"
                )
            quality[start:stop].copy_(value.detach().cpu().half())
            print(f"[quality-frame-view] {stop}/{len(rows)}", flush=True)
    payload = {
        "schema_version": 1,
        "contract": CONTRACT,
        "quality_tokens": quality,
        "sample_ids": [row.sample_id for row in rows],
        "shape": list(quality.shape),
        "dtype": str(quality.dtype),
        "frame_sampling_offset": sampling_offset,
        "frame_sampling_fractions": list(fractions),
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256": _sha256(checkpoint),
        "source_manifest": str(manifest.resolve()),
        "source_manifest_sha256": _sha256(manifest.resolve()),
        "interpretation": (
            "Frozen five-convolution quality input for one alternate uniform "
            "four-frame view; injected only into electronic residual E1"
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
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sampling-offset", required=True, type=float)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    print(
        json.dumps(
            build_cache(
                manifest=args.manifest,
                checkpoint=args.checkpoint,
                output=args.output,
                sampling_offset=args.sampling_offset,
                batch_size=args.batch_size,
                device_name=args.device,
            ),
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
