"""Cache an aligned raw-RGB view for alternate four-frame sampling.

The custom Conv E1 correction consumes raw frames.  Whenever the frozen Qwen
and Conv5 inputs switch to another temporal view, the raw frames must switch to
the same view as one atomic sample; otherwise training silently combines three
different moments of a video.
"""

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


CONTRACT = "lgvq_raw_rgb_four_frame_sampling_view_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_cache(
    *,
    manifest: Path,
    output: Path,
    sampling_offset: float,
) -> dict[str, Any]:
    fractions = frame_fractions(4, sampling_offset=sampling_offset)
    manifest = manifest.expanduser().resolve()
    output = output.expanduser().resolve()
    rows = read_manifest(manifest)
    frames = torch.empty(len(rows), 4, 3, 224, 224, dtype=torch.uint8)
    for index, row in enumerate(rows):
        decoded = decode_frames(
            Path(row.video_path),
            4,
            sampling_offset=sampling_offset,
            output_size=224,
        )
        frames[index].copy_(
            torch.stack(
                [
                    torch.from_numpy(np.asarray(image, dtype=np.uint8).copy()).permute(
                        2, 0, 1
                    )
                    for image in decoded
                ]
            )
        )
        if (index + 1) % 64 == 0 or index + 1 == len(rows):
            print(f"[raw-frame-view] {index + 1}/{len(rows)}", flush=True)
    payload = {
        "schema_version": 1,
        "feature_contract": CONTRACT,
        "frames": frames,
        "frame_count": 4,
        "frame_size": 224,
        "frame_sampling_offset": float(sampling_offset),
        "frame_fractions": list(fractions),
        "sample_ids": [row.sample_id for row in rows],
        "video_paths": [str(row.video_path) for row in rows],
        "splits": [row.split for row in rows],
        "manifest_path": str(manifest),
        "manifest_sha256": _sha256(manifest),
        "alignment_rule": (
            "Select this raw-frame tensor with the Vision and Conv5 cache that "
            "declares the identical frame_sampling_offset"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(output)
    report = {key: value for key, value in payload.items() if key != "frames"}
    report.update(
        {
            "path": str(output),
            "shape": list(frames.shape),
            "dtype": str(frames.dtype),
            "sha256": _sha256(output),
        }
    )
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sampling-offset", required=True, type=float)
    args = parser.parse_args()
    print(
        json.dumps(
            build_cache(
                manifest=args.manifest,
                output=args.output,
                sampling_offset=args.sampling_offset,
            ),
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
