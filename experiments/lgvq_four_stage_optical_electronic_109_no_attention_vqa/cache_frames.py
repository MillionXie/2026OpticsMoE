from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import build_frame_cache


def main() -> int:
    parser = argparse.ArgumentParser(description="Decode four compact uint8 frames per LGVQ video")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--frame-size", type=int, default=224)
    parser.add_argument("--crop-fraction", type=float, default=0.65)
    args = parser.parse_args()
    report = build_frame_cache(args.manifest, args.output, frame_size=args.frame_size, crop_fraction=args.crop_fraction)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
