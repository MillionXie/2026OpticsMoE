"""Open one configured camera and save a few lossless diagnostic frames."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from ..devices import build_camera
except ImportError:  # direct execution from hardware_sdk/
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from devices import build_camera


def run(config_path: str | Path, output_dir: str | Path, frame_count: int) -> dict:
    config_path = Path(config_path).expanduser().resolve()
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    camera = build_camera(dict(raw["camera"]), config_path.parent)
    camera.validate_runtime()
    rows = []
    with camera:
        print(json.dumps(camera.device_info(), ensure_ascii=False, indent=2))
        for index in range(int(frame_count)):
            path = output_dir / f"frame_{index:03d}.npy"
            camera.capture(path)
            value = np.load(path, allow_pickle=False)
            preview = value.astype(np.float32)
            low, high = np.percentile(preview, [1.0, 99.8])
            preview = np.clip((preview - low) / max(float(high - low), 1e-12), 0, 1)
            Image.fromarray(np.rint(preview * 255).astype(np.uint8), mode="L").save(
                output_dir / f"frame_{index:03d}_preview.png"
            )
            info = dict(camera.device_info().get("last_capture") or {})
            rows.append(
                {
                    "index": index,
                    "npy": path.name,
                    "shape_hw": list(value.shape),
                    "dtype": str(value.dtype),
                    "min": int(value.min()),
                    "max": int(value.max()),
                    "mean": float(value.mean()),
                    "capture": info,
                }
            )
    report = {
        "config": str(config_path),
        "camera": camera.device_info(),
        "frames": rows,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "camera_smoke_test.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved {len(rows)} raw frames and inspectable previews under {output_dir}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Lossless camera-only smoke test")
    parser.add_argument(
        "--config", default="configs/acquisition/tucam_windows.json"
    )
    parser.add_argument("--output-dir", default="artifacts/demos/camera_smoke_test")
    parser.add_argument("--frames", type=int, default=3)
    args = parser.parse_args()
    if args.frames <= 0:
        parser.error("--frames must be positive")
    run(args.config, args.output_dir, args.frames)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
