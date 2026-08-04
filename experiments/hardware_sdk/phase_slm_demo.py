"""Standalone Meadowlark Blink phase-SLM slideshow demo (Windows only)."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image, ImageOps


def _resolve(value: str | Path, base: Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def prepare_phase_frame(
    image_path: Path,
    expected_size_wh: tuple[int, int],
    *,
    wavefront_correction: Path | None,
    flip_vertical: bool,
) -> np.ndarray:
    image = Image.open(image_path).convert("L")
    if image.size != expected_size_wh:
        raise ValueError(f"{image_path.name} size={image.size}, expected={expected_size_wh}")
    if flip_vertical:
        image = ImageOps.flip(image)
    phase = np.asarray(image, dtype=np.uint16)
    if wavefront_correction is not None:
        correction_image = Image.open(wavefront_correction).convert("L")
        if correction_image.size != expected_size_wh:
            raise ValueError(
                f"WFC size={correction_image.size}, expected={expected_size_wh}"
            )
        correction = np.asarray(correction_image, dtype=np.uint16)
        phase = (phase + correction) % 256
    return np.ascontiguousarray(phase.astype(np.uint8))


class BlinkPhaseSLM:
    def __init__(self, sdk_root: Path, lut_file: Path) -> None:
        self.sdk_root = sdk_root
        self.lut_file = lut_file
        self.library: Any = None
        self.width = 0
        self.height = 0
        self.depth = 0

    def __enter__(self) -> "BlinkPhaseSLM":
        if not sys.platform.startswith("win"):
            raise RuntimeError("Meadowlark Blink HDMI SDK in hardware_sdk is Windows-only")
        sdk_dir = self.sdk_root / "SDK"
        wrapper = sdk_dir / "Blink_C_wrapper.dll"
        if not wrapper.is_file():
            raise FileNotFoundError(wrapper)
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(str(sdk_dir))
            os.add_dll_directory(str(self.sdk_root))
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            try:
                ctypes.windll.user32.SetProcessDPIAware()
            except Exception:
                pass
        self.library = ctypes.CDLL(str(wrapper))
        self.library.Create_SDK()
        self.width = int(self.library.Get_Width())
        self.height = int(self.library.Get_Height())
        self.depth = int(self.library.Get_Depth())
        self.library.Load_lut.argtypes = [ctypes.c_char_p]
        self.library.Load_lut.restype = ctypes.c_int
        loaded = self.library.Load_lut(str(self.lut_file).encode("utf-8"))
        if loaded <= 0:
            self.__exit__(None, None, None)
            raise RuntimeError(f"Meadowlark Load_lut failed: {self.lut_file}")
        self.library.Write_image.argtypes = [ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int]
        self.library.Write_image.restype = ctypes.c_int
        return self

    def show(self, frame: np.ndarray) -> None:
        if frame.shape != (self.height, self.width) or frame.dtype != np.uint8:
            raise ValueError(
                f"phase frame must be uint8 [{self.height},{self.width}], got {frame.shape}/{frame.dtype}"
            )
        success = self.library.Write_image(
            frame.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
            1,  # Blink_C_wrapper.h: one-byte 8-bit grayscale input
        )
        if success <= 0:
            raise RuntimeError("Meadowlark Write_image returned failure")

    def __exit__(self, *_: object) -> None:
        if self.library is not None:
            self.library.Delete_SDK()
            self.library = None


def run(config_path: str | Path, *, dry_run: bool = False) -> dict[str, Any]:
    config_path = Path(config_path).expanduser().resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    base = config_path.parent
    sdk_root = _resolve(raw["sdk_root"], base)
    lut_file = _resolve(raw["lut_file"], base)
    patterns_dir = _resolve(raw["patterns_dir"], base)
    output_dir = _resolve(raw.get("output_dir", "../demo_outputs/phase_slm"), base)
    expected = tuple(int(v) for v in raw.get("expected_resolution_wh", (1920, 1200)))
    correction = (
        _resolve(raw["wavefront_correction_file"], base)
        if raw.get("apply_wavefront_correction", False)
        else None
    )
    patterns = sorted(patterns_dir.glob(str(raw.get("pattern_glob", "*.bmp"))))
    if not patterns:
        raise FileNotFoundError(f"No phase BMPs found under {patterns_dir}")
    max_patterns = raw.get("max_patterns")
    if max_patterns is not None:
        patterns = patterns[: int(max_patterns)]
    output_dir.mkdir(parents=True, exist_ok=True)
    frames: list[tuple[Path, np.ndarray]] = []
    for pattern in patterns:
        frame = prepare_phase_frame(
            pattern,
            expected,
            wavefront_correction=correction,
            flip_vertical=bool(raw.get("flip_vertical", False)),
        )
        Image.fromarray(frame, mode="L").save(output_dir / f"displayed_{pattern.name}")
        frames.append((pattern, frame))
    manifest = {
        "config": str(config_path),
        "sdk_root": str(sdk_root),
        "lut_file": str(lut_file),
        "wavefront_correction_file": None if correction is None else str(correction),
        "expected_resolution_wh": list(expected),
        "patterns": [path.name for path, _ in frames],
        "dry_run": dry_run,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if dry_run:
        print(f"Validated {len(frames)} phase frames; previews: {output_dir}")
        return manifest
    duration = float(raw.get("display_seconds", 2.0))
    repeat_count = int(raw.get("repeat_count", 1))
    with BlinkPhaseSLM(sdk_root, lut_file) as slm:
        actual = (slm.width, slm.height)
        if actual != expected:
            raise RuntimeError(f"Meadowlark reports {actual}, config expects {expected}")
        print(f"Meadowlark ready: {slm.width}x{slm.height}, depth={slm.depth}-bit")
        input("Press Enter to begin the phase-mask slideshow...")
        for repeat in range(repeat_count):
            for index, (pattern, frame) in enumerate(frames):
                slm.show(frame)
                print(f"[phase demo] repeat={repeat + 1}/{repeat_count} frame={index + 1}/{len(frames)} {pattern.name}")
                time.sleep(duration)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Meadowlark Blink phase-SLM demo")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run(args.config, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
