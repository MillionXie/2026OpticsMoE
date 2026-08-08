"""Project ordered digit BMPs and audit SLM-to-camera timing/order.

The filenames and manifest are the source of truth for ordering.  The final
contact sheet puts each commanded frame beside the captured frame so a bench
operator can spot stale, duplicated, or permuted camera frames immediately.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

try:
    from ..devices import build_camera, build_slm, verify_camera_roi
except ImportError:  # direct execution inside demos/
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from devices import build_camera, build_slm, verify_camera_roi


def _resolve(value: str | Path, base: Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for name in ("DejaVuSans-Oblique.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def _synthetic_digit(digit: int, size: int = 28) -> np.ndarray:
    image = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(image)
    font = _font(int(size * 0.92))
    box = draw.textbbox((0, 0), str(digit), font=font)
    x = (size - (box[2] - box[0])) // 2 - box[0]
    y = (size - (box[3] - box[1])) // 2 - box[1]
    draw.text((x, y), str(digit), fill=255, font=font)
    return np.asarray(image, dtype=np.uint8)


def load_digit_images(config: dict[str, Any], base: Path) -> list[np.ndarray]:
    source = str(config.get("source", "mnist")).lower()
    if source == "synthetic":
        return [_synthetic_digit(digit) for digit in range(10)]
    if source != "mnist":
        raise ValueError("digits.source must be 'mnist' or 'synthetic'")
    try:
        from torchvision.datasets import MNIST
    except Exception as exc:
        raise RuntimeError("digits.source=mnist requires torchvision") from exc
    data_root = _resolve(config.get("data_root", "../../../data/mnist"), base)
    dataset = MNIST(root=str(data_root), train=False, download=bool(config.get("download", True)))
    chosen: dict[int, np.ndarray] = {}
    for image, label in dataset:
        label = int(label)
        if label not in chosen:
            chosen[label] = np.asarray(image.convert("L"), dtype=np.uint8)
        if len(chosen) == 10:
            break
    if len(chosen) != 10:
        raise RuntimeError(f"MNIST loader returned only digits {sorted(chosen)}")
    return [chosen[digit] for digit in range(10)]


def generate_digit_bmps(
    digit_images: list[np.ndarray],
    output_dir: Path,
    *,
    slm_size_wh: tuple[int, int] = (1920, 1080),
    active_size_wh: tuple[int, int] = (956, 956),
    digit_size_px: int = 700,
    foreground_level: int = 255,
    background_level: int = 0,
) -> list[Path]:
    if len(digit_images) != 10:
        raise ValueError("exactly ten digit images (0..9) are required")
    width, height = slm_size_wh
    active_width, active_height = active_size_wh
    if active_width > width or active_height > height:
        raise ValueError("active area exceeds amplitude SLM canvas")
    if not 1 <= digit_size_px <= min(active_width, active_height):
        raise ValueError("digit_size_px must fit inside the active area")
    if not 0 <= background_level <= foreground_level <= 255:
        raise ValueError("require 0 <= background_level <= foreground_level <= 255")
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    active_x = (width - active_width) // 2
    active_y = (height - active_height) // 2
    digit_x = active_x + (active_width - digit_size_px) // 2
    digit_y = active_y + (active_height - digit_size_px) // 2
    for digit, raw in enumerate(digit_images):
        source = Image.fromarray(np.asarray(raw, dtype=np.uint8), mode="L")
        source = source.resize((digit_size_px, digit_size_px), Image.Resampling.BILINEAR)
        source_array = np.asarray(source, dtype=np.float32) / 255.0
        encoded = np.rint(
            background_level + source_array * (foreground_level - background_level)
        ).astype(np.uint8)
        canvas = Image.new("L", (width, height), 0)
        canvas.paste(Image.fromarray(encoded, mode="L"), (digit_x, digit_y))
        path = output_dir / f"{digit:03d}_digit_{digit}.bmp"
        canvas.save(path, format="BMP")
        paths.append(path)
    return paths


def _load_capture(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        value = np.load(path)
    else:
        value = np.asarray(Image.open(path).convert("L"))
    if value.ndim != 2:
        raise RuntimeError(f"capture must be a 2-D monochrome image, got {value.shape}")
    return value


def _preview(array: np.ndarray) -> Image.Image:
    value = np.asarray(array, dtype=np.float32)
    positive = value[value > 0]
    high = float(np.percentile(positive, 99.5)) if positive.size else 1.0
    value = np.clip(value / max(high, 1.0), 0.0, 1.0)
    return Image.fromarray(np.rint(value * 255.0).astype(np.uint8), mode="L")


def save_contact_sheet(rows: list[dict[str, Any]], output: Path) -> None:
    if not rows:
        return
    thumb = 260
    margin = 24
    sheet = Image.new("RGB", (2 * thumb + 3 * margin, len(rows) * (thumb + margin) + margin), "white")
    draw = ImageDraw.Draw(sheet)
    for row_index, row in enumerate(rows):
        y = margin + row_index * (thumb + margin)
        source = Image.open(row["input_bmp"]).convert("L")
        captured = _preview(_load_capture(Path(row["capture_path"])))
        source.thumbnail((thumb, thumb), Image.Resampling.BILINEAR)
        captured.thumbnail((thumb, thumb), Image.Resampling.BILINEAR)
        sheet.paste(source.convert("RGB"), (margin, y))
        sheet.paste(captured.convert("RGB"), (2 * margin + thumb, y))
        draw.text((margin + 4, y + 4), f"command {row['order_index']}: digit {row['digit']}", fill="red")
        draw.text((2 * margin + thumb + 4, y + 4), f"capture {row['order_index']}", fill="red")
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)


def run(config_path: str | Path, *, generate_only: bool = False) -> dict[str, Any]:
    path = Path(config_path).expanduser().resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    base = path.parent
    output = _resolve(
        raw.get("output_dir", "../artifacts/demos/amplitude_digits"), base
    )
    inputs = output / "input_bmp"
    captures = output / "ccd_captured"
    digits = dict(raw.get("digits", {}))
    digit_images = load_digit_images(digits, base)
    slm_size = tuple(int(v) for v in digits.get("slm_size_wh", (1920, 1080)))
    active_size = tuple(int(v) for v in digits.get("active_size_wh", (956, 956)))
    files = generate_digit_bmps(
        digit_images,
        inputs,
        slm_size_wh=slm_size,
        active_size_wh=active_size,
        digit_size_px=int(digits.get("digit_size_px", 700)),
        foreground_level=int(digits.get("foreground_level", 255)),
        background_level=int(digits.get("background_level", 0)),
    )
    manifest: dict[str, Any] = {
        "config": str(path),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "slm_size_wh": list(slm_size),
        "active_size_wh": list(active_size),
        "play_order": [value.name for value in files],
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if generate_only:
        print(f"Generated {len(files)} ordered amplitude BMPs under {inputs}")
        return manifest

    device = dict(raw.get("devices", {}))
    camera_config = dict(device["camera"])
    extension = str(camera_config.get("output_extension", ".npy")).lower()
    if extension not in {".npy", ".png", ".tif", ".tiff"}:
        raise ValueError("camera.output_extension must be a lossless format")
    settle_seconds = float(raw.get("settle_delay_ms", 40.0)) / 1000.0
    rows: list[dict[str, Any]] = []
    captures.mkdir(parents=True, exist_ok=True)
    verify_camera_roi(camera_config)
    if bool(raw.get("confirm_before_start", True)):
        input(
            "请确认相位 SLM 已加载 phase_zero.bmp、配置中的手动 ROI 正确，"
            "然后按 Enter 开始按 0→9 顺序播放和采集："
        )
    with ExitStack() as stack:
        slm = stack.enter_context(build_slm(dict(device["amplitude_slm"]), base))
        camera_config.pop("output_extension", None)
        camera = stack.enter_context(build_camera(camera_config, base))
        verify_camera_roi(camera_config, camera.device_info())
        slm.preload_files(files)
        manifest["amplitude_slm"] = slm.device_info()
        manifest["camera"] = camera.device_info()
        (output / "resolved_devices.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        for index, input_path in enumerate(files):
            destination = captures / f"{index:03d}_digit_{index}{extension}"
            slm.display_file(input_path)
            time.sleep(settle_seconds)
            camera.capture(destination)
            row = {
                "order_index": index,
                "digit": index,
                "input_bmp": str(input_path),
                "capture_path": str(destination),
                "captured_utc": datetime.now(timezone.utc).isoformat(),
            }
            rows.append(row)
            print(f"[digit demo] {index + 1}/10 command=digit-{index} capture={destination.name}")
    with (output / "capture_order.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    save_contact_sheet(rows, output / "input_vs_capture_order.png")
    print(f"Order audit: {output / 'input_vs_capture_order.png'}")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Amplitude-SLM + camera ordered 0..9 timing/ROI audit"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--generate-only", action="store_true")
    args = parser.parse_args()
    run(args.config, generate_only=args.generate_only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
