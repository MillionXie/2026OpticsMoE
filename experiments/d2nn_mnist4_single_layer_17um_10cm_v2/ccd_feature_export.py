"""Export the exact played-BMP MNIST-4 CCD feature planes.

The scientific output is a compressed FP32 array of untouched linear CCD
intensity.  Grayscale, false-colour and detector-overlay PNGs are display-only
and are never fed back into the classifier or used for formal agreement metrics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw

from .simulation_agreement import (
    PlayedBMPSimulator,
    _load_amplitude,
    _roi_energies,
    _select_torch_device,
    binary_uint8,
    decode_played_phase,
    monochrome_uint8,
)


DETECTOR_COLORS = ((0, 114, 178), (230, 159, 0), (0, 158, 115), (204, 121, 167))


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError("Cannot write an empty CCD feature manifest")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _batches(rows: list[dict[str, str]], size: int) -> Iterable[list[dict[str, str]]]:
    for start in range(0, len(rows), size):
        yield rows[start : start + size]


def _viridis(gray: np.ndarray) -> np.ndarray:
    # Compact dependency-free approximation of matplotlib's viridis anchors.
    anchors = np.asarray(
        [
            [68, 1, 84],
            [59, 82, 139],
            [33, 145, 140],
            [94, 201, 98],
            [253, 231, 37],
        ],
        dtype=np.float32,
    )
    position = np.asarray(gray, dtype=np.float32) / 255.0 * (len(anchors) - 1)
    lower = np.floor(position).astype(np.int64).clip(0, len(anchors) - 1)
    upper = np.minimum(lower + 1, len(anchors) - 1)
    weight = (position - lower)[..., None]
    return np.rint(anchors[lower] * (1.0 - weight) + anchors[upper] * weight).astype(
        np.uint8
    )


def _save_displays(
    *,
    output: Path,
    key: str,
    intensity: np.ndarray,
    detector_bounds: list[list[int]],
) -> dict[str, str]:
    gray = monochrome_uint8(intensity)
    binary = binary_uint8(intensity)
    gray_path = output / "display_grayscale" / f"{key}.png"
    color_path = output / "display_viridis" / f"{key}.png"
    overlay_path = output / "display_detector_overlay" / f"{key}.png"
    binary_path = output / "display_binary" / f"{key}.png"
    for path in (gray_path, color_path, overlay_path, binary_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(gray, mode="L").save(gray_path)
    Image.fromarray(binary, mode="L").save(binary_path)
    color = Image.fromarray(_viridis(gray), mode="RGB")
    color.save(color_path)
    overlay = color.copy()
    draw = ImageDraw.Draw(overlay)
    for label, (left, top, right, bottom) in enumerate(detector_bounds):
        rgb = DETECTOR_COLORS[label % len(DETECTOR_COLORS)]
        draw.rectangle((left, top, right - 1, bottom - 1), outline=rgb, width=3)
        draw.text((left + 3, top + 3), str(label), fill=rgb)
    overlay.save(overlay_path)
    return {
        "grayscale_file": gray_path.relative_to(output).as_posix(),
        "grayscale_sha256": _sha256(gray_path),
        "viridis_file": color_path.relative_to(output).as_posix(),
        "viridis_sha256": _sha256(color_path),
        "detector_overlay_file": overlay_path.relative_to(output).as_posix(),
        "detector_overlay_sha256": _sha256(overlay_path),
        "binary_file": binary_path.relative_to(output).as_posix(),
        "binary_sha256": _sha256(binary_path),
    }


def _contact_sheet(output: Path, rows: list[dict[str, Any]]) -> Path:
    selected: list[dict[str, Any]] = []
    for rank in range(4):
        for label in range(4):
            matches = [row for row in rows if int(row["label"]) == label]
            if rank < len(matches):
                selected.append(matches[rank])
    tile, header = 239, 24
    sheet = Image.new("RGB", (tile * 4, (tile + header) * 4), color=(255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    for index, row in enumerate(selected):
        with Image.open(output / row["detector_overlay_file"]) as opened:
            image = opened.convert("RGB").resize((tile, tile), Image.Resampling.BILINEAR)
        grid_row, column = divmod(index, 4)
        x, y = column * tile, grid_row * (tile + header)
        sheet.paste(image, (x, y))
        draw.text(
            (x + 3, y + tile + 3),
            f"y={row['label']} pred={row['simulation_prediction']}",
            fill=(0, 0, 0),
        )
    path = output / "CONTACT_SHEET_CCD_WITH_DETECTOR_ROIS.png"
    sheet.save(path)
    return path


def export_ccd_features(
    export_dir: str | Path,
    *,
    profiles: list[str] | None = None,
    device: str = "auto",
    batch_size: int = 8,
    overwrite: bool = False,
) -> dict[str, Any]:
    root = Path(export_dir).expanduser().resolve()
    contract = _read_json(root / "hardware_contract.json")
    profile_names = profiles or list(contract["profiles"])
    phase_contract = {
        "phase_export": {
            "logical_shape_hw": [478, 478],
            "logical_pixel_pitch_um": 17.0,
            "native_pixel_pitch_um": 8.0,
            "native_active_bounds_xyxy": contract["phase_slm"]["active_bounds_xyxy"],
            "flip_vertical_before_rasterization": contract["phase_slm"].get(
                "flip_vertical_before_export", False
            ),
            "flip_horizontal_before_rasterization": contract["phase_slm"].get(
                "flip_horizontal_before_export", False
            ),
        }
    }
    phase = decode_played_phase(
        root / "phase_to_play" / contract["phase_slm"]["file"], phase_contract
    )
    _, selected_device = _select_torch_device(device)
    simulator = PlayedBMPSimulator(root / "lab_model_config.yaml", phase, selected_device)
    detector_bounds = [list(map(int, value)) for value in contract["logical_geometry"]["detector_bounds_xyxy"]]
    reports: list[dict[str, Any]] = []
    for profile in profile_names:
        stage = root / profile
        rows = _read_csv(stage / "samples.csv")
        output = stage / "theoretical_ccd_played_bmp"
        marker = output / "ccd_feature_summary.json"
        if output.exists() and not overwrite:
            raise FileExistsError(f"CCD feature output already exists: {output}")
        if output.exists():
            if not marker.is_file():
                raise RuntimeError(f"Refusing to replace unowned directory: {output}")
            import shutil

            shutil.rmtree(output)
        raw_dir = output / "raw_linear_fp32"
        raw_dir.mkdir(parents=True)
        manifest: list[dict[str, Any]] = []
        for batch_index, batch in enumerate(_batches(rows, max(1, int(batch_size)))):
            amplitudes = np.stack(
                [_load_amplitude(stage / "amplitude_to_play" / row["amplitude_file"]) for row in batch]
            )
            theories = simulator(amplitudes)
            for row, theory in zip(batch, theories):
                key = row["key"]
                raw_path = raw_dir / f"{key}.npz"
                np.savez_compressed(raw_path, intensity=np.asarray(theory, dtype=np.float32))
                energies = _roi_energies(theory)
                prediction = int(np.argmax(energies))
                display = _save_displays(
                    output=output,
                    key=key,
                    intensity=theory,
                    detector_bounds=detector_bounds,
                )
                manifest.append(
                    {
                        "key": key,
                        "profile": profile,
                        "label": int(row["label"]),
                        "simulation_prediction": prediction,
                        "simulation_correct": prediction == int(row["label"]),
                        "raw_linear_npz": raw_path.relative_to(output).as_posix(),
                        "raw_linear_npz_sha256": _sha256(raw_path),
                        "raw_mean": float(theory.mean()),
                        "raw_std": float(theory.std()),
                        "raw_max": float(theory.max()),
                        **{f"detector_energy_{index}": float(value) for index, value in enumerate(energies)},
                        **display,
                    }
                )
            print(
                f"[mnist4 CCD features:{profile}] "
                f"{min((batch_index + 1) * batch_size, len(rows))}/{len(rows)}",
                flush=True,
            )
        _write_csv(output / "ccd_feature_manifest.csv", manifest)
        contact = _contact_sheet(output, manifest)
        summary = {
            "schema_version": 1,
            "profile": profile,
            "samples": len(manifest),
            "simulation_accuracy": float(np.mean([row["simulation_correct"] for row in manifest])),
            "simulation_source": "exact exported uint8 amplitude BMP and uint8 phase BMP",
            "scientific_data": "raw_linear_fp32/*.npz key=intensity, 478x478, no postprocess",
            "display_only": [
                "display_grayscale",
                "display_viridis",
                "display_detector_overlay",
                "display_binary",
                contact.name,
            ],
            "ccd_classifier": "four untouched raw ROI sums followed by argmax",
            "background_subtraction": False,
            "normalization": False,
            "nonlinearity": False,
            "phase_sha256": contract["phase_slm"]["sha256"],
            "manifest_sha256": _sha256(output / "ccd_feature_manifest.csv"),
        }
        marker.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        reports.append(summary)
    report = {
        "schema_version": 1,
        "export_dir": str(root),
        "device": str(selected_device),
        "profiles": reports,
    }
    (root / "ccd_feature_export_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", required=True)
    parser.add_argument("--profile", action="append", dest="profiles")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    report = export_ccd_features(
        args.export_dir,
        profiles=args.profiles,
        device=args.device,
        batch_size=args.batch_size,
        overwrite=args.overwrite,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
