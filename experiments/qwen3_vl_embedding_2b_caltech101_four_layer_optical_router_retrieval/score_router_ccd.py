from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from .router import sparsify_probabilities
from .settings import load_settings


IMAGE_SUFFIXES = {".png", ".bmp", ".tif", ".tiff"}


def _read_intensity(path: Path, expected_size: int) -> np.ndarray:
    with Image.open(path) as image:
        array = np.asarray(image)
    if array.ndim == 3:
        array = array[..., :3].astype(np.float64).mean(axis=-1)
    if array.ndim != 2 or array.shape != (expected_size, expected_size):
        raise RuntimeError(
            f"Router CCD must already be canonical {expected_size}x{expected_size}: "
            f"{path} has {array.shape}"
        )
    value = array.astype(np.float64)
    if not np.isfinite(value).all() or np.any(value < 0.0):
        raise RuntimeError(f"Router CCD must be finite and nonnegative: {path}")
    return value


def score_directory(config: Path, input_dir: Path, output_dir: Path) -> dict[str, object]:
    settings = load_settings(config)
    paths = sorted(
        path
        for path in input_dir.expanduser().resolve().iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not paths:
        raise FileNotFoundError(f"No router CCD images found in {input_dir}")
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    intervals = settings.optical_router_detector_intervals
    rows: list[dict[str, object]] = []
    for path in paths:
        value = _read_intensity(path, settings.active_size)
        energies = []
        for top, bottom in intervals:
            for left, right in intervals:
                energies.append(float(value[top:bottom, left:right].sum()))
        energy = torch.tensor(energies, dtype=torch.float64).unsqueeze(0)
        fractions = energy / energy.sum(dim=-1, keepdim=True).clamp_min(
            settings.optical_router_energy_eps
        )
        if settings.optical_router_score_normalization == "standardized_region_energy":
            centered = energy - energy.mean(dim=-1, keepdim=True)
            logits = centered / centered.square().mean(
                dim=-1, keepdim=True
            ).add(settings.optical_router_energy_eps).sqrt()
        else:
            logits = fractions.clamp_min(settings.optical_router_energy_eps).log()
        probabilities = torch.softmax(logits / settings.router_temperature, dim=-1)
        weights, selected, indices = sparsify_probabilities(
            probabilities,
            settings.top_k,
            normalization=settings.router_weight_normalization,
            straight_through=False,
            eps=settings.optical_router_energy_eps,
        )
        captured = sum(energies) / max(float(value.sum()), settings.optical_router_energy_eps)
        row: dict[str, object] = {
            "filename": path.name,
            "selected_experts": ",".join(map(str, indices[0].tolist())),
            "capture_fraction": captured,
            "saturated_pixel_fraction": float(
                np.mean(value >= value.max()) if value.max() > 0 else 0.0
            ),
        }
        for index in range(4):
            row[f"energy_{index}"] = energies[index]
            row[f"energy_fraction_{index}"] = float(fractions[0, index])
            row[f"probability_{index}"] = float(probabilities[0, index])
            row[f"weight_{index}"] = float(weights[0, index])
            row[f"selected_{index}"] = bool(selected[0, index])
        rows.append(row)
    csv_path = output_dir / "routing.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    with Image.open(paths[0]) as source:
        preview = source.convert("RGB")
    draw = ImageDraw.Draw(preview)
    index = 0
    for top, bottom in intervals:
        for left, right in intervals:
            draw.rectangle((left, top, right - 1, bottom - 1), outline="red", width=3)
            draw.text((left + 3, top + 3), str(index), fill="red")
            index += 1
    preview_path = output_dir / "first_ccd_detector_overlay.png"
    preview.save(preview_path)
    report = {
        "schema_version": 1,
        "config": str(Path(config).expanduser().resolve()),
        "input_dir": str(input_dir.expanduser().resolve()),
        "images": len(rows),
        "canonical_ccd_size": settings.active_size,
        "top_k": settings.top_k,
        "weight_normalization": settings.router_weight_normalization,
        "score_normalization": settings.optical_router_score_normalization,
        "background_subtraction": False,
        "routing_csv": str(csv_path),
        "preview": str(preview_path),
        "mean_capture_fraction": float(
            np.mean([float(row["capture_fraction"]) for row in rows])
        ),
    }
    (output_dir / "routing_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Score canonical router CCD frames")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    report = score_directory(args.config, args.input_dir, args.output_dir)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
