from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from experiments.hardware_sdk.workflows.reconstruct_slm import (
    reconstruct_directory,
    save_active_png,
)

from .settings import load_settings


def build(
    config: Path,
    input_dir: Path,
    routing_csv: Path,
    output_dir: Path,
) -> dict[str, object]:
    settings = load_settings(config)
    input_dir = input_dir.expanduser().resolve()
    routing_csv = routing_csv.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    with routing_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError("routing.csv is empty")
    by_name = {
        path.name: path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".png", ".bmp", ".tif", ".tiff"}
    }
    by_stem = {path.stem: path for path in by_name.values()}
    compact = output_dir / "compact_amplitude"
    compact.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, object]] = []
    starts = ((0, 0), (0, 254), (254, 0), (254, 254))
    for row in rows:
        ccd_name = str(row["filename"])
        source = by_name.get(ccd_name) or by_stem.get(Path(ccd_name).stem)
        if source is None:
            raise FileNotFoundError(
                f"No 224x224 central router input matches CCD {ccd_name!r}"
            )
        with Image.open(source) as image:
            amplitude = np.asarray(image.convert("L"), dtype=np.float32)
        if amplitude.shape != (settings.expert_size, settings.expert_size):
            raise RuntimeError(
                f"Central router amplitude must be 224x224: {source} has {amplitude.shape}"
            )
        weights = np.asarray(
            [float(row[f"weight_{index}"]) for index in range(4)], dtype=np.float32
        )
        active = np.zeros((settings.active_size, settings.active_size), dtype=np.float32)
        for index, (top, left) in enumerate(starts):
            active[
                top : top + settings.expert_size,
                left : left + settings.expert_size,
            ] = amplitude * weights[index]
        encoded = np.rint(np.clip(active, 0.0, 255.0)).astype(np.uint8)
        destination = compact / f"{Path(ccd_name).stem}.png"
        save_active_png(encoded, destination)
        manifest.append(
            {
                "filename": destination.name,
                "central_input": source.name,
                "selected_experts": row["selected_experts"],
                **{f"amplitude_weight_{i}": float(weights[i]) for i in range(4)},
                "sum_amplitude_weight_squared": float(np.sum(weights**2)),
            }
        )
    with (output_dir / "routed_amplitude_manifest.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    reconstruction = reconstruct_directory(
        compact,
        output_dir / "amplitude_to_play",
        slm_size_wh=(
            settings.hardware_amplitude_slm_width,
            settings.hardware_amplitude_slm_height,
        ),
        scale_factor=None,
        center_xy=(
            settings.hardware_amplitude_slm_center_x,
            settings.hardware_amplitude_slm_center_y,
        ),
        logical_pixel_pitch_um=settings.language_optical_pixel_pitch_um,
        slm_pixel_pitch_um=settings.hardware_amplitude_slm_pixel_pitch_um,
    )
    report = {
        "schema_version": 1,
        "images": len(manifest),
        "source_is_central_224_amplitude": True,
        "routed_layout": "canonical MoE4 active478: [0:224] and [254:478] per axis",
        "weight_domain": "amplitude",
        "weight_normalization": settings.router_weight_normalization,
        "reconstruction": reconstruction,
    }
    (output_dir / "routed_amplitude_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the original 2x2 expert amplitude from optical-router scores"
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--routing-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    report = build(args.config, args.input_dir, args.routing_csv, args.output_dir)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
