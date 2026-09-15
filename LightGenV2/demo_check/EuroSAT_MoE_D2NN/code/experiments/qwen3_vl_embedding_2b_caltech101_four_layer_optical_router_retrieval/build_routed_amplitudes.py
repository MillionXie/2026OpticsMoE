from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from experiments.hardware_sdk.workflows.reconstruct_slm import (
    reconstruct_directory,
    save_active_png,
)

from .settings import load_settings
from .hardware_contract import (
    read_csv,
    require_empty_directory,
    sha256_file,
    unique_image_files,
)
from .router import sparsify_probabilities


def build(
    config: Path,
    input_dir: Path,
    routing_csv: Path,
    output_dir: Path,
) -> dict[str, object]:
    settings = load_settings(config)
    if settings.router_backend != "optical":
        raise ValueError("Measured routed amplitudes require an optical-router config")
    if bool(settings.hardware_amplitude_invert_before_export) or (
        int(settings.hardware_amplitude_bright_value_uint8),
        int(settings.hardware_amplitude_dark_value_uint8),
    ) != (255, 0):
        raise RuntimeError(
            "This builder emits the audited corrected Meadowlark polarity "
            "255=bright, 0=dark; the config requests a different polarity"
        )
    input_dir = input_dir.expanduser().resolve()
    routing_csv = routing_csv.expanduser().resolve()
    if not routing_csv.is_file():
        raise FileNotFoundError(f"routing.csv is missing: {routing_csv}")
    output_dir = require_empty_directory(output_dir, label="routed amplitude output")
    rows = read_csv(routing_csv)
    input_paths = unique_image_files(input_dir)
    by_name = {path.name: path for path in input_paths}
    by_stem = {path.stem: path for path in by_name.values()}
    routing_stems = [Path(str(row["filename"])).stem for row in rows]
    if len(routing_stems) != len(set(routing_stems)):
        raise RuntimeError("routing.csv contains duplicate sample stems")
    if set(routing_stems) != set(by_stem):
        raise RuntimeError(
            "Central Router input/routing sample sets differ: "
            f"missing_inputs={sorted(set(routing_stems).difference(by_stem))[:8]}, "
            f"unused_inputs={sorted(set(by_stem).difference(routing_stems))[:8]}"
        )
    compact = output_dir / "compact_amplitude"
    compact.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, object]] = []
    pitch = int(settings.expert_pitch)
    starts = ((0, 0), (0, pitch), (pitch, 0), (pitch, pitch))
    for row in rows:
        ccd_name = str(row["filename"])
        source = by_name.get(ccd_name) or by_stem.get(Path(ccd_name).stem)
        if source is None:
            raise FileNotFoundError(
                f"No 224x224 central router input matches CCD {ccd_name!r}"
            )
        with Image.open(source) as image:
            if image.mode != "L":
                raise RuntimeError(
                    f"Central Router amplitude must be uint8 mode L: {source} "
                    f"has mode {image.mode!r}"
                )
            source_array = np.asarray(image)
        if source_array.dtype != np.uint8:
            raise RuntimeError(f"Central Router amplitude must be uint8: {source}")
        amplitude = source_array.astype(np.float32)
        if amplitude.shape != (settings.expert_size, settings.expert_size):
            raise RuntimeError(
                f"Central router amplitude must be 224x224: {source} has {amplitude.shape}"
            )
        weights = np.asarray(
            [float(row[f"weight_{index}"]) for index in range(4)], dtype=np.float32
        )
        probabilities = torch.tensor(
            [[float(row[f"probability_{index}"]) for index in range(4)]],
            dtype=torch.float64,
        )
        if not np.isfinite(weights).all() or np.any(weights < 0.0):
            raise RuntimeError(f"Routing weights are nonfinite or negative for {ccd_name}")
        if not bool(torch.isfinite(probabilities).all()) or bool(
            (probabilities < 0).any()
        ):
            raise RuntimeError(
                f"Routing probabilities are nonfinite or negative for {ccd_name}"
            )
        torch.testing.assert_close(
            probabilities.sum(dim=-1),
            torch.ones(1, dtype=probabilities.dtype),
            rtol=1.0e-5,
            atol=1.0e-6,
        )
        expected_weights, expected_selected, expected_indices = sparsify_probabilities(
            probabilities,
            int(settings.top_k),
            normalization=settings.router_weight_normalization,
            straight_through=False,
            eps=settings.optical_router_energy_eps,
        )
        torch.testing.assert_close(
            torch.from_numpy(weights).double(),
            expected_weights[0],
            rtol=1.0e-4,
            atol=1.0e-6,
            msg=f"Routing weights do not match probabilities for {ccd_name}",
        )
        selected_text = [
            str(row[f"selected_{index}"]).strip().lower() == "true"
            for index in range(4)
        ]
        selected_indices = [
            int(value) for value in str(row["selected_experts"]).split(",")
        ]
        if selected_text != expected_selected[0].tolist() or selected_indices != expected_indices[0].tolist():
            raise RuntimeError(
                f"Routing selected experts do not match probabilities for {ccd_name}"
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
                "central_input_sha256": sha256_file(source),
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
        "config": str(Path(config).expanduser().resolve()),
        "config_sha256": sha256_file(Path(config)),
        "routing_csv": str(routing_csv),
        "routing_csv_sha256": sha256_file(routing_csv),
        "images": len(manifest),
        "source_is_central_224_amplitude": True,
        "routed_layout": (
            f"canonical MoE4 active{settings.active_size}: [0:{settings.expert_size}] "
            f"and [{pitch}:{pitch + settings.expert_size}] per axis"
        ),
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
