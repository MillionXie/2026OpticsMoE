"""Export all four trained logical phase masks to the native phase SLM BMP."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.hardware_sdk.workflows.reconstruct_slm import (
    reconstruct_directory,
    save_active_png,
)
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.io_utils import (
    write_json,
)

from .hardware_bridge import STAGES, _load_model, _phase_for_stage
from .settings import load_settings


def export_all(settings, checkpoint: Path, output_dir: Path) -> dict[str, object]:
    checkpoint = checkpoint.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    compact_dir = output_dir / "compact_phase"
    native_dir = output_dir / "phase_bmp"
    compact_dir.mkdir(parents=True, exist_ok=True)
    loaded, replacement, readout = _load_model(settings, checkpoint)
    del loaded, readout
    try:
        for stage in STAGES:
            save_active_png(
                _phase_for_stage(replacement, stage, settings),
                compact_dir / f"{stage}.png",
            )
    finally:
        replacement.close()
    reconstruction = reconstruct_directory(
        compact_dir,
        native_dir,
        slm_size_wh=(
            settings.hardware_phase_slm_width,
            settings.hardware_phase_slm_height,
        ),
        scale_factor=None,
        center_xy=(
            settings.hardware_phase_slm_center_x,
            settings.hardware_phase_slm_center_y,
        ),
        logical_pixel_pitch_um=settings.language_optical_pixel_pitch_um,
        slm_pixel_pitch_um=settings.hardware_phase_slm_pixel_pitch_um,
    )
    report = {
        "schema_version": 1,
        "checkpoint": str(checkpoint),
        "stages": list(STAGES),
        "logical_phase_shape": [settings.active_size, settings.active_size],
        "logical_pixel_pitch_um": settings.language_optical_pixel_pitch_um,
        "phase_slm": {
            "size_wh": [
                settings.hardware_phase_slm_width,
                settings.hardware_phase_slm_height,
            ],
            "pixel_pitch_um": settings.hardware_phase_slm_pixel_pitch_um,
            "center_xy": [
                settings.hardware_phase_slm_center_x,
                settings.hardware_phase_slm_center_y,
            ],
            "flip_vertical_before_raster": settings.hardware_phase_flip_vertical,
            "flip_horizontal_before_raster": settings.hardware_phase_flip_horizontal,
        },
        "reconstruction": reconstruction,
    }
    write_json(output_dir / "phase_export_report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    settings = load_settings(args.config)
    report = export_all(
        settings,
        Path(args.checkpoint),
        Path(args.output_dir),
    )
    print(
        f"Exported {len(report['stages'])} native phase BMPs to "
        f"{Path(args.output_dir).expanduser().resolve() / 'phase_bmp'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
