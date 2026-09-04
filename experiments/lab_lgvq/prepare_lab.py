"""Build Meadowlark/TUCam runtime files from the one operator config.

The audited implementation is shared with :mod:`experiments.lab_qwen` so that
ROI alignment, homography orientation, LUT hashing, exposure and timing rules
remain identical on the same laboratory bench.  This wrapper deliberately
keeps LGVQ results under ``experiments/lab_lgvq``.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import yaml

from experiments.lab_qwen.prepare_lab import prepare_lab


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="experiments/lab_lgvq/LAB_CONFIG.yaml")
    args = parser.parse_args()
    lab_directory = Path(__file__).resolve().parent
    repository = lab_directory.parents[1]
    lab_config = Path(args.config).expanduser().resolve()
    operator = yaml.safe_load(lab_config.read_text(encoding="utf-8")) or {}
    phase = operator.get("phase_slm", {})
    if not isinstance(phase, dict):
        raise ValueError("LAB_CONFIG.yaml: phase_slm must be a mapping")
    center = phase.get("center_xy", [980.0, 590.0])
    if not isinstance(center, (list, tuple)) or len(center) != 2:
        raise ValueError("LAB_CONFIG.yaml: phase_slm.center_xy must be [x,y]")
    center = [float(center[0]), float(center[1])]
    if not all(math.isfinite(value) for value in center):
        raise ValueError("LAB_CONFIG.yaml: phase_slm.center_xy must be finite")
    flip_vertical = bool(phase.get("flip_vertical_before_raster", True))
    flip_horizontal = bool(phase.get("flip_horizontal_before_raster", False))
    report = prepare_lab(
        lab_config,
        template_path=lab_directory / "internal/hardware_template.yaml",
        output_dir=lab_directory / "generated",
        repo_root=repository,
    )
    # Keep the generated acquisition configs and the model-side phase exporter
    # on the exact same operator-owned coordinate contract.
    for name in ("bootstrap_hardware.yaml", "formal_hardware.yaml"):
        runtime_path = lab_directory / "generated" / name
        if not runtime_path.is_file():
            continue
        runtime = yaml.safe_load(runtime_path.read_text(encoding="utf-8")) or {}
        runtime.setdefault("phase_slm", {})["center_xy"] = center
        runtime["phase_slm"]["flip_vertical_before_raster"] = flip_vertical
        runtime["phase_slm"]["flip_horizontal_before_raster"] = flip_horizontal
        runtime_path.write_text(
            yaml.safe_dump(runtime, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
    report["phase_slm_export"] = {
        "center_xy": center,
        "flip_vertical_before_raster": flip_vertical,
        "flip_horizontal_before_raster": flip_horizontal,
        "source": str(lab_config),
    }
    if report.get("formal_config"):
        report["next_command"] = (
            "python -m experiments.hardware_sdk.workflows.roi_calibration "
            "exposure --config experiments/lab_lgvq/generated/formal_hardware.yaml"
        )
        report_path = lab_directory / "generated" / "prepare_report.json"
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
