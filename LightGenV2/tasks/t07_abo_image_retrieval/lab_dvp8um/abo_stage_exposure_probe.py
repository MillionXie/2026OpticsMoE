"""Read-only-input exposure probe for six pinned ABO optical stages.

Reuses one pre-generated amplitude BMP and one selected phase BMP per stage.
Only exposure changes; no stage inputs, orientations, LUT, or camera geometry
are selected from the results. Never writes into the source four-query run.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

import four_image_flow as flow


STAGES = (
    "vision_router", "vision_expert", "vision_global",
    "language_router", "language_expert", "language_global",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--exposures-us", nargs="+", type=float, default=[20000, 40000, 60000, 80000])
    parser.add_argument("--corners-tltrbrbl", nargs=8, type=float, required=True)
    args = parser.parse_args()
    if any(not 100 <= e <= 80000 for e in args.exposures_us):
        raise ValueError("Exposure outside bounded 100..80000 us probe")
    source = args.source_run.resolve()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    flow.BASE_CORNERS = np.asarray(args.corners_tltrbrbl, dtype=np.float32).reshape(4, 2)
    prior = json.loads((source / "report.json").read_text(encoding="utf-8"))
    if prior["status"] != "complete":
        raise ValueError("Source four-query run is incomplete")
    if not np.array_equal(flow.BASE_CORNERS, np.asarray(prior["base_corners_screen_TL_TR_BR_BL"], np.float32)):
        raise ValueError("Camera corner mismatch with source run")
    sample_id = prior["samples"][0]["sample_id"]
    sim = np.load(source / "simulation_reference.npz")
    rows = []
    with flow.Bench(out, args.exposures_us[0], 240, {}) as bench:
        for stage in STAGES:
            phase_path = source / "phase" / f"{stage}.bmp"
            amplitude_path = source / "amplitude" / stage / f"{sample_id}.bmp"
            if not phase_path.is_file() or not amplitude_path.is_file():
                raise FileNotFoundError(f"Missing pinned stage files for {stage}")
            orientation = prior["stage_calibration"][stage]["camera_orientation"]
            for exposure in args.exposures_us:
                bench.settings = bench.camera.settings(exposure=exposure, gain=1.0)
                measured, receipt = bench.capture(stage, phase_path, [amplitude_path],
                                                   [sample_id], orientation, save=False)
                frame = measured[0]
                Image.fromarray(frame).save(out / f"{stage}_e{int(exposure):05d}.png")
                row = {"stage": stage, "sample_id": sample_id,
                       "requested_exposure_us": exposure,
                       "actual_exposure_us": bench.settings["exposure_us"],
                       "pcc_to_simulation": flow.pcc(frame, sim[stage][0]),
                       "mean": float(frame.mean()),
                       "p99": float(np.percentile(frame, 99)),
                       "maximum": int(frame.max()),
                       "saturated_fraction": float(np.mean(frame == 255)),
                       "orientation": orientation,
                       "phase_sha256": flow.sha(phase_path),
                       "amplitude_sha256": flow.sha(amplitude_path),
                       "phase_receipt": receipt}
                rows.append(row)
                print(json.dumps(row, ensure_ascii=False), flush=True)
    flow.write(out / "report.json", {"status": "complete", "source_run": str(source),
               "corners_TL_TR_BR_BL": flow.BASE_CORNERS.tolist(), "rows": rows,
               "no_per_image_photometric_normalization": True})


if __name__ == "__main__":
    main()
