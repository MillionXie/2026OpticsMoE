"""Check whether each pinned ABO phase BMP visibly changes a fixed CCD field."""
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
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-run", required=True, type=Path)
    p.add_argument("--flat-phase", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--exposure-us", type=float, default=60000)
    p.add_argument("--corners-tltrbrbl", nargs=8, type=float, required=True)
    a = p.parse_args()
    if not 100 <= a.exposure_us <= 80000:
        raise ValueError("Exposure outside bounded probe")
    source = a.source_run.resolve()
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    flow.BASE_CORNERS = np.asarray(a.corners_tltrbrbl, np.float32).reshape(4, 2)
    prior = json.loads((source / "report.json").read_text(encoding="utf-8"))
    if prior["status"] != "complete":
        raise ValueError("Source run incomplete")
    if not np.array_equal(flow.BASE_CORNERS, np.asarray(prior["base_corners_screen_TL_TR_BR_BL"], np.float32)):
        raise ValueError("Camera corner mismatch with source run")
    sample_id = prior["samples"][0]["sample_id"]
    flat = a.flat_phase.resolve()
    if not flat.is_file():
        raise FileNotFoundError(flat)
    sim = np.load(source / "simulation_reference.npz")
    rows = []
    with flow.Bench(out, a.exposure_us, 240, {}) as bench:
        for stage in STAGES:
            amp = source / "amplitude" / stage / f"{sample_id}.bmp"
            phase = source / "phase" / f"{stage}.bmp"
            orientation = prior["stage_calibration"][stage]["camera_orientation"]
            measured = {}
            for label, phase_file in (("trained", phase), ("flat", flat), ("repeat", phase)):
                images, receipt = bench.capture(stage, phase_file, [amp], [sample_id], orientation, save=False)
                measured[label] = images[0]
                Image.fromarray(images[0]).save(out / f"{stage}_{label}.png")
            row = {"stage": stage,
                   "trained_vs_flat_pcc": flow.pcc(measured["trained"], measured["flat"]),
                   "trained_repeat_pcc": flow.pcc(measured["trained"], measured["repeat"]),
                   "trained_vs_sim_pcc": flow.pcc(measured["trained"], sim[stage][0]),
                   "flat_vs_sim_pcc": flow.pcc(measured["flat"], sim[stage][0]),
                   "exposure_us": bench.settings["exposure_us"],
                   "trained_phase_sha256": flow.sha(phase),
                   "flat_phase_sha256": flow.sha(flat),
                   "amplitude_sha256": flow.sha(amp)}
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    flow.write(out / "report.json", {"status": "complete", "source_run": str(source),
                                     "corners_TL_TR_BR_BL": flow.BASE_CORNERS.tolist(),
                                     "rows": rows})


if __name__ == "__main__":
    main()
