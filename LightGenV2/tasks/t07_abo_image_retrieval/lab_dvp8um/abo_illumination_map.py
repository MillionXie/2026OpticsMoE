"""Six-pattern illumination check at the ABO logical 478x478 aperture."""
import json
from pathlib import Path

import numpy as np
from PIL import Image

import four_image_flow as flow


ROOT = Path(r"E:\code\guest\2026OpticsMoE\ABO_I2I_Lab_DVP_8um\runs\abo_i2i_20260924")
OUT = ROOT / "09_illumination_map"
FLAT = ROOT.parent / "mnist_20260924" / "02_exposure_scan" / "patterns" / "phase_flat_pi_inverted.bmp"


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    flow.BASE_CORNERS = np.float32([[995, 172], [4310, 172], [4303, 3440], [980, 3435]])
    patterns = {}
    for name, box in (
        ("full", (0, 0, 478, 478)),
        ("router_center", (127, 127, 351, 351)),
        ("expert_TL", (0, 0, 224, 224)),
        ("expert_TR", (254, 0, 478, 224)),
        ("expert_BL", (0, 254, 224, 478)),
        ("expert_BR", (254, 254, 478, 478)),
    ):
        x0, y0, x1, y1 = box
        active = np.zeros((478, 478), np.uint8)
        active[y0:y1, x0:x1] = 255
        path = OUT / f"{name}.bmp"
        Image.fromarray(flow.active_to_native(active)).save(path)
        patterns[name] = path
    rows = []
    with flow.Bench(OUT, 3000, 240, {}) as bench:
        for name, path in patterns.items():
            values, receipt = bench.capture("illumination", FLAT, [path], [name], "identity", save=False)
            frame = values[0]
            Image.fromarray(frame).save(OUT / f"ccd_{name}.png")
            row = {"pattern": name, "amplitude_sha256": flow.sha(path),
                   "phase_sha256": flow.sha(FLAT),
                   "exposure_us": bench.settings["exposure_us"],
                   "mean": float(frame.mean()), "p99": float(np.percentile(frame, 99)),
                   "maximum": int(frame.max()),
                   "saturated_fraction": float(np.mean(frame == 255))}
            rows.append(row)
            print(json.dumps(row), flush=True)
    flow.write(OUT / "report.json", {"status": "complete", "rows": rows,
                                     "corners_TL_TR_BR_BL": flow.BASE_CORNERS.tolist()})


if __name__ == "__main__":
    main()
