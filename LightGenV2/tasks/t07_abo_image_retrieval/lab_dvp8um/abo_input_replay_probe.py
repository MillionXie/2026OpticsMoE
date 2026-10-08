"""Replay the exact first query BMP from the four-query and invalid full runs."""
import json
from pathlib import Path

import numpy as np
from PIL import Image

import four_image_flow as flow


ROOT = Path(r"E:\code\guest\2026OpticsMoE\ABO_I2I_Lab_DVP_8um\runs\abo_i2i_20260924")
FOUR = ROOT / "01_four_query"
FULL = ROOT / "06_offline_input_audit"
OUT = ROOT / "08_input_replay_desktop_off"
SAMPLE = "a65bceb0d831b8b3"


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    flow.BASE_CORNERS = np.float32([[995, 172], [4310, 172], [4303, 3440], [980, 3435]])
    phase = FOUR / "phase" / "vision_router.bmp"
    old = FOUR / "amplitude" / "vision_router" / f"{SAMPLE}.bmp"
    new = FULL / "amplitude" / "vision_router" / f"{SAMPLE}.bmp"
    rows = []
    frames = {}
    with flow.Bench(OUT, 20000, 240, {}) as bench:
        for label, path in (("four_first", old), ("full", new), ("four_repeat", old)):
            values, receipt = bench.capture("vision_router", phase, [path], [SAMPLE], "flip_v", save=False)
            image = values[0]
            frames[label] = image
            Image.fromarray(image).save(OUT / f"{label}.png")
            row = {"label": label, "input_sha256": flow.sha(path), "phase_sha256": flow.sha(phase),
                   "p99": float(np.percentile(image, 99)), "maximum": int(image.max()),
                   "mean": float(image.mean()), "saturated_fraction": float(np.mean(image == 255)),
                   "exposure_us": bench.settings["exposure_us"]}
            rows.append(row)
            print(json.dumps(row), flush=True)
    report = {"status": "complete", "rows": rows,
              "four_repeat_pcc": flow.pcc(frames["four_first"], frames["four_repeat"]),
              "four_vs_full_pcc": flow.pcc(frames["four_first"], frames["full"])}
    flow.write(OUT / "report.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
