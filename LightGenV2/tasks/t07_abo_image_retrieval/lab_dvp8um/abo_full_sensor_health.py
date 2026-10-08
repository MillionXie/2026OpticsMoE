"""Compare full-sensor black/white images to distinguish dark optics from wrong ROI."""
import json
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import four_image_flow as flow


ROOT = Path(r"E:\code\guest\2026OpticsMoE\ABO_I2I_Lab_DVP_8um\runs\abo_i2i_20260924")
OUT = ROOT / "11_full_sensor_health"


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    flow.BASE_CORNERS = np.float32([[995, 172], [4310, 172], [4303, 3440], [980, 3435]])
    patterns = ROOT / "05_brightness_health" / "patterns"
    black = patterns / "amplitude_000.bmp"
    white = patterns / "amplitude_255.bmp"
    flat = patterns / "phase_flat_pi_inverted.bmp"
    amp = flow.HoloeyeSLM(flow.AMP_SDK, flow.AMP_BIN, (1920, 1080), None, True, True, 5)
    frames = {}
    rows = []
    with flow.PhaseHDMI(flow.PHASE_SDK, flow.PHASE_LUT, settle_s=.8, pixel_format="rgba") as phase, amp, flow.Camera(flow.CAMERA_DLL) as camera:
        phase.show(flat)
        amp.preload_files([black, white])
        actual = camera.settings(exposure=3000, gain=1.0)
        for label, path in (("black", black), ("white", white)):
            amp.display_file(path)
            time.sleep(.24)
            frames[label] = [camera.capture()[0] for _ in range(5)][-1]
    for label, frame in frames.items():
        roi = flow.warp(frame)
        small = cv2.resize(frame, (1370, 912), interpolation=cv2.INTER_AREA)
        Image.fromarray(small).save(OUT / f"{label}_full_sensor_4x.png")
        Image.fromarray(roi).save(OUT / f"{label}_roi.png")
        ys, xs = np.where(frame > 20)
        rows.append({"label": label, "amplitude_sha256": flow.sha(black if label == "black" else white),
                     "actual_exposure_us": actual["exposure_us"],
                     "shape": list(frame.shape), "full_p99": float(np.percentile(frame, 99)),
                     "full_maximum": int(frame.max()),
                     "full_fraction_gt20": float(np.mean(frame > 20)),
                     "full_bbox_gt20_xyxy": None if not len(xs) else [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
                     "roi_p99": float(np.percentile(roi, 99)), "roi_maximum": int(roi.max())})
    flow.write(OUT / "report.json", {"status": "complete", "rows": rows})
    print(json.dumps(rows, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
