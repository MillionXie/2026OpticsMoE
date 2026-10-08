"""Compare HOLOEYE multi-file preload and one-file re-preload in one session."""
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

import four_image_flow as flow


ROOT = Path(r"E:\code\guest\2026OpticsMoE\ABO_I2I_Lab_DVP_8um\runs\abo_i2i_20260924")
OUT = ROOT / "10_sdk_preload_probe"


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    flow.BASE_CORNERS = np.float32([[995, 172], [4310, 172], [4303, 3440], [980, 3435]])
    health = ROOT / "05_brightness_health" / "patterns"
    black = health / "amplitude_000.bmp"
    white = health / "amplitude_255.bmp"
    old = ROOT / "01_four_query" / "amplitude" / "vision_router" / "a65bceb0d831b8b3.bmp"
    flat = health / "phase_flat_pi_inverted.bmp"
    router = ROOT / "01_four_query" / "phase" / "vision_router.bmp"
    amp = flow.HoloeyeSLM(flow.AMP_SDK, flow.AMP_BIN, (1920, 1080), None, True, True, 5)
    rows = []
    with flow.PhaseHDMI(flow.PHASE_SDK, flow.PHASE_LUT, settle_s=.8, pixel_format="rgba") as phase, amp, flow.Camera(flow.CAMERA_DLL) as camera:
        phase.show(flat)
        amp.preload_files([black, white])
        for label, path, exposure, phase_path, preload in (
            ("multi_black", black, 3000, flat, None),
            ("multi_white", white, 3000, flat, None),
            ("single_white", white, 3000, flat, [white]),
            ("single_router", old, 20000, router, [old]),
            ("single_white_repeat", white, 3000, flat, [white]),
        ):
            if preload is not None:
                phase.show(phase_path)
                amp.preload_files(preload)
            actual = camera.settings(exposure=exposure, gain=1.0)
            amp.display_file(path)
            time.sleep(.24)
            frames = [camera.capture()[0] for _ in range(5)]
            roi = flow.warp(frames[-1])
            Image.fromarray(roi).save(OUT / f"{label}.png")
            row = {"label": label, "amplitude_sha256": flow.sha(path),
                   "phase_sha256": flow.sha(phase_path), "actual_exposure_us": actual["exposure_us"],
                   "mean": float(roi.mean()), "p99": float(np.percentile(roi, 99)),
                   "maximum": int(roi.max())}
            rows.append(row)
            print(json.dumps(row), flush=True)
    flow.write(OUT / "report.json", {"status": "complete", "rows": rows})


if __name__ == "__main__":
    main()
