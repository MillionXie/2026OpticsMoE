"""Bounded MNIST-4 physical checks with the 2026-09-24 DVP camera corners.

All CCD numbers are linear Mono8 values. There is one geometric warp and no
per-image normalization, background subtraction, denoising, or adaptive ROI.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]
OLD = ROOT / "ABO_Lab_SHS_8um"
ADAPTER = ROOT / "ABO_I2I_DVP_adapter_20260922"
sys.path[:0] = [str(OLD), str(ADAPTER)]
from lab_dvp import Camera  # noqa: E402
from phase_hdmi import PhaseHDMI  # noqa: E402
from vendor_driver import HoloeyeSLM  # noqa: E402

PACKAGE = ROOT.parent / "MNIST_10cm_8um_Bench_Test_20260923" / "02_mnist_10cm"
PHASE_B = PACKAGE / "phase" / "B_RECOMMENDED_native8_best.bmp"
PHASE_B_SHA = "9a5db11329f1fe397d7bed40f348651321850e1a11309e4ecf0a79690aa9aa54"
PHASE_A = PACKAGE / "phase" / "A_old_post_robust_resampled_8um.bmp"
PHASE_A_SHA = "395d2c60abb72ae9735d47e1fc7ec303c370662127f7571f5dbd24ba45fb9c0b"
FLAT_PHASE = ROOT / "ABO_I2I_Lab_DVP_8um" / "runs" / "mnist_20260924" / "02_exposure_scan" / "patterns" / "phase_flat_pi_inverted.bmp"
FLAT_PHASE_SHA = "821c30142c98faec439ab8a47e5c32b0ffe0f10794bff252ec0d1252635317db"
CAMERA_DLL = ROOT.parent / "小相机/SDK二次开发包/DVP2  SDK 中性版本/DVP2 SDK/library/Visual C++/bin/x64/DVPCamera64.dll"
PHASE_SDK = Path(r"C:\Program Files\Meadowlark Optics\Blink 1920 HDMI\SDK")
PHASE_LUT = Path(r"C:\Program Files\Meadowlark Optics\Blink 1920 HDMI\LUT Files\19x12_8bit_linearVoltage.lut")
AMP_SDK = OLD / "vendor/holoeye_python"
AMP_BIN = Path(r"C:\Program Files\HOLOEYE SLM SDK SlideshowPlayer 2.0")
CORNER_TXT = ROOT.parent / "20260924.txt"
EXPECTED_CORNERS = np.float32([[995, 172], [4310, 172], [4303, 3440], [980, 3435]])  # TL TR BR BL
BOUNDS = [(162, 162, 221, 221), (257, 162, 316, 221), (162, 257, 221, 316), (257, 257, 316, 316)]
EXAMPLES = ["mnist_i00013_y0.bmp", "mnist_i00014_y1.bmp", "mnist_i00038_y2.bmp", "mnist_i00032_y3.bmp"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pcc(a, b):
    x = np.asarray(a, np.float64).ravel()
    y = np.asarray(b, np.float64).ravel()
    x -= x.mean()
    y -= y.mean()
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    return float(x.dot(y) / denom) if denom else None


def canonical(frame):
    dst = np.float32([[0, 0], [477, 0], [477, 477], [0, 477]])
    h = cv2.getPerspectiveTransform(EXPECTED_CORNERS, dst)
    return cv2.warpPerspective(frame, h, (478, 478), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def stats(roi):
    a = roi.astype(np.float32)
    energies = [float(a[y0:y1, x0:x1].sum()) for x0, y0, x1, y1 in BOUNDS]
    return dict(mean=float(a.mean()), p99=float(np.percentile(a, 99)), maximum=int(a.max()),
                saturated_fraction=float(np.mean(a == 255)), energies=energies,
                prediction=int(np.argmax(energies)))


def write(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def acquire(camera, amplitude, wait_ms, out, name, save_raw=False):
    t = time.perf_counter()
    amplitude.display_file(name)
    visible_ms = (time.perf_counter() - t) * 1000
    time.sleep(wait_ms / 1000)
    frames = []
    metas = []
    for _ in range(5):
        frame, meta = camera.capture()
        frames.append(frame)
        metas.append(meta)
    first = canonical(frames[0])
    last = canonical(frames[-1])
    if save_raw:
        Image.fromarray(frames[-1]).save(out / "raw_first.png")
    return first, last, dict(visible_ms=visible_ms, requested_wait_ms=wait_ms,
                             first_frame=metas[0], last_frame=metas[-1],
                             first_vs_last_pcc=pcc(first, last), **stats(last))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["phase", "exposure", "timing", "mnist"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--exposure-us", type=float, default=10000)
    parser.add_argument("--wait-ms", type=float, default=240)
    parser.add_argument("--phase-variant", choices=["A", "B", "flat"], default="B")
    args = parser.parse_args()
    if not 0 <= args.wait_ms <= 1000 or not 100 <= args.exposure_us <= 50000:
        raise ValueError("Unbounded exposure or wait")
    phase_path, phase_expected_sha = {"A": (PHASE_A, PHASE_A_SHA),
                                      "B": (PHASE_B, PHASE_B_SHA),
                                      "flat": (FLAT_PHASE, FLAT_PHASE_SHA)}[args.phase_variant]
    if sha(phase_path) != phase_expected_sha:
        raise ValueError(f"Phase {args.phase_variant} hash mismatch")
    if args.mode == "phase" and args.phase_variant != "B":
        raise ValueError("Phase switching probe is pinned to B/flat/B")
    if not CORNER_TXT.is_file() or not all(s in CORNER_TXT.read_text(encoding="utf-8") for s in
                                       ("TL [995,172]", "TR [4310,172]", "BL [980,3435]", "BR [4303,3440]")):
        raise ValueError("20260924 corner file does not match pinned values")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    files = sorted((PACKAGE / "inputs_40_fixed").glob("mnist_i*_y*.bmp"))
    if len(files) != 40:
        raise ValueError("Expected exactly 40 fixed inputs")
    if args.mode == "phase":
        if not FLAT_PHASE.is_file():
            raise FileNotFoundError(FLAT_PHASE)
        files = [PACKAGE / "inputs_40_fixed" / EXAMPLES[0]]
    elif args.mode == "exposure":
        files = [PACKAGE / "inputs_40_fixed" / n for n in EXAMPLES]
    elif args.mode == "timing":
        files = [PACKAGE / "inputs_40_fixed" / n for n in EXAMPLES[:2]]
    for path in files:
        with Image.open(path) as im:
            if im.format != "BMP" or im.mode != "L" or im.size != (1920, 1080):
                raise ValueError(f"Invalid input BMP: {path}")
    report = dict(mode=args.mode, status="running", corners_TL_TR_BR_BL=EXPECTED_CORNERS.tolist(),
                  corner_file=str(CORNER_TXT), phase_variant=args.phase_variant,
                  phase_file=str(phase_path), phase_sha256=phase_expected_sha,
                  amplitude_inputs={p.name: sha(p) for p in files}, normalization="none",
                  background_subtraction=False, fixed_detector_bounds_xyxy=BOUNDS,
                  rows=[])
    write(out / "report.json", report)
    amp = HoloeyeSLM(AMP_SDK, AMP_BIN, (1920, 1080), None, True, True, 5)
    with PhaseHDMI(PHASE_SDK, PHASE_LUT, settle_s=.8, pixel_format="rgba") as phase, amp, Camera(CAMERA_DLL) as camera:
        amp.preload_files(files)
        report["camera_settings_initial"] = camera.settings()
        report["phase_receipt"] = phase.show(phase_path)
        if args.mode == "phase":
            camera.settings(exposure=args.exposure_us, gain=1.0)
            sequence = [("B_first", PHASE_B), ("flat", FLAT_PHASE), ("B_repeat", PHASE_B)]
            images = {}
            for label, phase_path in sequence:
                receipt = phase.show(phase_path)
                _, roi, row = acquire(camera, amp, args.wait_ms, out, files[0], save_raw=label == "B_first")
                Image.fromarray(roi).save(out / f"{label}.png")
                images[label] = roi
                row.update(label=label, phase_receipt=receipt)
                report["rows"].append(row)
                write(out / "report.json", report)
            report["B_repeat_pcc"] = pcc(images["B_first"], images["B_repeat"])
            report["B_vs_flat_pcc"] = pcc(images["B_first"], images["flat"])
        elif args.mode == "exposure":
            for exposure in (3000, 5000, 7000, 10000, 15000, 20000):
                actual = camera.settings(exposure=exposure, gain=1.0)
                for path in files:
                    _, roi, row = acquire(camera, amp, args.wait_ms, out, path)
                    stem = f"e{exposure:05d}_{path.stem}"
                    Image.fromarray(roi).save(out / f"{stem}.png")
                    row.update(name=stem, label=int(path.stem[-1]), camera_settings=actual)
                    report["rows"].append(row)
                    write(out / "report.json", report)
                    print(json.dumps({k: row[k] for k in ("name", "mean", "p99", "saturated_fraction", "prediction")}), flush=True)
        elif args.mode == "timing":
            report["camera_settings"] = camera.settings(exposure=args.exposure_us, gain=1.0)
            images = {}
            for repeat in range(3):
                for wait in (0, 80, 120, 160, 200, 240, 300):
                    for path in files:
                        first, roi, row = acquire(camera, amp, wait, out, path)
                        key = f"r{repeat}_w{wait}_{path.stem}"
                        images[key] = roi
                        row.update(name=key, label=int(path.stem[-1]), first_prediction=stats(first)["prediction"])
                        report["rows"].append(row)
                        write(out / "report.json", report)
            for row in report["rows"]:
                prefix, _, stem = row["name"].partition("_w")
                sample = stem.split("_", 1)[1]
                ref = images[f"{prefix}_w300_{sample}"]
                row["last_vs_wait300_pcc"] = pcc(images[row["name"]], ref)
        else:
            report["camera_settings"] = camera.settings(exposure=args.exposure_us, gain=1.0)
            for index, path in enumerate(files):
                _, roi, row = acquire(camera, amp, args.wait_ms, out, path, save_raw=index == 0)
                Image.fromarray(roi).save(out / f"{path.stem}.png")
                row.update(name=path.stem, label=int(path.stem[-1]), amplitude_sha256=sha(path))
                report["rows"].append(row)
                write(out / "report.json", report)
                print(f"CAPTURED {index+1}/40 {path.stem} p99={row['p99']} sat={row['saturated_fraction']:.6f}", flush=True)
            _, repeat_roi, row = acquire(camera, amp, args.wait_ms, out, files[0])
            report["first_repeat_pcc"] = pcc(np.asarray(Image.open(out / f"{files[0].stem}.png")), repeat_roi)
            row["name"] = "repeat_first"
            report["rows"].append(row)
            Image.fromarray(repeat_roi).save(out / "repeat_first.png")
        amp.display_file(files[0])
    report["status"] = "complete"
    write(out / "report.json", report)
    print("RESULT", out, flush=True)


if __name__ == "__main__":
    main()
