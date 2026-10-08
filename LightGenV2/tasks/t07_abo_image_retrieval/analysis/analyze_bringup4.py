"""Audit raw versus globally calibrated CCD agreement for the 2026-09-23 four-query run."""
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1] / "reports/20260923_dvp8um_bringup4/review"
STAGES = (
    "vision_router", "vision_expert", "vision_global",
    "language_router", "language_expert", "language_global",
)


def pcc(a, b):
    a = np.asarray(a, np.float64).ravel(); b = np.asarray(b, np.float64).ravel()
    a -= a.mean(); b -= b.mean()
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a.dot(b) / denominator) if denominator else 0.0


def orient(a, name):
    variants = {
        "identity": a, "flip_h": np.fliplr(a), "flip_v": np.flipud(a),
        "rot180": np.rot90(a, 2), "transpose": a.T, "rot90": np.rot90(a, 1),
        "rot270": np.rot90(a, 3), "anti_transpose": np.rot90(a.T, 2),
    }
    return variants[name].copy()


def main():
    report = json.loads((ROOT / "four_image_report.json").read_text(encoding="utf-8"))
    reference = np.load(ROOT / "simulation_reference.npz")
    orientation = report["camera_canonical_orientation"]
    dark = orient(np.asarray(Image.open(ROOT / "previews/e04000_g000.png"), np.float32), orientation)
    white = orient(np.asarray(Image.open(ROOT / "previews/e04000_g255.png"), np.float32), orientation)
    response = np.maximum(white - dark, 5.0)
    ids = [row["sample_id"] for row in report["samples"]]
    result = {"schema": 1, "orientation": orientation, "methods": {}}
    for method in ("raw", "ambient_subtracted", "flat_field"):
        result["methods"][method] = {}
        for stage in STAGES:
            values = []
            for sample_id, target in zip(ids, reference[stage]):
                image = np.asarray(Image.open(ROOT / "ccd" / stage / f"{sample_id}.png"), np.float32)
                if method == "ambient_subtracted":
                    image = np.maximum(image - dark, 0.0)
                elif method == "flat_field":
                    image = np.maximum(image - dark, 0.0) / response
                values.append(pcc(image, target))
            result["methods"][method][stage] = {
                "per_sample_pcc": values, "mean_pcc": float(np.mean(values))
            }
    out = ROOT.parent / "calibration_audit.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
