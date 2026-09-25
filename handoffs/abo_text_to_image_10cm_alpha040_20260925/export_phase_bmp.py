"""Export provisional 8-um phase BMPs from the pinned T08 checkpoint.

The gray/orientation choices come from the 2026-09-24 T07 calibration and
must be rechecked for this model before measuring retrieval accuracy.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
import torch


ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / "best_checkpoint.pt"
EXPECTED = "cc977b83286a8e90398ebd30064428886bc3c06f1eb0557e470060f7ff5c1cae"
STAGES = ("vision_router", "vision_expert", "vision_global",
          "language_router", "language_expert", "language_global")
HISTORICAL_CANDIDATES = {
    "vision_router": "hv_inverse",
    "vision_expert": "h_inverse",
    "vision_global": "h_inverse",
    "language_router": "h_inverse",
    "language_expert": "hv_inverse",
    "language_global": "hv_normal",
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def phase_map(weights: dict[str, torch.Tensor], stage: str) -> np.ndarray:
    prefix = "core.optical_branch.core."
    out = np.zeros((478, 478), dtype=np.float32)
    if stage.endswith("router"):
        raw = weights[prefix + "router.raw_router_phase"]
        out[127:351, 127:351] = (2 * math.pi * torch.sigmoid(raw)).numpy()
    elif stage.endswith("expert"):
        for i, (y, x) in enumerate(((0, 0), (0, 254), (254, 0), (254, 254))):
            raw = weights[prefix + f"expert_layers.0.experts.{i}.raw_phase"]
            out[y:y + 224, x:x + 224] = (2 * math.pi * torch.sigmoid(raw)).numpy()
    else:
        raw = weights[prefix + "global_phase.phase.raw_phase"]
        out[:] = (2 * math.pi * torch.sigmoid(raw)).numpy()
    return out


def pitch_nearest(gray: np.ndarray) -> np.ndarray:
    native_size = round(478 * 17 / 8)
    positions = (np.arange(native_size) + .5 - native_size / 2) * 8
    indices = np.floor(positions / 17 + 478 / 2).astype(int).clip(0, 477)
    return gray[np.ix_(indices, indices)]


def encode(radians: np.ndarray, candidate: str) -> np.ndarray:
    spatial, polarity = candidate.split("_")
    gray = np.floor(np.mod(radians, 2 * math.pi) / (2 * math.pi) * 256).clip(0, 255).astype(np.uint8)
    if "h" in spatial:
        gray = np.fliplr(gray)
    if "v" in spatial:
        gray = np.flipud(gray)
    active = pitch_nearest(gray)
    canvas = np.zeros((1200, 1920), dtype=np.uint8)
    top = (1200 - active.shape[0]) // 2
    left = (1920 - active.shape[1]) // 2
    canvas[top:top + active.shape[0], left:left + active.shape[1]] = active
    return 255 - canvas if polarity == "inverse" else canvas


def main() -> None:
    if digest(CHECKPOINT) != EXPECTED:
        raise RuntimeError("Wrong checkpoint: SHA256 mismatch")
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    output = ROOT / "phase_bmp_provisional"
    output.mkdir(exist_ok=True)
    rows = []
    for stage in STAGES:
        modality = stage.split("_")[0]
        radians = phase_map(payload[modality + "_optical"], stage)
        candidate = HISTORICAL_CANDIDATES[stage]
        path = output / (stage + ".bmp")
        Image.fromarray(encode(radians, candidate), mode="L").save(path)
        rows.append({"stage": stage, "candidate": candidate, "bmp": path.name,
                     "sha256": digest(path), "range_radians": [float(radians.min()), float(radians.max())]})
        if stage == "vision_router":
            candidate_dir = output / "vision_router_candidates"
            candidate_dir.mkdir(exist_ok=True)
            for spatial in ("none", "h", "v", "hv"):
                for polarity in ("normal", "inverse"):
                    name = f"{spatial}_{polarity}"
                    Image.fromarray(encode(radians, name)).save(candidate_dir / f"{name}.bmp")
    (output / "manifest.json").write_text(json.dumps({
        "status": "provisional_not_hardware_validated", "checkpoint_sha256": EXPECTED,
        "physical_pitch_um": 8, "logical_pitch_um": 17,
        "roi_center_xy": [960, 600], "stages": rows,
    }, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
