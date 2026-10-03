"""Stage one group's model/masks for deployment; not a capture-ready SDK kit."""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from study import GROUPS, make_config, sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=GROUPS, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device-pitch-um", type=float, default=8.0)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output must not already exist")
    sys.path.insert(0, str(ROOT / "runtime"))
    import torch
    import yaml
    from settings_adapter import load_settings
    from LightGenV2.tasks.t06_video_quality_assessment.models.multivideo9x4 import build_model
    from LightGenV2.tasks.t06_video_quality_assessment.lab_runtime import phase_planes
    from hardware import STAGES, rasterize_phase
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if saved.get("study_group") != args.group:
        parser.error("Checkpoint/group identity mismatch")
    args.output.mkdir(parents=True)
    raw = make_config(args.group, purpose="deployment", device_pitch=args.device_pitch_um,
                      output=args.output / "runs")
    config = args.output / "deployment.yaml"
    config.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    settings = load_settings(config)
    model = build_model(settings).eval()
    model.load_state_dict(saved["state_dict"], strict=True)
    planes, supports = phase_planes(model, "temporal")
    import numpy as np
    (args.output / "phases").mkdir()
    for stage in STAGES:
        np.save(args.output / "phases" / f"{stage}_logical.npy", planes[stage])
        mapped = rasterize_phase(torch.from_numpy(planes[stage]), device_pitch_um=args.device_pitch_um)
        np.save(args.output / "phases" / f"{stage}_device.npy", mapped.numpy())
    shutil.copy2(args.checkpoint, args.output / "checkpoint.pt")
    write_json(args.output / "deployment_manifest.json", {
        "group": args.group, "checkpoint_sha256": sha256(args.checkpoint),
        "stages": STAGES, "model_pitch_um": settings.pixel_pitch_um,
        "device_pitch_um": args.device_pitch_um, "active_device_pixels": settings.propagation_active_size,
        "eta": settings.unmodulated_power_fraction_eval, "automatic_phase_switching": False,
        "bounded_amplitude": saved["study_protocol"]["bounded_amplitude"],
        "status": "staged_not_capture_ready",
        "pending": ["verified panel LUT/orientation and amplitude encoder", "raw CCD calibration/ROI contract",
                    "field input identities", "per-layer capture adapter and amplitude forward equivalence"],
        "files": {str(p.relative_to(args.output)): sha256(p) for p in args.output.rglob("*") if p.is_file()},
    })
    print(args.output / "deployment_manifest.json")


if __name__ == "__main__":
    from amplitude import bounded_graph
    with bounded_graph():
        main()
