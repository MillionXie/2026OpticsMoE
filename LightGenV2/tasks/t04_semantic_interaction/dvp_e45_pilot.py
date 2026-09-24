"""Four-example, six-plane DVP pilot for the selected layered OpenMoji e45.

This is a physical diagnostic, not the 1000-row final evaluation. It preserves
all CCD frames separately from the earlier SHS release and ABO measurements.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch


STAGES = ("language_router", "language_expert", "language_global",
          "vision_router", "vision_expert", "vision_global")
CHECKPOINT_SHA = "03cb861c3ac344556601eb3eb6d7d1a22b77a54d2e7e68e85d77ee30fb09eb21"
CCD_CORNERS = np.float32([[995, 172], [4310, 172], [4303, 3440], [980, 3435]])


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def amplitude_bmp(active: np.ndarray) -> np.ndarray:
    import cv2
    image = np.asarray(active, np.float32)
    if image.shape != (478, 478) or not np.isfinite(image).all() or image.min() < 0:
        raise ValueError("Invalid 478x478 amplitude")
    positive = image[image > 0]
    scale = float(np.percentile(positive, 99.5)) if len(positive) else 0.0
    if scale <= 0:
        raise ValueError("Empty amplitude")
    gray = np.rint(np.clip(image / scale, 0, 1) * 255).astype(np.uint8)
    resized = cv2.resize(gray, (1016, 1016), interpolation=cv2.INTER_LINEAR)
    native = np.zeros((1080, 1920), np.uint8)
    native[32:1048, 452:1468] = resized
    return native


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--exposure-us", type=float, default=20000)
    parser.add_argument("--wait-ms", type=float, default=240)
    args = parser.parse_args()
    root, out = args.project.resolve(), args.output.resolve()
    if out.exists():
        raise FileExistsError(out)
    if not 100 <= args.exposure_us <= 20000 or not 150 <= args.wait_ms <= 500:
        raise ValueError("Outside bounded camera/SLM pilot settings")
    sys.path.insert(0, str(root / "runtime_exact"))
    abo = root.parent / "ABO_I2I_Lab_DVP_8um"
    sys.path.insert(0, str(abo / "lab_dvp8um"))
    import four_image_flow as hw
    from LightGenV2.tasks.t04_semantic_interaction.settings import load_settings
    from LightGenV2.tasks.t04_semantic_interaction.modeling import build_model
    from LightGenV2.tasks.t04_semantic_interaction.lab_runtime import OpticalBoundary
    from experiments.qwen3_vl_2b_openmoji_instruction_four_stage_optical_editing.datasets import (
        OpenMojiEditingDataset, collate_samples, load_prompt_cache,
    )
    from experiments.qwen3_vl_2b_openmoji_instruction_four_stage_optical_editing.metrics import MetricAccumulator

    hw.BASE_CORNERS = CCD_CORNERS
    cfg = load_settings(root / "runtime_exact/LightGenV2/tasks/t04_semantic_interaction/configs/layered_scene_exp05_dc30_ccdsmall.yaml")
    cfg.data_dir = root / "data"
    cfg.prompt_cache_path = cfg.data_dir / "token_embeddings_v1.pt"
    cfg.output_dir = out
    legacy = root.parent / "OpenMoji_Lab_SHS_8um"
    cfg.qwen_checkpoint = legacy / "frontend"
    cfg.asset_dir = legacy / "assets"
    if cfg.layout_version != "layered_anchor6_svg_v3":
        raise RuntimeError("Wrong OpenMoji layout")
    checkpoint = root / "weights/selected_checkpoint.pt"
    if sha(checkpoint) != CHECKPOINT_SHA:
        raise RuntimeError("Wrong selected checkpoint")
    out.mkdir(parents=True)
    (out / "phase_candidates").mkdir()
    (out / "phase").mkdir()
    (out / "amplitude").mkdir()
    device = torch.device("cuda")
    model = build_model(cfg, device)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload["architecture"] != model.checkpoint_architecture or payload["epoch"] != 45:
        raise RuntimeError("Checkpoint architecture/epoch mismatch")
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    for path in model._optical_paths():
        path.set_phase_dropout_active(False)
    data = OpenMojiEditingDataset(cfg.test_manifest, cfg, load_prompt_cache(cfg.prompt_cache_path))
    if len(data) != 1000:
        raise RuntimeError("Expected 1000 layered test samples")
    # First four rows cover add/replace/move/remove under the fixed split.
    indices = range(4)
    ids = [data.records[i]["sample_id"] for i in indices]
    batches = []
    for i in indices:
        batch = collate_samples([data[i]])
        batches.append({key: value.to(device) if torch.is_tensor(value) else value
                        for key, value in batch.items()})
    measured: list[dict[str, torch.Tensor]] = [dict() for _ in indices]
    stage_reports = {}
    with hw.Bench(out, args.exposure_us, args.wait_ms, {}) as bench:
        # Check actual optical throughput; SDK acknowledgements alone are not enough.
        health = abo / "runs/abo_i2i_20260924/05_brightness_health/patterns"
        flat = health / "phase_flat_pi_inverted.bmp"
        white = health / "amplitude_255.bmp"
        bench.phase.show(flat)
        bench.amp.preload_files([white])
        bench.amp.display_file(white)
        import time
        time.sleep(args.wait_ms / 1000)
        full = [bench.camera.capture()[0] for _ in range(6)][-1]
        white_roi = hw.warp(full)
        white_p99 = float(np.percentile(white_roi, 99))
        if white_p99 < 20:
            raise RuntimeError(f"Full-white optical health gate failed: p99={white_p99}")
        Image.fromarray(white_roi).save(out / "health_white_roi.png")
        for stage in STAGES:
            amplitudes, targets = [], []
            for batch, previous in zip(batches, measured):
                with torch.inference_mode(), OpticalBoundary(model, previous, stage) as tap:
                    try:
                        model(batch["source_image"], batch["prompt_hidden"])
                    except Exception as exc:
                        # Only the explicit stage stop is expected.
                        from LightGenV2.tasks.t04_semantic_interaction.lab_runtime import StopAtPlane
                        if not isinstance(exc, StopAtPlane):
                            raise
                amplitudes.append(tap.amplitudes[stage][0].cpu().numpy())
                targets.append(tap.detectors[stage][0].cpu())
            amp_folder = out / "amplitude" / stage
            amp_folder.mkdir()
            paths = []
            for sample_id, active in zip(ids, amplitudes):
                path = amp_folder / f"{sample_id}.bmp"
                Image.fromarray(amplitude_bmp(active)).save(path)
                paths.append(path)
            phase = tap.planes[stage]
            candidates = {}
            for spatial in ("none", "h", "v", "hv"):
                for tone in ("normal", "inverse"):
                    name = f"{spatial}_{tone}"
                    file = out / "phase_candidates" / f"{stage}_{name}.bmp"
                    Image.fromarray(hw.phase_gray(phase, spatial, tone == "inverse")).save(file)
                    candidates[name] = file
            selected, receipt, best, phase_file = hw.calibrate_stage(
                bench, out, stage, candidates, paths, ids, targets, minimum_pcc=None,
            )
            for index, frame in enumerate(selected):
                measured[index][stage] = torch.from_numpy(frame.astype(np.float32)[None] / 255.0)
            stage_reports[stage] = {"calibration": best, "phase_sha256": sha(phase_file),
                                    "p99_by_sample": [float(np.percentile(frame, 99)) for frame in selected],
                                    "saturated_fraction_max": float(np.max(np.mean(selected == 255, axis=(1, 2))))}
            write(out / "progress.json", {"status": "running", "completed_stages": list(stage_reports),
                                          "white_health_p99": white_p99})
    acc = MetricAccumulator()
    with torch.inference_mode():
        for batch, frames in zip(batches, measured):
            with OpticalBoundary(model, frames):
                output = model(batch["source_image"], batch["prompt_hidden"])
            acc.update(output, batch)
    report = {"status": "complete", "scope": "four-sample pilot, not independent test accuracy",
              "checkpoint_sha256": CHECKPOINT_SHA, "samples": ids,
              "exposure_us": args.exposure_us, "wait_ms": args.wait_ms,
              "white_health_p99": white_p99, "stages": stage_reports,
              "pilot_metrics": acc.compute()}
    write(out / "report.json", report)
    write(out / "progress.json", {"status": "complete", "completed_stages": list(STAGES)})
    print(json.dumps({"status": "complete", "white_health_p99": white_p99,
                      "pilot_metrics": report["pilot_metrics"]["overall"]}), flush=True)


if __name__ == "__main__":
    main()
