"""Audited 1000-row six-plane OpenMoji e45 DVP capture, stage by stage."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from PIL import Image
import torch

from pilot import CCD_CORNERS, CHECKPOINT_SHA, STAGES, amplitude_bmp, sha, write


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument("--pilot", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--exposure-us", type=float, default=20000)
    parser.add_argument("--wait-ms", type=float, default=240)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root, pilot, out = args.project.resolve(), args.pilot.resolve(), args.output.resolve()
    if not args.resume and out.exists():
        raise FileExistsError(out)
    if not 100 <= args.exposure_us <= 20000 or not 150 <= args.wait_ms <= 500:
        raise ValueError("Outside bounded camera/SLM settings")
    if not 1 <= args.batch_size <= 8:
        raise ValueError("Batch size must be 1..8")
    sys.path.insert(0, str(root / "runtime_exact"))
    abo = root.parent / "ABO_I2I_Lab_DVP_8um"
    sys.path.insert(0, str(abo / "lab_dvp8um"))
    import four_image_flow as hw
    from LightGenV2.tasks.t04_semantic_interaction.settings import load_settings
    from LightGenV2.tasks.t04_semantic_interaction.modeling import build_model
    from LightGenV2.tasks.t04_semantic_interaction.lab_runtime import OpticalBoundary, StopAtPlane
    from experiments.qwen3_vl_2b_openmoji_instruction_four_stage_optical_editing.datasets import (
        OpenMojiEditingDataset, collate_samples, load_prompt_cache,
    )
    from experiments.qwen3_vl_2b_openmoji_instruction_four_stage_optical_editing.metrics import MetricAccumulator

    hw.BASE_CORNERS = CCD_CORNERS
    pilot_report = json.loads((pilot / "report.json").read_text(encoding="utf-8"))
    if pilot_report["status"] != "complete" or pilot_report["checkpoint_sha256"] != CHECKPOINT_SHA:
        raise RuntimeError("Incomplete/wrong four-row pilot")
    if set(pilot_report["stages"]) != set(STAGES):
        raise RuntimeError("Pilot did not calibrate all six planes")
    cfg = load_settings(root / "runtime_exact/LightGenV2/tasks/t04_semantic_interaction/configs/layered_scene_exp05_dc30_ccdsmall.yaml")
    cfg.data_dir = root / "data"
    cfg.prompt_cache_path = cfg.data_dir / "token_embeddings_v1.pt"
    cfg.output_dir = out
    legacy = root.parent / "OpenMoji_Lab_SHS_8um"
    cfg.qwen_checkpoint = legacy / "frontend"
    cfg.asset_dir = legacy / "assets"
    ckpt = root / "weights/selected_checkpoint.pt"
    if sha(ckpt) != CHECKPOINT_SHA or cfg.layout_version != "layered_anchor6_svg_v3":
        raise RuntimeError("Wrong checkpoint/data layout")
    manifest_sha = sha(cfg.test_manifest)
    cache_sha = sha(cfg.prompt_cache_path)
    contract = {"checkpoint_sha256": CHECKPOINT_SHA, "pilot_sha256": sha(pilot / "report.json"),
                "test_manifest_sha256": manifest_sha, "prompt_cache_sha256": cache_sha,
                "exposure_us": args.exposure_us, "wait_ms": args.wait_ms,
                "corners_TL_TR_BR_BL": CCD_CORNERS.tolist(), "batch_size": args.batch_size,
                "query_count": 1000, "stages": list(STAGES)}
    if args.resume:
        if not out.exists() or json.loads((out / "run_contract.json").read_text(encoding="utf-8")) != contract:
            raise RuntimeError("Resume contract mismatch")
    else:
        out.mkdir(parents=True)
        (out / "phase").mkdir()
        write(out / "run_contract.json", contract)
        for stage in STAGES:
            source = pilot / "phase" / f"{stage}.bmp"
            if sha(source) != pilot_report["stages"][stage]["phase_sha256"]:
                raise RuntimeError("Pilot phase changed")
            shutil.copy2(source, out / "phase" / source.name)
    device = torch.device("cuda")
    model = build_model(cfg, device)
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    if payload["epoch"] != 45 or payload["architecture"] != model.checkpoint_architecture:
        raise RuntimeError("Checkpoint architecture/epoch mismatch")
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    for path in model._optical_paths():
        path.set_phase_dropout_active(False)
    data = OpenMojiEditingDataset(cfg.test_manifest, cfg, load_prompt_cache(cfg.prompt_cache_path))
    if len(data) != 1000:
        raise RuntimeError("Expected 1000 test samples")
    health = {}
    with hw.Bench(out, args.exposure_us, args.wait_ms, {}) as bench:
        for stage_index, stage in enumerate(STAGES):
            phase = out / "phase" / f"{stage}.bmp"
            orientation = pilot_report["stages"][stage]["calibration"]["camera_orientation"]
            reference_id = pilot_report["samples"][0]
            reference_amp = pilot / "amplitude" / stage / f"{reference_id}.bmp"
            reference_img = np.asarray(Image.open(pilot / "ccd" / stage / f"{reference_id}.png"))
            expected_p99 = float(np.percentile(reference_img, 99))
            def check_reference(position: int) -> None:
                measured, _ = bench.capture("reference_"+stage, phase,
                                             [reference_amp], [reference_id], orientation, save=False)
                frame = measured[0]
                p99 = float(np.percentile(frame, 99))
                pcc = hw.pcc(frame, reference_img)
                row = {"stage": stage, "after_query_count": position, "p99": p99,
                       "pilot_p99": expected_p99, "pcc_to_pilot": pcc}
                health.setdefault(stage, []).append(row)
                write(out / "health_checks.json", health)
                if p99 < max(8.0, expected_p99 * 0.5) or pcc < 0.75:
                    raise RuntimeError(f"Optical reference changed at {stage}/{position}: {row}")
            check_reference(0)
            stage_ccd = out / "ccd" / stage
            stage_amp = out / "amplitude" / stage
            stage_amp.mkdir(parents=True, exist_ok=True)
            for start in range(0, len(data), args.batch_size):
                paths, ids = [], []
                for index in range(start, min(start + args.batch_size, len(data))):
                    record = data.records[index]
                    sample_id = record["sample_id"]
                    captured = stage_ccd / f"{sample_id}.png"
                    receipt = stage_ccd / f"{sample_id}.json"
                    if args.resume and captured.exists() and receipt.exists():
                        saved = json.loads(receipt.read_text(encoding="utf-8"))
                        if saved["sample_id"] == sample_id and saved["phase_sha256"] == sha(phase):
                            continue
                    batch = collate_samples([data[index]])
                    batch = {key: value.to(device) if torch.is_tensor(value) else value
                             for key, value in batch.items()}
                    previous = {}
                    for prev in STAGES[:stage_index]:
                        file = out / "ccd" / prev / f"{sample_id}.png"
                        if not file.is_file():
                            raise FileNotFoundError(file)
                        previous[prev] = torch.from_numpy(np.asarray(Image.open(file), np.float32).copy()[None] / 255.0)
                    with torch.inference_mode(), OpticalBoundary(model, previous, stage) as tap:
                        try:
                            model(batch["source_image"], batch["prompt_hidden"])
                        except StopAtPlane:
                            pass
                    amp = stage_amp / f"{sample_id}.bmp"
                    Image.fromarray(amplitude_bmp(tap.amplitudes[stage][0].cpu().numpy())).save(amp)
                    paths.append(amp)
                    ids.append(sample_id)
                if paths:
                    bench.capture(stage, phase, paths, ids, orientation, save=True)
                completed = start + min(args.batch_size, len(data) - start)
                if completed % 50 == 0 or completed == len(data):
                    check_reference(completed)
                write(out / "progress.json", {"status": "running", "stage": stage,
                                              "completed_in_stage": completed,
                                              "total_per_stage": len(data),
                                              "completed_stages": list(STAGES[:stage_index])})
    acc = MetricAccumulator()
    with torch.inference_mode():
        for index in range(len(data)):
            batch = collate_samples([data[index]])
            batch = {key: value.to(device) if torch.is_tensor(value) else value
                     for key, value in batch.items()}
            frames = {}
            for stage in STAGES:
                file = out / "ccd" / stage / f"{data.records[index]['sample_id']}.png"
                frames[stage] = torch.from_numpy(np.asarray(Image.open(file), np.float32).copy()[None] / 255.0)
            with OpticalBoundary(model, frames):
                output = model(batch["source_image"], batch["prompt_hidden"])
            acc.update(output, batch)
            if (index + 1) % 100 == 0:
                print(f"EVALUATED {index + 1}/1000", flush=True)
    result = {"status": "complete", "checkpoint_sha256": CHECKPOINT_SHA,
              "test_count": len(data), "ccd_images": len(data) * len(STAGES),
              "metrics": acc.compute(), "health_checks": health,
              "simulated_reference_changed_cell_accuracy": 0.8765,
              "notes": "1000 selected test rows; no downstream retraining; no per-frame intensity normalization"}
    write(out / "report.json", result)
    write(out / "progress.json", {"status": "complete", "completed_stages": list(STAGES),
                                  "total_per_stage": len(data)})
    print(json.dumps({"status": "complete", "metrics": result["metrics"]["overall"]}), flush=True)


if __name__ == "__main__":
    main()
