"""Layerwise SHS capture for the matched low-rank OpenMoji ablation.

The optical model and devices stay alive across the entire capture. This is a
lab entry point, not an alternate training or simulation evaluation protocol.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from LightGenV2.tasks.t04_semantic_interaction import training as t
from LightGenV2.tasks.t04_semantic_interaction.lab_runtime import OpticalBoundary, StopAtPlane, STAGES, phase_planes
from LightGenV2.tasks.t04_semantic_interaction.settings import Settings
from experiments.qwen3_vl_2b_openmoji_instruction_four_stage_optical_editing.datasets import (
    OpenMojiEditingDataset, collate_samples, load_prompt_cache,
)
from experiments.qwen3_vl_2b_openmoji_instruction_four_stage_optical_editing.metrics import MetricAccumulator
from .profiles import install


GROUPS = {
    "g2": ("compact_lowrank16_r0_base.pt", "98fd922676b4faa7b5935e408f6aa31a0517b82f44b826c690ca4115518174c0"),
    "g3": ("compact_lowrank16_cleanmatch_r1_ccd.pt", "ae1525dc02306d9fd31bf9e0b675f279731b0543d228f3b2e3c48aecf011bb27"),
    "g4": ("compact_lowrank16_cleanmatch_r2_ccd_dc30.pt", "6234f977cacac2f78fbcba7ddebd9888d9c802bbba58249be9822dc21241c62f"),
    "g5": ("compact_lowrank16_cleanmatch_r3_ccd_dc30_grid.pt", "7a4b9ba5bbcf546e4af87ad3f03d99fb53e2ec8d7236b2c6fdbcfb65dc56154a"),
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    serialized = json.dumps(value, indent=2, ensure_ascii=False)
    for attempt in range(12):
        try:
            temporary.write_text(serialized, encoding="utf-8")
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 11:
                raise
            time.sleep(.25 * (attempt + 1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--group", choices=GROUPS, required=True)
    parser.add_argument("--scope", choices=("test", "train"), default="test")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--exposure-us", type=int, default=2000)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    project = args.project.resolve()
    output = args.output.resolve()
    original_lab = project.parent / "OpenMoji_Lab_SHS_8um"
    abo_lab = project.parent / "ABO_I2I_Lab_DVP_8um"
    sys.path.insert(0, str(abo_lab / "lab_dvp8um"))
    import four_image_flow as flow
    from shs_physical2400 import SHSBench, CORNERS

    # TRAIN contains some brighter source images than the pinned TEST set.
    # Keep the exact 2000 us/Gain X4 physical contract and record clipping,
    # but permit mild (<3%) saturated pixels for electronic-head adaptation.
    # TEST continues using the shared strict 1% SHSBench guard unchanged.
    if args.scope == "train":
        class TrainingSHSBench(SHSBench):
            def capture(self, stage, phase_path, amplitude_paths, ids, camera_orientation, save=True):
                digest = flow.sha(phase_path)
                if self.current_phase != digest:
                    self.receipt = self.phase.show(phase_path)
                    self.current_phase = digest
                    self.camera.fresh()
                receipt = self.receipt
                values = []
                folder = self.out / "ccd" / stage
                if save:
                    folder.mkdir(parents=True, exist_ok=True)
                for sample_id, path in zip(ids, amplitude_paths):
                    started = time.perf_counter()
                    self.controller.c["settle_delay_ms"] = self.wait * 1000
                    raw, meta = self.controller.capture(path)
                    if raw.shape != (1080, 1920):
                        raise RuntimeError(f"Unexpected SHS frame shape: {raw.shape}")
                    image = flow.orient(flow.warp(raw), camera_orientation)
                    row = {
                        "stage": stage, "sample_id": sample_id,
                        "phase_sha256": flow.sha(phase_path),
                        "amplitude_sha256": flow.sha(path),
                        "exposure": self.settings, "wait_ms": self.wait * 1000,
                        "frame_id": meta["frame_id"], "timestamp_ns": meta.get("timestamp_ns"),
                        "settle_drained_frames": meta["settle_drained_frames"],
                        "capture_total_ms": (time.perf_counter() - started) * 1000,
                        "mean": float(image.mean()), "p99": float(np.percentile(image, 99)),
                        "maximum": int(image.max()),
                        "saturation_fraction": float(np.mean(image == 255)),
                        "training_saturation_guard": 0.03,
                        "canonical_orientation": camera_orientation,
                        "no_photometric_normalization": True,
                    }
                    self.rows.append(row)
                    print(json.dumps(row), flush=True)
                    if row["saturation_fraction"] > 0.03:
                        raise RuntimeError("SHS TRAIN capture exceeds 3% saturation guard")
                    if save:
                        Image.fromarray(image).save(folder / (sample_id + ".png"))
                        flow.write(folder / (sample_id + ".json"), row)
                    values.append(image)
                return np.stack(values), receipt

        BenchClass = TrainingSHSBench
    else:
        BenchClass = SHSBench

    flow.BASE_CORNERS = CORNERS.copy()
    weight_name, expected_sha = GROUPS[args.group]
    checkpoint = project / "weights" / weight_name
    if sha(checkpoint) != expected_sha:
        raise ValueError("Weight SHA mismatch")
    cfg = Settings.__new__(Settings)
    cfg.__dict__.update(json.loads((project / "resolved_config.json").read_text(encoding="utf-8")))
    for key in ("config_path", "data_dir", "asset_dir", "output_dir", "qwen_checkpoint", "prompt_cache_path", "optical_base_config", "legacy_warmstart_checkpoint"):
        setattr(cfg, key, Path(getattr(cfg, key)))
    cfg.config_path = project / "source/LightGenV2/tasks/t04_semantic_interaction/configs/routerfill_shared.yaml"
    cfg.data_dir = original_lab / "data" if args.scope == "test" else project / "data_train_adapt1000"
    cfg.prompt_cache_path = cfg.data_dir / "token_embeddings_v1.pt"
    cfg.output_dir = output
    cfg.asset_dir = original_lab / "assets"
    cfg.svg_asset_dir = cfg.asset_dir / "openmoji-17.0.0-svg"
    cfg.qwen_checkpoint = original_lab / "frontend"
    cfg.optical_base_config = project / "source/experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval/configs/release/caltech101_four_layer_optical_joint.yaml"
    cfg.shared_readout_variant = "lowrank16"
    cfg.num_workers = 0
    device = torch.device(args.device)
    model = t.build_model(cfg, device)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("settings", {}).get("shared_readout_variant") != "lowrank16":
        raise ValueError("Checkpoint readout variant mismatch")
    model.load_state_dict(payload["model"], strict=True)
    install(model, "r0_base")  # same clean inference profile as the published five-row simulation
    model.eval().requires_grad_(False)
    for optic in model._optical_paths():
        optic.set_phase_dropout_active(False)
    manifest = cfg.test_manifest if args.scope == "test" else cfg.data_dir / "capture_train.jsonl"
    dataset = OpenMojiEditingDataset(manifest, cfg, load_prompt_cache(cfg.prompt_cache_path))
    if len(dataset) != 1000 or not 1 <= args.limit <= len(dataset):
        raise ValueError("Expected pinned 1000 records for this scope")
    contract = {
        "group": args.group, "checkpoint_sha256": expected_sha,
        "count": args.limit, "dataset_manifest_sha256": sha(manifest), "scope": args.scope,
        "phase_orientation": "hv", "phase_inverse": True, "camera_orientation": "flip_v",
        "amplitude_encoding": "round(255*bounded_amplitude), no peak rescale",
        "exposure_us": args.exposure_us, "gain": "Gain_X4", "wait_ms": 240,
        "corners": CORNERS.tolist(), "inference_profile": "r0_base_clean", "model_device": args.device,
    }
    if (output / "contract.json").exists():
        previous = json.loads((output / "contract.json").read_text())
        legacy_test = args.scope == "test" and "scope" not in previous and previous == {
            key: value for key, value in contract.items() if key != "scope"
        }
        if previous != contract and not legacy_test:
            raise ValueError("Resume contract mismatch")
    else:
        write(output / "contract.json", contract)

    def batch(index: int) -> dict:
        return {key: value.to(device) if torch.is_tensor(value) else value
                for key, value in collate_samples([dataset[index]]).items()}

    def sample_id(index: int) -> str:
        return f"test_{index:05d}" if args.scope == "test" else dataset.records[index]["sample_id"]

    def prior_frames(index: int, prior: tuple[str, ...]) -> dict:
        return {
            stage: torch.from_numpy(np.asarray(Image.open(output / "ccd" / stage / (sample_id(index) + ".png")), dtype=np.float32).copy()[None] / 255.0)
            for stage in prior
        }

    if args.selftest:
        x = batch(0)
        with torch.inference_mode():
            baseline = model(x["source_image"], x["prompt_hidden"])
            with OpticalBoundary(model) as tap:
                bridged = model(x["source_image"], x["prompt_hidden"])
            with OpticalBoundary(model, tap.detectors):
                replayed = model(x["source_image"], x["prompt_hidden"])
        keys = ["category_logits", "edit_logits", "task_logits"]
        error = max(float((baseline[k] - bridged[k]).abs().max()) for k in keys)
        replay_error = max(float((baseline[k] - replayed[k]).abs().max()) for k in keys)
        max_amplitude = max(float(a.max()) for a in tap.amplitudes.values())
        if error > 1e-4 or replay_error > 1e-4 or max_amplitude > 1.00001:
            raise ValueError(f"Optical boundary mismatch: {error=}, {replay_error=}, {max_amplitude=}")
        write(output / "selftest.json", {"status": "pass", "error": error, "replay_error": replay_error,
                                         "max_amplitude": max_amplitude, "contract": contract})
        return

    phases = phase_planes(model)
    phase_dir = output / "phase"
    phase_dir.mkdir(parents=True, exist_ok=True)
    for stage, phase in phases.items():
        bmp = phase_dir / (stage + ".bmp")
        image = Image.fromarray(flow.phase_gray(phase, "hv", True))
        if bmp.exists():
            if not np.array_equal(np.asarray(Image.open(bmp)), np.asarray(image)):
                raise ValueError("Phase image changed on resume")
        else:
            image.save(bmp)
    started = time.time()
    with BenchClass(output, args.exposure_us, 240, {}) as bench:
        for stage_index, stage in enumerate(STAGES):
            folder = output / "ccd" / stage
            folder.mkdir(parents=True, exist_ok=True)
            amplitude_dir = output / "amplitude" / stage
            amplitude_dir.mkdir(parents=True, exist_ok=True)
            for start in range(0, args.limit, 4):
                paths, ids = [], []
                for index in range(start, min(start + 4, args.limit)):
                    sid = sample_id(index)
                    if args.resume and (folder / (sid + ".png")).exists() and (folder / (sid + ".json")).exists():
                        continue
                    x = batch(index)
                    with torch.inference_mode(), OpticalBoundary(model, prior_frames(index, STAGES[:stage_index]), stage) as tap:
                        try:
                            model(x["source_image"], x["prompt_hidden"])
                        except StopAtPlane:
                            pass
                    amplitude = tap.amplitudes[stage][0].cpu().numpy()
                    if not np.isfinite(amplitude).all() or amplitude.min() < -1e-6 or amplitude.max() > 1.00001:
                        raise ValueError("Bounded amplitude contract violated")
                    gray = np.rint(np.clip(amplitude, 0, 1) * 255).astype(np.uint8)
                    path = amplitude_dir / (sid + ".bmp")
                    Image.fromarray(flow.active_to_native(gray)).save(path)
                    write(amplitude_dir / (sid + ".json"), {"input_bmp_sha256": sha(path),
                          "amplitude_max": float(amplitude.max()), "amplitude_mean": float(amplitude.mean()),
                          "phase_sha256": sha(phase_dir / (stage + ".bmp"))})
                    paths.append(path)
                    ids.append(sid)
                if paths:
                    values, _ = bench.capture(stage, phase_dir / (stage + ".bmp"), paths, ids, "flip_v")
                    checks = [float(np.percentile(v, 99)) for v in values]
                    if min(checks) < 15:
                        raise RuntimeError(f"Dark CCD batch {stage} {start}: {checks}")
                    for path in paths:
                        path.unlink()  # transient raster; its SHA and raw CCD receipt remain
                write(output / "progress.json", {"status": "capturing", "group": args.group,
                      "stage": stage, "stage_completed": min(start + 4, args.limit), "total": args.limit,
                      "completed_stages": list(STAGES[:stage_index]), "elapsed_seconds": time.time() - started})
    simulation, physical = MetricAccumulator(), MetricAccumulator()
    rows = []
    with torch.inference_mode():
        for index in range(args.limit):
            x = batch(index)
            sim = model(x["source_image"], x["prompt_hidden"])
            with OpticalBoundary(model, prior_frames(index, STAGES)):
                actual = model(x["source_image"], x["prompt_hidden"])
            simulation.update(sim, x)
            physical.update(actual, x)
            rows.append({"index": index, "sample_id": dataset.records[index]["sample_id"]})
    write(output / "report.json", {"status": "complete", "contract": contract,
          "simulation_metrics": simulation.compute(), "physical_metrics": physical.compute(),
          "samples": rows, "ccd_count": args.limit * len(STAGES), "elapsed_seconds": time.time() - started})
    write(output / "progress.json", {"status": "complete", "total": args.limit,
          "ccd_count": args.limit * len(STAGES), "completed_stages": list(STAGES)})


if __name__ == "__main__":
    main()
