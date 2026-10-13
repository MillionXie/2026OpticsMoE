"""A100-only recovery of missing LGVQ temporal heads using the original protocol.

No backbone training, no spatial head, no new split/architecture/seed search.
The historical source functions do all feature extraction and head fitting.
New outputs are explicitly marked as retrained, never as recovered old PTs.
"""
from __future__ import annotations

import argparse
import csv
import importlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

from . import evaluate_temporal_baseline_batches as frozen

TASK = Path(__file__).resolve().parent
REPO = TASK.parents[2]
OWNER = TASK / "runs" / "simulation"
ARCHIVE_REF = "refs/archive/local-inactive-20261004/codex/deepseek-lgvq-baseline"
CONFIG_NAMES = {
    "clip": "clip_vit_b32_lgvq_4f_r224.yaml",
    "yolo": "yolo11s_lgvq_4f_r640.yaml",
    "deepseek": "deepseek_vl2_tiny_lgvq_4f_r384.yaml",
}
MODEL_PATHS = {
    "clip": Path("/DATA/DATA1/guest3/models/openai-clip/ViT-B-32.pt"),
    "yolo": Path("/DATA/DATA1/guest3/models/yolo11s.pt"),
    "deepseek": Path("/DATA/DATA1/guest3/models/deepseek-vl2-tiny"),
}
ENV_PATHS = {
    "clip": "/DATA/DATA1/guest3/.venvs/lgvq-visual-baselines/bin/python",
    "yolo": "/DATA/DATA1/guest3/.venvs/lgvq-visual-baselines/bin/python",
    "deepseek": "/DATA/DATA1/guest3/.venvs/deepseek-vl2/bin/python",
}
CONTRACT = {
    "target": "temporal", "epochs": 50, "batch_size": 512, "seed": 42,
    "learning_rate": 0.001, "weight_decay": 0.0,
    "loss": "cross_entropy_only", "scheduler": "cosine",
    "train_samples": 2250, "test_samples": 558,
    "selection": "highest_TEST_SRCC_development_no_VAL",
}
OLD_SRCC = {"clip": 0.7251568, "yolo": 0.7015822, "deepseek": 0.5688169}


def require_new_output(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing existing reproduction output: {path}")
    if not path.resolve().is_relative_to(OWNER.resolve()) or path.resolve() == OWNER.resolve():
        raise ValueError("Output must be a new child of canonical T06 runs/simulation")


def validate_original_protocol(raw: dict, baseline: str) -> None:
    for key in ("epochs", "batch_size", "seed", "learning_rate", "weight_decay", "loss"):
        if raw["training"].get(key) != CONTRACT[key]:
            raise ValueError(f"Original training protocol changed: {key}")
    if (raw["task"]["train_samples"], raw["task"]["test_samples"]) != (2250, 558):
        raise ValueError("Original split changed")
    if raw["input"]["center_crop_short_side_fraction"] != 0.65:
        raise ValueError("Original crop changed")
    if raw["input"]["image_size"] != {"clip": 224, "yolo": 640, "deepseek": 384}[baseline]:
        raise ValueError("Original input resolution changed")


def validate_assets(manifest: Path, model: Path, baseline: str) -> Path:
    weight = frozen.backbone_file(baseline, model)
    for path, expected in ((manifest, frozen.MANIFEST_SHA), (weight, frozen.BACKBONE_SHA[baseline])):
        if not path.is_file() or frozen.sha256(path) != expected:
            raise ValueError(f"Original asset SHA mismatch: {path}")
    if baseline == "deepseek" and frozen.sha256(model / "config.json") != "377e10dceb803d9a843d9825d1aa7cba8121c2b825888c632ecf807d0834ce97":
        raise ValueError("Original DeepSeek model config SHA mismatch")
    return weight


def cuda_guard() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != frozen.A100_UUID:
        raise RuntimeError("Reproduction must bind only the original physical A100 UUID")
    frozen.process_audit()


def run_one(args) -> None:
    # Output rejection precedes imports/RNG/device initialization and asset reads.
    require_new_output(args.output)
    cuda_guard()
    weight = validate_assets(args.manifest, args.model, args.baseline)
    import torch
    from . import quality_token_common as core
    from LightGenV2.common.baseline_measurement import validate_cuda_device, gpu_power_limit_w

    module_name = "deepseek_vl2_quality" if args.baseline == "deepseek" else "visual_backbone_quality"
    original = importlib.import_module(f"{__package__}.{module_name}")
    config = TASK / "configs" / "baselines" / CONFIG_NAMES[args.baseline]
    raw = original.read_config(config)
    original.validate_config(raw)
    validate_original_protocol(raw, args.baseline)
    rows = core.read_manifest(args.manifest)
    if len(rows) != 2808 or len({row["sample_id"] for row in rows}) != 2808:
        raise ValueError("Expected original 2808 unique manifest rows")
    if sum(row["split"] == "train" for row in rows) != 2250:
        raise ValueError("Original TRAIN split mismatch")
    gpu_name = validate_cuda_device("A100")
    versions = {}
    for name in ("torch", "transformers", "xformers", "ultralytics", "clip", "numpy", "opencv-python"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    if args.baseline == "deepseek" and (versions["transformers"], versions["xformers"]) != ("4.38.2", "0.0.29.post2"):
        raise ValueError("Use the original pinned DeepSeek environment")
    if args.baseline == "yolo" and versions["ultralytics"] != "8.4.159":
        raise ValueError("Use the original pinned Ultralytics 8.4.159 environment")
    identity = {
        "origin": "retrained_original_protocol", "baseline": args.baseline,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "script_sha256": frozen.sha256(Path(__file__)),
        "original_source_sha256": frozen.sha256(Path(original.__file__)),
        "source_archive_ref": ARCHIVE_REF if args.baseline == "deepseek" else "95904ecf97bcefe4a7ae76a6eab08a1d7cb217f0",
        "config_sha256": frozen.sha256(config), "resolved_config": raw,
        "manifest": str(args.manifest), "manifest_sha256": frozen.MANIFEST_SHA,
        "backbone": str(args.model), "backbone_sha256": frozen.sha256(weight),
        "gpu_uuid": frozen.A100_UUID, "gpu_name": gpu_name, "pid": os.getpid(),
        "actual_power_limit_w": gpu_power_limit_w(frozen.A100_UUID), "packages": versions,
        "model_auxiliary_file_sha256": ({path.name: frozen.sha256(path) for path in sorted(args.model.iterdir())
            if path.is_file() and path.suffix in (".json", ".py", ".model", ".txt")}
            if args.model.is_dir() else {}),
        "python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
        "training_contract": CONTRACT, "historical_reference_srcc": OLD_SRCC[args.baseline],
        "historical_checkpoint_sha256": frozen.CHECKPOINT_SHA[args.baseline],
        "new_weights_do_not_recover_historical_checkpoint": True,
    }
    args.output.mkdir(parents=True, exist_ok=False)
    frozen.write_json(args.output / "run_identity.json", identity)
    status = {"status": "loading_frozen_backbone", **identity}
    frozen.write_json(args.output / "status.json", status)
    handles = []
    try:
        device = torch.device("cuda:0")
        if args.baseline == "deepseek":
            processor, model, row_data = original._load_backbone(args.model, device)
            if (row_data["metadata"]["quality_token_ids"] != [[44379], [82916], [72572], [17259], [81229]]
                    or row_data["metadata"]["row_initialization"] != "native_single_token_rows"
                    or row_data["metadata"]["initial_rows_sha256"] != "66696e2966da5db4afb82a22ec55fab998e1a29fca64e7c0b6f8ade262150123"):
                raise ValueError("Original DeepSeek tokenizer/native initial quality rows changed")
            layer = model.vision
            feature_path = args.output / "features" / "temporal" / original.FEATURE_FILENAME
        else:
            model, extractor, initial_rows, metadata = original.load_backbone(raw, args.model, device)
            layer = model.visual if args.baseline == "clip" else model
            feature_path = args.output / "features" / original.FEATURE_FILENAME
        if any(parameter.requires_grad for parameter in model.parameters()):
            raise RuntimeError("Backbone has trainable parameters")
        # Safety-only hook; does not replace any computation or precision policy.
        count = [0]
        def audit_progress(_module, _inputs):
            if count[0] % 25 == 0:
                cuda_guard()
                frozen.write_json(args.output / "status.json", {
                    "status": "extracting_frozen_features", "started_video_forwards": count[0] + 1,
                    "updated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"), **identity})
            count[0] += 1
        handles.append(layer.register_forward_pre_hook(audit_progress))
        if args.baseline == "deepseek":
            model_metadata = dict(row_data["metadata"])
            model_metadata["_initial_quality_rows"] = row_data["rows"]
            original.extract_target(raw=raw, rows=rows, target="temporal", model_path=args.model,
                output=feature_path, processor=processor, model=model,
                model_metadata=model_metadata, device=device)
            del processor, row_data
        else:
            original.extract_all(raw=raw, rows=rows, model_path=args.model, output=feature_path,
                extractor=extractor, initial_rows=initial_rows, model_metadata=metadata)
            if metadata.get("hook_handle") is not None:
                metadata["hook_handle"].remove()
            del extractor, initial_rows, metadata
        for handle in handles:
            handle.remove()
        handles.clear()
        del model, layer
        torch.cuda.empty_cache()
        cuda_guard()
        payload = torch.load(feature_path, map_location="cpu", weights_only=False)
        targets = payload["targets"] if args.baseline == "deepseek" else payload["temporal_targets"]
        expected_targets = torch.tensor([row["temporal"] for row in rows], dtype=torch.float32)
        if (payload["sample_ids"] != [row["sample_id"] for row in rows]
                or payload["splits"] != [row["split"] for row in rows]
                or not torch.equal(targets.float(), expected_targets)
                or payload["features"].dtype != torch.float16
                or not bool(torch.isfinite(payload["features"]).all())):
            raise RuntimeError("Extracted cache differs from original sample/split/MOS/FP16 contract")
        frozen.write_json(args.output / "status.json", {"status": "training_temporal_head", **identity})
        report = original.train_target(raw=raw, target="temporal", feature_path=feature_path,
            output_dir=args.output / "checkpoints" / "temporal")
        cuda_guard()
        if report["epochs"] != 50 or report["train_samples"] != 2250 or report["test_samples"] != 558:
            raise RuntimeError("Training report does not match original protocol")
        indices = torch.tensor([index for index, row in enumerate(rows) if row["split"] == "test"])
        features = payload["features"][indices].float()
        cpu_results = {}
        for choice in ("best", "last"):
            checkpoint = torch.load(args.output / "checkpoints" / "temporal" / f"{choice}_checkpoint.pt",
                                    map_location="cpu", weights_only=False)
            labels = torch.bucketize(targets[indices].float(), checkpoint["boundaries"], right=False)
            head = original.FiveQualityRows(checkpoint["state_dict"]["weight"]).eval().requires_grad_(False)
            metrics, predictions, _ = original.evaluate(head, features, targets[indices].float(), labels,
                checkpoint["level_scores"], torch.device("cpu"))
            if not bool(torch.isfinite(predictions).all()):
                raise RuntimeError("Non-finite strict CPU predictions")
            cpu_results[choice] = {"metrics": metrics, "epoch": checkpoint["selected_epoch"], "test_samples": len(indices)}
            if choice == "best":
                with (args.output / "cpu_predictions.csv").open("w", encoding="utf-8", newline="") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(["sample_id", "target_mos", "prediction"])
                    writer.writerows((rows[index]["sample_id"], float(targets[index]), float(value))
                                     for index, value in zip(indices.tolist(), predictions))
                old = list(csv.DictReader((args.output / "checkpoints" / "temporal" / "test_predictions.csv").open()))
                delta = max(abs(float(a["prediction_mos"]) - float(b)) for a, b in zip(old, predictions))
                if len(old) != 558 or delta > 0.0002:
                    raise RuntimeError(f"CPU/GPU prediction verification failed: max abs {delta}")
                cpu_results[choice]["max_abs_gpu_prediction_delta"] = delta
        frozen.write_json(args.output / "cpu_verification.json", cpu_results)
        after_sha = frozen.sha256(weight)
        if after_sha != frozen.BACKBONE_SHA[args.baseline]:
            raise RuntimeError("Backbone file changed during reproduction")
        artifacts = {path.relative_to(args.output).as_posix(): frozen.sha256(path)
                     for path in sorted(args.output.rglob("*"))
                     if path.is_file() and path.suffix in (".pt", ".csv", ".json") and path.name != "status.json"}
        receipt = {**identity, "status": "complete", "backbone_sha256_after": after_sha,
                   "best_test_srcc": report["best_test_metrics"]["srcc"],
                   "difference_from_historical_srcc": report["best_test_metrics"]["srcc"] - OLD_SRCC[args.baseline],
                   "cpu_verification": cpu_results, "artifacts": artifacts}
        frozen.write_json(args.output / "reproduction_receipt.json", receipt)
        frozen.write_json(args.output / "status.json", {"status": "complete", **identity})
        print(json.dumps({"baseline": args.baseline, "status": "trained_original_protocol", "srcc": receipt["best_test_srcc"]}), flush=True)
    except BaseException as error:
        frozen.write_json(args.output / "status.json", {"status": "failed", "error": repr(error), **identity})
        raise
    finally:
        for handle in handles:
            handle.remove()


def run_suite(args) -> None:
    require_new_output(args.output)
    cuda_guard()
    # A cooperative task lock prevents duplicate suites; foreign jobs are
    # additionally checked by UUID before each subprocess and within inference.
    import fcntl
    OWNER.mkdir(parents=True, exist_ok=True)
    with (OWNER / ".temporal_reproduction_a100.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        args.output.mkdir(parents=True, exist_ok=False)
        state = {"status": "running", "pid": os.getpid(), "completed": [],
                 "gpu_uuid": frozen.A100_UUID, "manifest": str(args.manifest),
                 "origin": "retrained_original_protocol", "training_contract": CONTRACT}
        frozen.write_json(args.output / "suite_status.json", state)
        try:
            for baseline in CONFIG_NAMES:
                cuda_guard()
                state["phase"] = f"{baseline}_extract_and_train"
                frozen.write_json(args.output / "suite_status.json", state)
                command = [ENV_PATHS[baseline], "-u", "-m", __package__ + ".reproduce_temporal_baseline",
                           "--baseline", baseline, "--manifest", str(args.manifest),
                           "--model", str(MODEL_PATHS[baseline]), "--output", str(args.output / baseline)]
                with (args.output / f"{baseline}_training.log").open("x") as log:
                    subprocess.run(command, cwd=REPO, check=True, stdout=log, stderr=subprocess.STDOUT)
                state["completed"].append(state["phase"])
                frozen.write_json(args.output / "suite_status.json", state)
            for baseline in CONFIG_NAMES:
                for batch in (1, 2, 4, 8):
                    cuda_guard()
                    state["phase"] = f"{baseline}_frozen_batch{batch}"
                    frozen.write_json(args.output / "suite_status.json", state)
                    command = [ENV_PATHS[baseline], "-u", "-m", __package__ + ".evaluate_temporal_baseline_batches",
                               "--baseline", baseline, "--batch", str(batch), "--manifest", str(args.manifest),
                               "--model", str(MODEL_PATHS[baseline]), "--checkpoint",
                               str(args.output / baseline / "checkpoints" / "temporal" / "best_checkpoint.pt"),
                               "--reproduction-receipt", str(args.output / baseline / "reproduction_receipt.json"),
                               "--output", str(args.output / baseline / f"batch{batch:02d}")]
                    with (args.output / f"{baseline}_batch{batch:02d}.log").open("x") as log:
                        subprocess.run(command, cwd=REPO, check=True, stdout=log, stderr=subprocess.STDOUT)
                    state["completed"].append(state["phase"])
            state["status"] = "complete"
        except BaseException as error:
            state.update(status="failed", error=repr(error))
            raise
        finally:
            frozen.write_json(args.output / "suite_status.json", state)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", choices=tuple(CONFIG_NAMES))
    parser.add_argument("--suite", action="store_true")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if bool(args.suite) == bool(args.baseline):
        parser.error("Choose exactly one of --suite or --baseline")
    for name in ("manifest", "output", "model"):
        if getattr(args, name) is not None:
            setattr(args, name, getattr(args, name).expanduser().resolve())
    if args.suite:
        run_suite(args)
    else:
        if args.model is None:
            parser.error("--baseline requires --model")
        run_one(args)


if __name__ == "__main__":
    main()
