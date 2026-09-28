"""Train and evaluate the in-training-interpolation temporal simulation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import subprocess
import sys

import torch
import yaml

from .camera import camera_operator, validate_profile
from .components.data import load_single_metric_cache
from .model import build_model
from .settings import REPO_ROOT, load_settings
from .simulation import synthetic_smoke
from . import training

CONFIG_ROOT = Path(__file__).resolve().parent / "configs"


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _read_config(path: Path, path_overrides: Path | None, output: Path, device: str, seed: int) -> dict:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if path_overrides:
        overrides = json.loads(path_overrides.read_text(encoding="utf-8"))
        for key, value in overrides.items():
            if key not in raw["data"]:
                raise ValueError(f"Unknown data path: {key}")
            raw["data"][key] = value
    raw["output_dir"] = str(output.resolve())
    raw["device"] = device
    raw["random_seed"] = seed
    return raw


def _asset_check(settings) -> dict:
    required = {
        "manifest": settings.manifest_path,
        "vision_cache": settings.vision_cache_path,
        "language_cache": settings.language_cache_path,
    }
    if settings.soft_target_weight > 0:
        required["training_soft_targets"] = settings.training_soft_targets_path
    missing = [name for name, path in required.items() if path is None or not path.is_file()]
    return {"ready": not missing, "missing": missing}


def _check_split(payload: dict) -> None:
    splits = payload["splits"]
    ids = list(map(str, payload["sample_ids"]))
    if len(splits) != len(ids) or len(set(ids)) != len(ids):
        raise ValueError("Duplicate or unaligned sample identities")
    if splits.count("train") != 2250 or splits.count("test") != 558 or set(splits) != {"train", "test"}:
        raise ValueError("Expected the original 2250 train / 558 test split")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("smoke", "preflight", "train", "evaluate"))
    parser.add_argument("--config", type=Path, default=CONFIG_ROOT / "simulation.yaml")
    parser.add_argument("--camera", type=Path, default=CONFIG_ROOT / "camera.json")
    parser.add_argument("--paths", type=Path, help="Local JSON file with private data/cache paths")
    parser.add_argument("--output", type=Path, help="New output directory; existing results are never overwritten")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=163)
    parser.add_argument("--noise-scale", type=float, default=0.0)
    parser.add_argument("--allow-uncalibrated-noise", action="store_true")
    args = parser.parse_args(argv)
    if args.phase == "evaluate" and args.checkpoint is None:
        parser.error("evaluate requires --checkpoint")
    if args.phase == "train" and args.noise_scale != 0:
        parser.error("--noise-scale is only used for evaluation")
    output = args.output or REPO_ROOT / "runs" / args.phase
    if output.exists() and any(output.iterdir()):
        parser.error("Output directory is not empty")
    camera = json.loads(args.camera.read_text(encoding="utf-8"))
    validate_profile(camera)
    if args.phase == "train" and not camera.get("calibrated", False) and not args.allow_uncalibrated_noise:
        parser.error("Camera parameters are a pilot; pass --allow-uncalibrated-noise to train")
    raw = _read_config(args.config, args.paths, output, args.device, args.seed)
    output.mkdir(parents=True, exist_ok=True)
    resolved = output / "resolved.yaml"
    resolved.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    settings = load_settings(resolved, synthetic=args.phase == "smoke")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if args.phase == "smoke":
        with camera_operator(camera):
            result = synthetic_smoke(settings)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    assets = _asset_check(settings)
    if args.phase == "preflight":
        print(json.dumps(assets, ensure_ascii=False, indent=2))
        return 0 if assets["ready"] else 2
    if not assets["ready"]:
        parser.error(f"Missing assets: {', '.join(assets['missing'])}")
    payload = load_single_metric_cache(settings)
    _check_split(payload)
    model = build_model(settings)
    device = torch.device(args.device)
    if args.phase == "train":
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                                capture_output=True, text=True)
        _write_json(output / "run_manifest.json", {
            "command": sys.argv if argv is None else argv,
            "git_commit": commit.stdout.strip() if commit.returncode == 0 else None,
            "train_count": 2250, "test_count": 558,
            "test_used_for_selection": True,
            "camera_calibrated": bool(camera.get("calibrated", False)),
            "resolved_config": str(resolved),
        })
        with camera_operator(camera):
            result = training.train(model, payload, settings, device)
    else:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.to(device).eval()
        with camera_operator(camera, evaluation_scale=args.noise_scale):
            result = training.evaluate(model, payload, settings, device,
                                       optical_enabled=True, prediction_path=output / "predictions.csv")
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0
