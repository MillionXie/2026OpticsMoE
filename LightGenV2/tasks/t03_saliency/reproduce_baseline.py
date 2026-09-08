"""Portable performance-only reproduction; no timing/power claims on training GPUs."""
from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from LightGenV2.common.baseline_measurement import sha256_file, write_json
from .baseline_5090d import _validation_only_bundle
from experiments.qwen3_vl_embedding_2b_salicon_vision_optical_saliency import datasets, modeling, training
from experiments.qwen3_vl_embedding_2b_salicon_vision_optical_saliency.objectives import SaliencyAccumulator, density_from_logits
from experiments.qwen3_vl_embedding_2b_salicon_vision_optical_saliency.settings import load_settings, save_resolved_config

ROOT = Path(__file__).resolve().parents[3]
LEGACY = ROOT / "experiments/qwen3_vl_embedding_2b_salicon_vision_optical_saliency"


def independent_cc(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Per-image Pearson in float64, independent of the training metric code."""
    x = np.asarray(prediction, dtype=np.float64).reshape(len(prediction), -1)
    y = np.asarray(target, dtype=np.float64).reshape(len(target), -1)
    x = x - x.mean(1, keepdims=True)
    y = y - y.mean(1, keepdims=True)
    denominator = np.sqrt((x*x).sum(1) * (y*y).sum(1))
    return np.divide((x*y).sum(1), denominator, out=np.zeros(len(x)), where=denominator > 0)


def run(args):
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        raise FileExistsError("Use an empty --run-dir; do not overwrite reproduction evidence")
    settings = load_settings(args.config.resolve())
    settings.model_id = str(args.model.resolve())
    settings.data_root = args.data_root.resolve()
    settings.local_files_only = True
    settings.download = False
    settings.output_dir = args.run_dir.resolve()
    settings.artifact_cache_dir = settings.output_dir / "prepared_maps"
    settings.inference_batch_size = args.batch_size
    settings.num_workers = args.workers
    settings.random_seed = args.seed
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    save_resolved_config(settings)
    bundle = datasets.prepare_salicon(settings, persist=True) if args.retrain_head else _validation_only_bundle(settings)
    if args.retrain_head and len(bundle.train_records) != 10000:
        raise RuntimeError("Baseline retraining requires all 10000 train2014 images")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded = modeling.load_vision_backbone(settings, device)
    checkpoint = training.train_teacher(loaded, bundle, settings) if args.retrain_head else args.checkpoint.resolve()
    model = modeling.build_teacher(loaded, settings)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.head.load_state_dict(payload["head"], strict=True)
    model.eval()
    _, loader = training.build_loaders(bundle, settings, training=False)
    accumulator = SaliencyAccumulator()
    rows = []
    try:
        with torch.inference_mode():
            for batch in loader:
                inputs = modeling.preprocess_vision(loaded.processor, batch["images"], loaded.device)
                logits = model(inputs["pixel_values"], inputs["image_grid_thw"])[0]
                target = batch["density"].to(device)
                accumulator.update(logits, target, batch["fixation"].to(device))
                cc = independent_cc(density_from_logits(logits).cpu().numpy(), target.cpu().numpy())
                rows.extend({"sample_id": sid, "cc_float64": float(value)} for sid, value in zip(batch["sample_ids"], cc))
                if len(rows) % 512 < args.batch_size:
                    print(f"[reproduce] {len(rows)}/5000 CC64={np.mean([r['cc_float64'] for r in rows]):.6f}", flush=True)
    finally:
        model.close()
    with (settings.output_dir / "per_image_cc.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["sample_id", "cc_float64"])
        writer.writeheader()
        writer.writerows(rows)
    weights = sorted(args.model.glob("*.safetensors"))
    hashed_files = [args.config.resolve(), checkpoint, datasets._annotation_path(settings.data_root, "validation"), *weights]
    if args.retrain_head:
        hashed_files.append(datasets._annotation_path(settings.data_root, "train"))
    manifest = [{"path": str(p), "bytes": p.stat().st_size, "sha256": sha256_file(p)} for p in hashed_files]
    metrics = accumulator.compute()
    cc64 = float(np.mean([row["cc_float64"] for row in rows]))
    report = {
        "status": "complete", "mode": "retrain_frozen_backbone_head" if args.retrain_head else "fixed_checkpoint_reevaluation",
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "command": [sys.executable, "-m", "LightGenV2.tasks.t03_saliency.reproduce_baseline", *sys.argv[1:]],
        "python": sys.version, "torch": torch.__version__, "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU",
        "test_samples": len(rows), "selected_epoch": payload.get("epoch"), "seed": args.seed,
        "performance": metrics, "independent_float64_cc": cc64,
        "cc_implementation_difference": abs(float(metrics["cc"]) - cc64),
        "historical_reference_cc": 0.881051770, "difference_from_historical": float(metrics["cc"]) - 0.881051770,
        "selection_biased": True, "protocol": "official val2014 as public test, selected by test CC; 224px output density, sigma=19 source-image pixels scaled per axis; not official hidden-test leaderboard",
        "native_vision_blocks": len(loaded.visual.blocks),
        "head_parameters": sum(p.numel() for p in model.head.parameters()),
        "files": manifest, "speed_and_power": "not measured",
    }
    write_json(settings.output_dir / "reproduction.json", report)
    (settings.output_dir / "environment.txt").write_text(subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=LEGACY / "configs/salicon.yaml")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/SALICON")
    parser.add_argument("--checkpoint", type=Path, default=LEGACY / "runs/salicon_vision_optical_saliency/checkpoints/teacher_best.pt")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--retrain-head", action="store_true", help="Train a new head for the original 30 epochs; backbone stays frozen")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
