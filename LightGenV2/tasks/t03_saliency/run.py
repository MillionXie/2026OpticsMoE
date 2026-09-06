"""Audited T03 entry point: optical Router, D2NN, and pending Qwen baseline."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.qwen3_vl_embedding_2b_salicon_vision_optical_saliency.datasets import (
    prepare_salicon,
)
from experiments.qwen3_vl_embedding_2b_salicon_vision_optical_saliency.io_utils import (
    environment_report,
    write_json,
)

from .modeling import architecture_report, build_student, load_vision_backbone
from .settings import load_settings, save_resolved_config
from .training import evaluate_selected_checkpoint, train


TASK_DIR = Path(__file__).resolve().parent
PROFILES = {
    "main_dc20": "moe_optical_router_scale_matched_dc20.yaml",
    "d2nn_dc20": "d2nn_active_expert_matched_dc20.yaml",
    "qwen_pending": "qwen_frozen_pending_5090d.yaml",
}
PHASES = {"prepare", "train", "evaluate", "all"}


def _seed(value: int) -> None:
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def _git_value(*arguments: str) -> str | None:
    try:
        return subprocess.run(
            ["git", *arguments],
            cwd=TASK_DIR,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _write_protocol(settings: Any, profile: str, seed: int) -> None:
    write_json(
        settings.output_dir / "run_manifest.json",
        {
            "schema_version": 1,
            "task": "t03_saliency",
            "profile": profile,
            "variant": (
                "frozen_qwen_pending_5090d"
                if profile == "qwen_pending"
                else settings.lightgen_model_variant
            ),
            "seed": int(seed),
            "git_commit": _git_value("rev-parse", "HEAD"),
            "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "selection": "highest public-test CC at epoch 1/every 5/final",
        },
    )
    write_json(
        settings.output_dir / "split_contract.json",
        {
            "dataset": "SALICON 2015r1",
            "train": "official train2014, 10000 labeled images",
            "test": "official val2014, 5000 labeled images",
            "validation": None,
            "private_official_test": "not used because ground truth is private",
            "image_id_disjoint": True,
            "selection_biased": profile != "qwen_pending",
        },
    )


def _pending_qwen(settings: Any) -> dict[str, Any]:
    report = {
        "status": "pending_5090d_measurement",
        "model": "Qwen/Qwen3-VL-Embedding-2B",
        "frozen": True,
        "performance": None,
        "speed_ms": None,
        "power_w": None,
        "timing_boundary": "input to first native Vision Transformer block -> saliency output",
        "hardware": "NVIDIA GeForce RTX 5090 D",
        "reason": "deliberately not measured on the shared training server",
    }
    write_json(settings.output_dir / "pending_5090d_baseline.json", report)
    return report


def run(args: argparse.Namespace) -> dict[str, Any]:
    settings = load_settings(TASK_DIR / "configs" / PROFILES[args.profile])
    if args.run_dir:
        settings.output_dir = Path(args.run_dir).expanduser().resolve()
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    settings.artifact_cache_dir.mkdir(parents=True, exist_ok=True)
    _seed(args.seed)
    settings.random_seed = int(args.seed)
    save_resolved_config(settings)
    write_json(settings.output_dir / "environment.json", environment_report())
    _write_protocol(settings, args.profile, args.seed)
    bundle = prepare_salicon(settings, persist=True)
    if args.profile == "qwen_pending":
        return _pending_qwen(settings)
    if args.phase == "prepare":
        return {
            "status": "prepared",
            "train": len(bundle.train_records),
            "test": len(bundle.validation_records),
        }
    device = torch.device(settings.device if torch.cuda.is_available() else "cpu")
    loaded = load_vision_backbone(settings, device)
    settings.resolve_architecture(loaded.model)
    save_resolved_config(settings)
    model = build_student(loaded, settings)
    try:
        write_json(
            settings.output_dir / "student_architecture.json",
            architecture_report(model, settings),
        )
    finally:
        model.restore_native()
    if args.phase in {"train", "all"}:
        result = train(loaded, bundle, settings)
        if args.phase == "train":
            return result
    checkpoint = (
        Path(args.checkpoint).expanduser().resolve()
        if args.checkpoint
        else settings.output_dir / "best_checkpoint.pt"
    )
    return evaluate_selected_checkpoint(loaded, bundle, settings, checkpoint)


def main() -> int:
    parser = argparse.ArgumentParser(description="LightGenV2 T03 SALICON formal comparison")
    parser.add_argument("--profile", choices=sorted(PROFILES), required=True)
    parser.add_argument("--phase", choices=sorted(PHASES), default="all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
