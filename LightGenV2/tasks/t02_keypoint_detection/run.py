"""Single formal entry point for LSP main and matched-D2NN runs."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.qwen3_vl_embedding_2b_lsp_pose_optical_moe16.datasets import prepare_lsp
from experiments.qwen3_vl_embedding_2b_lsp_pose_optical_router.protocol import (
    build_periodic_test_protocol,
    persist_protocol,
)

from .modeling import architecture_report, build_student, load_vision_backbone
from .settings import load_settings, save_resolved_config
from .training import evaluate_selected_checkpoint, train
from .visualize import render


TASK_DIR = Path(__file__).resolve().parent
PROFILES = {
    "alpha40_distill": "moe_alpha40_distill.yaml",
    "alpha40_polish": "moe_alpha40_polish.yaml",
    "alpha40": "moe_alpha40.yaml",
    "alpha50": "moe_alpha50.yaml",
    "main_dc20": "moe_optical_router_scale_matched_dc20.yaml",
    "main_dc20_no_shift": "moe_optical_router_scale_matched_dc20_no_shift.yaml",
    "main_dc20_no_shift_warmstart": "moe_optical_router_scale_matched_dc20_no_shift_warmstart.yaml",
    "d2nn_dc20": "d2nn_active_expert_matched_dc20.yaml",
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
            ["git", *arguments], cwd=TASK_DIR, check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _device(settings: Any) -> torch.device:
    requested = str(settings.device)
    return torch.device(requested if torch.cuda.is_available() else "cpu")


def _curate(settings: Any) -> dict[str, Any]:
    output = settings.output_dir
    source = output / "checkpoints" / "ema_best_periodic_test_pck.pt"
    if not source.is_file():
        raise FileNotFoundError(f"Selected LSP checkpoint missing: {source}")
    best = output / "best_checkpoint.pt"
    shutil.copy2(source, best)
    render(best, output / "best_visualization")
    removed: list[str] = []
    checkpoints = output / "checkpoints"
    if checkpoints.is_dir():
        shutil.rmtree(checkpoints)
        removed.append("checkpoints")
    for candidate in (output / "metrics").glob("periodic_test_epoch_*.json"):
        candidate.unlink()
        removed.append(str(candidate.relative_to(output)))
    report = {
        "policy": "retain best_checkpoint.pt and last_checkpoint.pt only",
        "best_checkpoint": str(best),
        "last_checkpoint": str(output / "last_checkpoint.pt"),
        "periodic_phase_checkpoints_retained": False,
        "removed": removed,
    }
    (output / "artifact_retention.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = TASK_DIR / "configs" / PROFILES[args.profile]
    settings = load_settings(config)
    if args.run_dir:
        settings.output_dir = Path(args.run_dir).expanduser().resolve()
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    _seed(int(args.seed))
    save_resolved_config(settings)
    (settings.output_dir / "run_manifest.json").write_text(
        json.dumps({
            "schema_version": 1,
            "task": "t02_keypoint_detection",
            "profile": args.profile,
            "variant": settings.lightgen_model_variant,
            "seed": int(args.seed),
            "git_commit": _git_value("rev-parse", "HEAD"),
            "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "selection": "maximum periodic-test PCK@0.2; test evaluated every 5 epochs",
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    official = prepare_lsp(settings, persist=False)
    bundle = build_periodic_test_protocol(official)
    persist_protocol(bundle, settings.output_dir)
    if args.phase == "prepare":
        return {"status": "prepared", "train": len(bundle.train), "test": len(bundle.test)}

    loaded = load_vision_backbone(settings, _device(settings))
    settings.resolve_architecture(loaded.model)
    save_resolved_config(settings)
    model = build_student(loaded, settings)
    try:
        (settings.output_dir / "student_architecture.json").write_text(
            json.dumps(architecture_report(model, settings), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    finally:
        model.restore_native()
    if args.phase in {"train", "all"}:
        result = train(loaded, bundle, settings)
        _curate(settings)
        if args.phase == "train":
            return result
    checkpoint = (
        Path(args.checkpoint).expanduser().resolve()
        if args.checkpoint
        else settings.output_dir / settings.lightgen_primary_checkpoint
    )
    return evaluate_selected_checkpoint(loaded, bundle, settings, checkpoint)


def main() -> int:
    parser = argparse.ArgumentParser(description="LightGenV2 T02 LSP formal comparison")
    parser.add_argument("--profile", choices=sorted(PROFILES), required=True)
    parser.add_argument("--phase", choices=sorted(PHASES), default="all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()
    if args.checkpoint and args.phase not in {"evaluate", "all"}:
        parser.error("--checkpoint is only valid for evaluate/all")
    print(json.dumps(run(args), ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
