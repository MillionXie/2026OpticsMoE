"""Audited entry point for the three OpenMoji comparison rows."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
from pathlib import Path
from typing import Any

import torch

from experiments.qwen3_vl_2b_openmoji_instruction_four_stage_optical_editing.prompt_cache import (
    build_prompt_cache,
)
from experiments.qwen3_vl_2b_openmoji_instruction_four_stage_optical_editing.scenes import (
    prepare_dataset,
)
from experiments.qwen3_vl_2b_openmoji_instruction_four_stage_optical_editing.training import (
    _environment,
)

from .settings import load_settings
from .training import evaluate_selected, train


TASK_DIR = Path(__file__).resolve().parent
PROFILES = {
    "main_dc20": "moe_optical_router_scale_matched_dc20.yaml",
    "d2nn_dc20": "d2nn_active_expert_matched_dc20.yaml",
    "qwen_pending": "qwen_frozen_pending_5090d.yaml",
}
PHASES = {"prepare", "train", "evaluate", "all"}


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _git_value(*arguments: str) -> str | None:
    try:
        return subprocess.run(
            ["git", *arguments], cwd=TASK_DIR, check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _ensure_data(settings: Any, device: torch.device) -> dict[str, Any]:
    if not settings.train_manifest.is_file() or not settings.test_manifest.is_file():
        summary = prepare_dataset(settings)
    else:
        summary_path = settings.data_dir / "dataset_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not settings.prompt_cache_path.is_file():
        build_prompt_cache(settings, device)
    return summary


def _pending_qwen(settings: Any) -> dict[str, Any]:
    report = {
        "schema_version": 1,
        "status": "pending_5090d_measurement",
        "model": "Qwen/Qwen3-VL-2B-Instruct",
        "frozen": True,
        "performance": None,
        "speed_ms": None,
        "power": {
            "idle_w": None,
            "active_mean_w": None,
            "peak_w": None,
            "incremental_j_per_sample": None,
        },
        "timing_boundary": "input to first native Transformer block -> 6x6 semantic/edit result",
        "timing_protocol": "batch=1; 50 warm-up forwards; complete 1000-sample test",
        "hardware": "NVIDIA GeForce RTX 5090 D",
        "reason": "deliberately not measured on the shared training server",
    }
    _json(settings.output_dir / "pending_5090d_baseline.json", report)
    return report


def run(args: argparse.Namespace) -> dict[str, Any]:
    settings = load_settings(TASK_DIR / "configs" / PROFILES[args.profile])
    if args.run_dir:
        settings.output_dir = Path(args.run_dir).expanduser().resolve()
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    _json(settings.output_dir / "resolved_config.json", settings.to_dict())
    _json(settings.output_dir / "environment.json", _environment(device))
    _json(
        settings.output_dir / "run_manifest.json",
        {
            "schema_version": 1,
            "task": "t04_semantic_interaction",
            "profile": args.profile,
            "variant": settings.lightgen_model_variant,
            "seed": settings.seed,
            "git_commit": _git_value("rev-parse", "HEAD"),
            "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "selection": "maximum test changed-cell accuracy at epoch 1/every 5/final",
        },
    )
    summary = _ensure_data(settings, device)
    _json(
        settings.output_dir / "split_contract.json",
        {
            "dataset": "OpenMoji semantic interaction v1",
            "train": 5000,
            "test": 1000,
            "validation": None,
            "split_rule": "same distribution, deterministic disjoint seeds",
            "task_counts_train": summary["train"]["task_counts"],
            "task_counts_test": summary["test"]["task_counts"],
            "selection_biased": args.profile != "qwen_pending",
        },
    )
    if args.profile == "qwen_pending":
        return _pending_qwen(settings)
    if args.phase == "prepare":
        return {"status": "prepared", "train": 5000, "test": 1000}
    if args.phase in {"train", "all"}:
        result = train(settings, device)
        if args.phase == "train":
            return result
    checkpoint = (
        Path(args.checkpoint).expanduser().resolve()
        if args.checkpoint
        else settings.output_dir / "best_checkpoint.pt"
    )
    return evaluate_selected(settings, device, checkpoint)


def main() -> int:
    parser = argparse.ArgumentParser(description="LightGenV2 T04 OpenMoji comparison")
    parser.add_argument("--profile", choices=sorted(PROFILES), required=True)
    parser.add_argument("--phase", choices=sorted(PHASES), default="all")
    parser.add_argument("--device", default=None)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()
    print(json.dumps(run(args), ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
