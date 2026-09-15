"""One-command reproduction of the canonical MoE4 Grocery retrieval release."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch


MODULE = "experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval"


def _epoch(path: Path) -> int:
    if not path.is_file():
        return 0
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return int(payload.get("epoch", 0))


def _run(command: list[str], *, cwd: Path, dry_run: bool) -> None:
    print("+ " + " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=cwd, check=True)


def _require_clean_or_complete(output: Path, expected_epoch: int) -> bool:
    ema = output / "ema_last_checkpoint.pt"
    last = output / "last_checkpoint.pt"
    if _epoch(ema) == expected_epoch and _epoch(last) == expected_epoch:
        return True
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(
            f"Incomplete output already exists: {output}. Move it aside or remove it "
            "before starting a clean reproducibility run; this command will not "
            "silently overwrite partial checkpoints."
        )
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    experiment = Path(__file__).resolve().parent
    repository = experiment.parents[1]
    stage1_config = experiment / "configs/release/stage1_grocery31_pretrain.yaml"
    stage2_config = experiment / "configs/release/stage2_grocery10_finetune.yaml"
    stage1_output = experiment / "runs/release_moe4_grocery31_pretrain"
    stage2_output = experiment / "runs/release_moe4_grocery10_epoch40"
    stage1_checkpoint = stage1_output / "ema_last_checkpoint.pt"
    stage2_checkpoint = stage2_output / "ema_last_checkpoint.pt"
    started = time.time()

    stage1_complete = _require_clean_or_complete(stage1_output, 26)
    if not stage1_complete:
        _run(
            [
                sys.executable, "-m", MODULE, "--config", str(stage1_config),
                "--phase", "all",
            ],
            cwd=repository,
            dry_run=args.dry_run,
        )

    if not args.dry_run and _epoch(stage1_checkpoint) != 26:
        raise RuntimeError(
            f"Stage-1 EMA checkpoint is not epoch 26: {stage1_checkpoint}"
        )

    stage2_complete = _require_clean_or_complete(stage2_output, 40)
    if not stage2_complete:
        _run(
            [
                sys.executable, "-m", MODULE, "--config", str(stage2_config),
                "--phase", "cache_teacher_embeddings",
            ],
            cwd=repository,
            dry_run=args.dry_run,
        )
        _run(
            [
                sys.executable, "-m", MODULE, "--config", str(stage2_config),
                "--phase", "train", "--resume-checkpoint", str(stage1_checkpoint),
            ],
            cwd=repository,
            dry_run=args.dry_run,
        )

    if not args.dry_run and _epoch(stage2_checkpoint) != 40:
        raise RuntimeError(
            f"Stage-2 EMA checkpoint is not absolute epoch 40: {stage2_checkpoint}"
        )

    for phase in ("evaluate", "visualize"):
        command = [
            sys.executable, "-m", MODULE, "--config", str(stage2_config),
            "--phase", phase,
        ]
        if phase == "evaluate":
            command.extend(["--checkpoint", str(stage2_checkpoint)])
        _run(command, cwd=repository, dry_run=args.dry_run)

    summary = {
        "pipeline": "canonical_moe4_grocery31_to_grocery10",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.time() - started,
        "dry_run": args.dry_run,
        "stage1_checkpoint": str(stage1_checkpoint),
        "stage1_expected_epoch": 26,
        "stage2_checkpoint": str(stage2_checkpoint),
        "stage2_expected_absolute_epoch": 40,
        "checkpoint_selection": "EMA at fixed final epoch; test is observational only",
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if not args.dry_run:
        (stage2_output / "release_pipeline.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
