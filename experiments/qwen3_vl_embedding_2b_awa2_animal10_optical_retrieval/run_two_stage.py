"""Run the reproducible AwA2 all-50 pretrain -> target-10 adaptation route."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

MODULE = "experiments.qwen3_vl_embedding_2b_awa2_animal10_optical_retrieval"
STAGE1_EPOCH = 30
FINAL_EPOCH = 50


def _epoch(path: Path) -> int:
    if not path.is_file():
        return 0
    import torch

    return int(torch.load(path, map_location="cpu", weights_only=False).get("epoch", 0))


def _run(command: list[str], repository: Path, dry_run: bool) -> None:
    print("+ " + " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=repository, check=True)


def _complete_or_empty(output: Path, expected_epoch: int) -> bool:
    if _epoch(output / "last_checkpoint.pt") == expected_epoch and _epoch(
        output / "ema_last_checkpoint.pt"
    ) == expected_epoch:
        return True
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(
            f"Incomplete output exists: {output}. Move it aside before a clean two-stage run; "
            "partial results are never overwritten silently."
        )
    return False


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    experiment = Path(__file__).resolve().parent
    repository = experiment.parents[1]
    stage1_config = experiment / "configs" / "awa2_50class_pretrain.yaml"
    stage2_config = experiment / "configs" / "awa2_10class_finetune.yaml"
    stage1_output = experiment / "runs" / "awa2_50class_pretrain"
    stage2_output = experiment / "runs" / "awa2_10class_finetune_epoch50"
    stage1_checkpoint = stage1_output / "ema_last_checkpoint.pt"
    fixed_checkpoint = stage2_output / "ema_last_checkpoint.pt"

    if not _complete_or_empty(stage1_output, STAGE1_EPOCH):
        _run(
            [sys.executable, "-m", MODULE, "--config", str(stage1_config), "--phase", "all"],
            repository,
            args.dry_run,
        )
    if not args.dry_run and _epoch(stage1_checkpoint) != STAGE1_EPOCH:
        raise RuntimeError(f"Stage-1 EMA checkpoint is not epoch {STAGE1_EPOCH}: {stage1_checkpoint}")

    if not _complete_or_empty(stage2_output, FINAL_EPOCH):
        _run(
            [
                sys.executable,
                "-m",
                MODULE,
                "--config",
                str(stage2_config),
                "--phase",
                "cache_teacher_embeddings",
            ],
            repository,
            args.dry_run,
        )
        _run(
            [
                sys.executable,
                "-m",
                MODULE,
                "--config",
                str(stage2_config),
                "--phase",
                "train",
                "--resume-checkpoint",
                str(stage1_checkpoint),
            ],
            repository,
            args.dry_run,
        )
    if args.dry_run:
        return 0
    if _epoch(fixed_checkpoint) != FINAL_EPOCH:
        raise RuntimeError(f"Stage-2 EMA checkpoint is not epoch {FINAL_EPOCH}: {fixed_checkpoint}")

    _run(
        [
            sys.executable,
            "-m",
            MODULE,
            "--config",
            str(stage2_config),
            "--phase",
            "evaluate",
            "--checkpoint",
            str(fixed_checkpoint),
        ],
        repository,
        False,
    )
    fixed_metrics = _read_json(stage2_output / "student_metrics.json")

    candidates: list[tuple[float, Path, dict]] = []
    for prefix in ("", "ema_"):
        metadata_path = stage2_output / "metrics" / f"{prefix}best_observed_test.json"
        checkpoint_path = stage2_output / f"{prefix}best_observed_test_checkpoint.pt"
        if metadata_path.is_file() and checkpoint_path.is_file():
            metadata = _read_json(metadata_path)
            candidates.append(
                (float(metadata.get("test_top1", -math.inf)), checkpoint_path, metadata)
            )
    comparison: dict[str, object] = {
        "fixed_epoch_50_ema": fixed_metrics,
        "fixed_result_is_primary": True,
        "selection_policy": "fixed two-stage schedule; no validation split",
    }
    if candidates:
        _, observed_checkpoint, observed_metadata = max(candidates, key=lambda row: row[0])
        _run(
            [
                sys.executable,
                "-m",
                MODULE,
                "--config",
                str(stage2_config),
                "--phase",
                "evaluate",
                "--checkpoint",
                str(observed_checkpoint),
            ],
            repository,
            False,
        )
        comparison.update(
            {
                "best_observed_test": _read_json(stage2_output / "student_metrics.json"),
                "best_observed_metadata": observed_metadata,
                "best_observed_is_selection_biased": True,
            }
        )
    _run(
        [sys.executable, "-m", MODULE, "--config", str(stage2_config), "--phase", "visualize"],
        repository,
        False,
    )
    destination = stage2_output / "two_stage_comparison.json"
    destination.write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(comparison, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
