"""Run the controlled response-preserving MoE4 Grocery31 -> Grocery10 route."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import torch


MODULE = "experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval"


def _epoch(path: Path) -> int:
    if not path.is_file():
        return 0
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
            f"Incomplete optimization output exists: {output}. Move it aside before "
            "a clean run; partial checkpoints are never silently overwritten."
        )
    return False


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    experiment = Path(__file__).resolve().parent
    repository = experiment.parents[1]
    config_dir = experiment / "configs" / "optimization"
    stage1_config = config_dir / "response_preserving_stage1_grocery31.yaml"
    stage2_config = config_dir / "response_preserving_stage2_grocery10.yaml"
    stage1_output = experiment / "runs" / "optimization_moe4_response_grocery31_pretrain"
    stage2_output = experiment / "runs" / "optimization_moe4_response_grocery10_epoch40"
    stage1_checkpoint = stage1_output / "ema_last_checkpoint.pt"
    fixed_checkpoint = stage2_output / "ema_last_checkpoint.pt"

    if not _complete_or_empty(stage1_output, 26):
        _run(
            [sys.executable, "-m", MODULE, "--config", str(stage1_config), "--phase", "all"],
            repository,
            args.dry_run,
        )
    if not args.dry_run and _epoch(stage1_checkpoint) != 26:
        raise RuntimeError(f"Stage-1 checkpoint is not epoch 26: {stage1_checkpoint}")

    if not _complete_or_empty(stage2_output, 40):
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
    if _epoch(fixed_checkpoint) != 40:
        raise RuntimeError(f"Stage-2 checkpoint is not epoch 40: {fixed_checkpoint}")

    # Report a non-selection-biased fixed-epoch result first.
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
    fixed_metrics = _json(stage2_output / "student_metrics.json")

    # The user explicitly wants the highest observed route as a diagnostic.
    # Keep its selection bias visible rather than presenting it as a held-out
    # model-selection result.
    observed_candidates: list[tuple[float, Path, dict]] = []
    for prefix in ("", "ema_"):
        metadata_path = stage2_output / "metrics" / f"{prefix}best_observed_test.json"
        checkpoint_path = stage2_output / f"{prefix}best_observed_test_checkpoint.pt"
        if metadata_path.is_file() and checkpoint_path.is_file():
            metadata = _json(metadata_path)
            observed_candidates.append(
                (float(metadata.get("test_top1", -math.inf)), checkpoint_path, metadata)
            )
    if not observed_candidates:
        raise RuntimeError("Training completed without an observed-test checkpoint")
    _, observed_checkpoint, observed_metadata = max(observed_candidates, key=lambda row: row[0])
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
    observed_metrics = _json(stage2_output / "student_metrics.json")
    _run(
        [sys.executable, "-m", MODULE, "--config", str(stage2_config), "--phase", "visualize"],
        repository,
        False,
    )
    comparison = {
        "fixed_epoch_40_ema": fixed_metrics,
        "best_observed_test": observed_metrics,
        "best_observed_metadata": observed_metadata,
        "best_observed_is_selection_biased": True,
        "canonical_release_top1_reference": 0.6769230962,
    }
    (stage2_output / "response_preserving_comparison.json").write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(comparison, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
