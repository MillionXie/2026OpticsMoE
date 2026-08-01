from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from .settings import load_settings


PACKAGE = "experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval"


def _resolve(path: str | Path, base: Path) -> Path:
    value = Path(os.path.expandvars(os.path.expanduser(str(path))))
    return (value if value.is_absolute() else base / value).resolve()


def _run(command: list[str], *, dry_run: bool) -> None:
    print("+ " + " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, check=True)


def _copy_teacher_cache(source: Path, target: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"Reusable Teacher cache is missing: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    shutil.copy2(source, temporary)
    temporary.replace(target)
    print(f"Reused identical replacement-10 Teacher cache: {source} -> {target}")


def run_pipeline(config_path: Path, *, dry_run: bool, skip_completed: bool) -> dict[str, Any]:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or int(raw.get("pipeline_version", 0)) != 1:
        raise ValueError("Expected Grocery10 canonical pipeline_version=1")
    base = config_path.parent
    stage_specs = raw.get("stages")
    if not isinstance(stage_specs, list) or len(stage_specs) != 3:
        raise ValueError("Canonical Grocery10 pipeline must define exactly three stages")
    stages: dict[str, dict[str, Any]] = {}
    for spec in stage_specs:
        name = str(spec["name"])
        stage_config = _resolve(spec["config"], base)
        settings = load_settings(stage_config)
        stages[name] = {
            "name": name,
            "config": stage_config,
            "settings": settings,
            "resume_from": spec.get("resume_from"),
            "checkpoint_name": str(spec["checkpoint"]),
            "checkpoint": settings.output_dir / str(spec["checkpoint"]),
        }
    started = time.time()
    completed: list[dict[str, Any]] = []
    cache_rule = raw.get("reuse_teacher_cache", {})
    for index, spec in enumerate(stage_specs):
        stage = stages[str(spec["name"])]
        checkpoint: Path = stage["checkpoint"]
        if skip_completed and checkpoint.is_file():
            print(f"[skip] {stage['name']} already has {checkpoint}")
            completed.append(
                {"stage": stage["name"], "checkpoint": str(checkpoint), "skipped": True}
            )
            continue
        if (
            stage["name"] == cache_rule.get("to_stage")
            and not dry_run
        ):
            source_stage = stages[str(cache_rule["from_stage"])]
            _copy_teacher_cache(
                source_stage["settings"].teacher_cache_path,
                stage["settings"].teacher_cache_path,
            )
        command = [
            sys.executable,
            "-m",
            PACKAGE,
            "--config",
            str(stage["config"]),
            "--phase",
            "all",
        ]
        resume_name = stage["resume_from"]
        if resume_name is not None:
            source_checkpoint = stages[str(resume_name)]["checkpoint"]
            if not dry_run and not source_checkpoint.is_file():
                raise FileNotFoundError(
                    f"Stage {stage['name']} requires {source_checkpoint}"
                )
            command.extend(["--resume-checkpoint", str(source_checkpoint)])
        # The final EMA checkpoint is produced during training and then used by
        # the same run invocation for evaluation.
        if index == len(stage_specs) - 1:
            command.extend(["--checkpoint", str(checkpoint)])
        _run(command, dry_run=dry_run)
        if not dry_run and not checkpoint.is_file():
            raise RuntimeError(
                f"Stage {stage['name']} completed without expected checkpoint {checkpoint}"
            )
        completed.append(
            {"stage": stage["name"], "checkpoint": str(checkpoint), "skipped": False}
        )
    export = raw["hardware_export"]
    final_stage = stages[str(export["stage"])]
    export_dir = final_stage["settings"].output_dir / "best_optical_artifacts"
    export_command = [
        sys.executable,
        "-m",
        f"{PACKAGE}.export_best_optical_artifacts",
        "--config",
        str(final_stage["config"]),
        "--checkpoint",
        str(final_stage["checkpoint"]),
        "--output-dir",
        str(export_dir),
        "--sample-count",
        str(int(export["sample_count"])),
        "--slm-pixel-pitch-um",
        str(float(export["slm_pixel_pitch_um"])),
    ]
    _run(export_command, dry_run=dry_run)
    report = {
        "pipeline_config": str(config_path),
        "stages": completed,
        "final_checkpoint": str(final_stage["checkpoint"]),
        "best_optical_artifacts": str(export_dir),
        "elapsed_seconds": time.time() - started,
        "dry_run": dry_run,
        "skip_completed": skip_completed,
    }
    if not dry_run:
        destination = final_stage["settings"].output_dir / "canonical_reproduction.json"
        destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the canonical three-stage Grocery10 best-reproduction chain"
    )
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parent / "configs" / "grocery10_best_reproduction.yaml"),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--no-skip-completed",
        action="store_true",
        help=(
            "Run every stage even if its expected checkpoint exists. Use only with empty "
            "canonical output directories; existing logs are deliberately not deleted."
        ),
    )
    args = parser.parse_args()
    report = run_pipeline(
        Path(args.config).expanduser().resolve(),
        dry_run=args.dry_run,
        skip_completed=not args.no_skip_completed,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
