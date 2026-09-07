"""Audited LightGenV2 entry point for the frozen-Qwen quality-token baseline."""

from __future__ import annotations

import argparse
import json
import platform
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
import yaml

from .project import REPO_ROOT, TASK_DIR, sha256
from .quality_token_common import FRAME_FRACTIONS, PROMPTS


MODULE_ROOT = "LightGenV2.tasks.t06_video_quality_assessment"


def _read_config(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise ValueError(f"Unsupported resolution-ablation config: {path}")
    return raw


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def _resolve_run_dir(raw: dict[str, Any], override: Path | None) -> Path:
    value = override if override is not None else Path(raw["run"]["run_dir"])
    path = value.expanduser()
    path = path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()
    owner = (TASK_DIR / "runs" / "simulation").resolve()
    if not path.is_relative_to(owner):
        raise ValueError(f"Simulation run must stay under {owner}, got {path}")
    return path


def _validate(raw: dict[str, Any], model: Path, manifest: Path) -> None:
    image_size = int(raw["input"]["image_size"])
    frames = tuple(int(value) for value in raw["input"]["frame_counts"])
    if image_size < 224 or image_size % 32:
        raise ValueError("Qwen3-VL input size must be >=224 and divisible by 32")
    input_contract = raw["input"]
    if (
        int(input_contract["qwen3vl_patch_size"]) != 16
        or int(input_contract["qwen3vl_spatial_merge_size"]) != 2
        or int(input_contract["qwen3vl_temporal_patch_size"]) != 2
        or int(input_contract["effective_spatial_stride"]) != 32
    ):
        raise ValueError("Qwen3-VL patch/merge geometry contract changed")
    target = str(raw["task"]["target"]).removesuffix("_mos")
    expected_frames = (4, 9, 16) if target == "temporal" else (4,)
    if frames != expected_frames:
        raise ValueError(
            f"Formal {target} frame-count contract must be {expected_frames}, got {frames}"
        )
    if not model.is_dir():
        raise FileNotFoundError(f"Qwen model directory is missing: {model}")
    if not manifest.is_file():
        raise FileNotFoundError(f"LGVQ manifest is missing: {manifest}")
    if raw["timing"]["explicit_warmup_forwards"] != 0:
        raise ValueError("This profile requires zero explicit warmup")
    if raw["timing"]["pretest_inference_forwards"] != 0:
        raise ValueError("This profile forbids pre-test inference")
    if raw["model"]["qwen_frozen"] is not True:
        raise ValueError("Qwen must remain frozen")
    if target not in PROMPTS or raw["task"]["prompt"] != PROMPTS[target]:
        raise ValueError(f"Formal {target} prompt differs from the executable contract")
    configured_fractions = raw["input"]["frame_fractions"]
    for count in frames:
        values = configured_fractions.get(count, configured_fractions.get(str(count)))
        if values is None or len(values) != count:
            raise ValueError(f"Missing {count}-frame sampling fractions")
        if any(abs(float(left) - float(right)) > 1e-8 for left, right in zip(values, FRAME_FRACTIONS[count])):
            raise ValueError(f"Configured {count}-frame fractions differ from executable contract")
    from LightGenV2.common.baseline_measurement import validate_cuda_device

    validate_cuda_device(str(raw["timing"]["gpu"]))


def _command_text(command: list[str]) -> str:
    return " ".join(shlex.quote(value) for value in command)


def _run_command(command: list[str], run_dir: Path) -> None:
    text = _command_text(command)
    with (run_dir / "commands.txt").open("a", encoding="utf-8") as stream:
        stream.write(text + "\n")
    print(f"[launch] {text}", flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def _record_identity(
    *,
    raw: dict[str, Any],
    config: Path,
    model: Path,
    manifest: Path,
    run_dir: Path,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    resolved = dict(raw)
    resolved["runtime_paths"] = {
        "model": str(model),
        "manifest": str(manifest),
        "run_dir": str(run_dir),
    }
    (run_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    identity = {
        "schema_version": 1,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "config": str(config),
        "config_sha256": sha256(config),
        "git_commit": commit,
        "git_worktree_clean": status == "",
        "git_status": status.splitlines(),
        "model": str(model),
        "manifest": str(manifest),
        "manifest_sha256": sha256(manifest),
        "run_dir": str(run_dir),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    }
    _write_json(run_dir / "run_identity.json", identity)
    (run_dir / "original_command.txt").write_text(
        _command_text([sys.executable, "-m", MODULE_ROOT + ".quality_token_resolution", *sys.argv[1:]])
        + "\n",
        encoding="utf-8",
    )


def _selected_frames(raw: dict[str, Any], value: int | None) -> list[int]:
    allowed = [int(item) for item in raw["input"]["frame_counts"]]
    if value is None:
        return allowed
    if value not in allowed:
        raise ValueError(f"frames must be one of {allowed}")
    return [value]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--phase", required=True, choices=("preflight", "extract", "train", "benchmark", "all")
    )
    parser.add_argument("--frames", type=int)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()

    config = args.config.expanduser().resolve()
    raw = _read_config(config)
    model = args.model.expanduser().resolve()
    manifest = args.manifest.expanduser().resolve()
    run_dir = _resolve_run_dir(raw, args.run_dir)
    _validate(raw, model, manifest)
    _record_identity(
        raw=raw, config=config, model=model, manifest=manifest, run_dir=run_dir
    )
    frames = _selected_frames(raw, args.frames)
    preflight = {
        "status": "ready",
        "image_size": int(raw["input"]["image_size"]),
        "frame_counts": frames,
        "model": str(model),
        "manifest": str(manifest),
        "run_dir": str(run_dir),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_worktree_clean": _git("status", "--porcelain") == "",
    }
    _write_json(run_dir / "preflight.json", preflight)
    if args.phase == "preflight":
        print(json.dumps(preflight, indent=2), flush=True)
        return 0

    python = sys.executable
    image_size = str(raw["input"]["image_size"])
    target = str(raw["task"]["target"]).removesuffix("_mos")
    feature_root = run_dir / "features"
    checkpoint_root = run_dir / "checkpoints"
    evaluation_root = run_dir / "evaluation"
    phases = ("extract", "train", "benchmark") if args.phase == "all" else (args.phase,)
    status_path = run_dir / "status.json"
    status = {"status": "running", "requested_phase": args.phase, "completed": []}
    _write_json(status_path, status)
    try:
        for phase in phases:
            for count in frames:
                if phase == "extract":
                    command = [
                        python,
                        "-m",
                        MODULE_ROOT + ".quality_token_extract",
                        "--frames",
                        str(count),
                        "--target",
                        target,
                        "--image-size",
                        image_size,
                        "--batch-size",
                        str(raw["feature_extraction"]["batch_size"]),
                        "--decode-workers",
                        str(raw["feature_extraction"]["decode_workers"]),
                        "--chunk-rows",
                        str(raw["feature_extraction"]["chunk_rows"]),
                        "--model",
                        str(model),
                        "--manifest",
                        str(manifest),
                        "--output",
                        str(feature_root / f"frames{count}" / "qwen_prompt_features.pt"),
                    ]
                elif phase == "train":
                    command = [
                        python,
                        "-m",
                        MODULE_ROOT + ".quality_token_train",
                        "--frames",
                        str(count),
                        "--target",
                        target,
                        "--epochs",
                        str(raw["training"]["epochs"]),
                        "--batch-size",
                        str(raw["training"]["batch_size"]),
                        "--learning-rate",
                        str(raw["training"]["learning_rate"]),
                        "--seed",
                        str(raw["training"]["seed"]),
                        "--model",
                        str(model),
                        "--expected-gpu",
                        str(raw["timing"]["gpu"]),
                        "--feature-root",
                        str(feature_root),
                        "--output",
                        str(checkpoint_root),
                    ]
                elif phase == "benchmark":
                    command = [
                        python,
                        "-m",
                        MODULE_ROOT + ".quality_token_dataset_once",
                        "--frames",
                        str(count),
                        "--target",
                        target,
                        "--scheme",
                        "scheme2_five_quality_tokens",
                        "--image-size",
                        image_size,
                        "--model",
                        str(model),
                        "--manifest",
                        str(manifest),
                        "--checkpoint",
                        str(checkpoint_root / f"frames{count}" / "best_checkpoint.pt"),
                        "--output",
                        str(evaluation_root),
                        "--expected-gpu",
                        str(raw["timing"]["gpu"]),
                    ]
                else:
                    raise AssertionError(phase)
                _run_command(command, run_dir)
                status["completed"].append({"phase": phase, "frames": count})
                _write_json(status_path, status)
        status["status"] = "complete"
    except Exception as error:
        status["status"] = "failed"
        status["error"] = repr(error)
        _write_json(status_path, status)
        raise
    _write_json(status_path, status)
    print(json.dumps(status, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
