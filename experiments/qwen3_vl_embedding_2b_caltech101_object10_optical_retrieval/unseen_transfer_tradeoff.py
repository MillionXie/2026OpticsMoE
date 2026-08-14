"""Reproduce held-out-class retention and checkpoint-interpolation analyses.

The ten evaluation categories were included in 101-category pretraining but
excluded from Target-10 adaptation.  Consequently this is a retained-transfer
analysis, not unseen-to-pretraining zero-shot evaluation.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
import yaml


MODULE = "experiments.qwen3_vl_embedding_2b_caltech101_object10_optical_retrieval"
MODEL_SECTIONS = ("vision_optical", "language_optical", "retrieval_readout")


def interpolate_checkpoints(
    generic_checkpoint: Path,
    target_checkpoint: Path,
    generic_weight: float,
    output_path: Path,
    *,
    status: str,
) -> Path:
    if not 0.0 <= generic_weight <= 1.0:
        raise ValueError("generic_weight must be between 0 and 1")
    generic = torch.load(generic_checkpoint, map_location="cpu", weights_only=False)
    target = torch.load(target_checkpoint, map_location="cpu", weights_only=False)
    output: dict[str, Any] = dict(target)
    for section in MODEL_SECTIONS:
        if section not in generic or section not in target:
            raise KeyError(f"Checkpoint section is missing: {section}")
        merged: dict[str, Any] = {}
        for name, target_value in target[section].items():
            generic_value = generic[section].get(name)
            if (
                torch.is_tensor(target_value)
                and torch.is_tensor(generic_value)
                and target_value.shape == generic_value.shape
                and target_value.is_floating_point()
            ):
                merged[name] = (
                    generic_value.float() * generic_weight
                    + target_value.float() * (1.0 - generic_weight)
                ).to(target_value.dtype)
            else:
                merged[name] = target_value
        output[section] = merged
    output["optimizer"] = {}
    output["metadata"] = dict(target.get("metadata", {}))
    output["metadata"]["checkpoint_interpolation"] = {
        "generic_checkpoint": str(generic_checkpoint),
        "target_checkpoint": str(target_checkpoint),
        "generic_weight": generic_weight,
        "target_weight": 1.0 - generic_weight,
        "status": status,
        "note": "No held-out query or gallery image contributes to the interpolation.",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, output_path)
    return output_path


def _write_runtime_config(base_config: Path, output_dir: Path, destination: Path) -> Path:
    raw = yaml.safe_load(base_config.read_text(encoding="utf-8"))
    parent = raw.get("base_config")
    if parent is not None:
        parent_path = Path(parent)
        if not parent_path.is_absolute():
            parent_path = (base_config.parent / parent_path).resolve()
        raw["base_config"] = str(parent_path)
    raw["output_dir"] = str(output_dir.resolve())
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return destination


def _run(repository: Path, config: Path, checkpoint: Path, device: str) -> dict[str, Any]:
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = device
    for phase in ("cache_teacher_embeddings", "evaluate"):
        command = [
            sys.executable,
            "-m",
            MODULE,
            "--config",
            str(config),
            "--phase",
            phase,
        ]
        if phase == "evaluate":
            command.extend(("--checkpoint", str(checkpoint)))
        print("+ " + " ".join(command), flush=True)
        subprocess.run(command, cwd=repository, check=True, env=environment)
    output_dir = Path(yaml.safe_load(config.read_text())["output_dir"])
    return json.loads((output_dir / "student_metrics.json").read_text())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="0", help="Physical CUDA index exposed to each evaluation")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    experiment = Path(__file__).resolve().parent
    repository = experiment.parents[1]
    generic = experiment / "runs/caltech101_101class_pretrain/ema_last_checkpoint.pt"
    target = experiment / "runs/caltech101_10class_router_rebalance_final_epoch56/last_checkpoint.pt"
    base = experiment / "configs/caltech101_10class_unseen_eval.yaml"
    artifacts = experiment / "runs/caltech101_unseen_transfer_tradeoff"
    definitions = [
        ("pretrain", generic, None, "predefined"),
        ("final_compromise", target, None, "main model"),
        ("interpolation50", artifacts / "generic50_target50.pt", 0.50, "predefined"),
        # This point was added after observing the predefined 50:50 result and
        # must therefore remain labelled exploratory in figures and manuscripts.
        ("interpolation75", artifacts / "generic75_target25.pt", 0.75, "exploratory"),
    ]
    rows: list[dict[str, Any]] = []
    for name, checkpoint, weight, status in definitions:
        if weight is not None:
            interpolate_checkpoints(generic, target, weight, checkpoint, status=status)
        output_dir = artifacts / name
        config = _write_runtime_config(base, output_dir, output_dir / "resolved_input.yaml")
        if args.prepare_only:
            continue
        metrics = _run(repository, config, checkpoint, args.device)
        rows.append({
            "condition": name,
            "status": status,
            "checkpoint": str(checkpoint),
            "top1": metrics["top1_retrieval_accuracy"],
            "top3": metrics["top3_retrieval_accuracy"],
            "mrr": metrics["mrr"],
        })
    artifacts.mkdir(parents=True, exist_ok=True)
    (artifacts / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
