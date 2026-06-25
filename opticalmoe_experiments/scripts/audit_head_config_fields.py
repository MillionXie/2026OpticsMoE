"""Audit task-specific readout head fields in multitask experiment YAML files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml


HEAD_FIELDS = [
    "detector_size",
    "detector_layout",
    "readout_type",
    "normalize_detector_energy",
    "logit_scale",
    "input_norm",
    "norm_affine",
    "hidden_dim",
    "hidden_layers",
    "activation",
    "dropout",
]


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8-sig") as handle:
        return yaml.safe_load(handle) or {}


def _missing_head_fields(head: Dict[str, Any]) -> List[str]:
    return [field for field in HEAD_FIELDS if field not in (head or {})]


def audit_dataset_switching(path: Path, cfg: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    tasks = cfg.get("training", {}).get("multitask", {}).get("tasks")
    if tasks is None:
        tasks = cfg.get("training", {}).get("tasks", [])
    for task in tasks or []:
        if not isinstance(task, dict) or "name" not in task:
            continue
        task_name = str(task["name"]).lower()
        head = task.get("head")
        if not isinstance(head, dict):
            errors.append(f"{path}:{task_name} missing head")
            continue
        for field in _missing_head_fields(head):
            errors.append(f"{path}:{task_name} missing head.{field}")
    return errors


def audit_same_input(path: Path, cfg: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    training = cfg.get("training", {}) or {}
    task_names = [str(task).lower() for task in training.get("tasks", [])]
    heads = training.get("task_heads")
    if not isinstance(heads, dict):
        return [f"{path}:training missing task_heads"]
    unknown = sorted(set(str(name).lower() for name in heads) - set(task_names))
    for task_name in unknown:
        errors.append(f"{path}:{task_name} unknown task_heads entry")
    for task_name in task_names:
        head = heads.get(task_name)
        if not isinstance(head, dict):
            errors.append(f"{path}:{task_name} missing training.task_heads entry")
            continue
        for field in _missing_head_fields(head):
            errors.append(f"{path}:{task_name} missing training.task_heads.{field}")
    return errors


def audit_configs(root: Path) -> List[str]:
    errors: List[str] = []
    for path in sorted((root / "dataset_switching" / "configs").glob("*.yaml")):
        errors.extend(audit_dataset_switching(path, _load_yaml(path)))
    for path in sorted((root / "same_input_multitask" / "configs").glob("*.yaml")):
        errors.extend(audit_same_input(path, _load_yaml(path)))
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=Path(__file__).resolve().parents[1], type=Path)
    args = parser.parse_args()
    errors = audit_configs(args.root)
    if errors:
        print("Head config audit failed:")
        for error in errors:
            print(f"  - {error}")
        return 1
    print("Head config audit passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
