"""Audit dataset size and DataLoader fields in experiment YAML configs.

Run from the repository root:

    python opticalmoe_experiments/scripts/audit_dataset_config_fields.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import yaml


REQUIRED_DATASET_FIELDS = [
    "batch_size",
    "num_workers",
    "pin_memory",
    "persistent_workers",
    "prefetch_factor",
    "smoke_train_size",
    "smoke_test_size",
    "max_train_samples",
    "max_val_samples",
    "max_test_samples",
]

REQUIRED_SAMPLING_FIELDS = [
    "enabled",
    "total_size",
    "train_test_ratio",
    "class_balanced",
    "seed_offset",
]

CONFIG_DIRS = [
    "single_task/configs",
    "dataset_switching/configs",
    "same_input_multitask/configs",
]


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8-sig") as handle:
        return yaml.safe_load(handle) or {}


def _looks_like_dataset_block(value: Any) -> bool:
    return isinstance(value, dict) and (
        "name" in value
        or "root" in value
        or "batch_size" in value
        or "sampling_protocol" in value
    )


def _iter_dataset_blocks(node: Any, prefix: str = "") -> Iterable[Tuple[str, Dict[str, Any]]]:
    if isinstance(node, dict):
        for key, value in node.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if key == "dataset" and _looks_like_dataset_block(value):
                yield path, value
            else:
                yield from _iter_dataset_blocks(value, path)
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from _iter_dataset_blocks(value, f"{prefix}[{index}]")


def audit_config_file(path: Path) -> List[str]:
    data = _load_yaml(path)
    errors: List[str] = []
    blocks = list(_iter_dataset_blocks(data))
    if not blocks:
        return errors
    for block_path, dataset_cfg in blocks:
        for field in REQUIRED_DATASET_FIELDS:
            if field not in dataset_cfg:
                errors.append(f"{path}:{block_path} missing {field}")
        sampling = dataset_cfg.get("sampling_protocol")
        if not isinstance(sampling, dict):
            errors.append(f"{path}:{block_path} missing sampling_protocol")
            continue
        for field in REQUIRED_SAMPLING_FIELDS:
            if field not in sampling:
                errors.append(f"{path}:{block_path} missing sampling_protocol.{field}")
    return errors


def audit_configs(root: Path) -> List[str]:
    errors: List[str] = []
    for rel_dir in CONFIG_DIRS:
        config_dir = root / rel_dir
        if not config_dir.exists():
            continue
        for path in sorted(config_dir.glob("*.yaml")):
            errors.extend(audit_config_file(path))
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=Path(__file__).resolve().parents[1], type=Path)
    args = parser.parse_args()
    errors = audit_configs(args.root)
    if errors:
        print("Dataset config audit failed:")
        for error in errors:
            print(f"  - {error}")
        return 1
    print("Dataset config audit passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
