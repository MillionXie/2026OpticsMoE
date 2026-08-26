"""Lightweight I/O helpers; importing this module never requires Torch."""

from __future__ import annotations

import csv
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np


def set_seed(value: int) -> None:
    random.seed(value)
    np.random.seed(value)
    # Training has Torch installed, while the laboratory raw-CCD evaluator
    # intentionally remains usable in a lightweight environment.
    import torch

    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
