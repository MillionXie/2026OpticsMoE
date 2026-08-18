from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .settings import REPO_ROOT, Settings, load_settings


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


@dataclass(frozen=True)
class FormalConfig:
    source_checkpoint: Path
    source_checkpoint_sha256: str
    common_checkpoint: Path
    head_warmup_seed: int
    head_warmup_epochs: int
    head_learning_rate: float
    finetune_epochs: int
    finetune_seeds: tuple[int, ...]
    phase_learning_rate: float
    residual_learning_rate: float
    electronic_learning_rate: float
    warmup_epochs: int
    min_learning_rate_ratio: float
    checkpoint_interval_epochs: int


@dataclass(frozen=True)
class FormalSettings:
    base: Settings
    formal: FormalConfig


def load_formal_settings(path: str | Path) -> FormalSettings:
    config_path = Path(path)
    raw: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    values = raw.get("formal")
    if not isinstance(values, dict):
        raise ValueError("formal configuration section is required")
    sha256 = str(values["source_checkpoint_sha256"]).lower()
    if len(sha256) != 64 or any(character not in "0123456789abcdef" for character in sha256):
        raise ValueError("formal.source_checkpoint_sha256 must be a lowercase SHA-256")
    formal = FormalConfig(
        source_checkpoint=_resolve(values["source_checkpoint"]),
        source_checkpoint_sha256=sha256,
        common_checkpoint=_resolve(values["common_checkpoint"]),
        head_warmup_seed=int(values.get("head_warmup_seed", 4242)),
        head_warmup_epochs=int(values.get("head_warmup_epochs", 10)),
        head_learning_rate=float(values.get("head_learning_rate", 1e-3)),
        finetune_epochs=int(values.get("finetune_epochs", 20)),
        finetune_seeds=tuple(int(seed) for seed in values.get("finetune_seeds", [2026])),
        phase_learning_rate=float(values.get("phase_learning_rate", 5e-4)),
        residual_learning_rate=float(values.get("residual_learning_rate", 3e-4)),
        electronic_learning_rate=float(values.get("electronic_learning_rate", 3e-4)),
        warmup_epochs=int(values.get("warmup_epochs", 2)),
        min_learning_rate_ratio=float(values.get("min_learning_rate_ratio", 0.1)),
        checkpoint_interval_epochs=int(values.get("checkpoint_interval_epochs", 10)),
    )
    if formal.head_warmup_epochs < 1 or formal.finetune_epochs < 1:
        raise ValueError("formal epoch counts must be positive")
    if not formal.finetune_seeds:
        raise ValueError("formal.finetune_seeds cannot be empty")
    return FormalSettings(base=load_settings(config_path), formal=formal)
