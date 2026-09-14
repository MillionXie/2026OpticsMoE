from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from FixedFeedbackSFT.paths import REPOSITORY_ROOT, resolve_repository_path
from .datasets import TASK_SPECS


TASKS = tuple(TASK_SPECS)
METHODS = ("noft", "bp", "fa_pretrained", "fa_random")


@dataclass(frozen=True)
class Settings:
    config_path: Path
    task: str
    method: str
    seed: int
    source_backbone: Path
    source_sha256: str
    stem_checkpoint: Path
    data_root: Path
    output_root: Path
    head_only_epochs: int
    adaptation_epochs: int
    train_batch_size: int
    evaluation_batch_size: int
    num_workers: int
    head_hidden_dim: int
    phase_learning_rate: float
    adapter_learning_rate: float
    residual_learning_rate: float
    head_learning_rate: float
    electronic_weight_decay: float
    warmup_epochs: int
    minimum_learning_rate_ratio: float
    use_amp: bool
    checkpoint_interval_epochs: int
    smoke_samples: int | None = None

    @property
    def run_dir(self) -> Path:
        return self.output_root / self.task / self.method / f"seed_{self.seed}"

    @property
    def common_checkpoint(self) -> Path:
        return self.output_root / self.task / "noft" / f"seed_{self.seed}" / "checkpoints" / "last.pt"

    @property
    def num_classes(self) -> int:
        return TASK_SPECS[self.task].num_classes

    @property
    def p11_config(self) -> dict[str, Any]:
        return {
            "canvas_size": 224,
            "optical_channels": 3,
            "num_stages": 8,
            "token_dim": 224,
            "num_classes": 1000,
            "head_hidden_dim": 448,
            "wavelength_m": 5.32e-7,
            "pixel_size_m": 1.6e-5,
            "distance_m": 0.05,
            "token_axis_propagation_distance_m": 0.05,
            "channel_axis_propagation_distance_m": 0.05,
            "phase_init_std": 0.10,
            "layernorm_eps": 1.0e-5,
            "optical_gate_init": 0.60,
            "optical_gate_min": 0.50,
            "mixer_width": 96,
            "seed": 2026,
        }

    def identity_payload(self) -> dict[str, Any]:
        payload = dict(self.__dict__)
        return {key: str(value) if isinstance(value, Path) else value for key, value in payload.items()}

    def digest(self) -> str:
        encoded = json.dumps(self.identity_payload(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _path(raw: Any, env_name: str) -> Path:
    override = os.environ.get(env_name)
    return resolve_repository_path(override if override else str(raw))


def load_settings(
    config_path: str | Path,
    *,
    task: str | None = None,
    method: str | None = None,
    seed: int | None = None,
    output_root: str | Path | None = None,
    smoke: bool = False,
) -> Settings:
    path = Path(config_path).expanduser().resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    selection = raw["selection"]
    selected_task = task or selection["task"]
    selected_method = method or selection["method"]
    selected_seed = int(selection["seed"] if seed is None else seed)
    if selected_task not in TASKS:
        raise ValueError(f"Unsupported task {selected_task}; choose from {TASKS}")
    if selected_method not in METHODS:
        raise ValueError(f"Unsupported method {selected_method}; choose from {METHODS}")
    paths = raw["paths"]
    training = raw["training"]
    optimizer = raw["optimizer"]
    dataloader = raw["dataloader"]
    model = raw["model"]
    selected_output = output_root or os.environ.get("P14_OUTPUT_ROOT") or paths["output_root"]
    settings = Settings(
        config_path=path,
        task=selected_task,
        method=selected_method,
        seed=selected_seed,
        source_backbone=_path(paths["source_backbone"], "P14_SOURCE_BACKBONE"),
        source_sha256=str(paths["source_backbone_sha256"]),
        stem_checkpoint=_path(paths["stem_checkpoint"], "P14_STEM_CHECKPOINT"),
        data_root=_path(paths["data_root"], "P14_DATA_ROOT"),
        output_root=resolve_repository_path(selected_output),
        head_only_epochs=1 if smoke else int(training["head_only_epochs"]),
        adaptation_epochs=1 if smoke else int(training["adaptation_epochs"]),
        train_batch_size=int(dataloader["train_batch_size"]),
        evaluation_batch_size=int(dataloader["evaluation_batch_size"]),
        num_workers=0 if smoke else int(dataloader["num_workers"]),
        head_hidden_dim=int(model["head_hidden_dim"]),
        phase_learning_rate=float(optimizer["phase_learning_rate"]),
        adapter_learning_rate=float(optimizer["adapter_learning_rate"]),
        residual_learning_rate=float(optimizer["residual_learning_rate"]),
        head_learning_rate=float(optimizer["head_learning_rate"]),
        electronic_weight_decay=float(optimizer["electronic_weight_decay"]),
        warmup_epochs=0 if smoke else int(optimizer["warmup_epochs"]),
        minimum_learning_rate_ratio=float(optimizer["minimum_learning_rate_ratio"]),
        use_amp=bool(training["use_amp"]),
        checkpoint_interval_epochs=int(training["checkpoint_interval_epochs"]),
        smoke_samples=8 if smoke else None,
    )
    if not settings.source_backbone.is_file():
        raise FileNotFoundError(f"Missing P11 source checkpoint: {settings.source_backbone}")
    if not settings.stem_checkpoint.is_file():
        raise FileNotFoundError(f"Missing frozen Qwen stem: {settings.stem_checkpoint}")
    if len(settings.source_sha256) != 64:
        raise ValueError("source_backbone_sha256 must be a 64-character digest")
    if settings.train_batch_size < 64:
        raise ValueError("Formal VTAB runs require train_batch_size >= 64")
    if settings.head_only_epochs != settings.adaptation_epochs:
        raise ValueError("Head-only and adaptation budgets must be matched")
    return settings


__all__ = ["METHODS", "TASKS", "Settings", "load_settings"]
