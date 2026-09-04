from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from experiments.qwen3_vl_2b_synthetic_instruction_four_stage_optical_editing.settings import (
    find_local_qwen_checkpoint,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _get(raw: dict[str, Any], path: str, default: Any = None) -> Any:
    value: Any = raw
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def _path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


@dataclass(slots=True)
class Settings:
    config_path: Path
    seed: int
    data_dir: Path
    output_dir: Path
    asset_dir: Path
    image_size: int
    grid_size: int
    icon_size: int
    icon_classes: int
    train_samples: int
    test_samples: int
    prompt_templates_per_operation: int
    qwen_model_id: str
    qwen_checkpoint: Path
    prompt_cache_path: Path
    prompt_cache_batch_size: int
    prompt_max_tokens: int
    optical_base_config: Path
    optical_enabled: bool
    electronic_width: int
    max_language_tokens: int
    optical_fusion_initial: float
    optical_shift_pixels: int
    phase_dropout_p: float
    epochs: int
    batch_size: int
    num_workers: int
    learning_rate: float
    adapter_learning_rate: float
    phase_learning_rate: float
    router_learning_rate: float
    readout_learning_rate: float
    decoder_learning_rate: float
    weight_decay: float
    warmup_electronic_epochs: int
    gradient_clip_norm: float
    ema_decay: float
    amp_enabled: bool
    resume: bool
    changed_cell_weight: float
    foreground_cell_weight: float
    category_loss_weight: float
    edit_loss_weight: float
    preservation_loss_weight: float
    task_loss_weight: float
    ccd_loss_weight: float
    router_balance_weight: float
    log_interval: int
    visualization_samples_per_task: int

    @property
    def train_manifest(self) -> Path:
        return self.data_dir / "train.jsonl"

    @property
    def test_manifest(self) -> Path:
        return self.data_dir / "test.jsonl"

    def to_dict(self) -> dict[str, Any]:
        return {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(self).items()
        }


def load_settings(path: str | Path) -> Settings:
    config_path = _path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Configuration root must be a mapping")
    model_id = str(_get(raw, "qwen.model_id", "Qwen/Qwen3-VL-2B-Instruct"))
    checkpoint_value = _get(raw, "qwen.checkpoint", "auto")
    checkpoint = (
        find_local_qwen_checkpoint(model_id)
        if checkpoint_value in {None, "auto"}
        else _path(str(checkpoint_value))
    )
    data_dir = _path(_get(raw, "dataset.data_dir"))
    prompt_cache = Path(str(_get(raw, "qwen.prompt_cache", "prompt_hidden.pt")))
    if not prompt_cache.is_absolute():
        prompt_cache = data_dir / prompt_cache
    settings = Settings(
        config_path=config_path,
        seed=int(_get(raw, "seed", 73)),
        data_dir=data_dir,
        output_dir=_path(_get(raw, "output_dir")),
        asset_dir=_path(_get(raw, "dataset.asset_dir")),
        image_size=int(_get(raw, "dataset.image_size", 224)),
        grid_size=int(_get(raw, "dataset.grid_size", 6)),
        icon_size=int(_get(raw, "dataset.icon_size", 30)),
        icon_classes=int(_get(raw, "dataset.icon_classes", 16)),
        train_samples=int(_get(raw, "dataset.train_samples", 5_000)),
        test_samples=int(_get(raw, "dataset.test_samples", 1_000)),
        prompt_templates_per_operation=int(
            _get(raw, "dataset.prompt_templates_per_operation", 3)
        ),
        qwen_model_id=model_id,
        qwen_checkpoint=checkpoint,
        prompt_cache_path=prompt_cache.resolve(),
        prompt_cache_batch_size=int(_get(raw, "qwen.prompt_cache_batch_size", 16)),
        prompt_max_tokens=int(_get(raw, "qwen.prompt_max_tokens", 64)),
        optical_base_config=_path(_get(raw, "model.optical_base_config")),
        optical_enabled=bool(_get(raw, "model.optical_enabled", True)),
        electronic_width=int(_get(raw, "model.electronic_width", 192)),
        max_language_tokens=int(_get(raw, "model.max_language_tokens", 64)),
        optical_fusion_initial=float(_get(raw, "model.optical_fusion_initial", 0.05)),
        optical_shift_pixels=int(_get(raw, "model.optical_shift_pixels", 1)),
        phase_dropout_p=float(_get(raw, "model.phase_dropout_p", 0.05)),
        epochs=int(_get(raw, "training.epochs", 20)),
        batch_size=int(_get(raw, "training.batch_size", 16)),
        num_workers=int(_get(raw, "training.num_workers", 4)),
        learning_rate=float(_get(raw, "training.learning_rate", 1.0e-4)),
        adapter_learning_rate=float(_get(raw, "training.adapter_learning_rate", 1.0e-4)),
        phase_learning_rate=float(_get(raw, "training.phase_learning_rate", 1.0e-4)),
        router_learning_rate=float(_get(raw, "training.router_learning_rate", 5.0e-5)),
        readout_learning_rate=float(_get(raw, "training.readout_learning_rate", 5.0e-5)),
        decoder_learning_rate=float(_get(raw, "training.decoder_learning_rate", 3.0e-4)),
        weight_decay=float(_get(raw, "training.weight_decay", 0.01)),
        warmup_electronic_epochs=int(_get(raw, "training.warmup_electronic_epochs", 2)),
        gradient_clip_norm=float(_get(raw, "training.gradient_clip_norm", 1.0)),
        ema_decay=float(_get(raw, "training.ema_decay", 0.995)),
        amp_enabled=bool(_get(raw, "training.amp_enabled", True)),
        resume=bool(_get(raw, "training.resume", False)),
        changed_cell_weight=float(_get(raw, "loss.changed_cell_weight", 8.0)),
        foreground_cell_weight=float(_get(raw, "loss.foreground_cell_weight", 2.0)),
        category_loss_weight=float(_get(raw, "loss.category", 1.0)),
        edit_loss_weight=float(_get(raw, "loss.edit", 1.0)),
        preservation_loss_weight=float(_get(raw, "loss.preservation", 0.2)),
        task_loss_weight=float(_get(raw, "loss.task", 0.1)),
        ccd_loss_weight=float(_get(raw, "loss.ccd", 0.02)),
        router_balance_weight=float(_get(raw, "loss.router_balance", 0.005)),
        log_interval=int(_get(raw, "logging.interval_batches", 20)),
        visualization_samples_per_task=int(
            _get(raw, "logging.visualization_samples_per_task", 4)
        ),
    )
    if settings.image_size != 224:
        raise ValueError("The frozen Qwen vision stem requires 224x224 images")
    if settings.grid_size < 4 or settings.grid_size > 8:
        raise ValueError("grid_size must be in [4,8]")
    if settings.icon_classes != 16:
        raise ValueError("OpenMoji v1 fixes sixteen semantic icon classes")
    return settings


__all__ = ["PROJECT_ROOT", "Settings", "load_settings"]

