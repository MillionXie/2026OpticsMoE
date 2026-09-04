from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _get(raw: dict[str, Any], path: str, default: Any = None) -> Any:
    value: Any = raw
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def _path(value: str | Path, *, base: Path = PROJECT_ROOT) -> Path:
    result = Path(value).expanduser()
    return result.resolve() if result.is_absolute() else (base / result).resolve()


def find_local_qwen_checkpoint(model_id: str) -> Path:
    repo = model_id.replace("/", "--")
    hubs: list[Path] = []
    if os.environ.get("HUGGINGFACE_HUB_CACHE"):
        hubs.append(Path(os.environ["HUGGINGFACE_HUB_CACHE"]))
    if os.environ.get("HF_HOME"):
        hubs.append(Path(os.environ["HF_HOME"]) / "hub")
    hubs.extend(
        [
            Path.home() / ".cache" / "huggingface" / "hub",
            PROJECT_ROOT.parent / ".cache" / "huggingface" / "hub",
        ]
    )
    snapshot_roots = [hub / f"models--{repo}" / "snapshots" for hub in dict.fromkeys(hubs)]
    choices = sorted(
        path
        for snapshots in snapshot_roots
        for path in snapshots.glob("*")
        if (path / "config.json").exists()
    )
    if not choices:
        raise FileNotFoundError(
            f"No local Hugging Face snapshot found for {model_id} under {snapshot_roots}"
        )
    return choices[-1].resolve()


@dataclass(slots=True)
class Settings:
    config_path: Path
    seed: int
    data_dir: Path
    output_dir: Path
    image_size: int
    train_samples: int
    test_samples: int
    grid_size: int
    object_half_size: int
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
    palette_classes: int
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
    changed_pixel_weight: float
    palette_loss_weight: float
    edit_mask_loss_weight: float
    preservation_loss_weight: float
    task_loss_weight: float
    edge_loss_weight: float
    ccd_loss_weight: float
    router_balance_weight: float
    log_interval: int
    visualization_samples: int

    @property
    def train_manifest(self) -> Path:
        return self.data_dir / "train.jsonl"

    @property
    def test_manifest(self) -> Path:
        return self.data_dir / "test.jsonl"

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        return {key: str(value) if isinstance(value, Path) else value for key, value in values.items()}


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
    data_dir = _path(_get(raw, "dataset.data_dir", "data/synthetic_instruction_editing/v1"))
    output_dir = _path(_get(raw, "output_dir", "runs/synthetic_instruction_editing"))
    prompt_cache_value = _get(raw, "qwen.prompt_cache", "prompt_hidden.pt")
    prompt_cache_path = Path(prompt_cache_value)
    if not prompt_cache_path.is_absolute():
        prompt_cache_path = data_dir / prompt_cache_path

    settings = Settings(
        config_path=config_path,
        seed=int(_get(raw, "seed", 42)),
        data_dir=data_dir,
        output_dir=output_dir,
        image_size=int(_get(raw, "dataset.image_size", 224)),
        train_samples=int(_get(raw, "dataset.train_samples", 20_000)),
        test_samples=int(_get(raw, "dataset.test_samples", 2_000)),
        grid_size=int(_get(raw, "dataset.grid_size", 7)),
        object_half_size=int(_get(raw, "dataset.object_half_size", 11)),
        prompt_templates_per_operation=int(
            _get(raw, "dataset.prompt_templates_per_operation", 3)
        ),
        qwen_model_id=model_id,
        qwen_checkpoint=checkpoint,
        prompt_cache_path=prompt_cache_path.resolve(),
        prompt_cache_batch_size=int(_get(raw, "qwen.prompt_cache_batch_size", 8)),
        prompt_max_tokens=int(_get(raw, "qwen.prompt_max_tokens", 64)),
        optical_base_config=_path(
            _get(
                raw,
                "model.optical_base_config",
                "experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval/"
                "configs/release/caltech101_four_layer_optical_joint.yaml",
            )
        ),
        optical_enabled=bool(_get(raw, "model.optical_enabled", True)),
        electronic_width=int(_get(raw, "model.electronic_width", 192)),
        max_language_tokens=int(_get(raw, "model.max_language_tokens", 64)),
        palette_classes=int(_get(raw, "model.palette_classes", 8)),
        optical_fusion_initial=float(_get(raw, "model.optical_fusion_initial", 0.05)),
        optical_shift_pixels=int(_get(raw, "model.optical_shift_pixels", 2)),
        phase_dropout_p=float(_get(raw, "model.phase_dropout_p", 0.05)),
        epochs=int(_get(raw, "training.epochs", 60)),
        batch_size=int(_get(raw, "training.batch_size", 8)),
        num_workers=int(_get(raw, "training.num_workers", 0)),
        learning_rate=float(_get(raw, "training.learning_rate", 1.0e-4)),
        adapter_learning_rate=float(_get(raw, "training.adapter_learning_rate", 1.0e-4)),
        phase_learning_rate=float(_get(raw, "training.phase_learning_rate", 1.0e-4)),
        router_learning_rate=float(_get(raw, "training.router_learning_rate", 5.0e-5)),
        readout_learning_rate=float(_get(raw, "training.readout_learning_rate", 5.0e-5)),
        decoder_learning_rate=float(_get(raw, "training.decoder_learning_rate", 3.0e-4)),
        weight_decay=float(_get(raw, "training.weight_decay", 0.01)),
        warmup_electronic_epochs=int(_get(raw, "training.warmup_electronic_epochs", 5)),
        gradient_clip_norm=float(_get(raw, "training.gradient_clip_norm", 1.0)),
        ema_decay=float(_get(raw, "training.ema_decay", 0.995)),
        amp_enabled=bool(_get(raw, "training.amp_enabled", True)),
        resume=bool(_get(raw, "training.resume", False)),
        changed_pixel_weight=float(_get(raw, "loss.changed_pixel_weight", 4.0)),
        palette_loss_weight=float(_get(raw, "loss.palette", 1.0)),
        edit_mask_loss_weight=float(_get(raw, "loss.edit_mask", 1.0)),
        preservation_loss_weight=float(_get(raw, "loss.preservation", 0.25)),
        task_loss_weight=float(_get(raw, "loss.task", 0.1)),
        edge_loss_weight=float(_get(raw, "loss.edge", 0.25)),
        ccd_loss_weight=float(_get(raw, "loss.ccd", 0.02)),
        router_balance_weight=float(_get(raw, "loss.router_balance", 0.005)),
        log_interval=int(_get(raw, "logging.interval_batches", 10)),
        visualization_samples=int(_get(raw, "logging.visualization_samples", 8)),
    )
    if settings.image_size != 224:
        raise ValueError("The frozen Qwen patch stem is fixed to 224x224 in this experiment")
    if settings.grid_size != 7:
        raise ValueError("Version 1 fixes the logical scene grid to 7x7")
    if settings.palette_classes != 8:
        raise ValueError("Version 1 palette is fixed to eight classes")
    if settings.max_language_tokens <= 0 or settings.max_language_tokens > 224:
        raise ValueError("max_language_tokens must be in [1,224]")
    return settings


__all__ = ["PROJECT_ROOT", "Settings", "find_local_qwen_checkpoint", "load_settings"]
